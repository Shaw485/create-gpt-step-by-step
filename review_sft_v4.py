"""Local browser UI for real human review of SFT v4 evaluation records.

The server binds to loopback only. It never grants approval automatically and
stores decisions separately from the immutable teacher source and candidates.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
from threading import RLock
from typing import Any, Sequence
from urllib.parse import parse_qs, urlparse

from build_sft_v4 import (
    SftV4ValidationError,
    atomic_write_text,
    configure_module_logger,
    jsonl_text,
    read_jsonl,
)
from repair_teacher_sft_v4 import ai_pre_review


SCHEMA_VERSION = "sft_v4_human_review_decision/1.0"
ALLOWED_DECISIONS = {"approved", "modified_approved", "rejected"}
ALLOWED_FILTERS = {"all", "pending", "transformed", "low_risk"}
MAX_REQUEST_BYTES = 64 * 1024


def configure_review_logging(log_dir: Path) -> dict[str, logging.Logger]:
    """Create independently filterable UI, decision-data, and validation logs."""

    return {
        "ui": configure_module_logger(
            "sft.review.ui",
            log_dir / "sft_review_ui.log",
            "SFT_REVIEW_UI_LOG_LEVEL",
        ),
        "data": configure_module_logger(
            "sft.review.data",
            log_dir / "sft_review_data.log",
            "SFT_REVIEW_DATA_LOG_LEVEL",
        ),
        "validation": configure_module_logger(
            "sft.review.validation",
            log_dir / "sft_review_validation.log",
            "SFT_REVIEW_VALIDATION_LOG_LEVEL",
        ),
    }


def candidate_digest(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


class ReviewValidationError(ValueError):
    pass


class ReviewStore:
    """Thread-safe decision store whose source candidates remain untouched."""

    def __init__(
        self,
        dataset_path: Path,
        decisions_path: Path,
        loggers: dict[str, logging.Logger],
    ) -> None:
        self.dataset_path = dataset_path
        self.decisions_path = decisions_path
        self.loggers = loggers
        self.lock = RLock()
        candidates = read_jsonl(dataset_path)
        self.records = sorted(
            (record for record in candidates if record.get("split") in {"val", "test"}),
            key=lambda record: (
                ai_pre_review(record)[0],
                str(record["split"]),
                str(record["task_family"]),
                str(record["id"]),
            ),
        )
        if len(self.records) != 600:
            raise ReviewValidationError(
                f"expected 600 val/test records, found {len(self.records)}"
            )
        self.by_id = {str(record["id"]): record for record in self.records}
        if len(self.by_id) != len(self.records):
            raise ReviewValidationError("evaluation record IDs are not unique")
        self.decisions = self._load_decisions()
        self.loggers["data"].info(
            "loaded evaluation candidates dataset=%s count=%d existing_decisions=%d",
            dataset_path,
            len(self.records),
            len(self.decisions),
        )

    def _load_decisions(self) -> dict[str, dict[str, Any]]:
        if not self.decisions_path.exists():
            return {}
        decisions: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(self.decisions_path):
            record_id = str(row.get("record_id", ""))
            if record_id not in self.by_id:
                raise ReviewValidationError(
                    f"decision refers to unknown record {record_id!r}"
                )
            if row.get("schema_version") != SCHEMA_VERSION:
                raise ReviewValidationError(
                    f"decision {record_id} has unsupported schema version"
                )
            if row.get("candidate_sha256") != candidate_digest(self.by_id[record_id]):
                raise ReviewValidationError(
                    f"candidate changed after decision was recorded: {record_id}"
                )
            self._validate_decision_payload(row, existing_record=self.by_id[record_id])
            decisions[record_id] = row
        return decisions

    @staticmethod
    def _validate_text(value: Any, field: str, maximum: int) -> str:
        if not isinstance(value, str):
            raise ReviewValidationError(f"{field} must be text")
        value = value.strip()
        if not value:
            raise ReviewValidationError(f"{field} cannot be empty")
        if len(value) > maximum:
            raise ReviewValidationError(f"{field} exceeds {maximum} characters")
        return value

    def _validate_decision_payload(
        self,
        payload: dict[str, Any],
        *,
        existing_record: dict[str, Any],
    ) -> dict[str, str]:
        decision = str(payload.get("decision", ""))
        if decision not in ALLOWED_DECISIONS:
            raise ReviewValidationError(f"invalid decision {decision!r}")
        reviewer = self._validate_text(payload.get("reviewer"), "reviewer", 100)
        notes = str(payload.get("notes", "")).strip()
        if len(notes) > 2000:
            raise ReviewValidationError("notes exceed 2000 characters")
        if decision == "rejected" and not notes:
            raise ReviewValidationError("rejected records require a reason in notes")
        question = str(payload.get("question", existing_record["question"])).strip()
        answer = str(payload.get("answer", existing_record["answer"])).strip()
        if decision == "approved" and (
            question != existing_record["question"]
            or answer != existing_record["answer"]
        ):
            raise ReviewValidationError(
                "edited content must use modified approval, not plain approval"
            )
        if decision == "modified_approved":
            question = self._validate_text(question, "question", 5000)
            answer = self._validate_text(answer, "answer", 5000)
            if (
                question == existing_record["question"]
                and answer == existing_record["answer"]
            ):
                raise ReviewValidationError(
                    "modified approval requires a changed question or answer"
                )
            if not notes:
                raise ReviewValidationError("modified approval requires change notes")
        return {
            "decision": decision,
            "reviewer": reviewer,
            "notes": notes,
            "question": question,
            "answer": answer,
        }

    def _persist(self) -> None:
        rows = [
            self.decisions[record["id"]]
            for record in self.records
            if record["id"] in self.decisions
        ]
        atomic_write_text(self.decisions_path, jsonl_text(rows))

    def submit(self, record_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if record_id not in self.by_id:
                raise ReviewValidationError(f"unknown record {record_id!r}")
            record = self.by_id[record_id]
            try:
                validated = self._validate_decision_payload(
                    payload, existing_record=record
                )
            except ReviewValidationError:
                self.loggers["validation"].warning(
                    "human decision rejected record_id=%s error_type=ReviewValidationError",
                    record_id,
                )
                raise
            decision = {
                "schema_version": SCHEMA_VERSION,
                "record_id": record_id,
                "candidate_sha256": candidate_digest(record),
                "decision": validated["decision"],
                "reviewer": validated["reviewer"],
                "reviewed_at": datetime.now(timezone.utc).astimezone().isoformat(
                    timespec="seconds"
                ),
                "notes": validated["notes"],
            }
            if validated["decision"] == "modified_approved":
                decision["question"] = validated["question"]
                decision["answer"] = validated["answer"]
            self.decisions[record_id] = decision
            self._persist()
            self.loggers["data"].info(
                "saved human decision record_id=%s decision=%s reviewed=%d remaining=%d",
                record_id,
                decision["decision"],
                len(self.decisions),
                len(self.records) - len(self.decisions),
            )
            return decision

    def _matches_filter(self, record: dict[str, Any], filter_name: str) -> bool:
        decision = self.decisions.get(str(record["id"]))
        precheck = ai_pre_review(record)[1]
        if filter_name == "all":
            return True
        if filter_name == "pending":
            return decision is None
        if filter_name == "transformed":
            return precheck == "review_transformed_task"
        if filter_name == "low_risk":
            return precheck == "low_risk_human_review"
        raise ReviewValidationError(f"invalid filter {filter_name!r}")

    def filtered_records(self, filter_name: str) -> list[dict[str, Any]]:
        if filter_name not in ALLOWED_FILTERS:
            raise ReviewValidationError(f"invalid filter {filter_name!r}")
        return [record for record in self.records if self._matches_filter(record, filter_name)]

    def status(self) -> dict[str, Any]:
        decision_counts = Counter(
            decision["decision"] for decision in self.decisions.values()
        )
        precheck_counts = Counter(ai_pre_review(record)[1] for record in self.records)
        return {
            "total": len(self.records),
            "reviewed": len(self.decisions),
            "pending": len(self.records) - len(self.decisions),
            "decision_counts": dict(sorted(decision_counts.items())),
            "precheck_counts": dict(sorted(precheck_counts.items())),
            "complete": len(self.decisions) == len(self.records),
            "all_approved": (
                len(self.decisions) == len(self.records)
                and all(
                    row["decision"] in {"approved", "modified_approved"}
                    for row in self.decisions.values()
                )
            ),
        }

    def record_payload(self, index: int, filter_name: str) -> dict[str, Any]:
        records = self.filtered_records(filter_name)
        if not records:
            return {"record": None, "index": 0, "filtered_total": 0, "status": self.status()}
        index = max(0, min(index, len(records) - 1))
        record = records[index]
        precheck = ai_pre_review(record)[1]
        return {
            "record": {
                "id": record["id"],
                "split": record["split"],
                "task_family": record["task_family"],
                "question": record["question"],
                "answer": record["answer"],
                "evidence": record.get("evidence", {}).get("text", ""),
                "chapter": record.get("evidence", {}).get("chapter"),
                "source_subcategory": record["origin"].get("source_subcategory"),
                "source_entities": record["origin"].get("source_entities", []),
                "repair_flags": record["origin"].get("repair_flags", []),
                "automatic_repairs": record["origin"].get("automatic_repairs", []),
                "ai_precheck": precheck,
                "decision": self.decisions.get(str(record["id"])),
            },
            "index": index,
            "filtered_total": len(records),
            "status": self.status(),
        }


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SFT v4 真人审核</title>
<style>
:root{color-scheme:light;--bg:#f4f1ea;--card:#fffdf8;--ink:#222;--muted:#706b63;--line:#d8d1c5;--green:#1f7a4d;--red:#a93b36;--blue:#245e9b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1120px;margin:0 auto;padding:24px}.top{display:flex;gap:16px;align-items:center;justify-content:space-between;flex-wrap:wrap}
h1{font-size:24px;margin:0}.muted{color:var(--muted)}.progress{height:10px;background:#ddd5c9;border-radius:10px;overflow:hidden;margin:14px 0 20px}.progress>div{height:100%;background:var(--green);width:0}
.toolbar,.actions{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin:14px 0;box-shadow:0 2px 10px #0000000a}
.meta{display:flex;gap:8px;flex-wrap:wrap}.tag{background:#eee8dd;border-radius:999px;padding:3px 9px;font-size:12px}.tag.warn{background:#f8dfb5}.tag.good{background:#d9eddf}
label{display:block;font-weight:650;margin:12px 0 6px}textarea,input,select{font:inherit;border:1px solid var(--line);border-radius:8px;background:white;padding:10px}textarea{width:100%;min-height:105px;resize:vertical}.evidence{white-space:pre-wrap;background:#f2eee6;border-left:4px solid #bcae97;padding:12px;border-radius:6px;max-height:260px;overflow:auto}
button{font:inherit;border:0;border-radius:8px;padding:10px 15px;cursor:pointer;background:#ddd5c9}button.primary{background:var(--green);color:white}button.modify{background:var(--blue);color:white}button.reject{background:var(--red);color:white}button:disabled{opacity:.45;cursor:not-allowed}
.notice{min-height:24px;font-weight:600}.notice.error{color:var(--red)}.notice.ok{color:var(--green)}kbd{border:1px solid #bbb;border-bottom-width:2px;border-radius:4px;padding:1px 5px;background:white;font-size:12px}
@media(max-width:700px){main{padding:14px}.card{padding:14px}.actions button{flex:1}}
</style>
</head>
<body><main>
<div class="top"><div><h1>SFT v4 真人审核</h1><div class="muted">决定只写入独立文件，不修改教师原始数据。</div></div><div><b id="count">0 / 600</b> <span id="pending" class="muted"></span></div></div>
<div class="progress"><div id="bar"></div></div>
<div class="toolbar">
<label style="margin:0">审核人 <input id="reviewer" maxlength="100" placeholder="请输入真实姓名或固定审核代号"></label>
<label style="margin:0">队列 <select id="filter"><option value="pending">仅待审核</option><option value="transformed">任务改写</option><option value="low_risk">低风险</option><option value="all">全部记录</option></select></label>
<button id="prev">← 上一条</button><button id="next">下一条 →</button><span id="position" class="muted"></span>
</div>
<section id="empty" class="card" hidden><h2>当前队列为空</h2><p>若待审核为0，请切换到“全部记录”查看结果。只有全部记录均通过或修改后通过，才可以进入数据冻结。</p></section>
<section id="record" hidden>
<div class="card"><div class="meta" id="meta"></div><label>问题</label><textarea id="question"></textarea><label>答案</label><textarea id="answer"></textarea></div>
<div class="card"><label>原文证据</label><div id="evidence" class="evidence"></div><label>修改/拒绝说明</label><textarea id="notes" placeholder="修改后通过或拒绝时必填；普通通过可选。"></textarea></div>
<div class="actions"><button class="primary" id="approve">通过 <kbd>Ctrl+Enter</kbd></button><button class="modify" id="modify">修改后通过 <kbd>Ctrl+M</kbd></button><button class="reject" id="reject">拒绝 <kbd>Ctrl+R</kbd></button></div>
</section>
<p id="notice" class="notice"></p>
<p class="muted">判断标准：问题是否自然且单一；答案是否完全被证据支持；不要凭印象补充全书事实。快捷键必须同时按 Ctrl，避免误操作。</p>
</main>
<script>
const $=id=>document.getElementById(id);let index=0,current=null;
$('reviewer').value=localStorage.getItem('sftReviewer')||'';
$('reviewer').addEventListener('input',()=>localStorage.setItem('sftReviewer',$('reviewer').value));
function tag(text,kind=''){const s=document.createElement('span');s.className='tag '+kind;s.textContent=text;return s}
function notice(text,kind=''){const n=$('notice');n.textContent=text;n.className='notice '+kind}
async function load(){notice('');const f=$('filter').value;const res=await fetch(`/api/record?index=${index}&filter=${encodeURIComponent(f)}`);const p=await res.json();if(!res.ok){notice(p.error||'加载失败','error');return}const st=p.status;$('count').textContent=`${st.reviewed} / ${st.total}`;$('pending').textContent=`待审核 ${st.pending}`;$('bar').style.width=`${st.reviewed/st.total*100}%`;$('position').textContent=p.filtered_total?`${p.index+1} / ${p.filtered_total}`:'0 / 0';current=p.record;$('empty').hidden=!!current;$('record').hidden=!current;if(!current)return;index=p.index;
const m=$('meta');m.replaceChildren(tag(current.split),tag(current.task_family),tag(current.source_subcategory),tag(current.ai_precheck,current.ai_precheck.includes('transformed')?'warn':'good'));for(const x of current.repair_flags)m.append(tag(x,'warn'));$('question').value=current.decision?.question||current.question;$('answer').value=current.decision?.answer||current.answer;$('evidence').textContent=current.evidence;$('notes').value=current.decision?.notes||'';if(current.decision)m.append(tag(`已记录：${current.decision.decision}`,'good'))}
async function submit(decision){if(!current)return;const reviewer=$('reviewer').value.trim();if(!reviewer){notice('请先填写真实审核人或固定审核代号。','error');$('reviewer').focus();return}const body={record_id:current.id,decision,reviewer,notes:$('notes').value,question:$('question').value,answer:$('answer').value};const res=await fetch('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const p=await res.json();if(!res.ok){notice(p.error||'保存失败','error');return}notice('审核决定已原子保存。','ok');if($('filter').value==='pending')index=0;else index++;await load()}
$('filter').addEventListener('change',()=>{index=0;load()});$('prev').onclick=()=>{index=Math.max(0,index-1);load()};$('next').onclick=()=>{index++;load()};$('approve').onclick=()=>submit('approved');$('modify').onclick=()=>submit('modified_approved');$('reject').onclick=()=>submit('rejected');
document.addEventListener('keydown',e=>{if(!e.ctrlKey)return;if(e.key==='Enter'){e.preventDefault();submit('approved')}else if(e.key.toLowerCase()==='m'){e.preventDefault();submit('modified_approved')}else if(e.key.toLowerCase()==='r'){e.preventDefault();submit('rejected')}});load();
</script></body></html>"""


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "SftReview/1.0"

    @property
    def review_server(self) -> "ReviewServer":
        return self.server  # type: ignore[return-value]

    def log_message(self, format: str, *args: Any) -> None:
        self.review_server.loggers["ui"].debug(
            "http client=%s method=%s path=%s status=%s",
            self.client_address[0],
            self.command,
            self.path.split("?", 1)[0],
            args[1] if len(args) > 1 else "unknown",
        )

    def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, error: Exception, status: HTTPStatus) -> None:
        self.review_server.loggers["validation"].warning(
            "request rejected method=%s path=%s error_type=%s",
            self.command,
            self.path.split("?", 1)[0],
            type(error).__name__,
        )
        self._json({"error": str(error)}, status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                body = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; style-src 'unsafe-inline'; "
                    "script-src 'unsafe-inline'; connect-src 'self'; "
                    "object-src 'none'; base-uri 'none'",
                )
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/health":
                self._json({"status": "ok"})
                return
            if parsed.path == "/api/status":
                self._json(self.review_server.store.status())
                return
            if parsed.path == "/api/record":
                query = parse_qs(parsed.query)
                index = int(query.get("index", ["0"])[0])
                filter_name = query.get("filter", ["pending"])[0]
                self._json(self.review_server.store.record_payload(index, filter_name))
                return
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (ReviewValidationError, ValueError) as error:
            self._error(error, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # pragma: no cover - safety boundary
            self.review_server.loggers["validation"].exception(
                "GET failed path=%s", parsed.path
            )
            self._error(RuntimeError("internal server error"), HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/decision":
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                raise ReviewValidationError("invalid request size")
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ReviewValidationError("request body must be an object")
            record_id = str(payload.pop("record_id", ""))
            decision = self.review_server.store.submit(record_id, payload)
            self._json({"saved": True, "decision": decision, "status": self.review_server.store.status()})
        except (ReviewValidationError, ValueError, json.JSONDecodeError) as error:
            self._error(error, HTTPStatus.BAD_REQUEST)
        except Exception:  # pragma: no cover - safety boundary
            self.review_server.loggers["validation"].exception(
                "POST failed path=%s", parsed.path
            )
            self._error(RuntimeError("internal server error"), HTTPStatus.INTERNAL_SERVER_ERROR)


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        store: ReviewStore,
        loggers: dict[str, logging.Logger],
    ) -> None:
        super().__init__(address, ReviewHandler)
        self.store = store
        self.loggers = loggers


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/sft/v4_teacher_repair/sft_v4_teacher_candidates.jsonl"),
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("data/sft/v4_teacher_repair/human_review_decisions.jsonl"),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("data/sft/v4_teacher_repair/review_logs"),
    )
    parser.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    loggers = configure_review_logging(args.log_dir)
    try:
        store = ReviewStore(args.dataset, args.decisions, loggers)
        server = ReviewServer((args.host, args.port), store, loggers)
    except (OSError, SftV4ValidationError, ReviewValidationError) as error:
        loggers["validation"].exception("review server startup failed")
        print(f"审核工具启动失败：{error}")
        return 2
    loggers["ui"].info(
        "review server started host=%s port=%d pending=%d",
        args.host,
        args.port,
        store.status()["pending"],
    )
    print(f"审核工具已启动：http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止；审核结果会在每次点击后立即保存。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        loggers["ui"].info("review server stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
