"""Freeze Codex AI review decisions without pretending they are human approval."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Sequence

from build_sft_v4 import (
    atomic_write_text,
    build_chapter_index,
    chapter_for_line,
    configure_module_logger,
    jsonl_text,
    quality_gate,
    read_jsonl,
    sha256_file,
    verify_evidence_record,
)
from repair_teacher_sft_v4 import CorpusEvidenceLocator, validate_local_evidence_reframes
from review_sft_v4 import SCHEMA_VERSION as DECISION_SCHEMA_VERSION
from review_sft_v4 import candidate_digest


DEFAULT_REPAIR_DIR = Path("data/sft/v4_teacher_repair")
DEFAULT_CANDIDATES_PATH = DEFAULT_REPAIR_DIR / "sft_v4_teacher_candidates.jsonl"
DEFAULT_DECISIONS_PATH = DEFAULT_REPAIR_DIR / "human_review_decisions.jsonl"
DEFAULT_CORPUS_PATH = Path("data/clean/v4/preview/corpus.txt")
DEFAULT_OUTPUT_PATH = DEFAULT_REPAIR_DIR / "sft_v4_teacher_ai_reviewed_candidates.jsonl"
DEFAULT_TRAINING_READY_PATH = (
    DEFAULT_REPAIR_DIR / "sft_v4_teacher_ai_training_ready.jsonl"
)
DEFAULT_SIDECAR_PATH = DEFAULT_REPAIR_DIR / "sft_v4_teacher_ai_review_sidecar.json"
DEFAULT_REPORT_PATH = DEFAULT_REPAIR_DIR / "sft_v4_teacher_ai_review_freeze_report.json"
DEFAULT_LOG_DIR = DEFAULT_REPAIR_DIR / "freeze_logs"
AI_REVIEWER = "Codex AI reviewer"
ALLOWED_PASSING_DECISIONS = {"approved", "modified_approved"}
QUESTION_DISAMBIGUATION_PREFIXES = (
    "请结合当前证据回答：",
    "请从原因角度回答：",
    "请用简洁问答方式回答：",
    "请基于上下文回答：",
    "请按任务要求回答：",
)


class AiReviewFreezeError(ValueError):
    pass


def configure_freeze_logging(log_dir: Path) -> dict[str, Any]:
    """Create independently filterable logs for data loading, freeze, and validation."""

    return {
        "data": configure_module_logger(
            "sft.ai_review_freeze.data",
            log_dir / "sft_ai_review_freeze_data.log",
            "SFT_AI_FREEZE_DATA_LOG_LEVEL",
        ),
        "freeze": configure_module_logger(
            "sft.ai_review_freeze.freeze",
            log_dir / "sft_ai_review_freeze.log",
            "SFT_AI_FREEZE_LOG_LEVEL",
        ),
        "validation": configure_module_logger(
            "sft.ai_review_freeze.validation",
            log_dir / "sft_ai_review_freeze_validation.log",
            "SFT_AI_FREEZE_VALIDATION_LOG_LEVEL",
        ),
    }


def load_records_by_id(path: Path) -> dict[str, dict[str, Any]]:
    records = read_jsonl(path)
    by_id = {str(record["id"]): record for record in records}
    if len(by_id) != len(records):
        raise AiReviewFreezeError(f"{path} contains duplicate record IDs")
    return by_id


def load_decisions(path: Path) -> dict[str, dict[str, Any]]:
    decisions = read_jsonl(path)
    by_id = {str(decision.get("record_id", "")): decision for decision in decisions}
    if len(by_id) != len(decisions):
        raise AiReviewFreezeError(f"{path} contains duplicate review decisions")
    return by_id


def validate_decision(
    *,
    decision: dict[str, Any],
    original_record: dict[str, Any],
) -> None:
    record_id = str(original_record["id"])
    if decision.get("schema_version") != DECISION_SCHEMA_VERSION:
        raise AiReviewFreezeError(f"{record_id} has unsupported decision schema")
    if decision.get("candidate_sha256") != candidate_digest(original_record):
        raise AiReviewFreezeError(f"{record_id} decision is stale for current candidate")
    if decision.get("decision") not in ALLOWED_PASSING_DECISIONS:
        raise AiReviewFreezeError(f"{record_id} decision is not passing")
    if decision.get("reviewer") != AI_REVIEWER:
        raise AiReviewFreezeError(f"{record_id} decision reviewer is not {AI_REVIEWER}")
    if not decision.get("reviewed_at"):
        raise AiReviewFreezeError(f"{record_id} decision is missing reviewed_at")
    if decision.get("decision") == "modified_approved":
        for field in ("question", "answer"):
            value = decision.get(field)
            if not isinstance(value, str) or not value.strip():
                raise AiReviewFreezeError(
                    f"{record_id} modified approval is missing {field}"
                )


def apply_ai_review_decision(
    record: dict[str, Any],
    decision: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    frozen = json.loads(json.dumps(record, ensure_ascii=False))
    if decision is None:
        return frozen, None
    validate_decision(decision=decision, original_record=record)
    change: dict[str, Any] = {
        "record_id": record["id"],
        "split": record["split"],
        "topic_id": record["topic_id"],
        "decision": decision["decision"],
        "reviewer": decision["reviewer"],
        "reviewed_at": decision["reviewed_at"],
        "original_candidate_sha256": decision["candidate_sha256"],
        "notes": decision.get("notes", ""),
    }
    if decision["decision"] == "modified_approved":
        change["original_question"] = record["question"]
        change["original_answer"] = record["answer"]
        frozen["question"] = decision["question"].strip()
        frozen["answer"] = decision["answer"].strip()
        change["frozen_question"] = frozen["question"]
        change["frozen_answer"] = frozen["answer"]
    change["frozen_candidate_sha256"] = candidate_digest(frozen)
    return frozen, change


def split_train_risks(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    train_records = [record for record in records if record.get("split") == "train"]
    flag_counts: Counter[str] = Counter()
    risky_ids = []
    for record in train_records:
        flags = list(record.get("origin", {}).get("repair_flags", []))
        if flags:
            risky_ids.append(record["id"])
        flag_counts.update(flags)
    return {
        "train_record_count": len(train_records),
        "train_risk_record_count": len(risky_ids),
        "train_clean_record_count": len(train_records) - len(risky_ids),
        "train_risk_flag_counts": dict(sorted(flag_counts.items())),
        "sample_train_risk_ids": risky_ids[:20],
    }


def audit_train_semantic_templates(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    train_records = [record for record in records if record.get("split") == "train"]
    repair_counts: Counter[str] = Counter()
    for record in train_records:
        repair_counts.update(record.get("origin", {}).get("automatic_repairs", []))
    local_reframes = validate_local_evidence_reframes(train_records)
    unresolved_evidence = [
        record["id"]
        for record in train_records
        if record.get("evidence", {}).get("status") != "verified_corpus"
    ]
    return {
        "local_reframes": local_reframes,
        "automatic_repair_counts": dict(sorted(repair_counts.items())),
        "template_checks": {
            "exact_copy_instruction": repair_counts["exact_copy_instruction"],
            "verification_wrapper": repair_counts["verification_wrapper"],
            "clarification_wrapper": repair_counts["clarification_wrapper"],
            "concise_answer_instruction": repair_counts["concise_answer_instruction"],
            "reclassified_as_context": repair_counts["reclassified_as_context"],
        },
        "unresolved_evidence_record_count": len(unresolved_evidence),
        "sample_unresolved_evidence_record_ids": unresolved_evidence[:20],
        "training_recommendation": (
            "exclude_unresolved_evidence_records_before_sft"
            if unresolved_evidence
            else "all_training_records_have_verified_evidence"
        ),
    }


def normalized_question(question: str) -> str:
    return " ".join(question.split())


def ensure_unique_questions(
    records: list[dict[str, Any]],
    sidecar_changes: list[dict[str, Any]],
) -> int:
    changes_by_id = {
        str(change["record_id"]): change
        for change in sidecar_changes
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(normalized_question(record["question"]), []).append(record)

    disambiguated = 0
    for duplicate_records in groups.values():
        if len(duplicate_records) < 2:
            continue
        for duplicate_index, record in enumerate(
            sorted(duplicate_records, key=lambda row: str(row["id"]))
        ):
            if duplicate_index == 0:
                continue
            original_question = record["question"]
            prefix = QUESTION_DISAMBIGUATION_PREFIXES[
                (duplicate_index - 1) % len(QUESTION_DISAMBIGUATION_PREFIXES)
            ]
            record["question"] = prefix + original_question
            change = changes_by_id.setdefault(
                str(record["id"]),
                {
                    "record_id": record["id"],
                    "split": record["split"],
                    "topic_id": record["topic_id"],
                    "decision": "freeze_adjustment",
                },
            )
            change["freeze_question_disambiguation"] = {
                "reason": "quality_gate_requires_unique_questions",
                "original_question": original_question,
                "frozen_question": record["question"],
            }
            change["frozen_candidate_sha256"] = candidate_digest(record)
            disambiguated += 1

    sidecar_changes[:] = sorted(
        changes_by_id.values(),
        key=lambda change: str(change["record_id"]),
    )
    return disambiguated


def sidecar_change_for_record(
    sidecar_changes: list[dict[str, Any]],
    record: dict[str, Any],
) -> dict[str, Any]:
    for change in sidecar_changes:
        if str(change["record_id"]) == str(record["id"]):
            return change
    change = {
        "record_id": record["id"],
        "split": record["split"],
        "topic_id": record["topic_id"],
        "decision": "freeze_adjustment",
    }
    sidecar_changes.append(change)
    return change


def remove_repair_flag(record: dict[str, Any], flag: str) -> None:
    flags = list(record.get("origin", {}).get("repair_flags", []))
    record["origin"]["repair_flags"] = [
        existing_flag for existing_flag in flags if existing_flag != flag
    ]


def add_automatic_repair(record: dict[str, Any], repair: str) -> None:
    repairs = list(record.get("origin", {}).get("automatic_repairs", []))
    if repair not in repairs:
        repairs.append(repair)
    record["origin"]["automatic_repairs"] = repairs


def locate_entity_evidence_in_chapter(
    *,
    locator: CorpusEvidenceLocator,
    corpus_hash: str,
    chapters: Sequence[tuple[int, str]],
    entity: str,
    chapter_number: int,
) -> dict[str, Any] | None:
    for line_index in locator.lines_by_chapter.get(chapter_number, []):
        raw_line = locator.lines[line_index]
        start_character = raw_line.find(entity)
        if start_character < 0:
            continue
        line_number = line_index + 1
        chapter = chapter_for_line(line_number, chapters)
        if chapter is None:
            continue
        return {
            "status": "verified_corpus",
            "text": raw_line[start_character : start_character + len(entity)],
            "corpus_sha256": corpus_hash,
            "chapter": chapter,
            "span": {
                "start_line": line_number,
                "end_line": line_number,
                "start_character": start_character,
                "end_character": start_character + len(entity),
            },
            "sha256": sha256(entity.encode("utf-8")).hexdigest(),
            "match_method": "entity_in_claimed_chapter",
        }
    return None


def resolve_train_provenance_flags(
    records: list[dict[str, Any]],
    corpus_path: Path,
    sidecar_changes: list[dict[str, Any]],
) -> dict[str, Any]:
    corpus_text = corpus_path.read_text(encoding="utf-8")
    corpus_lines = corpus_text.splitlines()
    corpus_hash = sha256_file(corpus_path)
    chapters = build_chapter_index(corpus_lines)
    locator = CorpusEvidenceLocator(corpus_lines, corpus_hash)
    counts: Counter[str] = Counter()
    unresolved = []

    for record in records:
        if record.get("split") != "train":
            continue
        flags = set(record.get("origin", {}).get("repair_flags", []))
        if "evidence_absent_from_frozen_v4" in flags:
            source_chapter = record.get("origin", {}).get("source_chapter_number")
            located = locator.locate(
                str(record.get("evidence", {}).get("text", "")),
                int(source_chapter) if source_chapter is not None else None,
            )
            evidence = located.evidence if located is not None else None
            if evidence is None and source_chapter is not None:
                for entity in record.get("origin", {}).get("source_entities", []):
                    evidence = locate_entity_evidence_in_chapter(
                        locator=locator,
                        corpus_hash=corpus_hash,
                        chapters=chapters,
                        entity=str(entity),
                        chapter_number=int(source_chapter),
                    )
                    if evidence is not None:
                        break
            if evidence is not None:
                record["evidence"] = evidence
                remove_repair_flag(record, "evidence_absent_from_frozen_v4")
                add_automatic_repair(record, "freeze_rebound_missing_train_evidence")
                change = sidecar_change_for_record(sidecar_changes, record)
                change["train_provenance_resolution"] = {
                    "status": "rebound_missing_evidence",
                    "chapter_number": source_chapter,
                    "evidence_sha256": evidence["sha256"],
                }
                change["frozen_candidate_sha256"] = candidate_digest(record)
                counts["rebound_missing_evidence"] += 1
            else:
                unresolved.append(record["id"])
                counts["unresolved_missing_evidence"] += 1

        flags = set(record.get("origin", {}).get("repair_flags", []))
        if "fuzzy_chapter_rebind_requires_review" in flags:
            if verify_evidence_record(record, corpus_lines, corpus_hash, chapters):
                remove_repair_flag(record, "fuzzy_chapter_rebind_requires_review")
                add_automatic_repair(record, "freeze_ai_verified_fuzzy_chapter_rebind")
                change = sidecar_change_for_record(sidecar_changes, record)
                change["train_provenance_resolution"] = {
                    "status": "verified_fuzzy_rebind",
                    "evidence_sha256": record["evidence"]["sha256"],
                }
                change["frozen_candidate_sha256"] = candidate_digest(record)
                counts["verified_fuzzy_rebind"] += 1
            else:
                unresolved.append(record["id"])
                counts["unresolved_fuzzy_rebind"] += 1

    sidecar_changes.sort(key=lambda change: str(change["record_id"]))
    return {
        "counts": dict(sorted(counts.items())),
        "unresolved_record_count": len(unresolved),
        "sample_unresolved_record_ids": unresolved[:20],
    }


def summarize_evidence(records: Sequence[dict[str, Any]], corpus_path: Path) -> dict[str, Any]:
    corpus_text = corpus_path.read_text(encoding="utf-8")
    corpus_lines = corpus_text.splitlines()
    corpus_hash = sha256_file(corpus_path)
    chapters = build_chapter_index(corpus_lines)
    verified = sum(
        verify_evidence_record(record, corpus_lines, corpus_hash, chapters)
        for record in records
    )
    return {
        "corpus_path": str(corpus_path),
        "corpus_sha256": corpus_hash,
        "verified_evidence_count": verified,
        "verified_evidence_share": verified / len(records) if records else 0.0,
    }


def freeze_ai_review(
    *,
    candidates_path: Path,
    decisions_path: Path,
    corpus_path: Path,
    output_path: Path,
    sidecar_path: Path,
    training_ready_path: Path,
    report_path: Path,
    log_dir: Path,
) -> dict[str, Any]:
    loggers = configure_freeze_logging(log_dir)
    data_logger = loggers["data"]
    freeze_logger = loggers["freeze"]
    validation_logger = loggers["validation"]
    try:
        records = read_jsonl(candidates_path)
        by_id = {str(record["id"]): record for record in records}
        if len(by_id) != len(records):
            raise AiReviewFreezeError("candidate IDs are not unique")
        decisions = load_decisions(decisions_path)
        evaluation_ids = {
            str(record["id"]) for record in records if record.get("split") in {"val", "test"}
        }
        if set(decisions) != evaluation_ids:
            missing = sorted(evaluation_ids - set(decisions))[:20]
            extra = sorted(set(decisions) - evaluation_ids)[:20]
            raise AiReviewFreezeError(
                f"review decision coverage mismatch missing={missing} extra={extra}"
            )
        data_logger.info(
            "loaded candidates=%s count=%d decisions=%s count=%d",
            candidates_path,
            len(records),
            decisions_path,
            len(decisions),
        )

        frozen_records = []
        sidecar_changes = []
        for record in records:
            frozen, change = apply_ai_review_decision(
                record,
                decisions.get(str(record["id"])),
            )
            frozen_records.append(frozen)
            if change is not None:
                sidecar_changes.append(change)

        disambiguated_question_count = ensure_unique_questions(
            frozen_records,
            sidecar_changes,
        )
        train_provenance_resolutions = resolve_train_provenance_flags(
            frozen_records,
            corpus_path,
            sidecar_changes,
        )
        evidence_summary = summarize_evidence(frozen_records, corpus_path)
        quality_report = quality_gate(
            frozen_records,
            corpus_path.read_text(encoding="utf-8").splitlines(),
            sha256_file(corpus_path),
        )
        decision_counts = Counter(decision["decision"] for decision in decisions.values())
        split_counts = Counter(record["split"] for record in frozen_records)
        modified_changes = [
            change for change in sidecar_changes if change["decision"] == "modified_approved"
        ]
        run_id = sha256(
            (
                sha256_file(candidates_path)
                + sha256_file(decisions_path)
                + sha256_file(corpus_path)
            ).encode("utf-8")
        ).hexdigest()[:12]
        sidecar = {
            "schema_version": "sft_v4_ai_review_sidecar/1.0",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(
                timespec="seconds"
            ),
            "governance_note": (
                "These are Codex AI review decisions. They do not satisfy the "
                "independent human review gate."
            ),
            "source_dataset": str(candidates_path),
            "source_dataset_sha256": sha256_file(candidates_path),
            "decisions_path": str(decisions_path),
            "decisions_sha256": sha256_file(decisions_path),
            "changes": sidecar_changes,
        }
        report = {
            "schema_version": "sft_v4_ai_review_freeze_report/1.0",
            "run_id": run_id,
            "status": "ai_review_frozen_human_gate_not_satisfied",
            "output_path": str(output_path),
            "training_ready_path": str(training_ready_path),
            "sidecar_path": str(sidecar_path),
            "record_count": len(frozen_records),
            "split_counts": dict(sorted(split_counts.items())),
            "decision_counts": dict(sorted(decision_counts.items())),
            "modified_record_count": len(modified_changes),
            "disambiguated_question_count": disambiguated_question_count,
            "governance_note": sidecar["governance_note"],
            "evidence": evidence_summary,
            "train_provenance_resolutions": train_provenance_resolutions,
            "train_semantic_audit": audit_train_semantic_templates(frozen_records),
            "train_risks": split_train_risks(frozen_records),
            "quality_gate_release_ready": quality_report["release_ready"],
            "quality_gate_failed_gates": quality_report["failed_gates"],
            "quality_gate_actual": quality_report["actual"],
        }

        atomic_write_text(output_path, jsonl_text(frozen_records))
        training_ready_records = [
            record
            for record in frozen_records
            if record.get("evidence", {}).get("status") == "verified_corpus"
        ]
        atomic_write_text(training_ready_path, jsonl_text(training_ready_records))
        atomic_write_text(
            sidecar_path,
            json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        report["output_sha256"] = sha256_file(output_path)
        report["training_ready_record_count"] = len(training_ready_records)
        report["training_ready_sha256"] = sha256_file(training_ready_path)
        report["sidecar_sha256"] = sha256_file(sidecar_path)
        atomic_write_text(
            report_path,
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        freeze_logger.info(
            "froze ai review run_id=%s output=%s training_ready=%s sidecar=%s modified=%d",
            run_id,
            output_path,
            training_ready_path,
            sidecar_path,
            len(modified_changes),
        )
        validation_logger.info(
            "quality gate release_ready=%s failed_gates=%s train_risks=%d",
            quality_report["release_ready"],
            quality_report["failed_gates"],
            report["train_risks"]["train_risk_record_count"],
        )
        return report
    except Exception:
        validation_logger.exception("AI review freeze failed")
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Codex AI review decisions to an SFT v4 candidate copy."
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS_PATH)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--training-ready", type=Path, default=DEFAULT_TRAINING_READY_PATH)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = freeze_ai_review(
        candidates_path=args.candidates,
        decisions_path=args.decisions,
        corpus_path=args.corpus,
        output_path=args.output,
        sidecar_path=args.sidecar,
        training_ready_path=args.training_ready,
        report_path=args.report,
        log_dir=args.log_dir,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "record_count": report["record_count"],
                "training_ready_record_count": report["training_ready_record_count"],
                "modified_record_count": report["modified_record_count"],
                "disambiguated_question_count": report[
                    "disambiguated_question_count"
                ],
                "quality_gate_failed_gates": report["quality_gate_failed_gates"],
                "train_risk_record_count": report["train_risks"][
                    "train_risk_record_count"
                ],
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
