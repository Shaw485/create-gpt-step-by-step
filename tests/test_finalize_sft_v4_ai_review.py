from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from build_sft_v4 import SCHEMA_VERSION, jsonl_text
from finalize_sft_v4_ai_review import (
    AI_REVIEWER,
    AiReviewFreezeError,
    apply_ai_review_decision,
    freeze_ai_review,
)
from review_sft_v4 import SCHEMA_VERSION as DECISION_SCHEMA_VERSION
from review_sft_v4 import candidate_digest


TEST_CORPUS_TEXT = "------------\n\n第一章 测试\n证据"
TEST_CORPUS_SHA256 = sha256(TEST_CORPUS_TEXT.encode("utf-8")).hexdigest()


def make_record(record_id: str, split: str, question: str | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": record_id,
        "question": question or f"原问题{record_id}？",
        "answer": "原答案。",
        "task_family": "direct_fact",
        "topic_id": f"topic:{record_id}",
        "fact_id": f"fact:{record_id}",
        "group_id": f"group:{record_id}",
        "split": split,
        "origin": {"repair_flags": []},
        "evidence": {
            "status": "verified_corpus",
            "text": "证据",
            "corpus_sha256": TEST_CORPUS_SHA256,
            "chapter": {"title": "第一章 测试", "heading_line": 3},
            "span": {
                "start_line": 4,
                "end_line": 4,
                "start_character": 0,
                "end_character": 2,
            },
            "sha256": sha256("证据".encode("utf-8")).hexdigest(),
        },
        "review": {
            "status": "pending",
            "reviewer": None,
            "reviewed_at": None,
            "notes": "Pending independent human review.",
        },
    }


def make_decision(record: dict, decision: str = "approved", **updates: str) -> dict:
    row = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "record_id": record["id"],
        "candidate_sha256": candidate_digest(record),
        "decision": decision,
        "reviewer": AI_REVIEWER,
        "reviewed_at": "2026-08-28T16:30:00+08:00",
        "notes": "fixture AI review",
    }
    row.update(updates)
    return row


class FinalizeSftV4AiReviewTests(unittest.TestCase):
    def test_modified_approval_updates_content_but_keeps_human_review_pending(self):
        record = make_record("v4_val", "val")
        decision = make_decision(
            record,
            "modified_approved",
            question="修改后问题？",
            answer="修改后答案。",
        )
        frozen, change = apply_ai_review_decision(record, decision)
        self.assertEqual(frozen["question"], "修改后问题？")
        self.assertEqual(frozen["answer"], "修改后答案。")
        self.assertEqual(frozen["review"]["status"], "pending")
        self.assertIsNotNone(change)
        assert change is not None
        self.assertEqual(change["decision"], "modified_approved")
        self.assertEqual(change["original_question"], "原问题v4_val？")

    def test_stale_candidate_hash_is_rejected(self):
        record = make_record("v4_val", "val")
        decision = make_decision(record)
        changed = dict(record)
        changed["question"] = "候选后来变了？"
        with self.assertRaisesRegex(AiReviewFreezeError, "stale"):
            apply_ai_review_decision(changed, decision)

    def test_freeze_requires_decisions_for_every_evaluation_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = make_record("v4_train", "train")
            val = make_record("v4_val", "val")
            test = make_record("v4_test", "test")
            candidates = root / "candidates.jsonl"
            decisions = root / "decisions.jsonl"
            corpus = root / "corpus.txt"
            candidates.write_text(jsonl_text([train, val, test]), encoding="utf-8")
            decisions.write_text(jsonl_text([make_decision(val)]), encoding="utf-8")
            corpus.write_text(TEST_CORPUS_TEXT, encoding="utf-8")
            with self.assertRaisesRegex(AiReviewFreezeError, "coverage mismatch"):
                freeze_ai_review(
                    candidates_path=candidates,
                    decisions_path=decisions,
                    corpus_path=corpus,
                    output_path=root / "out.jsonl",
                    sidecar_path=root / "sidecar.json",
                    training_ready_path=root / "training_ready.jsonl",
                    report_path=root / "report.json",
                    log_dir=root / "logs",
                )

    def test_freeze_writes_output_sidecar_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = make_record("v4_train", "train")
            val = make_record("v4_val", "val")
            test = make_record("v4_test", "test")
            candidates = root / "candidates.jsonl"
            decisions = root / "decisions.jsonl"
            corpus = root / "corpus.txt"
            output = root / "out.jsonl"
            sidecar = root / "sidecar.json"
            training_ready = root / "training_ready.jsonl"
            report = root / "report.json"
            candidates.write_text(jsonl_text([train, val, test]), encoding="utf-8")
            decisions.write_text(
                jsonl_text(
                    [
                        make_decision(
                            val,
                            "modified_approved",
                            question="修改后问题？",
                            answer="修改后答案。",
                        ),
                        make_decision(test),
                    ]
                ),
                encoding="utf-8",
            )
            corpus.write_text(TEST_CORPUS_TEXT, encoding="utf-8")
            result = freeze_ai_review(
                candidates_path=candidates,
                decisions_path=decisions,
                corpus_path=corpus,
                output_path=output,
                sidecar_path=sidecar,
                training_ready_path=training_ready,
                report_path=report,
                log_dir=root / "logs",
            )
            frozen_rows = [
                json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(frozen_rows[1]["question"], "修改后问题？")
            self.assertEqual(result["record_count"], 3)
            self.assertEqual(result["modified_record_count"], 1)
            self.assertTrue(sidecar.exists())
            self.assertTrue(training_ready.exists())
            self.assertTrue(report.exists())


if __name__ == "__main__":
    unittest.main()
