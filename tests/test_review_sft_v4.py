import json
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from build_sft_v4 import jsonl_text
from review_sft_v4 import (
    ReviewStore,
    ReviewValidationError,
    configure_review_logging,
)


class ReviewSftV4Tests(unittest.TestCase):
    def make_record(self, index, split, transformed=False):
        flags = ["transformed_task_requires_review"] if transformed else []
        return {
            "id": f"record-{index}",
            "split": split,
            "task_family": "direct_fact",
            "question": f"问题{index}",
            "answer": f"答案{index}",
            "topic_id": f"topic-{index}",
            "fact_id": f"fact-{index}",
            "group_id": f"group-{index}",
            "evidence": {
                "status": "verified_corpus",
                "text": f"证据{index}",
                "chapter": {"title": f"第{index}章", "heading_line": index},
            },
            "origin": {
                "source_subcategory": "fact",
                "source_entities": [],
                "repair_flags": flags,
                "automatic_repairs": [],
            },
            "review": {"status": "pending", "reviewer": None, "reviewed_at": None},
        }

    def make_store(self, root):
        records = [self.make_record(0, "train")]
        for index in range(1, 601):
            records.append(
                self.make_record(
                    index,
                    "val" if index <= 300 else "test",
                    transformed=index <= 406,
                )
            )
        dataset = root / "candidates.jsonl"
        dataset.write_text(jsonl_text(records), encoding="utf-8")
        logger = logging.getLogger(f"review-test-{id(root)}")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        return ReviewStore(
            dataset,
            root / "decisions.jsonl",
            {"ui": logger, "data": logger, "validation": logger},
        )

    def test_loads_only_evaluation_and_prioritizes_transformed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(Path(directory))
            self.assertEqual(store.status()["total"], 600)
            self.assertEqual(store.status()["precheck_counts"]["review_transformed_task"], 406)
            first = store.record_payload(0, "all")["record"]
            self.assertEqual(first["ai_precheck"], "review_transformed_task")

    def test_decision_is_atomic_persistent_and_does_not_mutate_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self.make_store(root)
            record = store.record_payload(0, "all")["record"]
            dataset_before = store.dataset_path.read_bytes()
            saved = store.submit(
                record["id"],
                {"decision": "approved", "reviewer": "reviewer-1", "notes": ""},
            )
            self.assertEqual(saved["decision"], "approved")
            self.assertEqual(dataset_before, store.dataset_path.read_bytes())
            self.assertFalse(store.decisions_path.with_suffix(".jsonl.tmp").exists())
            reloaded = ReviewStore(
                store.dataset_path,
                store.decisions_path,
                store.loggers,
            )
            self.assertEqual(reloaded.status()["reviewed"], 1)

    def test_modified_approval_and_rejection_require_explanation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(Path(directory))
            record = store.record_payload(0, "all")["record"]
            with self.assertRaisesRegex(ReviewValidationError, "requires a changed"):
                store.submit(
                    record["id"],
                    {
                        "decision": "modified_approved",
                        "reviewer": "reviewer-1",
                        "notes": "修改",
                        "question": record["question"],
                        "answer": record["answer"],
                    },
                )
            with self.assertRaisesRegex(ReviewValidationError, "require a reason"):
                store.submit(
                    record["id"],
                    {"decision": "rejected", "reviewer": "reviewer-1", "notes": ""},
                )
            with self.assertRaisesRegex(ReviewValidationError, "modified approval"):
                store.submit(
                    record["id"],
                    {
                        "decision": "approved",
                        "reviewer": "reviewer-1",
                        "notes": "",
                        "question": record["question"] + "改",
                        "answer": record["answer"],
                    },
                )

    def test_candidate_hash_prevents_stale_decision_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self.make_store(root)
            record = store.record_payload(0, "all")["record"]
            store.submit(
                record["id"],
                {"decision": "approved", "reviewer": "reviewer-1", "notes": ""},
            )
            rows = [json.loads(line) for line in store.decisions_path.read_text().splitlines()]
            rows[0]["candidate_sha256"] = "0" * 64
            store.decisions_path.write_text(jsonl_text(rows), encoding="utf-8")
            with self.assertRaisesRegex(ReviewValidationError, "candidate changed"):
                ReviewStore(store.dataset_path, store.decisions_path, store.loggers)

    def test_module_logs_are_separate_and_do_not_include_review_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                os.environ,
                {
                    "SFT_V4_CONSOLE_LOG": "0",
                    "SFT_REVIEW_UI_LOG_LEVEL": "INFO",
                    "SFT_REVIEW_DATA_LOG_LEVEL": "INFO",
                    "SFT_REVIEW_VALIDATION_LOG_LEVEL": "WARNING",
                },
            ):
                loggers = configure_review_logging(root / "logs")
            base_store = self.make_store(root)
            store = ReviewStore(
                base_store.dataset_path,
                root / "logged_decisions.jsonl",
                loggers,
            )
            record = store.record_payload(0, "all")["record"]
            store.submit(
                record["id"],
                {
                    "decision": "approved",
                    "reviewer": "private-reviewer",
                    "notes": "token-secret-note",
                },
            )
            with self.assertRaises(ReviewValidationError):
                store.submit(
                    record["id"],
                    {
                        "decision": "rejected",
                        "reviewer": "private-reviewer",
                        "notes": "",
                    },
                )
            for logger in loggers.values():
                for handler in logger.handlers:
                    handler.flush()
            files = sorted(path.name for path in (root / "logs").glob("*.log"))
            self.assertEqual(
                files,
                [
                    "sft_review_data.log",
                    "sft_review_ui.log",
                    "sft_review_validation.log",
                ],
            )
            combined = "".join(
                path.read_text(encoding="utf-8") for path in (root / "logs").glob("*.log")
            )
            self.assertIn(record["id"], combined)
            self.assertNotIn("private-reviewer", combined)
            self.assertNotIn("token-secret-note", combined)


if __name__ == "__main__":
    unittest.main()
