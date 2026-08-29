import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audit_sft_v7_semantic_density import (
    SemanticAuditError,
    audit_semantic_density,
    execute_audit,
    evidence_copy_ratio,
    load_jsonl,
    longest_contiguous_common_substring_length,
    main,
    normalize_overlap_text,
    portable_repo_path,
)
from sft_v7_vertical_catalog import CORE, DIMENSION_TOTALS, KNOWN_CORE_FACTS


SECRET_BODY = "绝不能出现在审计产物里的秘密正文"


def record(
    index: int,
    *,
    split: str,
    dimension: str = CORE,
    question: str = "萧炎是谁？",
    answer: str = "萧炎是萧战的儿子。",
    spans: list[str] | None = None,
    supervised_tokens: int = 12,
) -> dict:
    return {
        "schema_version": "sft_v7_vertical/1.0",
        "id": f"safe_{split}_{index}",
        "split": split,
        "primary_dimension": dimension,
        "task_family": "known_core_direct",
        "question": question,
        "answer": answer,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "encoding_audit": {"supervised_tokens": supervised_tokens},
        "answer_support": {
            "status": "supported" if spans else "not_applicable",
            "supporting_spans": spans or [],
        },
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


class SemanticDensityAuditTests(unittest.TestCase):
    def test_longest_contiguous_overlap_uses_documented_normalization(self):
        answer = normalize_overlap_text("原文：萧炎 是萧战的儿子。")
        span = normalize_overlap_text("旁白，萧炎是萧战的儿子！")
        self.assertEqual(
            longest_contiguous_common_substring_length(answer, span),
            len(normalize_overlap_text("萧炎是萧战的儿子")),
        )
        ratio, overlap, answer_length = evidence_copy_ratio(
            "萧炎是萧战的儿子。", ["记录说：萧炎是萧战的儿子！"]
        )
        self.assertEqual(overlap, answer_length)
        self.assertEqual(ratio, 1.0)

    def test_aggregate_success_contains_density_and_no_bodies(self):
        train = [
            record(
                1,
                split="train",
                answer=f"原文写道“{SECRET_BODY}”，其中涉及萧炎。",
                spans=[SECRET_BODY],
                supervised_tokens=10,
            )
        ]
        val = [
            record(
                2,
                split="val",
                dimension=next(d for d in DIMENSION_TOTALS if d != CORE),
                question=f"{SECRET_BODY}？",
                answer="萧炎是萧炎。",
                supervised_tokens=30,
            )
        ]
        report = audit_semantic_density({"train": train, "val": val})
        self.assertEqual(report["population"]["total_records"], 2)
        self.assertEqual(report["population"]["total_supervised_tokens"], 40)
        self.assertEqual(
            report["template_opening_analysis"]["records_matching_any_of_six"], 1
        )
        self.assertEqual(
            report["bare_core_question_coverage"]["matched_record_count"], 1
        )
        self.assertEqual(
            report["bare_core_question_coverage"]["by_split"]["train"][
                "matched_record_count"
            ],
            1,
        )
        self.assertEqual(
            report["bare_core_question_coverage"]["by_split"]["val"][
                "matched_record_count"
            ],
            0,
        )
        self.assertEqual(
            report["template_opening_analysis"]["by_split"]["train"],
            {"numerator": 1, "denominator": 1, "share": 1.0},
        )
        self.assertEqual(
            report["evidence_copy_analysis"]["by_split"]["train"]
            ["denominator_eligible_supported_records"],
            1,
        )
        self.assertEqual(report["risk_conclusion"]["decision_scope"], "train")
        self.assertTrue(
            all(
                item["decision_scope"] == "train"
                for item in report["risk_conclusion"]["findings"]
            )
        )
        self.assertEqual(
            report["self_reference_and_cycle_screen"]["exact_self_relation_records"], 1
        )
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn(SECRET_BODY, serialized)
        record_share_sum = sum(
            item["record_share"]
            for item in report["dimension_semantic_density"].values()
        )
        token_share_sum = sum(
            item["supervised_token_share"]
            for item in report["dimension_semantic_density"].values()
        )
        self.assertAlmostEqual(record_share_sum, 1.0)
        self.assertAlmostEqual(token_share_sum, 1.0)

    def test_all_decision_gates_use_train_even_when_val_is_worse(self):
        non_core_dimension = next(
            dimension for dimension in DIMENSION_TOTALS if dimension != CORE
        )
        train = [
            record(
                1,
                split="train",
                question="请回答：萧炎是谁？",
                answer="萧炎是故事中的人物。",
                spans=None,
            )
        ]
        val_fact_count = (len(KNOWN_CORE_FACTS) + 1) // 2
        val = [
            record(
                index + 2,
                split="val",
                dimension=non_core_dimension,
                question=fact.canonical_question,
                answer="原文写道“萧炎是萧炎。”",
                spans=["萧炎是萧炎。"],
            )
            for index, fact in enumerate(KNOWN_CORE_FACTS[:val_fact_count])
        ]

        report = audit_semantic_density({"train": train, "val": val})
        bare = report["bare_core_question_coverage"]
        self.assertEqual(bare["by_split"]["train"]["distinct_fact_coverage"], 0.0)
        self.assertGreaterEqual(
            bare["by_split"]["val"]["distinct_fact_coverage"], 0.50
        )
        self.assertGreaterEqual(bare["overall"]["distinct_fact_coverage"], 0.50)

        finding_codes = {
            item["code"] for item in report["risk_conclusion"]["findings"]
        }
        self.assertIn("bare_core_question_coverage_too_low", finding_codes)
        self.assertNotIn("fixed_opening_template_concentration", finding_codes)
        self.assertNotIn("core_supervision_density_too_low", finding_codes)
        self.assertNotIn("evidence_copy_concentration", finding_codes)
        self.assertNotIn("self_or_cycle_heuristic_requires_review", finding_codes)
        bare_finding = next(
            item
            for item in report["risk_conclusion"]["findings"]
            if item["code"] == "bare_core_question_coverage_too_low"
        )
        self.assertEqual(bare_finding["decision_scope"], "train")
        self.assertEqual(bare_finding["observed"], 0.0)

        self.assertEqual(
            report["template_opening_analysis"]["by_split"]["train"]["share"],
            0.0,
        )
        self.assertEqual(
            report["template_opening_analysis"]["by_split"]["val"]["share"],
            1.0,
        )
        self.assertEqual(
            report["dimension_semantic_density_by_split"]["train"][CORE][
                "relative_supervision_density_index"
            ],
            1.0,
        )
        self.assertGreater(
            report["evidence_copy_analysis"]["by_split"]["val"]
            ["share_at_or_above_0_50"],
            0.50,
        )

    def test_invalid_json_and_split_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "invalid.jsonl"
            invalid.write_text("not-json\n", encoding="utf-8")
            with self.assertRaisesRegex(SemanticAuditError, "invalid_jsonl"):
                load_jsonl(invalid, expected_split="train")

            mismatch = root / "mismatch.jsonl"
            write_jsonl(mismatch, [record(1, split="val")])
            with self.assertRaisesRegex(SemanticAuditError, "record_split_mismatch"):
                load_jsonl(mismatch, expected_split="train")

            with self.assertRaisesRegex(SemanticAuditError, "unauthorized_split"):
                load_jsonl(mismatch, expected_split="sealed_test")

    def test_paths_are_portable_and_outside_repo_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            inside = root / "data" / "train.jsonl"
            write_jsonl(inside, [record(1, split="train")])
            self.assertEqual(portable_repo_path(inside, root), "data/train.jsonl")
            outside = Path(directory) / "outside.jsonl"
            outside.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(SemanticAuditError, "path_outside_repo"):
                portable_repo_path(outside, root)

    def test_execute_reads_only_train_val_and_leaks_no_content_or_absolute_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            train = root / "data" / "sft" / "v7" / "train.jsonl"
            val = root / "data" / "sft" / "v7" / "val.jsonl"
            public = root / "data" / "sft" / "v7" / "public_diagnostic.jsonl"
            sealed = root / "data" / "sft" / "v7" / "sealed_test.jsonl"
            write_jsonl(
                train,
                [
                    record(
                        1,
                        split="train",
                        question=f"{SECRET_BODY}？",
                        answer=f"原文写道“{SECRET_BODY}”。",
                        spans=[SECRET_BODY],
                    )
                ],
            )
            write_jsonl(val, [record(2, split="val")])
            write_jsonl(public, [record(3, split="public_diagnostic")])
            write_jsonl(sealed, [record(4, split="sealed_test")])
            report_path = root / "reports" / "audit.json"
            markdown_path = root / "reports" / "audit.md"

            opened: list[str] = []
            original_open = Path.open

            def guarded_open(path: Path, *args, **kwargs):
                opened.append(path.name)
                if path.name in {"public_diagnostic.jsonl", "sealed_test.jsonl"}:
                    raise AssertionError("forbidden split was opened")
                return original_open(path, *args, **kwargs)

            with patch("pathlib.Path.open", new=guarded_open):
                report = execute_audit(
                    repo_root=root,
                    train_path=Path("data/sft/v7/train.jsonl"),
                    val_path=Path("data/sft/v7/val.jsonl"),
                    report_path=Path("reports/audit.json"),
                    markdown_path=Path("reports/audit.md"),
                    run_id="test-run",
                )

            self.assertNotIn("public_diagnostic.jsonl", opened)
            self.assertNotIn("sealed_test.jsonl", opened)
            self.assertFalse(report["privacy"]["public_diagnostic_body_accessed"])
            self.assertFalse(report["privacy"]["sealed_test_body_accessed"])
            combined = report_path.read_text(encoding="utf-8") + markdown_path.read_text(
                encoding="utf-8"
            )
            self.assertNotIn(SECRET_BODY, combined)
            self.assertNotIn(str(root), combined)
            self.assertIn('"path": "data/sft/v7/train.jsonl"', combined)
            implementation = report["implementation"]
            self.assertEqual(
                implementation["path"], "audit_sft_v7_semantic_density.py"
            )
            self.assertRegex(implementation["sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                implementation["algorithm_version"],
                "semantic-density-train-gates/2.0",
            )
            self.assertEqual(implementation["git_commit"], "unavailable")
            self.assertIsNone(implementation["git_dirty"])

    def test_module_logs_cover_success_and_failure_without_leaking_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            train = root / "data" / "sft" / "v7" / "train.jsonl"
            val = root / "data" / "sft" / "v7" / "val.jsonl"
            write_jsonl(
                train,
                [
                    record(
                        1,
                        split="train",
                        answer=f"原文写道“{SECRET_BODY}”。",
                        spans=[SECRET_BODY],
                    )
                ],
            )
            write_jsonl(val, [record(2, split="val")])
            common = [
                "--repo-root",
                str(root),
                "--train",
                "data/sft/v7/train.jsonl",
                "--val",
                "data/sft/v7/val.jsonl",
                "--report",
                "reports/audit.json",
                "--markdown",
                "reports/audit.md",
                "--log-dir",
                "logs/audit",
                "--no-console-log",
            ]
            with redirect_stdout(StringIO()):
                self.assertEqual(main(common), 0)
            success_logs = "".join(
                path.read_text(encoding="utf-8")
                for path in (root / "logs" / "audit").glob("*.jsonl")
            )
            self.assertIn("semantic density audit complete", success_logs)
            self.assertNotIn(SECRET_BODY, success_logs)
            self.assertNotIn(str(root), success_logs)

            train.write_text(f"not-json-{SECRET_BODY}\n", encoding="utf-8")
            with redirect_stdout(StringIO()):
                with self.assertRaisesRegex(SemanticAuditError, "invalid_jsonl"):
                    main(common)
            all_logs = "".join(
                path.read_text(encoding="utf-8")
                for path in (root / "logs" / "audit").glob("*.jsonl")
            )
            self.assertIn("semantic density audit failed", all_logs)
            self.assertIn("invalid_jsonl", all_logs)
            self.assertNotIn(SECRET_BODY, all_logs)
            self.assertNotIn(str(root), all_logs)


if __name__ == "__main__":
    unittest.main()
