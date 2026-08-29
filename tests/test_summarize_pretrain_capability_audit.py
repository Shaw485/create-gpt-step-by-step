import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from summarize_pretrain_capability_audit import (
    AUDIT_SCHEMA_VERSION,
    CLOZE_METRICS,
    SCHEMA_VERSION,
    SummaryError,
    build_comparison,
    build_csv,
    build_markdown,
    main,
)
from training_runtime import close_module_loggers, configure_module_loggers


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "docs/pretrain_capability_audit_protocol.md"
SHARED_HASHES = {
    "config": "a" * 64,
    "manifest": "b" * 64,
    "tokenizer": "c" * 64,
    "validation_tensor": "d" * 64,
    "probe": "e" * 64,
    "prompts": "f" * 64,
}


def ranking_metric(top1: float, mrr: float) -> dict:
    return {
        "top1_accuracy": top1,
        "mean_reciprocal_rank": mrr,
        "mean_correct_margin": 0.1,
    }


def make_audit(
    step: int,
    *,
    loss: float,
    perplexity: float,
    next_token_top1: float,
    empty_rate: float,
    eos_rate: float,
    degeneration_rate: float,
    unique_ratio: float,
    repetition: float,
    cloze_top1: float,
    cloze_mrr: float,
    context_lift_mrr: float,
) -> dict:
    metrics = {}
    tiers = {"high": {"case_count": 6, "metrics": {}}, "low": {"case_count": 6, "metrics": {}}}
    for metric in CLOZE_METRICS:
        metric_mrr = context_lift_mrr if metric == "context_lift" else cloze_mrr
        metrics[metric] = ranking_metric(cloze_top1, metric_mrr)
        tiers["high"]["metrics"][metric] = ranking_metric(
            min(1.0, cloze_top1 + 0.1), min(1.0, metric_mrr + 0.05)
        )
        tiers["low"]["metrics"][metric] = ranking_metric(
            max(0.0, cloze_top1 - 0.1), max(0.0, metric_mrr - 0.05)
        )
    cases = [
        {
            "id": f"case-{index}",
            "context": f"固定前缀{index}",
            "candidates": ["萧炎", "药老", "云韵", "紫研"],
            "correct": "萧炎",
            "frequency_tier": "high" if index < 6 else "low",
        }
        for index in range(12)
    ]
    generations = [
        {
            "prompt": f"固定小说提示{index}",
            "unique_character_ratio": unique_ratio,
        }
        for index in range(16)
    ]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "run_id": f"audit-{step}",
        "status": "FORMAL_AUDIT_COMPLETE_MANUAL_REVIEW_REQUIRED",
        "scope": {
            "formal_requested": True,
            "formal_status_eligible": True,
        },
        "config": {"canonical_sha256": SHARED_HASHES["config"]},
        "checkpoint": {
            "path": f"checkpoints/step_{step:05d}.pt",
            "sha256": f"{step:064x}"[-64:],
            "step": step,
        },
        "data": {
            "manifest_sha256": SHARED_HASHES["manifest"],
            "tokenizer_sha256": SHARED_HASHES["tokenizer"],
        },
        "validation_diagnostic": {
            "split": "val",
            "tensor_path": "data/scaling_a/bpe_3000/val_tokens.pt",
            "tensor_sha256": SHARED_HASHES["validation_tensor"],
            "loss": loss,
            "perplexity": perplexity,
            "top1_accuracy": next_token_top1,
            "windows_evaluated": 60,
            "tokens_evaluated": 30720,
        },
        "probe_provenance": {
            "formal_status_eligible": True,
            "artifact_sha256": SHARED_HASHES["probe"],
            "prompts_sha256": SHARED_HASHES["prompts"],
            "usage_checks": {
                "sft_false": True,
                "test_false": True,
                "training_allowed_false": True,
            },
        },
        "generation_summary": {
            "sample_count": 16,
            "empty_rate": empty_rate,
            "eos_stop_rate": eos_rate,
            "mechanical_degeneration_rate": degeneration_rate,
            "mean_four_gram_repetition": repetition,
            "maximum_character_run": 3,
        },
        "generations": generations,
        "cloze": {
            "case_count": 12,
            "primary_diagnostic_metric": "mean_token_log_probability",
            "metrics": metrics,
            "by_frequency_tier": tiers,
            "cases": cases,
        },
    }


class PretrainCapabilitySummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.audit_paths = {}
        specifications = {
            250: dict(
                loss=6.9,
                perplexity=992.0,
                next_token_top1=0.08,
                empty_rate=0.0625,
                eos_rate=0.0,
                degeneration_rate=0.125,
                unique_ratio=0.30,
                repetition=0.04,
                cloze_top1=0.25,
                cloze_mrr=0.52,
                context_lift_mrr=0.60,
            ),
            5750: dict(
                loss=4.4,
                perplexity=81.0,
                next_token_top1=0.24,
                empty_rate=0.0,
                eos_rate=0.0,
                degeneration_rate=0.0,
                unique_ratio=0.48,
                repetition=0.02,
                cloze_top1=0.66,
                cloze_mrr=0.80,
                context_lift_mrr=0.82,
            ),
            6000: dict(
                loss=4.39,
                perplexity=80.5,
                next_token_top1=0.243,
                empty_rate=0.0,
                eos_rate=0.0,
                degeneration_rate=0.0625,
                unique_ratio=0.47,
                repetition=0.03,
                cloze_top1=0.66,
                cloze_mrr=0.82,
                context_lift_mrr=0.80,
            ),
        }
        for step, values in specifications.items():
            path = self.root / f"audit-{step}.json"
            path.write_text(
                json.dumps(make_audit(step, **values), ensure_ascii=False),
                encoding="utf-8",
            )
            self.audit_paths[step] = path
        self.bpc_path = self.root / "pretrain_v4_report.json"
        self.bpc_path.write_text(
            json.dumps(
                {
                    "test_evaluated": False,
                    "history": [
                        {
                            "step": 250,
                            "val_bits_per_character": 5.914875678244785,
                            "val_loss": 7.01,
                            "train_loss": 7.00,
                        },
                        {
                            "step": 5750,
                            "val_bits_per_character": 3.761229497144957,
                            "val_loss": 4.45,
                            "train_loss": 4.20,
                        },
                        {
                            "step": 6000,
                            "val_bits_per_character": 3.7531531349649643,
                            "val_loss": 4.44,
                            "train_loss": 4.18,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.log_dir = self.root / "logs"

    def tearDown(self):
        self.temporary.cleanup()

    def build(self):
        loggers = configure_module_loggers(
            self.log_dir,
            "summary-test",
            {"data": "INFO", "validation": "INFO", "orchestrator": "INFO"},
            max_bytes=4096,
            backup_count=1,
            console=False,
        )
        try:
            return build_comparison(
                self.audit_paths,
                protocol_path=PROTOCOL_PATH,
                bpc_source_path=self.bpc_path,
                run_id="summary-test",
                loggers=loggers,
            )
        finally:
            close_module_loggers(loggers)

    def cli_args(self, *, audit250=None):
        return [
            "--audit-step250",
            str(audit250 or self.audit_paths[250]),
            "--audit-step5750",
            str(self.audit_paths[5750]),
            "--audit-step6000",
            str(self.audit_paths[6000]),
            "--protocol",
            str(PROTOCOL_PATH),
            "--bpc-source",
            str(self.bpc_path),
            "--output-json",
            str(self.root / "comparison.json"),
            "--output-csv",
            str(self.root / "comparison.csv"),
            "--output-markdown",
            str(self.root / "comparison.md"),
            "--log-dir",
            str(self.log_dir),
            "--run-id",
            "summary-cli-test",
            "--no-console-log",
        ]

    def test_summary_keeps_token_bits_and_bpc_strictly_separate(self):
        comparison = self.build()
        self.assertEqual(comparison["schema_version"], SCHEMA_VERSION)
        self.assertFalse(comparison["scope"]["test_read"])
        self.assertTrue(comparison["scope"]["fixed_window_token_diagnostic_is_not_bpc"])
        step250 = comparison["checkpoints"][0]["validation_diagnostic"]
        self.assertAlmostEqual(
            step250["token_bits_per_token_value"], 6.9 / math.log(2.0)
        )
        self.assertEqual(step250["validation_bits_per_character"], 5.914875678244785)
        self.assertNotAlmostEqual(
            step250["token_bits_per_token_value"],
            step250["validation_bits_per_character"],
        )
        self.assertIn("full-validation", comparison["inputs"]["bpc_source"]["definition"])
        self.assertIn("not BPC", comparison["inputs"]["bpc_source"]["forbidden_substitution"])

    def test_all_cloze_metrics_tiers_deltas_and_frozen_gates_are_present(self):
        comparison = self.build()
        for checkpoint in comparison["checkpoints"]:
            self.assertEqual(set(checkpoint["cloze"]["metrics"]), set(CLOZE_METRICS))
            for metric in CLOZE_METRICS:
                self.assertEqual(
                    set(checkpoint["cloze"]["metrics"][metric]),
                    {"overall", "high", "low"},
                )
        self.assertEqual(set(comparison["deltas"]), {"250_to_5750", "5750_to_6000"})
        first_delta = comparison["deltas"]["250_to_5750"]["validation"]
        self.assertGreater(first_delta["validation_bpc_relative_improvement"], 0.20)
        plateau_delta = comparison["deltas"]["5750_to_6000"]["validation"]
        self.assertLess(plateau_delta["validation_bpc_improvement"], 0.01)
        gates = comparison["frozen_gates"]
        self.assertTrue(gates["language_base"]["automatic_conditions_passed"])
        self.assertEqual(gates["language_base"]["final_status"], "manual_review_required")
        self.assertTrue(gates["practical_plateau"]["automatic_numeric_conditions_passed"])
        self.assertEqual(gates["practical_plateau"]["final_status"], "manual_review_required")
        self.assertIsNone(gates["mature_novel_generator"]["observed"])
        self.assertFalse(comparison["scope"]["subjective_manual_scores_inferred"])

    def test_csv_and_markdown_expose_required_fields_without_subjective_scores(self):
        comparison = self.build()
        csv_text = build_csv(comparison)
        self.assertNotIn("\r", csv_text)
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        self.assertEqual([int(row["step"]) for row in rows], [250, 5750, 6000])
        required = {
            "checkpoint_sha256",
            "fixed_window_validation_token_loss_nats",
            "token_bits_per_token_value",
            "validation_bits_per_character",
            "token_perplexity",
            "next_token_top1_accuracy",
            "generation_empty_rate",
            "generation_eos_stop_rate",
            "generation_mechanical_degeneration_rate",
            "generation_mean_unique_character_ratio",
            "generation_mean_four_gram_repetition",
            "cloze_context_lift_high_top1_accuracy",
            "cloze_context_lift_low_mean_reciprocal_rank",
        }
        self.assertTrue(required.issubset(rows[0]))
        markdown = build_markdown(comparison)
        self.assertIn("Token bits/token", markdown)
        self.assertIn("BPC 只引用", markdown)
        self.assertIn("不包含主观人工评分", markdown)

    def test_cli_writes_only_temp_outputs_and_structured_module_logs(self):
        self.assertEqual(main(self.cli_args()), 0)
        json_path = self.root / "comparison.json"
        csv_path = self.root / "comparison.csv"
        markdown_path = self.root / "comparison.md"
        self.assertTrue(json_path.is_file())
        self.assertTrue(csv_path.is_file())
        self.assertTrue(markdown_path.is_file())
        comparison = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertFalse(comparison["scope"]["test_read"])
        self.assertGreaterEqual(
            json_path.stat().st_mtime_ns,
            max(csv_path.stat().st_mtime_ns, markdown_path.stat().st_mtime_ns),
        )
        for module in ("data", "validation", "orchestrator"):
            path = self.log_dir / f"summary-cli-test.{module}.jsonl"
            events = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertTrue(events)
            self.assertTrue(all(event["run_id"] == "summary-cli-test" for event in events))
            self.assertNotIn("固定小说提示", path.read_text(encoding="utf-8"))

    def test_rejects_test_audit_and_bpc_source_that_used_test(self):
        report = json.loads(self.audit_paths[250].read_text(encoding="utf-8"))
        report["validation_diagnostic"]["split"] = "test"
        bad_audit = self.root / "test-audit.json"
        bad_audit.write_text(json.dumps(report), encoding="utf-8")
        paths = dict(self.audit_paths)
        paths[250] = bad_audit
        loggers = configure_module_loggers(
            self.root / "reject-logs",
            "reject-test",
            {"data": "INFO", "validation": "INFO", "orchestrator": "INFO"},
            console=False,
        )
        try:
            with self.assertRaisesRegex(SummaryError, "only val is allowed"):
                build_comparison(
                    paths,
                    protocol_path=PROTOCOL_PATH,
                    bpc_source_path=self.bpc_path,
                    run_id="reject-test",
                    loggers=loggers,
                )
        finally:
            close_module_loggers(loggers)

        bpc = json.loads(self.bpc_path.read_text(encoding="utf-8"))
        bpc["test_evaluated"] = True
        self.bpc_path.write_text(json.dumps(bpc), encoding="utf-8")
        with self.assertRaisesRegex(SummaryError, "test must remain sealed"):
            self.build()

    def test_failure_path_logs_actionable_context(self):
        missing = self.root / "missing-audit.json"
        self.assertEqual(main(self.cli_args(audit250=missing)), 2)
        log_path = self.log_dir / "summary-cli-test.orchestrator.jsonl"
        events = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        failure = next(event for event in reversed(events) if event["level"] == "ERROR")
        self.assertEqual(failure["context"]["test_read"], False)
        self.assertIn("Verify", failure["context"]["remediation"])
        self.assertEqual(failure["context"]["error_type"], "SummaryError")


if __name__ == "__main__":
    unittest.main()
