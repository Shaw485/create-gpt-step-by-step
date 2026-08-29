from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from summarize_sft_v7_m020 import (
    ArtifactSpec,
    CheckpointSpec,
    FIXED_SCHEMA,
    M020SummaryError,
    PUBLIC_SCHEMA,
    RETENTION_SCHEMA,
    SCHEMA_VERSION,
    TRAIN_SCHEMA,
    build_loss_csv,
    build_loss_svg,
    default_checkpoint_specs,
    main,
    resolve_log_levels,
    summarize_m020,
)
from training_runtime import close_module_loggers, configure_module_loggers


CHECKPOINT_SHA = "a" * 64
DATASET_SHA = "b" * 64
PUBLIC_TENSOR_SHA = "c" * 64
PROMPT_SET_SHA = "d" * 64


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return path


def make_train_report() -> dict:
    return {
        "schema_version": TRAIN_SCHEMA,
        "status": "training_complete_public_evaluation_pending",
        "target_step": 500,
        "best_val_loss": 1.2,
        "history": [
            {
                "step": 0,
                "train_loss": 3.0,
                "val_loss": 3.2,
                "active_phase": "phase1_core_chat_boundary",
                "coverage": {"coverage": 0.0},
            },
            {
                "step": 500,
                "train_loss": 1.0,
                "val_loss": 1.2,
                "active_phase": "phase2_full_vertical_mix",
                "coverage": {"coverage": 0.75},
            },
        ],
        "public_records_consumed": 0,
        "sealed_records_consumed": 0,
    }


def make_public_report(
    *,
    external_passed: bool | None = None,
    checkpoint_sha: str = CHECKPOINT_SHA,
    candidate_eligible: bool = False,
) -> dict:
    external_gate = {
        "id": "human_semantic_review",
        "threshold": ">= 0.8",
        "status": (
            "passed"
            if external_passed is True
            else "failed"
            if external_passed is False
            else "pending"
        ),
        "passed": external_passed,
    }
    return {
        "schema_version": PUBLIC_SCHEMA,
        "status": "diagnostic_complete_candidate_pending_or_ineligible",
        "checkpoint_step": 500,
        "checkpoint_mode": "sft-v7",
        "checkpoint_sha256": checkpoint_sha,
        "sft_dataset_manifest_sha256": DATASET_SHA,
        "public_tensor_sha256": PUBLIC_TENSOR_SHA,
        "teacher_forced": {"loss": 1.4, "perplexity": 4.05},
        "overall": {
            "records": 600,
            "generation_quality": {
                "eos_rate": 0.98,
                "empty_rate": 0.0,
                "truncation_rate": 0.02,
                "mechanical_repetition_rate": 0.01,
                "meta_phrase_rate": 0.0,
            },
        },
        "automatic_gates": [
            {"id": "core", "passed": True},
            {"id": "eos", "passed": True},
        ],
        "external_gates": [external_gate],
        "automatic_gates_passed": True,
        "external_gates_passed": external_passed is True,
        "candidate_eligible": candidate_eligible,
    }


def make_fixed_report(
    *,
    checkpoint_sha: str = CHECKPOINT_SHA,
    secret: str = "普通输出",
    step: int = 500,
) -> dict:
    return {
        "schema_version": FIXED_SCHEMA,
        "status": "complete",
        "checkpoint_step": step,
        "checkpoint_mode": "sft-v7",
        "checkpoint_sha256": checkpoint_sha,
        "prompt_set_sha256": PROMPT_SET_SHA,
        "results": [
            {
                "id": "one",
                "generated_text": secret,
                "generated_tokens": 4,
                "stopped_on_eos": True,
                "truncated": False,
            },
            {
                "id": "two",
                "generated_text": "第二条",
                "generated_tokens": 3,
                "stopped_on_eos": False,
                "truncated": True,
            },
        ],
    }


def make_retention_report(
    *, external_passed: bool = False, candidate_eligible: bool = False
) -> dict:
    if external_passed:
        reviews = {
            "ai_assisted_fluency": {
                "status": "passed",
                "minimum_score": 2.0,
                "score": 3.0,
            },
            "independent_human_review": {"status": "passed", "score": 4.0},
        }
    else:
        reviews = {
            "ai_assisted_fluency": {
                "status": "pending",
                "minimum_score": 2.0,
                "score": None,
            },
            "independent_human_review": {"status": "pending", "score": None},
            "automated_report_claims_human_review": False,
        }
    return {
        "schema_version": RETENTION_SCHEMA,
        "status": "automatic_retention_gates_passed_external_review_pending",
        "candidate_eligible": candidate_eligible,
        "checkpoint_lineage": {
            "step": 500,
            "checkpoint_sha256": CHECKPOINT_SHA,
            "public_records_consumed": 0,
            "sealed_records_consumed": 0,
            "blind_body_reads": 0,
        },
        "validation_diagnostic": {"loss": 4.5, "fixed_window_bpc": 3.8},
        "bpc_comparison": {
            "relative_degradation_candidate_minus_baseline": 0.015
        },
        "automatic_hard_gates": {
            "passed": True,
            "gates": [
                {"name": "bpc", "passed": True},
                {"name": "nonempty", "passed": True},
            ],
        },
        "external_reviews": reviews,
        "data_scope": {
            "pretraining_test_body_reads": 0,
            "sft_public_body_reads": 0,
            "sft_sealed_body_reads": 0,
        },
    }


class M020SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output_dir = self.root / "reports" / "m020"
        self.log_dir = self.root / "logs"
        self.run_id = "m020-summary-test"
        self.loggers = configure_module_loggers(
            self.log_dir,
            self.run_id,
            {
                "discovery": "INFO",
                "validation": "INFO",
                "rendering": "INFO",
                "orchestrator": "INFO",
            },
            max_bytes=4096,
            backup_count=1,
            console=False,
        )

    def tearDown(self):
        close_module_loggers(self.loggers)
        self.temporary.cleanup()

    def candidate_spec(
        self,
        public_path: Path,
        fixed_path: Path,
        retention_path: Path,
    ) -> CheckpointSpec:
        return CheckpointSpec(
            key="sft_step00500",
            display_name="SFT v7 Step 500",
            expected_step=500,
            checkpoint_mode="sft-v7",
            public=ArtifactSpec("public", (public_path,), required=True),
            fixed=ArtifactSpec("fixed", (fixed_path,), required=True),
            retention=ArtifactSpec("retention", (retention_path,), required=True),
        )

    def summarize(
        self,
        *,
        train_specs: dict[str, ArtifactSpec],
        checkpoint_specs: tuple[CheckpointSpec, ...],
    ) -> dict:
        return summarize_m020(
            train_specs=train_specs,
            checkpoint_specs=checkpoint_specs,
            output_dir=self.output_dir,
            project_root=self.root,
            run_id=self.run_id,
            loggers=self.loggers,
        )

    def test_missing_future_artifacts_are_pending_without_fabricated_metrics(self):
        report = self.summarize(
            train_specs={
                "smoke20": ArtifactSpec(
                    "train", (self.root / "missing-smoke.json",), required=True
                )
            },
            checkpoint_specs=(
                self.candidate_spec(
                    self.root / "missing-public.json",
                    self.root / "missing-fixed.json",
                    self.root / "missing-retention.json",
                ),
            ),
        )

        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["status"], "pending_required_artifacts")
        self.assertEqual(len(report["pending_required_artifacts"]), 4)
        self.assertEqual(report["loss_history_points"], 0)
        self.assertFalse(report["release_ready"])
        self.assertEqual(report["strict_candidate_keys"], [])
        public = report["checkpoints"][0]["public"]
        self.assertEqual(public["status"], "pending")
        self.assertNotIn("teacher_forced_loss", public)

        csv_text = (self.output_dir / "sft_v7_loss_curve.csv").read_text(
            encoding="utf-8"
        )
        self.assertEqual(len(list(csv.reader(io.StringIO(csv_text)))), 1)
        svg = (self.output_dir / "sft_v7_loss_curve.svg").read_text(
            encoding="utf-8"
        )
        self.assertIn("Pending: no completed training history", svg)
        self.assertIn("No zero or estimated loss values were inserted", svg)
        stored = json.loads(
            (self.output_dir / "checkpoint_comparison.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(stored["pending_required_artifacts"], report["pending_required_artifacts"])

    def test_smoke_public_is_optional_but_fixed16_remains_required(self):
        defaults = default_checkpoint_specs(self.root)
        smoke = next(spec for spec in defaults if spec.key == "smoke_step00020")
        self.assertFalse(smoke.public.required)
        self.assertTrue(smoke.fixed.required)

        fixed_path = write_json(
            smoke.fixed.paths[0], make_fixed_report(step=20)
        )
        self.assertTrue(fixed_path.is_file())
        report = self.summarize(
            train_specs={},
            checkpoint_specs=(smoke,),
        )
        row = report["checkpoints"][0]

        self.assertEqual(row["public"]["status"], "not_run_optional")
        self.assertEqual(
            row["summary_status"],
            "engineering_smoke_complete_public_not_run_optional",
        )
        self.assertEqual(report["pending_required_artifacts"], [])
        self.assertFalse(row["strict_candidate_eligible"])

        fixed_path.unlink()
        missing_fixed = self.summarize(
            train_specs={},
            checkpoint_specs=(smoke,),
        )
        self.assertEqual(len(missing_fixed["pending_required_artifacts"]), 1)
        self.assertEqual(
            missing_fixed["pending_required_artifacts"][0]["kind"], "fixed"
        )

    def test_complete_automatic_evidence_keeps_external_reviews_pending(self):
        secret = "小说正文不可进汇总 API_KEY=raw-secret-value"
        train_path = write_json(self.root / "train.json", make_train_report())
        public_path = write_json(self.root / "public.json", make_public_report())
        fixed_path = write_json(
            self.root / "fixed.json", make_fixed_report(secret=secret)
        )
        retention_path = write_json(
            self.root / "retention.json", make_retention_report()
        )
        report = self.summarize(
            train_specs={
                "formal2000": ArtifactSpec("train", (train_path,), required=True)
            },
            checkpoint_specs=(
                self.candidate_spec(public_path, fixed_path, retention_path),
            ),
        )

        self.assertEqual(
            report["status"],
            "automatic_diagnostics_complete_external_review_pending",
        )
        row = report["checkpoints"][0]
        self.assertEqual(row["public"]["automatic_gate_state"], "passed")
        self.assertEqual(row["public"]["external_gate_state"], "pending")
        self.assertEqual(row["retention"]["automatic_gate_state"], "passed")
        self.assertEqual(row["retention"]["external_review_state"], "pending")
        self.assertFalse(row["strict_candidate_eligible"])
        self.assertFalse(report["release_ready"])
        self.assertEqual(report["loss_history_points"], 2)

        csv_rows = list(
            csv.DictReader(
                io.StringIO(
                    (self.output_dir / "sft_v7_loss_curve.csv").read_text(
                        encoding="utf-8"
                    )
                )
            )
        )
        self.assertEqual([row["step"] for row in csv_rows], ["0", "500"])
        self.assertEqual(csv_rows[-1]["val_loss"], "1.2")
        combined_outputs = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                self.output_dir / "checkpoint_comparison.json",
                self.output_dir / "checkpoint_comparison.md",
                *self.log_dir.glob("*.jsonl"),
            )
        )
        self.assertNotIn(secret, combined_outputs)
        self.assertNotIn("raw-secret-value", combined_outputs)
        self.assertIn("external_review_pending", combined_outputs)

        hashes = (self.output_dir / "SHA256SUMS.md").read_text(encoding="utf-8")
        self.assertIn("checkpoint_comparison.json", hashes)
        self.assertIn("sft_v7_loss_curve.svg", hashes)
        self.assertNotIn("SHA256SUMS.md` |", hashes)

    def test_checkpoint_identity_mismatch_is_invalid_not_silently_selected(self):
        public_path = write_json(self.root / "public.json", make_public_report())
        fixed_path = write_json(
            self.root / "fixed.json",
            make_fixed_report(checkpoint_sha="e" * 64),
        )
        retention_path = write_json(
            self.root / "retention.json", make_retention_report()
        )
        report = self.summarize(
            train_specs={},
            checkpoint_specs=(
                self.candidate_spec(public_path, fixed_path, retention_path),
            ),
        )

        self.assertEqual(report["status"], "invalid_loaded_artifacts")
        row = report["checkpoints"][0]
        self.assertEqual(row["summary_status"], "invalid_artifact_identity")
        self.assertIn(
            "checkpoint SHA-256 differs across public/fixed/retention reports",
            row["integrity_errors"],
        )
        self.assertFalse(row["strict_candidate_eligible"])

    def test_present_wrong_schema_fails_instead_of_becoming_pending(self):
        bad = write_json(
            self.root / "public.json",
            {"schema_version": "wrong/v1", "status": "complete"},
        )
        spec = self.candidate_spec(
            bad,
            self.root / "missing-fixed.json",
            self.root / "missing-retention.json",
        )
        with self.assertRaisesRegex(M020SummaryError, "unexpected schema"):
            self.summarize(train_specs={}, checkpoint_specs=(spec,))

    def test_external_pass_requires_explicit_report_eligibility_and_release_stays_false(self):
        public_path = write_json(
            self.root / "public.json",
            make_public_report(external_passed=True, candidate_eligible=False),
        )
        fixed_path = write_json(self.root / "fixed.json", make_fixed_report())
        retention_path = write_json(
            self.root / "retention.json",
            make_retention_report(external_passed=True, candidate_eligible=False),
        )
        report = self.summarize(
            train_specs={},
            checkpoint_specs=(
                self.candidate_spec(public_path, fixed_path, retention_path),
            ),
        )

        row = report["checkpoints"][0]
        self.assertEqual(row["public"]["external_gate_state"], "passed")
        self.assertEqual(row["retention"]["external_review_state"], "passed")
        self.assertFalse(row["strict_candidate_eligible"])
        self.assertFalse(report["release_ready"])

    def test_sealed_filename_is_rejected_before_read(self):
        sealed = write_json(
            self.root / "sealed_test_report.json", make_public_report()
        )
        spec = self.candidate_spec(
            sealed,
            self.root / "missing-fixed.json",
            self.root / "missing-retention.json",
        )
        with self.assertRaisesRegex(M020SummaryError, "cannot be sealed"):
            self.summarize(train_specs={}, checkpoint_specs=(spec,))

    def test_loss_helpers_preserve_real_values_and_pending_state(self):
        rows = [
            {
                "run_label": "formal",
                "step": 0,
                "train_loss": 3.2,
                "val_loss": 3.4,
                "coverage": 0.0,
                "active_phase": "phase1",
                "source_report": "report.json",
            },
            {
                "run_label": "formal",
                "step": 500,
                "train_loss": 1.1,
                "val_loss": 1.3,
                "coverage": 0.8,
                "active_phase": "phase2",
                "source_report": "report.json",
            },
        ]
        csv_rows = list(csv.DictReader(io.StringIO(build_loss_csv(rows))))
        self.assertEqual(csv_rows[1]["val_loss"], "1.3")
        svg = build_loss_svg(rows)
        self.assertIn("solid train, dashed val", svg)
        self.assertNotIn("Pending: no completed", svg)
        self.assertIn("Pending: no completed", build_loss_svg([]))

    def test_log_levels_are_independently_configurable(self):
        levels = resolve_log_levels(
            ["discovery=DEBUG", "validation=OFF", "rendering=WARNING"]
        )
        self.assertEqual(levels["discovery"], "DEBUG")
        self.assertEqual(levels["validation"], "OFF")
        self.assertEqual(levels["rendering"], "WARNING")
        with self.assertRaisesRegex(M020SummaryError, "unknown M020 summary"):
            resolve_log_levels(["unknown=INFO"])

    def test_cli_success_and_failure_paths_write_actionable_redacted_logs(self):
        cli_output = self.root / "cli-output"
        cli_logs = self.root / "cli-logs"
        code = main(
            [
                "--output-dir",
                str(cli_output),
                "--project-root",
                str(self.root),
                "--log-dir",
                str(cli_logs),
                "--no-console-log",
            ]
        )
        self.assertEqual(code, 0)
        success_logs = "\n".join(
            path.read_text(encoding="utf-8") for path in cli_logs.glob("*.jsonl")
        )
        self.assertIn("M020 summary complete", success_logs)
        self.assertIn("pending_required_artifacts", success_logs)

        bad_public = write_json(
            self.root / "bad-public.json",
            {
                "schema_version": "wrong/v1",
                "api_key": "raw-secret-value",
            },
        )
        failure_logs = self.root / "failure-logs"
        code = main(
            [
                "--output-dir",
                str(self.root / "failure-output"),
                "--public-eval",
                f"sft_step00500={bad_public}",
                "--log-dir",
                str(failure_logs),
                "--no-console-log",
            ]
        )
        self.assertEqual(code, 1)
        failure_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in failure_logs.glob("*.jsonl")
        )
        self.assertIn("M020 summary failed", failure_text)
        self.assertIn("unexpected schema", failure_text)
        self.assertNotIn("raw-secret-value", failure_text)


if __name__ == "__main__":
    unittest.main()
