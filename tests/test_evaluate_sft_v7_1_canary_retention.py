from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from training_runtime import canonical_json_sha256, file_sha256

import evaluate_sft_v7_1_canary_retention as retention


class CanaryRetentionTests(unittest.TestCase):
    def _lineage_fixture(self, root: Path) -> dict[str, object]:
        manifest_path = root / "manifest.json"
        dataset_identity = "d" * 64
        manifest = {
            "manifest_schema_version": retention.EXPECTED_CANARY_MANIFEST_SCHEMA,
            "status": "frozen_canary_ready",
            "split_totals": {"train": 64, "holdout_eval": 16},
            "dataset_identity_sha256": dataset_identity,
            "access_audit": {
                "formal_corpus_body_read": False,
                "v7_public_body_read": False,
                "v7_sealed_body_read": False,
                "v7_train_body_read": False,
            },
        }
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        manifest_sha = file_sha256(manifest_path)

        base = {
            "path": retention.EXPECTED_BASE_CHECKPOINT_PATH,
            "sha256": retention.BASE_CHECKPOINT_SHA256,
            "step": retention.BASE_STEP,
            "config_canonical_sha256": retention.BASE_CONFIG_CANONICAL_SHA256,
            "token_manifest_sha256": retention.BASE_TOKEN_MANIFEST_SHA256,
            "parameter_count": retention.BASE_PARAMETER_COUNT,
        }
        base["binding_sha256"] = canonical_json_sha256(base)
        tensor_path = root / "train_eval_tensors.pt"
        torch.save(
            {
                "schema_version": retention.EXPECTED_CANARY_TENSOR_SCHEMA,
                "train_records": [{"id": f"train-{index}"} for index in range(64)],
                "eval_records": [{"id": f"eval-{index}"} for index in range(16)],
                "tokenizer_sha256": retention.TOKENIZER_SHA256,
                "bpe_token_manifest_sha256": retention.BASE_TOKEN_MANIFEST_SHA256,
                "canary_dataset_manifest_sha256": manifest_sha,
                "canary_dataset_identity_sha256": dataset_identity,
                "artifact_binding_sha256": "a" * 64,
                "required_base_checkpoint": base,
            },
            tensor_path,
        )
        tensor_sha = file_sha256(tensor_path)
        payload_summary = {
            "split_counts": {"train": 64, "holdout_eval": 16},
            "optimizer_record_count": 64,
            "holdout_optimizer_record_count": 0,
        }
        provenance = {
            "stage": "sft_v7_1_canary",
            "base_checkpoint_path": retention.EXPECTED_BASE_CHECKPOINT_PATH,
            "base_checkpoint_sha256": retention.BASE_CHECKPOINT_SHA256,
            "base_checkpoint_step": retention.BASE_STEP,
            "base_config_canonical_sha256": retention.BASE_CONFIG_CANONICAL_SHA256,
            "base_token_manifest_sha256": retention.BASE_TOKEN_MANIFEST_SHA256,
            "tokenizer_sha256": retention.TOKENIZER_SHA256,
            "sft_tensor_path": "data/sft/v7_1_canary/train_eval_tensors.pt",
            "sft_tensor_sha256": tensor_sha,
            "canary_tensor_path": "data/sft/v7_1_canary/train_eval_tensors.pt",
            "canary_tensor_sha256": tensor_sha,
            "canary_dataset_manifest_sha256": manifest_sha,
            "canary_dataset_identity_sha256": dataset_identity,
            "payload_summary": payload_summary,
            "optimization_train_records": 64,
            "optimization_holdout_records": 0,
            "holdout_records_consumed": 0,
            "teacher_loss_holdout_records": 16,
            "public_records_consumed": 0,
            "sealed_records_consumed": 0,
        }
        training = {
            "target_steps": 400,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "minimum_learning_rate": 1e-5,
            "warmup_steps": 20,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "gradient_clip": 1.0,
            "seed": 42,
            "sampler": "deterministic_shuffled_epoch/v1",
        }
        schedule = {
            "strategy": "linear_warmup_cosine_decay/v1",
            "target_steps": 400,
            "warmup_steps": 20,
            "peak_learning_rate": 1e-4,
            "minimum_learning_rate": 1e-5,
            "sampler": "deterministic_shuffled_epoch/v1",
        }
        effective = {
            "schema_version": "sft-v7.1-canary-training-signature/v1",
            "model": retention.EXPECTED_CONFIG_MODEL,
            "provenance": provenance,
            "training": training,
            "schedule": schedule,
        }
        effective["signature_sha256"] = canonical_json_sha256(
            retention._signature_payload(effective)
        )
        effective_path = root / "effective_config.json"
        effective_path.write_text(
            json.dumps(effective, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )

        checkpoint_path = root / "step_00020.pt"
        checkpoint_path.write_bytes(b"synthetic checkpoint fixture")
        checkpoint_sha = file_sha256(checkpoint_path)
        Path(f"{checkpoint_path}.sha256").write_text(
            f"{checkpoint_sha}  {checkpoint_path.name}\n", encoding="utf-8"
        )
        checkpoint = {
            "schema_version": "training-checkpoint/v1",
            "step": 20,
            "config_sha256": effective["signature_sha256"],
            "model_state_dict": {},
            "extra": {
                **provenance,
                "sampler_state": {"epoch": 2, "offset": 0},
                "learning_rate_schedule": schedule,
            },
        }
        return {
            "manifest_path": manifest_path,
            "tensor_path": tensor_path,
            "effective": effective,
            "effective_path": effective_path,
            "checkpoint_path": checkpoint_path,
            "checkpoint": checkpoint,
        }

    def test_five_percent_bpc_gate_is_signed_and_exact(self):
        base = 3.75
        passed = retention.compare_bpc(base * 1.05, base)
        failed = retention.compare_bpc(base * 1.050001, base)
        improved = retention.compare_bpc(base * 0.9, base)

        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])
        self.assertTrue(improved["passed"])
        self.assertAlmostEqual(
            passed["relative_degradation_candidate_minus_baseline"], 0.05
        )

    def test_lineage_binds_base_config_tensor_manifest_counts_and_blind_scope(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = self._lineage_fixture(Path(temp_dir))
            result = retention.validate_canary_lineage(
                fixture["checkpoint"],
                checkpoint_path=fixture["checkpoint_path"],
                effective_config=fixture["effective"],
                effective_config_path=fixture["effective_path"],
                canary_tensor_path=fixture["tensor_path"],
                canary_manifest_path=fixture["manifest_path"],
            )

        self.assertTrue(result["verified"])
        self.assertEqual(result["base_checkpoint_step"], 5750)
        self.assertEqual(result["training_split_counts"], {"train": 64})
        self.assertEqual(result["holdout_eval_count"], 16)
        self.assertEqual(result["holdout_records_consumed"], 0)
        self.assertTrue(
            result["development_provenance_inferred_from_legacy_checkpoint"]
        )
        self.assertEqual(
            result["development_records_consumed_for_teacher_loss"], 16
        )
        self.assertEqual(
            result["development_records_used_for_checkpoint_selection"], 16
        )
        self.assertEqual(result["public_records_consumed"], 0)
        self.assertEqual(result["sealed_records_consumed"], 0)
        self.assertEqual(result["pretraining_test_body_reads"], 0)

    def test_lineage_rejects_signature_tensor_and_blind_consumption_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = self._lineage_fixture(Path(temp_dir))
            fixture["effective"]["training"]["learning_rate"] = 9e-4
            with self.assertRaisesRegex(
                retention.CanaryRetentionError, "signatures do not agree"
            ):
                retention.validate_canary_lineage(
                    fixture["checkpoint"],
                    checkpoint_path=fixture["checkpoint_path"],
                    effective_config=fixture["effective"],
                    effective_config_path=fixture["effective_path"],
                    canary_tensor_path=fixture["tensor_path"],
                    canary_manifest_path=fixture["manifest_path"],
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = self._lineage_fixture(Path(temp_dir))
            fixture["tensor_path"].write_bytes(b"tampered")
            with self.assertRaisesRegex(
                retention.CanaryRetentionError, "tensor SHA changed"
            ):
                retention.validate_canary_lineage(
                    fixture["checkpoint"],
                    checkpoint_path=fixture["checkpoint_path"],
                    effective_config=fixture["effective"],
                    effective_config_path=fixture["effective_path"],
                    canary_tensor_path=fixture["tensor_path"],
                    canary_manifest_path=fixture["manifest_path"],
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = self._lineage_fixture(Path(temp_dir))
            fixture["checkpoint"]["extra"]["sealed_records_consumed"] = 1
            with self.assertRaisesRegex(
                retention.CanaryRetentionError, "sealed_records_consumed"
            ):
                retention.validate_canary_lineage(
                    fixture["checkpoint"],
                    checkpoint_path=fixture["checkpoint_path"],
                    effective_config=fixture["effective"],
                    effective_config_path=fixture["effective_path"],
                    canary_tensor_path=fixture["tensor_path"],
                    canary_manifest_path=fixture["manifest_path"],
                )

    def test_hard_gates_require_16_nonempty_and_limit_mechanical_degradation(self):
        comparison = retention.compare_bpc(3.8, 3.75)
        summary = {"empty_rate": 0.0, "mechanical_degeneration_rate": 0.25}
        passed = retention.build_automatic_gates(
            comparison=comparison, generation_summary=summary, generation_count=16
        )
        self.assertTrue(passed["passed"])

        summary["empty_rate"] = 1 / 16
        failed = retention.build_automatic_gates(
            comparison=comparison, generation_summary=summary, generation_count=16
        )
        self.assertFalse(failed["passed"])
        self.assertIn("all_16_continuations_nonempty", failed["failed_gate_names"])

    def test_cli_has_no_test_public_or_sealed_inputs_and_rejects_aliases(self):
        args = retention.parse_args([])
        self.assertFalse(hasattr(args, "test"))
        self.assertFalse(hasattr(args, "public"))
        self.assertFalse(hasattr(args, "sealed"))
        args.prompts = Path("data/sealed/prompts.txt")
        with self.assertRaisesRegex(ValueError, "test, public or sealed"):
            retention.validate_args(args)
        args.prompts = retention.DEFAULT_PROMPTS
        args.raw_validation = Path("data/test.txt")
        with self.assertRaisesRegex(ValueError, "test, public or sealed"):
            retention.validate_args(args)

    def test_generation_log_context_omits_prompt_continuation_and_token_ids(self):
        prompt = "绝不能进入日志的提示正文"
        continuation = "绝不能进入日志的续写正文"
        measured = {
            "continuation": continuation,
            "characters": len(continuation),
            "generated_tokens": 12,
            "generated_token_ids": [1, 2, 3],
            "stop_reason": "eos",
            "eos_emitted": True,
            "four_gram_repetition": 0.0,
            "mechanically_degenerate": False,
        }
        context = retention.safe_generation_log_context(
            prompt_index=1, prompt=prompt, measured=measured
        )
        serialized = json.dumps(context, ensure_ascii=False)
        self.assertNotIn(prompt, serialized)
        self.assertNotIn(continuation, serialized)
        self.assertNotIn("generated_token_ids", context)
        self.assertNotIn("prompt", context)
        self.assertNotIn("continuation", context)

    def test_markdown_reports_eos_and_both_truncation_reasons_with_full_samples(self):
        report = {
            "status": "retention_passed",
            "checkpoint_lineage": {"checkpoint_path": "runs/canary.pt", "step": 400},
            "validation_diagnostic": {"loss": 4.4},
            "bpc_comparison": {
                "candidate_bpc": 3.7,
                "baseline_bpc": 3.75,
                "relative_degradation_candidate_minus_baseline": -0.01,
            },
            "generation_summary": {
                "empty_rate": 0.0,
                "mechanical_degeneration_rate": 0.0,
                "stop_reason_counts": {
                    "eos": 8,
                    "max_characters": 6,
                    "max_new_tokens": 2,
                },
            },
            "automatic_hard_gates": {"gates": []},
            "generations": [
                {
                    "prompt_index": 1,
                    "prompt": "完整提示",
                    "continuation": "完整续写",
                    "stop_reason": "max_characters",
                    "eos_emitted": False,
                    "four_gram_repetition": 0.0,
                    "mechanically_degenerate": False,
                }
            ],
        }
        markdown = retention.build_markdown_report(report)
        self.assertIn("完整提示", markdown)
        self.assertIn("完整续写", markdown)
        self.assertIn("EOS 停止：8/16", markdown)
        self.assertIn("长度截断：6/16", markdown)
        self.assertIn("Token 上限截断：2/16", markdown)

    def test_success_and_failure_logs_are_modular_redacted_and_body_free(self):
        secret_prompt = "此提示只能存在于报告中"
        secret_answer = "此回答只能存在于报告中"
        fake_report = {
            "status": "retention_passed",
            "checkpoint_lineage": {"step": 400},
            "validation_diagnostic": {"fixed_window_bpc": 3.7},
            "bpc_comparison": {
                "relative_degradation_candidate_minus_baseline": -0.01
            },
            "generation_summary": {
                "stop_reason_counts": {
                    "eos": 16,
                    "max_characters": 0,
                    "max_new_tokens": 0,
                }
            },
            "automatic_hard_gates": {"passed": True},
            "generations": [
                {"prompt": secret_prompt, "continuation": secret_answer}
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "success-logs"
            with mock.patch.object(retention, "run_retention_audit", return_value=fake_report), mock.patch.object(
                retention, "write_outputs", return_value=fake_report
            ):
                result = retention.main(
                    ["--log-dir", str(log_dir), "--no-console-log"]
                )
            self.assertEqual(result, 0)
            log_paths = list(log_dir.glob("*.jsonl"))
            self.assertGreaterEqual(len(log_paths), len(retention.LOG_MODULES))
            success_logs = "".join(path.read_text(encoding="utf-8") for path in log_paths)
            self.assertNotIn(secret_prompt, success_logs)
            self.assertNotIn(secret_answer, success_logs)
            self.assertIn('"run_id"', success_logs)
            self.assertRegex(success_logs, r'"timestamp":\s*"[^\"]+\+00:00"')

            failure_dir = Path(temp_dir) / "failure-logs"
            failure_body = "failure body must not be logged"
            with mock.patch.object(
                retention,
                "run_retention_audit",
                side_effect=retention.CanaryRetentionError("synthetic_failure", failure_body),
            ):
                result = retention.main(
                    ["--log-dir", str(failure_dir), "--no-console-log"]
                )
            self.assertEqual(result, 1)
            failure_logs = "".join(
                path.read_text(encoding="utf-8")
                for path in failure_dir.glob("*.jsonl")
            )
            self.assertNotIn(failure_body, failure_logs)
            self.assertIn("synthetic_failure", failure_logs)
            self.assertRegex(failure_logs, r'"sft_sealed_body_reads":\s*0')

            redaction_dir = Path(temp_dir) / "redaction-logs"
            levels = {module: "OFF" for module in retention._ALL_LOG_MODULES}
            levels["data"] = "INFO"
            loggers = retention.configure_module_loggers(
                redaction_dir, "retention-redaction-run", levels, console=False
            )
            try:
                loggers["data"].info(
                    "redaction check",
                    extra={"context": {"api_key": "never-log-this-key"}},
                )
            finally:
                retention.close_module_loggers(loggers)
            redacted = "".join(
                path.read_text(encoding="utf-8")
                for path in redaction_dir.glob("*.jsonl")
            )
            self.assertNotIn("never-log-this-key", redacted)
            self.assertIn("[REDACTED]", redacted)


if __name__ == "__main__":
    unittest.main()
