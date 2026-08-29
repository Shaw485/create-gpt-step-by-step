from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from training_runtime import canonical_json_sha256, file_sha256

import evaluate_sft_v7_retention as retention


class SFTV7RetentionTests(unittest.TestCase):
    def _lineage_fixture(self, root: Path, *, step: int = 20):
        checkpoint_path = root / "step_00020.pt"
        checkpoint_path.write_bytes(b"synthetic checkpoint fixture")
        checkpoint_sha = file_sha256(checkpoint_path)
        Path(f"{checkpoint_path}.sha256").write_text(
            f"{checkpoint_sha}  {checkpoint_path.name}\n", encoding="utf-8"
        )
        tensor_path = root / "train_val_tensors.pt"
        tensor_path.write_bytes(b"reviewed train and validation tensors")
        tensor_sha = file_sha256(tensor_path)
        manifest_path = root / "manifest.json"
        manifest = {
            "manifest_schema_version": "sft-v7-vertical-manifest/v1",
            "frozen_status": "frozen_unspent",
            "split_totals": {
                "train": 8000,
                "val": 800,
                "public_diagnostic": 600,
                "sealed_test": 600,
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        manifest_sha = file_sha256(manifest_path)
        schedule = retention.schedule_contract(400)
        provenance = {
            "stage": "sft_v7_vertical",
            "base_checkpoint_path": (
                "runs/formal_pretrain_14m_bpe3000_6000/checkpoints/step_05750.pt"
            ),
            "base_checkpoint_sha256": retention.BASE_CHECKPOINT_SHA256,
            "base_checkpoint_step": retention.BASE_STEP,
            "base_config_canonical_sha256": retention.BASE_CONFIG_CANONICAL_SHA256,
            "base_token_manifest_sha256": retention.BASE_TOKEN_MANIFEST_SHA256,
            "tokenizer_sha256": retention.TOKENIZER_SHA256,
            "sft_tensor_path": "data/sft/v7/train_val_tensors.pt",
            "sft_tensor_sha256": tensor_sha,
            "sft_dataset_manifest_sha256": manifest_sha,
            "sampling_schedule": schedule,
            "payload_summary": {"split_counts": {"train": 8000, "val": 800}},
            "public_records_consumed": 0,
            "sealed_records_consumed": 0,
        }
        effective = {
            "schema_version": "sft-v7-training-signature/v2",
            "model": retention.EXPECTED_CONFIG_MODEL,
            "provenance": provenance,
            "batch_size": 4,
            "learning_rate": 2e-5,
            "weight_decay": 0.05,
            "betas": [0.9, 0.95],
            "gradient_clip": 1.0,
            "sampling_schedule": schedule,
            "seed": 42,
            "target_steps": 2000,
            "phase1_steps": 400,
            "eval_interval": 500,
            "checkpoint_interval": 500,
            "eval_batch_size": 8,
            "device": "mps",
        }
        effective["signature_sha256"] = canonical_json_sha256(
            retention._signature_payload(effective)
        )
        effective_path = root / "effective_config.json"
        effective_path.write_text(
            json.dumps(effective, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        checkpoint = {
            "schema_version": "training-checkpoint/v1",
            "step": step,
            "config_sha256": effective["signature_sha256"],
            "model_state_dict": {},
            "extra": {
                **provenance,
                "sampler_states": {
                    retention.PHASE_ORDER[0]: {},
                    retention.PHASE_ORDER[1]: {},
                },
                "current_phase": retention.phase_for_next_update(step, 400),
                "phase1_steps": 400,
            },
        }
        return {
            "checkpoint": checkpoint,
            "checkpoint_path": checkpoint_path,
            "effective": effective,
            "effective_path": effective_path,
            "tensor_path": tensor_path,
            "manifest_path": manifest_path,
        }

    def test_fixed_window_bpc_and_signed_degradation_are_explicit(self):
        base = retention.fixed_window_bpc(
            retention.EXPECTED_BASELINE_FIXED_WINDOW_LOSS,
            retention.EXPECTED_VALIDATION_TOKENS,
            retention.EXPECTED_VALIDATION_CHARACTERS,
        )
        self.assertAlmostEqual(base, 3.745440719321712)

        passed = retention.compare_bpc(base * 1.10, base)
        failed = retention.compare_bpc(base * 1.10001, base)

        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])
        self.assertAlmostEqual(
            passed["absolute_delta_candidate_minus_baseline"], base * 0.10
        )
        self.assertAlmostEqual(
            passed["relative_degradation_candidate_minus_baseline"], 0.10
        )

    def test_strict_lineage_binds_signature_tensor_manifest_and_frozen_base(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = self._lineage_fixture(Path(temp_dir))
            result = retention.validate_sft_lineage(
                fixture["checkpoint"],
                checkpoint_path=fixture["checkpoint_path"],
                effective_config=fixture["effective"],
                effective_config_path=fixture["effective_path"],
                sft_tensor_path=fixture["tensor_path"],
                sft_manifest_path=fixture["manifest_path"],
            )

        self.assertTrue(result["verified"])
        self.assertEqual(result["base_checkpoint_step"], 5750)
        self.assertEqual(result["training_split_counts"], {"train": 8000, "val": 800})
        self.assertEqual(result["public_records_consumed"], 0)
        self.assertEqual(result["sealed_records_consumed"], 0)
        self.assertEqual(result["blind_body_reads"], 0)

    def test_lineage_rejects_checksum_data_and_blind_consumption_mismatches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = self._lineage_fixture(Path(temp_dir))
            Path(f"{fixture['checkpoint_path']}.sha256").write_text(
                "0" * 64, encoding="utf-8"
            )
            with self.assertRaisesRegex(
                retention.SFTV7RetentionError, "checksum does not match"
            ):
                retention.validate_sft_lineage(
                    fixture["checkpoint"],
                    checkpoint_path=fixture["checkpoint_path"],
                    effective_config=fixture["effective"],
                    effective_config_path=fixture["effective_path"],
                    sft_tensor_path=fixture["tensor_path"],
                    sft_manifest_path=fixture["manifest_path"],
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = self._lineage_fixture(Path(temp_dir))
            fixture["tensor_path"].write_bytes(b"tampered")
            with self.assertRaisesRegex(
                retention.SFTV7RetentionError, "training tensor SHA"
            ):
                retention.validate_sft_lineage(
                    fixture["checkpoint"],
                    checkpoint_path=fixture["checkpoint_path"],
                    effective_config=fixture["effective"],
                    effective_config_path=fixture["effective_path"],
                    sft_tensor_path=fixture["tensor_path"],
                    sft_manifest_path=fixture["manifest_path"],
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = self._lineage_fixture(Path(temp_dir))
            fixture["checkpoint"]["extra"]["sealed_records_consumed"] = 1
            with self.assertRaisesRegex(
                retention.SFTV7RetentionError, "sealed_records_consumed"
            ):
                retention.validate_sft_lineage(
                    fixture["checkpoint"],
                    checkpoint_path=fixture["checkpoint_path"],
                    effective_config=fixture["effective"],
                    effective_config_path=fixture["effective_path"],
                    sft_tensor_path=fixture["tensor_path"],
                    sft_manifest_path=fixture["manifest_path"],
                )

    def test_lineage_rejects_effective_signature_and_phase_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = self._lineage_fixture(Path(temp_dir))
            fixture["effective"]["learning_rate"] = 9e-4
            with self.assertRaisesRegex(
                retention.SFTV7RetentionError, "signatures do not agree"
            ):
                retention.validate_sft_lineage(
                    fixture["checkpoint"],
                    checkpoint_path=fixture["checkpoint_path"],
                    effective_config=fixture["effective"],
                    effective_config_path=fixture["effective_path"],
                    sft_tensor_path=fixture["tensor_path"],
                    sft_manifest_path=fixture["manifest_path"],
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = self._lineage_fixture(Path(temp_dir), step=500)
            fixture["checkpoint"]["extra"]["current_phase"] = retention.PHASE_ORDER[0]
            with self.assertRaisesRegex(
                retention.SFTV7RetentionError, "phase does not match"
            ):
                retention.validate_sft_lineage(
                    fixture["checkpoint"],
                    checkpoint_path=fixture["checkpoint_path"],
                    effective_config=fixture["effective"],
                    effective_config_path=fixture["effective_path"],
                    sft_tensor_path=fixture["tensor_path"],
                    sft_manifest_path=fixture["manifest_path"],
                )

    def test_baseline_reference_requires_exact_m019_contract(self):
        baseline = {
            "schema_version": "pretrain-capability-audit/v1",
            "checkpoint": {
                "step": retention.BASE_STEP,
                "sha256": retention.BASE_CHECKPOINT_SHA256,
            },
            "validation_diagnostic": {
                "split": "val",
                "tensor_sha256": retention.EXPECTED_VALIDATION_TENSOR_SHA256,
                "windows_evaluated": retention.EXPECTED_VALIDATION_WINDOWS,
                "tokens_evaluated": retention.EXPECTED_WINDOW_TOKENS,
                "window_selection": "deterministic_evenly_spaced",
                "loss": retention.EXPECTED_BASELINE_FIXED_WINDOW_LOSS,
                "perplexity": math.exp(retention.EXPECTED_BASELINE_FIXED_WINDOW_LOSS),
            },
            "probe_provenance": {
                "artifact_sha256": retention.EXPECTED_PROBE_SHA256,
                "prompts_sha256": retention.EXPECTED_PROMPTS_SHA256,
            },
            "generation_settings": {
                **retention.GENERATION_SETTINGS,
                "prompt_count": retention.EXPECTED_PROMPT_COUNT,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.json"
            path.write_text(json.dumps(baseline, sort_keys=True), encoding="utf-8")
            with mock.patch.object(
                retention, "EXPECTED_BASELINE_AUDIT_SHA256", file_sha256(path)
            ):
                result = retention.validate_baseline_reference(baseline, path)
                baseline["validation_diagnostic"]["windows_evaluated"] = 59
                with self.assertRaisesRegex(
                    retention.SFTV7RetentionError, "window_count"
                ):
                    retention.validate_baseline_reference(baseline, path)

        self.assertAlmostEqual(
            result["fixed_window_bpc"],
            retention.fixed_window_bpc(
                retention.EXPECTED_BASELINE_FIXED_WINDOW_LOSS,
                retention.EXPECTED_VALIDATION_TOKENS,
                retention.EXPECTED_VALIDATION_CHARACTERS,
            ),
        )
        self.assertTrue(result["full_history_reference_not_used_for_gate"])

    def test_automatic_gates_keep_external_review_pending(self):
        comparison = retention.compare_bpc(3.8, 3.75)
        summary = {
            "empty_rate": 0.0,
            "mechanical_degeneration_rate": 0.25,
        }
        gates = retention.build_automatic_gates(
            comparison=comparison, generation_summary=summary, generation_count=16
        )
        self.assertTrue(gates["passed"])

        summary["empty_rate"] = 1 / 16
        failed = retention.build_automatic_gates(
            comparison=comparison, generation_summary=summary, generation_count=16
        )
        self.assertFalse(failed["passed"])
        self.assertIn("all_16_continuations_nonempty", failed["failed_gate_names"])

    def test_generation_log_context_has_no_text_body(self):
        prompt = "这是不能进入日志的固定提示"
        continuation = "这是不能进入日志的模型续写"
        measured = {
            "continuation": continuation,
            "characters": len(continuation),
            "generated_tokens": 9,
            "stop_reason": "eos",
            "four_gram_repetition": 0.0,
            "mechanically_degenerate": False,
        }
        context = retention.safe_generation_log_context(
            prompt_index=1, prompt=prompt, measured=measured
        )
        serialized = json.dumps(context, ensure_ascii=False)

        self.assertNotIn(prompt, serialized)
        self.assertNotIn(continuation, serialized)
        self.assertNotIn("prompt", context)
        self.assertNotIn("continuation", context)
        self.assertEqual(context["prompt_characters"], len(prompt))

    def test_recursive_path_normalization_keeps_internal_paths_portable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            root.mkdir()
            internal = root / "data" / "cloud_v4" / "val.txt"
            external = Path(temp_dir) / "outside" / "candidate.pt"
            payload = {
                "probe": {
                    "source": {"path": str(internal)},
                    "external_path": str(external),
                },
                "continuation": "正文保持不变",
            }
            normalized = retention.normalize_artifact_paths(
                payload, project_root=root
            )
            retention.assert_publishable_paths(normalized, project_root=root)

        self.assertEqual(normalized["probe"]["source"]["path"], "data/cloud_v4/val.txt")
        self.assertEqual(
            normalized["probe"]["external_path"],
            "artifact://external_path/candidate.pt",
        )
        self.assertEqual(normalized["continuation"], "正文保持不变")

    def test_written_json_and_markdown_never_publish_local_project_root(self):
        project_root_text = retention.PROJECT_ROOT.as_posix()
        report = {
            "status": "automatic_retention_gates_passed_external_review_pending",
            "checkpoint_lineage": {
                "checkpoint_path": str(
                    retention.PROJECT_ROOT
                    / "runs/sft_v7_vertical_2000/checkpoints/step_00500.pt"
                ),
                "step": 500,
            },
            "validation_diagnostic": {"loss": 4.5, "perplexity": 90.0},
            "bpc_comparison": {
                "candidate_bpc": 3.8,
                "baseline_bpc": 3.75,
                "absolute_delta_candidate_minus_baseline": 0.05,
                "relative_degradation_candidate_minus_baseline": 0.013333,
            },
            "generation_summary": {
                "empty_rate": 0.0,
                "eos_stop_rate": 0.5,
                "mechanical_degeneration_rate": 0.0,
            },
            "cloze": {
                "top1_accuracy": 0.5,
                "cases": [
                    {
                        "id": "case-1",
                        "correct": "甲",
                        "predicted": "甲",
                        "correct_rank": 1,
                        "source": {
                            "path": str(
                                retention.PROJECT_ROOT / "data/cloud_v4/val.txt"
                            )
                        },
                    }
                ],
            },
            "automatic_hard_gates": {"gates": []},
            "generations": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "retention.json"
            markdown_path = Path(temp_dir) / "retention.md"
            written = retention.write_retention_outputs(
                report, json_path, markdown_path
            )
            json_body = json_path.read_text(encoding="utf-8")
            markdown_body = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(
            written["cloze"]["cases"][0]["source"]["path"],
            "data/cloud_v4/val.txt",
        )
        for body in (json_body, markdown_body):
            self.assertNotIn("/Users/", body)
            self.assertNotIn(project_root_text, body)

    def test_logs_rotate_by_module_and_redact_secrets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            levels = retention.resolve_log_levels(
                ["generation=INFO", "cloze=OFF"]
            )
            loggers = retention.configure_module_loggers(
                Path(temp_dir),
                "retention-test",
                levels,
                max_bytes=1024,
                backup_count=1,
                console=False,
            )
            loggers["generation"].info(
                "aggregate only",
                extra={"context": {"api_key": "secret-value", "count": 16}},
            )
            retention.close_module_loggers(loggers)
            body = "\n".join(
                path.read_text(encoding="utf-8")
                for path in Path(temp_dir).glob("*.generation.jsonl")
            )

        self.assertIn("aggregate only", body)
        self.assertIn("[REDACTED]", body)
        self.assertNotIn("secret-value", body)

    def test_cli_has_no_blind_split_and_rejects_blind_aliases(self):
        args = retention.parse_args([])
        self.assertFalse(hasattr(args, "held_out_split"))
        self.assertFalse(hasattr(args, "allow_test"))
        retention.validate_args(args)

        with self.assertRaisesRegex(ValueError, "test or sealed"):
            retention.validate_args(
                retention.parse_args(["--raw-validation", "data/cloud_v4/test.txt"])
            )
        with self.assertRaisesRegex(ValueError, "test or sealed"):
            retention.validate_args(
                retention.parse_args(["--probes", "data/sealed/probes.json"])
            )

    def test_markdown_explicitly_disclaims_human_review(self):
        report = {
            "status": "automatic_retention_gates_passed_external_review_pending",
            "checkpoint_lineage": {"checkpoint_path": "candidate.pt", "step": 500},
            "validation_diagnostic": {"loss": 4.5, "perplexity": 90.0},
            "bpc_comparison": {
                "candidate_bpc": 3.8,
                "baseline_bpc": 3.75,
                "absolute_delta_candidate_minus_baseline": 0.05,
                "relative_degradation_candidate_minus_baseline": 0.013333,
            },
            "generation_summary": {
                "empty_rate": 0.0,
                "eos_stop_rate": 0.5,
                "mechanical_degeneration_rate": 0.0,
            },
            "cloze": {"top1_accuracy": 0.5, "cases": []},
            "automatic_hard_gates": {
                "gates": [
                    {
                        "name": "relative_fixed_window_bpc_degradation",
                        "observed": 0.013333,
                        "operator": "<=",
                        "threshold": 0.10,
                        "passed": True,
                    }
                ]
            },
            "generations": [],
        }
        markdown = retention.build_markdown_report(report)
        self.assertIn("不能冒充真人验收", markdown)
        self.assertIn("独立真人最终抽查：**pending**", markdown)
        self.assertIn("candidate_eligible=false", markdown)


if __name__ == "__main__":
    unittest.main()
