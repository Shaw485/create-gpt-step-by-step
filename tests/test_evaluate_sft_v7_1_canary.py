from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from evaluate_sft_v7_1_canary import (
    evaluate_gates,
    load_and_validate_effective_config,
    normalize_answer,
    parse_args,
    score_generation,
    self_relation_error_reasons,
    severe_repetition,
    summarize_by_fact,
    summarize_split,
    validate_canary_checkpoint_provenance,
)
from prepare_bpe_sft_v7 import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TOKENIZER_SHA256,
    REQUIRED_BASE_CHECKPOINT,
)
from train_sft_v7 import BASE_CONFIG_CANONICAL_SHA256
from train_sft_v7_1_canary import (
    DEFAULT_INIT_CHECKPOINT,
    EXPECTED_STAGE,
    TRAINING_SIGNATURE_SCHEMA,
    load_training_config,
)
from training_runtime import canonical_json_sha256, file_sha256


def record(
    *,
    fact_id: str = "xiaoyan_identity",
    split: str = "train",
    answer: str = "萧炎是萧战的儿子。",
) -> dict:
    return {
        "id": f"{split}-{fact_id}",
        "split": split,
        "fact_id": fact_id,
        "answer": answer,
        "evaluation": {
            "metric": "required_terms_all",
            "required_terms": ["萧炎", "萧战"],
            "forbidden_terms": ["资料不足"],
            "known_fact": True,
        },
    }


def generation(text: str, *, eos: bool = True) -> dict:
    return {
        "generated_text": text,
        "stopped_on_eos": eos,
        "truncated": not eos,
    }


def scored_sample(
    fact_id: str,
    split: str,
    *,
    exact: bool,
    eos: bool = True,
) -> dict:
    reference = "萧炎是萧战的儿子。"
    generated = reference if exact else "萧炎和萧战有关。"
    item = record(fact_id=fact_id, split=split, answer=reference)
    return {
        "fact_id": fact_id,
        "split": split,
        "score": score_generation(item, generation(generated, eos=eos)),
    }


class EvaluateSftV71CanaryTests(unittest.TestCase):
    def test_normalized_exact_ignores_only_spacing_and_punctuation(self):
        self.assertEqual(
            normalize_answer(" 萧炎是萧战的儿子！ "),
            normalize_answer("萧炎是萧战的儿子。"),
        )
        self.assertNotEqual(
            normalize_answer("萧炎不是萧战的儿子。"),
            normalize_answer("萧炎是萧战的儿子。"),
        )

    def test_required_terms_cannot_make_a_contradiction_pass(self):
        score = score_generation(
            record(), generation("萧炎不是萧战的儿子。", eos=True)
        )
        self.assertTrue(score["required_terms_all"])
        self.assertTrue(score["keypoint_pass"])
        self.assertFalse(score["normalized_exact_answer"])
        self.assertFalse(score["exact_answer_pass"])

        appended = score_generation(
            record(),
            generation("萧炎是萧战的儿子，但萧战也是萧炎的儿子。", eos=True),
        )
        self.assertFalse(appended["exact_answer_pass"])
        self.assertTrue(appended["self_relation_error"])

    def test_repetition_and_relation_direction_diagnostics(self):
        self.assertTrue(severe_repetition("药老药老药老药老"))
        self.assertFalse(severe_repetition("药老是萧炎的老师。"))
        self.assertIn(
            "direction_reversal:萧炎是药老的老师",
            self_relation_error_reasons(
                "yaolao_teacher", "萧炎是药老的老师。"
            ),
        )
        self.assertIn(
            "self_relation:萧炎",
            self_relation_error_reasons("xiaoyan_identity", "萧炎是萧炎本人。"),
        )

    def test_per_fact_gate_catches_failure_hidden_by_global_average(self):
        facts = [f"fact-{index}" for index in range(8)]
        samples = []
        for fact_index, fact_id in enumerate(facts):
            train_passes = 6 if fact_index == 0 else 8
            samples.extend(
                scored_sample(fact_id, "train", exact=index < train_passes)
                for index in range(8)
            )
            samples.extend(
                scored_sample(fact_id, "holdout_eval", exact=True)
                for _ in range(2)
            )
        split_metrics = {
            split: summarize_split(
                [sample for sample in samples if sample["split"] == split]
            )
            for split in ("train", "holdout_eval")
        }
        self.assertGreaterEqual(split_metrics["train"]["exact_answer_rate"], 0.95)
        fact_metrics = summarize_by_fact(samples)
        config = load_training_config(Path("configs/sft_v7_1_canary_train.json"))
        gates = evaluate_gates(split_metrics, config["gates"], fact_metrics)
        self.assertTrue(gates["checks"]["train_exact_answer_rate"])
        self.assertFalse(gates["checks"]["per_fact_train_exact_min"])
        self.assertEqual(gates["observed"]["per_fact_train_failures"], ["fact-0"])
        self.assertEqual(gates["status"], "fail")

    def test_all_exact_samples_pass_aggregate_and_per_fact_gates(self):
        facts = [f"fact-{index}" for index in range(8)]
        samples = []
        for fact_id in facts:
            samples.extend(scored_sample(fact_id, "train", exact=True) for _ in range(8))
            samples.extend(
                scored_sample(fact_id, "holdout_eval", exact=True) for _ in range(2)
            )
        metrics = {
            split: summarize_split(
                [sample for sample in samples if sample["split"] == split]
            )
            for split in ("train", "holdout_eval")
        }
        by_fact = summarize_by_fact(samples)
        config = load_training_config(Path("configs/sft_v7_1_canary_train.json"))
        gates = evaluate_gates(metrics, config["gates"], by_fact)
        self.assertEqual(gates["status"], "pass")
        self.assertTrue(all(gates["checks"].values()))

    def test_effective_config_signature_is_recomputed(self):
        signature = {
            "schema_version": TRAINING_SIGNATURE_SCHEMA,
            "model": {"layers": 10},
            "provenance": {"stage": EXPECTED_STAGE},
            "training": {"target_steps": 400},
            "schedule": {"strategy": "linear_warmup_cosine_decay/v1"},
        }
        effective = {**signature, "signature_sha256": canonical_json_sha256(signature)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "effective_config.json"
            path.write_text(json.dumps(effective), encoding="utf-8")
            loaded = load_and_validate_effective_config(path)
            self.assertEqual(loaded["signature_sha256"], effective["signature_sha256"])
            effective["training"]["target_steps"] = 401
            path.write_text(json.dumps(effective), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "signature"):
                load_and_validate_effective_config(path)

    def test_canary_checkpoint_provenance_binds_tensor_and_zero_training_holdout(self):
        signature = {
            "schema_version": TRAINING_SIGNATURE_SCHEMA,
            "model": {},
            "provenance": {},
            "training": {},
            "schedule": {},
        }
        effective = {**signature, "signature_sha256": canonical_json_sha256(signature)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensor_path = root / "train_eval_tensors.pt"
            tensor_path.write_bytes(b"tensor")
            checkpoint_path = root / "latest.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            tensor_payload = {
                "canary_dataset_manifest_sha256": "c" * 64,
                "canary_dataset_identity_sha256": "d" * 64,
            }
            checkpoint = {
                "schema_version": "training-checkpoint/v1",
                "config_sha256": effective["signature_sha256"],
                "step": 400,
                "extra": {
                    "stage": EXPECTED_STAGE,
                    "base_checkpoint_path": str(DEFAULT_INIT_CHECKPOINT),
                    "base_checkpoint_sha256": REQUIRED_BASE_CHECKPOINT["sha256"],
                    "base_checkpoint_step": REQUIRED_BASE_CHECKPOINT["step"],
                    "base_config_canonical_sha256": BASE_CONFIG_CANONICAL_SHA256,
                    "base_token_manifest_sha256": EXPECTED_MANIFEST_SHA256,
                    "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
                    "canary_tensor_path": "data/sft/v7_1_canary/train_eval_tensors.pt",
                    "canary_tensor_sha256": file_sha256(tensor_path),
                    "canary_dataset_manifest_sha256": "c" * 64,
                    "canary_dataset_identity_sha256": "d" * 64,
                    "optimization_train_records": 64,
                    "optimization_holdout_records": 0,
                    "holdout_records_consumed": 0,
                    "teacher_loss_holdout_records": 16,
                    "public_records_consumed": 0,
                    "sealed_records_consumed": 0,
                    "payload_summary": {
                        "split_counts": {"train": 64, "holdout_eval": 16}
                    },
                    "sampler_state": {"strategy": "deterministic_shuffled_epoch/v1"},
                },
            }
            result = validate_canary_checkpoint_provenance(
                checkpoint,
                checkpoint_path,
                tensor_path=tensor_path,
                tensor_payload=tensor_payload,
                effective_config=effective,
            )
            self.assertEqual(result["training_split_counts"], {"train": 64, "holdout_eval": 16})
            self.assertTrue(
                result["development_provenance_inferred_from_legacy_checkpoint"]
            )
            self.assertEqual(
                result["development_records_consumed_for_teacher_loss"], 16
            )
            self.assertEqual(
                result["development_records_used_for_checkpoint_selection"], 16
            )

            current = copy.deepcopy(checkpoint)
            current["extra"].update(
                {
                    "development_unseen_wording_records": 16,
                    "development_records_used_for_optimization": 0,
                    "development_records_consumed_for_teacher_loss": 16,
                    "development_records_used_for_checkpoint_selection": 16,
                }
            )
            current_result = validate_canary_checkpoint_provenance(
                current,
                checkpoint_path,
                tensor_path=tensor_path,
                tensor_payload=tensor_payload,
                effective_config=effective,
            )
            self.assertFalse(
                current_result[
                    "development_provenance_inferred_from_legacy_checkpoint"
                ]
            )

            broken = json.loads(json.dumps(checkpoint))
            broken["extra"]["holdout_records_consumed"] = 1
            with self.assertRaisesRegex(ValueError, "holdout_records_consumed"):
                validate_canary_checkpoint_provenance(
                    broken,
                    checkpoint_path,
                    tensor_path=tensor_path,
                    tensor_payload=tensor_payload,
                    effective_config=effective,
                )

    def test_generation_logging_level_is_independently_configurable(self):
        args = parse_args(
            [
                "--generation-log-level",
                "DEBUG",
                "--data-log-level",
                "OFF",
                "--checkpoint-mode",
                "base",
            ]
        )
        self.assertEqual(args.generation_log_level, "DEBUG")
        self.assertEqual(args.data_log_level, "OFF")


if __name__ == "__main__":
    unittest.main()
