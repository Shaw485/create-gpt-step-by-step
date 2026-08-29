from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from prepare_bpe_sft_v7 import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SPECIAL_TOKEN_IDS,
    EXPECTED_TOKENIZER_SHA256,
    REQUIRED_BASE_CHECKPOINT,
)
from prepare_sft_v7_1_canary import (
    EXPECTED_CANARY_DATASET_IDENTITY_SHA256,
    EXPECTED_CANARY_MANIFEST_SHA256,
    EXPECTED_CANARY_SOURCE_SHA256,
    TENSOR_SCHEMA,
)
from train_sft_v7_1_canary import (
    DeterministicShuffledEpochSampler,
    build_optimizer,
    checkpoint_payload,
    learning_rate_for_update,
    load_training_config,
    log_training_failure,
    schedule_contract,
    validate_canary_tensor_payload,
)
from training_runtime import (
    canonical_json_sha256,
    close_module_loggers,
    configure_module_loggers,
)


def encoded(identifier: str, split: str) -> dict:
    eos_id = EXPECTED_SPECIAL_TOKEN_IDS["<EOS>"]
    return {
        "id": identifier,
        "split": split,
        "primary_dimension": "parameter_core_fact_and_correction",
        "task_family": "canary_known_core",
        "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "labels": torch.tensor([-100, 12, eos_id], dtype=torch.long),
        "evaluation": {
            "metric": "required_terms_all",
            "required_terms": ["萧炎"],
            "forbidden_terms": ["资料不足"],
            "known_fact": True,
        },
    }


def tensor_payload() -> dict:
    base = dict(REQUIRED_BASE_CHECKPOINT)
    base["binding_sha256"] = canonical_json_sha256(REQUIRED_BASE_CHECKPOINT)
    source_hashes = dict(EXPECTED_CANARY_SOURCE_SHA256)
    payload = {
        "schema_version": TENSOR_SCHEMA,
        "train_records": [encoded(f"train-{index}", "train") for index in range(64)],
        "eval_records": [
            encoded(f"eval-{index}", "holdout_eval") for index in range(16)
        ],
        "vocab_size": 7465,
        "stoi": {},
        "itos": {},
        "special_token_ids": dict(EXPECTED_SPECIAL_TOKEN_IDS),
        "ignore_index": -100,
        "tokenizer_path": "data/scaling_a/bpe_3000/tokenizer.json",
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "bpe_token_manifest_path": "data/scaling_a/bpe_3000/token_manifest.json",
        "bpe_token_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "canary_dataset_manifest_path": "data/sft/v7_1_canary/manifest.json",
        "canary_dataset_manifest_sha256": EXPECTED_CANARY_MANIFEST_SHA256,
        "canary_dataset_identity_sha256": EXPECTED_CANARY_DATASET_IDENTITY_SHA256,
        "source_jsonl_paths": {
            "train": "data/sft/v7_1_canary/train.jsonl",
            "holdout_eval": "data/sft/v7_1_canary/holdout_eval.jsonl",
        },
        "source_jsonl_sha256": source_hashes,
        "required_base_checkpoint": base,
    }
    binding = {
        "schema_version": TENSOR_SCHEMA,
        "source_jsonl_sha256": source_hashes,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "bpe_token_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "canary_dataset_manifest_sha256": EXPECTED_CANARY_MANIFEST_SHA256,
        "canary_dataset_identity_sha256": EXPECTED_CANARY_DATASET_IDENTITY_SHA256,
        "required_base_checkpoint": base,
    }
    payload["artifact_binding_sha256"] = canonical_json_sha256(binding)
    return payload


class TrainSftV71CanaryTests(unittest.TestCase):
    def test_training_config_pins_canonical_tensor_and_development_role(self):
        config = load_training_config(Path("configs/sft_v7_1_canary_train.json"))
        self.assertEqual(
            config["data"]["holdout_eval_role"],
            "unseen_question_development_selection",
        )
        changed = copy.deepcopy(config)
        changed["data"]["tensor_path"] = "copied/train_eval_tensors.pt"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configuration path changed"):
                load_training_config(path)

    def test_training_failure_log_is_structured_and_has_no_traceback_or_path(self):
        with tempfile.TemporaryDirectory() as directory:
            loggers = configure_module_loggers(
                Path(directory),
                "test-run",
                {"training": "INFO"},
                max_bytes=4096,
                backup_count=1,
                console=False,
            )
            try:
                log_training_failure(
                    loggers["training"],
                    RuntimeError("private body /Users/example/secret.txt"),
                    step=7,
                )
            finally:
                close_module_loggers(loggers)
            text = next(Path(directory).glob("*.training.jsonl")).read_text(
                encoding="utf-8"
            )
            event = json.loads(text)
            self.assertEqual(event["context"]["error_code"], "CANARY_TRAINING_FAILED")
            self.assertEqual(event["context"]["error_type"], "RuntimeError")
            self.assertEqual(event["context"]["step"], 7)
            self.assertNotIn("Traceback", text)
            self.assertNotIn("/Users/", text)
            self.assertNotIn("private body", text)

    def test_tensor_contract_exposes_only_64_optimizer_records(self):
        summary = validate_canary_tensor_payload(tensor_payload(), 512)
        self.assertEqual(summary["split_counts"], {"train": 64, "holdout_eval": 16})
        self.assertEqual(summary["optimizer_record_count"], 64)
        self.assertEqual(summary["holdout_optimizer_record_count"], 0)

        bad = tensor_payload()
        bad["public_records"] = []
        with self.assertRaisesRegex(ValueError, "restricted key"):
            validate_canary_tensor_payload(bad, 512)

    def test_shuffled_epoch_visits_all_records_before_reuse(self):
        records = [encoded(f"record-{index}", "train") for index in range(64)]
        sampler = DeterministicShuffledEpochSampler(
            records, torch.Generator().manual_seed(7)
        )
        first_epoch = sampler.sample_indices(64)
        self.assertEqual(len(set(first_epoch)), 64)
        self.assertEqual(sampler.coverage_summary()["coverage"], 1.0)
        self.assertEqual(sampler.coverage_summary()["minimum_draws_per_record"], 1)
        sampler.sample_indices(8)
        self.assertEqual(sampler.coverage_summary()["epochs_started"], 2)

    def test_sampler_state_round_trip_preserves_next_draw(self):
        records = [encoded(f"record-{index}", "train") for index in range(64)]
        first_generator = torch.Generator().manual_seed(11)
        first = DeterministicShuffledEpochSampler(records, first_generator)
        first.sample_indices(13)
        state = first.state_dict()
        generator_state = first_generator.get_state()
        expected = first.sample_indices(19)

        second_generator = torch.Generator().manual_seed(999)
        second_generator.set_state(generator_state)
        second = DeterministicShuffledEpochSampler(records, second_generator)
        second.load_state_dict(state)
        self.assertEqual(second.sample_indices(19), expected)

    def test_learning_rate_warms_up_and_finishes_at_floor(self):
        values = [
            learning_rate_for_update(
                step,
                target_steps=400,
                warmup_steps=20,
                peak_learning_rate=1e-4,
                minimum_learning_rate=1e-5,
            )
            for step in range(400)
        ]
        self.assertAlmostEqual(values[0], 5e-6)
        self.assertAlmostEqual(values[19], 1e-4)
        self.assertAlmostEqual(values[-1], 1e-5)
        self.assertTrue(all(a <= b for a, b in zip(values[:19], values[1:20])))
        self.assertTrue(all(a >= b for a, b in zip(values[19:], values[20:])))

    def test_optimizer_does_not_decay_bias_or_norm_vectors(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.LayerNorm(4), torch.nn.Linear(4, 2)
        )
        optimizer = build_optimizer(
            model, learning_rate=1e-4, weight_decay=0.01, betas=(0.9, 0.95)
        )
        self.assertEqual([group["weight_decay"] for group in optimizer.param_groups], [0.01, 0.0])
        self.assertGreater(len(optimizer.param_groups[0]["params"]), 0)
        self.assertGreater(len(optimizer.param_groups[1]["params"]), 0)

    def test_checkpoint_records_all_capacity_provenance(self):
        records = [encoded(f"record-{index}", "train") for index in range(64)]
        generator = torch.Generator().manual_seed(13)
        sampler = DeterministicShuffledEpochSampler(records, generator)
        sampler.sample_indices(8)
        model = torch.nn.Linear(3, 3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        config = load_training_config(Path("configs/sft_v7_1_canary_train.json"))
        schedule = schedule_contract(config["training"])
        provenance = {
            "stage": "sft_v7_1_canary",
            "canary_tensor_path": "data/sft/v7_1_canary/train_eval_tensors.pt",
            "canary_tensor_sha256": "a" * 64,
            "payload_summary": {
                "split_counts": {"train": 64, "holdout_eval": 16}
            },
            "optimization_train_records": 64,
            "optimization_holdout_records": 0,
            "development_unseen_wording_records": 16,
            "development_optimizer_records": 0,
            "development_records_used_for_optimization": 0,
            "development_records_consumed_for_teacher_loss": 16,
            "development_records_used_for_checkpoint_selection": 16,
            "holdout_records_consumed": 0,
            "teacher_loss_holdout_records": 16,
            "public_records_consumed": 0,
            "sealed_records_consumed": 0,
        }
        payload = checkpoint_payload(
            model,
            optimizer,
            step=8,
            best_eval_loss=1.2,
            history=[{"step": 0}],
            generator=generator,
            signature_sha256="f" * 64,
            provenance=provenance,
            sampler=sampler,
            schedule=schedule,
        )
        extra = payload["extra"]
        self.assertEqual(extra["stage"], "sft_v7_1_canary")
        self.assertEqual(extra["payload_summary"]["split_counts"], {"train": 64, "holdout_eval": 16})
        self.assertEqual(extra["holdout_records_consumed"], 0)
        self.assertEqual(extra["development_records_consumed_for_teacher_loss"], 16)
        self.assertEqual(extra["development_records_used_for_checkpoint_selection"], 16)
        self.assertEqual(extra["development_records_used_for_optimization"], 0)
        self.assertEqual(extra["public_records_consumed"], 0)
        self.assertEqual(extra["sealed_records_consumed"], 0)
        self.assertIn("sampler_state", extra)
        self.assertIn("learning_rate_schedule", extra)


if __name__ == "__main__":
    unittest.main()
