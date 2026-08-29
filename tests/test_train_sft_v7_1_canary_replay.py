from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from evaluate_sft_v7_1_canary import validate_canary_checkpoint_provenance
from prepare_bpe_sft_v7 import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SPECIAL_TOKEN_IDS,
    EXPECTED_TOKENIZER_SHA256,
    REQUIRED_BASE_CHECKPOINT,
)
from train_sft_v7 import BASE_CONFIG_CANONICAL_SHA256
from train_sft_v7_1_canary import (
    DeterministicShuffledEpochSampler,
    TRAINING_SIGNATURE_SCHEMA,
)
from train_sft_v7_1_canary_replay import (
    DEFAULT_CANARY_DATA,
    DEFAULT_INIT_CHECKPOINT,
    DEFAULT_REPLAY_TENSOR,
    DEFAULT_TOKEN_MANIFEST,
    EXPECTED_REPLAY_TENSOR_SHA256,
    REPLAY_SAMPLER_SCHEMA,
    TRAINING_VARIANT,
    DeterministicReplayTrainSampler,
    checkpoint_payload,
    combine_joint_losses,
    load_and_validate_replay_train,
    load_training_config,
    log_run_failed,
    log_run_started,
    optimizer_consumption,
    schedule_contract,
    validate_replay_manifest_contract,
)
from training_runtime import (
    canonical_json_sha256,
    close_module_loggers,
    configure_module_loggers,
    file_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def encoded(identifier: str) -> dict:
    return {
        "id": identifier,
        "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "labels": torch.tensor([-100, 2, 3], dtype=torch.long),
    }


def frozen_manifest() -> dict:
    return json.loads(
        (ROOT / DEFAULT_TOKEN_MANIFEST).read_text(encoding="utf-8")
    )


def replay_config() -> dict:
    return copy.deepcopy(
        load_training_config(ROOT / "configs/sft_v7_1_canary_replay_train.json")
    )["replay"]


def canary_provenance(
    *, tensor_sha: str, manifest_sha: str, identity_sha: str, summary: dict
) -> dict:
    return {
        "stage": "sft_v7_1_canary",
        "training_variant": TRAINING_VARIANT,
        "base_checkpoint_path": str(DEFAULT_INIT_CHECKPOINT),
        "base_checkpoint_sha256": REQUIRED_BASE_CHECKPOINT["sha256"],
        "base_checkpoint_step": REQUIRED_BASE_CHECKPOINT["step"],
        "base_config_canonical_sha256": BASE_CONFIG_CANONICAL_SHA256,
        "base_token_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "sft_tensor_path": str(DEFAULT_CANARY_DATA),
        "sft_tensor_sha256": tensor_sha,
        "canary_tensor_path": str(DEFAULT_CANARY_DATA),
        "canary_tensor_sha256": tensor_sha,
        "canary_dataset_manifest_sha256": manifest_sha,
        "canary_dataset_identity_sha256": identity_sha,
        "payload_summary": summary,
        "optimization_train_records": 64,
        "optimization_holdout_records": 0,
        "holdout_records_consumed": 0,
        "teacher_loss_holdout_records": 16,
        "development_unseen_wording_records": 16,
        "development_optimizer_records": 0,
        "development_records_consumed_for_teacher_loss": 16,
        "development_records_used_for_checkpoint_selection": 16,
        "public_records_consumed": 0,
        "sealed_records_consumed": 0,
        "replay_optimizer_split": "train",
        "replay_train_tensor_path": str(DEFAULT_REPLAY_TENSOR),
        "replay_train_tensor_sha256": EXPECTED_REPLAY_TENSOR_SHA256,
        "replay_token_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "replay_weight": 0.5,
        "replay_batch_size": 4,
        "replay_block_size": 128,
        "pretrain_validation_batches_consumed": 0,
        "pretrain_test_batches_consumed": 0,
    }


def signature_payload(provenance: dict, schedule: dict) -> dict:
    base_model = json.loads(
        (ROOT / "configs/formal_pretrain_14m_bpe3000.json").read_text(
            encoding="utf-8"
        )
    )["model"]
    return {
        "schema_version": TRAINING_SIGNATURE_SCHEMA,
        "model": base_model,
        "provenance": provenance,
        "training": {
            "target_steps": 400,
            "batch_size": 8,
            "sampler": "deterministic_shuffled_epoch/v1",
            "sft_batch_size": 8,
            "replay_batch_size": 4,
            "replay_block_size": 128,
            "replay_weight": 0.5,
            "learning_rate": 1e-5,
            "minimum_learning_rate": 1e-6,
            "warmup_steps": 20,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "gradient_clip": 1.0,
            "seed": 42,
            "replay_sampler": REPLAY_SAMPLER_SCHEMA,
            "joint_loss": TRAINING_VARIANT,
        },
        "schedule": schedule,
    }


class TrainSftV71CanaryReplayTests(unittest.TestCase):
    def test_default_config_is_conservative_and_train_only(self):
        config = load_training_config(
            ROOT / "configs/sft_v7_1_canary_replay_train.json"
        )
        settings = config["training"]
        self.assertEqual(settings["target_steps"], 400)
        self.assertEqual(settings["sft_batch_size"], 8)
        self.assertEqual(settings["replay_batch_size"], 4)
        self.assertEqual(settings["replay_block_size"], 128)
        self.assertEqual(settings["replay_weight"], 0.5)
        self.assertEqual(settings["learning_rate"], 1e-5)
        self.assertEqual(settings["minimum_learning_rate"], 1e-6)
        self.assertEqual(settings["warmup_steps"], 20)
        self.assertEqual(settings["eval_interval"], 25)
        self.assertEqual(settings["checkpoint_interval"], 25)
        self.assertEqual(config["replay"]["split"], "train")
        self.assertFalse(config["replay"]["validation_used_for_optimization"])
        self.assertFalse(config["replay"]["test_used_for_optimization"])
        self.assertFalse(config["canary"]["holdout_used_for_optimization"])
        self.assertEqual(config["canary"]["development_unseen_wording_records"], 16)
        self.assertTrue(config["canary"]["development_used_for_teacher_loss"])
        self.assertTrue(
            config["canary"]["development_used_for_checkpoint_selection"]
        )

    def test_joint_loss_formula_and_gradients_use_requested_weight(self):
        sft = torch.tensor(2.0, requires_grad=True)
        replay = torch.tensor(4.0, requires_grad=True)
        total = combine_joint_losses(sft, replay, 0.5)
        self.assertEqual(total.item(), 4.0)
        total.backward()
        self.assertEqual(sft.grad.item(), 1.0)
        self.assertEqual(replay.grad.item(), 0.5)

    def test_manifest_contract_rejects_non_train_replay(self):
        manifest = frozen_manifest()
        config = replay_config()
        summary = validate_replay_manifest_contract(
            manifest,
            config,
            manifest_sha256=EXPECTED_MANIFEST_SHA256,
            train_tensor_sha256=EXPECTED_REPLAY_TENSOR_SHA256,
        )
        self.assertEqual(summary["optimizer_split"], "train")
        self.assertEqual(summary["validation_batches_consumed"], 0)
        self.assertEqual(summary["test_batches_consumed"], 0)

        config["split"] = "val"
        with self.assertRaisesRegex(ValueError, "train split"):
            validate_replay_manifest_contract(
                manifest,
                config,
                manifest_sha256=EXPECTED_MANIFEST_SHA256,
                train_tensor_sha256=EXPECTED_REPLAY_TENSOR_SHA256,
            )

    def test_loader_touches_only_manifest_and_train_tensor(self):
        calls: list[tuple[str, str]] = []
        manifest = frozen_manifest()

        def digest(path: Path) -> str:
            calls.append(("digest", path.name))
            return (
                EXPECTED_MANIFEST_SHA256
                if path.name == "token_manifest.json"
                else EXPECTED_REPLAY_TENSOR_SHA256
            )

        def json_loader(path: Path) -> dict:
            calls.append(("json", path.name))
            return manifest

        def tensor_loader(path: Path) -> torch.Tensor:
            calls.append(("tensor", path.name))
            return torch.zeros(3_223_207, dtype=torch.long)

        tensor, summary = load_and_validate_replay_train(
            replay_config(),
            root=ROOT,
            digest=digest,
            tensor_loader=tensor_loader,
            json_loader=json_loader,
        )
        self.assertEqual(len(tensor), 3_223_207)
        self.assertEqual(summary["optimizer_split"], "train")
        names = [name for _, name in calls]
        self.assertEqual(
            names,
            ["token_manifest.json", "train_tokens.pt", "token_manifest.json", "train_tokens.pt"],
        )
        self.assertNotIn("val_tokens.pt", names)
        self.assertNotIn("test_tokens.pt", names)

    def test_replay_sampler_state_round_trip_restores_next_windows(self):
        data = torch.arange(500, dtype=torch.long)
        generator = torch.Generator().manual_seed(13)
        first = DeterministicReplayTrainSampler(
            data,
            generator,
            train_tensor_sha256="a" * 64,
            batch_size=3,
            block_size=8,
        )
        first.sample_batch(torch.device("cpu"))
        state = first.state_dict()
        expected_inputs, expected_targets = first.sample_batch(torch.device("cpu"))

        second = DeterministicReplayTrainSampler(
            data,
            torch.Generator().manual_seed(999),
            train_tensor_sha256="a" * 64,
            batch_size=3,
            block_size=8,
        )
        second.load_state_dict(state)
        actual_inputs, actual_targets = second.sample_batch(torch.device("cpu"))
        self.assertTrue(torch.equal(actual_inputs, expected_inputs))
        self.assertTrue(torch.equal(actual_targets, expected_targets))
        self.assertEqual(second.batches_drawn, 2)
        self.assertEqual(second.target_tokens_consumed, 48)

    def test_checkpoint_preserves_two_samplers_rng_and_restricted_split_zeros(self):
        records = [encoded(f"train-{index}") for index in range(64)]
        sft_generator = torch.Generator().manual_seed(5)
        sft_sampler = DeterministicShuffledEpochSampler(records, sft_generator)
        sft_sampler.sample_indices(8)
        replay_sampler = DeterministicReplayTrainSampler(
            torch.arange(500, dtype=torch.long),
            torch.Generator().manual_seed(6),
            train_tensor_sha256=EXPECTED_REPLAY_TENSOR_SHA256,
            batch_size=4,
            block_size=16,
        )
        replay_sampler.sample_batch(torch.device("cpu"))
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        settings = load_training_config(
            ROOT / "configs/sft_v7_1_canary_replay_train.json"
        )["training"]
        schedule = schedule_contract(settings)
        provenance = canary_provenance(
            tensor_sha="b" * 64,
            manifest_sha="c" * 64,
            identity_sha="d" * 64,
            summary={"split_counts": {"train": 64, "holdout_eval": 16}},
        )
        payload = checkpoint_payload(
            model,
            optimizer,
            step=1,
            best_eval_loss=2.0,
            history=[],
            sft_generator=sft_generator,
            signature_sha256="e" * 64,
            provenance=provenance,
            sft_sampler=sft_sampler,
            replay_sampler=replay_sampler,
            schedule=schedule,
        )
        extra = payload["extra"]
        self.assertEqual(extra["stage"], "sft_v7_1_canary")
        self.assertEqual(extra["sampler_state"], extra["sft_sampler_state"])
        self.assertIsInstance(extra["replay_sampler_state"]["generator_state"], torch.Tensor)
        consumption = extra["optimizer_consumption"]
        self.assertEqual(consumption["canary_train_records_drawn"], 8)
        self.assertEqual(consumption["pretrain_train_target_tokens_consumed"], 64)
        for key in (
            "canary_holdout_records_drawn",
            "sft_public_records_drawn",
            "sft_sealed_records_drawn",
            "pretrain_validation_batches_drawn",
            "pretrain_test_batches_drawn",
        ):
            self.assertEqual(consumption[key], 0)

    def test_checkpoint_shape_is_accepted_by_primary_canary_lineage_validator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensor_path = root / "train_eval_tensors.pt"
            tensor_path.write_bytes(b"synthetic canary tensor identity")
            tensor_sha = file_sha256(tensor_path)
            summary = {"split_counts": {"train": 64, "holdout_eval": 16}}
            provenance = canary_provenance(
                tensor_sha=tensor_sha,
                manifest_sha="c" * 64,
                identity_sha="d" * 64,
                summary=summary,
            )
            settings = load_training_config(
                ROOT / "configs/sft_v7_1_canary_replay_train.json"
            )["training"]
            schedule = schedule_contract(settings)
            signature = signature_payload(provenance, schedule)
            effective = {
                **signature,
                "signature_sha256": canonical_json_sha256(signature),
            }
            records = [encoded(f"train-{index}") for index in range(64)]
            sft_generator = torch.Generator().manual_seed(5)
            sft_sampler = DeterministicShuffledEpochSampler(records, sft_generator)
            replay_sampler = DeterministicReplayTrainSampler(
                torch.arange(500, dtype=torch.long),
                torch.Generator().manual_seed(6),
                train_tensor_sha256=EXPECTED_REPLAY_TENSOR_SHA256,
                batch_size=4,
                block_size=128,
            )
            model = torch.nn.Linear(2, 2)
            checkpoint = checkpoint_payload(
                model,
                torch.optim.AdamW(model.parameters(), lr=1e-5),
                step=1,
                best_eval_loss=2.0,
                history=[],
                sft_generator=sft_generator,
                signature_sha256=effective["signature_sha256"],
                provenance=provenance,
                sft_sampler=sft_sampler,
                replay_sampler=replay_sampler,
                schedule=schedule,
            )
            checkpoint_path = root / "candidate.pt"
            checkpoint_path.write_bytes(b"synthetic checkpoint identity")
            result = validate_canary_checkpoint_provenance(
                checkpoint,
                checkpoint_path,
                tensor_path=tensor_path,
                tensor_payload={
                    "canary_dataset_manifest_sha256": "c" * 64,
                    "canary_dataset_identity_sha256": "d" * 64,
                },
                effective_config=effective,
            )
            self.assertEqual(result["checkpoint_mode"], "canary")
            self.assertEqual(result["training_split_counts"], summary["split_counts"])

    def test_module_logs_record_success_failure_without_body_secret_or_path(self):
        with tempfile.TemporaryDirectory() as directory:
            loggers = configure_module_loggers(
                directory,
                "replay-log-test",
                {
                    "data": "INFO",
                    "sft_training": "INFO",
                    "replay_training": "INFO",
                    "validation": "INFO",
                    "checkpoint": "INFO",
                    "orchestrator": "INFO",
                },
                max_bytes=2048,
                backup_count=1,
                console=False,
            )
            try:
                log_run_started(
                    loggers,
                    start_step=0,
                    target_step=400,
                    run_signature="a" * 64,
                )
                log_run_failed(
                    loggers,
                    ValueError(
                        "question=萧炎是谁 password=hidden /private/sensitive/path"
                    ),
                    step=7,
                )
            finally:
                close_module_loggers(loggers)
            for module in (
                "data",
                "sft_training",
                "replay_training",
                "validation",
                "checkpoint",
                "orchestrator",
            ):
                self.assertTrue(
                    (Path(directory) / f"replay-log-test.{module}.jsonl").is_file()
                )
            text = (
                Path(directory) / "replay-log-test.orchestrator.jsonl"
            ).read_text(encoding="utf-8")
            events = [json.loads(line) for line in text.splitlines()]
            self.assertEqual(len(events), 2)
            self.assertTrue(all(event["run_id"] == "replay-log-test" for event in events))
            self.assertTrue(all(event["timestamp"].endswith("+00:00") for event in events))
            self.assertNotIn("萧炎", text)
            self.assertNotIn("hidden", text)
            self.assertNotIn("/private/sensitive/path", text)

    def test_optimizer_consumption_distinguishes_development_from_optimizer_use(self):
        records = [encoded(f"train-{index}") for index in range(64)]
        sft_sampler = DeterministicShuffledEpochSampler(
            records, torch.Generator().manual_seed(1)
        )
        replay_sampler = DeterministicReplayTrainSampler(
            torch.arange(500, dtype=torch.long),
            torch.Generator().manual_seed(2),
            train_tensor_sha256="f" * 64,
            batch_size=4,
            block_size=32,
        )
        consumption = optimizer_consumption(
            sft_sampler, replay_sampler, development_evaluations=3
        )
        self.assertEqual(consumption["canary_holdout_records_drawn"], 0)
        self.assertEqual(consumption["canary_development_optimizer_records_drawn"], 0)
        self.assertEqual(consumption["canary_development_teacher_evaluations"], 3)
        self.assertEqual(
            consumption["canary_development_teacher_forward_records"], 48
        )
        self.assertEqual(
            consumption["canary_development_checkpoint_selection_events"], 3
        )
        self.assertEqual(
            consumption["canary_development_records_per_checkpoint_selection"],
            16,
        )
        self.assertEqual(consumption["sft_public_records_drawn"], 0)
        self.assertEqual(consumption["sft_sealed_records_drawn"], 0)
        self.assertEqual(consumption["pretrain_validation_batches_drawn"], 0)
        self.assertEqual(consumption["pretrain_test_batches_drawn"], 0)


if __name__ == "__main__":
    unittest.main()
