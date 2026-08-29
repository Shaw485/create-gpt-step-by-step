from __future__ import annotations

import copy
from pathlib import Path
import signal
import tempfile
import unittest

import torch

from train_sft_v7 import (
    BASE_CONFIG_CANONICAL_SHA256,
    BASE_PARAMETER_COUNT,
    BASE_STEP,
    BASE_TOKEN_MANIFEST_SHA256,
    DEFAULT_PHASE1_STEPS,
    DIMENSION_ORDER,
    EXPECTED_MODEL_CONFIG,
    PHASE1_DIMENSION_WEIGHTS,
    PHASE2_DIMENSION_WEIGHTS,
    PHASE_ORDER,
    _reject_forbidden_keys,
    build_family_sampler,
    build_phase_samplers,
    checkpoint_payload,
    parse_args,
    phase_for_next_update,
    schedule_contract,
    validate_base_checkpoint_payload,
    validate_frozen_config,
    validate_record_tensor,
)
from training_runtime import EmergencyCheckpointHook, load_checkpoint, restore_checkpoint


def encoded(record_id: str, dimension: str) -> dict:
    return {
        "id": record_id,
        "split": "train",
        "primary_dimension": dimension,
        "task_family": "unit",
        "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "labels": torch.tensor([-100, 2, 3], dtype=torch.long),
    }


def base_payload() -> dict:
    return {
        "schema_version": "training-checkpoint/v1",
        "step": BASE_STEP,
        "config_sha256": BASE_CONFIG_CANONICAL_SHA256,
        "extra": {
            "initial_checkpoint": None,
            "parameter_count": BASE_PARAMETER_COUNT,
            "token_manifest_sha256": BASE_TOKEN_MANIFEST_SHA256,
            "model_config": dict(EXPECTED_MODEL_CONFIG),
        },
    }


class TrainSftV7Tests(unittest.TestCase):
    def test_trainer_rejects_any_public_or_blind_split_field(self):
        for key in ("public_records", "sealed_records", "test_records"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "forbidden key"):
                    _reject_forbidden_keys({key: []})

    def test_frozen_base_checkpoint_contract(self):
        validate_base_checkpoint_payload(base_payload())

        wrong = copy.deepcopy(base_payload())
        wrong["step"] = 6000
        with self.assertRaisesRegex(ValueError, "Step 5750"):
            validate_base_checkpoint_payload(wrong)

        sft = copy.deepcopy(base_payload())
        sft["extra"]["sampler_state"] = {}
        with self.assertRaisesRegex(ValueError, "SFT checkpoint"):
            validate_base_checkpoint_payload(sft)

    def test_family_sampler_visits_every_record_once_for_frozen_mix(self):
        counts = (18, 32, 14, 18, 13, 5)
        records = []
        for dimension, count in zip(DIMENSION_ORDER, counts):
            records.extend(
                encoded(f"{dimension}-{index}", dimension) for index in range(count)
            )
        sampler = build_family_sampler(records, torch.Generator().manual_seed(7))

        indexes = sampler.sample_indices(len(records))

        self.assertEqual(len(set(indexes)), len(records))
        self.assertEqual(sampler.coverage_summary()["coverage"], 1.0)
        self.assertEqual(
            [sampler.pool_draw_counts[name] for name in DIMENSION_ORDER],
            list(counts),
        )

    def test_record_tensor_contract_rejects_missing_supervision(self):
        record = encoded("x", DIMENSION_ORDER[0])
        validate_record_tensor(record, "train", 512)
        record["labels"] = torch.full((3,), -100, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "no supervised"):
            validate_record_tensor(record, "train", 512)

    def test_frozen_config_rejects_dropout_and_layer_norm_changes(self):
        from train_pretrain_v4 import load_config

        config = load_config(Path("configs/formal_pretrain_14m_bpe3000.json"))
        validate_frozen_config(config)
        dropout_changed = copy.deepcopy(config)
        dropout_changed["model"]["dropout"] = 0.2
        with self.assertRaisesRegex(ValueError, "frozen Step 5750"):
            validate_frozen_config(dropout_changed)
        layer_norm_changed = copy.deepcopy(config)
        layer_norm_changed["model"]["layer_norm_epsilon"] = 1e-6
        with self.assertRaisesRegex(ValueError, "frozen Step 5750"):
            validate_frozen_config(layer_norm_changed)

    def test_two_phase_schedule_is_explicit_and_not_scaled_by_target_steps(self):
        args = parse_args(["--target-steps", "20"])
        self.assertEqual(args.phase1_steps, DEFAULT_PHASE1_STEPS)
        self.assertEqual(phase_for_next_update(0, args.phase1_steps), PHASE_ORDER[0])
        self.assertEqual(phase_for_next_update(399, 400), PHASE_ORDER[0])
        self.assertEqual(phase_for_next_update(400, 400), PHASE_ORDER[1])
        self.assertEqual(
            schedule_contract(400)["phase1"]["weights"],
            PHASE1_DIMENSION_WEIGHTS,
        )
        self.assertEqual(
            schedule_contract(400)["phase2"]["weights"],
            PHASE2_DIMENSION_WEIGHTS,
        )

    def test_phase1_draws_only_core_chat_boundary_at_frozen_ratio(self):
        records = [
            encoded(f"record-{index}", dimension)
            for index, dimension in enumerate(DIMENSION_ORDER)
        ]
        samplers = build_phase_samplers(
            records, torch.Generator().manual_seed(17)
        )
        phase1 = samplers[PHASE_ORDER[0]]

        phase1.sample_indices(100)

        self.assertEqual(
            [phase1.pool_draw_counts[name] for name in DIMENSION_ORDER],
            [45, 0, 0, 40, 0, 15],
        )

    def test_emergency_checkpoint_round_trips_samplers_optimizer_and_rng(self):
        records = [
            encoded(f"record-{index}", dimension)
            for index, dimension in enumerate(DIMENSION_ORDER)
        ]
        generator = torch.Generator().manual_seed(29)
        samplers = build_phase_samplers(records, generator)
        samplers[PHASE_ORDER[0]].sample_indices(11)
        model = torch.nn.Linear(3, 3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        signature_sha256 = "a" * 64
        provenance = {"sampling_schedule": schedule_contract(7)}

        def payload_factory():
            return checkpoint_payload(
                model,
                optimizer,
                step=7,
                best_val_loss=1.25,
                history=[{"step": 0, "val_loss": 2.0}],
                generator=generator,
                signature_sha256=signature_sha256,
                provenance=provenance,
                samplers=samplers,
                phase1_steps=7,
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "emergency.pt"
            hook = EmergencyCheckpointHook(path, payload_factory, exit_after_save=False)
            hook._handle_signal(signal.SIGTERM, None)
            loaded = load_checkpoint(path)
            self.assertEqual(loaded["step"], 7)
            self.assertEqual(loaded["history"][0]["val_loss"], 2.0)
            self.assertEqual(loaded["extra"]["current_phase"], PHASE_ORDER[1])
            self.assertEqual(set(loaded["extra"]["sampler_states"]), set(PHASE_ORDER))

            expected_next = samplers[PHASE_ORDER[1]].sample_indices(12)
            restored_model = torch.nn.Linear(3, 3)
            restored_optimizer = torch.optim.AdamW(
                restored_model.parameters(), lr=2e-5
            )
            restored_generator = torch.Generator().manual_seed(999)
            restored_samplers = build_phase_samplers(records, restored_generator)
            resumed = restore_checkpoint(
                path,
                restored_model,
                restored_optimizer,
                restored_generator,
                expected_config_sha256=signature_sha256,
                restore_cuda_rng=False,
            )
            for phase in PHASE_ORDER:
                restored_samplers[phase].load_state_dict(
                    resumed.extra["sampler_states"][phase]
                )
            self.assertEqual(
                restored_samplers[PHASE_ORDER[1]].sample_indices(12),
                expected_next,
            )


if __name__ == "__main__":
    unittest.main()
