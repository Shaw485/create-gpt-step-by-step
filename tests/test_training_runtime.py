import json
import logging
import os
from pathlib import Path
import signal
import tempfile
import unittest

import torch

from training_runtime import (
    ConfigurationError,
    EarlyStopping,
    EmergencyCheckpointHook,
    NumericalFailure,
    RunStateWriter,
    WallClockExceeded,
    WallClockLimit,
    assert_finite_gradients,
    assert_finite_tensor,
    atomic_save_checkpoint,
    build_checkpoint_payload,
    configure_module_loggers,
    file_sha256,
    load_checkpoint,
    load_versioned_config,
    restore_checkpoint,
    resolve_module_log_levels,
)


CONFIG_SECTIONS = {
    "project": {},
    "data": {},
    "model": {},
    "pretraining": {},
    "sft": {},
    "evaluation": {},
    "hardware": {},
    "runtime": {},
    "logging": {},
    "safety": {},
    "artifacts": {},
}


class FakeScaler:
    def __init__(self, scale=16.0):
        self.scale = scale

    def state_dict(self):
        return {"scale": self.scale}

    def load_state_dict(self, state):
        self.scale = state["scale"]


class TrainingRuntimeTests(unittest.TestCase):
    def test_module_log_levels_can_be_overridden_independently(self):
        variable = "GPT_LOG_LEVEL_VALIDATION"
        previous = os.environ.get(variable)
        os.environ[variable] = "DEBUG"
        try:
            resolved = resolve_module_log_levels(
                {"data": "INFO", "validation": "OFF"}
            )
        finally:
            if previous is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous

        self.assertEqual(resolved, {"data": "INFO", "validation": "DEBUG"})

    def test_versioned_config_requires_matching_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            payload = {
                "schema_version": "cloud-training-config/v1",
                "config_version": "1.2.3",
                **CONFIG_SECTIONS,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            Path(f"{path}.sha256").write_text(
                f"{file_sha256(path)}  config.json\n", encoding="utf-8"
            )
            loaded = load_versioned_config(path)
            self.assertEqual(loaded.values["config_version"], "1.2.3")
            self.assertEqual(loaded.file_sha256, file_sha256(path))

            path.write_text(json.dumps({**payload, "config_version": "1.2.4"}), encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "SHA-256 mismatch"):
                load_versioned_config(path)

    def test_module_logs_are_independent_json_rotating_and_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            loggers = configure_module_loggers(
                directory,
                "run-test",
                {"preflight": "DEBUG", "gpu": "OFF"},
                max_bytes=2048,
                backup_count=2,
                console=False,
            )
            loggers["preflight"].info(
                "authorization=top-secret token=abc123 operation=probe",
                extra={
                    "context": {
                        "password": "never-log-me",
                        "step": 7,
                        "tokenizer_sha256": "safe-dataset-hash",
                    }
                },
            )
            loggers["gpu"].critical("must stay disabled")
            for handler in loggers["preflight"].handlers:
                handler.flush()
            path = Path(directory) / "run-test.preflight.jsonl"
            event = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(event["run_id"], "run-test")
            self.assertEqual(event["module"], "cloud.preflight")
            self.assertEqual(event["context"]["password"], "[REDACTED]")
            self.assertEqual(event["context"]["tokenizer_sha256"], "safe-dataset-hash")
            self.assertNotIn("top-secret", path.read_text(encoding="utf-8"))
            self.assertNotIn("abc123", path.read_text(encoding="utf-8"))
            self.assertEqual(
                (Path(directory) / "run-test.gpu.jsonl").read_text(encoding="utf-8"),
                "",
            )
            for logger in loggers.values():
                for handler in logger.handlers:
                    handler.close()
                logger.handlers.clear()
                logger.disabled = False

    def test_atomic_checkpoint_reload_restores_full_state(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(11)
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            inputs = torch.randn(4, 3)
            model(inputs).sum().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            generator = torch.Generator().manual_seed(55)
            scaler = FakeScaler(32.0)
            expected_weight = model.weight.detach().clone()
            payload = build_checkpoint_payload(
                model,
                optimizer,
                step=123,
                best_metric=1.25,
                history=[{"step": 100, "val_loss": 1.25}],
                sampling_generator=generator,
                config_sha256="a" * 64,
                amp_scaler=scaler,
                early_stopping_state={"bad_evaluations": 2},
                extra={"stage": "pretrain"},
            )
            expected_generator_values = torch.rand(4, generator=generator)
            checkpoint = Path(directory) / "latest.pt"
            result = atomic_save_checkpoint(checkpoint, payload)
            self.assertTrue(result.reload_verified)
            self.assertEqual(result.sha256, file_sha256(checkpoint))
            self.assertTrue(Path(f"{checkpoint}.sha256").is_file())

            with torch.no_grad():
                model.weight.zero_()
            _ = torch.rand(20, generator=generator)
            restored_scaler = FakeScaler(1.0)
            state = restore_checkpoint(
                checkpoint,
                model,
                optimizer,
                generator,
                amp_scaler=restored_scaler,
                expected_config_sha256="a" * 64,
            )
            self.assertEqual(state.step, 123)
            self.assertEqual(state.best_metric, 1.25)
            self.assertEqual(state.early_stopping_state["bad_evaluations"], 2)
            self.assertEqual(state.extra["stage"], "pretrain")
            self.assertEqual(restored_scaler.scale, 32.0)
            self.assertTrue(torch.equal(model.weight, expected_weight))
            self.assertTrue(
                torch.equal(torch.rand(4, generator=generator), expected_generator_values)
            )

    def test_checkpoint_hash_detects_post_save_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            generator = torch.Generator().manual_seed(1)
            path = Path(directory) / "latest.pt"
            atomic_save_checkpoint(
                path,
                build_checkpoint_payload(
                    model,
                    optimizer,
                    step=1,
                    best_metric=2.0,
                    history=[],
                    sampling_generator=generator,
                    config_sha256="b" * 64,
                ),
            )
            with path.open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_checkpoint(path)

    def test_signal_hook_creates_emergency_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            generator = torch.Generator().manual_seed(7)
            path = Path(directory) / "emergency.pt"

            def payload_factory():
                return build_checkpoint_payload(
                    model,
                    optimizer,
                    step=9,
                    best_metric=3.0,
                    history=[],
                    sampling_generator=generator,
                    config_sha256="c" * 64,
                )

            hook = EmergencyCheckpointHook(
                path,
                payload_factory,
                exit_after_save=False,
            )
            hook._handle_signal(signal.SIGTERM, None)
            self.assertEqual(hook.received_signal, signal.SIGTERM)
            self.assertEqual(hook.save_result.step, 9)
            self.assertEqual(load_checkpoint(path)["step"], 9)

    def test_nonfinite_guards_stop_before_optimizer_step(self):
        with self.assertRaises(NumericalFailure):
            assert_finite_tensor(torch.tensor(float("nan")), "loss")
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        parameter.grad = torch.tensor(float("inf"))
        with self.assertRaisesRegex(NumericalFailure, "weight"):
            assert_finite_gradients([("weight", parameter)])

    def test_early_stopping_state_and_wall_clock(self):
        stopping = EarlyStopping(patience=2, min_delta=0.01)
        self.assertTrue(stopping.update(2.0, 10).improved)
        self.assertFalse(stopping.update(1.995, 20).should_stop)
        decision = stopping.update(2.1, 30)
        self.assertTrue(decision.should_stop)
        restored = EarlyStopping(patience=1)
        restored.load_state_dict(stopping.state_dict())
        self.assertEqual(restored.best_step, 10)
        self.assertEqual(restored.bad_evaluations, 2)

        current = [100.0]
        limit = WallClockLimit(10.0, started_monotonic=100.0, clock=lambda: current[0])
        current[0] = 106.0
        limit.check(reserve_seconds=3.0)
        with self.assertRaises(WallClockExceeded):
            limit.check(reserve_seconds=4.0)

    def test_done_and_failed_status_redact_secret_details(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = RunStateWriter(directory, "run-1", "d" * 64)
            done_path = writer.mark_done({"step": 12})
            failed_path = writer.mark_failed(
                RuntimeError("authorization=hidden-value"),
                {"api_key": "never-store"},
            )
            self.assertEqual(json.loads(done_path.read_text())["status"], "DONE")
            failed_text = failed_path.read_text(encoding="utf-8")
            self.assertNotIn("hidden-value", failed_text)
            self.assertNotIn("never-store", failed_text)
            self.assertEqual(json.loads(failed_text)["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
