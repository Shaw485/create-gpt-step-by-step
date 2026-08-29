"""Train the M021 Canary with assistant-only SFT plus train-only LM replay.

Every optimizer update minimizes exactly::

    total_loss = sft_assistant_only_loss + replay_weight * pretrain_next_token_loss

The SFT optimizer can see only the 64 Canary training records.  The 16 unseen
question wordings form a development/selection set: every scheduled evaluation
computes their teacher loss and that metric may select a checkpoint, but these
records never enter an optimizer batch.  The replay optimizer can see only
random windows from the frozen formal pretraining ``train_tokens.pt``.  SFT
public/sealed data and pretraining validation/test tensors are never loaded for
optimization.

Runtime diagnostics are independently controlled rotating JSONL streams for
``data``, ``sft_training``, ``replay_training``, ``validation``, ``checkpoint``
and ``orchestrator``.  Logs contain hashes, counts and numeric metrics only;
they never contain questions, answers, source text, token IDs, credentials or
absolute failure paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from prepare_bpe_sft_v7 import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SPECIAL_TOKEN_IDS,
    EXPECTED_TOKENIZER_SHA256,
    REQUIRED_BASE_CHECKPOINT,
)
from train_pretrain_v4 import get_batch, load_config, load_tensor
from train_sft_v5 import evaluate_all_records
from train_sft_v7 import (
    BASE_CONFIG_CANONICAL_SHA256,
    BASE_PARAMETER_COUNT,
    BASE_STEP,
    validate_base_checkpoint_payload,
    validate_frozen_config,
)
from train_sft_v7_1_canary import (
    DeterministicShuffledEpochSampler,
    EXPECTED_EVAL_COUNT,
    EXPECTED_TRAIN_COUNT,
    TRAINING_SIGNATURE_SCHEMA,
    build_optimizer,
    learning_rate_for_update,
    load_canary_tensor_payload,
    validate_canary_tensor_payload,
)
from train_sft_v4 import build_model, select_device, supervised_loss
from training_runtime import (
    EmergencyCheckpointHook,
    RunStateWriter,
    assert_finite_gradients,
    assert_finite_tensor,
    atomic_save_checkpoint,
    atomic_write_json,
    atomic_write_text,
    build_checkpoint_payload,
    canonical_json_sha256,
    close_module_loggers,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
    load_checkpoint,
    resolve_module_log_levels,
    restore_checkpoint,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAINING_CONFIG = Path("configs/sft_v7_1_canary_replay_train.json")
DEFAULT_BASE_CONFIG = Path("configs/formal_pretrain_14m_bpe3000.json")
DEFAULT_CANARY_DATA = Path("data/sft/v7_1_canary/train_eval_tensors.pt")
DEFAULT_INIT_CHECKPOINT = Path(str(REQUIRED_BASE_CHECKPOINT["path"]))
DEFAULT_REPLAY_TENSOR = Path("data/scaling_a/bpe_3000/train_tokens.pt")
DEFAULT_TOKEN_MANIFEST = Path("data/scaling_a/bpe_3000/token_manifest.json")
DEFAULT_RUN_DIR = Path("runs/sft_v7_1_canary_replay")
DEFAULT_REPORT = Path(
    "reports/milestones/021_sft_v7_1_canary/canary_replay_train_report.json"
)

CONFIG_SCHEMA = "sft-v7.1-canary-replay-training-config/v1"
REPORT_SCHEMA = "sft-v7.1-canary-replay-train-report/v1"
REPLAY_SAMPLER_SCHEMA = "deterministic_random_train_windows/v1"
EXPECTED_STAGE = "sft_v7_1_canary"  # retained for existing Canary evaluators
TRAINING_VARIANT = "assistant_only_sft_plus_train_only_pretrain_replay/v1"
EXPECTED_REPLAY_TENSOR_SHA256 = (
    "3e152945535e1711471b343e1acafabe7a9f423d5f553a11d532306d1986a712"
)
EXPECTED_REPLAY_TOKEN_COUNT = 3_223_207
LOG_MODULES = (
    "data",
    "sft_training",
    "replay_training",
    "validation",
    "checkpoint",
    "orchestrator",
)
_HEX_SHA = re.compile(r"[0-9a-f]{64}")


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return f"artifact://canary-replay/{resolved.name}"


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("required replay metadata cannot be parsed") from error
    if not isinstance(value, dict):
        raise ValueError("required replay metadata root must be an object")
    return value


def load_training_config(path: Path) -> dict[str, Any]:
    """Load and validate the reviewed replay experiment contract."""

    config = _load_json_object(path)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported Canary replay training config schema")
    if config.get("base_model_config") != str(DEFAULT_BASE_CONFIG):
        raise ValueError("Canary replay base model binding changed")
    if config.get("base_checkpoint") != str(DEFAULT_INIT_CHECKPOINT):
        raise ValueError("Canary replay base checkpoint binding changed")
    canary = config.get("canary")
    replay = config.get("replay")
    training = config.get("training")
    logging_config = config.get("logging")
    if not all(
        isinstance(value, Mapping)
        for value in (canary, replay, training, logging_config)
    ):
        raise ValueError("Canary replay config sections are incomplete")
    if (
        canary.get("tensor_path") != str(DEFAULT_CANARY_DATA)
        or int(canary.get("train_records", -1)) != EXPECTED_TRAIN_COUNT
        or int(canary.get("holdout_eval_records", -1)) != EXPECTED_EVAL_COUNT
        or int(canary.get("development_unseen_wording_records", -1))
        != EXPECTED_EVAL_COUNT
        or canary.get("assistant_only_loss") is not True
        or canary.get("holdout_used_for_optimization") is not False
        or canary.get("development_used_for_teacher_loss") is not True
        or canary.get("development_used_for_checkpoint_selection") is not True
    ):
        raise ValueError("Canary optimizer data-role contract changed")
    if (
        replay.get("split") != "train"
        or replay.get("train_tensor_path") != str(DEFAULT_REPLAY_TENSOR)
        or replay.get("token_manifest_path") != str(DEFAULT_TOKEN_MANIFEST)
        or replay.get("train_tensor_sha256") != EXPECTED_REPLAY_TENSOR_SHA256
        or replay.get("token_manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or replay.get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA256
        or replay.get("validation_used_for_optimization") is not False
        or replay.get("test_used_for_optimization") is not False
    ):
        raise ValueError("pretraining replay must be bound only to frozen train data")
    positive_fields = (
        "target_steps",
        "sft_batch_size",
        "replay_batch_size",
        "replay_block_size",
        "replay_weight",
        "learning_rate",
        "minimum_learning_rate",
        "warmup_steps",
        "gradient_clip",
        "eval_interval",
        "checkpoint_interval",
        "eval_batch_size",
        "log_interval",
    )
    for name in positive_fields:
        if float(training.get(name, 0)) <= 0:
            raise ValueError(f"Canary replay training field must be positive: {name}")
    if int(training["warmup_steps"]) >= int(training["target_steps"]):
        raise ValueError("Canary replay warmup must finish before target step")
    if float(training["minimum_learning_rate"]) > float(training["learning_rate"]):
        raise ValueError("Canary replay minimum learning rate exceeds peak")
    if float(training.get("weight_decay", -1)) < 0:
        raise ValueError("Canary replay weight decay cannot be negative")
    betas = training.get("betas")
    if not isinstance(betas, list) or len(betas) != 2 or not all(
        0.0 < float(value) < 1.0 for value in betas
    ):
        raise ValueError("Canary replay AdamW betas are invalid")
    if training.get("sft_sampler") != "deterministic_shuffled_epoch/v1":
        raise ValueError("Canary replay SFT sampler changed")
    if training.get("replay_sampler") != REPLAY_SAMPLER_SCHEMA:
        raise ValueError("Canary replay train-window sampler changed")
    return config


def validate_replay_manifest_contract(
    manifest: Mapping[str, Any],
    replay_config: Mapping[str, Any],
    *,
    manifest_sha256: str,
    train_tensor_sha256: str,
) -> dict[str, Any]:
    """Validate only the frozen manifest's train entry for optimizer replay."""

    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError("formal replay token manifest SHA changed")
    if train_tensor_sha256 != EXPECTED_REPLAY_TENSOR_SHA256:
        raise ValueError("formal replay train tensor SHA changed")
    if (
        manifest.get("schema_version") != "bpe-v4/v1"
        or manifest.get("status") != "ready"
        or manifest.get("train_only_merge_learning") is not True
        or manifest.get("validation_test_characters_used_for_merge_counts") is not False
        or manifest.get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA256
        or manifest.get("special_tokens") != EXPECTED_SPECIAL_TOKEN_IDS
        or int(manifest.get("vocab_size", -1)) != 7465
    ):
        raise ValueError("formal replay token manifest contract changed")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("formal replay token manifest lacks split metadata")
    train = splits.get("train")
    if not isinstance(train, Mapping):
        raise ValueError("formal replay token manifest lacks train metadata")
    if (
        replay_config.get("split") != "train"
        or train.get("tensor_path") != replay_config.get("train_tensor_path")
        or train.get("tensor_sha256") != train_tensor_sha256
        or int(train.get("tokens", -1)) != EXPECTED_REPLAY_TOKEN_COUNT
    ):
        raise ValueError("formal replay train split binding changed")
    return {
        "optimizer_split": "train",
        "train_tensor_sha256": train_tensor_sha256,
        "token_manifest_sha256": manifest_sha256,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "train_token_count": int(train["tokens"]),
        "validation_batches_consumed": 0,
        "test_batches_consumed": 0,
    }


def load_and_validate_replay_train(
    replay_config: Mapping[str, Any],
    *,
    root: Path = REPOSITORY_ROOT,
    digest: Callable[[Path], str] | None = None,
    tensor_loader: Callable[[Path], torch.Tensor] | None = None,
    json_loader: Callable[[Path], dict[str, Any]] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load exactly one optimizer replay tensor: the formal train split."""

    digest_fn = digest or file_sha256
    tensor_fn = tensor_loader or load_tensor
    json_fn = json_loader or _load_json_object
    manifest_path = root / Path(str(replay_config["token_manifest_path"]))
    train_path = root / Path(str(replay_config["train_tensor_path"]))
    expected_manifest_path = root / DEFAULT_TOKEN_MANIFEST
    expected_train_path = root / DEFAULT_REPLAY_TENSOR
    if (
        manifest_path.resolve() != expected_manifest_path.resolve()
        or train_path.resolve() != expected_train_path.resolve()
    ):
        raise ValueError("replay optimizer source is outside the frozen train contract")
    manifest_sha = digest_fn(manifest_path)
    train_sha = digest_fn(train_path)
    manifest = json_fn(manifest_path)
    summary = validate_replay_manifest_contract(
        manifest,
        replay_config,
        manifest_sha256=manifest_sha,
        train_tensor_sha256=train_sha,
    )
    tensor = tensor_fn(train_path)
    if len(tensor) != summary["train_token_count"]:
        raise ValueError("replay train tensor length differs from frozen manifest")
    return tensor, summary


def combine_joint_losses(
    sft_assistant_only_loss: torch.Tensor,
    pretrain_next_token_loss: torch.Tensor,
    replay_weight: float,
) -> torch.Tensor:
    """Return the sole scalar differentiated by the joint optimizer update."""

    if sft_assistant_only_loss.ndim != 0 or pretrain_next_token_loss.ndim != 0:
        raise ValueError("joint training losses must be scalar tensors")
    if replay_weight <= 0:
        raise ValueError("replay weight must be positive")
    return sft_assistant_only_loss + float(replay_weight) * pretrain_next_token_loss


class DeterministicReplayTrainSampler:
    """Sample reproducible next-token windows from one frozen train tensor."""

    def __init__(
        self,
        train_data: torch.Tensor,
        generator: torch.Generator,
        *,
        train_tensor_sha256: str,
        batch_size: int,
        block_size: int,
    ) -> None:
        if (
            not isinstance(train_data, torch.Tensor)
            or train_data.dtype != torch.long
            or train_data.ndim != 1
        ):
            raise ValueError("replay source must be one-dimensional int64 train data")
        if len(train_data) <= block_size or batch_size <= 0 or block_size <= 0:
            raise ValueError("replay batch/block settings are incompatible with train data")
        if not _HEX_SHA.fullmatch(train_tensor_sha256):
            raise ValueError("replay train tensor identity is invalid")
        self.train_data = train_data
        self.generator = generator
        self.train_tensor_sha256 = train_tensor_sha256
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)
        self.batches_drawn = 0
        self.windows_drawn = 0
        self.target_tokens_consumed = 0

    def sample_batch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = get_batch(
            self.train_data,
            self.batch_size,
            self.block_size,
            self.generator,
            device,
        )
        self.batches_drawn += 1
        self.windows_drawn += self.batch_size
        self.target_tokens_consumed += self.batch_size * self.block_size
        return inputs, targets

    def state_dict(self) -> dict[str, Any]:
        return {
            "strategy": REPLAY_SAMPLER_SCHEMA,
            "optimizer_split": "train",
            "train_tensor_sha256": self.train_tensor_sha256,
            "batch_size": self.batch_size,
            "block_size": self.block_size,
            "batches_drawn": self.batches_drawn,
            "windows_drawn": self.windows_drawn,
            "target_tokens_consumed": self.target_tokens_consumed,
            "generator_state": self.generator.get_state(),
            "validation_batches_consumed": 0,
            "test_batches_consumed": 0,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "strategy": REPLAY_SAMPLER_SCHEMA,
            "optimizer_split": "train",
            "train_tensor_sha256": self.train_tensor_sha256,
            "batch_size": self.batch_size,
            "block_size": self.block_size,
            "validation_batches_consumed": 0,
            "test_batches_consumed": 0,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"replay sampler state mismatch: {key}")
        batches = int(state.get("batches_drawn", -1))
        windows = int(state.get("windows_drawn", -1))
        tokens = int(state.get("target_tokens_consumed", -1))
        if (
            batches < 0
            or windows != batches * self.batch_size
            or tokens != windows * self.block_size
        ):
            raise ValueError("replay sampler consumption counters are invalid")
        generator_state = state.get("generator_state")
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("replay sampler RNG state is missing")
        self.batches_drawn = batches
        self.windows_drawn = windows
        self.target_tokens_consumed = tokens
        self.generator.set_state(generator_state.cpu())

    def consumption_summary(self) -> dict[str, Any]:
        return {
            "strategy": REPLAY_SAMPLER_SCHEMA,
            "optimizer_split": "train",
            "batches": self.batches_drawn,
            "windows": self.windows_drawn,
            "target_tokens": self.target_tokens_consumed,
            "block_size": self.block_size,
            "batch_size": self.batch_size,
            "validation_batches": 0,
            "test_batches": 0,
        }


def schedule_contract(settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "strategy": "linear_warmup_cosine_decay/v1",
        "target_steps": int(settings["target_steps"]),
        "warmup_steps": int(settings["warmup_steps"]),
        "peak_learning_rate": float(settings["learning_rate"]),
        "minimum_learning_rate": float(settings["minimum_learning_rate"]),
        "sampler": "deterministic_shuffled_epoch/v1",
        "replay_sampler": REPLAY_SAMPLER_SCHEMA,
        "joint_loss": TRAINING_VARIANT,
    }


def optimizer_consumption(
    sft_sampler: DeterministicShuffledEpochSampler,
    replay_sampler: DeterministicReplayTrainSampler,
    *,
    development_evaluations: int = 0,
) -> dict[str, Any]:
    if development_evaluations < 0:
        raise ValueError("development evaluation count cannot be negative")
    sft_draws = int(sft_sampler.coverage_summary()["draws"])
    return {
        "canary_train_records_drawn": sft_draws,
        "canary_holdout_records_drawn": 0,
        "canary_development_optimizer_records_drawn": 0,
        "canary_development_teacher_evaluations": development_evaluations,
        "canary_development_teacher_forward_records": (
            development_evaluations * EXPECTED_EVAL_COUNT
        ),
        "canary_development_checkpoint_selection_events": development_evaluations,
        "canary_development_records_per_teacher_evaluation": EXPECTED_EVAL_COUNT,
        "canary_development_records_per_checkpoint_selection": EXPECTED_EVAL_COUNT,
        "sft_public_records_drawn": 0,
        "sft_sealed_records_drawn": 0,
        "pretrain_train_batches_drawn": replay_sampler.batches_drawn,
        "pretrain_train_windows_drawn": replay_sampler.windows_drawn,
        "pretrain_train_target_tokens_consumed": replay_sampler.target_tokens_consumed,
        "pretrain_validation_batches_drawn": 0,
        "pretrain_test_batches_drawn": 0,
    }


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    best_eval_loss: float,
    history: Sequence[Mapping[str, Any]],
    sft_generator: torch.Generator,
    signature_sha256: str,
    provenance: Mapping[str, Any],
    sft_sampler: DeterministicShuffledEpochSampler,
    replay_sampler: DeterministicReplayTrainSampler,
    schedule: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a Canary-compatible checkpoint with both sampler/RNG states."""

    sft_state = sft_sampler.state_dict()
    replay_state = replay_sampler.state_dict()
    return build_checkpoint_payload(
        model,
        optimizer,
        step=step,
        best_metric=best_eval_loss,
        history=history,
        sampling_generator=sft_generator,
        config_sha256=signature_sha256,
        extra={
            **dict(provenance),
            # Legacy name retained so both existing Canary evaluators accept it.
            "sampler_state": sft_state,
            "sft_sampler_state": sft_state,
            "replay_sampler_state": replay_state,
            "learning_rate_schedule": dict(schedule),
            "optimizer_consumption": optimizer_consumption(
                sft_sampler,
                replay_sampler,
                development_evaluations=len(history),
            ),
            "selection_policy": (
                "teacher_loss_is_diagnostic_only; generation_and_retention_gates_decide"
            ),
            "mps_resume_reproducibility": (
                "CPU/Python/SFT/replay RNG and both samplers restored; exact MPS dropout "
                "replay is not claimed"
            ),
        },
    )


def _resolve_settings(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> dict[str, Any]:
    settings = dict(config["training"])
    for name in (
        "target_steps",
        "sft_batch_size",
        "replay_batch_size",
        "replay_block_size",
        "replay_weight",
        "learning_rate",
        "minimum_learning_rate",
        "warmup_steps",
        "weight_decay",
        "gradient_clip",
        "eval_interval",
        "checkpoint_interval",
        "eval_batch_size",
        "log_interval",
        "seed",
    ):
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    for name in (
        "target_steps",
        "sft_batch_size",
        "replay_batch_size",
        "replay_block_size",
        "replay_weight",
        "learning_rate",
        "minimum_learning_rate",
        "warmup_steps",
        "gradient_clip",
        "eval_interval",
        "checkpoint_interval",
        "eval_batch_size",
        "log_interval",
    ):
        if float(settings[name]) <= 0:
            raise ValueError(f"Canary replay setting must be positive: {name}")
    if int(settings["warmup_steps"]) >= int(settings["target_steps"]):
        raise ValueError("Canary replay warmup must finish before target step")
    if float(settings["minimum_learning_rate"]) > float(settings["learning_rate"]):
        raise ValueError("Canary replay learning-rate bounds are invalid")
    if float(settings["weight_decay"]) < 0:
        raise ValueError("Canary replay weight decay cannot be negative")
    return settings


def _add_log_arguments(parser: argparse.ArgumentParser) -> None:
    for module in LOG_MODULES:
        parser.add_argument(
            f"--{module.replace('_', '-')}-log-level",
            choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "OFF"),
            default=None,
            help=f"independently set the {module} rotating JSONL log level",
        )
    parser.add_argument("--no-console-log", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--canary-data", type=Path, default=DEFAULT_CANARY_DATA)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--init-checkpoint", type=Path, default=None)
    source.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--target-steps", type=int, default=None)
    parser.add_argument("--sft-batch-size", type=int, default=None)
    parser.add_argument("--replay-batch-size", type=int, default=None)
    parser.add_argument("--replay-block-size", type=int, default=None)
    parser.add_argument("--replay-weight", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--minimum-learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--gradient-clip", type=float, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    _add_log_arguments(parser)
    return parser.parse_args(argv)


def _log_levels(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> dict[str, str]:
    configured = dict(config["logging"].get("module_levels", {}))
    levels = resolve_module_log_levels(
        {module: str(configured.get(module, "INFO")) for module in LOG_MODULES},
        env_prefix="GPT_CANARY_REPLAY_LOG_LEVEL",
    )
    for module in LOG_MODULES:
        override = getattr(args, f"{module}_log_level")
        if override is not None:
            levels[module] = override
    return levels


def _set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer, learning_rate: float
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def log_run_started(
    loggers: Mapping[str, Any], *, start_step: int, target_step: int, run_signature: str
) -> None:
    loggers["orchestrator"].info(
        "Canary replay training started",
        extra={
            "context": {
                "start_step": start_step,
                "target_step": target_step,
                "run_signature": run_signature,
                "optimizer_replay_split": "train",
            }
        },
    )


def log_run_failed(
    loggers: Mapping[str, Any], error: BaseException, *, step: int
) -> None:
    """Emit an actionable but path/body-safe failure event."""

    loggers["orchestrator"].error(
        "Canary replay training failed; inspect redacted module logs",
        extra={
            "context": {
                "error_type": type(error).__name__,
                "step": step,
                "record_bodies_logged": False,
                "token_ids_logged": False,
            }
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    training_config = load_training_config(args.training_config)
    settings = _resolve_settings(args, training_config)
    base_config = load_config(args.base_config)
    validate_frozen_config(base_config)
    if int(settings["replay_block_size"]) > int(base_config["model"]["block_size"]):
        raise ValueError("replay block size exceeds frozen model context")

    run_id = generate_run_id("sft-v7-1-canary-replay")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    loggers = configure_module_loggers(
        args.run_dir / "logs",
        run_id,
        _log_levels(args, training_config),
        max_bytes=int(training_config["logging"]["max_bytes"]),
        backup_count=int(training_config["logging"]["backup_count"]),
        console=(
            bool(training_config["logging"]["console"])
            and not args.no_console_log
        ),
    )
    state_writer: RunStateWriter | None = None
    emergency_hook: EmergencyCheckpointHook | None = None
    current_step = 0
    best_eval_loss = float("inf")
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        device = select_device(args.device)
        seed = int(settings["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        sft_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        replay_generator = torch.Generator(device="cpu").manual_seed(seed + 2)

        canary_payload = load_canary_tensor_payload(args.canary_data)
        model = build_model(base_config, int(canary_payload["vocab_size"])).to(device)
        if model.parameter_count() != BASE_PARAMETER_COUNT:
            raise ValueError("Canary replay model is not the frozen 14.9M architecture")
        payload_summary = validate_canary_tensor_payload(
            canary_payload, int(model.config.block_size)
        )
        canary_tensor_sha = file_sha256(args.canary_data)
        canary_manifest_path = REPOSITORY_ROOT / Path(
            str(canary_payload["canary_dataset_manifest_path"])
        )
        if (
            not canary_manifest_path.is_file()
            or file_sha256(canary_manifest_path)
            != str(canary_payload["canary_dataset_manifest_sha256"])
        ):
            raise ValueError("Canary tensor no longer matches its dataset manifest")

        replay_data, replay_summary = load_and_validate_replay_train(
            training_config["replay"]
        )
        train_records = list(canary_payload["train_records"])
        development_records = list(canary_payload["eval_records"])
        sft_sampler = DeterministicShuffledEpochSampler(
            train_records, sft_generator
        )
        replay_sampler = DeterministicReplayTrainSampler(
            replay_data,
            replay_generator,
            train_tensor_sha256=replay_summary["train_tensor_sha256"],
            batch_size=int(settings["replay_batch_size"]),
            block_size=int(settings["replay_block_size"]),
        )
        optimizer = build_optimizer(
            model,
            learning_rate=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
            betas=tuple(float(value) for value in settings["betas"]),
        )
        schedule = schedule_contract(settings)
        provenance = {
            "stage": EXPECTED_STAGE,
            "training_variant": TRAINING_VARIANT,
            "base_checkpoint_path": str(DEFAULT_INIT_CHECKPOINT),
            "base_checkpoint_sha256": REQUIRED_BASE_CHECKPOINT["sha256"],
            "base_checkpoint_step": BASE_STEP,
            "base_config_canonical_sha256": BASE_CONFIG_CANONICAL_SHA256,
            "base_token_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
            "sft_tensor_path": _portable(args.canary_data),
            "sft_tensor_sha256": canary_tensor_sha,
            "canary_tensor_path": _portable(args.canary_data),
            "canary_tensor_sha256": canary_tensor_sha,
            "canary_dataset_manifest_sha256": canary_payload[
                "canary_dataset_manifest_sha256"
            ],
            "canary_dataset_identity_sha256": canary_payload[
                "canary_dataset_identity_sha256"
            ],
            "payload_summary": payload_summary,
            "optimization_train_records": EXPECTED_TRAIN_COUNT,
            "optimization_holdout_records": 0,
            # Legacy zero-consumption field means optimizer consumption only.
            "holdout_records_consumed": 0,
            "teacher_loss_holdout_records": EXPECTED_EVAL_COUNT,
            "development_unseen_wording_records": EXPECTED_EVAL_COUNT,
            "development_optimizer_records": 0,
            "development_records_consumed_for_teacher_loss": EXPECTED_EVAL_COUNT,
            "development_records_used_for_checkpoint_selection": EXPECTED_EVAL_COUNT,
            "public_records_consumed": 0,
            "sealed_records_consumed": 0,
            "replay_optimizer_split": "train",
            "replay_train_tensor_path": str(DEFAULT_REPLAY_TENSOR),
            "replay_train_tensor_sha256": replay_summary["train_tensor_sha256"],
            "replay_token_manifest_sha256": replay_summary[
                "token_manifest_sha256"
            ],
            "replay_weight": float(settings["replay_weight"]),
            "replay_batch_size": int(settings["replay_batch_size"]),
            "replay_block_size": int(settings["replay_block_size"]),
            "pretrain_validation_batches_consumed": 0,
            "pretrain_test_batches_consumed": 0,
        }
        signature_training = {
            "target_steps": int(settings["target_steps"]),
            # Existing evaluators require the legacy Canary sampler key.
            "batch_size": int(settings["sft_batch_size"]),
            "sampler": "deterministic_shuffled_epoch/v1",
            "sft_batch_size": int(settings["sft_batch_size"]),
            "replay_batch_size": int(settings["replay_batch_size"]),
            "replay_block_size": int(settings["replay_block_size"]),
            "replay_weight": float(settings["replay_weight"]),
            "learning_rate": float(settings["learning_rate"]),
            "minimum_learning_rate": float(settings["minimum_learning_rate"]),
            "warmup_steps": int(settings["warmup_steps"]),
            "weight_decay": float(settings["weight_decay"]),
            "betas": [float(value) for value in settings["betas"]],
            "gradient_clip": float(settings["gradient_clip"]),
            "seed": seed,
            "replay_sampler": REPLAY_SAMPLER_SCHEMA,
            "joint_loss": TRAINING_VARIANT,
        }
        signature = {
            "schema_version": TRAINING_SIGNATURE_SCHEMA,
            "model": base_config["model"],
            "provenance": provenance,
            "training": signature_training,
            "schedule": schedule,
        }
        signature_sha = canonical_json_sha256(signature)
        atomic_write_json(
            args.run_dir / "effective_config.json",
            {
                **signature,
                "signature_sha256": signature_sha,
                "eval_interval": int(settings["eval_interval"]),
                "checkpoint_interval": int(settings["checkpoint_interval"]),
                "eval_batch_size": int(settings["eval_batch_size"]),
                "log_interval": int(settings["log_interval"]),
                "device": str(device),
                "runtime_logging": {
                    "modules": list(LOG_MODULES),
                    "directory": _portable(args.run_dir / "logs"),
                    "record_bodies_logged": False,
                    "token_ids_logged": False,
                },
            },
        )
        state_writer = RunStateWriter(args.run_dir, run_id, signature_sha)

        start_step = 0
        source_checkpoint: Path
        initialization_policy = "fresh_from_frozen_pretrain_step5750"
        if args.resume is not None:
            resumed = restore_checkpoint(
                args.resume,
                model,
                optimizer,
                sft_generator,
                expected_config_sha256=signature_sha,
                map_location=device,
                restore_cuda_rng=False,
            )
            for key, expected in provenance.items():
                if resumed.extra.get(key) != expected:
                    raise ValueError(f"Canary replay resume provenance mismatch: {key}")
            sft_state = resumed.extra.get("sft_sampler_state")
            replay_state = resumed.extra.get("replay_sampler_state")
            if not isinstance(sft_state, Mapping) or not isinstance(
                replay_state, Mapping
            ):
                raise ValueError("Canary replay checkpoint lacks both sampler states")
            sft_sampler.load_state_dict(sft_state)
            replay_sampler.load_state_dict(replay_state)
            if resumed.extra.get("learning_rate_schedule") != schedule:
                raise ValueError("Canary replay learning-rate schedule changed")
            start_step = resumed.step
            current_step = start_step
            best_eval_loss = resumed.best_metric
            history = resumed.history
            source_checkpoint = args.resume
            initialization_policy = "full_resume_model_optimizer_rng_two_samplers"
        else:
            source_checkpoint = args.init_checkpoint or DEFAULT_INIT_CHECKPOINT
            if file_sha256(source_checkpoint) != REQUIRED_BASE_CHECKPOINT["sha256"]:
                raise ValueError("Canary replay fresh source is not frozen Step 5750")
            initial = load_checkpoint(source_checkpoint, map_location=device)
            validate_base_checkpoint_payload(initial)
            model.load_state_dict(initial["model_state_dict"], strict=True)

        target_steps = int(settings["target_steps"])
        if target_steps <= start_step:
            raise ValueError("Canary replay target step must exceed resumed step")
        pad_id = int(canary_payload["special_token_ids"]["<PAD>"])
        loggers["data"].info(
            "Canary and train-only replay artifacts validated",
            extra={
                "context": {
                    "canary_tensor_sha256": canary_tensor_sha,
                    "canary_train_records": EXPECTED_TRAIN_COUNT,
                    "development_optimizer_records": 0,
                    "development_teacher_eval_records": EXPECTED_EVAL_COUNT,
                    "development_checkpoint_selection_records": EXPECTED_EVAL_COUNT,
                    "replay_train_tensor_sha256": replay_summary[
                        "train_tensor_sha256"
                    ],
                    "replay_train_token_count": replay_summary[
                        "train_token_count"
                    ],
                    "replay_validation_batches": 0,
                    "replay_test_batches": 0,
                }
            },
        )
        log_run_started(
            loggers,
            start_step=start_step,
            target_step=target_steps,
            run_signature=signature_sha,
        )

        def make_current_checkpoint() -> dict[str, Any]:
            return checkpoint_payload(
                model,
                optimizer,
                step=current_step,
                best_eval_loss=best_eval_loss,
                history=history,
                sft_generator=sft_generator,
                signature_sha256=signature_sha,
                provenance=provenance,
                sft_sampler=sft_sampler,
                replay_sampler=replay_sampler,
                schedule=schedule,
            )

        emergency_hook = EmergencyCheckpointHook(
            args.run_dir / "emergency.pt",
            make_current_checkpoint,
            logger=loggers["checkpoint"],
        ).install()

        last_batch_metrics: dict[str, float | int] = {
            "sft_assistant_only_loss": 0.0,
            "pretrain_next_token_loss": 0.0,
            "joint_total_loss": 0.0,
            "learning_rate": 0.0,
        }

        def evaluate_full_sets(step: int) -> None:
            nonlocal best_eval_loss
            train_loss = evaluate_all_records(
                model,
                train_records,
                pad_id,
                int(settings["eval_batch_size"]),
                device,
            )
            development_loss = evaluate_all_records(
                model,
                development_records,
                pad_id,
                int(settings["eval_batch_size"]),
                device,
            )
            row = {
                "step": step,
                "train_teacher_loss": train_loss,
                "development_teacher_loss": development_loss,
                # Compatibility alias for existing report readers.
                "holdout_teacher_loss": development_loss,
                **last_batch_metrics,
                "sft_sampler_coverage": sft_sampler.coverage_summary(),
                "replay_consumption": replay_sampler.consumption_summary(),
            }
            history.append(row)
            loggers["validation"].info(
                "Canary full train/development teacher losses evaluated",
                extra={
                    "context": {
                        "step": step,
                        "train_teacher_loss": train_loss,
                        "development_teacher_loss": development_loss,
                        "development_forward_records": EXPECTED_EVAL_COUNT,
                        "development_optimizer_records": 0,
                        "checkpoint_selection_records": EXPECTED_EVAL_COUNT,
                    }
                },
            )
            if development_loss < best_eval_loss:
                best_eval_loss = development_loss
                result = atomic_save_checkpoint(
                    args.run_dir / "best_teacher_loss.pt", make_current_checkpoint()
                )
                loggers["checkpoint"].info(
                    "Provisional best teacher-loss replay checkpoint saved",
                    extra={
                        "context": {
                            "step": step,
                            "development_teacher_loss": development_loss,
                            "checkpoint_sha256": result.sha256,
                        }
                    },
                )

        def save_milestone(step: int) -> None:
            result = atomic_save_checkpoint(
                args.run_dir / "checkpoints" / f"step_{step:05d}.pt",
                make_current_checkpoint(),
            )
            loggers["checkpoint"].info(
                "Canary replay milestone checkpoint saved",
                extra={"context": {"step": step, "checkpoint_sha256": result.sha256}},
            )

        if not history:
            evaluate_full_sets(start_step)
        for completed_steps in range(start_step, target_steps):
            learning_rate = learning_rate_for_update(
                completed_steps,
                target_steps=target_steps,
                warmup_steps=int(settings["warmup_steps"]),
                peak_learning_rate=float(settings["learning_rate"]),
                minimum_learning_rate=float(settings["minimum_learning_rate"]),
            )
            _set_optimizer_learning_rate(optimizer, learning_rate)
            sft_inputs, sft_labels, _ = sft_sampler.sample_batch(
                int(settings["sft_batch_size"]), pad_id
            )
            replay_inputs, replay_targets = replay_sampler.sample_batch(device)
            optimizer.zero_grad(set_to_none=True)
            sft_loss = supervised_loss(
                model, sft_inputs.to(device), sft_labels.to(device)
            )
            _, replay_loss = model(replay_inputs, replay_targets)
            if replay_loss is None:
                raise RuntimeError("model did not return pretraining replay loss")
            total_loss = combine_joint_losses(
                sft_loss, replay_loss, float(settings["replay_weight"])
            )
            assert_finite_tensor(sft_loss, "Canary assistant-only SFT loss")
            assert_finite_tensor(replay_loss, "formal train-only replay loss")
            assert_finite_tensor(total_loss, "Canary plus replay joint loss")
            total_loss.backward()
            assert_finite_gradients(model.named_parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(settings["gradient_clip"])
            )
            assert_finite_tensor(grad_norm, "Canary replay gradient norm")
            optimizer.step()
            current_step = completed_steps + 1
            last_batch_metrics = {
                "sft_assistant_only_loss": float(sft_loss.detach().cpu()),
                "pretrain_next_token_loss": float(replay_loss.detach().cpu()),
                "joint_total_loss": float(total_loss.detach().cpu()),
                "learning_rate": learning_rate,
            }
            if current_step % int(settings["log_interval"]) == 0:
                loggers["sft_training"].info(
                    "Canary assistant-only loss update complete",
                    extra={
                        "context": {
                            "step": current_step,
                            "sft_assistant_only_loss": last_batch_metrics[
                                "sft_assistant_only_loss"
                            ],
                            "supervised_token_count": int(
                                (sft_labels != -100).sum()
                            ),
                            "learning_rate": learning_rate,
                        }
                    },
                )
                loggers["replay_training"].info(
                    "Formal train-only next-token replay update complete",
                    extra={
                        "context": {
                            "step": current_step,
                            "pretrain_next_token_loss": last_batch_metrics[
                                "pretrain_next_token_loss"
                            ],
                            "replay_weight": float(settings["replay_weight"]),
                            "train_target_tokens_consumed": replay_sampler.target_tokens_consumed,
                            "validation_batches_consumed": 0,
                            "test_batches_consumed": 0,
                        }
                    },
                )
                loggers["orchestrator"].info(
                    "Joint optimizer update complete",
                    extra={
                        "context": {
                            "step": current_step,
                            "joint_total_loss": last_batch_metrics[
                                "joint_total_loss"
                            ],
                            "gradient_norm": float(grad_norm.detach().cpu()),
                        }
                    },
                )
            if (
                current_step % int(settings["eval_interval"]) == 0
                or current_step == target_steps
            ):
                evaluate_full_sets(current_step)
            if current_step % int(settings["checkpoint_interval"]) == 0:
                save_milestone(current_step)
            elif current_step == target_steps:
                save_milestone(current_step)

        latest = atomic_save_checkpoint(
            args.run_dir / "latest.pt", make_current_checkpoint()
        )
        loss_csv = args.report.with_name(args.report.stem + "_loss.csv")
        atomic_write_text(
            loss_csv,
            (
                "step,train_teacher_loss,development_teacher_loss,sft_loss,replay_loss,"
                "joint_loss,learning_rate,replay_train_tokens\n"
                + "".join(
                    f"{row['step']},{row['train_teacher_loss']},"
                    f"{row['development_teacher_loss']},"
                    f"{row['sft_assistant_only_loss']},"
                    f"{row['pretrain_next_token_loss']},"
                    f"{row['joint_total_loss']},{row['learning_rate']},"
                    f"{row['replay_consumption']['target_tokens']}\n"
                    for row in history
                )
            ),
        )
        consumption = optimizer_consumption(
            sft_sampler,
            replay_sampler,
            development_evaluations=len(history),
        )
        report = {
            "report_schema_version": REPORT_SCHEMA,
            "status": "training_complete_generation_and_retention_gates_pending",
            "run_id": run_id,
            "parameter_count": model.parameter_count(),
            "device": str(device),
            "initialization_policy": initialization_policy,
            "source_checkpoint": _portable(source_checkpoint),
            "source_checkpoint_sha256": file_sha256(source_checkpoint),
            "base_checkpoint_step": BASE_STEP,
            "canary_tensor_sha256": canary_tensor_sha,
            "canary_dataset_manifest_sha256": canary_payload[
                "canary_dataset_manifest_sha256"
            ],
            "replay_train_tensor_sha256": replay_summary["train_tensor_sha256"],
            "replay_token_manifest_sha256": replay_summary[
                "token_manifest_sha256"
            ],
            "signature_sha256": signature_sha,
            "start_step": start_step,
            "target_step": target_steps,
            "optimizer_steps_this_run": target_steps - start_step,
            "joint_loss": {
                "formula": "sft_assistant_only_loss + replay_weight * pretrain_next_token_loss",
                "replay_weight": float(settings["replay_weight"]),
            },
            "sft_batch_size": int(settings["sft_batch_size"]),
            "replay_batch_size": int(settings["replay_batch_size"]),
            "replay_block_size": int(settings["replay_block_size"]),
            "learning_rate_schedule": schedule,
            "payload_summary": payload_summary,
            "sft_sampler_coverage": sft_sampler.coverage_summary(),
            "replay_consumption": replay_sampler.consumption_summary(),
            "optimizer_consumption": consumption,
            "development_consumption": {
                "role": "canary_dev_unseen_wording",
                "records": EXPECTED_EVAL_COUNT,
                "optimizer_records_per_step": 0,
                "teacher_loss_records_per_evaluation": EXPECTED_EVAL_COUNT,
                "checkpoint_selection_records_per_evaluation": EXPECTED_EVAL_COUNT,
                "teacher_evaluation_events": len(history),
                "teacher_forward_records_total": (
                    len(history) * EXPECTED_EVAL_COUNT
                ),
                "checkpoint_selection_events": len(history),
            },
            "best_development_teacher_loss": best_eval_loss,
            "history": history,
            "latest_checkpoint": _portable(args.run_dir / "latest.pt"),
            "latest_checkpoint_sha256": latest.sha256,
            "loss_csv": _portable(loss_csv),
            "elapsed_seconds": time.monotonic() - started,
            "validation_policy": {
                "canary_train": "full 64-record teacher loss every eval interval",
                "canary_dev_unseen_wording": (
                    "full 16-record teacher loss and checkpoint selection; zero optimizer use"
                ),
                "pretrain_replay": "frozen formal train tensor only",
                "pretrain_validation_test": "zero reads for optimizer",
            },
            "logging": {
                "directory": _portable(args.run_dir / "logs"),
                "modules": list(LOG_MODULES),
                "format": "rotating JSONL with UTC timestamp and run_id",
                "record_bodies_logged": False,
                "token_ids_logged": False,
                "max_bytes": int(training_config["logging"]["max_bytes"]),
                "backup_count": int(training_config["logging"]["backup_count"]),
            },
        }
        atomic_write_json(args.report, report)
        loggers["orchestrator"].info(
            "Canary replay training complete",
            extra={
                "context": {
                    "target_step": target_steps,
                    "best_development_teacher_loss": best_eval_loss,
                    "latest_checkpoint_sha256": latest.sha256,
                    "generation_and_retention_gates_pending": True,
                }
            },
        )
        state_writer.mark_done(
            {
                "step": target_steps,
                "best_development_teacher_loss": best_eval_loss,
                "latest_checkpoint_sha256": latest.sha256,
                "optimizer_consumption": consumption,
                "generation_and_retention_gates_pending": True,
            }
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "target_step": target_steps,
                    "best_development_teacher_loss": best_eval_loss,
                    "replay_consumption": report["replay_consumption"],
                    "latest_checkpoint": report["latest_checkpoint"],
                    "report": _portable(args.report),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except BaseException as error:
        log_run_failed(loggers, error, step=current_step)
        if state_writer is not None:
            state_writer.mark_failed(
                "Canary replay training failed; inspect redacted module logs",
                {
                    "step": current_step,
                    "error_type": type(error).__name__,
                    "emergency_checkpoint_written": bool(
                        emergency_hook is not None
                        and emergency_hook.save_result is not None
                    ),
                },
            )
        raise
    finally:
        if emergency_hook is not None:
            emergency_hook.uninstall()
        close_module_loggers(loggers)


if __name__ == "__main__":
    raise SystemExit(main())
