"""Run the M021 64-record SFT v7.1 capacity experiment.

This is a deliberately small, falsifiable experiment.  It always starts from
the exact pure-pretraining Step 5750 checkpoint unless a complete Canary
checkpoint is resumed.  The 16 unseen-question development paraphrases never
enter the optimizer sampler, but their teacher loss is measured repeatedly and
used for provisional checkpoint selection.  They are therefore development
data, not an untouched blind test.

Runtime diagnostics are isolated into ``data``, ``training``, ``validation``,
``checkpoint`` and ``orchestrator`` rotating JSONL logs.  Set one module with
``--training-log-level DEBUG`` (or ``GPT_CANARY_LOG_LEVEL_TRAINING=DEBUG``), and
use ``OFF`` to disable it.  Logs contain numeric summaries and immutable hashes,
never questions, answers, text bodies, token IDs or credentials.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any, Mapping, Sequence

import torch

from prepare_bpe_sft_v7 import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SPECIAL_TOKEN_IDS,
    EXPECTED_TOKENIZER_SHA256,
    REQUIRED_BASE_CHECKPOINT,
)
from prepare_sft_v7_1_canary import (
    CANONICAL_CANARY_PATHS,
    EXPECTED_CANARY_DATASET_IDENTITY_SHA256,
    EXPECTED_CANARY_MANIFEST_SHA256,
    EXPECTED_CANARY_SOURCE_SHA256,
    EXPECTED_CANARY_TENSOR_SHA256,
    TENSOR_SCHEMA,
    reject_restricted_keys,
)
from train_pretrain_v4 import load_config
from train_sft_v4 import build_model, collate_records, select_device, supervised_loss
from train_sft_v5 import evaluate_all_records
from train_sft_v7 import (
    BASE_CONFIG_CANONICAL_SHA256,
    BASE_PARAMETER_COUNT,
    BASE_STEP,
    EXPECTED_CONFIG_MODEL,
    validate_base_checkpoint_payload,
    validate_frozen_config,
)
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
DEFAULT_TRAINING_CONFIG = Path("configs/sft_v7_1_canary_train.json")
DEFAULT_BASE_CONFIG = Path("configs/formal_pretrain_14m_bpe3000.json")
DEFAULT_DATA = Path("data/sft/v7_1_canary/train_eval_tensors.pt")
DEFAULT_INIT_CHECKPOINT = Path(str(REQUIRED_BASE_CHECKPOINT["path"]))
DEFAULT_RUN_DIR = Path("runs/sft_v7_1_canary")
DEFAULT_REPORT = Path(
    "reports/milestones/021_sft_v7_1_canary/canary_train_report.json"
)

TRAINING_CONFIG_SCHEMA = "sft-v7.1-canary-training-config/v1"
EXPECTED_TRAINING_CONFIG_SHA256 = (
    "6648a4326d5b20335effd22f2907a631d777cbd36de4c551cb279d132abb3207"
)
TRAIN_REPORT_SCHEMA = "sft-v7.1-canary-train-report/v1"
TRAINING_SIGNATURE_SCHEMA = "sft-v7.1-canary-training-signature/v1"
EXPECTED_STAGE = "sft_v7_1_canary"
EXPECTED_TRAIN_COUNT = 64
EXPECTED_EVAL_COUNT = 16
EXPECTED_DIMENSION = "parameter_core_fact_and_correction"
EXPECTED_FAMILY = "canary_known_core"
_HEX_SHA = re.compile(r"[0-9a-f]{64}")


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - old PyTorch compatibility
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Canary tensor artifact root must be a dictionary")
    return payload


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return f"artifact://canary/{resolved.name}"


def load_training_config(path: Path) -> dict[str, Any]:
    candidate = path if path.is_absolute() else REPOSITORY_ROOT / path
    expected = REPOSITORY_ROOT / DEFAULT_TRAINING_CONFIG
    if candidate.resolve() != expected.resolve():
        raise ValueError("Canary training configuration path changed")
    if file_sha256(expected) != EXPECTED_TRAINING_CONFIG_SHA256:
        raise ValueError("Canary training configuration SHA-256 changed")
    try:
        config = json.loads(expected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Canary training config cannot be parsed") from error
    if not isinstance(config, dict) or config.get("schema_version") != TRAINING_CONFIG_SCHEMA:
        raise ValueError("unsupported Canary training configuration schema")
    if config.get("base_model_config") != str(DEFAULT_BASE_CONFIG):
        raise ValueError("Canary base model configuration binding changed")
    if config.get("base_checkpoint") != str(DEFAULT_INIT_CHECKPOINT):
        raise ValueError("Canary base checkpoint binding changed")
    data = config.get("data")
    training = config.get("training")
    logging_config = config.get("logging")
    if not all(isinstance(value, Mapping) for value in (data, training, logging_config)):
        raise ValueError("Canary training configuration sections are incomplete")
    if (
        data.get("tensor_path") != str(DEFAULT_DATA)
        or int(data.get("train_records", -1)) != EXPECTED_TRAIN_COUNT
        or int(data.get("holdout_eval_records", -1)) != EXPECTED_EVAL_COUNT
        or data.get("assistant_only_loss") is not True
        or data.get("holdout_used_for_optimization") is not False
        or data.get("holdout_eval_role")
        != "unseen_question_development_selection"
        or data.get("holdout_is_blind_test") is not False
        or data.get("holdout_used_for_teacher_loss") is not True
        or data.get("holdout_used_for_checkpoint_selection") is not True
    ):
        raise ValueError("Canary data-role contract changed")
    required_positive = (
        "target_steps",
        "batch_size",
        "learning_rate",
        "minimum_learning_rate",
        "warmup_steps",
        "gradient_clip",
        "eval_interval",
        "checkpoint_interval",
        "eval_batch_size",
        "log_interval",
    )
    for name in required_positive:
        if float(training.get(name, 0)) <= 0:
            raise ValueError(f"Canary training field must be positive: {name}")
    if float(training.get("minimum_learning_rate")) > float(training.get("learning_rate")):
        raise ValueError("minimum learning rate cannot exceed peak learning rate")
    if int(training.get("warmup_steps")) >= int(training.get("target_steps")):
        raise ValueError("warmup must finish before the Canary target step")
    betas = training.get("betas")
    if not isinstance(betas, list) or len(betas) != 2 or not all(
        0.0 < float(value) < 1.0 for value in betas
    ):
        raise ValueError("Canary AdamW betas are invalid")
    if float(training.get("weight_decay", -1)) < 0:
        raise ValueError("Canary weight decay cannot be negative")
    if training.get("sampler") != "deterministic_shuffled_epoch/v1":
        raise ValueError("Canary sampler contract changed")
    return config


def _validate_encoded_record(
    record: Mapping[str, Any],
    *,
    expected_split: str,
    block_size: int,
    eos_id: int,
) -> tuple[int, int]:
    if record.get("split") != expected_split:
        raise ValueError("Canary encoded record has the wrong split")
    if record.get("primary_dimension") != EXPECTED_DIMENSION:
        raise ValueError("Canary encoded record dimension changed")
    if record.get("task_family") != EXPECTED_FAMILY:
        raise ValueError("Canary encoded record family changed")
    input_ids = record.get("input_ids")
    labels = record.get("labels")
    if (
        not isinstance(input_ids, torch.Tensor)
        or not isinstance(labels, torch.Tensor)
        or input_ids.dtype != torch.long
        or labels.dtype != torch.long
        or input_ids.ndim != 1
        or labels.ndim != 1
        or len(input_ids) != len(labels)
        or not 1 <= len(input_ids) < block_size
    ):
        raise ValueError("Canary encoded tensor shape is invalid")
    supervised = labels[labels != -100]
    if len(supervised) <= 0 or int(supervised[-1]) != eos_id:
        raise ValueError("Canary record must supervise an answer ending in EOS")
    if int((supervised == eos_id).sum()) != 1:
        raise ValueError("Canary single-turn record must supervise exactly one EOS")
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, Mapping) or (
        evaluation.get("metric") != "required_terms_all"
        or evaluation.get("known_fact") is not True
        or not isinstance(evaluation.get("required_terms"), list)
        or not evaluation.get("required_terms")
        or not isinstance(evaluation.get("forbidden_terms"), list)
    ):
        raise ValueError("Canary encoded evaluation metadata changed")
    return len(input_ids), len(supervised)


def validate_canary_tensor_payload(
    payload: Mapping[str, Any], block_size: int
) -> dict[str, Any]:
    """Prove the optimizer-visible set is only the 64 Canary records."""

    reject_restricted_keys(payload)
    required = {
        "schema_version",
        "train_records",
        "eval_records",
        "vocab_size",
        "stoi",
        "itos",
        "special_token_ids",
        "ignore_index",
        "tokenizer_path",
        "tokenizer_sha256",
        "bpe_token_manifest_path",
        "bpe_token_manifest_sha256",
        "canary_dataset_manifest_path",
        "canary_dataset_manifest_sha256",
        "canary_dataset_identity_sha256",
        "source_jsonl_paths",
        "source_jsonl_sha256",
        "required_base_checkpoint",
        "artifact_binding_sha256",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Canary tensor artifact is missing {len(missing)} fields")
    if payload.get("schema_version") != TENSOR_SCHEMA:
        raise ValueError("unsupported Canary tensor schema")
    record_keys = {str(key) for key in payload if str(key).endswith("_records")}
    if record_keys != {"train_records", "eval_records"}:
        raise ValueError("Canary tensor record scope is not isolated")
    if int(payload["vocab_size"]) != 7465 or int(payload["ignore_index"]) != -100:
        raise ValueError("Canary vocabulary or ignore index changed")
    actual_special = {
        str(key): int(value) for key, value in dict(payload["special_token_ids"]).items()
    }
    if actual_special != EXPECTED_SPECIAL_TOKEN_IDS:
        raise ValueError("Canary special-token IDs changed")
    if payload["tokenizer_sha256"] != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("Canary tokenizer SHA changed")
    if payload["bpe_token_manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Canary BPE manifest SHA changed")
    if payload["canary_dataset_manifest_sha256"] != EXPECTED_CANARY_MANIFEST_SHA256:
        raise ValueError("Canary dataset manifest SHA changed")
    if (
        payload["canary_dataset_identity_sha256"]
        != EXPECTED_CANARY_DATASET_IDENTITY_SHA256
    ):
        raise ValueError("Canary dataset identity changed")
    source_paths = payload["source_jsonl_paths"]
    source_hashes = payload["source_jsonl_sha256"]
    if not isinstance(source_paths, Mapping) or set(source_paths) != {
        "train",
        "holdout_eval",
    }:
        raise ValueError("Canary source path scope changed")
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != {
        "train",
        "holdout_eval",
    } or dict(source_hashes) != EXPECTED_CANARY_SOURCE_SHA256:
        raise ValueError("Canary source hashes changed")
    expected_source_paths = {
        "train": CANONICAL_CANARY_PATHS["train"].as_posix(),
        "holdout_eval": CANONICAL_CANARY_PATHS["holdout_eval"].as_posix(),
    }
    if dict(source_paths) != expected_source_paths:
        raise ValueError("Canary source paths changed")
    if (
        payload["canary_dataset_manifest_path"]
        != CANONICAL_CANARY_PATHS["manifest"].as_posix()
    ):
        raise ValueError("Canary dataset manifest path changed")
    base = payload["required_base_checkpoint"]
    expected_base = dict(REQUIRED_BASE_CHECKPOINT)
    expected_base["binding_sha256"] = canonical_json_sha256(REQUIRED_BASE_CHECKPOINT)
    if not isinstance(base, Mapping) or dict(base) != expected_base:
        raise ValueError("Canary required base checkpoint changed")
    binding = {
        "schema_version": TENSOR_SCHEMA,
        "source_jsonl_sha256": dict(source_hashes),
        "tokenizer_sha256": payload["tokenizer_sha256"],
        "bpe_token_manifest_sha256": payload["bpe_token_manifest_sha256"],
        "canary_dataset_manifest_sha256": payload["canary_dataset_manifest_sha256"],
        "canary_dataset_identity_sha256": payload["canary_dataset_identity_sha256"],
        "required_base_checkpoint": dict(base),
    }
    if payload["artifact_binding_sha256"] != canonical_json_sha256(binding):
        raise ValueError("Canary tensor artifact binding is invalid")

    train_records = list(payload["train_records"])
    eval_records = list(payload["eval_records"])
    if len(train_records) != EXPECTED_TRAIN_COUNT or len(eval_records) != EXPECTED_EVAL_COUNT:
        raise ValueError("Canary tensor split counts must be 64 and 16")
    seen: set[str] = set()
    lengths: list[int] = []
    supervised_counts = {"train": 0, "holdout_eval": 0}
    eos_id = int(actual_special["<EOS>"])
    for split, records in (("train", train_records), ("holdout_eval", eval_records)):
        for record in records:
            identifier = record.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise ValueError("Canary tensor record ID is empty or duplicated")
            seen.add(identifier)
            length, supervised = _validate_encoded_record(
                record,
                expected_split=split,
                block_size=block_size,
                eos_id=eos_id,
            )
            lengths.append(length)
            supervised_counts[split] += supervised
    return {
        "split_counts": {"train": len(train_records), "holdout_eval": len(eval_records)},
        "optimizer_record_count": len(train_records),
        "holdout_optimizer_record_count": 0,
        "supervised_token_counts": supervised_counts,
        "minimum_sequence_length": min(lengths),
        "maximum_sequence_length": max(lengths),
        "mean_sequence_length": sum(lengths) / len(lengths),
    }


def load_canary_tensor_payload(path: Path) -> dict[str, Any]:
    candidate = path if path.is_absolute() else REPOSITORY_ROOT / path
    expected = REPOSITORY_ROOT / CANONICAL_CANARY_PATHS["tensor"]
    if candidate.resolve() != expected.resolve():
        raise ValueError("Canary tensor artifact path changed")
    sidecar = Path(f"{path}.sha256")
    if not sidecar.is_file():
        raise ValueError("Canary tensor checksum sidecar is missing")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if not fields or not _HEX_SHA.fullmatch(fields[0].lower()):
        raise ValueError("Canary tensor checksum sidecar is malformed")
    calculated = file_sha256(path)
    if calculated != fields[0].lower():
        raise ValueError("Canary tensor SHA-256 does not match its sidecar")
    if calculated != EXPECTED_CANARY_TENSOR_SHA256:
        raise ValueError("Canary tensor SHA-256 is not the reviewed artifact")
    return _torch_load(path)


class DeterministicShuffledEpochSampler:
    """Shuffle each 64-record epoch once and visit every record before reuse."""

    def __init__(
        self, records: Sequence[dict[str, Any]], generator: torch.Generator
    ) -> None:
        if not records:
            raise ValueError("Canary sampler records cannot be empty")
        identifiers = [str(record.get("id", "")) for record in records]
        if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError("Canary sampler record IDs must be unique")
        self.records = records
        self.generator = generator
        self.queue: list[int] = []
        self.position = 0
        self.epochs_started = 0
        self.draw_counts = [0 for _ in records]

    def _reshuffle(self) -> None:
        self.queue = torch.randperm(len(self.records), generator=self.generator).tolist()
        self.position = 0
        self.epochs_started += 1

    def sample_indices(self, batch_size: int) -> list[int]:
        if batch_size <= 0:
            raise ValueError("Canary batch size must be positive")
        indexes: list[int] = []
        while len(indexes) < batch_size:
            if self.position >= len(self.queue):
                self._reshuffle()
            take = min(batch_size - len(indexes), len(self.queue) - self.position)
            chunk = self.queue[self.position : self.position + take]
            self.position += take
            for index in chunk:
                self.draw_counts[index] += 1
            indexes.extend(chunk)
        return indexes

    def sample_batch(
        self, batch_size: int, pad_token_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        indexes = self.sample_indices(batch_size)
        inputs, labels = collate_records(
            [self.records[index] for index in indexes], pad_token_id
        )
        return inputs, labels, indexes

    def state_dict(self) -> dict[str, Any]:
        return {
            "strategy": "deterministic_shuffled_epoch/v1",
            "record_ids": [str(record["id"]) for record in self.records],
            "queue_record_ids": [str(self.records[index]["id"]) for index in self.queue],
            "position": self.position,
            "epochs_started": self.epochs_started,
            "draw_counts": {
                str(self.records[index]["id"]): count
                for index, count in enumerate(self.draw_counts)
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected_ids = [str(record["id"]) for record in self.records]
        if state.get("strategy") != "deterministic_shuffled_epoch/v1" or list(
            state.get("record_ids", [])
        ) != expected_ids:
            raise ValueError("Canary sampler state belongs to different records")
        id_to_index = {identifier: index for index, identifier in enumerate(expected_ids)}
        queue_ids = list(state.get("queue_record_ids", []))
        if queue_ids and (len(queue_ids) != len(expected_ids) or set(queue_ids) != set(expected_ids)):
            raise ValueError("Canary sampler queue is invalid")
        self.queue = [id_to_index[str(identifier)] for identifier in queue_ids]
        self.position = int(state.get("position", 0))
        if not 0 <= self.position <= len(self.queue):
            raise ValueError("Canary sampler position is invalid")
        self.epochs_started = int(state.get("epochs_started", 0))
        counts = state.get("draw_counts")
        if not isinstance(counts, Mapping) or set(counts) != set(expected_ids):
            raise ValueError("Canary sampler draw counts are invalid")
        self.draw_counts = [int(counts[identifier]) for identifier in expected_ids]

    def coverage_summary(self) -> dict[str, Any]:
        seen = sum(count > 0 for count in self.draw_counts)
        return {
            "strategy": "deterministic_shuffled_epoch/v1",
            "records": len(self.records),
            "seen_records": seen,
            "coverage": seen / len(self.records),
            "draws": sum(self.draw_counts),
            "epochs_started": self.epochs_started,
            "minimum_draws_per_record": min(self.draw_counts),
            "maximum_draws_per_record": max(self.draw_counts),
        }


def learning_rate_for_update(
    completed_steps: int,
    *,
    target_steps: int,
    warmup_steps: int,
    peak_learning_rate: float,
    minimum_learning_rate: float,
) -> float:
    """Warm up linearly, then cosine-decay so the final update uses the floor."""

    if not 0 <= completed_steps < target_steps:
        raise ValueError("completed_steps must identify a pending optimizer update")
    if not 0 < warmup_steps < target_steps:
        raise ValueError("warmup_steps must be between zero and target_steps")
    if not 0 < minimum_learning_rate <= peak_learning_rate:
        raise ValueError("learning-rate bounds are invalid")
    update_number = completed_steps + 1
    if update_number <= warmup_steps:
        return peak_learning_rate * update_number / warmup_steps
    decay_updates = target_steps - warmup_steps
    decay_position = update_number - warmup_steps
    progress = decay_position / decay_updates
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_learning_rate + coefficient * (
        peak_learning_rate - minimum_learning_rate
    )


def build_optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
) -> torch.optim.AdamW:
    """Decay matrix weights, but not biases or normalization scales."""

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for _name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=betas,
    )


def schedule_contract(settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "strategy": "linear_warmup_cosine_decay/v1",
        "target_steps": int(settings["target_steps"]),
        "warmup_steps": int(settings["warmup_steps"]),
        "peak_learning_rate": float(settings["learning_rate"]),
        "minimum_learning_rate": float(settings["minimum_learning_rate"]),
        "sampler": "deterministic_shuffled_epoch/v1",
    }


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    best_eval_loss: float,
    history: Sequence[Mapping[str, Any]],
    generator: torch.Generator,
    signature_sha256: str,
    provenance: Mapping[str, Any],
    sampler: DeterministicShuffledEpochSampler,
    schedule: Mapping[str, Any],
) -> dict[str, Any]:
    return build_checkpoint_payload(
        model,
        optimizer,
        step=step,
        best_metric=best_eval_loss,
        history=history,
        sampling_generator=generator,
        config_sha256=signature_sha256,
        extra={
            **dict(provenance),
            "sampler_state": sampler.state_dict(),
            "learning_rate_schedule": dict(schedule),
            "development_teacher_loss_evaluation_events": len(history),
            "development_teacher_loss_record_forwards": (
                len(history) * EXPECTED_EVAL_COUNT
            ),
            "selection_policy": (
                "development_teacher_loss_selects_provisional_best; capacity release also "
                "requires generation gates"
            ),
            "mps_resume_reproducibility": (
                "optimizer/sampler/CPU RNG restored; exact MPS dropout replay is not claimed"
            ),
        },
    )


def _resolve_settings(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    settings = dict(config["training"])
    cli_names = (
        "target_steps",
        "batch_size",
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
    )
    for name in cli_names:
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    if int(settings["target_steps"]) <= 0 or int(settings["batch_size"]) <= 0:
        raise ValueError("Canary target steps and batch size must be positive")
    if not 0 < int(settings["warmup_steps"]) < int(settings["target_steps"]):
        raise ValueError("Canary warmup must finish before target steps")
    if not 0 < float(settings["minimum_learning_rate"]) <= float(
        settings["learning_rate"]
    ):
        raise ValueError("Canary learning-rate bounds are invalid")
    for name in ("gradient_clip", "eval_interval", "checkpoint_interval", "eval_batch_size", "log_interval"):
        if float(settings[name]) <= 0:
            raise ValueError(f"Canary setting must be positive: {name}")
    if float(settings["weight_decay"]) < 0:
        raise ValueError("Canary weight decay cannot be negative")
    return settings


def _add_log_arguments(parser: argparse.ArgumentParser, modules: Sequence[str]) -> None:
    for module in modules:
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
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--init-checkpoint", type=Path, default=None)
    source.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--target-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
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
    _add_log_arguments(
        parser, ("data", "training", "validation", "checkpoint", "orchestrator")
    )
    return parser.parse_args(argv)


def _log_levels(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    modules: Sequence[str],
) -> dict[str, str]:
    configured = dict(config["logging"].get("module_levels", {}))
    levels = resolve_module_log_levels(
        {module: str(configured.get(module, "INFO")) for module in modules},
        env_prefix="GPT_CANARY_LOG_LEVEL",
    )
    for module in modules:
        override = getattr(args, f"{module}_log_level")
        if override is not None:
            levels[module] = override
    return levels


def _set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer, learning_rate: float
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def log_training_failure(logger: Any, error: BaseException, *, step: int) -> None:
    """Write a body/path-safe failure event without serializing a traceback."""

    logger.error(
        "Canary training failed",
        extra={
            "context": {
                "error_code": "CANARY_TRAINING_FAILED",
                "error_type": type(error).__name__,
                "step": step,
                "inspect_module_logs": True,
            }
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    training_config = load_training_config(args.training_config)
    settings = _resolve_settings(args, training_config)
    base_config = load_config(args.base_config)
    validate_frozen_config(base_config)
    modules = ("data", "training", "validation", "checkpoint", "orchestrator")
    run_id = generate_run_id("sft-v7-1-canary")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    loggers = configure_module_loggers(
        args.run_dir / "logs",
        run_id,
        _log_levels(args, training_config, modules),
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
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)

        payload = load_canary_tensor_payload(args.data)
        model = build_model(base_config, int(payload["vocab_size"])).to(device)
        if model.parameter_count() != BASE_PARAMETER_COUNT:
            raise ValueError("Canary model parameter count is not the frozen 14.9M architecture")
        payload_summary = validate_canary_tensor_payload(
            payload, int(model.config.block_size)
        )
        tensor_sha = file_sha256(args.data)
        manifest_path = REPOSITORY_ROOT / Path(
            str(payload["canary_dataset_manifest_path"])
        )
        if (
            not manifest_path.is_file()
            or file_sha256(manifest_path)
            != str(payload["canary_dataset_manifest_sha256"])
        ):
            raise ValueError("Canary tensor no longer matches its dataset manifest")
        train_records = list(payload["train_records"])
        eval_records = list(payload["eval_records"])
        sampler = DeterministicShuffledEpochSampler(train_records, generator)
        optimizer = build_optimizer(
            model,
            learning_rate=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
            betas=tuple(float(value) for value in settings["betas"]),
        )
        schedule = schedule_contract(settings)
        provenance = {
            "stage": EXPECTED_STAGE,
            "base_checkpoint_path": str(DEFAULT_INIT_CHECKPOINT),
            "base_checkpoint_sha256": REQUIRED_BASE_CHECKPOINT["sha256"],
            "base_checkpoint_step": BASE_STEP,
            "base_config_canonical_sha256": BASE_CONFIG_CANONICAL_SHA256,
            "base_token_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
            "sft_tensor_path": _portable(args.data),
            "sft_tensor_sha256": tensor_sha,
            # Explicit Canary names are the canonical retention/evaluation API;
            # the sft_* aliases keep existing training tooling compatible.
            "canary_tensor_path": _portable(args.data),
            "canary_tensor_sha256": tensor_sha,
            "canary_dataset_manifest_sha256": payload[
                "canary_dataset_manifest_sha256"
            ],
            "canary_dataset_identity_sha256": payload[
                "canary_dataset_identity_sha256"
            ],
            "payload_summary": payload_summary,
            "optimization_train_records": EXPECTED_TRAIN_COUNT,
            "optimization_holdout_records": 0,
            "development_unseen_wording_records": EXPECTED_EVAL_COUNT,
            "development_optimizer_records": 0,
            "development_records_used_for_optimization": 0,
            "development_records_consumed_for_teacher_loss": EXPECTED_EVAL_COUNT,
            "development_records_used_for_checkpoint_selection": EXPECTED_EVAL_COUNT,
            "development_split_role": "unseen_question_development_selection",
            # Legacy field retained for old checkpoint readers. Its historical
            # meaning was optimizer consumption, not whether teacher loss ran.
            "holdout_records_consumed": 0,
            "teacher_loss_holdout_records": EXPECTED_EVAL_COUNT,
            "public_records_consumed": 0,
            "sealed_records_consumed": 0,
        }
        signature = {
            "schema_version": TRAINING_SIGNATURE_SCHEMA,
            "model": base_config["model"],
            "provenance": provenance,
            "training": {
                key: settings[key]
                for key in (
                    "target_steps",
                    "batch_size",
                    "learning_rate",
                    "minimum_learning_rate",
                    "warmup_steps",
                    "weight_decay",
                    "betas",
                    "gradient_clip",
                    "seed",
                    "sampler",
                )
            },
            "schedule": schedule,
        }
        signature_sha = canonical_json_sha256(signature)
        atomic_write_json(
            args.run_dir / "effective_config.json",
            {
                **signature,
                "signature_sha256": signature_sha,
                "eval_interval": settings["eval_interval"],
                "checkpoint_interval": settings["checkpoint_interval"],
                "eval_batch_size": settings["eval_batch_size"],
                "log_interval": settings["log_interval"],
                "device": str(device),
                "runtime_logging": {
                    "modules": list(modules),
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
                generator,
                expected_config_sha256=signature_sha,
                map_location=device,
                restore_cuda_rng=False,
            )
            for key, expected in provenance.items():
                if resumed.extra.get(key) != expected:
                    raise ValueError(f"Canary resume provenance mismatch: {key}")
            sampler_state = resumed.extra.get("sampler_state")
            if not isinstance(sampler_state, Mapping):
                raise ValueError("Canary resume checkpoint lacks sampler state")
            sampler.load_state_dict(sampler_state)
            if resumed.extra.get("learning_rate_schedule") != schedule:
                raise ValueError("Canary resume learning-rate schedule changed")
            start_step = resumed.step
            best_eval_loss = resumed.best_metric
            history = resumed.history
            source_checkpoint = args.resume
            initialization_policy = "full_resume_model_optimizer_rng_sampler"
        else:
            source_checkpoint = args.init_checkpoint or DEFAULT_INIT_CHECKPOINT
            if file_sha256(source_checkpoint) != REQUIRED_BASE_CHECKPOINT["sha256"]:
                raise ValueError("Canary fresh source is not frozen Step 5750")
            initial = load_checkpoint(source_checkpoint, map_location=device)
            validate_base_checkpoint_payload(initial)
            model.load_state_dict(initial["model_state_dict"], strict=True)
        target_steps = int(settings["target_steps"])
        if target_steps <= start_step:
            raise ValueError("Canary target step must exceed resumed step")
        current_step = start_step
        pad_id = int(payload["special_token_ids"]["<PAD>"])
        loggers["data"].info(
            "Canary tensor validated",
            extra={
                "context": {
                    "tensor_sha256": tensor_sha,
                    "train_count": EXPECTED_TRAIN_COUNT,
                    "holdout_eval_count": EXPECTED_EVAL_COUNT,
                    "optimizer_holdout_count": 0,
                    "maximum_sequence_length": payload_summary[
                        "maximum_sequence_length"
                    ],
                }
            },
        )
        loggers["orchestrator"].info(
            "Canary training started",
            extra={
                "context": {
                    "policy": initialization_policy,
                    "start_step": start_step,
                    "target_step": target_steps,
                    "batch_size": int(settings["batch_size"]),
                    "device": str(device),
                    "signature_sha256": signature_sha,
                }
            },
        )

        def make_current_checkpoint() -> dict[str, Any]:
            return checkpoint_payload(
                model,
                optimizer,
                step=current_step,
                best_eval_loss=best_eval_loss,
                history=history,
                generator=generator,
                signature_sha256=signature_sha,
                provenance=provenance,
                sampler=sampler,
                schedule=schedule,
            )

        emergency_hook = EmergencyCheckpointHook(
            args.run_dir / "emergency.pt",
            make_current_checkpoint,
            logger=loggers["checkpoint"],
        ).install()

        def evaluate_and_checkpoint(step: int) -> None:
            nonlocal best_eval_loss
            train_loss = evaluate_all_records(
                model,
                train_records,
                pad_id,
                int(settings["eval_batch_size"]),
                device,
            )
            eval_loss = evaluate_all_records(
                model,
                eval_records,
                pad_id,
                int(settings["eval_batch_size"]),
                device,
            )
            row = {
                "step": step,
                "train_teacher_loss": train_loss,
                "holdout_teacher_loss": eval_loss,
                "learning_rate": (
                    0.0
                    if step == 0
                    else float(optimizer.param_groups[0]["lr"])
                ),
                "sampler_coverage": sampler.coverage_summary(),
            }
            history.append(row)
            loggers["validation"].info(
                "Canary full-set teacher loss evaluated",
                extra={
                    "context": {
                        "step": step,
                        "train_teacher_loss": train_loss,
                        "holdout_teacher_loss": eval_loss,
                        "coverage": row["sampler_coverage"]["coverage"],
                    }
                },
            )
            improved = eval_loss < best_eval_loss
            if improved:
                best_eval_loss = eval_loss
                result = atomic_save_checkpoint(
                    args.run_dir / "best_teacher_loss.pt", make_current_checkpoint()
                )
                loggers["checkpoint"].info(
                    "Provisional best teacher-loss checkpoint saved",
                    extra={
                        "context": {
                            "step": step,
                            "holdout_teacher_loss": eval_loss,
                            "sha256": result.sha256,
                        }
                    },
                )
            if step > 0 and (
                step % int(settings["checkpoint_interval"]) == 0
                or step == target_steps
            ):
                result = atomic_save_checkpoint(
                    args.run_dir / "checkpoints" / f"step_{step:05d}.pt",
                    make_current_checkpoint(),
                )
                loggers["checkpoint"].info(
                    "Canary milestone checkpoint saved",
                    extra={"context": {"step": step, "sha256": result.sha256}},
                )

        if not history:
            evaluate_and_checkpoint(start_step)
        for completed_steps in range(start_step, target_steps):
            learning_rate = learning_rate_for_update(
                completed_steps,
                target_steps=target_steps,
                warmup_steps=int(settings["warmup_steps"]),
                peak_learning_rate=float(settings["learning_rate"]),
                minimum_learning_rate=float(settings["minimum_learning_rate"]),
            )
            _set_optimizer_learning_rate(optimizer, learning_rate)
            inputs, labels, _indexes = sampler.sample_batch(
                int(settings["batch_size"]), pad_id
            )
            optimizer.zero_grad(set_to_none=True)
            loss = supervised_loss(model, inputs.to(device), labels.to(device))
            assert_finite_tensor(loss, "SFT v7.1 Canary supervised loss")
            loss.backward()
            assert_finite_gradients(model.named_parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(settings["gradient_clip"])
            )
            assert_finite_tensor(grad_norm, "SFT v7.1 Canary gradient norm")
            optimizer.step()
            current_step = completed_steps + 1
            if current_step % int(settings["log_interval"]) == 0:
                loggers["training"].info(
                    "Canary optimizer update complete",
                    extra={
                        "context": {
                            "step": current_step,
                            "batch_loss": float(loss.detach().cpu()),
                            "gradient_norm": float(grad_norm.detach().cpu()),
                            "learning_rate": learning_rate,
                            "supervised_token_count": int((labels != -100).sum()),
                        }
                    },
                )
            if (
                current_step % int(settings["eval_interval"]) == 0
                or current_step == target_steps
            ):
                evaluate_and_checkpoint(current_step)

        latest = atomic_save_checkpoint(args.run_dir / "latest.pt", make_current_checkpoint())
        loss_csv = args.report.with_name(args.report.stem + "_loss.csv")
        atomic_write_text(
            loss_csv,
            "step,train_teacher_loss,holdout_teacher_loss,learning_rate,coverage\n"
            + "".join(
                f"{row['step']},{row['train_teacher_loss']},"
                f"{row['holdout_teacher_loss']},{row['learning_rate']},"
                f"{row['sampler_coverage']['coverage']}\n"
                for row in history
            ),
        )
        report = {
            "report_schema_version": TRAIN_REPORT_SCHEMA,
            "status": "training_complete_generation_gates_pending",
            "run_id": run_id,
            "parameter_count": model.parameter_count(),
            "device": str(device),
            "initialization_policy": initialization_policy,
            "source_checkpoint": _portable(source_checkpoint),
            "source_checkpoint_sha256": file_sha256(source_checkpoint),
            "base_checkpoint_step": BASE_STEP,
            "data_path": _portable(args.data),
            "data_sha256": tensor_sha,
            "canary_dataset_manifest_sha256": payload[
                "canary_dataset_manifest_sha256"
            ],
            "signature_sha256": signature_sha,
            "start_step": start_step,
            "target_step": target_steps,
            "optimizer_steps_this_run": target_steps - start_step,
            "batch_size": int(settings["batch_size"]),
            "learning_rate_schedule": schedule,
            "weight_decay": float(settings["weight_decay"]),
            "sampler_coverage": sampler.coverage_summary(),
            "payload_summary": payload_summary,
            "validation_policy": {
                "train": "all 64 records",
                "holdout_eval": (
                    "all 16 unseen-question development paraphrases; teacher loss and "
                    "provisional checkpoint selection; zero gradient use"
                ),
                "selection": "development teacher loss plus generation gates",
            },
            "best_holdout_teacher_loss": best_eval_loss,
            "history": history,
            "latest_checkpoint": _portable(args.run_dir / "latest.pt"),
            "latest_checkpoint_sha256": latest.sha256,
            "best_teacher_loss_checkpoint": _portable(
                args.run_dir / "best_teacher_loss.pt"
            ),
            "loss_csv": _portable(loss_csv),
            "elapsed_seconds": time.monotonic() - started,
            "optimization_train_records": EXPECTED_TRAIN_COUNT,
            "optimization_holdout_records": 0,
            "development_optimizer_records": 0,
            "development_records_consumed_for_teacher_loss": EXPECTED_EVAL_COUNT,
            "development_records_used_for_checkpoint_selection": EXPECTED_EVAL_COUNT,
            "development_teacher_loss_evaluation_events": len(history),
            "development_teacher_loss_record_forwards": (
                len(history) * EXPECTED_EVAL_COUNT
            ),
            "public_records_consumed": 0,
            "sealed_records_consumed": 0,
            "logging": {
                "directory": _portable(args.run_dir / "logs"),
                "modules": list(modules),
                "format": "rotating JSONL with UTC timestamp and run_id",
                "record_bodies_logged": False,
                "token_ids_logged": False,
                "max_bytes": int(training_config["logging"]["max_bytes"]),
                "backup_count": int(training_config["logging"]["backup_count"]),
            },
        }
        atomic_write_json(args.report, report)
        loggers["orchestrator"].info(
            "Canary training complete",
            extra={
                "context": {
                    "target_step": target_steps,
                    "best_holdout_teacher_loss": best_eval_loss,
                    "latest_sha256": latest.sha256,
                    "generation_gates_pending": True,
                }
            },
        )
        state_writer.mark_done(
            {
                "step": target_steps,
                "best_holdout_teacher_loss": best_eval_loss,
                "latest_checkpoint_sha256": latest.sha256,
                "generation_gates_pending": True,
                "public_records_consumed": 0,
                "sealed_records_consumed": 0,
            }
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "target_step": target_steps,
                    "best_holdout_teacher_loss": best_eval_loss,
                    "sampler_coverage": report["sampler_coverage"],
                    "latest_checkpoint": report["latest_checkpoint"],
                    "report": _portable(args.report),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except BaseException as error:
        log_training_failure(loggers["training"], error, step=current_step)
        if state_writer is not None:
            state_writer.mark_failed(
                f"{type(error).__name__}: Canary training failed; inspect redacted logs",
                {
                    "step": current_step,
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
