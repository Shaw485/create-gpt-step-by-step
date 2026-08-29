"""Train the sealed-safe, novel-vertical SFT v7 dataset.

The trainer accepts only a train/validation tensor artifact. Public diagnostic
records are evaluated by a separate process and sealed records are physically
unreachable from this entry point.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import time
from typing import Any, Mapping, Sequence

import torch

from train_pretrain_v4 import load_config
from train_sft_v4 import build_model, collate_records, select_device, supervised_loss
from train_sft_v5 import WeightedEpochSampler, evaluate_all_records, select_fixed_train_records
from sft_v7_vertical_catalog import (
    BOUNDARY,
    CHAT,
    CORE,
    DIMENSION_SPLIT_QUOTAS,
    EVIDENCE,
    EXPRESSION,
    RAG,
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
    restore_checkpoint,
    resolve_module_log_levels,
)


DEFAULT_CONFIG = Path("configs/formal_pretrain_14m_bpe3000.json")
DEFAULT_DATA = Path("data/sft/v7/train_val_tensors.pt")
DEFAULT_INIT_CHECKPOINT = Path(
    "runs/formal_pretrain_14m_bpe3000_6000/checkpoints/step_05750.pt"
)
DEFAULT_RUN_DIR = Path("runs/sft_v7_vertical")
DEFAULT_REPORT = Path("reports/milestones/020_sft_v7_vertical/train_report.json")

BASE_CHECKPOINT_SHA256 = (
    "bfe4fec5e6045d4c06d22393e7c2079fdc03897be71829c9d9dcbaf0fcaf5c1e"
)
BASE_CONFIG_CANONICAL_SHA256 = (
    "faac0f759a5ce9cd5e827f95c511b7f9bbbda06f6e7642e5e0d90d5ec5635974"
)
BASE_TOKEN_MANIFEST_SHA256 = (
    "5d10245eac86e4dbafef908cb2d915bb1effcf61ad977b4de96d8d64d30809c7"
)
TOKENIZER_SHA256 = (
    "e70cf3dc0ed185a6b22ab7dc08b6a850eeb59864ba161dd156c644e003862822"
)
BASE_STEP = 5750
BASE_PARAMETER_COUNT = 14_880_745
EXPECTED_MODEL_CONFIG = {
    "vocab_size": 7465,
    "block_size": 512,
    "embedding_size": 320,
    "num_layers": 10,
    "num_heads": 8,
    "ffn_multiplier": 4,
    "dropout": 0.1,
    "layer_norm_epsilon": 1e-5,
    "initialization_std": 0.02,
    "tie_embeddings": True,
}
EXPECTED_CONFIG_MODEL = {
    "block_size": 512,
    "embedding_size": 320,
    "num_layers": 10,
    "num_heads": 8,
    "ffn_multiplier": 4,
    "dropout": 0.1,
    "tie_embeddings": True,
}

DIMENSION_ORDER = (
    CORE,
    EVIDENCE,
    RAG,
    CHAT,
    EXPRESSION,
    BOUNDARY,
)
DIMENSION_WEIGHTS = {
    CORE: 0.18,
    EVIDENCE: 0.32,
    RAG: 0.14,
    CHAT: 0.18,
    EXPRESSION: 0.13,
    BOUNDARY: 0.05,
}
PHASE1_DIMENSION_WEIGHTS = {
    CORE: 0.45,
    EVIDENCE: 0.0,
    RAG: 0.0,
    CHAT: 0.40,
    EXPRESSION: 0.0,
    BOUNDARY: 0.15,
}
PHASE2_DIMENSION_WEIGHTS = dict(DIMENSION_WEIGHTS)
PHASE_ORDER = ("phase1_route_eos_core", "phase2_full_vertical_mix")
DEFAULT_PHASE1_STEPS = 400
FORBIDDEN_PAYLOAD_MARKERS = ("sealed", "test", "public")


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - old PyTorch compatibility
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError("SFT v7 tensor artifact must contain a dictionary")
    return value


def _reject_forbidden_keys(value: Any, path: str = "payload") -> None:
    """Reject hidden public/test/sealed material recursively by key name."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            if any(marker in key_text for marker in FORBIDDEN_PAYLOAD_MARKERS):
                raise ValueError(f"train/val payload contains forbidden key: {path}.{key}")
            _reject_forbidden_keys(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_forbidden_keys(nested, f"{path}[{index}]")


def load_train_val_payload(path: Path) -> dict[str, Any]:
    payload = _torch_load(path)
    _reject_forbidden_keys(payload)
    required = {
        "schema_version",
        "train_records",
        "val_records",
        "vocab_size",
        "itos",
        "special_token_ids",
        "ignore_index",
        "tokenizer_path",
        "tokenizer_sha256",
        "bpe_token_manifest_sha256",
        "sft_dataset_manifest_sha256",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"SFT v7 tensor artifact is missing keys: {missing}")
    if payload["schema_version"] != "sft-v7-train-val-tensors/v1":
        raise ValueError("unsupported SFT v7 train/val tensor schema")
    if int(payload["ignore_index"]) != -100:
        raise ValueError("SFT v7 labels must use -100 as ignore_index")
    if str(payload["tokenizer_sha256"]) != TOKENIZER_SHA256:
        raise ValueError("SFT v7 tensor tokenizer does not match the frozen tokenizer")
    if str(payload["bpe_token_manifest_sha256"]) != BASE_TOKEN_MANIFEST_SHA256:
        raise ValueError("SFT v7 tensor BPE token manifest does not match")
    if not re.fullmatch(r"[0-9a-f]{64}", str(payload["sft_dataset_manifest_sha256"])):
        raise ValueError("SFT v7 dataset manifest SHA-256 is invalid")
    return payload


def validate_frozen_config(config: Mapping[str, Any]) -> None:
    if canonical_json_sha256(config) != BASE_CONFIG_CANONICAL_SHA256:
        raise ValueError("configuration is not the frozen Step 5750 configuration")
    if config.get("model") != EXPECTED_CONFIG_MODEL:
        raise ValueError("model configuration is not the complete frozen architecture")


def validate_record_tensor(record: Mapping[str, Any], split: str, block_size: int) -> None:
    if record.get("split") != split:
        raise ValueError(f"{split} tensor record has the wrong split")
    input_ids = record.get("input_ids")
    labels = record.get("labels")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise ValueError("encoded records must contain tensor input_ids and labels")
    if input_ids.dtype != torch.long or labels.dtype != torch.long:
        raise ValueError("encoded inputs and labels must be torch.long")
    if input_ids.ndim != 1 or labels.ndim != 1 or len(input_ids) != len(labels):
        raise ValueError("encoded inputs and labels must be aligned one-dimensional tensors")
    if not 1 <= len(input_ids) <= block_size:
        raise ValueError("encoded record exceeds the model context window")
    if int((labels != -100).sum()) <= 0:
        raise ValueError("encoded record contains no supervised answer tokens")


def validate_train_val_payload(
    payload: Mapping[str, Any],
    block_size: int,
) -> dict[str, Any]:
    train_records = list(payload["train_records"])
    val_records = list(payload["val_records"])
    if len(train_records) != 8000 or len(val_records) != 800:
        raise ValueError("SFT v7 train/val split counts must be 8000/800")
    ids: dict[str, set[str]] = {"train": set(), "val": set()}
    dimension_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
    }
    supervised: Counter[str] = Counter()
    lengths: list[int] = []
    for split, records in (("train", train_records), ("val", val_records)):
        for record in records:
            validate_record_tensor(record, split, block_size)
            record_id = str(record.get("id", ""))
            if not record_id or record_id in ids[split]:
                raise ValueError(f"duplicate or empty record id in {split}")
            ids[split].add(record_id)
            dimension = str(record.get("primary_dimension", ""))
            if dimension not in DIMENSION_WEIGHTS:
                raise ValueError(f"unknown SFT v7 dimension: {dimension}")
            dimension_counts[split][dimension] += 1
            supervised[split] += int((record["labels"] != -100).sum())
            lengths.append(len(record["input_ids"]))
    if ids["train"] & ids["val"]:
        raise ValueError("train and validation IDs overlap")
    expected_train = {
        dimension: quotas["train"]
        for dimension, quotas in DIMENSION_SPLIT_QUOTAS.items()
    }
    expected_val = {
        dimension: quotas["val"]
        for dimension, quotas in DIMENSION_SPLIT_QUOTAS.items()
    }
    if dict(dimension_counts["train"]) != expected_train:
        raise ValueError("SFT v7 train dimension counts do not match the frozen protocol")
    if dict(dimension_counts["val"]) != expected_val:
        raise ValueError("SFT v7 validation dimension counts do not match the frozen protocol")
    return {
        "split_counts": {"train": len(train_records), "val": len(val_records)},
        "dimension_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in dimension_counts.items()
        },
        "supervised_tokens": dict(supervised),
        "min_sequence_length": min(lengths),
        "max_sequence_length": max(lengths),
        "mean_sequence_length": sum(lengths) / len(lengths),
    }


def validate_base_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "training-checkpoint/v1":
        raise ValueError("base checkpoint schema is not a pretraining checkpoint")
    if int(payload.get("step", -1)) != BASE_STEP:
        raise ValueError("SFT v7 must initialize from pure pretraining Step 5750")
    if payload.get("config_sha256") != BASE_CONFIG_CANONICAL_SHA256:
        raise ValueError("base checkpoint configuration hash is not frozen Step 5750")
    extra = payload.get("extra")
    if not isinstance(extra, Mapping):
        raise ValueError("base checkpoint is missing provenance metadata")
    if extra.get("initial_checkpoint") is not None:
        raise ValueError("base checkpoint is not a fresh pure-pretraining checkpoint")
    if int(extra.get("parameter_count", -1)) != BASE_PARAMETER_COUNT:
        raise ValueError("base checkpoint parameter count is incompatible")
    if extra.get("token_manifest_sha256") != BASE_TOKEN_MANIFEST_SHA256:
        raise ValueError("base checkpoint token manifest is incompatible")
    if dict(extra.get("model_config", {})) != EXPECTED_MODEL_CONFIG:
        raise ValueError("base checkpoint model architecture is incompatible")
    sft_only_markers = {
        "payload_summary",
        "data_sha256",
        "sampler_state",
        "sampler_states",
    }
    if sft_only_markers & set(extra):
        raise ValueError("an SFT checkpoint cannot be used as the fresh v7 base")


def dimension_for_record(record: Mapping[str, Any]) -> str:
    dimension = str(record.get("primary_dimension", ""))
    if dimension not in DIMENSION_WEIGHTS:
        raise ValueError(f"unknown SFT v7 dimension: {dimension}")
    return dimension


def build_family_sampler(
    records: Sequence[dict[str, Any]],
    generator: torch.Generator,
) -> WeightedEpochSampler:
    return WeightedEpochSampler(
        records,
        generator,
        {name: DIMENSION_WEIGHTS[name] for name in DIMENSION_ORDER},
        dimension_for_record,
    )


def build_phase_samplers(
    records: Sequence[dict[str, Any]],
    generator: torch.Generator,
) -> dict[str, WeightedEpochSampler]:
    """Build both frozen samplers before training for deterministic resume."""

    weights_by_phase = {
        PHASE_ORDER[0]: PHASE1_DIMENSION_WEIGHTS,
        PHASE_ORDER[1]: PHASE2_DIMENSION_WEIGHTS,
    }
    return {
        phase: WeightedEpochSampler(
            records,
            generator,
            {name: weights[name] for name in DIMENSION_ORDER},
            dimension_for_record,
        )
        for phase, weights in weights_by_phase.items()
    }


def phase_for_next_update(completed_steps: int, phase1_steps: int) -> str:
    """Select the phase for the next optimizer update at a stable boundary."""

    if completed_steps < 0:
        raise ValueError("completed_steps cannot be negative")
    if phase1_steps <= 0:
        raise ValueError("phase1_steps must be positive")
    return PHASE_ORDER[0] if completed_steps < phase1_steps else PHASE_ORDER[1]


def schedule_contract(phase1_steps: int) -> dict[str, Any]:
    """Return the signature/provenance representation of the frozen schedule."""

    return {
        "strategy": "two_phase_dimension_weighted_epoch/v1",
        "phase1_steps": phase1_steps,
        "phase1": {
            "name": PHASE_ORDER[0],
            "weights": PHASE1_DIMENSION_WEIGHTS,
        },
        "phase2": {
            "name": PHASE_ORDER[1],
            "weights": PHASE2_DIMENSION_WEIGHTS,
        },
        "switch_rule": "phase1 iff completed_optimizer_steps < phase1_steps",
    }


def phase_sampler_coverage(
    samplers: Mapping[str, WeightedEpochSampler],
) -> dict[str, dict[str, Any]]:
    return {phase: samplers[phase].coverage_summary() for phase in PHASE_ORDER}


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    best_val_loss: float,
    history: Sequence[dict[str, Any]],
    generator: torch.Generator,
    signature_sha256: str,
    provenance: Mapping[str, Any],
    samplers: Mapping[str, WeightedEpochSampler],
    phase1_steps: int,
) -> dict[str, Any]:
    return build_checkpoint_payload(
        model,
        optimizer,
        step=step,
        best_metric=best_val_loss,
        history=history,
        sampling_generator=generator,
        config_sha256=signature_sha256,
        extra={
            **dict(provenance),
            "sampler_states": {
                phase: samplers[phase].state_dict() for phase in PHASE_ORDER
            },
            "current_phase": phase_for_next_update(step, phase1_steps),
            "phase1_steps": phase1_steps,
            "selection_policy": (
                "provisional_best_validation_loss_only; release requires separate "
                "public task gates"
            ),
            "mps_resume_reproducibility": (
                "optimizer/sampler/CPU RNG restored; exact MPS dropout replay is not claimed"
            ),
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--init-checkpoint", type=Path, default=None)
    source.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--target-steps", type=int, default=20)
    parser.add_argument("--phase1-steps", type=int, default=DEFAULT_PHASE1_STEPS)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--train-eval-records", type=int, default=160)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "target_steps",
        "phase1_steps",
        "batch_size",
        "eval_interval",
        "checkpoint_interval",
        "eval_batch_size",
        "train_eval_records",
        "log_interval",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.learning_rate <= 0 or args.gradient_clip <= 0:
        raise ValueError("learning_rate and gradient_clip must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    base_config = load_config(args.config)
    validate_frozen_config(base_config)
    run_id = generate_run_id("sft-v7")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    loggers = configure_module_loggers(
        args.run_dir / "logs",
        run_id,
        resolve_module_log_levels(
            {
                "data": "INFO",
                "sft": "INFO",
                "validation": "INFO",
                "checkpoint": "INFO",
                "orchestrator": "INFO",
            }
        ),
        max_bytes=int(base_config["logging"]["max_bytes"]),
        backup_count=int(base_config["logging"]["backup_count"]),
        console=bool(base_config["logging"]["console"]),
    )
    started = time.monotonic()
    state_writer: RunStateWriter | None = None
    emergency_hook: EmergencyCheckpointHook | None = None
    current_step = 0
    try:
        device = select_device(args.device)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        generator = torch.Generator().manual_seed(args.seed + 1)

        payload = load_train_val_payload(args.data)
        model = build_model(base_config, int(payload["vocab_size"])).to(device)
        payload_summary = validate_train_val_payload(payload, model.config.block_size)
        data_sha256 = file_sha256(args.data)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=tuple(base_config["training"]["betas"]),
        )
        train_records = list(payload["train_records"])
        val_records = list(payload["val_records"])
        fixed_train = select_fixed_train_records(train_records, args.train_eval_records)
        samplers = build_phase_samplers(train_records, generator)
        schedule = schedule_contract(args.phase1_steps)
        provenance = {
            "stage": "sft_v7_vertical",
            "base_checkpoint_path": str(DEFAULT_INIT_CHECKPOINT),
            "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
            "base_checkpoint_step": BASE_STEP,
            "base_config_canonical_sha256": BASE_CONFIG_CANONICAL_SHA256,
            "base_token_manifest_sha256": BASE_TOKEN_MANIFEST_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "sft_tensor_path": str(args.data),
            "sft_tensor_sha256": data_sha256,
            "sft_dataset_manifest_sha256": str(
                payload["sft_dataset_manifest_sha256"]
            ),
            "sampling_schedule": schedule,
            "payload_summary": payload_summary,
            "public_records_consumed": 0,
            "sealed_records_consumed": 0,
        }
        signature = {
            "schema_version": "sft-v7-training-signature/v2",
            "model": base_config["model"],
            "provenance": provenance,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "betas": list(base_config["training"]["betas"]),
            "gradient_clip": args.gradient_clip,
            "sampling_schedule": schedule,
            "seed": args.seed,
        }
        signature_sha256 = canonical_json_sha256(signature)
        atomic_write_json(
            args.run_dir / "effective_config.json",
            {
                **signature,
                "signature_sha256": signature_sha256,
                "target_steps": args.target_steps,
                "phase1_steps": args.phase1_steps,
                "eval_interval": args.eval_interval,
                "checkpoint_interval": args.checkpoint_interval,
                "eval_batch_size": args.eval_batch_size,
                "device": str(device),
            },
        )
        state_writer = RunStateWriter(args.run_dir, run_id, signature_sha256)

        history: list[dict[str, Any]] = []
        start_step = 0
        best_val_loss = float("inf")
        initialization_policy = "fresh_from_frozen_pretrain_step5750"
        if args.resume is not None:
            resumed = restore_checkpoint(
                args.resume,
                model,
                optimizer,
                generator,
                expected_config_sha256=signature_sha256,
                map_location=device,
                restore_cuda_rng=False,
            )
            for key, expected in provenance.items():
                if resumed.extra.get(key) != expected:
                    raise ValueError(f"resume provenance mismatch: {key}")
            sampler_states = resumed.extra.get("sampler_states")
            if not isinstance(sampler_states, Mapping):
                raise ValueError("resume checkpoint is missing both phase sampler states")
            for phase in PHASE_ORDER:
                if phase not in sampler_states:
                    raise ValueError(f"resume checkpoint is missing sampler state: {phase}")
                samplers[phase].load_state_dict(sampler_states[phase])
            start_step = resumed.step
            expected_phase = phase_for_next_update(start_step, args.phase1_steps)
            if resumed.extra.get("current_phase") != expected_phase:
                raise ValueError("resume checkpoint phase does not match its completed step")
            if int(resumed.extra.get("phase1_steps", -1)) != args.phase1_steps:
                raise ValueError("resume checkpoint phase1 boundary changed")
            best_val_loss = resumed.best_metric
            history = resumed.history
            source_checkpoint = args.resume
            initialization_policy = "full_resume_model_optimizer_rng_sampler"
        else:
            source_checkpoint = args.init_checkpoint or DEFAULT_INIT_CHECKPOINT
            actual_sha = file_sha256(source_checkpoint)
            if actual_sha != BASE_CHECKPOINT_SHA256:
                raise ValueError("fresh SFT source is not the frozen Step 5750 checkpoint")
            initial = load_checkpoint(source_checkpoint, map_location=device)
            validate_base_checkpoint_payload(initial)
            model.load_state_dict(initial["model_state_dict"], strict=True)
        if args.target_steps <= start_step:
            raise ValueError(
                f"target_steps must exceed resumed step {start_step}, got {args.target_steps}"
            )
        current_step = start_step

        loggers["data"].info(
            "loaded train_val counts=%s dimensions=%s data_sha256=%s manifest_sha256=%s",
            payload_summary["split_counts"],
            payload_summary["dimension_counts"],
            data_sha256,
            payload["sft_dataset_manifest_sha256"],
        )
        loggers["orchestrator"].info(
            "training start policy=%s start=%d target=%d batch=%d device=%s "
            "phase1_steps=%d next_phase=%s",
            initialization_policy,
            start_step,
            args.target_steps,
            args.batch_size,
            device,
            args.phase1_steps,
            phase_for_next_update(start_step, args.phase1_steps),
        )

        def emergency_payload_factory() -> dict[str, Any]:
            return checkpoint_payload(
                model,
                optimizer,
                step=current_step,
                best_val_loss=best_val_loss,
                history=history,
                generator=generator,
                signature_sha256=signature_sha256,
                provenance=provenance,
                samplers=samplers,
                phase1_steps=args.phase1_steps,
            )

        emergency_hook = EmergencyCheckpointHook(
            args.run_dir / "emergency.pt",
            emergency_payload_factory,
            logger=loggers["checkpoint"],
        ).install()

        pad_id = int(payload["special_token_ids"]["<PAD>"])
        for step in range(start_step, args.target_steps + 1):
            evaluate_now = (
                (step == start_step and not history)
                or (step > start_step and step % args.eval_interval == 0)
                or step == args.target_steps
            )
            if evaluate_now:
                train_loss = evaluate_all_records(
                    model, fixed_train, pad_id, args.eval_batch_size, device
                )
                val_loss = evaluate_all_records(
                    model, val_records, pad_id, args.eval_batch_size, device
                )
                active_phase = phase_for_next_update(step, args.phase1_steps)
                coverage = samplers[active_phase].coverage_summary()
                coverage_by_phase = phase_sampler_coverage(samplers)
                row = {
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "active_phase": active_phase,
                    "coverage": coverage,
                    "coverage_by_phase": coverage_by_phase,
                }
                history.append(row)
                loggers["validation"].info(
                    "step=%d train_loss=%.6f full_val_loss=%.6f phase=%s "
                    "active_coverage=%.4f",
                    step,
                    train_loss,
                    val_loss,
                    active_phase,
                    coverage["coverage"],
                )
                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss = val_loss
                save_payload = checkpoint_payload(
                    model,
                    optimizer,
                    step=step,
                    best_val_loss=best_val_loss,
                    history=history,
                    generator=generator,
                    signature_sha256=signature_sha256,
                    provenance=provenance,
                    samplers=samplers,
                    phase1_steps=args.phase1_steps,
                )
                if improved:
                    result = atomic_save_checkpoint(args.run_dir / "best_val.pt", save_payload)
                    loggers["checkpoint"].info(
                        "saved provisional best-val step=%d val_loss=%.6f sha256=%s",
                        step,
                        val_loss,
                        result.sha256,
                    )
                if step > 0 and (
                    step % args.checkpoint_interval == 0 or step == args.target_steps
                ):
                    result = atomic_save_checkpoint(
                        args.run_dir / "checkpoints" / f"step_{step:05d}.pt",
                        save_payload,
                    )
                    loggers["checkpoint"].info(
                        "saved milestone step=%d sha256=%s", step, result.sha256
                    )
            if step == args.target_steps:
                break

            active_phase = phase_for_next_update(step, args.phase1_steps)
            active_sampler = samplers[active_phase]
            inputs, labels, indices = active_sampler.sample_batch(
                args.batch_size, pad_id
            )
            optimizer.zero_grad(set_to_none=True)
            loss = supervised_loss(model, inputs.to(device), labels.to(device))
            assert_finite_tensor(loss, "SFT v7 supervised loss")
            loss.backward()
            assert_finite_gradients(model.named_parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip
            )
            assert_finite_tensor(grad_norm, "SFT v7 gradient norm")
            optimizer.step()
            current_step = step + 1
            if (step + 1) % args.log_interval == 0:
                dimensions = Counter(
                    dimension_for_record(train_records[index]) for index in indices
                )
                loggers["sft"].info(
                    "step=%d batch_loss=%.6f grad_norm=%.6f supervised_tokens=%d "
                    "phase=%s batch_dimensions=%s",
                    step + 1,
                    float(loss.detach().cpu()),
                    float(grad_norm.detach().cpu()),
                    int((labels != -100).sum()),
                    active_phase,
                    dict(dimensions),
                )

        final_payload = checkpoint_payload(
            model,
            optimizer,
            step=args.target_steps,
            best_val_loss=best_val_loss,
            history=history,
            generator=generator,
            signature_sha256=signature_sha256,
            provenance=provenance,
            samplers=samplers,
            phase1_steps=args.phase1_steps,
        )
        latest = atomic_save_checkpoint(args.run_dir / "latest.pt", final_payload)
        loss_csv = args.report.with_name(args.report.stem + "_loss.csv")
        atomic_write_text(
            loss_csv,
            "step,train_loss,val_loss,coverage\n"
            + "".join(
                f"{row['step']},{row['train_loss']},{row['val_loss']},"
                f"{row['coverage']['coverage']}\n"
                for row in history
            ),
        )
        report = {
            "schema_version": "sft-v7-train-report/v1",
            "status": "training_complete_public_evaluation_pending",
            "run_id": run_id,
            "parameter_count": model.parameter_count(),
            "device": str(device),
            "initialization_policy": initialization_policy,
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": file_sha256(source_checkpoint),
            "data_path": str(args.data),
            "data_sha256": data_sha256,
            "signature_sha256": signature_sha256,
            "start_step": start_step,
            "target_step": args.target_steps,
            "optimizer_steps_this_run": args.target_steps - start_step,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "sampling_schedule": schedule,
            "sampler_coverage_by_phase": phase_sampler_coverage(samplers),
            "current_phase": phase_for_next_update(
                args.target_steps, args.phase1_steps
            ),
            "validation_policy": {
                "train": f"fixed stratified {len(fixed_train)} records",
                "validation": f"all {len(val_records)} records",
                "checkpoint_status": "provisional_only_until_public_task_gates",
            },
            "best_val_loss": best_val_loss,
            "history": history,
            "best_val_checkpoint": str(args.run_dir / "best_val.pt"),
            "latest_checkpoint": str(args.run_dir / "latest.pt"),
            "latest_checkpoint_sha256": latest.sha256,
            "loss_csv": str(loss_csv),
            "elapsed_seconds": time.monotonic() - started,
            "public_records_consumed": 0,
            "sealed_records_consumed": 0,
        }
        atomic_write_json(args.report, report)
        loggers["orchestrator"].info(
            "training complete step=%d best_val_loss=%.6f latest_sha256=%s",
            args.target_steps,
            best_val_loss,
            latest.sha256,
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "target_step": args.target_steps,
                    "best_val_loss": best_val_loss,
                    "sampler_coverage_by_phase": report[
                        "sampler_coverage_by_phase"
                    ],
                    "public_records_consumed": 0,
                    "sealed_records_consumed": 0,
                    "report": str(args.report),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        state_writer.mark_done(
            {
                "step": args.target_steps,
                "best_val_loss": best_val_loss,
                "latest_checkpoint_sha256": latest.sha256,
                "public_records_consumed": 0,
                "sealed_records_consumed": 0,
            }
        )
        return 0
    except BaseException as error:
        loggers["sft"].exception("SFT v7 training failed")
        if state_writer is not None:
            state_writer.mark_failed(
                f"{type(error).__name__}: SFT v7 training failed; inspect redacted logs",
                {
                    "step": current_step,
                    "phase": phase_for_next_update(current_step, args.phase1_steps),
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
