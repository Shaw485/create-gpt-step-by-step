"""Train SFT v5 with balanced epoch sampling and complete resumability.

This entry point keeps the historical v4 trainer unchanged for reproducibility.
It fixes four issues found in M013: sampling with replacement, repeated sampling
after a continuation run, optimizer-state loss, and checkpoint selection from a
tiny random validation sample.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from bpe_tokenizer import BPETokenizer
from evaluate_sft_v4_categories import EVAL_ITEMS, score_item, summarize
from sample_sft_v4_custom import build_prompt_ids
from train_pretrain_v4 import load_config
from train_sft_v4 import (
    build_model,
    collate_records,
    generate_answer,
    load_sft_payload,
    select_device,
    supervised_loss,
    validate_sft_payload,
)
from training_runtime import (
    assert_finite_gradients,
    assert_finite_tensor,
    atomic_save_checkpoint,
    atomic_write_json,
    atomic_write_text,
    build_checkpoint_payload,
    canonical_json_sha256,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
    load_checkpoint,
    restore_checkpoint,
)


DEFAULT_CONFIG_PATH = Path("configs/local_m4_8m_continue_6000.json")
DEFAULT_DATA_PATH = Path("data/cloud_v4/sft_v5_2_2_core_routing_tensors.pt")
DEFAULT_INIT_CHECKPOINT = Path("runs/pretrain_v4_m4_continue6000/best.pt")
DEFAULT_RUN_DIR = Path("runs/sft_v5_2_2_core_routing")
DEFAULT_REPORT_PATH = Path(
    "reports/milestones/014_v5_2_entity_routing/train_report.json"
)
DEFAULT_TARGET_STEPS = 1000
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WEIGHT_DECAY = 0.05
DEFAULT_GRADIENT_CLIP = 1.0
DEFAULT_EVAL_INTERVAL = 250
DEFAULT_TRAIN_EVAL_RECORDS = 160
DEFAULT_EVAL_BATCH_SIZE = 8
DEFAULT_LOG_INTERVAL = 25
DEFAULT_SEED = 42
EVALUATION_SUITE_VERSION = "no-math-v2.1-strict-entity"

KNOWN_FAMILIES = {
    "novel_known_entity_v5_2",
    "novel_relation_v5_2",
}
CORE_FAMILIES = {"novel_core_entity_v5_2"}
CONTRAST_FAMILIES = {"novel_unknown_grounded_v5_2"}
POOL_ORDER = ("replay", "core", "known", "contrast")


class WeightedEpochSampler:
    """Draw weighted pools while traversing each pool without replacement."""

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        generator: torch.Generator,
        weights: Mapping[str, float],
        pool_selector: Callable[[dict[str, Any]], str],
    ) -> None:
        self.records = records
        self.generator = generator
        self.pool_order = tuple(weights)
        total_weight = sum(float(weights[name]) for name in self.pool_order)
        if total_weight <= 0:
            raise ValueError("sampler weights must sum to a positive value")
        self.weights = {
            name: float(weights[name]) / total_weight for name in self.pool_order
        }
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("sampler weights cannot be negative")

        pools: dict[str, list[int]] = {name: [] for name in self.pool_order}
        for index, record in enumerate(records):
            pool = pool_selector(record)
            if pool not in pools:
                raise ValueError(f"pool selector returned unknown pool: {pool}")
            pools[pool].append(index)
        for name, weight in self.weights.items():
            if weight > 0 and not pools[name]:
                raise ValueError(f"weighted sampler pool is empty: {name}")
        self.pools = pools
        self.queues = {name: [] for name in self.pool_order}
        self.positions = {name: 0 for name in self.pool_order}
        self.credits = {name: 0.0 for name in self.pool_order}
        self.pool_draw_counts: Counter[str] = Counter()
        self.record_draw_counts: Counter[int] = Counter()
        self.pool_cycles_started: Counter[str] = Counter()

    def _reshuffle(self, pool: str) -> None:
        source = self.pools[pool]
        permutation = torch.randperm(len(source), generator=self.generator).tolist()
        self.queues[pool] = [source[index] for index in permutation]
        self.positions[pool] = 0
        self.pool_cycles_started[pool] += 1

    def _choose_pool(self) -> str:
        for name in self.pool_order:
            self.credits[name] += self.weights[name]
        active = [name for name in self.pool_order if self.weights[name] > 0]
        selected = max(active, key=lambda name: (self.credits[name], -self.pool_order.index(name)))
        self.credits[selected] -= 1.0
        return selected

    def sample_indices(self, batch_size: int) -> list[int]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        selected: list[int] = []
        for _ in range(batch_size):
            pool = self._choose_pool()
            if self.positions[pool] >= len(self.queues[pool]):
                self._reshuffle(pool)
            record_index = self.queues[pool][self.positions[pool]]
            self.positions[pool] += 1
            self.pool_draw_counts[pool] += 1
            self.record_draw_counts[record_index] += 1
            selected.append(record_index)
        return selected

    def sample_batch(
        self,
        batch_size: int,
        pad_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        indices = self.sample_indices(batch_size)
        inputs, labels = collate_records(
            [self.records[index] for index in indices],
            pad_token_id,
        )
        return inputs, labels, indices

    def state_dict(self) -> dict[str, Any]:
        return {
            "pool_order": list(self.pool_order),
            "weights": dict(self.weights),
            "pool_record_ids": {
                name: [self.records[index]["id"] for index in self.pools[name]]
                for name in self.pool_order
            },
            "queue_record_ids": {
                name: [self.records[index]["id"] for index in self.queues[name]]
                for name in self.pool_order
            },
            "positions": dict(self.positions),
            "credits": dict(self.credits),
            "pool_draw_counts": dict(self.pool_draw_counts),
            "record_draw_counts": {
                self.records[index]["id"]: count
                for index, count in self.record_draw_counts.items()
            },
            "pool_cycles_started": dict(self.pool_cycles_started),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if tuple(state.get("pool_order", ())) != self.pool_order:
            raise ValueError("sampler pool order does not match checkpoint")
        for name in self.pool_order:
            expected_ids = [self.records[index]["id"] for index in self.pools[name]]
            if list(state["pool_record_ids"][name]) != expected_ids:
                raise ValueError(f"sampler pool records changed for {name}")
        id_to_index = {record["id"]: index for index, record in enumerate(self.records)}
        self.queues = {
            name: [id_to_index[record_id] for record_id in state["queue_record_ids"][name]]
            for name in self.pool_order
        }
        self.positions = {name: int(state["positions"][name]) for name in self.pool_order}
        self.credits = {name: float(state["credits"][name]) for name in self.pool_order}
        self.pool_draw_counts = Counter(
            {name: int(value) for name, value in state["pool_draw_counts"].items()}
        )
        self.record_draw_counts = Counter(
            {
                id_to_index[record_id]: int(value)
                for record_id, value in state["record_draw_counts"].items()
            }
        )
        self.pool_cycles_started = Counter(
            {
                name: int(value)
                for name, value in state["pool_cycles_started"].items()
            }
        )

    def coverage_summary(self) -> dict[str, Any]:
        by_pool: dict[str, dict[str, Any]] = {}
        for name in self.pool_order:
            indexes = self.pools[name]
            seen = sum(1 for index in indexes if self.record_draw_counts[index] > 0)
            by_pool[name] = {
                "records": len(indexes),
                "seen_records": seen,
                "coverage": seen / len(indexes) if indexes else 0.0,
                "draws": self.pool_draw_counts[name],
                "cycles_started": self.pool_cycles_started[name],
            }
        total_seen = len(self.record_draw_counts)
        return {
            "records": len(self.records),
            "seen_records": total_seen,
            "coverage": total_seen / len(self.records),
            "draws": sum(self.pool_draw_counts.values()),
            "by_pool": by_pool,
        }


def pool_for_record(record: dict[str, Any]) -> str:
    family = str(record["task_family"])
    if family in CORE_FAMILIES:
        return "core"
    if family in KNOWN_FAMILIES:
        return "known"
    if family in CONTRAST_FAMILIES:
        return "contrast"
    return "replay"


def build_sampler(
    records: Sequence[dict[str, Any]],
    generator: torch.Generator,
    strategy: str,
    replay_weight: float,
    core_weight: float,
    known_weight: float,
    contrast_weight: float,
) -> WeightedEpochSampler:
    if strategy == "epoch":
        return WeightedEpochSampler(
            records,
            generator,
            {"all": 1.0},
            lambda _record: "all",
        )
    if strategy != "mixture":
        raise ValueError(f"unsupported sampling strategy: {strategy}")
    return WeightedEpochSampler(
        records,
        generator,
        {
            "replay": replay_weight,
            "core": core_weight,
            "known": known_weight,
            "contrast": contrast_weight,
        },
        pool_for_record,
    )


def select_fixed_train_records(
    records: Sequence[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(records):
        return list(records)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_family[str(record["task_family"])].append(record)
    for family_records in by_family.values():
        family_records.sort(key=lambda record: str(record["id"]))
    selected: list[dict[str, Any]] = []
    offsets = Counter()
    families = sorted(by_family)
    while len(selected) < limit:
        added = False
        for family in families:
            index = offsets[family]
            if index < len(by_family[family]):
                selected.append(by_family[family][index])
                offsets[family] += 1
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
    return selected


@torch.no_grad()
def evaluate_all_records(
    model: torch.nn.Module,
    records: Sequence[dict[str, Any]],
    pad_token_id: int,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    weighted_loss = 0.0
    supervised_tokens = 0
    for start in range(0, len(records), batch_size):
        inputs, labels = collate_records(records[start : start + batch_size], pad_token_id)
        token_count = int((labels != -100).sum())
        loss = supervised_loss(model, inputs.to(device), labels.to(device))
        weighted_loss += float(loss.detach().cpu()) * token_count
        supervised_tokens += token_count
    model.train()
    if supervised_tokens == 0:
        raise ValueError("evaluation set contains no supervised tokens")
    return weighted_loss / supervised_tokens


@torch.no_grad()
def evaluate_behavior(
    model: torch.nn.Module,
    payload: dict[str, Any],
    tokenizer: BPETokenizer,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    for index, item in enumerate(EVAL_ITEMS):
        prompt_ids = build_prompt_ids(
            tokenizer,
            item["question"],
            payload["special_token_ids"],
        )
        answer, stopped_on_eos = generate_answer(
            model,
            prompt_ids,
            payload["itos"],
            payload["special_token_ids"],
            max_new_tokens=30,
            temperature=0.3,
            top_k=1,
            seed=seed + index,
            device=device,
        )
        score = score_item(item, answer, stopped_on_eos)
        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "generated_answer": answer,
                "stopped_on_eos": stopped_on_eos,
                "metric": score["metric"],
                "passed": score["passed"],
            }
        )
    return summarize(results), results


def selection_score(behavior: Mapping[str, Any], val_loss: float) -> float:
    entity_passed = int(behavior["by_category"]["小说人物"]["passed"])
    total_passed = int(behavior["passed"])
    eos_count = int(behavior["eos_count"])
    return total_passed + entity_passed + 0.05 * eos_count - 0.01 * val_loss


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    best_score: float,
    history: Sequence[dict[str, Any]],
    generator: torch.Generator,
    signature_sha256: str,
    payload_summary: Mapping[str, Any],
    data_sha256: str,
    sampler: WeightedEpochSampler,
) -> dict[str, Any]:
    return build_checkpoint_payload(
        model,
        optimizer,
        step=step,
        best_metric=best_score,
        history=history,
        sampling_generator=generator,
        config_sha256=signature_sha256,
        extra={
            "payload_summary": dict(payload_summary),
            "data_sha256": data_sha256,
            "sampler_state": sampler.state_dict(),
            "selection_policy": "total_passed + entity_passed + 0.05*eos - 0.01*val_loss",
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--init-checkpoint", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--target-steps", type=int, default=DEFAULT_TARGET_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--gradient-clip", type=float, default=DEFAULT_GRADIENT_CLIP)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--train-eval-records", type=int, default=DEFAULT_TRAIN_EVAL_RECORDS)
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--sampling-strategy", choices=("epoch", "mixture"), default="mixture")
    parser.add_argument("--replay-weight", type=float, default=0.45)
    parser.add_argument("--core-weight", type=float, default=0.35)
    parser.add_argument("--known-weight", type=float, default=0.15)
    parser.add_argument("--contrast-weight", type=float, default=0.05)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "target_steps",
        "batch_size",
        "eval_interval",
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
    weights = (
        args.replay_weight,
        args.core_weight,
        args.known_weight,
        args.contrast_weight,
    )
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("sampling weights must be non-negative and sum positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    base_config = load_config(args.config)
    run_id = generate_run_id("sft-v5")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    loggers = configure_module_loggers(
        args.run_dir / "logs",
        run_id,
        {
            "data": "INFO",
            "sft": "INFO",
            "validation": "INFO",
            "checkpoint": "INFO",
            "orchestrator": "INFO",
        },
        max_bytes=int(base_config["logging"]["max_bytes"]),
        backup_count=int(base_config["logging"]["backup_count"]),
        console=bool(base_config["logging"]["console"]),
    )
    started = time.monotonic()
    try:
        device = select_device(args.device)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        generator = torch.Generator().manual_seed(args.seed + 1)
        payload = load_sft_payload(args.data)
        model = build_model(base_config, int(payload["vocab_size"])).to(device)
        payload_summary = validate_sft_payload(payload, model.config.block_size)
        data_sha256 = file_sha256(args.data)
        tokenizer = BPETokenizer.load(Path(payload["tokenizer_path"]))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=tuple(base_config["training"]["betas"]),
        )
        train_records = payload["train_records"]
        val_records = payload["val_records"]
        fixed_train_records = select_fixed_train_records(
            train_records,
            args.train_eval_records,
        )
        sampler = build_sampler(
            train_records,
            generator,
            args.sampling_strategy,
            args.replay_weight,
            args.core_weight,
            args.known_weight,
            args.contrast_weight,
        )
        signature = {
            "schema_version": "sft-v5-training-signature/v1",
            "base_model": base_config["model"],
            "data_sha256": data_sha256,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "betas": list(base_config["training"]["betas"]),
            "gradient_clip": args.gradient_clip,
            "sampling_strategy": args.sampling_strategy,
            "weights": {
                "replay": args.replay_weight,
                "core": args.core_weight,
                "known": args.known_weight,
                "contrast": args.contrast_weight,
            },
            "seed": args.seed,
            "evaluation_suite_version": EVALUATION_SUITE_VERSION,
        }
        signature_sha256 = canonical_json_sha256(signature)
        atomic_write_json(args.run_dir / "effective_config.json", {
            **signature,
            "target_steps": args.target_steps,
            "eval_interval": args.eval_interval,
            "eval_batch_size": args.eval_batch_size,
            "train_eval_records": args.train_eval_records,
            "device": str(device),
        })

        history: list[dict[str, Any]] = []
        start_step = 0
        best_score = float("-inf")
        initialization_policy = "fresh_optimizer_from_model_checkpoint"
        if args.resume:
            resumed = restore_checkpoint(
                args.resume,
                model,
                optimizer,
                generator,
                expected_config_sha256=signature_sha256,
                map_location=device,
                restore_cuda_rng=False,
            )
            if resumed.extra.get("data_sha256") != data_sha256:
                raise ValueError("resume checkpoint data does not match current SFT data")
            sampler.load_state_dict(resumed.extra["sampler_state"])
            start_step = resumed.step
            best_score = resumed.best_metric
            history = resumed.history
            initialization_policy = "full_resume_model_optimizer_rng_sampler"
            source_checkpoint = args.resume
        else:
            source_checkpoint = args.init_checkpoint or DEFAULT_INIT_CHECKPOINT
            initial = load_checkpoint(source_checkpoint, map_location=device)
            model.load_state_dict(initial["model_state_dict"], strict=True)
        if args.target_steps <= start_step:
            raise ValueError(
                f"target_steps must exceed resumed step {start_step}, got {args.target_steps}"
            )

        loggers["data"].info(
            "loaded data splits=%s families=%s fixed_train_eval=%d data_sha256=%s",
            payload_summary["split_counts"],
            payload_summary["task_family_counts"],
            len(fixed_train_records),
            data_sha256,
        )
        loggers["orchestrator"].info(
            "training start source=%s policy=%s start_step=%d target_step=%d "
            "device=%s strategy=%s",
            source_checkpoint,
            initialization_policy,
            start_step,
            args.target_steps,
            device,
            args.sampling_strategy,
        )

        pad_token_id = int(payload["special_token_ids"]["<PAD>"])
        for step in range(start_step, args.target_steps + 1):
            should_evaluate = (
                (step == start_step and not history)
                or (step > start_step and step % args.eval_interval == 0)
                or step == args.target_steps
            )
            if should_evaluate:
                train_loss = evaluate_all_records(
                    model,
                    fixed_train_records,
                    pad_token_id,
                    args.eval_batch_size,
                    device,
                )
                val_loss = evaluate_all_records(
                    model,
                    val_records,
                    pad_token_id,
                    args.eval_batch_size,
                    device,
                )
                behavior, behavior_results = evaluate_behavior(
                    model,
                    payload,
                    tokenizer,
                    device,
                    args.seed + 1000,
                )
                score = selection_score(behavior, val_loss)
                row = {
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "selection_score": score,
                    "behavior_passed": behavior["passed"],
                    "behavior_total": behavior["total"],
                    "entity_passed": behavior["by_category"]["小说人物"]["passed"],
                    "eos_count": behavior["eos_count"],
                    "coverage": sampler.coverage_summary(),
                    "behavior_results": behavior_results,
                }
                history.append(row)
                loggers["validation"].info(
                    "step=%d train_loss=%.6f full_val_loss=%.6f behavior=%d/%d "
                    "entity=%d/5 eos=%d/30 selection_score=%.4f coverage=%.3f",
                    step,
                    train_loss,
                    val_loss,
                    behavior["passed"],
                    behavior["total"],
                    behavior["by_category"]["小说人物"]["passed"],
                    behavior["eos_count"],
                    score,
                    sampler.coverage_summary()["coverage"],
                )
                payload_for_save = checkpoint_payload(
                    model,
                    optimizer,
                    step=step,
                    best_score=max(best_score, score),
                    history=history,
                    generator=generator,
                    signature_sha256=signature_sha256,
                    payload_summary=payload_summary,
                    data_sha256=data_sha256,
                    sampler=sampler,
                )
                milestone = atomic_save_checkpoint(
                    args.run_dir / "checkpoints" / f"step_{step:05d}.pt",
                    payload_for_save,
                )
                loggers["checkpoint"].info(
                    "saved milestone step=%d sha256=%s",
                    step,
                    milestone.sha256,
                )
                if score > best_score:
                    best_score = score
                    best = atomic_save_checkpoint(args.run_dir / "best.pt", payload_for_save)
                    loggers["checkpoint"].info(
                        "saved best step=%d selection_score=%.4f sha256=%s",
                        step,
                        score,
                        best.sha256,
                    )
            if step == args.target_steps:
                break

            inputs, labels, batch_indices = sampler.sample_batch(
                args.batch_size,
                pad_token_id,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = supervised_loss(model, inputs.to(device), labels.to(device))
            assert_finite_tensor(loss, "SFT v5 supervised loss")
            loss.backward()
            assert_finite_gradients(model.named_parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.gradient_clip,
            )
            optimizer.step()
            if (step + 1) % args.log_interval == 0:
                loggers["sft"].info(
                    "step=%d batch_loss=%.6f grad_norm=%.6f supervised_tokens=%d "
                    "batch_records=%d",
                    step + 1,
                    float(loss.detach().cpu()),
                    float(grad_norm.detach().cpu()),
                    int((labels != -100).sum()),
                    len(batch_indices),
                )

        latest_payload = checkpoint_payload(
            model,
            optimizer,
            step=args.target_steps,
            best_score=best_score,
            history=history,
            generator=generator,
            signature_sha256=signature_sha256,
            payload_summary=payload_summary,
            data_sha256=data_sha256,
            sampler=sampler,
        )
        latest = atomic_save_checkpoint(args.run_dir / "latest.pt", latest_payload)
        best_path = args.run_dir / "best.pt"
        if not best_path.exists():
            fallback_best = atomic_save_checkpoint(best_path, latest_payload)
            loggers["checkpoint"].warning(
                "best checkpoint absent after resume; saved latest as fallback sha256=%s",
                fallback_best.sha256,
            )
        loss_csv = args.report.with_name(args.report.stem + "_loss.csv")
        atomic_write_text(
            loss_csv,
            "step,train_loss,val_loss,selection_score,behavior_passed,entity_passed,eos_count\n"
            + "".join(
                f"{row['step']},{row['train_loss']},{row['val_loss']},"
                f"{row['selection_score']},{row['behavior_passed']},"
                f"{row['entity_passed']},{row['eos_count']}\n"
                for row in history
            ),
        )
        report = {
            "schema_version": "sft-v5-report/v1",
            "status": "complete",
            "run_id": run_id,
            "parameter_count": model.parameter_count(),
            "device": str(device),
            "source_checkpoint": str(source_checkpoint),
            "initialization_policy": initialization_policy,
            "data_path": str(args.data),
            "data_sha256": data_sha256,
            "signature_sha256": signature_sha256,
            "start_step": start_step,
            "target_step": args.target_steps,
            "optimizer_steps_this_run": args.target_steps - start_step,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "sampling_strategy": args.sampling_strategy,
            "sampler_coverage": sampler.coverage_summary(),
            "validation_policy": {
                "train": f"fixed stratified {len(fixed_train_records)} records",
                "validation": f"all {len(val_records)} records",
                "behavior": "fixed 30-item no-math diagnostic suite",
                "selection": "total_passed + entity_passed + 0.05*eos - 0.01*val_loss",
            },
            "best_selection_score": best_score,
            "history": history,
            "best_checkpoint": str(best_path),
            "latest_checkpoint": str(args.run_dir / "latest.pt"),
            "latest_checkpoint_sha256": latest.sha256,
            "loss_csv": str(loss_csv),
            "elapsed_seconds": time.monotonic() - started,
            "test_records_consumed": 0,
        }
        atomic_write_json(args.report, report)
        loggers["checkpoint"].info(
            "training complete step=%d best_score=%.4f latest_sha256=%s report=%s",
            args.target_steps,
            best_score,
            latest.sha256,
            args.report,
        )
        print(json.dumps({
            "status": "complete",
            "start_step": start_step,
            "target_step": args.target_steps,
            "best_selection_score": best_score,
            "final_behavior": f"{history[-1]['behavior_passed']}/30",
            "final_entity": f"{history[-1]['entity_passed']}/5",
            "final_eos": f"{history[-1]['eos_count']}/30",
            "full_val_loss": history[-1]["val_loss"],
            "coverage": sampler.coverage_summary()["coverage"],
            "report": str(args.report),
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        loggers["sft"].exception("SFT v5 training failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
