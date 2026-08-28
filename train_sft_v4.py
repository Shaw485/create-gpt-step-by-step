"""Run a supervised fine-tuning smoke test on the v4 GPT checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time
from typing import Any, Sequence

import torch

from gpt_model_v4 import GPTConfig, GPTLanguageModelV4
from train_pretrain_v4 import load_config
from training_runtime import (
    atomic_save_checkpoint,
    atomic_write_json,
    build_checkpoint_payload,
    canonical_json_sha256,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
)


DEFAULT_CONFIG_PATH = Path("configs/local_m4_8m_continue_6000.json")
DEFAULT_DATA_PATH = Path("data/cloud_v4/sft_v4_ai_training_ready_tensors.pt")
DEFAULT_INIT_CHECKPOINT = Path("runs/pretrain_v4_m4_continue6000/best.pt")
DEFAULT_RUN_DIR = Path("runs/sft_v4_smoke20")
DEFAULT_REPORT_PATH = Path(
    "reports/milestones/009_v4_sft_smoke20/sft_v4_smoke20_report.json"
)
DEFAULT_MAX_STEPS = 20
DEFAULT_EVAL_INTERVAL = 5
DEFAULT_EVAL_BATCHES = 4
DEFAULT_MICRO_BATCH_SIZE = 1
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_WEIGHT_DECAY = 0.05
DEFAULT_GRADIENT_CLIP = 1.0
DEFAULT_SEED = 42


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device(
            "mps"
            if torch.backends.mps.is_built() and torch.backends.mps.is_available()
            else "cpu"
        )
    if requested == "mps":
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device: {requested}")


def load_sft_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "train_records",
        "val_records",
        "test_records",
        "vocab_size",
        "special_token_ids",
        "itos",
        "ignore_index",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"SFT tensor payload is missing keys: {missing}")
    if int(payload["ignore_index"]) != -100:
        raise ValueError("SFT payload must use -100 as ignore_index")
    return payload


def validate_sft_payload(payload: dict[str, Any], block_size: int) -> dict[str, Any]:
    train_records = list(payload["train_records"])
    val_records = list(payload["val_records"])
    test_records = list(payload["test_records"])
    if not train_records or not val_records or not test_records:
        raise ValueError("train, val and test splits must all be non-empty")

    ids_by_split = {
        "train": {record["id"] for record in train_records},
        "val": {record["id"] for record in val_records},
        "test": {record["id"] for record in test_records},
    }
    if ids_by_split["train"] & ids_by_split["val"]:
        raise ValueError("train and val record IDs overlap")
    if ids_by_split["train"] & ids_by_split["test"]:
        raise ValueError("train and test record IDs overlap")
    if ids_by_split["val"] & ids_by_split["test"]:
        raise ValueError("val and test record IDs overlap")

    all_records = train_records + val_records + test_records
    lengths = [int(record["sequence_length"]) for record in all_records]
    if max(lengths) > block_size:
        raise ValueError("an SFT sequence exceeds model context window")
    supervised_tokens = sum(int((record["labels"] != -100).sum()) for record in all_records)
    if supervised_tokens <= 0:
        raise ValueError("SFT payload contains no supervised tokens")

    return {
        "split_counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "task_family_counts": dict(
            sorted(Counter(record["task_family"] for record in all_records).items())
        ),
        "min_sequence_length": min(lengths),
        "max_sequence_length": max(lengths),
        "mean_sequence_length": sum(lengths) / len(lengths),
        "supervised_tokens": supervised_tokens,
    }


def collate_records(
    records: Sequence[dict[str, Any]],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(record["input_ids"]) for record in records)
    inputs = torch.full((len(records), max_length), pad_token_id, dtype=torch.long)
    labels = torch.full((len(records), max_length), -100, dtype=torch.long)
    for row, record in enumerate(records):
        input_ids = record["input_ids"]
        record_labels = record["labels"]
        length = len(input_ids)
        inputs[row, :length] = input_ids
        labels[row, :length] = record_labels
    return inputs, labels


def sample_batch(
    records: Sequence[dict[str, Any]],
    batch_size: int,
    pad_token_id: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(0, len(records), (batch_size,), generator=generator)
    selected = [records[int(index)] for index in indices]
    return collate_records(selected, pad_token_id)


def build_model(config: dict[str, Any], vocab_size: int) -> GPTLanguageModelV4:
    model_config = GPTConfig(vocab_size=vocab_size, **config["model"])
    return GPTLanguageModelV4(model_config)


def load_model_checkpoint(
    model: GPTLanguageModelV4,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


def supervised_loss(
    model: GPTLanguageModelV4,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    _, loss = model(inputs, labels)
    if loss is None:
        raise RuntimeError("model did not return a supervised loss")
    return loss


@torch.no_grad()
def evaluate_records(
    model: GPTLanguageModelV4,
    records: Sequence[dict[str, Any]],
    pad_token_id: int,
    batch_size: int,
    eval_batches: int,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        inputs, labels = sample_batch(records, batch_size, pad_token_id, generator)
        loss = supervised_loss(model, inputs.to(device), labels.to(device))
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses)


def token_to_text(itos: dict[Any, str], token_id: int) -> str:
    return itos.get(token_id, itos.get(str(token_id), ""))


def decode_ids(itos: dict[Any, str], token_ids: Sequence[int]) -> str:
    return "".join(token_to_text(itos, int(token_id)) for token_id in token_ids)


@torch.no_grad()
def generate_answer(
    model: GPTLanguageModelV4,
    prompt_ids: list[int],
    itos: dict[Any, str],
    special_token_ids: dict[str, int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
    device: torch.device,
) -> tuple[str, bool]:
    generator = torch.Generator().manual_seed(seed)
    current_ids = list(prompt_ids)
    generated_ids: list[int] = []
    eos_id = int(special_token_ids["<EOS>"])
    forbidden = [
        int(special_token_ids[token])
        for token in ("<BOS>", "<USER>", "<ASSISTANT>", "<PAD>")
    ]
    stopped_on_eos = False
    model.eval()
    for _ in range(max_new_tokens):
        context = current_ids[-model.config.block_size :]
        inputs = torch.tensor([context], dtype=torch.long, device=device)
        logits, _ = model(inputs)
        scores = logits[0, -1].float().cpu() / max(temperature, 1e-6)
        scores[forbidden] = float("-inf")
        if top_k > 0:
            values, indices = torch.topk(scores, min(top_k, scores.numel()))
            probabilities = torch.softmax(values, dim=-1)
            sampled = torch.multinomial(probabilities, 1, generator=generator)
            next_id = int(indices[sampled].item())
        else:
            probabilities = torch.softmax(scores, dim=-1)
            next_id = int(torch.multinomial(probabilities, 1, generator=generator).item())
        if next_id == eos_id:
            stopped_on_eos = True
            break
        current_ids.append(next_id)
        generated_ids.append(next_id)
    model.train()
    return "".join(token_to_text(itos, token_id) for token_id in generated_ids), stopped_on_eos


def monitor_answers(
    model: GPTLanguageModelV4,
    records: Sequence[dict[str, Any]],
    payload: dict[str, Any],
    device: torch.device,
    seed: int,
) -> list[dict[str, Any]]:
    outputs = []
    for index, record in enumerate(records):
        prompt_length = int(record["assistant_index"]) + 1
        prompt_ids = record["input_ids"][:prompt_length].tolist()
        question_ids = record["input_ids"][2 : int(record["assistant_index"])].tolist()
        generated, stopped_on_eos = generate_answer(
            model,
            prompt_ids,
            payload["itos"],
            payload["special_token_ids"],
            max_new_tokens=30,
            temperature=0.8,
            top_k=20,
            seed=seed + index,
            device=device,
        )
        outputs.append(
            {
                "id": record["id"],
                "split": record["split"],
                "task_family": record["task_family"],
                "question": decode_ids(payload["itos"], question_ids),
                "generated_answer": generated,
                "stopped_on_eos": stopped_on_eos,
            }
        )
    return outputs


def make_checkpoint(
    model: GPTLanguageModelV4,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    best_val_loss: float,
    history: Sequence[dict[str, Any]],
    generator: torch.Generator,
    config_sha256: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    return build_checkpoint_payload(
        model,
        optimizer,
        step=step,
        best_metric=best_val_loss,
        history=history,
        sampling_generator=generator,
        config_sha256=config_sha256,
        extra=extra,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--eval-batches", type=int, default=DEFAULT_EVAL_BATCHES)
    parser.add_argument("--micro-batch-size", type=int, default=DEFAULT_MICRO_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--gradient-clip", type=float, default=DEFAULT_GRADIENT_CLIP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = generate_run_id("sft-v4-smoke")
    base_config = load_config(args.config)
    run_config = {
        "schema_version": "sft-v4-smoke/v1",
        "base_config": base_config,
        "data_path": str(args.data),
        "init_checkpoint": str(args.init_checkpoint),
        "max_steps": args.max_steps,
        "eval_interval": args.eval_interval,
        "eval_batches": args.eval_batches,
        "micro_batch_size": args.micro_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "seed": args.seed,
        "device": args.device,
    }
    config_sha256 = canonical_json_sha256(run_config)
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
        batch_generator = torch.Generator().manual_seed(args.seed + 1)
        eval_generator = torch.Generator().manual_seed(args.seed + 2)
        payload = load_sft_payload(args.data)
        model = build_model(base_config, int(payload["vocab_size"])).to(device)
        init_checkpoint = load_model_checkpoint(model, args.init_checkpoint, device)
        payload_summary = validate_sft_payload(payload, model.config.block_size)
        pad_token_id = int(payload["special_token_ids"]["<PAD>"])
        train_records = payload["train_records"]
        val_records = payload["val_records"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=tuple(base_config["training"]["betas"]),
        )
        atomic_write_json(args.run_dir / "effective_config.json", run_config)
        loggers["data"].info(
            "loaded sft data records=%s max_length=%d data_sha256=%s",
            payload_summary["split_counts"],
            payload_summary["max_sequence_length"],
            file_sha256(args.data),
        )
        loggers["orchestrator"].info(
            "loaded init checkpoint path=%s source_step=%s device=%s",
            args.init_checkpoint,
            init_checkpoint.get("step"),
            device,
        )

        history: list[dict[str, Any]] = []
        best_val_loss = float("inf")
        for step in range(0, args.max_steps + 1):
            if step == 0 or step % args.eval_interval == 0 or step == args.max_steps:
                train_loss = evaluate_records(
                    model,
                    train_records,
                    pad_token_id,
                    args.micro_batch_size,
                    args.eval_batches,
                    eval_generator,
                    device,
                )
                val_loss = evaluate_records(
                    model,
                    val_records,
                    pad_token_id,
                    args.micro_batch_size,
                    args.eval_batches,
                    eval_generator,
                    device,
                )
                history.append(
                    {"step": step, "train_loss": train_loss, "val_loss": val_loss}
                )
                loggers["validation"].info(
                    "step=%d train_loss=%.6f val_loss=%.6f",
                    step,
                    train_loss,
                    val_loss,
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    result = atomic_save_checkpoint(
                        args.run_dir / "best.pt",
                        make_checkpoint(
                            model,
                            optimizer,
                            step=step,
                            best_val_loss=best_val_loss,
                            history=history,
                            generator=batch_generator,
                            config_sha256=config_sha256,
                            extra={"payload_summary": payload_summary},
                        ),
                    )
                    loggers["checkpoint"].info(
                        "saved best checkpoint step=%d sha256=%s",
                        result.step,
                        result.sha256,
                    )
            if step == args.max_steps:
                break

            inputs, labels = sample_batch(
                train_records,
                args.micro_batch_size,
                pad_token_id,
                batch_generator,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = supervised_loss(model, inputs.to(device), labels.to(device))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.gradient_clip,
            )
            optimizer.step()
            loggers["sft"].info(
                "step=%d batch_loss=%.6f grad_norm=%.6f supervised_tokens=%d",
                step + 1,
                float(loss.detach().cpu()),
                float(grad_norm.detach().cpu()),
                int((labels != -100).sum()),
            )

        latest = atomic_save_checkpoint(
            args.run_dir / "latest.pt",
            make_checkpoint(
                model,
                optimizer,
                step=args.max_steps,
                best_val_loss=best_val_loss,
                history=history,
                generator=batch_generator,
                config_sha256=config_sha256,
                extra={"payload_summary": payload_summary},
            ),
        )
        monitor_records = [train_records[0], val_records[0]]
        samples = monitor_answers(model, monitor_records, payload, device, args.seed + 100)
        report = {
            "schema_version": "sft-v4-smoke-report/v1",
            "run_id": run_id,
            "stage": "sft_v4_smoke20",
            "status": "complete",
            "steps": args.max_steps,
            "device": str(device),
            "parameter_count": model.parameter_count(),
            "data_path": str(args.data),
            "data_sha256": file_sha256(args.data),
            "init_checkpoint": str(args.init_checkpoint),
            "init_checkpoint_sha256": file_sha256(args.init_checkpoint),
            "payload_summary": payload_summary,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "micro_batch_size": args.micro_batch_size,
            "eval_interval": args.eval_interval,
            "eval_batches": args.eval_batches,
            "best_val_loss": best_val_loss,
            "initial_train_loss": history[0]["train_loss"],
            "initial_val_loss": history[0]["val_loss"],
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1]["val_loss"],
            "loss_history": history,
            "monitor_outputs": samples,
            "best_checkpoint": str(args.run_dir / "best.pt"),
            "latest_checkpoint": str(args.run_dir / "latest.pt"),
            "latest_checkpoint_sha256": latest.sha256,
            "elapsed_seconds": time.monotonic() - started,
            "test_records_consumed": 0,
        }
        atomic_write_json(args.report, report)
        loggers["checkpoint"].info(
            "sft smoke complete latest=%s report=%s",
            args.run_dir / "latest.pt",
            args.report,
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "steps": report["steps"],
                    "device": report["device"],
                    "initial_val_loss": report["initial_val_loss"],
                    "final_val_loss": report["final_val_loss"],
                    "best_val_loss": report["best_val_loss"],
                    "report": str(args.report),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception:
        loggers["sft"].exception("SFT v4 smoke failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
