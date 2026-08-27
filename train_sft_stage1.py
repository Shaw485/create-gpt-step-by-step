from pathlib import Path
from typing import Any
import json
import logging
import os
import time

import torch
from logging.handlers import RotatingFileHandler

from evaluate_sft_baseline import generate_answer
from prepare_sft_data import parse_log_level, sha256_file
from train_gpt_stage3 import GPTLanguageModel, set_global_seed


DATA_PATH = Path(
    os.getenv("SFT_DATA_PATH", "data/sft/sft_pilot50_v1_tensors.pt")
)
INIT_CHECKPOINT_PATH = Path(
    os.getenv(
        "SFT_INIT_CHECKPOINT_PATH",
        "checkpoints/archive/sft_stage1_init_pre_sft.pt",
    )
)
LATEST_CHECKPOINT_PATH = Path(
    os.getenv("SFT_LATEST_CHECKPOINT_PATH", "checkpoints/sft_stage1_smoke20.pt")
)
BEST_CHECKPOINT_PATH = Path(
    os.getenv(
        "SFT_BEST_CHECKPOINT_PATH", "checkpoints/sft_stage1_smoke20_best.pt"
    )
)
REPORT_PATH = Path(
    os.getenv(
        "SFT_REPORT_PATH",
        "reports/milestones/003a_sft_smoke20/sft_smoke20_report.json",
    )
)
STAGE = os.getenv("SFT_STAGE", "sft_stage1_smoke20")
MILESTONE = os.getenv("SFT_MILESTONE", "M003a")
MAX_STEPS = int(os.getenv("SFT_MAX_STEPS", "20"))
EVAL_INTERVAL = int(os.getenv("SFT_EVAL_INTERVAL", "5"))
BATCH_SIZE = int(os.getenv("SFT_BATCH_SIZE", "4"))
LEARNING_RATE = float(os.getenv("SFT_LEARNING_RATE", "1e-4"))
WEIGHT_DECAY = float(os.getenv("SFT_WEIGHT_DECAY", "0.01"))
GRAD_CLIP = float(os.getenv("SFT_GRAD_CLIP", "1.0"))
SEED = int(os.getenv("SFT_SEED", "42"))
DEVICE = torch.device(os.getenv("SFT_DEVICE", "cpu"))


def configure_logger(name: str, file_name: str, env_name: str) -> logging.Logger:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(parse_log_level(os.getenv(env_name, "INFO")))
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler = RotatingFileHandler(
        Path("logs") / file_name,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if os.getenv("SFT_CONSOLE_LOG", "1") == "1":
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    logger.propagate = False
    return logger


def configure_logging() -> dict[str, logging.Logger]:
    return {
        "data": configure_logger("sft.data", "sft_train_data.log", "SFT_DATA_LOG_LEVEL"),
        "train": configure_logger(
            "sft.train", "sft_train_step.log", "SFT_TRAIN_LOG_LEVEL"
        ),
        "validation": configure_logger(
            "sft.validation", "sft_train_validation.log", "SFT_VALIDATION_LOG_LEVEL"
        ),
        "checkpoint": configure_logger(
            "sft.checkpoint", "sft_train_checkpoint.log", "SFT_CHECKPOINT_LOG_LEVEL"
        ),
    }


def build_model(meta: dict[str, Any]) -> GPTLanguageModel:
    return GPTLanguageModel(
        vocab_size=int(meta["vocab_size"]),
        embedding_size=int(meta["embedding_dim"]),
        num_heads=int(meta["num_heads"]),
        context_size=int(meta["block_size"]),
        num_layers=int(meta["num_layers"]),
    )


def load_training_splits(payload: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    train_records = payload["train_records"]
    val_records = payload["val_records"]
    if not train_records or not val_records:
        raise ValueError("train and validation records must both be non-empty")
    train_ids = {record["id"] for record in train_records}
    val_ids = {record["id"] for record in val_records}
    if train_ids & val_ids:
        raise ValueError("train and validation record IDs overlap")
    return train_records, val_records


def collate_records(
    records: list[dict[str, Any]], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(record["input_ids"]) for record in records)
    inputs = torch.full(
        (len(records), max_length), pad_token_id, dtype=torch.long
    )
    labels = torch.full((len(records), max_length), -100, dtype=torch.long)
    for row, record in enumerate(records):
        length = len(record["input_ids"])
        inputs[row, :length] = record["input_ids"]
        labels[row, :length] = record["labels"]
    return inputs, labels


def sample_training_batch(
    records: list[dict[str, Any]],
    batch_size: int,
    pad_token_id: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(
        0, len(records), (batch_size,), generator=generator
    ).tolist()
    selected = [records[index] for index in indices]
    return collate_records(selected, pad_token_id)


def supervised_loss(
    model: GPTLanguageModel,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    logits, _ = model(inputs)
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, model.vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
        reduction=reduction,
    )


@torch.no_grad()
def evaluate_records(
    model: GPTLanguageModel,
    records: list[dict[str, Any]],
    pad_token_id: int,
    batch_size: int,
) -> float:
    model.eval()
    total_loss = 0.0
    supervised_tokens = 0
    for start in range(0, len(records), batch_size):
        inputs, labels = collate_records(records[start : start + batch_size], pad_token_id)
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        loss_sum = supervised_loss(model, inputs, labels, reduction="sum")
        total_loss += float(loss_sum)
        supervised_tokens += int((labels != -100).sum())
    if supervised_tokens == 0:
        raise ValueError("evaluation records contain no supervised labels")
    model.train()
    return total_loss / supervised_tokens


def checkpoint_payload(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    init_meta: dict[str, Any],
    step: int,
    best_val_loss: float,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    meta = dict(init_meta)
    meta.update(
        {
            "stage": STAGE,
            "sft_steps": step,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "best_val_loss": best_val_loss,
            "seed": SEED,
            "test_records_consumed": 0,
        }
    )
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "meta": meta,
        "loss_history": history,
    }


def save_checkpoint(
    path: Path,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    init_meta: dict[str, Any],
    step: int,
    best_val_loss: float,
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            model, optimizer, init_meta, step, best_val_loss, history
        ),
        path,
    )


def monitor_answers(
    model: GPTLanguageModel,
    records: list[dict[str, Any]],
    data_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    outputs = []
    for index, record in enumerate(records):
        prompt_length = record["assistant_index"] + 1
        prompt_ids = record["input_ids"][:prompt_length].tolist()
        answer, _, stopped_on_eos = generate_answer(
            model=model,
            prompt_ids=prompt_ids,
            itos=data_payload["itos"],
            special_token_ids=data_payload["special_token_ids"],
            max_new_tokens=30,
            temperature=0.8,
            top_k=20,
            seed=SEED + index,
        )
        outputs.append(
            {
                "id": record["id"],
                "generated_answer": answer,
                "stopped_on_eos": stopped_on_eos,
            }
        )
    return outputs


def main() -> None:
    loggers = configure_logging()
    set_global_seed(SEED, deterministic=True)
    generator = torch.Generator().manual_seed(SEED)
    try:
        if MAX_STEPS <= 0 or BATCH_SIZE <= 0:
            raise ValueError("MAX_STEPS and BATCH_SIZE must be positive")
        data_payload = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
        init_checkpoint = torch.load(
            INIT_CHECKPOINT_PATH, map_location="cpu", weights_only=False
        )
        train_records, val_records = load_training_splits(data_payload)
        pad_token_id = data_payload["special_token_ids"]["<PAD>"]
        model = build_model(init_checkpoint["meta"]).to(DEVICE)
        model.load_state_dict(init_checkpoint["model_state_dict"])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        loggers["data"].info(
            "loaded train=%d val=%d test_consumed=0 batch=%d device=%s",
            len(train_records),
            len(val_records),
            BATCH_SIZE,
            DEVICE,
        )

        history: list[dict[str, Any]] = []
        initial_train_loss = evaluate_records(
            model, train_records, pad_token_id, BATCH_SIZE
        )
        initial_val_loss = evaluate_records(model, val_records, pad_token_id, BATCH_SIZE)
        best_val_loss = initial_val_loss
        history.append(
            {
                "step": 0,
                "train_loss": initial_train_loss,
                "val_loss": initial_val_loss,
            }
        )
        save_checkpoint(
            BEST_CHECKPOINT_PATH,
            model,
            optimizer,
            init_checkpoint["meta"],
            0,
            best_val_loss,
            history,
        )
        loggers["validation"].info(
            "step=0 train_loss=%.6f val_loss=%.6f", initial_train_loss, initial_val_loss
        )

        last_loss = float("nan")
        last_grad_norm = float("nan")
        started_at = time.time()
        for step in range(1, MAX_STEPS + 1):
            inputs, labels = sample_training_batch(
                train_records, BATCH_SIZE, pad_token_id, generator
            )
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = supervised_loss(model, inputs, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            last_loss = loss.detach().item()
            last_grad_norm = grad_norm.detach().item()
            loggers["train"].info(
                "step=%d batch_loss=%.6f supervised_tokens=%d grad_norm=%.6f",
                step,
                last_loss,
                int((labels != -100).sum()),
                last_grad_norm,
            )

            if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
                train_loss = evaluate_records(
                    model, train_records, pad_token_id, BATCH_SIZE
                )
                val_loss = evaluate_records(model, val_records, pad_token_id, BATCH_SIZE)
                history.append(
                    {
                        "step": step,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    }
                )
                loggers["validation"].info(
                    "step=%d train_loss=%.6f val_loss=%.6f",
                    step,
                    train_loss,
                    val_loss,
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        BEST_CHECKPOINT_PATH,
                        model,
                        optimizer,
                        init_checkpoint["meta"],
                        step,
                        best_val_loss,
                        history,
                    )
                    loggers["checkpoint"].info(
                        "best checkpoint saved step=%d val_loss=%.6f path=%s",
                        step,
                        val_loss,
                        BEST_CHECKPOINT_PATH,
                    )

        save_checkpoint(
            LATEST_CHECKPOINT_PATH,
            model,
            optimizer,
            init_checkpoint["meta"],
            MAX_STEPS,
            best_val_loss,
            history,
        )
        monitor_records = [train_records[0], val_records[0]]
        monitors = monitor_answers(model, monitor_records, data_payload)
        report = {
            "milestone": MILESTONE,
            "stage": STAGE,
            "steps": MAX_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRAD_CLIP,
            "seed": SEED,
            "device": str(DEVICE),
            "train_records": len(train_records),
            "val_records": len(val_records),
            "test_records_consumed": 0,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "initial_train_loss": initial_train_loss,
            "initial_val_loss": initial_val_loss,
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1]["val_loss"],
            "best_val_loss": best_val_loss,
            "last_batch_loss": last_loss,
            "last_gradient_norm": last_grad_norm,
            "elapsed_seconds": time.time() - started_at,
            "loss_history": history,
            "monitor_outputs": monitors,
            "data_path": str(DATA_PATH),
            "data_sha256": sha256_file(DATA_PATH),
            "init_checkpoint": str(INIT_CHECKPOINT_PATH),
            "init_checkpoint_sha256": sha256_file(INIT_CHECKPOINT_PATH),
            "latest_checkpoint": str(LATEST_CHECKPOINT_PATH),
            "best_checkpoint": str(BEST_CHECKPOINT_PATH),
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        loggers["checkpoint"].info(
            "smoke complete latest=%s best=%s report=%s",
            LATEST_CHECKPOINT_PATH,
            BEST_CHECKPOINT_PATH,
            REPORT_PATH,
        )
    except Exception:
        loggers["train"].exception("SFT smoke training failed")
        raise


if __name__ == "__main__":
    main()
