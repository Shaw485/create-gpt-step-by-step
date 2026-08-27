"""Pretrain the handwritten GPT with BPE tokens from scratch."""

from __future__ import annotations

import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import time

import torch

from bpe_tokenizer import BPETokenizer
from train_gpt_stage3 import GPTLanguageModel, get_batch, set_global_seed


DATA_PATH = Path(os.getenv("BPE_PRETRAIN_DATA", "data/bpe/doupo_bpe_v1_tensors.pt"))
TOKENIZER_PATH = Path(os.getenv("BPE_PRETRAIN_TOKENIZER", "data/bpe/tokenizer_v1.json"))
LATEST_PATH = Path(os.getenv("BPE_PRETRAIN_LATEST", "checkpoints/bpe_pretrain_step10000.pt"))
BEST_PATH = Path(os.getenv("BPE_PRETRAIN_BEST", "checkpoints/bpe_pretrain_step10000_best.pt"))
REPORT_PATH = Path(os.getenv("BPE_PRETRAIN_REPORT", "reports/bpe_pretrain_report.json"))
PROMPT_FILE = Path(os.getenv("BPE_PRETRAIN_PROMPTS", "data/prompt10_eval.txt"))
MAX_STEPS = int(os.getenv("BPE_PRETRAIN_STEPS", "10000"))
EVAL_INTERVAL = int(os.getenv("BPE_PRETRAIN_EVAL_INTERVAL", "500"))
EVAL_ITERS = int(os.getenv("BPE_PRETRAIN_EVAL_ITERS", "20"))
BLOCK_SIZE = int(os.getenv("BPE_PRETRAIN_BLOCK_SIZE", "256"))
BATCH_SIZE = int(os.getenv("BPE_PRETRAIN_BATCH_SIZE", "4"))
EMBEDDING_DIM = int(os.getenv("BPE_PRETRAIN_EMBEDDING_DIM", "256"))
NUM_HEADS = int(os.getenv("BPE_PRETRAIN_HEADS", "8"))
NUM_LAYERS = int(os.getenv("BPE_PRETRAIN_LAYERS", "6"))
LEARNING_RATE = float(os.getenv("BPE_PRETRAIN_LR", "3e-4"))
WEIGHT_DECAY = float(os.getenv("BPE_PRETRAIN_WEIGHT_DECAY", "0.01"))
GRAD_CLIP = float(os.getenv("BPE_PRETRAIN_GRAD_CLIP", "1.0"))
MAX_NEW_CHARACTERS = int(os.getenv("BPE_PRETRAIN_MAX_NEW_CHARS", "30"))
TEMPERATURE = float(os.getenv("BPE_PRETRAIN_TEMPERATURE", "0.8"))
TOP_K = int(os.getenv("BPE_PRETRAIN_TOP_K", "20"))
SEED = int(os.getenv("BPE_PRETRAIN_SEED", "42"))
DEVICE = torch.device(os.getenv("BPE_PRETRAIN_DEVICE", "cpu"))


def parse_level(value: str) -> int:
    return getattr(logging, value.upper(), logging.INFO)


def configure_logger(name: str, filename: str, env_name: str) -> logging.Logger:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    level = parse_level(os.getenv(env_name, "INFO"))
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler = RotatingFileHandler(
        Path("logs") / filename,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if os.getenv("BPE_PRETRAIN_CONSOLE_LOG", "1") == "1":
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger


def configure_logging() -> dict[str, logging.Logger]:
    return {
        "data": configure_logger("bpe.pretrain.data", "bpe_pretrain_data.log", "BPE_PRETRAIN_DATA_LOG_LEVEL"),
        "step": configure_logger("bpe.pretrain.step", "bpe_pretrain_step.log", "BPE_PRETRAIN_STEP_LOG_LEVEL"),
        "validation": configure_logger("bpe.pretrain.validation", "bpe_pretrain_validation.log", "BPE_PRETRAIN_VALIDATION_LOG_LEVEL"),
        "generation": configure_logger("bpe.pretrain.generation", "bpe_pretrain_generation.log", "BPE_PRETRAIN_GENERATION_LOG_LEVEL"),
        "checkpoint": configure_logger("bpe.pretrain.checkpoint", "bpe_pretrain_checkpoint.log", "BPE_PRETRAIN_CHECKPOINT_LOG_LEVEL"),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def evaluate(
    model: GPTLanguageModel,
    data: torch.Tensor,
    generator: torch.Generator,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(EVAL_ITERS):
        inputs, targets = get_batch(data, BATCH_SIZE, BLOCK_SIZE)
        _, loss = model(inputs.to(DEVICE), targets.to(DEVICE))
        losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def generate_continuation(
    model: GPTLanguageModel,
    tokenizer: BPETokenizer,
    prompt: str,
    seed: int,
) -> tuple[str, int]:
    set_global_seed(seed, deterministic=True)
    ids = tokenizer.encode(prompt)
    prompt_tokens = len(ids)
    generated: list[int] = []
    model.eval()
    safety_limit = max(30, MAX_NEW_CHARACTERS * 2)
    for _ in range(safety_limit):
        context = ids[-BLOCK_SIZE:]
        tensor = torch.tensor([context], dtype=torch.long, device=DEVICE)
        logits, _ = model(tensor)
        logits = logits[:, -1, :] / max(TEMPERATURE, 1e-8)
        k = min(TOP_K, logits.shape[-1])
        if k > 0:
            values, indices = torch.topk(logits, k, dim=-1)
            sampled = torch.multinomial(torch.softmax(values, dim=-1), 1)
            next_id = int(indices.gather(1, sampled)[0, 0])
        else:
            next_id = int(torch.multinomial(torch.softmax(logits, dim=-1), 1)[0, 0])
        ids.append(next_id)
        generated.append(next_id)
        if len(tokenizer.decode(generated)) >= MAX_NEW_CHARACTERS:
            break
    model.train()
    return tokenizer.decode(generated)[:MAX_NEW_CHARACTERS], prompt_tokens


def checkpoint_payload(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    vocab_size: int,
    best_val_loss: float,
    history: list[dict],
    data_payload: dict,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "meta": {
            "stage": "bpe_pretrain",
            "step": step,
            "vocab_size": vocab_size,
            "block_size": BLOCK_SIZE,
            "batch_size": BATCH_SIZE,
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "best_val_loss": best_val_loss,
            "seed": SEED,
            "tokenizer_type": "character_seeded_bpe",
            "tokenizer_path": str(TOKENIZER_PATH),
            "tokenizer_sha256": data_payload["tokenizer_sha256"],
            "num_merges": data_payload["num_merges"],
        },
        "loss_history": history,
    }


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    loggers = configure_logging()
    set_global_seed(SEED, deterministic=True)
    generator = torch.Generator().manual_seed(SEED)
    started = time.time()
    try:
        data_payload = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
        tokenizer = BPETokenizer.load(TOKENIZER_PATH)
        train_data = data_payload["train_data"]
        val_data = data_payload["val_data"]
        vocab_size = int(data_payload["vocab_size"])
        if tokenizer.vocab_size != vocab_size:
            raise ValueError("tokenizer and training tensor vocabulary sizes differ")
        prompts = [
            line.strip()
            for line in PROMPT_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][:10]
        model = GPTLanguageModel(
            vocab_size=vocab_size,
            embedding_size=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            context_size=BLOCK_SIZE,
            num_layers=NUM_LAYERS,
        ).to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        loggers["data"].info(
            "loaded train_tokens=%d val_tokens=%d vocab=%d merges=%d device=%s params=%d block=%d batch=%d",
            len(train_data), len(val_data), vocab_size, len(tokenizer.merges), DEVICE,
            parameter_count, BLOCK_SIZE, BATCH_SIZE,
        )

        history: list[dict] = []
        prompt_history: list[dict] = []
        best_val_loss = float("inf")
        last_loss = float("nan")
        for step in range(MAX_STEPS + 1):
            inputs, targets = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            _, loss = model(inputs, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            last_loss = float(loss.detach())

            if step % 100 == 0:
                loggers["step"].info(
                    "step=%d/%d batch_loss=%.6f grad_norm=%.6f elapsed_seconds=%.1f",
                    step, MAX_STEPS, last_loss, float(grad_norm), time.time() - started,
                )

            if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
                train_loss = evaluate(model, train_data, generator)
                val_loss = evaluate(model, val_data, generator)
                entry = {"step": step, "train_loss": train_loss, "val_loss": val_loss}
                history.append(entry)
                loggers["validation"].info(
                    "step=%d train_loss=%.6f val_loss=%.6f",
                    step, train_loss, val_loss,
                )
                generations = []
                for index, prompt in enumerate(prompts):
                    continuation, prompt_tokens = generate_continuation(
                        model, tokenizer, prompt, SEED + step + index
                    )
                    generations.append({
                        "prompt": prompt,
                        "prompt_tokens": prompt_tokens,
                        "continuation": continuation,
                        "continuation_characters": len(continuation),
                    })
                    loggers["generation"].info(
                        "step=%d prompt_index=%d prompt=%r continuation=%r",
                        step, index + 1, prompt, continuation,
                    )
                prompt_history.append({"step": step, "samples": generations})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        BEST_PATH,
                        checkpoint_payload(
                            model, optimizer, step, vocab_size, best_val_loss,
                            history, data_payload,
                        ),
                    )
                    loggers["checkpoint"].info(
                        "best saved step=%d val_loss=%.6f path=%s",
                        step, val_loss, BEST_PATH,
                    )

        save_checkpoint(
            LATEST_PATH,
            checkpoint_payload(
                model, optimizer, MAX_STEPS, vocab_size, best_val_loss,
                history, data_payload,
            ),
        )
        report = {
            "stage": "bpe_pretrain",
            "steps": MAX_STEPS,
            "parameter_count": parameter_count,
            "device": str(DEVICE),
            "architecture": {
                "block_size": BLOCK_SIZE,
                "batch_size": BATCH_SIZE,
                "embedding_dim": EMBEDDING_DIM,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
                "vocab_size": vocab_size,
            },
            "best_val_loss": best_val_loss,
            "final_batch_loss": last_loss,
            "elapsed_seconds": time.time() - started,
            "loss_history": history,
            "prompt_history": prompt_history,
            "data_path": str(DATA_PATH),
            "data_sha256": sha256(DATA_PATH),
            "tokenizer_path": str(TOKENIZER_PATH),
            "tokenizer_sha256": sha256(TOKENIZER_PATH),
            "latest_checkpoint": str(LATEST_PATH),
            "best_checkpoint": str(BEST_PATH),
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        loggers["checkpoint"].info(
            "training complete latest=%s best=%s report=%s elapsed_seconds=%.1f",
            LATEST_PATH, BEST_PATH, REPORT_PATH, time.time() - started,
        )
    except Exception:
        loggers["step"].exception("BPE pretraining failed")
        raise


if __name__ == "__main__":
    main()
