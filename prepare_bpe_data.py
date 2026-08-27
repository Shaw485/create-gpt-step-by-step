"""Learn BPE rules and encode the complete cleaned pretraining corpus."""

from __future__ import annotations

import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import time

import torch

from bpe_tokenizer import BPETokenizer, learn_bpe


SOURCE_PATH = Path(os.getenv("BPE_SOURCE_PATH", "data/clean/doupo_stage3.txt"))
CHAR_TENSOR_PATH = Path(
    os.getenv("BPE_CHAR_TENSOR_PATH", "data/clean/doupo_stage3_tensors.pt")
)
OUTPUT_DIR = Path(os.getenv("BPE_OUTPUT_DIR", "data/bpe"))
TOKENIZER_PATH = OUTPUT_DIR / os.getenv("BPE_TOKENIZER_NAME", "tokenizer_v1.json")
TENSOR_PATH = OUTPUT_DIR / os.getenv("BPE_TENSOR_NAME", "doupo_bpe_v1_tensors.pt")
METRICS_PATH = OUTPUT_DIR / os.getenv("BPE_METRICS_NAME", "bpe_v1_metrics.json")
NUM_MERGES = int(os.getenv("BPE_NUM_MERGES", "2000"))
TRAIN_CHARS = int(os.getenv("BPE_TRAIN_CHARS", "750000"))
SAMPLE_CHUNKS = int(os.getenv("BPE_SAMPLE_CHUNKS", "128"))
MIN_FREQUENCY = int(os.getenv("BPE_MIN_FREQUENCY", "3"))
TRAIN_RATIO = float(os.getenv("BPE_TRAIN_RATIO", "0.9"))


def configure_logging() -> None:
    """Create independent rotating logs for learning, encoding and validation."""
    Path("logs").mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    console_enabled = os.getenv("BPE_LOG_CONSOLE", "1") == "1"
    for suffix in ("learning", "encoding", "validation"):
        name = f"bpe.{suffix}"
        logger = logging.getLogger(name)
        logger.handlers.clear()
        level_name = os.getenv(f"BPE_LOG_{suffix.upper()}_LEVEL", "INFO")
        level = getattr(logging, level_name.upper(), logging.INFO)
        logger.setLevel(level)
        logger.propagate = False
        file_handler = RotatingFileHandler(
            f"logs/bpe_{suffix}.log", maxBytes=1_000_000, backupCount=3
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        if console_enabled:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evenly_sample_sequences(
    text: str, target_chars: int, chunk_count: int
) -> list[str]:
    """Take deterministic chunks spread from the start to end of the corpus."""
    if target_chars <= 0 or target_chars >= len(text):
        return [text]
    chunk_count = max(1, min(chunk_count, target_chars))
    chunk_size = max(1, target_chars // chunk_count)
    max_start = max(0, len(text) - chunk_size)
    starts = [
        round(index * max_start / max(1, chunk_count - 1))
        for index in range(chunk_count)
    ]
    return [text[start : start + chunk_size] for start in starts]


def encode_complete_corpus(
    text: str,
    tokenizer: BPETokenizer,
    logger: logging.Logger,
) -> list[int]:
    """Encode line by line to keep peak memory bounded and report progress."""
    lines = text.splitlines(keepends=True)
    encoded: list[int] = []
    started = time.monotonic()
    report_every = max(1, len(lines) // 20)
    for index, line in enumerate(lines, start=1):
        encoded.extend(tokenizer.encode(line))
        if index % report_every == 0 or index == len(lines):
            logger.info(
                "encoding progress lines=%d/%d tokens=%d elapsed_seconds=%.1f",
                index,
                len(lines),
                len(encoded),
                time.monotonic() - started,
            )
    return encoded


def main() -> None:
    configure_logging()
    learning_logger = logging.getLogger("bpe.learning")
    encoding_logger = logging.getLogger("bpe.encoding")
    validation_logger = logging.getLogger("bpe.validation")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        text = SOURCE_PATH.read_text(encoding="utf-8")
        char_payload = torch.load(
            CHAR_TENSOR_PATH, map_location="cpu", weights_only=False
        )
    except Exception:
        learning_logger.exception(
            "input load failed source=%s char_tensor=%s",
            SOURCE_PATH,
            CHAR_TENSOR_PATH,
        )
        raise

    base_tokens = [char_payload["itos"][index] for index in range(char_payload["vocab_size"])]
    if set(base_tokens) != set(text):
        raise ValueError("character tensor vocabulary does not exactly match source text")

    sample_sequences = evenly_sample_sequences(text, TRAIN_CHARS, SAMPLE_CHUNKS)
    sampled_chars = sum(map(len, sample_sequences))
    learning_logger.info(
        "learning start source_chars=%d sampled_chars=%d chunks=%d base_vocab=%d requested_merges=%d min_frequency=%d",
        len(text),
        sampled_chars,
        len(sample_sequences),
        len(base_tokens),
        NUM_MERGES,
        MIN_FREQUENCY,
    )
    started = time.monotonic()

    def report_progress(
        learned: int, requested: int, frequency: int, token: str, vocab_size: int
    ) -> None:
        if learned <= 10 or learned % 100 == 0 or learned == requested:
            learning_logger.info(
                "merge=%d/%d frequency=%d token=%r vocab=%d elapsed_seconds=%.1f",
                learned,
                requested,
                frequency,
                token,
                vocab_size,
                time.monotonic() - started,
            )

    tokenizer = learn_bpe(
        sequences=sample_sequences,
        base_tokens=base_tokens,
        num_merges=NUM_MERGES,
        min_frequency=MIN_FREQUENCY,
        progress_callback=report_progress,
    )
    tokenizer.save(TOKENIZER_PATH)
    learning_seconds = time.monotonic() - started
    learning_logger.info(
        "learning complete learned_merges=%d vocab=%d seconds=%.1f path=%s",
        len(tokenizer.merges),
        tokenizer.vocab_size,
        learning_seconds,
        TOKENIZER_PATH,
    )

    encoded = encode_complete_corpus(text, tokenizer, encoding_logger)
    data = torch.tensor(encoded, dtype=torch.long)
    split_index = int(len(data) * TRAIN_RATIO)
    train_data = data[:split_index].clone()
    val_data = data[split_index:].clone()
    compression_ratio = 1.0 - len(encoded) / len(text)

    tensor_payload = {
        "train_data": train_data,
        "val_data": val_data,
        "vocab_size": tokenizer.vocab_size,
        "stoi": {token: index for index, token in enumerate(tokenizer.tokens)},
        "itos": {index: token for index, token in enumerate(tokenizer.tokens)},
        "tokenizer_type": "character_seeded_bpe",
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_sha256": sha256(TOKENIZER_PATH),
        "num_merges": len(tokenizer.merges),
        "source_path": str(SOURCE_PATH),
        "source_sha256": sha256(SOURCE_PATH),
    }
    torch.save(tensor_payload, TENSOR_PATH)

    decoded = tokenizer.decode(encoded)
    round_trip_ok = decoded == text
    metrics = {
        "source_characters": len(text),
        "bpe_tokens": len(encoded),
        "compression_ratio": compression_ratio,
        "characters_per_token": len(text) / len(encoded),
        "base_vocab_size": len(base_tokens),
        "learned_merges": len(tokenizer.merges),
        "bpe_vocab_size": tokenizer.vocab_size,
        "sampled_characters": sampled_chars,
        "training_tokens": len(train_data),
        "validation_tokens": len(val_data),
        "round_trip_ok": round_trip_ok,
        "learning_seconds": learning_seconds,
        "source_sha256": tensor_payload["source_sha256"],
        "tokenizer_sha256": tensor_payload["tokenizer_sha256"],
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validation_logger.info(
        "validation round_trip=%s chars=%d tokens=%d compression=%.2f%% chars_per_token=%.3f train=%d val=%d",
        round_trip_ok,
        len(text),
        len(encoded),
        compression_ratio * 100,
        metrics["characters_per_token"],
        len(train_data),
        len(val_data),
    )
    validation_logger.info(
        "artifacts tokenizer=%s tensors=%s metrics=%s",
        TOKENIZER_PATH,
        TENSOR_PATH,
        METRICS_PATH,
    )
    if not round_trip_ok:
        raise RuntimeError("BPE round-trip validation failed")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
