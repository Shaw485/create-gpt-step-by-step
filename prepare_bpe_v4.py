"""Train the formal v4 BPE tokenizer and encode chapter splits with EOS."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import torch

from bpe_tokenizer import BPETokenizer, learn_bpe
from prepare_bpe_data import evenly_sample_sequences
from prepare_corpus_v4 import parse_complete_chapters


SPECIAL_TOKENS = ["<UNK>", "<BOS>", "<USER>", "<ASSISTANT>", "<EOS>", "<PAD>"]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp": datetime.fromtimestamp(
                    record.created, timezone.utc
                ).isoformat(timespec="milliseconds"),
                "level": record.levelname,
                "module": record.name,
                "message": record.getMessage(),
            },
            ensure_ascii=False,
        )


def configure_loggers(log_dir: Path) -> dict[str, logging.Logger]:
    log_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for module in ("learning", "encoding", "validation"):
        logger = logging.getLogger(f"bpe_v4.{module}")
        logger.handlers.clear()
        logger.propagate = False
        level_name = os.getenv(f"BPE_V4_{module.upper()}_LOG_LEVEL", "INFO")
        logger.setLevel(getattr(logging, level_name.upper(), logging.INFO))
        handler = RotatingFileHandler(
            log_dir / f"bpe_v4_{module}.jsonl",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        console = logging.StreamHandler()
        console.setFormatter(JsonFormatter())
        logger.addHandler(console)
        result[module] = logger
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temp = Path(name)
    try:
        torch.save(value, temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def split_chapter_texts(path: Path) -> list[str]:
    preamble, chapters = parse_complete_chapters(path.read_text(encoding="utf-8"))
    if preamble.strip():
        raise ValueError(f"unexpected non-empty split preamble: {path}")
    return [chapter.cleaned_text or chapter.source_text for chapter in chapters]


def encode_with_eos(
    chapter_texts: list[str],
    tokenizer: BPETokenizer,
    eos_id: int,
    logger: logging.Logger,
    split: str,
) -> torch.Tensor:
    encoded: list[int] = []
    started = time.monotonic()
    interval = max(1, len(chapter_texts) // 20)
    for index, chapter_text in enumerate(chapter_texts, start=1):
        encoded.extend(tokenizer.encode(chapter_text))
        encoded.append(eos_id)
        if index % interval == 0 or index == len(chapter_texts):
            logger.info(
                "split=%s chapters=%d/%d tokens=%d elapsed_seconds=%.2f",
                split,
                index,
                len(chapter_texts),
                len(encoded),
                time.monotonic() - started,
            )
    return torch.tensor(encoded, dtype=torch.long)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cloud_v4"))
    parser.add_argument("--num-merges", type=int, default=2000)
    parser.add_argument("--learn-characters", type=int, default=1_500_000)
    parser.add_argument("--sample-chunks", type=int, default=192)
    parser.add_argument("--min-frequency", type=int, default=3)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    args = parser.parse_args()
    loggers = configure_loggers(args.log_dir)
    started = time.monotonic()

    try:
        paths = {
            split: args.data_dir / f"{split}.txt"
            for split in ("train", "val", "test")
        }
        texts = {split: path.read_text(encoding="utf-8") for split, path in paths.items()}
        chapters = {split: split_chapter_texts(path) for split, path in paths.items()}
        full_alphabet = sorted(set().union(*(set(text) for text in texts.values())))
        train_alphabet = set(texts["train"])
        coverage_only = sorted(set(full_alphabet) - train_alphabet)
        samples = evenly_sample_sequences(
            texts["train"], args.learn_characters, args.sample_chunks
        )
        sampled_characters = sum(len(item) for item in samples)
        loggers["learning"].info(
            "start train_chars=%d sampled_chars=%d chunks=%d base_vocab=%d "
            "coverage_only_chars=%d requested_merges=%d",
            len(texts["train"]),
            sampled_characters,
            len(samples),
            len(full_alphabet),
            len(coverage_only),
            args.num_merges,
        )

        def progress(learned: int, requested: int, frequency: int, token: str, vocab: int) -> None:
            if learned <= 10 or learned % 100 == 0 or learned == requested:
                loggers["learning"].info(
                    "merge=%d/%d frequency=%d token_length=%d vocab=%d elapsed_seconds=%.2f",
                    learned,
                    requested,
                    frequency,
                    len(token),
                    vocab,
                    time.monotonic() - started,
                )

        learned = learn_bpe(
            sequences=samples,
            base_tokens=full_alphabet,
            num_merges=args.num_merges,
            min_frequency=args.min_frequency,
            progress_callback=progress,
        )
        tokenizer = learned.with_special_tokens(SPECIAL_TOKENS)
        tokenizer_path = args.data_dir / "tokenizer.json"
        atomic_json(tokenizer_path, tokenizer.to_dict())
        eos_id = tokenizer.special_to_id["<EOS>"]

        tensors = {}
        tensor_paths = {}
        for split in ("train", "val", "test"):
            tensor = encode_with_eos(
                chapters[split], tokenizer, eos_id, loggers["encoding"], split
            )
            tensor_path = args.data_dir / f"{split}_tokens.pt"
            atomic_torch_save(tensor_path, tensor)
            tensors[split] = tensor
            tensor_paths[split] = tensor_path

        split_reports = {}
        for split in ("train", "val", "test"):
            tensor = tensors[split]
            without_specials = tokenizer.decode(
                tensor.tolist(), skip_special_tokens=True
            )
            round_trip = without_specials == texts[split]
            eos_count = int((tensor == eos_id).sum().item())
            if not round_trip or eos_count != len(chapters[split]):
                raise RuntimeError(
                    f"{split} validation failed: round_trip={round_trip} "
                    f"eos={eos_count}/{len(chapters[split])}"
                )
            split_reports[split] = {
                "text_path": str(paths[split]),
                "text_sha256": sha256_file(paths[split]),
                "tensor_path": str(tensor_paths[split]),
                "tensor_sha256": sha256_file(tensor_paths[split]),
                "characters": len(texts[split]),
                "tokens": len(tensor),
                "chapter_count": len(chapters[split]),
                "eos_count": eos_count,
                "round_trip_without_specials": round_trip,
                "characters_per_token": len(texts[split]) / len(tensor),
            }
            loggers["validation"].info(
                "split=%s round_trip=%s chapters=%d eos=%d chars=%d tokens=%d",
                split,
                round_trip,
                len(chapters[split]),
                eos_count,
                len(texts[split]),
                len(tensor),
            )

        manifest = {
            "schema_version": "bpe-v4/v1",
            "status": "ready",
            "tokenizer_type": "character_seeded_bpe",
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "base_vocab_size": len(full_alphabet),
            "train_only_merge_learning": True,
            "validation_test_characters_used_for_merge_counts": False,
            "coverage_only_base_characters": coverage_only,
            "sampled_train_characters": sampled_characters,
            "requested_merges": args.num_merges,
            "learned_merges": len(learned.merges),
            "special_tokens": tokenizer.special_to_id,
            "vocab_size": tokenizer.vocab_size,
            "eos_policy": "append one EOS token after every parsed chapter section",
            "splits": split_reports,
            "elapsed_seconds": time.monotonic() - started,
        }
        manifest_path = args.data_dir / "token_manifest.json"
        atomic_json(manifest_path, manifest)
        (args.data_dir / "token_manifest.json.sha256").write_text(
            f"{sha256_file(manifest_path)}  token_manifest.json\n",
            encoding="utf-8",
        )
        loggers["validation"].info(
            "complete vocab=%d merges=%d eos_id=%d elapsed_seconds=%.2f",
            tokenizer.vocab_size,
            len(learned.merges),
            eos_id,
            manifest["elapsed_seconds"],
        )
    except Exception:
        loggers["validation"].exception("BPE v4 preparation failed")
        raise
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
