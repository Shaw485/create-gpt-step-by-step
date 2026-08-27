"""Encode the balanced 1,000-record SFT dataset with the learned BPE tokenizer."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

import torch

from bpe_tokenizer import BPETokenizer


DATASET_PATH = Path(os.getenv("BPE_SFT_DATASET", "data/sft/sft_balanced_v3.jsonl"))
TOKENIZER_PATH = Path(os.getenv("BPE_SFT_TOKENIZER", "data/bpe/tokenizer_v1.json"))
OUTPUT_PATH = Path(os.getenv("BPE_SFT_OUTPUT", "data/bpe/sft_balanced_v3_bpe_tensors.pt"))
REPORT_PATH = Path(os.getenv("BPE_SFT_REPORT", "reports/bpe_sft_data_report.json"))
SPECIAL_TOKENS = ("<BOS>", "<USER>", "<ASSISTANT>", "<EOS>", "<PAD>")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_logging() -> dict[str, logging.Logger]:
    Path("logs").mkdir(parents=True, exist_ok=True)
    result = {}
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    for suffix in ("data", "validation", "output"):
        logger = logging.getLogger(f"bpe.sft.{suffix}")
        logger.handlers.clear()
        logger.propagate = False
        level_name = os.getenv(f"BPE_SFT_{suffix.upper()}_LOG_LEVEL", "INFO")
        level = getattr(logging, level_name.upper(), logging.INFO)
        logger.setLevel(level)
        file_handler = RotatingFileHandler(
            f"logs/bpe_sft_{suffix}.log",
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        if os.getenv("BPE_SFT_CONSOLE_LOG", "1") == "1":
            console = logging.StreamHandler()
            console.setLevel(level)
            console.setFormatter(formatter)
            logger.addHandler(console)
        result[suffix] = logger
    return result


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}") from error
    return records


def serialize_record(
    record: dict,
    tokenizer: BPETokenizer,
    special_ids: dict[str, int],
) -> dict:
    question_ids = tokenizer.encode(record["question"])
    answer_ids = tokenizer.encode(record["answer"])
    sequence = [
        special_ids["<BOS>"],
        special_ids["<USER>"],
        *question_ids,
        special_ids["<ASSISTANT>"],
        *answer_ids,
        special_ids["<EOS>"],
    ]
    assistant_index = 2 + len(question_ids)
    input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
    labels = torch.tensor(sequence[1:], dtype=torch.long)
    labels[:assistant_index] = -100
    return {
        "id": record["id"],
        "topic": record["topic"],
        "task_family": record["task_family"],
        "split": record["split"],
        "input_ids": input_ids,
        "labels": labels,
        "assistant_index": assistant_index,
        "sequence_length": len(sequence),
    }


def main() -> None:
    loggers = configure_logging()
    try:
        tokenizer = BPETokenizer.load(TOKENIZER_PATH)
        records = load_jsonl(DATASET_PATH)
        loggers["data"].info(
            "loaded records=%d tokenizer_vocab=%d merges=%d",
            len(records), tokenizer.vocab_size, len(tokenizer.merges),
        )
        split_counts = Counter(record["split"] for record in records)
        if split_counts != {"train": 800, "val": 100, "test": 100}:
            raise ValueError(f"unexpected SFT split counts: {dict(split_counts)}")
        if len({record["id"] for record in records}) != len(records):
            raise ValueError("duplicate SFT record IDs")
        if len({record["question"] for record in records}) != len(records):
            raise ValueError("duplicate SFT questions")

        special_ids = {
            token: tokenizer.vocab_size + offset
            for offset, token in enumerate(SPECIAL_TOKENS)
        }
        prepared = [
            serialize_record(record, tokenizer, special_ids) for record in records
        ]
        if max(record["sequence_length"] for record in prepared) > 256:
            raise ValueError("a BPE SFT sequence exceeds the 256-token context")
        by_split = {
            split: [record for record in prepared if record["split"] == split]
            for split in ("train", "val", "test")
        }
        itos = {index: token for index, token in enumerate(tokenizer.tokens)}
        stoi = {token: index for index, token in enumerate(tokenizer.tokens)}
        payload = {
            "train_records": by_split["train"],
            "val_records": by_split["val"],
            "test_records": by_split["test"],
            "base_vocab_size": tokenizer.vocab_size,
            "vocab_size": tokenizer.vocab_size + len(SPECIAL_TOKENS),
            "stoi": stoi,
            "itos": itos,
            "special_token_ids": special_ids,
            "ignore_index": -100,
            "tokenizer_type": "character_seeded_bpe",
            "tokenizer_path": str(TOKENIZER_PATH),
            "tokenizer_sha256": sha256(TOKENIZER_PATH),
        }
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, OUTPUT_PATH)

        lengths = [record["sequence_length"] for record in prepared]
        report = {
            "stage": "bpe_sft_data",
            "records": len(records),
            "split_counts": dict(split_counts),
            "task_family_counts": dict(Counter(r["task_family"] for r in records)),
            "base_vocab_size": tokenizer.vocab_size,
            "extended_vocab_size": payload["vocab_size"],
            "special_token_ids": special_ids,
            "min_sequence_length": min(lengths),
            "max_sequence_length": max(lengths),
            "mean_sequence_length": sum(lengths) / len(lengths),
            "supervised_tokens": sum(int((r["labels"] != -100).sum()) for r in prepared),
            "masked_tokens": sum(int((r["labels"] == -100).sum()) for r in prepared),
            "dataset_sha256": sha256(DATASET_PATH),
            "tokenizer_sha256": sha256(TOKENIZER_PATH),
            "output_sha256": sha256(OUTPUT_PATH),
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        loggers["validation"].info(
            "validated train=800 val=100 test=100 min_length=%d max_length=%d mean_length=%.2f",
            min(lengths), max(lengths), report["mean_sequence_length"],
        )
        loggers["output"].info(
            "wrote tensors=%s report=%s sha256=%s",
            OUTPUT_PATH, REPORT_PATH, report["output_sha256"],
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception:
        loggers["validation"].exception("BPE SFT preparation failed")
        raise


if __name__ == "__main__":
    main()
