from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import hashlib
import json
import logging
import os

import torch
from logging.handlers import RotatingFileHandler


CORPUS_PATH = Path(os.getenv("SFT_CORPUS_PATH", "data/clean/doupo_stage3.txt"))
BASE_TENSOR_PATH = Path(
    os.getenv("SFT_BASE_TENSOR_PATH", "data/clean/doupo_stage3_tensors.pt")
)
DATASET_PATH = Path(os.getenv("SFT_DATASET_PATH", "data/sft/sft_seed_v1.jsonl"))
OUTPUT_PATH = Path(
    os.getenv("SFT_OUTPUT_PATH", "data/sft/sft_seed_v1_tensors.pt")
)
REPORT_PATH = Path(
    os.getenv(
        "SFT_REPORT_PATH",
        "reports/milestones/002_sft_dataset_v1/sft_data_report.json",
    )
)
SPECIAL_TOKENS = ("<BOS>", "<USER>", "<ASSISTANT>", "<EOS>", "<PAD>")
REQUIRED_FIELDS = {
    "id",
    "question",
    "answer",
    "evidence",
    "source_line",
    "topic",
    "split",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log_level(value: str) -> int:
    if value.upper() == "OFF":
        return logging.CRITICAL + 1
    return getattr(logging, value.upper(), logging.INFO)


def configure_logger(name: str, path: Path, env_name: str) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(parse_log_level(os.getenv(env_name, "INFO")))
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler = RotatingFileHandler(
        path,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if os.getenv("SFT_CONSOLE_LOG", "1") == "1":
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def configure_logging() -> dict[str, logging.Logger]:
    return {
        "data": configure_logger(
            "sft.data", Path("logs/sft_data.log"), "SFT_DATA_LOG_LEVEL"
        ),
        "validation": configure_logger(
            "sft.validation",
            Path("logs/sft_validation.log"),
            "SFT_VALIDATION_LOG_LEVEL",
        ),
        "output": configure_logger(
            "sft.output", Path("logs/sft_output.log"), "SFT_OUTPUT_LOG_LEVEL"
        ),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at dataset line {line_number}") from error
        if not isinstance(record, dict):
            raise TypeError(f"dataset line {line_number} must be a JSON object")
        records.append(record)
    if not records:
        raise ValueError("SFT dataset is empty")
    return records


def validate_records(
    records: list[dict[str, Any]], corpus_text: str, stoi: dict[str, int]
) -> None:
    corpus_lines = corpus_text.splitlines()
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    topic_splits: dict[str, set[str]] = {}

    for record in records:
        missing_fields = REQUIRED_FIELDS - record.keys()
        if missing_fields:
            raise ValueError(f"{record.get('id', '<unknown>')} missing {missing_fields}")
        if record["split"] not in {"train", "val"}:
            raise ValueError(f"{record['id']} has invalid split {record['split']}")
        if not record["question"].strip() or not record["answer"].strip():
            raise ValueError(f"{record['id']} has an empty question or answer")
        if record["id"] in seen_ids:
            raise ValueError(f"duplicate id {record['id']}")
        if record["question"] in seen_questions:
            raise ValueError(f"duplicate question {record['question']}")
        seen_ids.add(record["id"])
        seen_questions.add(record["question"])

        source_line = record["source_line"]
        if not isinstance(source_line, int) or not 1 <= source_line <= len(corpus_lines):
            raise ValueError(f"{record['id']} has invalid source_line {source_line}")
        if record["evidence"] not in corpus_lines[source_line - 1]:
            raise ValueError(
                f"{record['id']} evidence is absent from source line {source_line}"
            )

        missing_chars = sorted(
            set(record["question"] + record["answer"]) - set(stoi)
        )
        if missing_chars:
            raise ValueError(f"{record['id']} has out-of-vocabulary chars {missing_chars}")

        topic_splits.setdefault(record["topic"], set()).add(record["split"])

    leaked_topics = sorted(
        topic for topic, splits in topic_splits.items() if len(splits) > 1
    )
    if leaked_topics:
        raise ValueError(f"topics leak across train and val: {leaked_topics}")


def build_special_token_ids(vocab_size: int) -> dict[str, int]:
    return {
        token: vocab_size + offset
        for offset, token in enumerate(SPECIAL_TOKENS)
    }


def serialize_record(
    record: dict[str, Any],
    stoi: dict[str, int],
    special_token_ids: dict[str, int],
) -> dict[str, Any]:
    question_ids = [stoi[char] for char in record["question"]]
    answer_ids = [stoi[char] for char in record["answer"]]
    sequence = [
        special_token_ids["<BOS>"],
        special_token_ids["<USER>"],
        *question_ids,
        special_token_ids["<ASSISTANT>"],
        *answer_ids,
        special_token_ids["<EOS>"],
    ]
    assistant_index = 2 + len(question_ids)
    input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
    labels = torch.tensor(sequence[1:], dtype=torch.long)
    labels[:assistant_index] = -100
    return {
        "id": record["id"],
        "topic": record["topic"],
        "split": record["split"],
        "input_ids": input_ids,
        "labels": labels,
        "assistant_index": assistant_index,
        "sequence_length": len(sequence),
    }


def build_report(
    records: list[dict[str, Any]],
    prepared_records: list[dict[str, Any]],
    base_vocab_size: int,
    special_token_ids: dict[str, int],
) -> dict[str, Any]:
    split_counts = Counter(record["split"] for record in records)
    lengths = [record["sequence_length"] for record in prepared_records]
    masked_tokens = sum(int((record["labels"] == -100).sum()) for record in prepared_records)
    supervised_tokens = sum(int((record["labels"] != -100).sum()) for record in prepared_records)
    return {
        "milestone": "M002",
        "dataset_version": "sft_seed_v1",
        "dataset_path": str(DATASET_PATH),
        "output_path": str(OUTPUT_PATH),
        "corpus_path": str(CORPUS_PATH),
        "record_count": len(records),
        "train_count": split_counts["train"],
        "val_count": split_counts["val"],
        "topic_count": len({record["topic"] for record in records}),
        "topic_split_leakage": False,
        "evidence_verified": True,
        "base_vocab_size": base_vocab_size,
        "extended_vocab_size": base_vocab_size + len(SPECIAL_TOKENS),
        "special_token_ids": special_token_ids,
        "loss_policy": "assistant answer and EOS only; prompt labels use -100",
        "min_sequence_length": min(lengths),
        "max_sequence_length": max(lengths),
        "mean_sequence_length": sum(lengths) / len(lengths),
        "masked_label_count": masked_tokens,
        "supervised_label_count": supervised_tokens,
        "dataset_sha256": sha256_file(DATASET_PATH),
        "corpus_sha256": sha256_file(CORPUS_PATH),
    }


def main() -> None:
    loggers = configure_logging()
    data_logger = loggers["data"]
    validation_logger = loggers["validation"]
    output_logger = loggers["output"]
    try:
        data_logger.info("loading corpus path=%s", CORPUS_PATH)
        corpus_text = CORPUS_PATH.read_text(encoding="utf-8")
        payload = torch.load(BASE_TENSOR_PATH, map_location="cpu", weights_only=False)
        records = load_jsonl(DATASET_PATH)
        stoi = payload["stoi"]
        base_vocab_size = payload["vocab_size"]
        data_logger.info(
            "loaded records=%d corpus_chars=%d base_vocab=%d",
            len(records),
            len(corpus_text),
            base_vocab_size,
        )

        validate_records(records, corpus_text, stoi)
        validation_logger.info(
            "validation passed records=%d evidence=exact topic_leakage=none",
            len(records),
        )

        special_token_ids = build_special_token_ids(base_vocab_size)
        prepared_records = [
            serialize_record(record, stoi, special_token_ids) for record in records
        ]
        train_records = [
            record for record in prepared_records if record["split"] == "train"
        ]
        val_records = [
            record for record in prepared_records if record["split"] == "val"
        ]
        report = build_report(
            records, prepared_records, base_vocab_size, special_token_ids
        )

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "train_records": train_records,
                "val_records": val_records,
                "base_vocab_size": base_vocab_size,
                "vocab_size": report["extended_vocab_size"],
                "stoi": stoi,
                "itos": payload["itos"],
                "special_token_ids": special_token_ids,
                "ignore_index": -100,
            },
            OUTPUT_PATH,
        )
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_logger.info(
            "wrote tensors=%s report=%s train=%d val=%d vocab=%d",
            OUTPUT_PATH,
            REPORT_PATH,
            len(train_records),
            len(val_records),
            report["extended_vocab_size"],
        )
    except Exception:
        validation_logger.exception("SFT data preparation failed")
        raise


if __name__ == "__main__":
    main()
