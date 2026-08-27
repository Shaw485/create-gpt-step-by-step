from collections import Counter
from pathlib import Path
from typing import Any
import json

import torch

from prepare_sft_data import (
    BASE_TENSOR_PATH,
    CORPUS_PATH,
    REQUIRED_FIELDS,
    build_special_token_ids,
    configure_logging,
    load_jsonl,
    serialize_record,
    sha256_file,
)


DATASET_PATH = Path("data/sft/sft_pilot50_v1.jsonl")
OUTPUT_PATH = Path("data/sft/sft_pilot50_v1_tensors.pt")
REPORT_PATH = Path(
    "reports/milestones/002a_sft_pilot50_v1/sft_data_report.json"
)
EXPECTED_SPLIT_COUNTS = {"train": 40, "val": 5, "test": 5}


def validate_pilot_records(
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
        if record["split"] not in EXPECTED_SPLIT_COUNTS:
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

    split_counts = Counter(record["split"] for record in records)
    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"expected split counts {EXPECTED_SPLIT_COUNTS}, got {dict(split_counts)}"
        )
    leaked_topics = sorted(
        topic for topic, splits in topic_splits.items() if len(splits) > 1
    )
    if leaked_topics:
        raise ValueError(f"topics leak across splits: {leaked_topics}")


def build_pilot_report(
    records: list[dict[str, Any]],
    prepared_records: list[dict[str, Any]],
    base_vocab_size: int,
    special_token_ids: dict[str, int],
) -> dict[str, Any]:
    split_counts = Counter(record["split"] for record in records)
    lengths = [record["sequence_length"] for record in prepared_records]
    return {
        "milestone": "M002a",
        "dataset_version": "sft_pilot50_v1",
        "record_count": len(records),
        "split_counts": dict(split_counts),
        "topic_count": len({record["topic"] for record in records}),
        "topic_split_leakage": False,
        "evidence_verified": True,
        "base_vocab_size": base_vocab_size,
        "extended_vocab_size": base_vocab_size + len(special_token_ids),
        "special_token_ids": special_token_ids,
        "loss_policy": "assistant answer and EOS only; prompt labels use -100",
        "test_policy": "test records are exported separately and never used for updates",
        "min_sequence_length": min(lengths),
        "max_sequence_length": max(lengths),
        "mean_sequence_length": sum(lengths) / len(lengths),
        "masked_label_count": sum(
            int((record["labels"] == -100).sum()) for record in prepared_records
        ),
        "supervised_label_count": sum(
            int((record["labels"] != -100).sum()) for record in prepared_records
        ),
        "dataset_sha256": sha256_file(DATASET_PATH),
        "corpus_sha256": sha256_file(CORPUS_PATH),
    }


def main() -> None:
    loggers = configure_logging()
    try:
        corpus_text = CORPUS_PATH.read_text(encoding="utf-8")
        payload = torch.load(BASE_TENSOR_PATH, map_location="cpu", weights_only=False)
        records = load_jsonl(DATASET_PATH)
        stoi = payload["stoi"]
        base_vocab_size = payload["vocab_size"]
        loggers["data"].info(
            "pilot loading records=%d corpus_chars=%d", len(records), len(corpus_text)
        )

        validate_pilot_records(records, corpus_text, stoi)
        loggers["validation"].info(
            "pilot validation passed train=40 val=5 test=5 evidence=exact"
        )

        special_token_ids = build_special_token_ids(base_vocab_size)
        prepared_records = [
            serialize_record(record, stoi, special_token_ids) for record in records
        ]
        split_records = {
            split: [record for record in prepared_records if record["split"] == split]
            for split in EXPECTED_SPLIT_COUNTS
        }
        report = build_pilot_report(
            records, prepared_records, base_vocab_size, special_token_ids
        )

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "train_records": split_records["train"],
                "val_records": split_records["val"],
                "test_records": split_records["test"],
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
        loggers["output"].info(
            "pilot wrote tensors=%s report=%s", OUTPUT_PATH, REPORT_PATH
        )
    except Exception:
        loggers["validation"].exception("pilot SFT preparation failed")
        raise


if __name__ == "__main__":
    main()
