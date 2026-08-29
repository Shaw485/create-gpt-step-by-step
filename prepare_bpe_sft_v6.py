"""Encode the audited SFT v6 JSONL dataset into multi-turn BPE tensors."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import torch

from bpe_tokenizer import BPETokenizer
from training_runtime import (
    atomic_write_json,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
    resolve_module_log_levels,
)


DEFAULT_DATASET = Path("data/sft/v6/sft_v6_10000.jsonl")
DEFAULT_TOKENIZER = Path("data/scaling_a/bpe_3000/tokenizer.json")
DEFAULT_OUTPUT = Path("data/sft/v6/sft_v6_bpe_tensors.pt")
DEFAULT_REPORT = Path("reports/milestones/018_sft_v6_10000/tensor_report.json")
DEFAULT_LOG_DIR = Path("logs/sft_v6_tensor")
SPECIAL_TOKENS = ("<BOS>", "<USER>", "<ASSISTANT>", "<EOS>", "<PAD>")
SPLIT_TARGETS = {
    "train": 8000,
    "val": 800,
    "public_diagnostic": 600,
    "sealed_test": 600,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is invalid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def require_special_ids(tokenizer: BPETokenizer) -> dict[str, int]:
    missing = [token for token in SPECIAL_TOKENS if token not in tokenizer.special_to_id]
    if missing:
        raise ValueError(f"tokenizer is missing required special tokens: {missing}")
    return {token: int(tokenizer.special_to_id[token]) for token in SPECIAL_TOKENS}


def serialize_messages(
    record: dict[str, Any],
    tokenizer: BPETokenizer,
    special_ids: dict[str, int],
) -> dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
        raise ValueError(f"record {record.get('id')} has invalid messages")

    sequence = [special_ids["<BOS>"]]
    supervised_targets = [False]
    assistant_spans: list[tuple[int, int]] = []
    expected_roles = ("user", "assistant")
    for index, message in enumerate(messages):
        expected_role = expected_roles[index % 2]
        if not isinstance(message, dict) or message.get("role") != expected_role:
            raise ValueError(f"record {record.get('id')} has invalid role order")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"record {record.get('id')} has empty message content")
        content_ids = tokenizer.encode(content)
        if expected_role == "user":
            sequence.append(special_ids["<USER>"])
            supervised_targets.append(False)
            sequence.extend(content_ids)
            supervised_targets.extend(False for _ in content_ids)
        else:
            sequence.append(special_ids["<ASSISTANT>"])
            supervised_targets.append(False)
            answer_start = len(sequence)
            sequence.extend(content_ids)
            supervised_targets.extend(True for _ in content_ids)
            sequence.append(special_ids["<EOS>"])
            supervised_targets.append(True)
            assistant_spans.append((answer_start, len(sequence)))

    input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
    labels = torch.tensor(sequence[1:], dtype=torch.long)
    target_mask = torch.tensor(supervised_targets[1:], dtype=torch.bool)
    labels[~target_mask] = -100
    if not bool((labels != -100).any()):
        raise ValueError(f"record {record.get('id')} has no supervised assistant tokens")
    return {
        "id": str(record["id"]),
        "primary_dimension": str(record["primary_dimension"]),
        "task_family": str(record["task_family"]),
        "split": str(record["split"]),
        "input_ids": input_ids,
        "labels": labels,
        "assistant_spans": assistant_spans,
        "assistant_turns": len(assistant_spans),
        "sequence_length": len(sequence),
    }


def prepare_payload(
    records: Sequence[dict[str, Any]],
    tokenizer: BPETokenizer,
    *,
    tokenizer_path: Path,
    tokenizer_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    split_counts = Counter(str(record.get("split")) for record in records)
    if dict(split_counts) != SPLIT_TARGETS:
        raise ValueError(f"unexpected split counts: {dict(split_counts)}")
    ids = [str(record.get("id")) for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate SFT record IDs")

    special_ids = require_special_ids(tokenizer)
    prepared = [serialize_messages(record, tokenizer, special_ids) for record in records]
    lengths = [int(record["sequence_length"]) for record in prepared]
    if max(lengths) > 512:
        raise ValueError("an SFT v6 sequence exceeds the 512-token context window")
    by_split = {
        split: [record for record in prepared if record["split"] == split]
        for split in SPLIT_TARGETS
    }
    payload = {
        "schema_version": "sft-v6-tensors/v1",
        "train_records": by_split["train"],
        "val_records": by_split["val"],
        "public_diagnostic_records": by_split["public_diagnostic"],
        "test_records": by_split["sealed_test"],
        "sealed_test_records": by_split["sealed_test"],
        "base_vocab_size": tokenizer.vocab_size,
        "vocab_size": tokenizer.vocab_size,
        "stoi": {token: index for index, token in enumerate(tokenizer.tokens)},
        "itos": {index: token for index, token in enumerate(tokenizer.tokens)},
        "special_token_ids": special_ids,
        "ignore_index": -100,
        "tokenizer_type": "character_seeded_bpe",
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_sha256,
        "test_policy": "sealed_test_is_stored_but_not_consumed_during_training_or_selection",
    }
    supervised_by_split = {
        split: sum(int((record["labels"] != -100).sum()) for record in values)
        for split, values in by_split.items()
    }
    report = {
        "schema_version": "sft-v6-tensor-report/v1",
        "status": "prepared",
        "records": len(prepared),
        "split_counts": dict(split_counts),
        "task_family_counts": dict(Counter(record["task_family"] for record in prepared)),
        "vocab_size": tokenizer.vocab_size,
        "special_token_ids": special_ids,
        "min_sequence_length": min(lengths),
        "max_sequence_length": max(lengths),
        "mean_sequence_length": sum(lengths) / len(lengths),
        "supervised_tokens_by_split": supervised_by_split,
        "supervised_tokens": sum(supervised_by_split.values()),
        "masked_tokens": sum(int((record["labels"] == -100).sum()) for record in prepared),
        "multiturn_records": sum(record["assistant_turns"] > 1 for record in prepared),
        "sealed_test_records_consumed": 0,
    }
    return payload, report


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = generate_run_id("sft-v6-tensor")
    loggers = configure_module_loggers(
        args.log_dir,
        run_id,
        resolve_module_log_levels(
            {"data": "INFO", "sft": "INFO", "validation": "INFO", "orchestrator": "INFO"}
        ),
        max_bytes=10 * 1024 * 1024,
        backup_count=5,
        console=True,
    )
    try:
        records = read_jsonl(args.dataset)
        tokenizer = BPETokenizer.load(args.tokenizer)
        loggers["data"].info(
            "loaded records=%d dataset_sha256=%s tokenizer_vocab=%d tokenizer_sha256=%s",
            len(records),
            file_sha256(args.dataset),
            tokenizer.vocab_size,
            file_sha256(args.tokenizer),
        )
        payload, report = prepare_payload(
            records,
            tokenizer,
            tokenizer_path=args.tokenizer,
            tokenizer_sha256=file_sha256(args.tokenizer),
        )
        atomic_torch_save(payload, args.output)
        report.update(
            {
                "run_id": run_id,
                "dataset_path": str(args.dataset),
                "dataset_sha256": file_sha256(args.dataset),
                "tokenizer_path": str(args.tokenizer),
                "tokenizer_sha256": file_sha256(args.tokenizer),
                "output_path": str(args.output),
                "output_sha256": file_sha256(args.output),
            }
        )
        atomic_write_json(args.report, report)
        loggers["validation"].info(
            "tensor gate passed splits=%s max_length=%d supervised=%d multiturn=%d sealed_consumed=0",
            report["split_counts"],
            report["max_sequence_length"],
            report["supervised_tokens"],
            report["multiturn_records"],
        )
        loggers["orchestrator"].info(
            "wrote tensors=%s sha256=%s report=%s",
            args.output,
            report["output_sha256"],
            args.report,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        loggers["validation"].exception("SFT v6 tensor preparation failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
