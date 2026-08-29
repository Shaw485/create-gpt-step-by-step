"""Encode isolated SFT v7 splits with the formal BPE 3000 tokenizer.

The routine training artifact contains only train and validation records.  The
public diagnostic artifact is written separately.  The formal blind-evaluation
split is deliberately outside this program's CLI and payload schemas.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import torch

from bpe_tokenizer import BPETokenizer
from training_runtime import (
    atomic_write_json,
    atomic_write_text,
    canonical_json_sha256,
    close_module_loggers,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
    resolve_module_log_levels,
)


DEFAULT_TRAIN = Path("data/sft/v7/train.jsonl")
DEFAULT_VAL = Path("data/sft/v7/val.jsonl")
DEFAULT_PUBLIC_DIAGNOSTIC = Path("data/sft/v7/public_diagnostic.jsonl")
DEFAULT_TOKENIZER = Path("data/scaling_a/bpe_3000/tokenizer.json")
DEFAULT_TOKEN_MANIFEST = Path("data/scaling_a/bpe_3000/token_manifest.json")
DEFAULT_DATASET_MANIFEST = Path("data/sft/v7/manifest.json")
DEFAULT_TRAIN_VAL_OUTPUT = Path("data/sft/v7/train_val_tensors.pt")
DEFAULT_PUBLIC_OUTPUT = Path("data/sft/v7/public_diagnostic_tensors.pt")
DEFAULT_REPORT = Path("reports/milestones/020_sft_v7_vertical/tensor_report.json")
DEFAULT_LOG_DIR = Path("logs/sft_v7_tensor")

MAX_SEQUENCE_LENGTH = 512
IGNORE_INDEX = -100

EXPECTED_SPECIAL_TOKEN_IDS = {
    "<UNK>": 7459,
    "<BOS>": 7460,
    "<USER>": 7461,
    "<ASSISTANT>": 7462,
    "<EOS>": 7463,
    "<PAD>": 7464,
}
EXPECTED_SPECIAL_TOKEN_ORDER = tuple(EXPECTED_SPECIAL_TOKEN_IDS)
EXPECTED_VOCAB_SIZE = 7465
EXPECTED_TOKENIZER_SHA256 = (
    "e70cf3dc0ed185a6b22ab7dc08b6a850eeb59864ba161dd156c644e003862822"
)
EXPECTED_MANIFEST_SHA256 = (
    "5d10245eac86e4dbafef908cb2d915bb1effcf61ad977b4de96d8d64d30809c7"
)
EXPECTED_DATASET_MANIFEST_SCHEMA = "sft-v7-vertical-manifest/v1"
EXPECTED_DATASET_RECORD_SCHEMA = "sft_v7_vertical/1.0"
EXPECTED_ROUTINE_SPLIT_COUNTS = {
    "train": 8000,
    "val": 800,
    "public_diagnostic": 600,
}
REQUIRED_BASE_CHECKPOINT = {
    "path": "runs/formal_pretrain_14m_bpe3000_6000/checkpoints/step_05750.pt",
    "sha256": "bfe4fec5e6045d4c06d22393e7c2079fdc03897be71829c9d9dcbaf0fcaf5c1e",
    "step": 5750,
    "config_canonical_sha256": (
        "faac0f759a5ce9cd5e827f95c511b7f9bbbda06f6e7642e5e0d90d5ec5635974"
    ),
    "token_manifest_sha256": EXPECTED_MANIFEST_SHA256,
    "parameter_count": 14_880_745,
}

_FORBIDDEN_SCOPE_TOKEN = re.compile(r"(?:^|[^a-z])(sealed|test)(?:$|[^a-z])")
_PLAINTEXT_EVALUATION_KEYS = {
    "answer",
    "content",
    "evidence",
    "messages",
    "prompt",
    "question",
    "reference_answer",
    "source_path",
    "text",
}


class SFTV7EncodingError(ValueError):
    """A data or provenance failure with a log-safe diagnostic code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _raise(code: str, message: str) -> None:
    raise SFTV7EncodingError(code, message)


def _scope_token_forbidden(value: str) -> bool:
    return bool(_FORBIDDEN_SCOPE_TOKEN.search(value.lower()))


def _assert_no_forbidden_scope(value: Any, *, location: str = "payload") -> None:
    """Reject metadata that could reconnect routine artifacts to a blind split."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if _scope_token_forbidden(key_text):
                _raise("forbidden_scope_metadata", f"{location} contains a forbidden field")
            _assert_no_forbidden_scope(item, location=f"{location}.{key_text}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_forbidden_scope(item, location=f"{location}[{index}]")
        return
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and _scope_token_forbidden(value):
        _raise("forbidden_scope_metadata", f"{location} contains a forbidden value")


def _copy_evaluation_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, Mapping) or not evaluation:
        _raise("missing_evaluation_metadata", "public record lacks evaluation metadata")

    def inspect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                normalized = str(key).strip().lower()
                if normalized in _PLAINTEXT_EVALUATION_KEYS:
                    _raise(
                        "plaintext_evaluation_metadata",
                        "evaluation metadata contains a plaintext field",
                    )
                inspect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                inspect(item)
        elif value is not None and not isinstance(value, (str, int, float, bool)):
            _raise("invalid_evaluation_metadata", "evaluation metadata is not JSON-compatible")

    inspect(evaluation)
    copied = deepcopy(dict(evaluation))
    _assert_no_forbidden_scope(copied, location="evaluation")
    return copied


def read_jsonl(path: Path, expected_split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SFTV7EncodingError(
                    "invalid_jsonl",
                    f"{path.name}:{line_number} is invalid JSON",
                ) from error
            if not isinstance(record, dict):
                _raise("invalid_record", f"{path.name}:{line_number} is not an object")
            if record.get("split") != expected_split:
                _raise("split_mismatch", f"{path.name}:{line_number} has the wrong split")
            records.append(record)
    if not records:
        _raise("empty_split", f"{path.name} contains no records")
    return records


def require_formal_special_ids(tokenizer: BPETokenizer) -> dict[str, int]:
    actual = {str(key): int(value) for key, value in tokenizer.special_to_id.items()}
    if actual != EXPECTED_SPECIAL_TOKEN_IDS:
        _raise("special_token_ids_mismatch", "tokenizer special-token IDs do not match BPE 3000")
    if tuple(tokenizer.special_tokens) != EXPECTED_SPECIAL_TOKEN_ORDER:
        _raise("special_token_order_mismatch", "tokenizer special-token order is not frozen")
    if tokenizer.vocab_size != EXPECTED_VOCAB_SIZE:
        _raise("vocab_size_mismatch", "tokenizer vocabulary size is not frozen")
    return dict(EXPECTED_SPECIAL_TOKEN_IDS)


def load_and_validate_formal_tokenizer(
    tokenizer_path: Path,
    manifest_path: Path,
) -> tuple[BPETokenizer, dict[str, int], dict[str, str]]:
    tokenizer_sha256 = file_sha256(tokenizer_path)
    manifest_sha256 = file_sha256(manifest_path)
    if tokenizer_sha256 != EXPECTED_TOKENIZER_SHA256:
        _raise("tokenizer_sha256_mismatch", "tokenizer SHA-256 is not the frozen BPE 3000 hash")
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        _raise("manifest_sha256_mismatch", "manifest SHA-256 is not the frozen BPE 3000 hash")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SFTV7EncodingError("invalid_manifest", "token manifest is invalid JSON") from error
    if not isinstance(manifest, dict):
        _raise("invalid_manifest", "token manifest root must be an object")

    # Deliberately inspect only tokenizer identity fields.  Paths under split
    # metadata are neither resolved nor opened by this encoder.
    if manifest.get("schema_version") != "bpe-v4/v1":
        _raise("manifest_schema_mismatch", "token manifest schema is not supported")
    if manifest.get("status") != "ready":
        _raise("manifest_not_ready", "token manifest is not ready")
    if manifest.get("tokenizer_sha256") != tokenizer_sha256:
        _raise("manifest_tokenizer_mismatch", "manifest does not bind the tokenizer")
    if int(manifest.get("vocab_size", -1)) != EXPECTED_VOCAB_SIZE:
        _raise("manifest_vocab_mismatch", "manifest vocabulary size is not frozen")
    if manifest.get("special_tokens") != EXPECTED_SPECIAL_TOKEN_IDS:
        _raise("manifest_special_tokens_mismatch", "manifest special-token IDs are not frozen")

    tokenizer = BPETokenizer.load(tokenizer_path)
    special_ids = require_formal_special_ids(tokenizer)
    return tokenizer, special_ids, {
        "tokenizer_sha256": tokenizer_sha256,
        "bpe_token_manifest_sha256": manifest_sha256,
        # Compatibility alias for callers that predate the dataset manifest.
        "manifest_sha256": manifest_sha256,
    }


def load_and_validate_dataset_manifest(
    dataset_manifest_path: Path,
    split_paths: Mapping[str, Path],
) -> dict[str, str]:
    """Bind routine JSONL inputs to the immutable builder manifest.

    Only train, validation and public metadata are inspected.  The blind split
    entry is never resolved, hashed or opened by this function.
    """

    try:
        manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SFTV7EncodingError(
            "invalid_dataset_manifest", "dataset manifest is invalid JSON"
        ) from error
    if not isinstance(manifest, Mapping):
        _raise("invalid_dataset_manifest", "dataset manifest root must be an object")
    if manifest.get("manifest_schema_version") != EXPECTED_DATASET_MANIFEST_SCHEMA:
        _raise("dataset_manifest_schema_mismatch", "dataset manifest schema changed")
    if manifest.get("record_schema_version") != EXPECTED_DATASET_RECORD_SCHEMA:
        _raise("dataset_record_schema_mismatch", "dataset record schema changed")
    split_files = manifest.get("split_files")
    if not isinstance(split_files, Mapping):
        _raise("dataset_manifest_splits_missing", "dataset manifest lacks split metadata")

    current_hashes: dict[str, str] = {}
    for split, expected_count in EXPECTED_ROUTINE_SPLIT_COUNTS.items():
        metadata = split_files.get(split)
        if not isinstance(metadata, Mapping):
            _raise("dataset_manifest_split_missing", f"dataset manifest lacks {split}")
        current_path = split_paths[split]
        declared_path = Path(str(metadata.get("path", "")))
        expected_path = (dataset_manifest_path.parent / declared_path).resolve()
        if current_path.resolve() != expected_path:
            _raise("dataset_manifest_path_mismatch", f"{split} path is not builder-bound")
        if int(metadata.get("count", -1)) != expected_count:
            _raise("dataset_manifest_count_mismatch", f"{split} count is not frozen")
        with current_path.open(encoding="utf-8") as handle:
            actual_count = sum(1 for line in handle if line.strip())
        if actual_count != expected_count:
            _raise("dataset_jsonl_count_mismatch", f"{split} JSONL count changed")
        declared_sha = str(metadata.get("sha256", ""))
        actual_sha = file_sha256(current_path)
        if declared_sha != actual_sha:
            _raise("dataset_manifest_sha_mismatch", f"{split} SHA-256 changed")
        current_hashes[split] = actual_sha
    return {
        "sft_dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        **{f"{split}_sha256": digest for split, digest in current_hashes.items()},
    }


def serialize_messages(
    record: Mapping[str, Any],
    tokenizer: BPETokenizer,
    special_ids: Mapping[str, int],
    *,
    retain_evaluation: bool = False,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
) -> dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
        _raise("invalid_messages", "record has an invalid message count")

    sequence = [int(special_ids["<BOS>"])]
    supervised_targets = [False]
    assistant_spans: list[tuple[int, int]] = []
    expected_roles = ("user", "assistant")
    for index, message in enumerate(messages):
        expected_role = expected_roles[index % 2]
        if not isinstance(message, Mapping) or message.get("role") != expected_role:
            _raise("invalid_role_order", "record roles must alternate user and assistant")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            _raise("empty_message_content", "record contains empty message content")
        try:
            content_ids = tokenizer.encode(content)
        except ValueError as error:
            raise SFTV7EncodingError(
                "unencodable_content",
                "record contains characters outside the frozen BPE vocabulary",
            ) from error

        if expected_role == "user":
            sequence.append(int(special_ids["<USER>"]))
            supervised_targets.append(False)
            sequence.extend(content_ids)
            supervised_targets.extend(False for _ in content_ids)
        else:
            sequence.append(int(special_ids["<ASSISTANT>"]))
            supervised_targets.append(False)
            answer_start = len(sequence)
            sequence.extend(content_ids)
            supervised_targets.extend(True for _ in content_ids)
            sequence.append(int(special_ids["<EOS>"]))
            supervised_targets.append(True)
            assistant_spans.append((answer_start, len(sequence)))

    if len(sequence) > max_sequence_length:
        _raise("sequence_too_long", "record exceeds the 512-token context limit")
    input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
    labels = torch.tensor(sequence[1:], dtype=torch.long)
    target_mask = torch.tensor(supervised_targets[1:], dtype=torch.bool)
    labels[~target_mask] = IGNORE_INDEX
    if not bool((labels != IGNORE_INDEX).any()):
        _raise("no_supervision", "record has no supervised assistant targets")

    dimension = record.get("primary_dimension", record.get("dimension"))
    family = record.get("task_family", record.get("family"))
    if not isinstance(record.get("id"), str) or not record["id"].strip():
        _raise("invalid_record_id", "record ID must be a non-empty string")
    if not isinstance(dimension, str) or not dimension.strip():
        _raise("invalid_dimension", "record dimension must be a non-empty string")
    if not isinstance(family, str) or not family.strip():
        _raise("invalid_family", "record family must be a non-empty string")

    encoded: dict[str, Any] = {
        "id": record["id"],
        "primary_dimension": dimension,
        "task_family": family,
        "split": record["split"],
        "input_ids": input_ids,
        "labels": labels,
        "assistant_spans": assistant_spans,
        "assistant_turns": len(assistant_spans),
        "sequence_length": len(sequence),
    }
    if retain_evaluation:
        encoded["evaluation"] = _copy_evaluation_metadata(record)
    return encoded


def _encode_split(
    records: Sequence[dict[str, Any]],
    tokenizer: BPETokenizer,
    special_ids: Mapping[str, int],
    *,
    retain_evaluation: bool,
) -> list[dict[str, Any]]:
    encoded = [
        serialize_messages(
            record,
            tokenizer,
            special_ids,
            retain_evaluation=retain_evaluation,
        )
        for record in records
    ]
    identifiers = [record["id"] for record in encoded]
    if len(identifiers) != len(set(identifiers)):
        _raise("duplicate_record_ids", "a split contains duplicate record IDs")
    return encoded


def _common_payload(
    tokenizer: BPETokenizer,
    special_ids: Mapping[str, int],
    *,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    token_manifest_path: Path,
    token_manifest_sha256: str,
    sft_dataset_manifest_sha256: str,
    source_paths: Mapping[str, Path],
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    base_provenance = dict(REQUIRED_BASE_CHECKPOINT)
    base_provenance["binding_sha256"] = canonical_json_sha256(base_provenance)
    return {
        "vocab_size": tokenizer.vocab_size,
        "stoi": {token: index for index, token in enumerate(tokenizer.tokens)},
        "itos": {index: token for index, token in enumerate(tokenizer.tokens)},
        "special_token_ids": dict(special_ids),
        "ignore_index": IGNORE_INDEX,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_sha256,
        "bpe_token_manifest_path": str(token_manifest_path),
        "bpe_token_manifest_sha256": token_manifest_sha256,
        "sft_dataset_manifest_sha256": sft_dataset_manifest_sha256,
        "source_jsonl_paths": {name: str(path) for name, path in source_paths.items()},
        "source_jsonl_sha256": dict(source_sha256),
        "required_base_checkpoint": base_provenance,
    }


def prepare_payloads(
    train_records: Sequence[dict[str, Any]],
    val_records: Sequence[dict[str, Any]],
    public_records: Sequence[dict[str, Any]],
    tokenizer: BPETokenizer,
    special_ids: Mapping[str, int],
    *,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    token_manifest_path: Path,
    token_manifest_sha256: str,
    sft_dataset_manifest_sha256: str,
    train_path: Path,
    train_sha256: str,
    val_path: Path,
    val_sha256: str,
    public_path: Path,
    public_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    train_encoded = _encode_split(
        train_records,
        tokenizer,
        special_ids,
        retain_evaluation=False,
    )
    val_encoded = _encode_split(
        val_records,
        tokenizer,
        special_ids,
        retain_evaluation=False,
    )
    public_encoded = _encode_split(
        public_records,
        tokenizer,
        special_ids,
        retain_evaluation=True,
    )

    identifiers = [
        {record["id"] for record in records}
        for records in (train_encoded, val_encoded, public_encoded)
    ]
    if identifiers[0] & identifiers[1] or identifiers[0] & identifiers[2] or identifiers[1] & identifiers[2]:
        _raise("record_id_split_overlap", "record IDs overlap across routine splits")

    common = {
        "tokenizer": tokenizer,
        "special_ids": special_ids,
        "tokenizer_path": tokenizer_path,
        "tokenizer_sha256": tokenizer_sha256,
        "token_manifest_path": token_manifest_path,
        "token_manifest_sha256": token_manifest_sha256,
        "sft_dataset_manifest_sha256": sft_dataset_manifest_sha256,
    }
    train_val_payload = {
        "schema_version": "sft-v7-train-val-tensors/v1",
        "train_records": train_encoded,
        "val_records": val_encoded,
        **_common_payload(
            **common,
            source_paths={"train": train_path, "val": val_path},
            source_sha256={"train": train_sha256, "val": val_sha256},
        ),
    }
    public_payload = {
        "schema_version": "sft-v7-public-tensors/v1",
        "public_records": public_encoded,
        **_common_payload(
            **common,
            source_paths={"public_diagnostic": public_path},
            source_sha256={"public_diagnostic": public_sha256},
        ),
    }
    for payload in (train_val_payload, public_payload):
        binding = {
            "schema_version": payload["schema_version"],
            "source_jsonl_sha256": payload["source_jsonl_sha256"],
            "tokenizer_sha256": payload["tokenizer_sha256"],
            "bpe_token_manifest_sha256": payload["bpe_token_manifest_sha256"],
            "sft_dataset_manifest_sha256": payload["sft_dataset_manifest_sha256"],
            "required_base_checkpoint": payload["required_base_checkpoint"],
        }
        payload["artifact_binding_sha256"] = canonical_json_sha256(binding)
        _assert_no_forbidden_scope(payload)

    all_encoded = train_encoded + val_encoded + public_encoded
    report = {
        "schema_version": "sft-v7-tensor-report/v1",
        "status": "prepared",
        "split_counts": {
            "train": len(train_encoded),
            "val": len(val_encoded),
            "public_diagnostic": len(public_encoded),
        },
        "supervised_tokens": {
            name: sum(int((record["labels"] != IGNORE_INDEX).sum()) for record in records)
            for name, records in (
                ("train", train_encoded),
                ("val", val_encoded),
                ("public_diagnostic", public_encoded),
            )
        },
        "max_sequence_length": max(int(record["sequence_length"]) for record in all_encoded),
        "multiturn_records": sum(int(record["assistant_turns"]) > 1 for record in all_encoded),
        "task_family_counts": dict(Counter(record["task_family"] for record in all_encoded)),
        "train_val_binding_sha256": train_val_payload["artifact_binding_sha256"],
        "public_binding_sha256": public_payload["artifact_binding_sha256"],
    }
    _assert_no_forbidden_scope(report)
    return train_val_payload, public_payload, report


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> str:
    _assert_no_forbidden_scope(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    digest = file_sha256(path)
    atomic_write_text(f"{path}.sha256", f"{digest}  {path.name}\n")
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    if reloaded.get("artifact_binding_sha256") != payload.get("artifact_binding_sha256"):
        raise IOError("tensor artifact reload verification failed")
    return digest


def validate_paths(args: argparse.Namespace) -> None:
    expected_input_names = {
        "train": "train.jsonl",
        "val": "val.jsonl",
        "public_diagnostic": "public_diagnostic.jsonl",
    }
    for attribute, expected_name in expected_input_names.items():
        if getattr(args, attribute).name != expected_name:
            _raise("unexpected_input_name", f"{attribute} input must be named {expected_name}")
    expected_outputs = {
        "train_val_output": "train_val_tensors.pt",
        "public_output": "public_diagnostic_tensors.pt",
    }
    for attribute, expected_name in expected_outputs.items():
        if getattr(args, attribute).name != expected_name:
            _raise("unexpected_output_name", f"{attribute} must be named {expected_name}")
    paths = [
        args.train,
        args.val,
        args.public_diagnostic,
        args.tokenizer,
        args.token_manifest,
        args.dataset_manifest,
        args.train_val_output,
        args.public_output,
    ]
    if len({str(path.resolve()) for path in paths}) != len(paths):
        _raise("path_overlap", "input and output paths must be distinct")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument(
        "--public-diagnostic",
        type=Path,
        default=DEFAULT_PUBLIC_DIAGNOSTIC,
    )
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--token-manifest", type=Path, default=DEFAULT_TOKEN_MANIFEST)
    parser.add_argument(
        "--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST
    )
    parser.add_argument("--train-val-output", type=Path, default=DEFAULT_TRAIN_VAL_OUTPUT)
    parser.add_argument("--public-output", type=Path, default=DEFAULT_PUBLIC_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = generate_run_id("sft-v7-tensor")
    loggers = configure_module_loggers(
        args.log_dir,
        run_id,
        resolve_module_log_levels(
            {
                "data": "INFO",
                "sft": "INFO",
                "validation": "INFO",
                "orchestrator": "INFO",
            }
        ),
        max_bytes=10 * 1024 * 1024,
        backup_count=5,
        console=True,
    )
    try:
        validate_paths(args)
        tokenizer, special_ids, tokenizer_identity = load_and_validate_formal_tokenizer(
            args.tokenizer,
            args.token_manifest,
        )
        dataset_identity = load_and_validate_dataset_manifest(
            args.dataset_manifest,
            {
                "train": args.train,
                "val": args.val,
                "public_diagnostic": args.public_diagnostic,
            },
        )
        train_sha256 = file_sha256(args.train)
        val_sha256 = file_sha256(args.val)
        public_sha256 = file_sha256(args.public_diagnostic)
        train_records = read_jsonl(args.train, "train")
        val_records = read_jsonl(args.val, "val")
        public_records = read_jsonl(args.public_diagnostic, "public_diagnostic")
        loggers["data"].info(
            "loaded routine splits train=%d val=%d public=%d train_sha256=%s "
            "val_sha256=%s public_sha256=%s",
            len(train_records),
            len(val_records),
            len(public_records),
            train_sha256,
            val_sha256,
            public_sha256,
        )
        loggers["validation"].info(
            "formal tokenizer binding passed vocab=%d tokenizer_sha256=%s manifest_sha256=%s "
            "base_step=%d base_sha256=%s",
            tokenizer.vocab_size,
            tokenizer_identity["tokenizer_sha256"],
            tokenizer_identity["manifest_sha256"],
            REQUIRED_BASE_CHECKPOINT["step"],
            REQUIRED_BASE_CHECKPOINT["sha256"],
        )
        train_val_payload, public_payload, report = prepare_payloads(
            train_records,
            val_records,
            public_records,
            tokenizer,
            special_ids,
            tokenizer_path=args.tokenizer,
            tokenizer_sha256=tokenizer_identity["tokenizer_sha256"],
            token_manifest_path=args.token_manifest,
            token_manifest_sha256=tokenizer_identity["bpe_token_manifest_sha256"],
            sft_dataset_manifest_sha256=dataset_identity[
                "sft_dataset_manifest_sha256"
            ],
            train_path=args.train,
            train_sha256=train_sha256,
            val_path=args.val,
            val_sha256=val_sha256,
            public_path=args.public_diagnostic,
            public_sha256=public_sha256,
        )
        loggers["sft"].info(
            "encoded train=%d val=%d public=%d max_sequence=%d multiturn=%d",
            len(train_val_payload["train_records"]),
            len(train_val_payload["val_records"]),
            len(public_payload["public_records"]),
            report["max_sequence_length"],
            report["multiturn_records"],
        )
        train_val_output_sha256 = atomic_torch_save(
            train_val_payload,
            args.train_val_output,
        )
        public_output_sha256 = atomic_torch_save(public_payload, args.public_output)
        report.update(
            {
                "run_id": run_id,
                "train_val_output_path": str(args.train_val_output),
                "train_val_output_sha256": train_val_output_sha256,
                "public_output_path": str(args.public_output),
                "public_output_sha256": public_output_sha256,
                "source_jsonl_paths": {
                    "train": str(args.train),
                    "val": str(args.val),
                    "public_diagnostic": str(args.public_diagnostic),
                },
                "source_jsonl_sha256": {
                    "train": train_sha256,
                    "val": val_sha256,
                    "public_diagnostic": public_sha256,
                },
                "tokenizer_path": str(args.tokenizer),
                "tokenizer_sha256": tokenizer_identity["tokenizer_sha256"],
                "bpe_token_manifest_path": str(args.token_manifest),
                "bpe_token_manifest_sha256": tokenizer_identity[
                    "bpe_token_manifest_sha256"
                ],
                "sft_dataset_manifest_path": str(args.dataset_manifest),
                "sft_dataset_manifest_sha256": dataset_identity[
                    "sft_dataset_manifest_sha256"
                ],
                "required_base_checkpoint": train_val_payload["required_base_checkpoint"],
            }
        )
        _assert_no_forbidden_scope(report)
        atomic_write_json(args.report, report)
        loggers["orchestrator"].info(
            "wrote isolated tensor artifacts train_val_sha256=%s public_sha256=%s report=%s",
            train_val_output_sha256,
            public_output_sha256,
            args.report,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        loggers["validation"].error(
            "tensor preparation failed error_code=%s error_type=%s",
            getattr(error, "code", "unexpected_failure"),
            type(error).__name__,
        )
        raise
    finally:
        close_module_loggers(loggers)


if __name__ == "__main__":
    raise SystemExit(main())
