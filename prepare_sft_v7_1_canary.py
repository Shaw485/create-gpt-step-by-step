"""Encode the isolated M021 Canary train and held-out paraphrase splits.

The tensor artifact contains exactly 64 optimization records and 16 evaluation
records.  It cannot accept the formal v7 diagnostic or blind splits.  Assistant
answers (including EOS) are the only supervised targets.

Diagnostics are separated into ``data``, ``encoding``, ``validation``,
``artifact`` and ``orchestrator`` rotating JSONL logs.  Each module can be set
to DEBUG/INFO/WARNING/ERROR/OFF with its CLI option or with
``GPT_CANARY_LOG_LEVEL_<MODULE>``.  Logs contain counts, hashes and error codes;
they never contain message bodies or token IDs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from build_sft_v7_1_canary import CANARY_CONFIG_BINDING
from prepare_bpe_sft_v7 import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SPECIAL_TOKEN_IDS,
    EXPECTED_TOKENIZER_SHA256,
    EXPECTED_VOCAB_SIZE,
    IGNORE_INDEX,
    REQUIRED_BASE_CHECKPOINT,
    atomic_torch_save,
    load_and_validate_formal_tokenizer,
    read_jsonl,
    serialize_messages,
)
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


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN = Path("data/sft/v7_1_canary/train.jsonl")
DEFAULT_EVAL = Path("data/sft/v7_1_canary/holdout_eval.jsonl")
DEFAULT_DATASET_MANIFEST = Path("data/sft/v7_1_canary/manifest.json")
DEFAULT_TOKENIZER = Path("data/scaling_a/bpe_3000/tokenizer.json")
DEFAULT_TOKEN_MANIFEST = Path("data/scaling_a/bpe_3000/token_manifest.json")
DEFAULT_OUTPUT = Path("data/sft/v7_1_canary/train_eval_tensors.pt")
DEFAULT_REPORT_JSON = Path(
    "reports/milestones/021_sft_v7_1_canary/canary_tensor_report.json"
)
DEFAULT_REPORT_MD = Path(
    "reports/milestones/021_sft_v7_1_canary/canary_tensor_report.md"
)
DEFAULT_LOG_DIR = Path("logs/sft_v7_1_canary_prepare")

DATASET_MANIFEST_SCHEMA = "sft-v7.1-canary-manifest/v1"
DATASET_RECORD_SCHEMA = "sft_v7_1_canary/1.0"
TENSOR_SCHEMA = "sft-v7.1-canary-tensors/v1"
REPORT_SCHEMA = "sft-v7.1-canary-tensor-report/v1"
EXPECTED_COUNTS = {"train": 64, "holdout_eval": 16}
EXPECTED_DIMENSION = "parameter_core_fact_and_correction"
EXPECTED_FAMILY = "canary_known_core"
EXPECTED_CANARY_MANIFEST_SHA256 = (
    "68908fdabe4f8ae470f6bcd4ec6d11b59304829119835af901df7bf9888ef50d"
)
EXPECTED_CANARY_DATASET_IDENTITY_SHA256 = (
    "b2012953980c823d018494d2a5212e79b51370c632dceb1e559b552f152f92b9"
)
EXPECTED_CANARY_SOURCE_SHA256 = {
    "train": "e5f0f90b26f9dbacb68017bbaa4243a41ccdacb4ac60484677964061fd4d008a",
    "holdout_eval": "fe8a72efcd8e3f179d61ca8e4b2de2b3c775dbd9841d25692fe6743f3548c64d",
}
EXPECTED_CANARY_TENSOR_SHA256 = (
    "8511bea2aa449f9dc29cc268239951786d512d9895f29a8730ccb7179f26914e"
)
CANONICAL_CANARY_PATHS = {
    "manifest": DEFAULT_DATASET_MANIFEST,
    "train": DEFAULT_TRAIN,
    # The filename is retained for artifact compatibility. Semantically this is
    # an unseen-question development/selection split, not a blind test.
    "holdout_eval": DEFAULT_EVAL,
    "tensor": DEFAULT_OUTPUT,
}
_HEX_SHA = re.compile(r"[0-9a-f]{64}")
_RESTRICTED_KEY = re.compile(r"(?:^|_)(?:public|sealed)(?:_|$)", re.IGNORECASE)


class CanaryEncodingError(ValueError):
    """A log-safe Canary data, identity or encoding failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str) -> None:
    raise CanaryEncodingError(code, message)


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return f"artifact://canary/{resolved.name}"


def _require_canonical_repository_path(
    path: Path,
    expected_relative: Path,
    *,
    code: str,
) -> Path:
    """Resolve one reviewed artifact and reject copies or external lookalikes."""

    candidate = path if path.is_absolute() else REPOSITORY_ROOT / path
    expected = REPOSITORY_ROOT / expected_relative
    if candidate.resolve() != expected.resolve():
        _fail(code, "Canary artifact is not the reviewed repository path")
    return expected.resolve()


def reject_restricted_keys(value: Any, *, location: str = "payload") -> None:
    """Keep formal diagnostic/blind split names out of the tensor schema."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _RESTRICTED_KEY.search(str(key)):
                _fail("restricted_scope_key", f"{location} contains a restricted key")
            reject_restricted_keys(nested, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            reject_restricted_keys(nested, location=f"{location}[{index}]")


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CanaryEncodingError(
            "invalid_canary_manifest", "Canary manifest cannot be parsed"
        ) from error
    if not isinstance(payload, dict):
        _fail("invalid_canary_manifest", "Canary manifest root must be an object")
    return payload


def load_and_validate_canary_manifest(
    manifest_path: Path,
    split_paths: Mapping[str, Path],
    *,
    enforce_frozen_binding: bool = True,
) -> dict[str, Any]:
    """Bind only the Canary train/evaluation files to their builder manifest."""

    if enforce_frozen_binding:
        manifest_path = _require_canonical_repository_path(
            manifest_path,
            CANONICAL_CANARY_PATHS["manifest"],
            code="manifest_path_mismatch",
        )
        for split in EXPECTED_COUNTS:
            _require_canonical_repository_path(
                split_paths[split],
                CANONICAL_CANARY_PATHS[split],
                code="manifest_path_mismatch",
            )
        if file_sha256(manifest_path) != EXPECTED_CANARY_MANIFEST_SHA256:
            _fail("manifest_sha_mismatch", "Reviewed Canary manifest SHA-256 changed")
    manifest = _load_manifest(manifest_path)
    if manifest.get("manifest_schema_version") != DATASET_MANIFEST_SCHEMA:
        _fail("manifest_schema_mismatch", "Canary manifest schema changed")
    if manifest.get("record_schema_version") != DATASET_RECORD_SCHEMA:
        _fail("record_schema_mismatch", "Canary record schema changed")
    if manifest.get("status") != "frozen_canary_ready":
        _fail("manifest_not_ready", "Canary manifest is not frozen and ready")
    config_binding = manifest.get("config")
    if not isinstance(config_binding, Mapping) or (
        config_binding.get("path") != CANARY_CONFIG_BINDING["path"]
        or config_binding.get("sha256") != CANARY_CONFIG_BINDING["sha256"]
    ):
        _fail("config_binding_mismatch", "Canary manifest config binding changed")
    totals = manifest.get("split_totals")
    if not isinstance(totals, Mapping) or {
        str(key): int(value) for key, value in totals.items()
    } != EXPECTED_COUNTS:
        _fail("manifest_count_mismatch", "Canary manifest split counts changed")
    split_files = manifest.get("split_files")
    if not isinstance(split_files, Mapping) or set(split_files) != set(EXPECTED_COUNTS):
        _fail("manifest_split_scope", "Canary manifest split scope is not isolated")

    hashes: dict[str, str] = {}
    for split, expected_count in EXPECTED_COUNTS.items():
        metadata = split_files.get(split)
        if not isinstance(metadata, Mapping):
            _fail("manifest_split_missing", f"Canary manifest lacks {split}")
        declared = manifest_path.parent / Path(str(metadata.get("path", "")))
        actual_path = split_paths[split]
        if declared.resolve() != actual_path.resolve():
            _fail("manifest_path_mismatch", f"{split} is not manifest-bound")
        if int(metadata.get("count", -1)) != expected_count:
            _fail("manifest_count_mismatch", f"{split} count changed")
        if metadata.get("schema_version") != DATASET_RECORD_SCHEMA:
            _fail("manifest_record_schema_mismatch", f"{split} record schema changed")
        actual_hash = file_sha256(actual_path)
        if metadata.get("sha256") != actual_hash:
            _fail("manifest_sha_mismatch", f"{split} SHA-256 changed")
        if enforce_frozen_binding and actual_hash != EXPECTED_CANARY_SOURCE_SHA256[split]:
            _fail("manifest_sha_mismatch", f"Reviewed {split} SHA-256 changed")
        with actual_path.open(encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
        if count != expected_count:
            _fail("jsonl_count_mismatch", f"{split} JSONL count changed")
        hashes[split] = actual_hash

    binding = manifest.get("training_binding")
    if not isinstance(binding, Mapping):
        _fail("training_binding_missing", "Canary manifest lacks training binding")
    base = binding.get("base_checkpoint")
    tokenizer = binding.get("tokenizer")
    if not isinstance(base, Mapping) or (
        base.get("path") != REQUIRED_BASE_CHECKPOINT["path"]
        or base.get("sha256") != REQUIRED_BASE_CHECKPOINT["sha256"]
        or int(base.get("step", -1)) != REQUIRED_BASE_CHECKPOINT["step"]
        or int(base.get("parameter_count", -1))
        != REQUIRED_BASE_CHECKPOINT["parameter_count"]
    ):
        _fail("base_binding_mismatch", "Canary base checkpoint binding changed")
    if not isinstance(tokenizer, Mapping) or (
        tokenizer.get("sha256") != EXPECTED_TOKENIZER_SHA256
        or int(tokenizer.get("vocab_size", -1)) != EXPECTED_VOCAB_SIZE
        or int(tokenizer.get("context_limit", -1)) != 512
    ):
        _fail("tokenizer_binding_mismatch", "Canary tokenizer binding changed")
    dataset_identity = str(manifest.get("dataset_identity_sha256", ""))
    if not _HEX_SHA.fullmatch(dataset_identity):
        _fail("dataset_identity_invalid", "Canary dataset identity is invalid")
    if (
        enforce_frozen_binding
        and dataset_identity != EXPECTED_CANARY_DATASET_IDENTITY_SHA256
    ):
        _fail("dataset_identity_invalid", "Reviewed Canary dataset identity changed")
    return {
        "manifest": manifest,
        "canary_manifest_sha256": file_sha256(manifest_path),
        "dataset_identity_sha256": dataset_identity,
        "source_jsonl_sha256": hashes,
    }


def validate_source_records(
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate split roles and scoring metadata without returning text bodies."""

    if len(train_records) != 64 or len(eval_records) != 16:
        _fail("source_count_mismatch", "Canary source counts must be 64 and 16")
    all_ids: set[str] = set()
    fact_counts: dict[str, Counter[str]] = {}
    for split, records in (("train", train_records), ("holdout_eval", eval_records)):
        for record in records:
            identifier = record.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in all_ids:
                _fail("record_id_invalid", "Canary record IDs are empty or duplicated")
            all_ids.add(identifier)
            if record.get("schema_version") != DATASET_RECORD_SCHEMA:
                _fail("source_record_schema", "Canary source record schema changed")
            if record.get("split") != split:
                _fail("source_split_mismatch", "Canary source record split changed")
            if record.get("primary_dimension") != EXPECTED_DIMENSION:
                _fail("source_dimension_mismatch", "Canary source dimension changed")
            if record.get("task_family") != EXPECTED_FAMILY:
                _fail("source_family_mismatch", "Canary source task family changed")
            fact_id = str(record.get("fact_id", ""))
            if not fact_id:
                _fail("source_fact_missing", "Canary source fact ID is missing")
            fact_counts.setdefault(fact_id, Counter())[split] += 1
            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) != 2:
                _fail("source_messages_invalid", "Canary record must be single-turn")
            evaluation = record.get("evaluation")
            if not isinstance(evaluation, Mapping):
                _fail("evaluation_missing", "Canary scoring metadata is missing")
            required = evaluation.get("required_terms")
            forbidden = evaluation.get("forbidden_terms")
            if (
                evaluation.get("metric") != "required_terms_all"
                or evaluation.get("known_fact") is not True
                or not isinstance(required, list)
                or not required
                or not all(isinstance(term, str) and term for term in required)
                or not isinstance(forbidden, list)
                or not all(isinstance(term, str) and term for term in forbidden)
            ):
                _fail("evaluation_contract", "Canary scoring metadata changed")
            supervision = record.get("supervision")
            if not isinstance(supervision, Mapping) or (
                supervision.get("assistant_only_loss") is not True
                or supervision.get("eos_appended_by_encoder") is not True
                or supervision.get("use_for_training") is not (split == "train")
            ):
                _fail("supervision_contract", "Canary supervision role changed")
    if len(fact_counts) != 8 or any(
        counts["train"] != 8 or counts["holdout_eval"] != 2
        for counts in fact_counts.values()
    ):
        _fail("fact_quota_mismatch", "Canary fact quotas must be 8/2")
    return {
        "split_counts": {"train": len(train_records), "holdout_eval": len(eval_records)},
        "fact_count": len(fact_counts),
        "record_ids_unique": True,
    }


def prepare_canary_payload(
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    special_ids: Mapping[str, int],
    *,
    tokenizer_path: Path,
    token_manifest_path: Path,
    tokenizer_sha256: str,
    token_manifest_sha256: str,
    dataset_manifest_path: Path,
    dataset_manifest_sha256: str,
    dataset_identity_sha256: str,
    train_path: Path,
    eval_path: Path,
    source_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Encode both roles while preserving scoring metadata, never message text."""

    summary = validate_source_records(train_records, eval_records)
    encoded_train = [
        serialize_messages(record, tokenizer, special_ids, retain_evaluation=True)
        for record in train_records
    ]
    encoded_eval = [
        serialize_messages(record, tokenizer, special_ids, retain_evaluation=True)
        for record in eval_records
    ]
    required_base = dict(REQUIRED_BASE_CHECKPOINT)
    required_base["binding_sha256"] = canonical_json_sha256(REQUIRED_BASE_CHECKPOINT)
    payload: dict[str, Any] = {
        "schema_version": TENSOR_SCHEMA,
        "train_records": encoded_train,
        "eval_records": encoded_eval,
        "vocab_size": int(tokenizer.vocab_size),
        "stoi": {token: index for index, token in enumerate(tokenizer.tokens)},
        "itos": {index: token for index, token in enumerate(tokenizer.tokens)},
        "special_token_ids": dict(special_ids),
        "ignore_index": IGNORE_INDEX,
        "tokenizer_path": _portable(tokenizer_path),
        "tokenizer_sha256": tokenizer_sha256,
        "bpe_token_manifest_path": _portable(token_manifest_path),
        "bpe_token_manifest_sha256": token_manifest_sha256,
        "canary_dataset_manifest_path": _portable(dataset_manifest_path),
        "canary_dataset_manifest_sha256": dataset_manifest_sha256,
        "canary_dataset_identity_sha256": dataset_identity_sha256,
        "source_jsonl_paths": {
            "train": _portable(train_path),
            "holdout_eval": _portable(eval_path),
        },
        "source_jsonl_sha256": dict(source_hashes),
        "required_base_checkpoint": required_base,
    }
    binding = {
        "schema_version": TENSOR_SCHEMA,
        "source_jsonl_sha256": dict(source_hashes),
        "tokenizer_sha256": tokenizer_sha256,
        "bpe_token_manifest_sha256": token_manifest_sha256,
        "canary_dataset_manifest_sha256": dataset_manifest_sha256,
        "canary_dataset_identity_sha256": dataset_identity_sha256,
        "required_base_checkpoint": required_base,
    }
    payload["artifact_binding_sha256"] = canonical_json_sha256(binding)
    reject_restricted_keys(payload)

    all_records = encoded_train + encoded_eval
    report = {
        "report_schema_version": REPORT_SCHEMA,
        "status": "prepared",
        "split_counts": summary["split_counts"],
        "fact_count": summary["fact_count"],
        "supervised_token_counts": {
            "train": sum(int((record["labels"] != IGNORE_INDEX).sum()) for record in encoded_train),
            "holdout_eval": sum(
                int((record["labels"] != IGNORE_INDEX).sum()) for record in encoded_eval
            ),
        },
        "sequence_lengths": {
            "minimum": min(int(record["sequence_length"]) for record in all_records),
            "maximum": max(int(record["sequence_length"]) for record in all_records),
            "mean": sum(int(record["sequence_length"]) for record in all_records)
            / len(all_records),
        },
        "assistant_only_supervision": True,
        "eos_appended_and_supervised": all(
            int(record["labels"][record["labels"] != IGNORE_INDEX][-1])
            == int(special_ids["<EOS>"])
            for record in all_records
        ),
        "artifact_binding_sha256": payload["artifact_binding_sha256"],
        "tokenizer_sha256": tokenizer_sha256,
        "bpe_token_manifest_sha256": token_manifest_sha256,
        "canary_dataset_manifest_sha256": dataset_manifest_sha256,
        "canary_dataset_identity_sha256": dataset_identity_sha256,
    }
    reject_restricted_keys(report)
    return payload, report


def render_report_markdown(report: Mapping[str, Any]) -> str:
    lengths = report["sequence_lengths"]
    supervised = report["supervised_token_counts"]
    return "\n".join(
        [
            "# M021 SFT v7.1 Canary 编码报告",
            "",
            f"状态：**{report['status']}**",
            "",
            "| 项目 | Train | Holdout eval |",
            "|---|---:|---:|",
            f"| 记录数 | {report['split_counts']['train']} | {report['split_counts']['holdout_eval']} |",
            f"| 监督 Token | {supervised['train']} | {supervised['holdout_eval']} |",
            "",
            f"- 事实数：{report['fact_count']}",
            f"- 序列长度：min={lengths['minimum']}，mean={lengths['mean']:.2f}，max={lengths['maximum']}",
            f"- 仅 assistant 参与 Loss：{report['assistant_only_supervision']}",
            f"- EOS 已追加且参与监督：{report['eos_appended_and_supervised']}",
            f"- Tensor SHA-256：`{report['output_sha256']}`",
            f"- Manifest SHA-256：`{report['canary_dataset_manifest_sha256']}`",
            "",
            "## 日志与独立调试",
            "",
            "data、encoding、validation、artifact、orchestrator 各自写入轮转 JSONL。"
            "可使用 `--data-log-level DEBUG` 等参数，或设置 "
            "`GPT_CANARY_LOG_LEVEL_DATA=DEBUG` 单独打开某一类；传入 `OFF` 可关闭。"
            "日志仅包含数量、长度、SHA、状态和错误码，不包含问题、答案或 Token ID。"
            "定位完成后恢复 INFO；默认单文件 1 MiB，保留 3 份备份。",
            "",
        ]
    )


def write_frozen_canary_tensor(
    payload: Mapping[str, Any],
    output_path: Path,
    *,
    run_id: str,
) -> str:
    """Verify deterministic bytes before replacing the reviewed tensor artifact."""

    candidate = output_path.with_name(f".{output_path.name}.{run_id}.candidate")
    candidate_sidecar = Path(f"{candidate}.sha256")
    try:
        output_sha = atomic_torch_save(payload, candidate)
        if output_sha != EXPECTED_CANARY_TENSOR_SHA256:
            _fail(
                "output_sha_mismatch",
                "Encoded Canary tensor bytes differ from the reviewed artifact",
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(candidate, output_path)
        atomic_write_text(
            Path(f"{output_path}.sha256"),
            f"{output_sha}  {output_path.name}\n",
        )
        return output_sha
    finally:
        if candidate.exists():
            candidate.unlink()
        if candidate_sidecar.exists():
            candidate_sidecar.unlink()


def _add_log_arguments(parser: argparse.ArgumentParser, modules: Sequence[str]) -> None:
    for module in modules:
        parser.add_argument(
            f"--{module.replace('_', '-')}-log-level",
            choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "OFF"),
            default=None,
            help=f"independently set the {module} rotating JSONL log level",
        )
    parser.add_argument("--log-max-bytes", type=int, default=1_048_576)
    parser.add_argument("--log-backup-count", type=int, default=3)
    parser.add_argument("--no-console-log", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--holdout-eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--token-manifest", type=Path, default=DEFAULT_TOKEN_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    _add_log_arguments(
        parser, ("data", "encoding", "validation", "artifact", "orchestrator")
    )
    return parser.parse_args(argv)


def _log_levels(args: argparse.Namespace, modules: Sequence[str]) -> dict[str, str]:
    levels = resolve_module_log_levels(
        {module: "INFO" for module in modules}, env_prefix="GPT_CANARY_LOG_LEVEL"
    )
    for module in modules:
        override = getattr(args, f"{module}_log_level")
        if override is not None:
            levels[module] = override
    return levels


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    modules = ("data", "encoding", "validation", "artifact", "orchestrator")
    run_id = generate_run_id("sft-v7-1-canary-prepare")
    loggers = configure_module_loggers(
        args.log_dir,
        run_id,
        _log_levels(args, modules),
        max_bytes=args.log_max_bytes,
        backup_count=args.log_backup_count,
        console=not args.no_console_log,
    )
    try:
        _require_canonical_repository_path(
            args.train, CANONICAL_CANARY_PATHS["train"], code="source_path_mismatch"
        )
        _require_canonical_repository_path(
            args.holdout_eval,
            CANONICAL_CANARY_PATHS["holdout_eval"],
            code="source_path_mismatch",
        )
        _require_canonical_repository_path(
            args.dataset_manifest,
            CANONICAL_CANARY_PATHS["manifest"],
            code="manifest_path_mismatch",
        )
        _require_canonical_repository_path(
            args.tokenizer, DEFAULT_TOKENIZER, code="tokenizer_path_mismatch"
        )
        _require_canonical_repository_path(
            args.token_manifest,
            DEFAULT_TOKEN_MANIFEST,
            code="token_manifest_path_mismatch",
        )
        _require_canonical_repository_path(
            args.output, CANONICAL_CANARY_PATHS["tensor"], code="output_path_mismatch"
        )
        tokenizer, special_ids, tokenizer_identity = load_and_validate_formal_tokenizer(
            args.tokenizer, args.token_manifest
        )
        manifest_identity = load_and_validate_canary_manifest(
            args.dataset_manifest,
            {"train": args.train, "holdout_eval": args.holdout_eval},
        )
        train_records = read_jsonl(args.train, "train")
        eval_records = read_jsonl(args.holdout_eval, "holdout_eval")
        loggers["data"].info(
            "Canary sources loaded",
            extra={
                "context": {
                    "train_count": len(train_records),
                    "holdout_eval_count": len(eval_records),
                    "manifest_sha256": manifest_identity["canary_manifest_sha256"],
                }
            },
        )
        payload, report = prepare_canary_payload(
            train_records,
            eval_records,
            tokenizer,
            special_ids,
            tokenizer_path=args.tokenizer,
            token_manifest_path=args.token_manifest,
            tokenizer_sha256=tokenizer_identity["tokenizer_sha256"],
            token_manifest_sha256=tokenizer_identity["bpe_token_manifest_sha256"],
            dataset_manifest_path=args.dataset_manifest,
            dataset_manifest_sha256=manifest_identity["canary_manifest_sha256"],
            dataset_identity_sha256=manifest_identity["dataset_identity_sha256"],
            train_path=args.train,
            eval_path=args.holdout_eval,
            source_hashes=manifest_identity["source_jsonl_sha256"],
        )
        loggers["encoding"].info(
            "Canary encoding complete",
            extra={
                "context": {
                    "train_count": len(payload["train_records"]),
                    "eval_count": len(payload["eval_records"]),
                    "maximum_sequence_length": report["sequence_lengths"]["maximum"],
                    "train_supervised_tokens": report["supervised_token_counts"]["train"],
                    "eval_supervised_tokens": report["supervised_token_counts"]["holdout_eval"],
                }
            },
        )
        if special_ids != EXPECTED_SPECIAL_TOKEN_IDS:
            _fail("special_ids_mismatch", "Formal special-token IDs changed")
        if tokenizer_identity["tokenizer_sha256"] != EXPECTED_TOKENIZER_SHA256:
            _fail("tokenizer_sha_mismatch", "Formal tokenizer SHA changed")
        if tokenizer_identity["bpe_token_manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
            _fail("token_manifest_sha_mismatch", "Formal token manifest SHA changed")
        loggers["validation"].info(
            "Canary identities and supervision passed",
            extra={
                "context": {
                    "vocab_size": tokenizer.vocab_size,
                    "eos_supervised": report["eos_appended_and_supervised"],
                    "artifact_binding_sha256": payload["artifact_binding_sha256"],
                }
            },
        )
        output_sha = write_frozen_canary_tensor(
            payload,
            args.output,
            run_id=run_id,
        )
        report.update(
            {
                "run_id": run_id,
                "output_path": _portable(args.output),
                "output_sha256": output_sha,
                "source_jsonl_paths": {
                    "train": _portable(args.train),
                    "holdout_eval": _portable(args.holdout_eval),
                },
                "source_jsonl_sha256": manifest_identity["source_jsonl_sha256"],
                "logging": {
                    "directory": _portable(args.log_dir),
                    "modules": list(modules),
                    "format": "rotating JSONL with UTC timestamp and run_id",
                    "record_bodies_logged": False,
                    "token_ids_logged": False,
                    "max_bytes": args.log_max_bytes,
                    "backup_count": args.log_backup_count,
                },
            }
        )
        reject_restricted_keys(report)
        atomic_write_json(args.report_json, report)
        atomic_write_text(args.report_md, render_report_markdown(report))
        loggers["artifact"].info(
            "Canary tensor and reports written",
            extra={
                "context": {
                    "tensor_sha256": output_sha,
                    "report_json": _portable(args.report_json),
                    "report_markdown": _portable(args.report_md),
                }
            },
        )
        loggers["orchestrator"].info(
            "Canary preparation finished",
            extra={"context": {"status": "prepared", "run_id": run_id}},
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "split_counts": report["split_counts"],
                    "output": report["output_path"],
                    "output_sha256": output_sha,
                    "report": _portable(args.report_json),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as error:
        loggers["validation"].error(
            "Canary preparation failed",
            extra={
                "context": {
                    "error_code": getattr(error, "code", "unexpected_failure"),
                    "error_type": type(error).__name__,
                }
            },
        )
        raise
    finally:
        close_module_loggers(loggers)


if __name__ == "__main__":
    raise SystemExit(main())
