"""Audit the non-sealed SFT v6 prefix against the frozen v7 vertical contract.

This module has a deliberately unusual input boundary: it consumes exactly the
first 9,400 JSON objects (train, val, and public diagnostic) and never asks the
input iterator for another item.  The final 600 v6 objects remain sealed.  In
particular, do not replace :func:`read_nonsealed_prefix` with a whole-file hash,
``read_text()``, or a list conversion.

Reports and logs contain only counts, hashes, identifiers, and risk codes.  No
question, answer, evidence, or novel body is emitted.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Iterator, Mapping, Sequence

from build_sft_v5_1_no_math import is_arithmetic_text, is_math_topic_text
from training_runtime import (
    atomic_write_json,
    configure_module_loggers,
    generate_run_id,
    resolve_module_log_levels,
)


AUDITED_RECORD_LIMIT = 9_400
DECLARED_SEALED_RECORDS = 600
ALLOWED_PREFIX_SPLITS = frozenset({"train", "val", "public_diagnostic"})

DEFAULT_DATASET = Path("data/sft/v6/sft_v6_10000.jsonl")
DEFAULT_REPORT = Path(
    "reports/milestones/020_sft_v7_vertical/v6_vertical_gap_audit.json"
)
DEFAULT_LOG_DIR = Path("logs/sft_v6_vertical_gap")

V7_NONSEALED_BUCKET_TARGETS: Mapping[str, int] = {
    "parametric_core": 1_692,
    "grounded_single_evidence": 3_008,
    "rag_multi_evidence": 1_316,
    "vertical_interaction": 1_692,
    "novel_expression": 1_222,
    "capability_boundary": 470,
}

V6_FAMILY_TO_BUCKET: Mapping[str, str] = {
    "curated_core_novel_identity": "parametric_core",
    "known_novel_correction": "parametric_core",
    "grounded_novel_entity_fact": "grounded_single_evidence",
    "long_context_exact_paragraph_extraction": "grounded_single_evidence",
    "grounded_format_control": "grounded_single_evidence",
    "natural_multiturn_support": "vertical_interaction",
    "natural_single_turn_support": "vertical_interaction",
    "grounded_concise_rewrite": "novel_expression",
    "grounded_one_sentence_summary": "novel_expression",
    "grounded_next_paragraph_continuation": "novel_expression",
    "grounded_unknown": "capability_boundary",
    "realtime_boundary": "capability_boundary",
    "practice_correction": "capability_boundary",
}

# These are deterministic statistics, not a semantic classifier.  They make the
# already observed v6 template/domain drift independently reproducible.
FORBIDDEN_TEMPLATE_MARKERS = (
    "可以先",
    "先先",
    "原问题是",
    "现只做局部证据核验",
    "当前证据片段",
    "正确，证据支持",
    "材料写到",
    "根据这段文字，可以确认",
    "这段材料明确提到",
)
PROJECT_CONCEPT_FAMILIES = frozenset({"project_concept_explanation"})
GENERIC_SUPPORT_FAMILIES = frozenset(
    {"natural_multiturn_support", "natural_single_turn_support", "practice_correction"}
)
PROJECT_CONCEPT_MARKERS = (
    "Block Size",
    "Embedding",
    "Tokenizer",
    "Token",
    "监督微调",
    "模型训练",
    "注意力机制",
    "学习计划",
    "日志应该",
)
GENERAL_ENCYCLOPEDIA_MARKERS = (
    "绿巨人",
    "爱因斯坦",
    "牛顿",
    "世界首都",
    "百科知识",
    "游乐园在哪里",
)


def _raw_line_bytes(line: str | bytes) -> bytes:
    return line if isinstance(line, bytes) else line.encode("utf-8")


def _parse_json_object(raw_line: str | bytes, logical_index: int) -> dict[str, Any]:
    try:
        text = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        value = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"non-sealed record {logical_index} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"non-sealed record {logical_index} is not a JSON object")
    return value


def read_prefix_from_lines(
    lines: Iterable[str | bytes],
    *,
    limit: int = AUDITED_RECORD_LIMIT,
) -> tuple[list[dict[str, Any]], str]:
    """Consume ``limit`` non-empty JSON lines without one-item lookahead.

    This helper exists so a sentinel iterator can prove the sealed element is
    never requested.  It intentionally avoids ``list(lines)``, ``enumerate``
    over the full iterator, and any post-loop ``next`` call.
    """

    iterator: Iterator[str | bytes] = iter(lines)
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    while len(records) < limit:
        try:
            raw_line = next(iterator)
        except StopIteration as error:
            raise ValueError(
                f"dataset ended after {len(records)} records; expected at least {limit}"
            ) from error
        raw_bytes = _raw_line_bytes(raw_line)
        if not raw_bytes.strip():
            continue
        digest.update(raw_bytes)
        records.append(_parse_json_object(raw_line, len(records) + 1))
    return records, digest.hexdigest()


def read_nonsealed_prefix(
    path: Path,
    *,
    limit: int = AUDITED_RECORD_LIMIT,
) -> tuple[list[dict[str, Any]], str]:
    """Read only the declared non-sealed JSONL prefix.

    Buffered IO may prefetch bytes inside the operating-system/file-object
    implementation, but no prefetched 9,401st line is requested, decoded,
    parsed, hashed, returned, logged, or otherwise exposed to the program.  The
    iterator sentinel test fixes that observable security boundary while
    keeping the real 9,400-record audit practical.
    """

    with path.open("rb") as handle:
        return read_prefix_from_lines(handle, limit=limit)


def canonical_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s，。！？、；：“”‘’（）()《》【】\[\]：:,.!?;'\"`]+", "", normalized)


def opening_fingerprint(answer: str, *, width: int = 12) -> str:
    """Return a non-reversible hash of a reproducible normalized answer prefix."""

    opening = canonical_text(answer)[:width]
    return hashlib.sha256(opening.encode("utf-8")).hexdigest()[:16]


def classify_v6_bucket(record: Mapping[str, Any]) -> str:
    family = str(record.get("task_family", ""))
    if family.startswith("rag_"):
        return "rag_multi_evidence"
    return V6_FAMILY_TO_BUCKET.get(family, "unmapped_or_out_of_scope")


def _combined_text(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        parts = [
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, Mapping)
        ]
        if parts:
            return "\n".join(parts)
    return f"{record.get('question', '')}\n{record.get('answer', '')}"


def audit_v6_vertical_gap(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_records: int = AUDITED_RECORD_LIMIT,
) -> dict[str, Any]:
    """Calculate aggregate, reproducible gap statistics without emitting text."""

    split_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    dimension_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    marker_counts: Counter[str] = Counter()
    opening_counts: Counter[str] = Counter()
    domain_drift_families: Counter[str] = Counter()

    sealed_in_prefix = 0
    math_positive = 0
    encyclopedia_positive = 0
    project_concept_records = 0
    generic_support_records = 0
    rag_bundle_records = 0
    known_core_refusals = 0
    refusal_markers = ("无法确认", "资料不足", "不能直接", "需要证据", "需要检索")

    for record in records:
        split = str(record.get("split", ""))
        family = str(record.get("task_family", ""))
        dimension = str(record.get("primary_dimension", ""))
        answer = str(record.get("answer", ""))
        combined = _combined_text(record)
        messages = record.get("messages")
        assistant_texts = (
            [
                str(message.get("content", ""))
                for message in messages
                if isinstance(message, Mapping) and message.get("role") == "assistant"
            ]
            if isinstance(messages, list)
            else []
        )
        if not assistant_texts:
            assistant_texts = [answer]
        split_counts[split] += 1
        family_counts[family] += 1
        dimension_counts[dimension] += 1
        bucket_counts[classify_v6_bucket(record)] += 1
        opening_counts[opening_fingerprint(answer)] += 1

        if split not in ALLOWED_PREFIX_SPLITS:
            sealed_in_prefix += 1
        for assistant_text in assistant_texts:
            for marker in FORBIDDEN_TEMPLATE_MARKERS:
                if marker in assistant_text:
                    marker_counts[marker] += 1

        is_boundary = family in {"grounded_unknown", "realtime_boundary"}
        if not is_boundary and (is_math_topic_text(combined) or is_arithmetic_text(combined)):
            math_positive += 1
        if not is_boundary and any(marker in combined for marker in GENERAL_ENCYCLOPEDIA_MARKERS):
            encyclopedia_positive += 1
        if family in PROJECT_CONCEPT_FAMILIES or any(
            marker in combined for marker in PROJECT_CONCEPT_MARKERS
        ):
            project_concept_records += 1
            domain_drift_families[family or "unknown"] += 1
        if family in GENERIC_SUPPORT_FAMILIES:
            generic_support_records += 1
            domain_drift_families[family] += 1

        evidence = record.get("evidence")
        if isinstance(evidence, list) and 2 <= len(evidence) <= 4:
            rag_bundle_records += 1
        if family == "curated_core_novel_identity" and any(
            marker in answer for marker in refusal_markers
        ):
            known_core_refusals += 1

    largest_opening_count = max(opening_counts.values(), default=0)
    largest_opening_share = largest_opening_count / len(records) if records else 0.0
    top_openings = [
        {"opening_sha256_prefix": key, "records": count, "share": count / len(records)}
        for key, count in opening_counts.most_common(20)
    ] if records else []

    bucket_gaps = {
        bucket: bucket_counts.get(bucket, 0) - target
        for bucket, target in V7_NONSEALED_BUCKET_TARGETS.items()
    }
    hard_failures: list[str] = []
    checks = (
        (len(records) == expected_records, "audited_record_count"),
        (sealed_in_prefix == 0, "sealed_split_in_audited_prefix"),
        (math_positive == 0, "positive_math_domain_drift"),
        (encyclopedia_positive == 0, "positive_general_encyclopedia_drift"),
        (project_concept_records == 0, "project_concept_domain_drift"),
        (generic_support_records == 0, "generic_support_domain_drift"),
        (not marker_counts, "forbidden_answer_templates"),
        (largest_opening_share <= 0.02, "answer_opening_template_share"),
        (
            rag_bundle_records >= V7_NONSEALED_BUCKET_TARGETS["rag_multi_evidence"],
            "rag_bundle_coverage",
        ),
        (known_core_refusals == 0, "known_core_false_refusal"),
    )
    for passed, name in checks:
        if not passed:
            hard_failures.append(name)

    return {
        "schema_version": "sft-v6-vertical-gap-audit/v1",
        "status": "passed" if not hard_failures else "needs_revision",
        "hard_failures": sorted(hard_failures),
        "audited_records": len(records),
        "audit_boundary": {
            "maximum_records": expected_records,
            "allowed_splits": sorted(ALLOWED_PREFIX_SPLITS),
            "sealed_records_declared_but_not_read": DECLARED_SEALED_RECORDS,
            "sealed_split_records_in_prefix": sealed_in_prefix,
        },
        "split_counts": dict(sorted(split_counts.items())),
        "dimension_counts": dict(sorted(dimension_counts.items())),
        "task_family_counts": dict(sorted(family_counts.items())),
        "vertical_bucket_counts": dict(sorted(bucket_counts.items())),
        "vertical_bucket_targets_nonsealed": dict(V7_NONSEALED_BUCKET_TARGETS),
        "vertical_bucket_deltas": bucket_gaps,
        "template_statistics": {
            "normalization": "NFKC+lower+remove_space_and_punctuation+first_12_codepoints+sha256_prefix16",
            "forbidden_marker_count_unit": "assistant_messages_containing_marker",
            "opening_metric_scope": "final_answer_only",
            "forbidden_marker_counts": dict(sorted(marker_counts.items())),
            "distinct_opening_fingerprints": len(opening_counts),
            "largest_opening_count": largest_opening_count,
            "largest_opening_share": largest_opening_share,
            "top_opening_fingerprints": top_openings,
        },
        "domain_alignment": {
            "positive_math_records": math_positive,
            "positive_general_encyclopedia_records": encyclopedia_positive,
            "project_concept_records": project_concept_records,
            "generic_support_records": generic_support_records,
            "domain_drift_family_counts": dict(sorted(domain_drift_families.items())),
            "rag_bundle_records": rag_bundle_records,
            "known_core_false_refusals": known_core_refusals,
        },
        "privacy": {
            "record_body_emitted": False,
            "full_dataset_hashed": False,
            "sealed_body_accessed": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--data-log-level", default="INFO")
    parser.add_argument("--validation-log-level", default="INFO")
    parser.add_argument("--sft-log-level", default="INFO")
    parser.add_argument("--orchestrator-log-level", default="INFO")
    parser.add_argument("--preflight-log-level", default="INFO")
    parser.add_argument("--no-console-log", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = generate_run_id("sft-v6-vertical-gap")
    levels = resolve_module_log_levels(
        {
            "data": args.data_log_level,
            "validation": args.validation_log_level,
            "sft": args.sft_log_level,
            "orchestrator": args.orchestrator_log_level,
            "preflight": args.preflight_log_level,
            "pretrain": "OFF",
            "checkpoint": "OFF",
            "gpu": "OFF",
        }
    )
    loggers = configure_module_loggers(
        args.log_dir,
        run_id,
        levels,
        max_bytes=10 * 1024 * 1024,
        backup_count=5,
        console=not args.no_console_log,
    )
    try:
        records, prefix_sha256 = read_nonsealed_prefix(args.dataset)
        loggers["data"].info(
            "loaded non-sealed SFT v6 prefix",
            extra={
                "context": {
                    "dataset_path": str(args.dataset),
                    "audited_records": len(records),
                    "audited_prefix_sha256": prefix_sha256,
                    "sealed_body_accessed": False,
                }
            },
        )
        report = audit_v6_vertical_gap(records)
        report.update(
            {
                "run_id": run_id,
                "dataset_path": str(args.dataset),
                "audited_prefix_sha256": prefix_sha256,
            }
        )
        atomic_write_json(args.report, report)
        loggers["validation"].info(
            "vertical gap audit complete",
            extra={
                "context": {
                    "status": report["status"],
                    "hard_failures": report["hard_failures"],
                    "audited_records": report["audited_records"],
                }
            },
        )
        loggers["preflight"].info(
            "sealed boundary preserved",
            extra={
                "context": {
                    "maximum_records_read": AUDITED_RECORD_LIMIT,
                    "sealed_records_not_read": DECLARED_SEALED_RECORDS,
                }
            },
        )
        loggers["orchestrator"].info(
            "wrote aggregate gap report",
            extra={"context": {"report_path": str(args.report), "status": report["status"]}},
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["status"] == "passed" else 2
    except Exception:
        loggers["validation"].exception("SFT v6 vertical gap audit failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
