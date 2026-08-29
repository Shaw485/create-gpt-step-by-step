"""Audit an SFT JSONL dataset before encoding or formal fine-tuning.

The audit is intentionally tokenizer-aware and split-aware.  It reports facts
instead of silently rewriting data, so a reviewed source JSONL remains immutable
until a separate build step creates the next frozen dataset version.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from bpe_tokenizer import BPETokenizer
from training_runtime import (
    atomic_write_json,
    atomic_write_text,
    close_module_loggers,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
)


DEFAULT_DATASET = Path(
    "data/sft/v5_2_2_core_routing/"
    "sft_v5_2_2_core_routing_training_ready.jsonl"
)
DEFAULT_TOKENIZER = Path("data/scaling_a/bpe_3000/tokenizer.json")
DEFAULT_OUTPUT = Path(
    "reports/milestones/017_sft_data_readiness_audit/automatic_audit.json"
)
DEFAULT_RISK_OUTPUT = Path(
    "data/sft/v6_readiness/sft_v5_2_2_risk_queue.jsonl"
)
DEFAULT_LOG_DIR = Path("logs/sft_readiness_audit")
VALID_SPLITS = ("train", "val", "test")
REQUIRED_FIELDS = (
    "id",
    "question",
    "answer",
    "task_family",
    "split",
    "topic_id",
    "fact_id",
    "group_id",
)
REFUSAL_MARKERS = (
    "资料不足",
    "无法确定",
    "无法确认",
    "不能确定",
    "不能确认",
    "不能硬编",
    "没有足够",
    "请补充",
    "请先说明",
)
META_MARKERS = (
    "当前证据片段",
    "依据当前事实卡",
    "原问题是",
    "现只做局部证据核验",
    "如果用户只说",
    "如果用户现在只说",
)
STYLE_PREFIXES = (
    "请简短回答：",
    "用一句话回答：",
    "直接说结论：",
    "自然一点回答：",
    "像聊天一样回答：",
    "简单说：",
    "请直接回答：",
    "请友好地回答：",
    "请根据已知资料回答：",
    "用资料口径回答：",
    "这是小说事实题：",
    "只回答这个事实：",
)
CHAPTER_PATTERN = re.compile(
    r"第([零〇一二两三四五六七八九十百千万\d]+)章"
)

DIMENSION_FAMILIES = {
    "novel_factual_and_grounded_qa": {
        "direct_fact",
        "context_understanding",
        "fact_verification_correction",
        "relationship_reason_timeline",
        "evidence_entity_match",
        "novel_fact_anchor",
        "novel_known_entity",
        "novel_core_entity_v5_2",
        "novel_known_entity_v5_2",
        "novel_relation_v5_2",
        "novel_correction_v5_2",
        "novel_unknown_grounded_v5_2",
    },
    "natural_conversation": {
        "ambiguity_unknown_clarification",
        "conversation_control",
        "general_chat",
        "natural_conversation_repair",
    },
    "instruction_following": {
        "instruction_following",
        "instruction_following_repair",
    },
    "capability_and_uncertainty": {
        "capability_boundary",
        "capability_boundary_specific",
    },
    "concept_explanation": {"concept_explanation_repair"},
    "project_and_study_assistance": {"project_explanation", "study_planning"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--risk-output", type=Path, default=DEFAULT_RISK_OUTPUT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--console-log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the complete report instead of a compact completion summary.",
    )
    parser.add_argument("--data-log-level", default="INFO")
    parser.add_argument("--validation-log-level", default="INFO")
    parser.add_argument("--sft-log-level", default="INFO")
    parser.add_argument("--orchestrator-log-level", default="INFO")
    parser.add_argument("--log-max-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--log-backups", type=int, default=5)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(payload, dict):
                raise ValueError(f"record at {path}:{line_number} is not an object")
            payload["_audit_line_number"] = line_number
            records.append(payload)
    return records


def canonical_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def canonical_prompt(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    changed = True
    while changed:
        changed = False
        for prefix in STYLE_PREFIXES:
            normalized_prefix = unicodedata.normalize("NFKC", prefix)
            if normalized.startswith(normalized_prefix):
                normalized = normalized[len(normalized_prefix) :].strip()
                changed = True
                break
    return canonical_text(normalized)


def parse_chinese_integer(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    if not value or any(character not in digits and character not in units for character in value):
        return None
    if not any(character in units for character in value):
        return int("".join(str(digits[character]) for character in value))
    total = 0
    current = 0
    for character in value:
        if character in digits:
            current = digits[character]
        else:
            unit = units[character]
            total += (current or 1) * unit
            current = 0
    return total + current


def first_chapter_number(text: str) -> int | None:
    match = CHAPTER_PATTERN.search(text)
    return parse_chinese_integer(match.group(1)) if match else None


def has_unbalanced_punctuation(text: str) -> bool:
    return any(
        text.count(opening) != text.count(closing)
        for opening, closing in (("“", "”"), ("《", "》"), ("【", "】"), ("（", "）"))
    )


def chapter_order_mismatch(record: Mapping[str, Any]) -> bool:
    origin = record.get("origin")
    if not isinstance(origin, Mapping) or origin.get("source_subcategory") != "chapter_order":
        return False
    asked_number = first_chapter_number(str(record.get("question", "")))
    source_number = origin.get("source_chapter_number")
    return (
        asked_number is not None
        and isinstance(source_number, int)
        and source_number != asked_number + 1
    )


def percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def length_summary(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {key: 0 for key in ("min", "p50", "p90", "p95", "p99", "max", "mean")}
    return {
        "min": min(values),
        "p50": round(percentile(values, 0.50), 2),
        "p90": round(percentile(values, 0.90), 2),
        "p95": round(percentile(values, 0.95), 2),
        "p99": round(percentile(values, 0.99), 2),
        "max": max(values),
        "mean": round(statistics.fmean(values), 2),
    }


def repeated_values(values: Iterable[str], sample_limit: int = 20) -> dict[str, Any]:
    counts = Counter(value for value in values if value)
    duplicates = [(value, count) for value, count in counts.items() if count > 1]
    duplicates.sort(key=lambda item: (-item[1], item[0]))
    return {
        "duplicate_value_count": len(duplicates),
        "duplicate_record_count": sum(count - 1 for _, count in duplicates),
        "maximum_repeat": duplicates[0][1] if duplicates else 1 if counts else 0,
        "samples": [
            {"value": value, "count": count}
            for value, count in duplicates[:sample_limit]
        ],
    }


def values_crossing_splits(
    records: Sequence[Mapping[str, Any]],
    value_getter,
    sample_limit: int = 20,
) -> dict[str, Any]:
    split_sets: dict[str, set[str]] = defaultdict(set)
    value_counts: Counter[str] = Counter()
    for record in records:
        split = str(record.get("split", ""))
        value = str(value_getter(record) or "")
        if not value or split not in VALID_SPLITS:
            continue
        split_sets[value].add(split)
        value_counts[value] += 1
    leaking = [value for value, splits in split_sets.items() if len(splits) > 1]
    leaking.sort(key=lambda value: (-value_counts[value], value))
    return {
        "value_count": len(leaking),
        "record_count": sum(value_counts[value] for value in leaking),
        "samples": [
            {
                "value": value,
                "splits": sorted(split_sets[value]),
                "records": value_counts[value],
            }
            for value in leaking[:sample_limit]
        ],
    }


def chapter_key(record: Mapping[str, Any]) -> str:
    evidence = record.get("evidence")
    if not isinstance(evidence, Mapping):
        return ""
    chapter = evidence.get("chapter")
    if isinstance(chapter, Mapping):
        title = str(chapter.get("title", "")).strip()
        if title:
            return title
    origin = record.get("origin")
    if isinstance(origin, Mapping):
        number = origin.get("source_chapter_number")
        title = str(origin.get("source_chapter_title", "")).strip()
        if number is not None or title:
            return f"{number}|{title}"
    return ""


def marker_counts(records: Sequence[Mapping[str, Any]], markers: Sequence[str]) -> dict[str, Any]:
    by_marker: Counter[str] = Counter()
    record_count = 0
    by_family: Counter[str] = Counter()
    for record in records:
        text = f"{record.get('question', '')}\n{record.get('answer', '')}"
        matched = [marker for marker in markers if marker in text]
        if matched:
            record_count += 1
            by_family[str(record.get("task_family", ""))] += 1
            by_marker.update(matched)
    return {
        "records": record_count,
        "share": round(record_count / len(records), 6) if records else 0.0,
        "by_marker": dict(by_marker.most_common()),
        "by_family": dict(by_family.most_common()),
    }


def family_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("task_family", ""))].append(record)
    report: dict[str, Any] = {}
    for family, family_records in sorted(grouped.items()):
        answers = [str(record.get("answer", "")) for record in family_records]
        questions = [str(record.get("question", "")) for record in family_records]
        answer_counts = Counter(answers)
        report[family] = {
            "records": len(family_records),
            "share": round(len(family_records) / len(records), 6),
            "split_counts": dict(
                sorted(Counter(str(record.get("split", "")) for record in family_records).items())
            ),
            "unique_question_ratio": round(len(set(questions)) / len(questions), 6),
            "unique_answer_ratio": round(len(set(answers)) / len(answers), 6),
            "maximum_exact_answer_repeat": max(answer_counts.values()),
            "mean_question_characters": round(statistics.fmean(map(len, questions)), 2),
            "mean_answer_characters": round(statistics.fmean(map(len, answers)), 2),
            "refusal_records": sum(
                any(marker in answer for marker in REFUSAL_MARKERS) for answer in answers
            ),
        }
    return report


def dimension_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    family_counts = Counter(str(record.get("task_family", "")) for record in records)
    assigned: set[str] = set()
    dimensions: dict[str, Any] = {}
    for dimension, families in DIMENSION_FAMILIES.items():
        assigned.update(families)
        count = sum(family_counts[family] for family in families)
        dimensions[dimension] = {
            "records": count,
            "share": round(count / len(records), 6) if records else 0.0,
            "families_present": {
                family: family_counts[family]
                for family in sorted(families)
                if family_counts[family]
            },
        }
    dimensions["unmapped"] = {
        "records": sum(count for family, count in family_counts.items() if family not in assigned),
        "families_present": {
            family: count
            for family, count in sorted(family_counts.items())
            if family not in assigned
        },
    }
    return dimensions


def audit_records(
    records: Sequence[dict[str, Any]],
    tokenizer: BPETokenizer,
) -> dict[str, Any]:
    missing_fields: Counter[str] = Counter()
    blank_fields: Counter[str] = Counter()
    invalid_splits: Counter[str] = Counter()
    for record in records:
        for field in REQUIRED_FIELDS:
            if field not in record:
                missing_fields[field] += 1
            elif not str(record[field]).strip():
                blank_fields[field] += 1
        split = str(record.get("split", ""))
        if split not in VALID_SPLITS:
            invalid_splits[split] += 1

    question_tokens: list[int] = []
    answer_tokens: list[int] = []
    sequence_tokens: list[int] = []
    supervised_by_split: Counter[str] = Counter()
    sequence_by_split: Counter[str] = Counter()
    unencodable: list[dict[str, Any]] = []
    for record in records:
        question = str(record.get("question", ""))
        answer = str(record.get("answer", ""))
        try:
            question_ids = tokenizer.encode(question)
            answer_ids = tokenizer.encode(answer)
        except ValueError as error:
            unencodable.append(
                {
                    "id": str(record.get("id", "")),
                    "line": int(record.get("_audit_line_number", 0)),
                    "error": str(error),
                }
            )
            continue
        sequence_length = len(question_ids) + len(answer_ids) + 4
        supervised_length = len(answer_ids) + 1
        split = str(record.get("split", ""))
        question_tokens.append(len(question_ids))
        answer_tokens.append(len(answer_ids))
        sequence_tokens.append(sequence_length)
        supervised_by_split[split] += supervised_length
        sequence_by_split[split] += sequence_length

    split_counts = Counter(str(record.get("split", "")) for record in records)
    family_counts = Counter(str(record.get("task_family", "")) for record in records)
    review_statuses = Counter()
    evidence_statuses = Counter()
    evidence_sources = Counter()
    for record in records:
        review = record.get("review")
        review_statuses[
            str(review.get("status", "missing")) if isinstance(review, Mapping) else "missing"
        ] += 1
        evidence = record.get("evidence")
        if isinstance(evidence, Mapping):
            evidence_statuses[str(evidence.get("status", "missing"))] += 1
            evidence_sources[str(evidence.get("source", "missing"))] += 1
        else:
            evidence_statuses["missing"] += 1
            evidence_sources["missing"] += 1

    exact_questions = [str(record.get("question", "")) for record in records]
    normalized_questions = [canonical_text(question) for question in exact_questions]
    style_stripped_questions = [canonical_prompt(question) for question in exact_questions]
    exact_pairs = [
        f"{record.get('question', '')}\u241f{record.get('answer', '')}"
        for record in records
    ]
    answers = [str(record.get("answer", "")) for record in records]
    answer_counts = Counter(answers)
    top_answers = [
        {"answer": answer, "count": count, "share": round(count / len(records), 6)}
        for answer, count in answer_counts.most_common(20)
    ]

    known_names = (
        "萧炎",
        "药尘",
        "药老",
        "异火",
        "萧战",
        "韩枫",
        "紫研",
        "云韵",
        "美杜莎",
        "萧薰儿",
        "小医仙",
        "海波东",
        "纳兰嫣然",
        "云山",
        "古河",
    )
    entity_coverage: dict[str, dict[str, int]] = {}
    for name in known_names:
        counts: Counter[str] = Counter()
        for record in records:
            if name in f"{record.get('question', '')}\n{record.get('answer', '')}":
                counts[str(record.get("split", ""))] += 1
        entity_coverage[name] = dict(sorted(counts.items()))

    chapter_values = [chapter_key(record) for record in records]
    unique_chapters = sorted({value for value in chapter_values if value})
    multi_turn_records = sum(
        isinstance(record.get("messages"), list) and len(record["messages"]) > 2
        for record in records
    )

    return {
        "record_count": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "task_family_counts": dict(sorted(family_counts.items())),
        "record_integrity": {
            "missing_required_fields": dict(missing_fields),
            "blank_required_fields": dict(blank_fields),
            "invalid_splits": dict(invalid_splits),
            "duplicate_ids": repeated_values(str(record.get("id", "")) for record in records),
            "duplicate_exact_questions": repeated_values(exact_questions),
            "duplicate_normalized_questions": repeated_values(normalized_questions),
            "duplicate_style_stripped_questions": repeated_values(style_stripped_questions),
            "duplicate_exact_question_answer_pairs": repeated_values(exact_pairs),
        },
        "tokenization": {
            "vocab_size": tokenizer.vocab_size,
            "merge_count": len(tokenizer.merges),
            "unencodable_records": len(unencodable),
            "unencodable_samples": unencodable[:20],
            "question_tokens": length_summary(question_tokens),
            "answer_tokens": length_summary(answer_tokens),
            "sequence_tokens": length_summary(sequence_tokens),
            "sequences_over_256": sum(value > 256 for value in sequence_tokens),
            "sequences_over_384": sum(value > 384 for value in sequence_tokens),
            "sequences_over_512": sum(value > 512 for value in sequence_tokens),
            "questions_at_least_128": sum(value >= 128 for value in question_tokens),
            "answers_at_least_128": sum(value >= 128 for value in answer_tokens),
            "answers_at_most_16": sum(value <= 16 for value in answer_tokens),
            "supervised_tokens_by_split": dict(sorted(supervised_by_split.items())),
            "sequence_tokens_by_split": dict(sorted(sequence_by_split.items())),
        },
        "split_isolation": {
            "exact_question": values_crossing_splits(
                records, lambda record: str(record.get("question", ""))
            ),
            "normalized_question": values_crossing_splits(
                records, lambda record: canonical_text(str(record.get("question", "")))
            ),
            "style_stripped_question": values_crossing_splits(
                records, lambda record: canonical_prompt(str(record.get("question", "")))
            ),
            "exact_question_answer_pair": values_crossing_splits(
                records,
                lambda record: (
                    f"{record.get('question', '')}\u241f{record.get('answer', '')}"
                ),
            ),
            "topic_id": values_crossing_splits(records, lambda record: record.get("topic_id")),
            "fact_id": values_crossing_splits(records, lambda record: record.get("fact_id")),
            "group_id": values_crossing_splits(records, lambda record: record.get("group_id")),
            "evidence_chapter": values_crossing_splits(records, chapter_key),
            "evidence_sha256": values_crossing_splits(
                records,
                lambda record: (
                    record.get("evidence", {}).get("sha256", "")
                    if isinstance(record.get("evidence"), Mapping)
                    else ""
                ),
            ),
        },
        "provenance_and_review": {
            "review_status_counts": dict(review_statuses.most_common()),
            "evidence_status_counts": dict(evidence_statuses.most_common()),
            "evidence_source_counts": dict(evidence_sources.most_common()),
            "records_with_chapter_provenance": sum(bool(value) for value in chapter_values),
            "unique_evidence_chapters": len(unique_chapters),
        },
        "content_shape": {
            "multi_turn_records": multi_turn_records,
            "single_turn_records": len(records) - multi_turn_records,
            "chapter_number_records": sum(
                bool(CHAPTER_PATTERN.search(f"{record.get('question', '')}\n{record.get('answer', '')}"))
                for record in records
            ),
            "style_prefix_records": sum(
                str(record.get("question", "")).startswith(STYLE_PREFIXES)
                for record in records
            ),
            "verification_wrapper_records": sum(
                (
                    isinstance(record.get("origin"), Mapping)
                    and record["origin"].get("task_transformation") == "verification_wrapper"
                )
                for record in records
            ),
            "transformed_review_flag_records": sum(
                (
                    isinstance(record.get("origin"), Mapping)
                    and "transformed_task_requires_review"
                    in record["origin"].get("repair_flags", [])
                )
                for record in records
            ),
            "unbalanced_answer_punctuation_records": sum(
                has_unbalanced_punctuation(str(record.get("answer", "")))
                for record in records
            ),
            "chapter_order_mismatch_records": sum(
                chapter_order_mismatch(record) for record in records
            ),
            "refusal_markers": marker_counts(records, REFUSAL_MARKERS),
            "meta_markers": marker_counts(records, META_MARKERS),
            "unique_answer_ratio": round(len(answer_counts) / len(records), 6) if records else 0.0,
            "top_exact_answers": top_answers,
        },
        "families": family_report(records),
        "capability_dimensions": dimension_report(records),
        "known_entity_mentions": entity_coverage,
    }


def build_risk_queue(
    records: Sequence[Mapping[str, Any]],
    tokenizer: BPETokenizer,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    answer_counts = Counter(str(record.get("answer", "")) for record in records)
    queue: list[dict[str, Any]] = []
    flag_counts: Counter[str] = Counter()
    for record in records:
        flags: list[str] = []
        question = str(record.get("question", ""))
        answer = str(record.get("answer", ""))
        origin = record.get("origin")
        review = record.get("review")
        if isinstance(review, Mapping) and review.get("status") == "pending":
            flags.append("pending_semantic_review")
        if isinstance(origin, Mapping):
            if "transformed_task_requires_review" in origin.get("repair_flags", []):
                flags.append("transformed_task_requires_review")
            if origin.get("task_transformation") == "verification_wrapper":
                flags.append("verification_wrapper")
        if any(marker in f"{question}\n{answer}" for marker in META_MARKERS):
            flags.append("meta_evidence_wrapper")
        if question.startswith(STYLE_PREFIXES):
            flags.append("style_prefix_template")
        if has_unbalanced_punctuation(answer):
            flags.append("unbalanced_answer_punctuation")
        if chapter_order_mismatch(record):
            flags.append("chapter_order_mismatch")
        if answer_counts[answer] >= 10:
            flags.append("high_exact_answer_repetition")
        try:
            if len(tokenizer.encode(answer)) <= 4:
                flags.append("very_short_answer")
        except ValueError:
            flags.append("tokenizer_incompatible")
        if not flags:
            continue
        flag_counts.update(flags)
        critical = {
            "chapter_order_mismatch",
            "unbalanced_answer_punctuation",
            "tokenizer_incompatible",
        }
        high = {
            "verification_wrapper",
            "transformed_task_requires_review",
            "meta_evidence_wrapper",
        }
        priority = "P0" if critical.intersection(flags) else "P1" if high.intersection(flags) else "P2"
        queue.append(
            {
                "id": str(record.get("id", "")),
                "line": int(record.get("_audit_line_number", 0)),
                "split": str(record.get("split", "")),
                "task_family": str(record.get("task_family", "")),
                "priority": priority,
                "flags": flags,
            }
        )
    queue.sort(key=lambda item: (item["priority"], item["split"], item["task_family"], item["id"]))
    return queue, flag_counts


def main() -> None:
    args = parse_args()
    run_id = generate_run_id("sft-readiness-audit")
    loggers = configure_module_loggers(
        args.log_dir,
        run_id,
        {
            "data": args.data_log_level,
            "validation": args.validation_log_level,
            "sft": args.sft_log_level,
            "orchestrator": args.orchestrator_log_level,
            "preflight": "OFF",
            "pretrain": "OFF",
            "checkpoint": "OFF",
            "gpu": "OFF",
        },
        max_bytes=args.log_max_bytes,
        backup_count=args.log_backups,
        console=args.console_log,
    )
    try:
        records = load_jsonl(args.dataset)
        tokenizer = BPETokenizer.load(args.tokenizer)
        loggers["data"].info(
            "SFT audit inputs loaded",
            extra={
                "context": {
                    "dataset": str(args.dataset),
                    "records": len(records),
                    "tokenizer": str(args.tokenizer),
                    "vocab_size": tokenizer.vocab_size,
                }
            },
        )
        audit = audit_records(records, tokenizer)
        risk_queue, risk_flag_counts = build_risk_queue(records, tokenizer)
        payload = {
            "schema_version": "sft-readiness-audit/v1",
            "status": "complete",
            "run_id": run_id,
            "dataset_path": str(args.dataset),
            "dataset_sha256": file_sha256(args.dataset),
            "tokenizer_path": str(args.tokenizer),
            "tokenizer_sha256": file_sha256(args.tokenizer),
            "risk_queue": {
                "path": str(args.risk_output),
                "records": len(risk_queue),
                "priority_counts": dict(
                    sorted(Counter(item["priority"] for item in risk_queue).items())
                ),
                "flag_counts": dict(risk_flag_counts.most_common()),
            },
            **audit,
        }
        args.risk_output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            args.risk_output,
            "".join(
                json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                for item in risk_queue
            ),
        )
        atomic_write_json(args.output, payload)
        loggers["validation"].info(
            "SFT audit checks complete",
            extra={
                "context": {
                    "records": audit["record_count"],
                    "unencodable_records": audit["tokenization"]["unencodable_records"],
                    "sequences_over_512": audit["tokenization"]["sequences_over_512"],
                    "normalized_question_leaks": audit["split_isolation"][
                        "normalized_question"
                    ]["value_count"],
                }
            },
        )
        loggers["orchestrator"].info(
            "SFT readiness artifact written",
            extra={"context": {"output": str(args.output)}},
        )
        console_payload = payload if args.print_json else {
            "status": payload["status"],
            "records": payload["record_count"],
            "risk_records": payload["risk_queue"]["records"],
            "priority_counts": payload["risk_queue"]["priority_counts"],
            "unencodable_records": payload["tokenization"]["unencodable_records"],
            "output": str(args.output),
            "risk_output": str(args.risk_output),
        }
        print(json.dumps(console_payload, ensure_ascii=False, indent=2))
    except Exception:
        loggers["validation"].exception("SFT readiness audit failed")
        raise
    finally:
        close_module_loggers(loggers)


if __name__ == "__main__":
    main()
