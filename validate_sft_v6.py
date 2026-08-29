"""Independently validate the 10,000-record SFT v6 release candidate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Sequence

from bpe_tokenizer import BPETokenizer
from build_sft_v5_1_no_math import is_arithmetic_text, is_math_topic_text
from build_sft_v6 import (
    DEFAULT_CORPUS,
    DEFAULT_OUTPUT,
    DEFAULT_TOKENIZER,
    DIMENSION_SPLIT_TARGETS,
    DIMENSION_TARGETS,
    META_MARKERS,
    REFUSAL_MARKERS,
    SCHEMA_VERSION,
    SPLIT_ORDER,
    SPLIT_TARGETS,
)
from training_runtime import (
    atomic_write_json,
    atomic_write_text,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
    resolve_module_log_levels,
)


DEFAULT_REPORT = Path("reports/milestones/018_sft_v6_10000/validation_report.json")
DEFAULT_RISK_QUEUE = Path("data/sft/v6/sft_v6_validation_risks.jsonl")
DEFAULT_LOG_DIR = Path("logs/sft_v6_validation")
SPECIAL_TOKENS = ("<BOS>", "<USER>", "<ASSISTANT>", "<EOS>", "<PAD>")
ALLOWED_EVIDENCE_STATUS = {
    "verified_train_corpus",
    "curated_project_fact",
    "not_applicable",
}
REQUIRED_FIELDS = {
    "schema_version",
    "id",
    "split",
    "primary_dimension",
    "task_family",
    "semantic_group",
    "question",
    "answer",
    "messages",
    "evidence",
    "coverage",
    "provenance",
    "review",
}
PUNCTUATION_PAIRS = (("（", "）"), ("(", ")"), ("《", "》"), ("【", "】"))


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is invalid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def canonical_question(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"[\s，。！？、；：“”‘’（）()《》【】\[\]：:,.!?;'\"`]+", "", normalized)
    return normalized


def sequence_metrics(
    record: dict[str, Any],
    tokenizer: BPETokenizer,
) -> tuple[int, int, int]:
    sequence_length = 1
    supervised_tokens = 0
    last_answer_tokens = 0
    for index, message in enumerate(record["messages"]):
        content_tokens = len(tokenizer.encode(message["content"]))
        sequence_length += 1 + content_tokens
        if message["role"] == "assistant":
            sequence_length += 1
            supervised_tokens += content_tokens + 1
            if index == len(record["messages"]) - 1:
                last_answer_tokens = content_tokens
    return sequence_length, supervised_tokens, last_answer_tokens


def add_risk(
    risks: list[dict[str, Any]],
    record: dict[str, Any],
    code: str,
    severity: str = "P0",
    detail: str = "",
) -> None:
    risks.append(
        {
            "id": record.get("id", ""),
            "split": record.get("split", ""),
            "task_family": record.get("task_family", ""),
            "severity": severity,
            "code": code,
            "detail": detail,
        }
    )


def validate_records(
    records: Sequence[dict[str, Any]],
    corpus_path: Path,
    tokenizer: BPETokenizer,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    corpus_text = corpus_path.read_text(encoding="utf-8")
    corpus_lines = corpus_text.splitlines()
    risks: list[dict[str, Any]] = []
    ids: list[str] = []
    questions: list[str] = []
    canonical_questions: list[str] = []
    semantic_splits: dict[str, set[str]] = defaultdict(set)
    evidence_splits: dict[str, set[str]] = defaultdict(set)
    chapter_splits: dict[int, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    dimension_counts: Counter[str] = Counter()
    dimension_splits: dict[str, Counter[str]] = defaultdict(Counter)
    family_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    general_answer_counts: Counter[str] = Counter()
    sequence_lengths: list[int] = []
    answer_lengths: list[int] = []
    evidence_lengths: list[int] = []
    supervised_by_split: Counter[str] = Counter()
    total_sequence_by_split: Counter[str] = Counter()
    all_entities: set[str] = set()
    all_concepts: set[str] = set()
    verified_chapters: set[tuple[int, str]] = set()
    refusal_records = 0
    multiturn_records = 0
    math_records = 0
    meta_records = 0
    unencodable_records = 0
    over_context_records = 0
    pending_reviews = 0
    long_context_records = 0
    punctuation_errors = 0

    for line_number, record in enumerate(records, 1):
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            add_risk(record=record, risks=risks, code="missing_fields", detail=",".join(sorted(missing)))
            continue
        ids.append(str(record["id"]))
        questions.append(str(record["question"]))
        canonical_questions.append(canonical_question(str(record["question"])))
        split = str(record["split"])
        dimension = str(record["primary_dimension"])
        split_counts[split] += 1
        dimension_counts[dimension] += 1
        dimension_splits[dimension][split] += 1
        family_counts[str(record["task_family"])] += 1
        semantic_splits[str(record["semantic_group"])].add(split)
        answer_counts[str(record["answer"])] += 1
        if dimension in {
            "natural_chat_and_multiturn",
            "project_study_and_general_concepts",
            "correction_grounded_unknown_and_capability_boundary",
        }:
            general_answer_counts[str(record["answer"])] += 1

        if record["schema_version"] != SCHEMA_VERSION:
            add_risk(risks, record, "schema_version_mismatch")
        messages = record["messages"]
        if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
            add_risk(risks, record, "invalid_message_count")
            continue
        expected_roles = ["user" if index % 2 == 0 else "assistant" for index in range(len(messages))]
        roles = [message.get("role") for message in messages if isinstance(message, dict)]
        if roles != expected_roles:
            add_risk(risks, record, "invalid_role_alternation")
            continue
        if any(not isinstance(message.get("content"), str) or not message["content"].strip() for message in messages):
            add_risk(risks, record, "empty_message_content")
            continue
        if record["question"] != messages[-2]["content"]:
            add_risk(risks, record, "question_last_user_mismatch")
        if record["answer"] != messages[-1]["content"]:
            add_risk(risks, record, "answer_last_assistant_mismatch")
        if len(messages) >= 4:
            multiturn_records += 1

        combined = "\n".join(message["content"] for message in messages)
        evidence_status = str(record.get("evidence", {}).get("status", ""))
        is_explicit_math = evidence_status != "verified_train_corpus" and (
            is_math_topic_text(combined) or is_arithmetic_text(combined)
        )
        if is_explicit_math:
            math_records += 1
            add_risk(risks, record, "math_training_record")
        if any(marker in combined for marker in META_MARKERS):
            meta_records += 1
            add_risk(risks, record, "forbidden_meta_wrapper")
        if any(marker in str(record["answer"]) for marker in REFUSAL_MARKERS):
            refusal_records += 1
        for opening, closing in PUNCTUATION_PAIRS:
            if str(record["answer"]).count(opening) != str(record["answer"]).count(closing):
                punctuation_errors += 1
                add_risk(risks, record, "unbalanced_answer_punctuation", detail=f"{opening}{closing}")
                break

        review_status = str(record.get("review", {}).get("status", ""))
        if review_status != "codex_generated_and_rule_checked":
            pending_reviews += 1
            add_risk(risks, record, "review_not_frozen")

        evidence = record["evidence"]
        status = evidence.get("status")
        if status not in ALLOWED_EVIDENCE_STATUS:
            add_risk(risks, record, "unknown_evidence_status")
        elif status == "verified_train_corpus":
            start_line = int(evidence.get("start_line", 0))
            end_line = int(evidence.get("end_line", 0))
            if not 1 <= start_line <= end_line <= len(corpus_lines):
                add_risk(risks, record, "evidence_line_range_invalid")
            else:
                expected_text = "\n".join(corpus_lines[start_line - 1 : end_line])
                if evidence.get("text") != expected_text:
                    add_risk(risks, record, "evidence_line_text_mismatch")
            if evidence.get("source_path") != str(DEFAULT_CORPUS):
                add_risk(risks, record, "evidence_source_not_formal_train")
            if evidence.get("sha256") != text_sha256(str(evidence.get("text", ""))):
                add_risk(risks, record, "evidence_sha256_mismatch")
            if str(evidence.get("text", "")) not in corpus_text:
                add_risk(risks, record, "evidence_not_exact_substring")
            chapter_key = int(evidence.get("heading_line", 0))
            chapter_splits[chapter_key].add(split)
            evidence_splits[str(evidence.get("sha256", ""))].add(split)
            verified_chapters.add((chapter_key, str(evidence.get("chapter_title", ""))))
            try:
                evidence_token_count = len(tokenizer.encode(str(evidence.get("text", "")).strip()))
                evidence_lengths.append(evidence_token_count)
                if 128 <= evidence_token_count <= 384:
                    long_context_records += 1
            except ValueError as error:
                unencodable_records += 1
                add_risk(risks, record, "unencodable_evidence", detail=str(error))
        else:
            text = str(evidence.get("text", ""))
            if evidence.get("sha256", "") != (text_sha256(text) if text else ""):
                add_risk(risks, record, "non_corpus_evidence_sha256_mismatch")

        coverage = record["coverage"]
        all_entities.update(str(value) for value in coverage.get("entities", []))
        all_concepts.update(str(value) for value in coverage.get("concepts", []))

        try:
            sequence_length, supervised_tokens, answer_tokens = sequence_metrics(record, tokenizer)
        except ValueError as error:
            unencodable_records += 1
            add_risk(risks, record, "unencodable_messages", detail=str(error))
            continue
        sequence_lengths.append(sequence_length)
        answer_lengths.append(answer_tokens)
        supervised_by_split[split] += supervised_tokens
        total_sequence_by_split[split] += sequence_length
        if sequence_length > 512:
            over_context_records += 1
            add_risk(risks, record, "sequence_over_512", detail=str(sequence_length))

    duplicate_ids = sum(count - 1 for count in Counter(ids).values() if count > 1)
    duplicate_questions = sum(count - 1 for count in Counter(questions).values() if count > 1)
    duplicate_canonical = sum(
        count - 1 for count in Counter(canonical_questions).values() if count > 1
    )
    semantic_leaks = {
        key: sorted(splits) for key, splits in semantic_splits.items() if len(splits) > 1
    }
    evidence_leaks = {
        key: sorted(splits) for key, splits in evidence_splits.items() if len(splits) > 1
    }
    chapter_leaks = {
        str(key): sorted(splits) for key, splits in chapter_splits.items() if len(splits) > 1
    }
    maximum_general_answer_repeat = max(general_answer_counts.values(), default=0)
    refusal_share = refusal_records / len(records) if records else 0.0
    medium_long_answers = sum(length >= 33 for length in answer_lengths)
    very_long_answers = sum(length >= 97 for length in answer_lengths)

    hard_failures: list[str] = []
    expected_checks = (
        (len(records) == 10000, "record_count"),
        (dict(split_counts) == SPLIT_TARGETS, "split_counts"),
        (dict(dimension_counts) == DIMENSION_TARGETS, "dimension_counts"),
        (
            all(dict(dimension_splits[name]) == expected for name, expected in DIMENSION_SPLIT_TARGETS.items()),
            "dimension_split_counts",
        ),
        (duplicate_ids == 0, "duplicate_ids"),
        (duplicate_questions == 0, "duplicate_questions"),
        (duplicate_canonical == 0, "canonical_question_duplicates"),
        (unencodable_records == 0, "unencodable_records"),
        (over_context_records == 0, "sequences_over_512"),
        (not semantic_leaks, "semantic_group_split_leakage"),
        (not evidence_leaks, "evidence_sha_split_leakage"),
        (not chapter_leaks, "chapter_split_leakage"),
        (multiturn_records >= 1000, "multiturn_coverage"),
        (long_context_records >= 1000, "long_context_coverage"),
        (len(all_entities) >= 60, "entity_coverage"),
        (len(all_concepts) >= 40, "concept_coverage"),
        (len(verified_chapters) >= 1000, "chapter_coverage"),
        (math_records == 0, "math_records"),
        (meta_records == 0, "meta_wrappers"),
        (pending_reviews == 0, "pending_reviews"),
        (punctuation_errors == 0, "punctuation_errors"),
        (refusal_share <= 0.05, "refusal_share"),
        (maximum_general_answer_repeat <= 5, "general_answer_repeat"),
        (300000 <= supervised_by_split["train"] <= 500000, "train_supervised_tokens"),
        (800000 <= sum(total_sequence_by_split.values()) <= 1500000, "total_sequence_tokens"),
        (medium_long_answers >= 1500, "medium_long_answer_coverage"),
    )
    for passed, name in expected_checks:
        if not passed:
            hard_failures.append(name)
    if risks:
        hard_failures.append("record_level_risks")

    report = {
        "schema_version": "sft-v6-validation-report/v1",
        "status": "passed" if not hard_failures else "needs_revision",
        "hard_failures": sorted(set(hard_failures)),
        "records": len(records),
        "split_counts": dict(split_counts),
        "dimension_counts": dict(dimension_counts),
        "dimension_split_counts": {
            name: dict(counter) for name, counter in dimension_splits.items()
        },
        "task_family_counts": dict(family_counts),
        "duplicate_ids": duplicate_ids,
        "duplicate_exact_questions": duplicate_questions,
        "duplicate_canonical_questions": duplicate_canonical,
        "semantic_group_split_leaks": len(semantic_leaks),
        "evidence_sha_split_leaks": len(evidence_leaks),
        "chapter_split_leaks": len(chapter_leaks),
        "unencodable_records": unencodable_records,
        "sequences_over_512": over_context_records,
        "min_sequence_tokens": min(sequence_lengths, default=0),
        "max_sequence_tokens": max(sequence_lengths, default=0),
        "mean_sequence_tokens": sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else 0,
        "total_sequence_tokens": sum(total_sequence_by_split.values()),
        "sequence_tokens_by_split": dict(total_sequence_by_split),
        "supervised_tokens_by_split": dict(supervised_by_split),
        "min_answer_tokens": min(answer_lengths, default=0),
        "max_answer_tokens": max(answer_lengths, default=0),
        "mean_answer_tokens": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
        "answers_33_tokens_or_more": medium_long_answers,
        "answers_97_tokens_or_more": very_long_answers,
        "multiturn_records": multiturn_records,
        "long_context_128_to_384_records": long_context_records,
        "verified_corpus_records": sum(
            record.get("evidence", {}).get("status") == "verified_train_corpus"
            for record in records
        ),
        "verified_chapters": len(verified_chapters),
        "entity_coverage": len(all_entities),
        "concept_coverage": len(all_concepts),
        "math_records": math_records,
        "meta_wrapper_records": meta_records,
        "pending_review_records": pending_reviews,
        "punctuation_error_records": punctuation_errors,
        "refusal_records": refusal_records,
        "refusal_share": refusal_share,
        "maximum_general_exact_answer_repeat": maximum_general_answer_repeat,
        "record_level_risks": len(risks),
        "risk_code_counts": dict(Counter(risk["code"] for risk in risks)),
    }
    return report, risks


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--risk-queue", type=Path, default=DEFAULT_RISK_QUEUE)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = generate_run_id("sft-v6-validation")
    loggers = configure_module_loggers(
        args.log_dir,
        run_id,
        resolve_module_log_levels(
            {"data": "INFO", "validation": "INFO", "sft": "INFO", "orchestrator": "INFO"}
        ),
        max_bytes=10 * 1024 * 1024,
        backup_count=5,
        console=True,
    )
    try:
        records = read_jsonl(args.dataset)
        tokenizer = BPETokenizer.load(args.tokenizer)
        loggers["data"].info(
            "loaded dataset=%s records=%d sha256=%s tokenizer=%s corpus=%s",
            args.dataset,
            len(records),
            file_sha256(args.dataset),
            args.tokenizer,
            args.corpus,
        )
        report, risks = validate_records(records, args.corpus, tokenizer)
        report.update(
            {
                "run_id": run_id,
                "dataset_path": str(args.dataset),
                "dataset_sha256": file_sha256(args.dataset),
                "corpus_path": str(args.corpus),
                "corpus_sha256": file_sha256(args.corpus),
                "tokenizer_path": str(args.tokenizer),
                "tokenizer_sha256": file_sha256(args.tokenizer),
                "risk_queue_path": str(args.risk_queue),
            }
        )
        atomic_write_json(args.report, report)
        atomic_write_text(
            args.risk_queue,
            "".join(json.dumps(risk, ensure_ascii=False, sort_keys=True) + "\n" for risk in risks),
        )
        loggers["validation"].info(
            "validation status=%s hard_failures=%s risks=%d sequences_over_512=%d "
            "train_supervised=%d total_sequence=%d",
            report["status"],
            report["hard_failures"],
            len(risks),
            report["sequences_over_512"],
            report["supervised_tokens_by_split"].get("train", 0),
            report["total_sequence_tokens"],
        )
        loggers["orchestrator"].info(
            "wrote report=%s risk_queue=%s dataset_sha256=%s",
            args.report,
            args.risk_queue,
            report["dataset_sha256"],
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "passed" else 2
    except Exception:
        loggers["validation"].exception("SFT v6 validation failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
