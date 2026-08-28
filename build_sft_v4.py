from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from hashlib import sha256
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from audit_chapter_versions import find_boundaries


SCHEMA_VERSION = "sft_v4_candidate/1.0"
SOURCE_DATASET_PATH = Path("data/sft/sft_balanced_v3.jsonl")
CORPUS_PATH = Path("data/clean/doupo_stage3.txt")
CANDIDATE_DIR = Path("data/sft/v4")
CANDIDATE_PATH = CANDIDATE_DIR / "sft_v4_candidates.jsonl"
REJECTION_PATH = CANDIDATE_DIR / "sft_v4_rejections.jsonl"
REVIEW_QUEUE_PATH = CANDIDATE_DIR / "sft_v4_review_queue.jsonl"
AUDIT_REPORT_PATH = CANDIDATE_DIR / "sft_v4_audit.json"
SCHEMA_PATH = CANDIDATE_DIR / "sft_v4_schema.json"
RELEASE_DIR = Path("data/cloud_v4")
RELEASE_PATHS = {
    "train": RELEASE_DIR / "sft_train.jsonl",
    "val": RELEASE_DIR / "sft_val.jsonl",
    "test": RELEASE_DIR / "sft_test.jsonl",
}
RELEASE_MANIFEST_PATH = RELEASE_DIR / "sft_manifest.json"

TARGET_RECORD_COUNT = 3000
TARGET_SPLITS = {"train": 2400, "val": 300, "test": 300}
TASK_FAMILY_QUOTAS = {
    "direct_fact": 750,
    "relationship_reason_timeline": 600,
    "context_understanding": 450,
    "continuation_rewrite_instruction": 450,
    "fact_verification_correction": 300,
    "ambiguity_unknown_clarification": 300,
    "conversation_control": 150,
}
MIN_TOPIC_COUNT = 1200
MAX_QUESTIONS_PER_FACT = 2
MIN_EVIDENCE_SHARE = 0.70
MAX_IDENTICAL_ANSWER_SHARE = 0.02
VALID_SPLITS = frozenset(TARGET_SPLITS)
VALID_REVIEW_STATUSES = frozenset({"pending", "approved", "rejected"})

LEGACY_FAMILY_MAP = {
    "grounded_fact": "direct_fact",
    "concept_identity": "direct_fact",
    "fact_verification": "fact_verification_correction",
    "explicit_instruction": "continuation_rewrite_instruction",
    "honest_unknown": "ambiguity_unknown_clarification",
    "conversation": "conversation_control",
}

CHAPTER_PATTERN = re.compile(
    r"^\s*第[〇零一二三四五六七八九十百千万两0-9]+章(?:\s+.*)?\s*$"
)


class SftV4ValidationError(ValueError):
    """Raised when a record is malformed or its provenance cannot be verified."""


class SftV4ReleaseBlocked(RuntimeError):
    """Raised when candidate data fails one or more release quality gates."""


def stable_hash(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log_level(value: str) -> int:
    if value.upper() == "OFF":
        return logging.CRITICAL + 1
    level = getattr(logging, value.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"invalid log level {value!r}")
    return level


def configure_module_logger(
    name: str,
    path: Path,
    env_name: str,
) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    for existing_handler in logger.handlers[:]:
        logger.removeHandler(existing_handler)
        existing_handler.close()
    logger.setLevel(parse_log_level(os.getenv(env_name, "INFO")))
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler = RotatingFileHandler(
        path,
        maxBytes=10_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if os.getenv("SFT_V4_CONSOLE_LOG", "1") == "1":
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    logger.propagate = False
    return logger


def configure_sft_v4_logging(log_dir: Path = Path("logs")) -> dict[str, logging.Logger]:
    """Create independently filterable data, build, and validation logs."""

    return {
        "data": configure_module_logger(
            "sft.v4.data",
            log_dir / "sft_v4_data.log",
            "SFT_V4_DATA_LOG_LEVEL",
        ),
        "build": configure_module_logger(
            "sft.v4.build",
            log_dir / "sft_v4_build.log",
            "SFT_V4_BUILD_LOG_LEVEL",
        ),
        "validation": configure_module_logger(
            "sft.v4.validation",
            log_dir / "sft_v4_validation.log",
            "SFT_V4_VALIDATION_LOG_LEVEL",
        ),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise SftV4ValidationError(
                f"{path}:{line_number} contains invalid JSON"
            ) from error
        if not isinstance(record, dict):
            raise SftV4ValidationError(
                f"{path}:{line_number} must contain a JSON object"
            )
        records.append(record)
    if not records:
        raise SftV4ValidationError(f"dataset is empty: {path}")
    return records


def jsonl_text(records: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def build_chapter_index(corpus_lines: Sequence[str]) -> list[tuple[int, str]]:
    boundaries = find_boundaries(list(corpus_lines))
    if boundaries:
        chapters = [
            (
                boundary["heading_index"] + 1,
                corpus_lines[boundary["heading_index"]].strip(),
            )
            for boundary in boundaries
        ]
    else:
        chapters = [
            (line_number, line.strip())
            for line_number, line in enumerate(corpus_lines, 1)
            if CHAPTER_PATTERN.fullmatch(line)
        ]
    if not chapters:
        raise SftV4ValidationError("no chapter headings were found in the corpus")
    return chapters


def chapter_for_line(
    source_line: int,
    chapters: Sequence[tuple[int, str]],
) -> dict[str, Any] | None:
    positions = [line_number for line_number, _ in chapters]
    index = bisect_right(positions, source_line) - 1
    if index < 0:
        return None
    line_number, title = chapters[index]
    return {"title": title, "heading_line": line_number}


def build_evidence_provenance(
    source: dict[str, Any],
    corpus_lines: Sequence[str],
    corpus_sha256: str,
    chapters: Sequence[tuple[int, str]],
) -> dict[str, Any]:
    evidence_text = source.get("evidence") or source.get("evidence_text")
    source_line = source.get("source_line")
    evidence_type = source.get("evidence_type")

    if source_line is not None and evidence_text:
        if not isinstance(source_line, int) or not 1 <= source_line <= len(corpus_lines):
            raise SftV4ValidationError(
                f"record {source.get('id', '<import>')} has invalid source_line"
            )
        corpus_line = corpus_lines[source_line - 1]
        start_character = corpus_line.find(str(evidence_text))
        if start_character < 0:
            raise SftV4ValidationError(
                f"record {source.get('id', '<import>')} evidence is absent "
                f"from source line {source_line}"
            )
        chapter = chapter_for_line(source_line, chapters)
        if chapter is None:
            raise SftV4ValidationError(
                f"record {source.get('id', '<import>')} has no preceding chapter"
            )
        evidence_text = str(evidence_text)
        return {
            "status": "verified_corpus",
            "text": evidence_text,
            "corpus_sha256": corpus_sha256,
            "chapter": chapter,
            "span": {
                "start_line": source_line,
                "end_line": source_line,
                "start_character": start_character,
                "end_character": start_character + len(evidence_text),
            },
            "sha256": sha256(evidence_text.encode("utf-8")).hexdigest(),
        }

    if evidence_type == "verified_absence":
        return {
            "status": "verified_absence_legacy_claim",
            "text": str(evidence_text or ""),
            "corpus_sha256": corpus_sha256,
            "chapter": None,
            "span": None,
            "sha256": None,
        }

    if evidence_type == "curated_behavior":
        return {
            "status": "legacy_behavior_claim_unreviewed",
            "text": (
                "Legacy data labeled this as curated behavior; "
                "v4 has not verified or recorded human approval."
            ),
            "corpus_sha256": None,
            "chapter": None,
            "span": None,
            "sha256": None,
        }

    return {
        "status": "missing",
        "text": str(evidence_text or ""),
        "corpus_sha256": None,
        "chapter": None,
        "span": None,
        "sha256": None,
    }


def schema_document() -> dict[str, Any]:
    """Return the stable public record contract used by candidate and release data."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.invalid/create-gpt-step-by-step/sft-v4.schema.json",
        "title": "SFT v4 candidate record",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "id",
            "question",
            "answer",
            "task_family",
            "topic_id",
            "fact_id",
            "group_id",
            "split",
            "origin",
            "evidence",
            "review",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "id": {"type": "string", "minLength": 1},
            "question": {"type": "string", "minLength": 1},
            "answer": {"type": "string", "minLength": 1},
            "task_family": {"enum": list(TASK_FAMILY_QUOTAS)},
            "topic_id": {"type": "string", "minLength": 1},
            "fact_id": {"type": "string", "minLength": 1},
            "group_id": {"type": "string", "minLength": 1},
            "split": {"enum": list(TARGET_SPLITS)},
            "origin": {"type": "object"},
            "evidence": {"type": "object"},
            "review": {"type": "object"},
        },
        "quality_contract": {
            "record_count": TARGET_RECORD_COUNT,
            "split_counts": TARGET_SPLITS,
            "task_family_counts": TASK_FAMILY_QUOTAS,
            "minimum_topic_count": MIN_TOPIC_COUNT,
            "maximum_questions_per_fact": MAX_QUESTIONS_PER_FACT,
            "minimum_verified_evidence_share": MIN_EVIDENCE_SHARE,
            "maximum_identical_answer_share": MAX_IDENTICAL_ANSWER_SHARE,
            "evaluation_review_status": "approved",
            "grouping": "topic and source chapter must not cross splits",
        },
    }


def deterministic_split(group_id: str) -> str:
    bucket = int(stable_hash("sft-v4-split", group_id)[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def make_group_id(topic_id: str, evidence: dict[str, Any]) -> str:
    chapter = evidence.get("chapter")
    if evidence.get("status") == "verified_corpus" and isinstance(chapter, dict):
        return f"chapter:{chapter['heading_line']}:{chapter['title']}"
    return f"topic:{topic_id}"


def stable_record_id(record: dict[str, Any]) -> str:
    return "v4_" + stable_hash(
        SCHEMA_VERSION,
        record["question"],
        record["answer"],
        record["task_family"],
        record["topic_id"],
        record["fact_id"],
    )[:20]


def normalize_review(review: Any = None) -> dict[str, Any]:
    if not isinstance(review, dict):
        review = {}
    status = str(review.get("status", "pending"))
    if status not in VALID_REVIEW_STATUSES:
        raise SftV4ValidationError(f"invalid review status {status!r}")
    return {
        "status": status,
        "reviewer": review.get("reviewer"),
        "reviewed_at": review.get("reviewed_at"),
        "notes": review.get(
            "notes",
            "Pending independent v4 human review; legacy labels are not approval.",
        ),
    }


def make_candidate(
    *,
    question: str,
    answer: str,
    task_family: str,
    topic_id: str,
    fact_id: str,
    origin: dict[str, Any],
    evidence: dict[str, Any],
    review: Any = None,
) -> dict[str, Any]:
    question = question.strip()
    answer = answer.strip()
    topic_id = str(topic_id).strip()
    fact_id = str(fact_id).strip()
    if not question or not answer or not topic_id or not fact_id:
        raise SftV4ValidationError(
            "question, answer, topic_id, and fact_id must be non-empty"
        )
    if task_family not in TASK_FAMILY_QUOTAS:
        raise SftV4ValidationError(f"unknown task_family {task_family!r}")
    group_id = make_group_id(topic_id, evidence)
    record = {
        "schema_version": SCHEMA_VERSION,
        "id": "",
        "question": question,
        "answer": answer,
        "task_family": task_family,
        "topic_id": topic_id,
        "fact_id": fact_id,
        "group_id": group_id,
        "split": deterministic_split(group_id),
        "origin": origin,
        "evidence": evidence,
        "review": normalize_review(review),
    }
    record["id"] = stable_record_id(record)
    return record


def import_legacy_v3(
    path: Path,
    corpus_lines: Sequence[str],
    corpus_sha256: str,
    chapters: Sequence[tuple[int, str]],
) -> list[dict[str, Any]]:
    candidates = []
    for source in read_jsonl(path):
        legacy_family = source.get("task_family")
        if legacy_family not in LEGACY_FAMILY_MAP:
            raise SftV4ValidationError(
                f"legacy record {source.get('id')} has unknown family {legacy_family!r}"
            )
        topic = str(source.get("topic", "")).strip()
        evidence = build_evidence_provenance(
            source, corpus_lines, corpus_sha256, chapters
        )
        candidates.append(
            make_candidate(
                question=str(source.get("question", "")),
                answer=str(source.get("answer", "")),
                task_family=LEGACY_FAMILY_MAP[legacy_family],
                topic_id=f"legacy:{topic}",
                fact_id=f"legacy:{topic}",
                origin={
                    "kind": "legacy_v3_import",
                    "source_dataset": str(path),
                    "source_record_id": source.get("id"),
                    "generation_method": source.get("generation_method"),
                },
                evidence=evidence,
            )
        )
    return candidates


def import_external_candidates(
    path: Path,
    corpus_lines: Sequence[str],
    corpus_sha256: str,
    chapters: Sequence[tuple[int, str]],
) -> list[dict[str, Any]]:
    """Import a simple JSONL contract without inventing missing provenance.

    Required fields are question, answer, task_family, and topic_id (or topic).
    fact_id defaults to topic_id. Evidence becomes verified only when both an exact
    corpus substring and its one-based source_line are supplied.
    """

    candidates = []
    for line_number, source in enumerate(read_jsonl(path), 1):
        family = str(source.get("task_family", ""))
        family = LEGACY_FAMILY_MAP.get(family, family)
        topic = str(source.get("topic_id", source.get("topic", ""))).strip()
        fact_id = str(source.get("fact_id", topic)).strip()
        evidence = build_evidence_provenance(
            source, corpus_lines, corpus_sha256, chapters
        )
        candidates.append(
            make_candidate(
                question=str(source.get("question", "")),
                answer=str(source.get("answer", "")),
                task_family=family,
                topic_id=topic,
                fact_id=fact_id,
                origin={
                    "kind": "external_import",
                    "source_dataset": str(path),
                    "source_line_number": line_number,
                    "source_record_id": source.get("id"),
                },
                evidence=evidence,
                review=source.get("review"),
            )
        )
    return candidates


def candidate_priority(record: dict[str, Any]) -> tuple[int, str]:
    origin_priority = 0 if record["origin"].get("kind") == "external_import" else 1
    return origin_priority, stable_hash(
        "sft-v4-candidate-priority",
        record["fact_id"],
        record["task_family"],
        record["question"],
    )


def deduplicate_and_limit_facts(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rejections: list[dict[str, Any]] = []
    by_question: dict[str, dict[str, Any]] = {}
    for record in sorted(records, key=candidate_priority):
        normalized_question = " ".join(record["question"].split())
        if normalized_question in by_question:
            rejections.append(
                {
                    "id": record["id"],
                    "question": record["question"],
                    "fact_id": record["fact_id"],
                    "reason": "duplicate_question",
                }
            )
            continue
        by_question[normalized_question] = record

    by_fact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in by_question.values():
        by_fact[record["fact_id"]].append(record)

    selected: list[dict[str, Any]] = []
    for fact_id in sorted(by_fact):
        ordered = sorted(by_fact[fact_id], key=candidate_priority)
        kept: list[dict[str, Any]] = []
        seen_families: set[str] = set()
        for record in ordered:
            if record["task_family"] in seen_families:
                continue
            kept.append(record)
            seen_families.add(record["task_family"])
            if len(kept) == MAX_QUESTIONS_PER_FACT:
                break
        if len(kept) < MAX_QUESTIONS_PER_FACT:
            for record in ordered:
                if record in kept:
                    continue
                kept.append(record)
                if len(kept) == MAX_QUESTIONS_PER_FACT:
                    break
        kept_ids = {record["id"] for record in kept}
        selected.extend(kept)
        for record in ordered:
            if record["id"] not in kept_ids:
                rejections.append(
                    {
                        "id": record["id"],
                        "question": record["question"],
                        "fact_id": fact_id,
                        "reason": "more_than_two_questions_for_fact",
                    }
                )

    selected.sort(key=lambda row: (row["split"], row["task_family"], row["id"]))
    rejections.sort(key=lambda row: (row["reason"], row["id"]))
    return selected, rejections


def _validate_record_shape(record: dict[str, Any]) -> None:
    required = set(schema_document()["required"])
    missing = required - record.keys()
    if missing:
        raise SftV4ValidationError(
            f"record {record.get('id', '<unknown>')} missing fields {sorted(missing)}"
        )
    unexpected = set(record) - required
    if unexpected:
        raise SftV4ValidationError(
            f"record {record.get('id', '<unknown>')} has unexpected fields "
            f"{sorted(unexpected)}"
        )
    if record["schema_version"] != SCHEMA_VERSION:
        raise SftV4ValidationError(f"record {record['id']} has wrong schema version")
    if record["task_family"] not in TASK_FAMILY_QUOTAS:
        raise SftV4ValidationError(f"record {record['id']} has invalid task family")
    if record["split"] not in VALID_SPLITS:
        raise SftV4ValidationError(f"record {record['id']} has invalid split")
    for field in ("id", "question", "answer", "topic_id", "fact_id", "group_id"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise SftV4ValidationError(f"record {record['id']} has empty {field}")
    if not isinstance(record["origin"], dict):
        raise SftV4ValidationError(f"record {record['id']} has invalid origin")
    if not isinstance(record["evidence"], dict):
        raise SftV4ValidationError(f"record {record['id']} has invalid evidence")
    if not isinstance(record["review"], dict):
        raise SftV4ValidationError(f"record {record['id']} has invalid review")
    if record["review"].get("status") not in VALID_REVIEW_STATUSES:
        raise SftV4ValidationError(f"record {record['id']} has invalid review status")


def verify_evidence_record(
    record: dict[str, Any],
    corpus_lines: Sequence[str],
    corpus_sha256: str,
    chapters: Sequence[tuple[int, str]],
) -> bool:
    evidence = record["evidence"]
    if evidence.get("status") != "verified_corpus":
        return False
    span = evidence.get("span")
    chapter = evidence.get("chapter")
    text = evidence.get("text")
    if not isinstance(span, dict) or not isinstance(chapter, dict) or not isinstance(text, str):
        raise SftV4ValidationError(f"record {record['id']} has incomplete provenance")
    if evidence.get("corpus_sha256") != corpus_sha256:
        raise SftV4ValidationError(f"record {record['id']} corpus hash mismatch")
    start_line = span.get("start_line")
    end_line = span.get("end_line")
    start_character = span.get("start_character")
    end_character = span.get("end_character")
    if start_line != end_line or not isinstance(start_line, int):
        raise SftV4ValidationError(f"record {record['id']} has unsupported line span")
    if not 1 <= start_line <= len(corpus_lines):
        raise SftV4ValidationError(f"record {record['id']} line span is out of range")
    corpus_line = corpus_lines[start_line - 1]
    if not isinstance(start_character, int) or not isinstance(end_character, int):
        raise SftV4ValidationError(f"record {record['id']} has invalid character span")
    if corpus_line[start_character:end_character] != text:
        raise SftV4ValidationError(f"record {record['id']} evidence span mismatch")
    if evidence.get("sha256") != sha256(text.encode("utf-8")).hexdigest():
        raise SftV4ValidationError(f"record {record['id']} evidence hash mismatch")
    if not chapter.get("title") or not isinstance(chapter.get("heading_line"), int):
        raise SftV4ValidationError(f"record {record['id']} chapter metadata is invalid")
    expected_chapter = chapter_for_line(start_line, chapters)
    if chapter != expected_chapter:
        raise SftV4ValidationError(f"record {record['id']} chapter provenance mismatch")
    return True


def quality_gate(
    records: Sequence[dict[str, Any]],
    corpus_lines: Sequence[str],
    corpus_sha256: str,
) -> dict[str, Any]:
    if not records:
        raise SftV4ValidationError("candidate dataset is empty")
    for record in records:
        _validate_record_shape(record)

    ids = [record["id"] for record in records]
    questions = [" ".join(record["question"].split()) for record in records]
    if len(set(ids)) != len(ids):
        raise SftV4ValidationError("record IDs are not unique")
    if len(set(questions)) != len(questions):
        raise SftV4ValidationError("questions are not unique")

    chapters = build_chapter_index(corpus_lines)
    verified_evidence_count = sum(
        verify_evidence_record(record, corpus_lines, corpus_sha256, chapters)
        for record in records
    )
    split_counts = Counter(record["split"] for record in records)
    family_counts = Counter(record["task_family"] for record in records)
    fact_counts = Counter(record["fact_id"] for record in records)
    answer_counts = Counter(record["answer"] for record in records)
    topic_splits: dict[str, set[str]] = defaultdict(set)
    chapter_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        topic_splits[record["topic_id"]].add(record["split"])
        group_splits[record["group_id"]].add(record["split"])
        chapter = record["evidence"].get("chapter")
        if isinstance(chapter, dict):
            chapter_key = f"{chapter.get('heading_line')}:{chapter.get('title')}"
            chapter_splits[chapter_key].add(record["split"])

    leaked_topics = sorted(key for key, values in topic_splits.items() if len(values) > 1)
    leaked_chapters = sorted(
        key for key, values in chapter_splits.items() if len(values) > 1
    )
    leaked_groups = sorted(key for key, values in group_splits.items() if len(values) > 1)
    evaluation_records = [record for record in records if record["split"] != "train"]
    unapproved_evaluation_ids = [
        record["id"]
        for record in evaluation_records
        if record["review"].get("status") != "approved"
        or not record["review"].get("reviewer")
        or not record["review"].get("reviewed_at")
    ]
    evidence_share = verified_evidence_count / len(records)
    most_common_answer, most_common_answer_count = answer_counts.most_common(1)[0]
    most_common_answer_share = most_common_answer_count / len(records)
    maximum_fact_questions = max(fact_counts.values())

    gate_rows = [
        {
            "name": "record_count",
            "expected": TARGET_RECORD_COUNT,
            "actual": len(records),
            "passed": len(records) == TARGET_RECORD_COUNT,
        },
        {
            "name": "split_counts",
            "expected": TARGET_SPLITS,
            "actual": dict(split_counts),
            "passed": split_counts == Counter(TARGET_SPLITS),
        },
        {
            "name": "task_family_counts",
            "expected": TASK_FAMILY_QUOTAS,
            "actual": dict(family_counts),
            "passed": family_counts == Counter(TASK_FAMILY_QUOTAS),
        },
        {
            "name": "minimum_topic_count",
            "expected": MIN_TOPIC_COUNT,
            "actual": len(topic_splits),
            "passed": len(topic_splits) >= MIN_TOPIC_COUNT,
        },
        {
            "name": "maximum_questions_per_fact",
            "expected": MAX_QUESTIONS_PER_FACT,
            "actual": maximum_fact_questions,
            "passed": maximum_fact_questions <= MAX_QUESTIONS_PER_FACT,
        },
        {
            "name": "maximum_identical_answer_share",
            "expected": f"<{MAX_IDENTICAL_ANSWER_SHARE:.2%}",
            "actual": most_common_answer_share,
            "passed": most_common_answer_share < MAX_IDENTICAL_ANSWER_SHARE,
        },
        {
            "name": "minimum_verified_evidence_share",
            "expected": f">={MIN_EVIDENCE_SHARE:.0%}",
            "actual": evidence_share,
            "passed": evidence_share >= MIN_EVIDENCE_SHARE,
        },
        {
            "name": "topic_chapter_group_leakage",
            "expected": 0,
            "actual": len(leaked_topics) + len(leaked_chapters) + len(leaked_groups),
            "passed": not leaked_topics and not leaked_chapters and not leaked_groups,
        },
        {
            "name": "val_test_human_review",
            "expected": "all approved with reviewer and reviewed_at",
            "actual": len(unapproved_evaluation_ids),
            "passed": not unapproved_evaluation_ids,
        },
    ]
    deficits = {
        family: max(0, target - family_counts[family])
        for family, target in TASK_FAMILY_QUOTAS.items()
    }
    split_deficits = {
        split: max(0, target - split_counts[split])
        for split, target in TARGET_SPLITS.items()
    }
    failed_gates = [row["name"] for row in gate_rows if not row["passed"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "release_ready": not failed_gates,
        "failed_gates": failed_gates,
        "quality_gates": gate_rows,
        "actual": {
            "record_count": len(records),
            "split_counts": dict(split_counts),
            "task_family_counts": dict(family_counts),
            "topic_count": len(topic_splits),
            "fact_count": len(fact_counts),
            "maximum_questions_per_fact": maximum_fact_questions,
            "unique_answer_count": len(answer_counts),
            "most_common_answer": most_common_answer,
            "most_common_answer_count": most_common_answer_count,
            "most_common_answer_share": most_common_answer_share,
            "verified_evidence_count": verified_evidence_count,
            "verified_evidence_share": evidence_share,
            "evaluation_record_count": len(evaluation_records),
            "unapproved_evaluation_count": len(unapproved_evaluation_ids),
        },
        "gaps": {
            "records_missing": max(0, TARGET_RECORD_COUNT - len(records)),
            "topics_missing": max(0, MIN_TOPIC_COUNT - len(topic_splits)),
            "task_family_deficits": deficits,
            "split_deficits": split_deficits,
            "unapproved_evaluation_count": len(unapproved_evaluation_ids),
        },
        "leakage": {
            "topics": leaked_topics[:20],
            "chapters": leaked_chapters[:20],
            "groups": leaked_groups[:20],
        },
        "warnings": [
            "Imported or generated candidates are not human-approved by default.",
            "A passing candidate audit is required before cloud release export.",
            "Absence and behavioral claims do not count toward the 70% corpus-provenance gate.",
        ],
    }


def build_review_queue(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    queue = []
    for record in records:
        reasons = []
        if record["split"] in {"val", "test"}:
            reasons.append("evaluation_split_requires_human_approval")
        if record["evidence"].get("status") != "verified_corpus":
            reasons.append("missing_chapter_span_hash_provenance")
        if record["review"].get("status") != "approved":
            reasons.append("pending_human_review")
        priority = (
            0
            if record["split"] in {"val", "test"}
            else 1
            if record["evidence"].get("status") != "verified_corpus"
            else 2
        )
        queue.append(
            {
                "id": record["id"],
                "priority": priority,
                "split": record["split"],
                "task_family": record["task_family"],
                "topic_id": record["topic_id"],
                "fact_id": record["fact_id"],
                "question": record["question"],
                "answer": record["answer"],
                "evidence": record["evidence"],
                "review": record["review"],
                "reason_codes": reasons,
            }
        )
    queue.sort(key=lambda row: (row["priority"], row["split"], row["id"]))
    return queue


def release_records(
    records: Sequence[dict[str, Any]],
    audit: dict[str, Any],
    corpus_path: Path,
) -> dict[str, Any]:
    if not audit["release_ready"]:
        raise SftV4ReleaseBlocked(
            "SFT v4 release blocked by gates: " + ", ".join(audit["failed_gates"])
        )
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for split, path in RELEASE_PATHS.items():
        split_records = [record for record in records if record["split"] == split]
        atomic_write_text(path, jsonl_text(split_records))
        artifacts.append(
            {
                "path": str(path),
                "split": split,
                "record_count": len(split_records),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "release_ready": True,
        "corpus_path": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
        "artifacts": artifacts,
        "quality_gates": audit["quality_gates"],
    }
    atomic_write_text(
        RELEASE_MANIFEST_PATH,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(
        RELEASE_MANIFEST_PATH.with_suffix(RELEASE_MANIFEST_PATH.suffix + ".sha256"),
        sha256_file(RELEASE_MANIFEST_PATH) + "\n",
    )
    return manifest


def build_pipeline(
    *,
    source_path: Path,
    corpus_path: Path,
    import_paths: Sequence[Path] = (),
    candidate_dir: Path = CANDIDATE_DIR,
    release: bool = False,
    log_dir: Path = Path("logs"),
) -> dict[str, Any]:
    loggers = configure_sft_v4_logging(log_dir)
    data_logger = loggers["data"]
    build_logger = loggers["build"]
    validation_logger = loggers["validation"]
    run_id = "unavailable"
    try:
        corpus_text = corpus_path.read_text(encoding="utf-8")
        corpus_lines = corpus_text.splitlines()
        corpus_hash = sha256_file(corpus_path)
        chapters = build_chapter_index(corpus_lines)
        run_id = stable_hash(
            "sft-v4-run",
            sha256_file(source_path),
            corpus_hash,
            *(str(path) for path in import_paths),
        )[:12]
        data_logger.info(
            "run_id=%s loaded corpus path=%s chars=%d lines=%d chapters=%d sha256=%s",
            run_id,
            corpus_path,
            len(corpus_text),
            len(corpus_lines),
            len(chapters),
            corpus_hash,
        )

        raw_candidates = import_legacy_v3(
            source_path, corpus_lines, corpus_hash, chapters
        )
        data_logger.info(
            "run_id=%s imported legacy candidates path=%s count=%d",
            run_id,
            source_path,
            len(raw_candidates),
        )
        for import_path in import_paths:
            imported = import_external_candidates(
                import_path, corpus_lines, corpus_hash, chapters
            )
            raw_candidates.extend(imported)
            data_logger.info(
                "run_id=%s imported external candidates path=%s count=%d",
                run_id,
                import_path,
                len(imported),
            )

        candidates, rejections = deduplicate_and_limit_facts(raw_candidates)
        build_logger.info(
            "run_id=%s built candidate batch raw=%d accepted=%d rejected=%d",
            run_id,
            len(raw_candidates),
            len(candidates),
            len(rejections),
        )
        audit = quality_gate(candidates, corpus_lines, corpus_hash)
        audit.update(
            {
                "source_dataset": str(source_path),
                "source_dataset_sha256": sha256_file(source_path),
                "corpus_path": str(corpus_path),
                "corpus_sha256": corpus_hash,
                "raw_candidate_count": len(raw_candidates),
                "accepted_candidate_count": len(candidates),
                "rejected_candidate_count": len(rejections),
                "rejection_reason_counts": dict(
                    Counter(item["reason"] for item in rejections)
                ),
                "import_paths": [str(path) for path in import_paths],
            }
        )
        validation_logger.info(
            "run_id=%s audit complete release_ready=%s records=%d topics=%d evidence_share=%.4f "
            "failed_gates=%s",
            run_id,
            audit["release_ready"],
            audit["actual"]["record_count"],
            audit["actual"]["topic_count"],
            audit["actual"]["verified_evidence_share"],
            audit["failed_gates"],
        )

        candidate_path = candidate_dir / CANDIDATE_PATH.name
        rejection_path = candidate_dir / REJECTION_PATH.name
        review_queue_path = candidate_dir / REVIEW_QUEUE_PATH.name
        audit_path = candidate_dir / AUDIT_REPORT_PATH.name
        schema_path = candidate_dir / SCHEMA_PATH.name
        atomic_write_text(candidate_path, jsonl_text(candidates))
        atomic_write_text(rejection_path, jsonl_text(rejections))
        atomic_write_text(review_queue_path, jsonl_text(build_review_queue(candidates)))
        atomic_write_text(
            audit_path,
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        atomic_write_text(
            schema_path,
            json.dumps(schema_document(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
        )
        build_logger.info(
            "run_id=%s wrote candidates=%s rejections=%s review_queue=%s audit=%s schema=%s",
            run_id,
            candidate_path,
            rejection_path,
            review_queue_path,
            audit_path,
            schema_path,
        )

        if release:
            manifest = release_records(candidates, audit, corpus_path)
            build_logger.info(
                "run_id=%s released cloud SFT data manifest=%s artifacts=%s",
                run_id,
                RELEASE_MANIFEST_PATH,
                manifest["artifacts"],
            )
        return audit
    except Exception:
        validation_logger.exception("run_id=%s SFT v4 pipeline failed", run_id)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and audit honest SFT v4 candidates."
    )
    parser.add_argument("--source", type=Path, default=SOURCE_DATASET_PATH)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument(
        "--import-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Optional candidate JSONL; may be repeated.",
    )
    parser.add_argument("--candidate-dir", type=Path, default=CANDIDATE_DIR)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument(
        "--release",
        action="store_true",
        help="Export data/cloud_v4 splits only when every quality gate passes.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_pipeline(
        source_path=args.source,
        corpus_path=args.corpus,
        import_paths=args.import_jsonl,
        candidate_dir=args.candidate_dir,
        release=args.release,
        log_dir=args.log_dir,
    )
    print(json.dumps({
        "release_ready": audit["release_ready"],
        "accepted_candidates": audit["accepted_candidate_count"],
        "records_missing": audit["gaps"]["records_missing"],
        "topics_missing": audit["gaps"]["topics_missing"],
        "failed_gates": audit["failed_gates"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
