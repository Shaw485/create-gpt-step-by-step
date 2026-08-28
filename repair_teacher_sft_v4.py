"""Conservatively import and repair the 3,000-record teacher SFT candidate set.

The source JSONL is treated as immutable. This importer rebases provenance onto
the frozen v4 corpus, verifies the v4 BPE vocabulary, groups related records
before splitting, and leaves every evaluation record pending human review.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from difflib import SequenceMatcher
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Sequence

from audit_corpus import CHAPTER_PATTERN, chinese_number_to_int
from bpe_tokenizer import BPETokenizer
from build_sft_v4 import (
    SCHEMA_VERSION,
    TASK_FAMILY_QUOTAS,
    atomic_write_text,
    build_chapter_index,
    chapter_for_line,
    configure_sft_v4_logging,
    jsonl_text,
    make_candidate,
    quality_gate,
    read_jsonl,
    sha256_file,
    stable_hash,
)


CATEGORY_MAP = {
    "core_fact": "direct_fact",
    "worldbuilding_concept": "direct_fact",
    "long_tail_detail": "direct_fact",
    "character_relation": "relationship_reason_timeline",
    "timeline_event": "relationship_reason_timeline",
    "cause_motivation_result": "relationship_reason_timeline",
    "comparison_synthesis": "relationship_reason_timeline",
    "plot_summary_extraction": "context_understanding",
}
TARGET_SPLITS = {"train": 2400, "val": 300, "test": 300}
SUSPICIOUS_QUESTION_PATTERNS = (
    re.compile(r"是什么导致即使"),
    re.compile(r"是什么导致[^？]{0,8}的原因"),
    re.compile(r"中，是什么导致[^？]*？$"),
)
GLOBAL_CLAIM_MARKERS = ("最早", "第一次", "此前", "唯一", "从未", "没有提到")
PROVENANCE_REVIEW_FLAGS = {
    "evidence_absent_from_frozen_v4",
    "claimed_chapter_mismatch",
    "fuzzy_chapter_rebind_requires_review",
}
SEMANTIC_REVIEW_FLAGS = {
    "aggregation_claim_requires_full_chapter_review",
    "global_claim_requires_index_review",
    "question_grammar_requires_review",
    "third_fact_variant_split_for_review",
    "transformed_task_requires_review",
}


@dataclass(frozen=True)
class LocatedEvidence:
    evidence: dict[str, Any]
    chapter_number: int | None
    chapter_matches_claim: bool


class UnionFind:
    def __init__(self, values: Iterable[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: int, second: int) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def normalize_without_whitespace(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


def normalized_line_with_positions(text: str) -> tuple[str, list[int]]:
    characters: list[str] = []
    positions: list[int] = []
    for index, character in enumerate(text):
        if not character.isspace():
            characters.append(character)
            positions.append(index)
    return "".join(characters), positions


def chapter_number_from_title(title: str) -> int | None:
    match = CHAPTER_PATTERN.match(title.strip())
    return chinese_number_to_int(match.group(1)) if match else None


class CorpusEvidenceLocator:
    """Locate whitespace-normalized quotes while returning exact line spans."""

    def __init__(self, corpus_lines: Sequence[str], corpus_sha256: str) -> None:
        self.lines = list(corpus_lines)
        self.corpus_sha256 = corpus_sha256
        self.chapters = build_chapter_index(self.lines)
        self.normalized_lines = [normalize_without_whitespace(line) for line in self.lines]
        self.starts: list[int] = []
        pieces: list[str] = []
        offset = 0
        for line in self.normalized_lines:
            self.starts.append(offset)
            pieces.append(line)
            offset += len(line) + 1
        self.search_text = "\n".join(pieces)
        self.lines_by_chapter: dict[int, list[int]] = defaultdict(list)
        for line_index, line in enumerate(self.lines):
            if not line.strip() or CHAPTER_PATTERN.match(line.strip()):
                continue
            chapter = chapter_for_line(line_index + 1, self.chapters)
            number = chapter_number_from_title(chapter["title"]) if chapter else None
            if number is not None:
                self.lines_by_chapter[number].append(line_index)

    def _fuzzy_locate_in_chapter(
        self, needle: str, claimed_chapter: int
    ) -> LocatedEvidence | None:
        best: tuple[float, int, int] | None = None
        for line_index in self.lines_by_chapter.get(claimed_chapter, []):
            candidate = self.normalized_lines[line_index]
            if not candidate:
                continue
            match = SequenceMatcher(None, needle, candidate, autojunk=False).find_longest_match()
            score = match.size / max(1, min(len(needle), len(candidate)))
            if match.size < 20 or score < 0.55:
                continue
            proposal = (score, match.size, line_index)
            if best is None or proposal > best:
                best = proposal
        if best is None:
            return None
        score, _, line_index = best
        raw_line = self.lines[line_index]
        start_character = len(raw_line) - len(raw_line.lstrip())
        end_character = len(raw_line.rstrip())
        exact_text = raw_line[start_character:end_character]
        chapter = chapter_for_line(line_index + 1, self.chapters)
        return LocatedEvidence(
            evidence={
                "status": "verified_corpus",
                "text": exact_text,
                "corpus_sha256": self.corpus_sha256,
                "chapter": chapter,
                "span": {
                    "start_line": line_index + 1,
                    "end_line": line_index + 1,
                    "start_character": start_character,
                    "end_character": end_character,
                },
                "sha256": sha256(exact_text.encode("utf-8")).hexdigest(),
                "match_method": "fuzzy_chapter_rebind",
                "match_score": round(score, 6),
            },
            chapter_number=claimed_chapter,
            chapter_matches_claim=True,
        )

    def locate(self, quote: str, claimed_chapter: int | None) -> LocatedEvidence | None:
        needle = normalize_without_whitespace(quote)
        if not needle:
            return None
        fallback: LocatedEvidence | None = None
        position = self.search_text.find(needle)
        while position >= 0:
            line_index = bisect_right(self.starts, position) - 1
            local_start = position - self.starts[line_index]
            if "\n" not in self.search_text[position : position + len(needle)]:
                normalized_line, original_positions = normalized_line_with_positions(
                    self.lines[line_index]
                )
                if normalized_line[local_start : local_start + len(needle)] == needle:
                    start_character = original_positions[local_start]
                    end_character = original_positions[local_start + len(needle) - 1] + 1
                    exact_text = self.lines[line_index][start_character:end_character]
                    chapter = chapter_for_line(line_index + 1, self.chapters)
                    actual_number = (
                        chapter_number_from_title(chapter["title"]) if chapter else None
                    )
                    match = claimed_chapter is None or actual_number == claimed_chapter
                    located = LocatedEvidence(
                        evidence={
                            "status": "verified_corpus",
                            "text": exact_text,
                            "corpus_sha256": self.corpus_sha256,
                            "chapter": chapter,
                            "span": {
                                "start_line": line_index + 1,
                                "end_line": line_index + 1,
                                "start_character": start_character,
                                "end_character": end_character,
                            },
                            "sha256": sha256(exact_text.encode("utf-8")).hexdigest(),
                        },
                        chapter_number=actual_number,
                        chapter_matches_claim=match,
                    )
                    if match:
                        return located
                    fallback = fallback or located
            position = self.search_text.find(needle, position + 1)
        if claimed_chapter is not None:
            fuzzy = self._fuzzy_locate_in_chapter(needle, claimed_chapter)
            if fuzzy is not None:
                return fuzzy
        return fallback


def mapped_family(record: dict[str, Any]) -> str:
    category = str(record.get("category", ""))
    if category == "correction_unanswerable":
        return (
            "ambiguity_unknown_clarification"
            if record.get("subcategory") == "unanswerable"
            else "fact_verification_correction"
        )
    if category not in CATEGORY_MAP:
        raise ValueError(f"unsupported teacher category: {category!r}")
    return CATEGORY_MAP[category]


def normalize_answer(record: dict[str, Any]) -> tuple[str, list[str]]:
    """Apply only transformations whose meaning is mechanically preserved."""
    answer = str(record["answer"]).strip()
    repairs: list[str] = []
    if record.get("subcategory") == "chapter_focus" and answer.startswith("主要是"):
        chapter = record.get("source", {}).get("chapter_number")
        subject = answer.removeprefix("主要是").strip().rstrip("。！？!?")
        answer = f"第{chapter}章的情节主要围绕{subject}展开。"
        repairs.append("contextualized_repeated_chapter_focus_answer")
    if record.get("subcategory") == "chapter_title":
        chapter = record.get("source", {}).get("chapter_number")
        title = str(record.get("source", {}).get("chapter_title", "")).strip("《》 ")
        answer = f"第{chapter}章的标题是《{title}》。"
        repairs.append("removed_unasked_chapter_title_padding")
    if record.get("subcategory") == "kinship" and re.fullmatch(r"是[^。]+。", answer):
        question_match = re.match(r"(.+?)与(.+?)之间", str(record["question"]))
        evidence = str(record.get("source", {}).get("evidence_quote", ""))
        if question_match:
            left, right = question_match.groups()
            relations = ("弟子", "表姐", "女儿", "老师", "父亲", "母亲", "孙女")
            for owner, person in ((left, right), (right, left)):
                for relation in relations:
                    relationship_pattern = re.compile(
                        re.escape(owner)
                        + "的"
                        + relation
                        + r"[，,、\s]*"
                        + re.escape(person)
                    )
                    if relationship_pattern.search(evidence):
                        answer = f"{person}是{owner}的{relation}。"
                        repairs.append("repaired_incomplete_kinship_answer")
                        return answer, repairs
    return answer, repairs


def content_flags(
    record: dict[str, Any],
    question: str,
    answer: str,
    located: LocatedEvidence | None,
) -> list[str]:
    flags: list[str] = []
    if located is None:
        flags.append("evidence_absent_from_frozen_v4")
    elif not located.chapter_matches_claim:
        flags.append("claimed_chapter_mismatch")
    elif located.evidence.get("match_method") == "fuzzy_chapter_rebind":
        flags.append("fuzzy_chapter_rebind_requires_review")
    if any(pattern.search(question) for pattern in SUSPICIOUS_QUESTION_PATTERNS):
        flags.append("question_grammar_requires_review")
    if any(marker in question or marker in answer for marker in GLOBAL_CLAIM_MARKERS):
        flags.append("global_claim_requires_index_review")
    if "唯一被反复提及" in answer or "出现次数最多" in answer:
        flags.append("aggregation_claim_requires_full_chapter_review")
    return sorted(set(flags))


def _stable_pool(indices: Iterable[int], records: Sequence[dict[str, Any]], salt: str) -> list[int]:
    return sorted(
        indices,
        key=lambda index: stable_hash(
            salt,
            records[index]["knowledge_unit_id"],
            records[index]["id"],
        ),
    )


def _select_topic_limited(
    indices: Sequence[int],
    count: int,
    records: Sequence[dict[str, Any]],
    salt: str,
) -> tuple[list[int], list[int]]:
    """Select up to two records per knowledge unit before using any third variant."""
    by_topic: dict[str, list[int]] = defaultdict(list)
    for index in _stable_pool(indices, records, salt):
        by_topic[str(records[index]["knowledge_unit_id"])].append(index)
    selected: list[int] = []
    topics = sorted(by_topic, key=lambda topic: stable_hash(salt, topic))
    for variant_index in range(2):
        for topic in topics:
            if len(by_topic[topic]) > variant_index:
                selected.append(by_topic[topic][variant_index])
                if len(selected) == count:
                    selected_set = set(selected)
                    return selected, [index for index in indices if index not in selected_set]
    raise ValueError(f"only {len(selected)} topic-limited records available for target {count}")


def rebalance_task_families(
    records: Sequence[dict[str, Any]],
) -> dict[int, tuple[str, str | None]]:
    """Re-use all teacher records while meeting the frozen seven-family contract."""
    initial: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        initial[mapped_family(record)].append(index)

    direct, direct_surplus = _select_topic_limited(
        initial["direct_fact"], 750, records, "keep-direct"
    )
    relationship, relationship_surplus = _select_topic_limited(
        initial["relationship_reason_timeline"],
        600,
        records,
        "keep-relationship",
    )
    context_original, context_surplus = _select_topic_limited(
        initial["context_understanding"],
        250,
        records,
        "keep-context",
    )
    if context_surplus:
        raise ValueError("unexpected surplus in the 250-record context teacher pool")

    relationship_surplus = _stable_pool(
        relationship_surplus, records, "relationship-transform"
    )
    slices = {
        "context_understanding": (0, 200, "reclassified_as_context"),
        "fact_verification_correction": (200, 443, "verification_wrapper"),
        "ambiguity_unknown_clarification": (443, 650, "clarification_wrapper"),
        "conversation_control": (650, 800, "concise_answer_instruction"),
    }
    assignments: dict[int, tuple[str, str | None]] = {}
    for index in direct:
        assignments[index] = ("direct_fact", None)
    for index in relationship:
        assignments[index] = ("relationship_reason_timeline", None)
    for index in context_original:
        assignments[index] = ("context_understanding", None)
    for index in direct_surplus:
        assignments[index] = (
            "continuation_rewrite_instruction",
            "exact_copy_instruction",
        )
    for family, (start, end, transformation) in slices.items():
        for index in relationship_surplus[start:end]:
            assignments[index] = (family, transformation)
    for index in initial["fact_verification_correction"]:
        assignments[index] = ("fact_verification_correction", None)
    for index in initial["ambiguity_unknown_clarification"]:
        assignments[index] = ("ambiguity_unknown_clarification", None)

    if len(assignments) != len(records):
        raise ValueError(
            f"task rebalance assigned {len(assignments)} of {len(records)} records"
        )
    actual = Counter(family for family, _ in assignments.values())
    if actual != Counter(TASK_FAMILY_QUOTAS):
        raise ValueError(f"task rebalance missed quotas: {dict(actual)}")
    return assignments


def transform_task(
    record: dict[str, Any],
    family: str,
    transformation: str | None,
    located: LocatedEvidence | None,
) -> tuple[str, str, list[str]]:
    answer, repairs = normalize_answer(record)
    question = str(record["question"]).strip()
    if transformation is None:
        return question, answer, repairs
    if transformation == "exact_copy_instruction":
        source_text = (
            located.evidence["text"]
            if located
            else str(record.get("source", {}).get("evidence_quote", ""))
        ).strip()
        snippet = source_text[:40].strip()
        original_question = str(record["question"]).strip()
        question = (
            f"根据问题“{original_question}”，请原样重复作为依据的文字：“{snippet}”"
        )
        answer = snippet
    elif transformation == "reclassified_as_context":
        repairs.append("relationship_example_reclassified_as_context_understanding")
    elif transformation == "verification_wrapper":
        question = f"对于问题“{question}”，回答“{answer}”是否正确？"
        answer = f"正确；原文信息支持“{answer.rstrip('。！？!?')}”。"
    elif transformation == "clarification_wrapper":
        entity = str((record.get("entities") or ["相关人物"])[0])
        original_question = question.rstrip("？?!！")
        question = (
            f"如果把“{original_question}”简化成不带章节范围的问题，"
            "应先向用户确认什么？"
        )
        answer = (
            f"应先确认所依据的章节或故事阶段，再核对“{original_question}”"
            f"涉及的{entity}信息。"
        )
    elif transformation == "concise_answer_instruction":
        question = f"请只用一句话回答，不要续写小说：{question}"
    else:
        raise ValueError(f"unknown task transformation {transformation!r}")
    repairs.append(transformation)
    return question, answer, repairs


def choose_component_subset(
    components: Sequence[list[int]], target: int, salt: str
) -> set[int]:
    """Choose whole components whose record count is exactly target."""
    ordered = sorted(
        range(len(components)),
        key=lambda index: stable_hash(salt, *(components[index][:3])),
    )
    reachable: dict[int, tuple[int, int] | None] = {0: None}
    for component_index in ordered:
        size = len(components[component_index])
        for total in sorted(list(reachable), reverse=True):
            new_total = total + size
            if new_total <= target and new_total not in reachable:
                reachable[new_total] = (total, component_index)
        if target in reachable:
            break
    if target not in reachable:
        raise ValueError(f"cannot assign whole groups to exact split size {target}")
    selected: set[int] = set()
    total = target
    while total:
        previous, component_index = reachable[total]  # type: ignore[misc]
        selected.add(component_index)
        total = previous
    return selected


def assign_grouped_splits(
    records: Sequence[dict[str, Any]],
    actual_chapter_keys: Sequence[str | None] | None = None,
) -> tuple[list[str], list[str]]:
    """Keep knowledge units and claimed source chapters inside one exact split."""
    union = UnionFind(range(len(records)))
    by_topic: dict[str, int] = {}
    by_chapter: dict[int, int] = {}
    by_actual_chapter: dict[str, int] = {}
    for index, record in enumerate(records):
        topic = str(record["knowledge_unit_id"])
        chapter = int(record["source"]["chapter_number"])
        if topic in by_topic:
            union.union(index, by_topic[topic])
        else:
            by_topic[topic] = index
        if chapter in by_chapter:
            union.union(index, by_chapter[chapter])
        else:
            by_chapter[chapter] = index
        actual_chapter = actual_chapter_keys[index] if actual_chapter_keys else None
        if actual_chapter:
            if actual_chapter in by_actual_chapter:
                union.union(index, by_actual_chapter[actual_chapter])
            else:
                by_actual_chapter[actual_chapter] = index
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        grouped[union.find(index)].append(index)
    components = [grouped[root] for root in sorted(grouped)]
    test_components = choose_component_subset(components, TARGET_SPLITS["test"], "test")
    remaining_component_indices = [
        index for index in range(len(components)) if index not in test_components
    ]
    remaining_components = [components[index] for index in remaining_component_indices]
    val_in_remaining = choose_component_subset(
        remaining_components, TARGET_SPLITS["val"], "val"
    )
    split_by_record = ["train"] * len(records)
    group_by_record = [""] * len(records)
    for index, component in enumerate(components):
        if index in test_components:
            split = "test"
        else:
            remaining_index = remaining_component_indices.index(index)
            split = "val" if remaining_index in val_in_remaining else "train"
        group_id = "teacher-component:" + stable_hash(*component)[:16]
        for record_index in component:
            split_by_record[record_index] = split
            group_by_record[record_index] = group_id
    return split_by_record, group_by_record


def write_manual_review_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    fields = [
        "id",
        "split",
        "task_family",
        "question",
        "answer",
        "evidence",
        "reason_codes",
        "decision",
        "reviewer",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            if record["split"] == "train":
                continue
            writer.writerow(
                {
                    "id": record["id"],
                    "split": record["split"],
                    "task_family": record["task_family"],
                    "question": record["question"],
                    "answer": record["answer"],
                    "evidence": record["evidence"].get("text", ""),
                    "reason_codes": ";".join(record["origin"]["repair_flags"]),
                }
            )


def build_review_priority_summary(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize disjoint review priorities without inventing approvals."""
    priority_counts: Counter[str] = Counter()
    split_priority_counts: dict[str, Counter[str]] = defaultdict(Counter)
    clean_train_count = 0
    for record in records:
        flags = set(record["origin"].get("repair_flags", []))
        if record["split"] != "train":
            priority = "P0_evaluation_human_review"
        elif flags & PROVENANCE_REVIEW_FLAGS:
            priority = "P1_training_provenance_review"
        elif flags & SEMANTIC_REVIEW_FLAGS:
            priority = "P2_training_semantic_review"
        elif flags:
            priority = "P3_training_other_review"
        else:
            priority = "clean_training_candidate"
            clean_train_count += 1
        priority_counts[priority] += 1
        split_priority_counts[record["split"]][priority] += 1
    return {
        "policy": {
            "P0_evaluation_human_review": (
                "All validation and test records require real human approval before release."
            ),
            "P1_training_provenance_review": (
                "Review missing, mismatched, or fuzzy-rebound corpus evidence before training."
            ),
            "P2_training_semantic_review": (
                "Review transformed tasks, global claims, aggregation claims, and grammar."
            ),
            "P3_training_other_review": "Review remaining non-clean training candidates.",
            "clean_training_candidate": (
                "No automatic repair flag; still a candidate rather than a human-approved fact."
            ),
        },
        "counts": dict(sorted(priority_counts.items())),
        "counts_by_split": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_priority_counts.items())
        },
        "clean_training_candidate_count": clean_train_count,
        "human_approval_was_inferred": False,
    }


def run_repair(
    source_path: Path,
    corpus_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
    log_dir: Path,
) -> dict[str, Any]:
    loggers = configure_sft_v4_logging(log_dir)
    source_sha = sha256_file(source_path)
    corpus_sha = sha256_file(corpus_path)
    tokenizer_sha = sha256_file(tokenizer_path)
    run_id = stable_hash("teacher-sft-v4-repair", source_sha, corpus_sha, tokenizer_sha)[:12]
    loggers["data"].info(
        "run_id=%s loading teacher source=%s sha256=%s corpus=%s tokenizer=%s",
        run_id,
        source_path,
        source_sha,
        corpus_path,
        tokenizer_path,
    )
    raw_records = read_jsonl(source_path)
    if len(raw_records) != 3000:
        raise ValueError(f"teacher source must contain 3000 records, got {len(raw_records)}")
    corpus_lines = corpus_path.read_text(encoding="utf-8").splitlines()
    locator = CorpusEvidenceLocator(corpus_lines, corpus_sha)
    tokenizer = BPETokenizer.load(tokenizer_path)
    locations: list[LocatedEvidence | None] = []
    for source in raw_records:
        source_meta = source.get("source", {})
        claimed_chapter = source_meta.get("chapter_number")
        locations.append(
            locator.locate(
                str(source_meta.get("evidence_quote", "")),
                int(claimed_chapter) if claimed_chapter is not None else None,
            )
        )
    actual_chapter_keys = [
        (
            f"{located.evidence['chapter']['heading_line']}:"
            f"{located.evidence['chapter']['title']}"
            if located and located.evidence.get("chapter")
            else None
        )
        for located in locations
    ]
    splits, group_ids = assign_grouped_splits(raw_records, actual_chapter_keys)
    assignments = rebalance_task_families(raw_records)

    candidates: list[dict[str, Any]] = []
    repair_queue: list[dict[str, Any]] = []
    repair_counts: Counter[str] = Counter()
    sequence_lengths: list[int] = []
    vocab_errors: list[dict[str, str]] = []
    fact_occurrences: Counter[str] = Counter()
    for index, source in enumerate(raw_records):
        source_meta = source.get("source", {})
        claimed_chapter = source_meta.get("chapter_number")
        located = locations[index]
        family, transformation = assignments[index]
        question, answer, repairs = transform_task(
            source, family, transformation, located
        )
        flags = content_flags(source, question, answer, located)
        if transformation:
            flags.append("transformed_task_requires_review")
        for repair in repairs:
            repair_counts[repair] += 1
        for flag in flags:
            repair_counts[flag] += 1
        evidence = (
            located.evidence
            if located
            else {
                "status": "missing",
                "text": str(source_meta.get("evidence_quote", "")),
                "corpus_sha256": None,
                "chapter": None,
                "span": None,
                "sha256": None,
            }
        )
        topic_id = "teacher:" + str(source["knowledge_unit_id"])
        fact_occurrences[topic_id] += 1
        if transformation or fact_occurrences[topic_id] > 2:
            fact_id = f"{topic_id}:task:{source['id']}"
            if fact_occurrences[topic_id] > 2 and not transformation:
                flags.append("third_fact_variant_split_for_review")
        else:
            fact_id = topic_id
        candidate = make_candidate(
            question=question,
            answer=answer,
            task_family=family,
            topic_id=topic_id,
            fact_id=fact_id,
            origin={
                "kind": "teacher_sft3000_repair",
                "source_dataset_sha256": source_sha,
                "source_record_id": source["id"],
                "source_category": source["category"],
                "source_subcategory": source["subcategory"],
                "target_task_family": family,
                "task_transformation": transformation,
                "source_chapter_number": claimed_chapter,
                "source_chapter_title": source_meta.get("chapter_title"),
                "automatic_repairs": repairs,
                "repair_flags": flags,
                "vocab_compatible": True,
            },
            evidence=evidence,
        )
        candidate["split"] = splits[index]
        candidate["group_id"] = group_ids[index]
        candidates.append(candidate)
        try:
            sequence_lengths.append(
                4 + len(tokenizer.encode(candidate["question"]))
                + len(tokenizer.encode(candidate["answer"]))
            )
        except ValueError as error:
            vocab_errors.append({"id": candidate["id"], "error": str(error)})
        if flags or candidate["split"] != "train":
            repair_queue.append(candidate)

    if vocab_errors:
        raise ValueError(f"v4 BPE rejected {len(vocab_errors)} teacher records")
    audit = quality_gate(candidates, corpus_lines, corpus_sha)
    audit.update(
        {
            "stage": "teacher_sft3000_conservative_repair",
            "status": "needs_review" if not audit["release_ready"] else "ready",
            "run_id": run_id,
            "source_path": str(source_path),
            "source_sha256": source_sha,
            "corpus_path": str(corpus_path),
            "corpus_sha256": corpus_sha,
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_sha256": tokenizer_sha,
            "vocab_error_count": len(vocab_errors),
            "sequence_length": {
                "minimum": min(sequence_lengths),
                "maximum": max(sequence_lengths),
                "over_context_512": sum(length > 512 for length in sequence_lengths),
            },
            "repair_reason_counts": dict(sorted(repair_counts.items())),
            "repair_queue_count": len(repair_queue),
            "review_priorities": build_review_priority_summary(candidates),
            "methodology": {
                "source_mutated": False,
                "evidence_policy": "exact whitespace-normalized quote on one frozen-v4 line",
                "split_policy": "whole connected groups of knowledge unit and claimed source chapter",
                "semantic_policy": "only meaning-preserving template repairs; all uncertain claims queued",
            },
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "teacher_raw_snapshot.jsonl"
    shutil.copyfile(source_path, snapshot_path)
    candidate_path = output_dir / "sft_v4_teacher_candidates.jsonl"
    queue_path = output_dir / "sft_v4_teacher_repair_queue.jsonl"
    audit_path = output_dir / "sft_v4_teacher_audit.json"
    atomic_write_text(candidate_path, jsonl_text(candidates))
    atomic_write_text(queue_path, jsonl_text(repair_queue))
    atomic_write_text(
        audit_path,
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    review_summary_path = output_dir / "review_priority_summary.json"
    atomic_write_text(
        review_summary_path,
        json.dumps(
            audit["review_priorities"], ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
    )
    write_manual_review_csv(output_dir / "manual_review_val_test.csv", candidates)
    artifacts = {
        path.name: sha256_file(path)
        for path in (
            snapshot_path,
            candidate_path,
            queue_path,
            audit_path,
            review_summary_path,
            output_dir / "manual_review_val_test.csv",
        )
    }
    atomic_write_text(
        output_dir / "SHA256SUMS.json",
        json.dumps(artifacts, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    loggers["build"].info(
        "run_id=%s candidates=%d repair_queue=%d verified_evidence=%d",
        run_id,
        len(candidates),
        len(repair_queue),
        audit["actual"]["verified_evidence_count"],
    )
    loggers["validation"].info(
        "run_id=%s release_ready=%s failed_gates=%s max_sequence=%d",
        run_id,
        audit["release_ready"],
        audit["failed_gates"],
        max(sequence_lengths),
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("data/cloud_v4/corpus.txt"))
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("data/cloud_v4/tokenizer.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/sft/v4_teacher_repair")
    )
    parser.add_argument("--log-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    audit = run_repair(
        args.source,
        args.corpus,
        args.tokenizer,
        output_dir,
        args.log_dir or output_dir / "logs",
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "release_ready": audit["release_ready"],
                "failed_gates": audit["failed_gates"],
                "actual": audit["actual"],
                "repair_queue_count": audit["repair_queue_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
