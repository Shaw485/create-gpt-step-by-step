"""Build the 10,000-record, evidence-aware SFT v6 dataset.

The builder reads only the formal pretraining train split for novel excerpts.
It creates fresh SFT splits by chapter so the sealed test never shares a
chapter-backed fact with train. General-language cards are grouped before
splitting for the same reason.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from bpe_tokenizer import BPETokenizer
from build_sft_hq1000 import CATEGORY_TERMS
from prepare_corpus_v4 import Chapter, parse_complete_chapters
from sft_v6_catalog import (
    ADDITIONAL_PERSON_LABELS,
    BOUNDARY_TOPICS,
    CORE_NOVEL_FACTS,
    NATURAL_CHALLENGES,
    NATURAL_SUBJECTS,
    PRACTICE_CORRECTIONS,
    PROJECT_CONCEPTS,
    UNKNOWN_NOVEL_NAMES,
)
from training_runtime import (
    atomic_write_json,
    atomic_write_text,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
    resolve_module_log_levels,
)


SCHEMA_VERSION = "sft_v6/1.0"
DEFAULT_CORPUS = Path("data/cloud_v4/train.txt")
DEFAULT_TOKENIZER = Path("data/scaling_a/bpe_3000/tokenizer.json")
DEFAULT_OUTPUT = Path("data/sft/v6/sft_v6_10000.jsonl")
DEFAULT_REPORT = Path("reports/milestones/018_sft_v6_10000/build_report.json")
DEFAULT_SAMPLES = Path("reports/milestones/018_sft_v6_10000/build_samples.md")
DEFAULT_LOG_DIR = Path("logs/sft_v6_build")

SPLIT_ORDER = ("train", "val", "public_diagnostic", "sealed_test")
SPLIT_TARGETS = {
    "train": 8000,
    "val": 800,
    "public_diagnostic": 600,
    "sealed_test": 600,
}
DIMENSION_TARGETS = {
    "novel_entities_facts_relations_timeline": 3300,
    "evidence_reading_and_retrieval_qa": 2000,
    "natural_chat_and_multiturn": 1500,
    "summarization_rewrite_continuation_expression": 1000,
    "instruction_format_and_length_control": 900,
    "project_study_and_general_concepts": 800,
    "correction_grounded_unknown_and_capability_boundary": 500,
}
DIMENSION_SPLIT_TARGETS = {
    dimension: {
        "train": total * 80 // 100,
        "val": total * 8 // 100,
        "public_diagnostic": total * 6 // 100,
        "sealed_test": total * 6 // 100,
    }
    for dimension, total in DIMENSION_TARGETS.items()
}
CHAPTER_SPLIT_COUNTS = {
    "train": 1279,
    "val": 128,
    "public_diagnostic": 96,
    "sealed_test": 96,
}

META_MARKERS = (
    "原问题是",
    "现只做局部证据核验",
    "当前证据片段明确",
    "正确，证据支持",
)
UNSAFE_TEXT_MARKERS = (
    "草你奶奶",
    "操你",
    "强奸",
    "性交",
    "乳房",
    "裸体",
    "一丝不挂",
    "手机访问",
    "未完待续",
    "支持正版",
    "更新到",
    "登陆com",
)
REFUSAL_MARKERS = ("不能直接", "无法凭空", "无法确认", "资料不足", "不能编造")


def stable_hash(*parts: object) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def jsonl_text(records: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )


def compact_text(text: str, limit: int = 72) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    compacted = normalized if len(normalized) <= limit else normalized[:limit].rstrip() + "…"
    if not has_balanced_structural_punctuation(compacted):
        compacted = re.sub(r"[（）()《》【】]", "", compacted)
    return compacted


def has_balanced_structural_punctuation(text: str) -> bool:
    """Return whether brackets used by the source line are structurally balanced."""

    pairs = (("（", "）"), ("(", ")"), ("《", "》"), ("【", "】"))
    return all(text.count(opening) == text.count(closing) for opening, closing in pairs)


def source_identity(text: str) -> str:
    """Normalize harmless punctuation variants before source deduplication."""

    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def category_label(category: str) -> str:
    return {
        "人物": "人物",
        "斗技或功法": "斗技或功法",
        "异火": "异火",
        "丹药": "丹药",
        "势力": "势力",
        "地点": "地点",
        "物品或体质": "物品或体质",
    }[category]


@dataclass(frozen=True)
class ParagraphSource:
    chapter_number: int
    chapter_title: str
    heading_line: int
    start_line: int
    end_line: int
    local_index: int
    raw_text: str
    clean_text: str
    label: str
    category: str
    chapter: Chapter


def evidence_payload(
    source: ParagraphSource | None,
    *,
    text: str = "",
    start_line: int = 0,
    end_line: int = 0,
    status: str = "not_applicable",
    source_path: str = "",
) -> dict[str, Any]:
    if source is None:
        return {
            "status": status,
            "source_path": source_path,
            "chapter_number": 0,
            "chapter_title": "",
            "heading_line": 0,
            "start_line": 0,
            "end_line": 0,
            "text": text,
            "sha256": text_sha256(text) if text else "",
        }
    evidence_text = text or source.raw_text
    return {
        "status": "verified_train_corpus",
        "source_path": str(DEFAULT_CORPUS),
        "chapter_number": source.chapter_number,
        "chapter_title": source.chapter_title,
        "heading_line": source.heading_line,
        "start_line": start_line or source.start_line,
        "end_line": end_line or source.end_line,
        "text": evidence_text,
        "sha256": text_sha256(evidence_text),
    }


def make_record(
    *,
    split: str,
    dimension: str,
    family: str,
    semantic_group: str,
    messages: list[dict[str, str]],
    evidence: dict[str, Any],
    entities: Sequence[str] = (),
    concepts: Sequence[str] = (),
    method: str,
) -> dict[str, Any]:
    if len(messages) < 2 or messages[0]["role"] != "user":
        raise ValueError("messages must begin with a user turn")
    if messages[-1]["role"] != "assistant":
        raise ValueError("messages must end with an assistant turn")
    question = next(
        message["content"] for message in reversed(messages) if message["role"] == "user"
    )
    answer = messages[-1]["content"]
    digest = stable_hash("sft-v6", split, dimension, family, semantic_group, question, answer)
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"v6_{digest[:20]}",
        "split": split,
        "primary_dimension": dimension,
        "task_family": family,
        "semantic_group": semantic_group,
        "question": question,
        "answer": answer,
        "messages": messages,
        "evidence": evidence,
        "coverage": {
            "entities": sorted(set(entities)),
            "concepts": sorted(set(concepts)),
        },
        "provenance": {
            "generation_method": method,
            "source_scope": "formal_train_only_or_curated_general_card",
        },
        "review": {
            "status": "codex_generated_and_rule_checked",
            "reviewer": "Codex",
            "note": "Generated for SFT v6 and subject to the independent v6 validator.",
        },
    }


def partition_chapters(chapters: Sequence[Chapter]) -> dict[str, list[Chapter]]:
    ordered = sorted(
        chapters,
        key=lambda chapter: stable_hash(
            "sft-v6-chapter-split",
            chapter.chapter_number,
            chapter.title,
            chapter.source_sha256,
        ),
    )
    if len(ordered) != sum(CHAPTER_SPLIT_COUNTS.values()):
        raise ValueError(
            f"expected {sum(CHAPTER_SPLIT_COUNTS.values())} train chapters, got {len(ordered)}"
        )
    result: dict[str, list[Chapter]] = {}
    cursor = 0
    for split in SPLIT_ORDER:
        count = CHAPTER_SPLIT_COUNTS[split]
        result[split] = ordered[cursor : cursor + count]
        cursor += count
    return result


def label_catalog() -> tuple[dict[str, str], list[str]]:
    mapping: dict[str, str] = {}
    for category, labels in CATEGORY_TERMS.items():
        for label in labels:
            mapping.setdefault(label, category)
    mapping.setdefault("药尘", "人物")
    mapping.setdefault("药老", "人物")
    mapping.setdefault("萧薰儿", "人物")
    mapping.setdefault("紫研", "人物")
    for label in ADDITIONAL_PERSON_LABELS:
        mapping.setdefault(label, "人物")
    ordered = sorted(mapping, key=lambda value: (-len(value), value))
    return mapping, ordered


def chapter_paragraphs(
    chapters: Sequence[Chapter],
    mapping: dict[str, str],
    labels: Sequence[str],
) -> dict[str, list[ParagraphSource]]:
    result: dict[str, list[ParagraphSource]] = {}
    for chapter in chapters:
        lines = chapter.source_text.splitlines()
        items: list[ParagraphSource] = []
        for local_index, raw_line in enumerate(lines):
            clean = raw_line.strip()
            if not 24 <= len(clean) <= 220:
                continue
            if any(marker in clean for marker in UNSAFE_TEXT_MARKERS):
                continue
            if not has_balanced_structural_punctuation(clean):
                continue
            found = [label for label in labels if label in clean]
            if not found:
                continue
            label = found[0]
            global_line = chapter.range_start_line + local_index
            items.append(
                ParagraphSource(
                    chapter_number=chapter.chapter_number,
                    chapter_title=chapter.title,
                    heading_line=chapter.start_line,
                    start_line=global_line,
                    end_line=global_line,
                    local_index=local_index,
                    raw_text=raw_line,
                    clean_text=clean,
                    label=label,
                    category=mapping[label],
                    chapter=chapter,
                )
            )
        if items:
            result[chapter.section_id] = sorted(
                items,
                key=lambda item: stable_hash(
                    "sft-v6-paragraph", item.start_line, item.label, item.clean_text
                ),
            )
    text_counts = Counter(
        source_identity(item.clean_text)
        for items in result.values()
        for item in items
    )
    return {
        section_id: [item for item in items if text_counts[source_identity(item.clean_text)] == 1]
        for section_id, items in result.items()
        if any(text_counts[source_identity(item.clean_text)] == 1 for item in items)
    }


class SourcePool:
    def __init__(
        self,
        split: str,
        chapters: Sequence[Chapter],
        paragraphs: dict[str, list[ParagraphSource]],
        tokenizer: BPETokenizer,
    ) -> None:
        self.split = split
        self.tokenizer = tokenizer
        chapter_items = [
            (chapter, paragraphs.get(chapter.section_id, []))
            for chapter in chapters
            if paragraphs.get(chapter.section_id)
        ]
        chapter_items.sort(
            key=lambda pair: stable_hash(
                "sft-v6-pool", split, pair[0].chapter_number, pair[0].title
            )
        )
        self.entries: list[ParagraphSource] = []
        maximum = max(len(items) for _, items in chapter_items)
        for offset in range(maximum):
            for _, items in chapter_items:
                if offset < len(items):
                    self.entries.append(items[offset])
        self.position = 0
        self.used_purposes: set[tuple[str, int]] = set()

    def next_short(
        self,
        purpose: str,
        *,
        answer_token_limit: int = 120,
    ) -> ParagraphSource:
        attempts = 0
        while attempts < len(self.entries) * 2:
            source = self.entries[self.position % len(self.entries)]
            self.position += 1
            attempts += 1
            key = (purpose, source.start_line)
            if key in self.used_purposes:
                continue
            if len(self.tokenizer.encode(source.clean_text)) > answer_token_limit:
                continue
            self.used_purposes.add(key)
            return source
        raise RuntimeError(f"source pool exhausted for {self.split}/{purpose}")

    def long_context(
        self,
        purpose: str,
        *,
        minimum_tokens: int = 128,
        maximum_tokens: int = 280,
    ) -> tuple[ParagraphSource, str, int, int]:
        attempts = 0
        while attempts < len(self.entries) * 3:
            source = self.next_short(purpose, answer_token_limit=96)
            lines = source.chapter.source_text.splitlines()
            start = source.local_index
            end = source.local_index
            context = "\n".join(lines[start : end + 1])
            turn_left = True
            while len(self.tokenizer.encode(context.strip())) < minimum_tokens:
                changed = False
                if turn_left and start > 0:
                    start -= 1
                    changed = True
                elif end + 1 < len(lines):
                    end += 1
                    changed = True
                elif start > 0:
                    start -= 1
                    changed = True
                if not changed:
                    break
                candidate = "\n".join(lines[start : end + 1])
                candidate_tokens = len(self.tokenizer.encode(candidate.strip()))
                if candidate_tokens > maximum_tokens:
                    if turn_left:
                        start += 1
                    else:
                        end -= 1
                    break
                context = candidate
                turn_left = not turn_left
            token_count = len(self.tokenizer.encode(context.strip()))
            if not minimum_tokens <= token_count <= maximum_tokens:
                attempts += 1
                continue
            if any(marker in context for marker in UNSAFE_TEXT_MARKERS):
                attempts += 1
                continue
            paragraphs_with_label = sum(
                1 for line in lines[start : end + 1] if source.label in line
            )
            if paragraphs_with_label != 1:
                attempts += 1
                continue
            start_line = source.chapter.range_start_line + start
            end_line = source.chapter.range_start_line + end
            return source, context, start_line, end_line
        raise RuntimeError(f"cannot build long context for {self.split}/{purpose}")

    def continuation_pair(
        self,
        purpose: str,
    ) -> tuple[ParagraphSource, str, int, int]:
        attempts = 0
        while attempts < len(self.entries) * 3:
            source = self.next_short(purpose, answer_token_limit=100)
            lines = source.chapter.source_text.splitlines()
            following_index = source.local_index + 1
            while following_index < len(lines) and not lines[following_index].strip():
                following_index += 1
            if following_index >= len(lines):
                attempts += 1
                continue
            answer = lines[following_index].strip()
            if not 20 <= len(answer) <= 220:
                attempts += 1
                continue
            if any(marker in answer for marker in UNSAFE_TEXT_MARKERS):
                attempts += 1
                continue
            if not has_balanced_structural_punctuation(answer):
                attempts += 1
                continue
            answer_tokens = len(self.tokenizer.encode(answer))
            prompt_tokens = len(self.tokenizer.encode(source.clean_text))
            if answer_tokens > 160 or prompt_tokens + answer_tokens > 430:
                attempts += 1
                continue
            start_line = source.start_line
            end_line = source.chapter.range_start_line + following_index
            evidence = "\n".join(lines[source.local_index : following_index + 1])
            return source, answer, start_line, end_line
        raise RuntimeError(f"cannot build continuation for {self.split}/{purpose}")


def allocate_candidates(
    candidates: Sequence[Any],
    targets: dict[str, int],
    *,
    salt: str,
) -> dict[str, list[Any]]:
    ordered = sorted(candidates, key=lambda value: stable_hash(salt, repr(value)))
    if len(ordered) != sum(targets.values()):
        raise ValueError(
            f"{salt} expected {sum(targets.values())} candidates, got {len(ordered)}"
        )
    result: dict[str, list[Any]] = {}
    cursor = 0
    for split in SPLIT_ORDER:
        count = targets[split]
        result[split] = ordered[cursor : cursor + count]
        cursor += count
    return result


def build_core_fact_records() -> list[dict[str, Any]]:
    dimension = "novel_entities_facts_relations_timeline"
    question_templates = (
        "小说中的{name}是什么身份？",
        "请用一句话介绍{name}。",
        "第一次了解{name}时，应该先记住什么？",
        "怎样概括{name}在故事中的基本定位？",
        "{name}属于哪类小说设定？",
        "请说明{name}与故事的基本关系。",
        "如果读者问{name}是谁，应怎样回答？",
        "只说基础事实，{name}是什么？",
        "请给出{name}的简要身份说明。",
        "用新读者能听懂的话解释{name}。",
        "不展开后续剧情，介绍一下{name}。",
        "请直接概括{name}的基本信息。",
    )
    answer_templates = (
        "{name}{description}。",
        "简单说，{name}{description}。",
        "最基础的信息是：{name}{description}。",
        "在故事中，{name}{description}。",
        "{name}属于{kind}；{description}。",
        "可以先记住：{name}{description}。",
        "对新读者而言，{name}{description}。",
        "直接概括，{name}{description}。",
        "{name}的基本定位是{description}。",
        "不涉及后续情节，{name}{description}。",
        "{name}是{kind}，并且{description}。",
        "一句话概括：{name}{description}。",
    )
    records: list[dict[str, Any]] = []
    for name, kind, description in CORE_NOVEL_FACTS:
        fact_text = f"{name}：{description}"
        for index, (question_template, answer_template) in enumerate(
            zip(question_templates, answer_templates, strict=True)
        ):
            records.append(
                make_record(
                    split="train",
                    dimension=dimension,
                    family="curated_core_novel_identity",
                    semantic_group=f"core:{name}",
                    messages=[
                        {"role": "user", "content": question_template.format(name=name)},
                        {
                            "role": "assistant",
                            "content": answer_template.format(
                                name=name,
                                kind=kind,
                                description=description,
                            ),
                        },
                    ],
                    evidence=evidence_payload(
                        None,
                        text=fact_text,
                        status="curated_project_fact",
                        source_path="sft_v6_catalog.py",
                    ),
                    entities=[name] if kind == "人物" else [],
                    concepts=[name] if kind != "人物" else [],
                    method="codex_curated_fact_card",
                )
            )
    if len(records) != 300:
        raise AssertionError(f"core fact pack must contain 300 records, got {len(records)}")
    return records


def build_novel_records(
    pools: dict[str, SourcePool],
) -> list[dict[str, Any]]:
    dimension = "novel_entities_facts_relations_timeline"
    records = build_core_fact_records()
    templates = (
        "阅读这句话：“{quote}”其中明确提到的{category}是什么？",
        "把空缺补完整：“{masked}”只填写缺少的专名。",
        "在“{quote}”这句话里，被点名的是哪个{category}？",
        "根据材料“{quote}”，请指出其中出现的{category}。",
        "这段小说文字写到“{quote}”。其中的专名是什么？",
        "从“{quote}”中找出被明确提到的{category}。",
    )
    targets = dict(DIMENSION_SPLIT_TARGETS[dimension])
    targets["train"] -= len(records)
    for split in SPLIT_ORDER:
        for index in range(targets[split]):
            source = pools[split].next_short("novel")
            quote = compact_text(source.clean_text, 92)
            masked = quote.replace(source.label, "【空缺】", 1)
            template = templates[index % len(templates)]
            question = template.format(
                quote=quote,
                masked=masked,
                category=category_label(source.category),
            )
            fragment = compact_text(source.clean_text, 30)
            answer_forms = (
                f"被明确提到的是{source.label}；原文写到“{fragment}”。",
                f"空缺处应填{source.label}，它在这里作为{category_label(source.category)}出现；原句写到“{fragment}”。",
                f"这句话点名的是{source.label}，对应类别是{category_label(source.category)}；原句要点是“{fragment}”。",
                f"根据这段文字，可以确认其中出现了{source.label}；直接依据是“{fragment}”。",
                f"材料中的专名是{source.label}；相关表述为“{fragment}”。",
                f"应回答{source.label}，它是这段话明确写出的{category_label(source.category)}；原句为“{fragment}”。",
            )
            records.append(
                make_record(
                    split=split,
                    dimension=dimension,
                    family="grounded_novel_entity_fact",
                    semantic_group=f"train-line:{source.start_line}",
                    messages=[
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer_forms[index % len(answer_forms)]},
                    ],
                    evidence=evidence_payload(source),
                    entities=[source.label] if source.category == "人物" else [],
                    concepts=[source.label] if source.category != "人物" else [],
                    method="grounded_train_paragraph_fact",
                )
            )
    return records


def build_evidence_records(
    pools: dict[str, SourcePool],
) -> list[dict[str, Any]]:
    dimension = "evidence_reading_and_retrieval_qa"
    records: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        target = DIMENSION_SPLIT_TARGETS[dimension][split]
        for index in range(target):
            source, context, start_line, end_line = pools[split].long_context("evidence")
            context_display = context.strip()
            question_forms = (
                "阅读材料：\n{context}\n\n请原样摘录其中包含“{label}”的完整段落。",
                "材料如下：\n{context}\n\n从材料中找出写到“{label}”的那一段，并保持原文措辞。",
                "请阅读：\n{context}\n\n哪一段明确提到了“{label}”？请完整摘录该段。",
                "依据下面的材料完成信息提取：\n{context}\n\n输出包含“{label}”的完整段落。",
            )
            question = question_forms[index % len(question_forms)].format(
                context=context_display,
                label=source.label,
            )
            records.append(
                make_record(
                    split=split,
                    dimension=dimension,
                    family="long_context_exact_paragraph_extraction",
                    semantic_group=f"evidence-line:{source.start_line}",
                    messages=[
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": source.clean_text},
                    ],
                    evidence=evidence_payload(
                        source,
                        text=context,
                        start_line=start_line,
                        end_line=end_line,
                    ),
                    entities=[source.label] if source.category == "人物" else [],
                    concepts=[source.label] if source.category != "人物" else [],
                    method="grounded_long_context_extraction",
                )
            )
    return records


def natural_multiturn_candidates() -> list[tuple[str, str, str, str, str, str]]:
    records = []
    for subject in NATURAL_SUBJECTS:
        for challenge, suggestion in NATURAL_CHALLENGES:
            first_user = f"我在学{subject}，{challenge}。"
            first_assistant = (
                f"可以先{suggestion}。这一步只处理{subject}里的一个具体问题，完成后马上检查结果。"
            )
            records.append(
                (
                    subject,
                    challenge,
                    first_user,
                    first_assistant,
                    f"针对刚才提到的“{challenge}”，如果今天只有半小时处理{subject}，应该怎样安排？",
                    f"因为你遇到“{challenge}”，前十分钟明确{subject}的小目标，中间十五分钟动手完成，最后五分钟记录结果和下一步。",
                )
            )
            records.append(
                (
                    subject,
                    challenge,
                    first_user,
                    first_assistant,
                    f"针对“{challenge}”这个困难，我完成{subject}的第一步以后，接下来做什么？",
                    f"针对“{challenge}”，先检查刚才的{subject}结果是否符合目标，再只增加一个小难度，并把新问题写进复盘记录。",
                )
            )
    return records


def natural_single_candidates() -> list[tuple[str, str, str, str]]:
    records = []
    for subject in NATURAL_SUBJECTS:
        for challenge, suggestion in NATURAL_CHALLENGES:
            question = f"我在处理{subject}时{challenge}，给我一个可以马上开始的建议。"
            answer = f"先{suggestion}。针对{subject}只做一个能在短时间内验证的小步骤，完成后再决定是否继续。"
            records.append((subject, challenge, question, answer))
    return records


def build_natural_records() -> list[dict[str, Any]]:
    dimension = "natural_chat_and_multiturn"
    multi_targets = {
        "train": 800,
        "val": 80,
        "public_diagnostic": 60,
        "sealed_test": 60,
    }
    single_targets = {
        "train": 400,
        "val": 40,
        "public_diagnostic": 30,
        "sealed_test": 30,
    }
    multi = allocate_candidates(
        natural_multiturn_candidates(), multi_targets, salt="natural-multiturn"
    )
    single = allocate_candidates(
        natural_single_candidates(), single_targets, salt="natural-single"
    )
    records: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        for subject, challenge, first_user, first_assistant, followup, answer in multi[split]:
            records.append(
                make_record(
                    split=split,
                    dimension=dimension,
                    family="natural_multiturn_support",
                    semantic_group=f"natural-multi:{stable_hash(subject, challenge, followup)[:16]}",
                    messages=[
                        {"role": "user", "content": first_user},
                        {"role": "assistant", "content": first_assistant},
                        {"role": "user", "content": followup},
                        {"role": "assistant", "content": answer},
                    ],
                    evidence=evidence_payload(None),
                    concepts=[subject],
                    method="curated_multiturn_scenario",
                )
            )
        for subject, challenge, question, answer in single[split]:
            records.append(
                make_record(
                    split=split,
                    dimension=dimension,
                    family="natural_single_turn_support",
                    semantic_group=f"natural-single:{stable_hash(subject, challenge)[:16]}",
                    messages=[
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    evidence=evidence_payload(None),
                    concepts=[subject],
                    method="curated_single_turn_scenario",
                )
            )
    return records


def build_expression_records(
    pools: dict[str, SourcePool],
) -> list[dict[str, Any]]:
    dimension = "summarization_rewrite_continuation_expression"
    records: list[dict[str, Any]] = []
    subtype_targets = {
        "continuation": {
            "train": 400,
            "val": 40,
            "public_diagnostic": 30,
            "sealed_test": 30,
        },
        "summary": {
            "train": 200,
            "val": 20,
            "public_diagnostic": 15,
            "sealed_test": 15,
        },
        "rewrite": {
            "train": 200,
            "val": 20,
            "public_diagnostic": 15,
            "sealed_test": 15,
        },
    }
    for split in SPLIT_ORDER:
        for index in range(subtype_targets["continuation"][split]):
            source, answer, start_line, end_line = pools[split].continuation_pair("continuation")
            lines = source.chapter.source_text.splitlines()
            end_local = end_line - source.chapter.range_start_line
            full_evidence = "\n".join(lines[source.local_index : end_local + 1])
            question = (
                f"下面是小说中的一段文字：\n{source.clean_text}\n\n"
                "请按原文接写紧随其后的下一段，不要自行扩展。"
            )
            records.append(
                make_record(
                    split=split,
                    dimension=dimension,
                    family="grounded_next_paragraph_continuation",
                    semantic_group=f"continuation:{source.start_line}",
                    messages=[
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    evidence=evidence_payload(
                        source,
                        text=full_evidence,
                        start_line=start_line,
                        end_line=end_line,
                    ),
                    entities=[source.label] if source.category == "人物" else [],
                    concepts=[source.label] if source.category != "人物" else [],
                    method="grounded_next_paragraph",
                )
            )
        for subtype in ("summary", "rewrite"):
            for index in range(subtype_targets[subtype][split]):
                source = pools[split].next_short(f"expression-{subtype}")
                quote = compact_text(source.clean_text, 150)
                fragment = compact_text(source.clean_text, 36)
                if subtype == "summary":
                    question = (
                        f"材料：“{quote}”\n请用一句话概括这段材料明确提到的对象，不补充材料外信息。"
                    )
                    answer = f"这段材料明确提到了{source.label}，相关文字是“{fragment}”。"
                    family = "grounded_one_sentence_summary"
                else:
                    question = (
                        f"请把下面材料改写成简洁陈述，并保留专名“{source.label}”：\n{quote}"
                    )
                    answer = f"材料写到{source.label}，并提及“{fragment}”。"
                    family = "grounded_concise_rewrite"
                records.append(
                    make_record(
                        split=split,
                        dimension=dimension,
                        family=family,
                        semantic_group=f"expression:{subtype}:{source.start_line}",
                        messages=[
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                        ],
                        evidence=evidence_payload(source),
                        entities=[source.label] if source.category == "人物" else [],
                        concepts=[source.label] if source.category != "人物" else [],
                        method=f"grounded_{subtype}",
                    )
                )
    return records


def build_instruction_records(
    pools: dict[str, SourcePool],
) -> list[dict[str, Any]]:
    dimension = "instruction_format_and_length_control"
    records: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        target = DIMENSION_SPLIT_TARGETS[dimension][split]
        for index in range(target):
            source = pools[split].next_short("instruction")
            category = category_label(source.category)
            quote = compact_text(source.clean_text, 110)
            format_index = index % 6
            if format_index == 0:
                question = f"材料：“{quote}”请只输出其中的专名，不加解释。"
                answer = source.label
            elif format_index == 1:
                question = f"从材料“{quote}”提取专名，并按“名称：类别”格式输出。"
                answer = f"{source.label}：{category}"
            elif format_index == 2:
                question = (
                    f"材料“{quote}”提到了{source.label}。"
                    "请用“对象：名称；类型：类别”的格式回答。"
                )
                answer = f"对象：{source.label}；类型：{category}。"
            elif format_index == 3:
                question = f"请根据“{quote}”给出一个项目符号，只写专名和类别。"
                answer = f"- {source.label}：{category}"
            elif format_index == 4:
                question = (
                    f"根据材料“{quote}”，把{source.label}和它的类别"
                    "写成一句不超过二十字的话。"
                )
                answer = f"{source.label}属于{category}。"
            else:
                question = (
                    f"阅读“{quote}”后完成填空：“{source.label}在此处属于____。”"
                    "只填写类别。"
                )
                answer = category
            records.append(
                make_record(
                    split=split,
                    dimension=dimension,
                    family="grounded_format_control",
                    semantic_group=f"instruction:{source.start_line}",
                    messages=[
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    evidence=evidence_payload(source),
                    entities=[source.label] if source.category == "人物" else [],
                    concepts=[source.label] if source.category != "人物" else [],
                    method="grounded_positive_format_instruction",
                )
            )
    return records


def project_variants(concept: str, definition: str) -> list[tuple[str, str]]:
    return [
        (f"{concept}是什么？", f"{concept}是{definition}。"),
        (f"请给新手解释{concept}。", f"可以把{concept}理解为{definition}。"),
        (f"{concept}在手搓GPT项目中有什么作用？", f"在这个项目中，{concept}用于说明{definition}。"),
        (f"为什么训练模型时要关心{concept}？", f"因为{concept}关系到{definition}，会影响训练或生成过程。"),
        (f"用一句话概括{concept}。", f"一句话说，{concept}就是{definition}。"),
        (f"第一次学习{concept}应该先记住什么？", f"先记住它的核心：{concept}指{definition}。"),
        (f"{concept}和模型训练有什么联系？", f"它与模型训练的联系在于：{definition}。"),
        (f"请说明{concept}解决的基本问题。", f"{concept}解决的基本问题可以概括为{definition}。"),
        (f"怎样判断自己理解了{concept}？", f"如果你能解释“{definition}”并举出项目中的用途，就开始理解{concept}了。"),
        (f"请用通俗语言介绍{concept}。", f"通俗地说，{concept}就是{definition}。"),
        (f"学习{concept}时容易忽略什么？", f"不要只记名称，还要理解它表示{definition}。"),
        (f"给{concept}写一条复习笔记。", f"复习笔记：{concept}——{definition}。"),
        (f"把{concept}解释成“概念加作用”的形式。", f"概念：{concept}；作用或含义：{definition}。"),
        (f"如果同学问起{concept}，怎么简短回答？", f"可以回答：{concept}是{definition}。"),
        (f"{concept}属于这个项目的哪类知识？", f"它属于模型训练与生成的基础知识，具体指{definition}。"),
        (f"请写出{concept}的核心定义和学习提醒。", f"核心定义：{definition}。学习时要结合实际代码或样本理解{concept}。"),
    ]


def build_project_records() -> list[dict[str, Any]]:
    dimension = "project_study_and_general_concepts"
    grouped = allocate_candidates(
        list(PROJECT_CONCEPTS),
        {"train": 40, "val": 4, "public_diagnostic": 3, "sealed_test": 3},
        salt="project-concept-groups",
    )
    records: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        for concept, definition in grouped[split]:
            fact = f"{concept}：{definition}"
            for question, answer in project_variants(concept, definition):
                records.append(
                    make_record(
                        split=split,
                        dimension=dimension,
                        family="project_concept_explanation",
                        semantic_group=f"project-concept:{concept}",
                        messages=[
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                        ],
                        evidence=evidence_payload(
                            None,
                            text=fact,
                            status="curated_project_fact",
                            source_path="sft_v6_catalog.py",
                        ),
                        concepts=[concept],
                        method="curated_project_concept_card",
                    )
                )
    return records


def correction_groups() -> list[tuple[str, str, str, str]]:
    groups: list[tuple[str, str, str, str]] = []
    for claim, correction in PRACTICE_CORRECTIONS:
        groups.append((f"practice:{claim}", claim, correction, "practice_correction"))
    for name, kind, description in CORE_NOVEL_FACTS[:15]:
        claim = f"{name}与《斗破苍穹》无关"
        correction = f"这个说法不对。{name}{description}"
        groups.append((f"novel:{name}", claim, correction, "known_novel_correction"))
    for topic, need in BOUNDARY_TOPICS:
        claim = f"你现在可以直接确认{topic}"
        correction = f"我不能直接确认{topic}；提供{need}后，我可以继续分析"
        groups.append((f"boundary:{topic}", claim, correction, "realtime_boundary"))
    for name in UNKNOWN_NOVEL_NAMES:
        claim = f"{name}一定是小说中的重要人物"
        correction = f"现有正式训练语料不能确认{name}的身份，不能把它当成已知小说人物"
        groups.append((f"unknown:{name}", claim, correction, "grounded_unknown"))
    if len(groups) != 50:
        raise AssertionError(f"correction groups must be 50, got {len(groups)}")
    return groups


def correction_variants(claim: str, correction: str) -> list[tuple[str, str]]:
    return [
        (f"有人说“{claim}”，请判断并纠正。", f"{correction}。"),
        (f"“{claim}”这个说法可靠吗？", f"不可靠。{correction}。"),
        (f"请核对这句话：“{claim}”。", f"核对结果：{correction}。"),
        (f"如果“{claim}”不准确，应该怎样改？", f"可以改为：{correction}。"),
        (f"请直接指出“{claim}”的问题。", f"问题在于结论没有可靠依据。{correction}。"),
        (f"怎样向新手解释“{claim}”是否成立？", f"应告诉新手：{correction}。"),
        (f"请用一句话纠正“{claim}”。", f"正确处理是：{correction}。"),
        (f"面对“{claim}”这种判断，应该怎么回答？", f"不要直接接受该判断；{correction}。"),
        (f"只给结论和修正：“{claim}”。", f"结论：不应这样断定；修正：{correction}。"),
        (f"请把“{claim}”改写成更可靠的说法。", f"更可靠的说法是：{correction}。"),
    ]


def build_correction_records() -> list[dict[str, Any]]:
    dimension = "correction_grounded_unknown_and_capability_boundary"
    grouped = allocate_candidates(
        correction_groups(),
        {"train": 40, "val": 4, "public_diagnostic": 3, "sealed_test": 3},
        salt="correction-groups",
    )
    records: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        for group_key, claim, correction, family in grouped[split]:
            for question, answer in correction_variants(claim, correction):
                records.append(
                    make_record(
                        split=split,
                        dimension=dimension,
                        family=family,
                        semantic_group=group_key,
                        messages=[
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                        ],
                        evidence=evidence_payload(
                            None,
                            text=f"判断：{claim}\n修正：{correction}",
                            status="curated_project_fact",
                            source_path="sft_v6_catalog.py",
                        ),
                        concepts=[claim],
                        method="curated_correction_or_boundary_card",
                    )
                )
    return records


def record_counts(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "splits": dict(Counter(record["split"] for record in records)),
        "dimensions": dict(Counter(record["primary_dimension"] for record in records)),
        "dimension_splits": {
            dimension: dict(
                Counter(
                    record["split"]
                    for record in records
                    if record["primary_dimension"] == dimension
                )
            )
            for dimension in DIMENSION_TARGETS
        },
        "families": dict(Counter(record["task_family"] for record in records)),
        "multiturn": sum(len(record["messages"]) >= 4 for record in records),
        "verified_corpus": sum(
            record["evidence"]["status"] == "verified_train_corpus"
            for record in records
        ),
        "chapters": len(
            {
                record["evidence"]["chapter_title"]
                for record in records
                if record["evidence"]["status"] == "verified_train_corpus"
            }
        ),
        "entities": len(
            {
                entity
                for record in records
                for entity in record["coverage"]["entities"]
            }
        ),
        "concepts": len(
            {
                concept
                for record in records
                for concept in record["coverage"]["concepts"]
            }
        ),
    }


def validate_build_shape(records: Sequence[dict[str, Any]]) -> None:
    counts = record_counts(records)
    if counts["records"] != 10000:
        raise ValueError(f"expected 10000 records, got {counts['records']}")
    if counts["splits"] != SPLIT_TARGETS:
        raise ValueError(f"split target mismatch: {counts['splits']}")
    if counts["dimensions"] != DIMENSION_TARGETS:
        raise ValueError(f"dimension target mismatch: {counts['dimensions']}")
    for dimension, expected in DIMENSION_SPLIT_TARGETS.items():
        if counts["dimension_splits"][dimension] != expected:
            raise ValueError(
                f"dimension split mismatch for {dimension}: "
                f"{counts['dimension_splits'][dimension]} != {expected}"
            )
    ids = [record["id"] for record in records]
    questions = [record["question"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate record IDs")
    if len(questions) != len(set(questions)):
        raise ValueError("duplicate exact questions")
    if counts["multiturn"] < 1000:
        raise ValueError("fewer than 1000 multiturn records")
    if any(
        marker in record["question"] or marker in record["answer"]
        for record in records
        for marker in META_MARKERS
    ):
        raise ValueError("forbidden meta wrapper leaked into v6")


def sample_markdown(records: Sequence[dict[str, Any]]) -> str:
    sections = ["# SFT v6 build samples", ""]
    for dimension in DIMENSION_TARGETS:
        sections.extend([f"## {dimension}", ""])
        selected = sorted(
            (record for record in records if record["primary_dimension"] == dimension),
            key=lambda record: stable_hash("sample", record["id"]),
        )[:5]
        for record in selected:
            sections.append(f"- `{record['id']}` / `{record['split']}` / `{record['task_family']}`")
            sections.append(f"  - 问：{compact_text(record['question'], 180)}")
            sections.append(f"  - 答：{compact_text(record['answer'], 180)}")
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = generate_run_id("sft-v6-build")
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
        corpus_text = args.corpus.read_text(encoding="utf-8")
        preamble, chapters = parse_complete_chapters(corpus_text)
        if preamble.strip():
            raise ValueError("formal train corpus has an unexpected preamble")
        tokenizer = BPETokenizer.load(args.tokenizer)
        mapping, labels = label_catalog()
        partitions = partition_chapters(chapters)
        paragraphs = chapter_paragraphs(chapters, mapping, labels)
        loggers["data"].info(
            "loaded corpus chapters=%d labelled_chapters=%d labelled_paragraphs=%d vocab=%d",
            len(chapters),
            len(paragraphs),
            sum(len(items) for items in paragraphs.values()),
            tokenizer.vocab_size,
        )
        pools = {
            split: SourcePool(split, partitions[split], paragraphs, tokenizer)
            for split in SPLIT_ORDER
        }
        records = (
            build_novel_records(pools)
            + build_evidence_records(pools)
            + build_natural_records()
            + build_expression_records(pools)
            + build_instruction_records(pools)
            + build_project_records()
            + build_correction_records()
        )
        records.sort(
            key=lambda record: (
                SPLIT_ORDER.index(record["split"]),
                record["primary_dimension"],
                record["id"],
            )
        )
        validate_build_shape(records)
        counts = record_counts(records)
        loggers["validation"].info(
            "build shape passed records=%d splits=%s dimensions=%s multiturn=%d chapters=%d",
            len(records),
            counts["splits"],
            counts["dimensions"],
            counts["multiturn"],
            counts["chapters"],
        )
        atomic_write_text(args.output, jsonl_text(records))
        atomic_write_text(args.samples, sample_markdown(records))
        report = {
            "schema_version": "sft-v6-build-report/v1",
            "status": "built_pending_independent_validation",
            "run_id": run_id,
            "corpus_path": str(args.corpus),
            "corpus_sha256": file_sha256(args.corpus),
            "tokenizer_path": str(args.tokenizer),
            "tokenizer_sha256": file_sha256(args.tokenizer),
            "dataset_path": str(args.output),
            "dataset_sha256": file_sha256(args.output),
            "samples_path": str(args.samples),
            "chapter_partition_counts": {
                split: len(partitions[split]) for split in SPLIT_ORDER
            },
            **counts,
            "next_gate": "validate_sft_v6.py",
        }
        atomic_write_json(args.report, report)
        loggers["orchestrator"].info(
            "wrote dataset=%s sha256=%s report=%s samples=%s",
            args.output,
            report["dataset_sha256"],
            args.report,
            args.samples,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        loggers["validation"].exception("SFT v6 build failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
