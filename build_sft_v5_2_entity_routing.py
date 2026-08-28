"""Build SFT v5.2 with balanced known-entity and grounded-unknown routing."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from bpe_tokenizer import BPETokenizer
from build_sft_v5_1_no_math import (
    assert_no_math_or_pollution,
    canonicalize_question,
    held_out_prompt_matches,
)
from build_sft_v5_repair import (
    filter_encodable_candidates,
    jsonl_text,
    read_jsonl,
    validate_records,
)
from sft_v5_entity_spec import (
    CONCEPT_IDENTITY_QUESTIONS,
    CORE_CONCEPT_EXTRA_QUESTIONS,
    CORE_ENTITY_NAMES,
    CORE_PERSON_EXTRA_QUESTIONS,
    HIDDEN_ENTITY_EVAL_ITEMS,
    IDENTITY_ANSWERS,
    KNOWN_ENTITY_PROFILES,
    PERSON_IDENTITY_QUESTIONS,
    RELATION_FACTS,
    UNKNOWN_ENTITY_NAMES,
)
from training_runtime import (
    atomic_write_json,
    atomic_write_text,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
)


SCHEMA_VERSION = "sft_v5_2_entity_routing/1.2"
DEFAULT_BASE_PATH = Path(
    "data/sft/v5_1_no_math/sft_v5_1_no_math_training_ready.jsonl"
)
DEFAULT_TOKENIZER_PATH = Path("data/cloud_v4/tokenizer.json")
DEFAULT_OUTPUT_PATH = Path(
    "data/sft/v5_2_2_core_routing/sft_v5_2_2_core_routing_training_ready.jsonl"
)
DEFAULT_REPORT_PATH = Path(
    "reports/milestones/014_v5_2_entity_routing/v5_2_2_data_report.json"
)
EXPECTED_BASE_SPLITS = {"train": 3629, "val": 520, "test": 527}

REFUSAL_MARKERS = (
    "资料不足",
    "无法确定",
    "不能硬编",
    "没有足够",
    "无法确认",
    "不能确认",
    "请先说明",
    "请补充",
)
NEW_FAMILIES = {
    "novel_core_entity_v5_2",
    "novel_known_entity_v5_2",
    "novel_relation_v5_2",
    "novel_correction_v5_2",
    "novel_unknown_grounded_v5_2",
}
POSITIVE_FAMILIES = {
    "novel_core_entity_v5_2",
    "novel_known_entity_v5_2",
    "novel_relation_v5_2",
    "novel_correction_v5_2",
}
HIDDEN_CANONICAL_QUESTIONS = frozenset(
    canonicalize_question(item["question"]) for item in HIDDEN_ENTITY_EVAL_ITEMS
)


def stable_hash(*parts: object) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hidden_prompt_matches(question: str) -> tuple[str, ...]:
    canonical = canonicalize_question(question)
    return tuple(
        sorted(prompt for prompt in HIDDEN_CANONICAL_QUESTIONS if prompt in canonical)
    )


def is_refusal_answer(answer: str) -> bool:
    return any(marker in answer for marker in REFUSAL_MARKERS)


def is_meta_introduction_clarification(record: dict[str, Any]) -> bool:
    question = str(record.get("question", ""))
    answer = str(record.get("answer", ""))
    family = str(record.get("task_family", ""))
    return (
        family == "ambiguity_unknown_clarification"
        and "请介绍" in question
        and ("如果用户现在只说" in question or "如果用户只说" in question)
        and is_refusal_answer(answer)
    )


def clean_v5_1_records(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for record in records:
        family = str(record.get("task_family", ""))
        reason = ""
        if family == "novel_known_entity":
            reason = "replace_sparse_entity_pack"
        elif family == "honest_unknown_general":
            reason = "replace_bare_unknown_refusal_pack"
        elif is_meta_introduction_clarification(record):
            reason = "remove_meta_introduction_clarification"
        if reason:
            removed.append(record)
            reasons[reason] += 1
        else:
            kept.append(record)
    return kept, removed, reasons


def make_record(
    *,
    task_family: str,
    group_key: str,
    split: str,
    question: str,
    answer: str,
    evidence_source: str,
    evidence_text: str,
) -> dict[str, Any]:
    digest = stable_hash("sft-v5-2", task_family, question, answer)[:16]
    group_digest = stable_hash("sft-v5-2-group", task_family, group_key)[:16]
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"v5_2_{digest}",
        "question": question,
        "answer": answer,
        "task_family": task_family,
        "split": split,
        "topic_id": f"v5_2:{task_family}:{group_digest}",
        "fact_id": f"v5_2:{task_family}:{group_digest}",
        "group_id": f"v5_2:{task_family}:{group_digest}",
        "evidence": {
            "status": "codex_curated_repair",
            "source": evidence_source,
            "text": evidence_text,
            "sha256": stable_hash(evidence_source, evidence_text),
        },
        "review": {
            "status": "codex_generated_and_rule_checked",
            "reviewer": "Codex",
            "note": (
                "v5.2 entity-routing repair; known entities answer directly, "
                "unknown entities require an explicit corpus-evidence premise."
            ),
        },
        "origin": {
            "source": "m014_entity_routing_repair",
            "generation_method": "deterministic_fact_card_templates",
        },
    }


def identity_candidates() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name, profile in KNOWN_ENTITY_PROFILES.items():
        base_questions = (
            PERSON_IDENTITY_QUESTIONS
            if profile["kind"] == "person"
            else CONCEPT_IDENTITY_QUESTIONS
        )
        extra_questions = (
            CORE_PERSON_EXTRA_QUESTIONS
            if profile["kind"] == "person"
            else CORE_CONCEPT_EXTRA_QUESTIONS
        )
        questions = (
            base_questions + extra_questions
            if name in CORE_ENTITY_NAMES
            else base_questions
        )
        description = str(profile["description"])
        for index, question_template in enumerate(questions):
            answer_template = IDENTITY_ANSWERS[index % len(IDENTITY_ANSWERS)]
            train_limit = 20 if name in CORE_ENTITY_NAMES else 8
            val_limit = train_limit + 2
            split = "train" if index < train_limit else "val" if index < val_limit else "test"
            question = question_template.format(name=name)
            answer = answer_template.format(
                name=name,
                description=description,
            )
            records.append(
                make_record(
                    task_family=(
                        "novel_core_entity_v5_2"
                        if name in CORE_ENTITY_NAMES
                        else "novel_known_entity_v5_2"
                    ),
                    group_key=f"{name}|identity|{index}",
                    split=split,
                    question=question,
                    answer=answer,
                    evidence_source="curated_known_novel_entities_v3",
                    evidence_text=f"{name}: {description}",
                )
            )
    return records


def relation_candidates() -> list[dict[str, Any]]:
    question_templates = (
        "请说明{subject}之间的基本关系。",
        "怎样概括{subject}的关系？",
        "小说中{subject}是什么关系？",
        "用一句话说清{subject}的联系。",
        "只按《斗破苍穹》说明{subject}之间的联系。",
    )
    answer_templates = (
        "{answer}。",
        "简单说，{answer}。",
        "在小说设定中，{answer}。",
        "两者的基本联系是：{answer}。",
        "依据当前事实卡，{answer}。",
    )
    records: list[dict[str, Any]] = []
    for fact_index, (subject, relation, answer) in enumerate(RELATION_FACTS):
        for index, (question_template, answer_template) in enumerate(
            zip(question_templates, answer_templates, strict=True)
        ):
            split = "train" if index < 3 else "val" if index == 3 else "test"
            records.append(
                make_record(
                    task_family="novel_relation_v5_2",
                    group_key=f"relation|{fact_index}|{index}",
                    split=split,
                    question=question_template.format(subject=subject),
                    answer=answer_template.format(answer=answer),
                    evidence_source="curated_novel_relations_v3",
                    evidence_text=f"{subject}: {relation}; {answer}",
                )
            )
    return records


def correction_candidates() -> list[dict[str, Any]]:
    corrections = (
        ("萧炎就是一种异火", "萧炎是故事的主要人物，异火才是特殊火焰力量"),
        ("药尘是萧炎的父亲", "药尘是萧炎的重要老师，萧炎的父亲是萧战"),
        ("药老和药尘是两个不同人物", "药老就是药尘，两者指同一个人物"),
        ("萧战是萧炎的老师", "萧战是萧炎的父亲"),
        ("古河就是药老", "古河是被称为丹王的人物，药老是药尘"),
        ("云山就是云韵", "云山和云韵是不同人物，云山与云岚宗关系密切"),
        ("异火是小说中的普通人物", "异火不是人物，而是特殊火焰力量"),
        ("萧薰儿与萧炎毫无关系", "萧薰儿是与萧炎关系密切的重要人物"),
        ("韩枫与药尘没有关系", "韩枫是与药尘有关的重要人物"),
        ("丹王是萧战的称号", "小说中被称为丹王的人物是古河"),
    )
    question_templates = (
        "有人说“{claim}”，这个说法正确吗？",
        "请判断并纠正这句话：“{claim}”。",
        "只按小说事实核对：“{claim}”是否成立？",
        "如果“{claim}”不对，正确说法是什么？",
    )
    answer_templates = (
        "不正确。{correction}。",
        "这句话需要纠正：{correction}。",
        "不成立；按小说事实，{correction}。",
        "正确说法是：{correction}。",
    )
    records: list[dict[str, Any]] = []
    for fact_index, (claim, correction) in enumerate(corrections):
        for index, (question_template, answer_template) in enumerate(
            zip(question_templates, answer_templates, strict=True)
        ):
            split = "train" if index < 2 else "val" if index == 2 else "test"
            records.append(
                make_record(
                    task_family="novel_correction_v5_2",
                    group_key=f"correction|{fact_index}|{index}",
                    split=split,
                    question=question_template.format(claim=claim),
                    answer=answer_template.format(correction=correction),
                    evidence_source="curated_entity_corrections_v1",
                    evidence_text=correction,
                )
            )
    return records


def grounded_unknown_candidates() -> list[dict[str, Any]]:
    question_templates = (
        "已经核对本项目语料，没有找到{name}；现在能确认它的身份吗？",
        "限定只用当前小说语料：检索不到{name}时，应该怎样回答？",
    )
    answer_templates = (
        "当前语料中没有找到该名称，因此不能确认它在本书中的身份。",
        "只依据现有语料，无法核实这个名称；需要新的可靠证据后才能介绍。",
    )
    records: list[dict[str, Any]] = []
    for entity_index, name in enumerate(UNKNOWN_ENTITY_NAMES):
        split = "train" if entity_index < 14 else "val" if entity_index < 17 else "test"
        for index, (question_template, answer_template) in enumerate(
            zip(question_templates, answer_templates, strict=True)
        ):
            records.append(
                make_record(
                    task_family="novel_unknown_grounded_v5_2",
                    group_key=f"unknown|{name}",
                    split=split,
                    question=question_template.format(name=name),
                    answer=answer_template,
                    evidence_source="negative_entity_lookup_v1",
                    evidence_text=f"not found in curated project corpus: {name}",
                )
            )
    return records


def repair_candidates() -> list[dict[str, Any]]:
    return (
        identity_candidates()
        + relation_candidates()
        + correction_candidates()
        + grounded_unknown_candidates()
    )


def filter_duplicate_questions(
    candidates: Sequence[dict[str, Any]],
    forbidden_questions: Iterable[str],
) -> tuple[list[dict[str, Any]], int]:
    seen = {canonicalize_question(question) for question in forbidden_questions}
    accepted: list[dict[str, Any]] = []
    rejected = 0
    for record in candidates:
        canonical = canonicalize_question(record["question"])
        if canonical in seen:
            rejected += 1
            continue
        seen.add(canonical)
        accepted.append(record)
    return accepted, rejected


def validate_entity_routing(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    new_records = [r for r in records if r.get("task_family") in NEW_FAMILIES]
    families = {r["task_family"] for r in new_records}
    if families != NEW_FAMILIES:
        raise ValueError(f"missing v5.2 repair families: {sorted(NEW_FAMILIES - families)}")

    for record in records:
        question = str(record.get("question", ""))
        answer = str(record.get("answer", ""))
        if hidden_prompt_matches(question):
            raise ValueError(f"hidden evaluation prompt overlap: {record.get('id')}")
        if is_meta_introduction_clarification(record):
            raise ValueError(f"meta introduction clarification returned: {record.get('id')}")
        if record.get("task_family") in POSITIVE_FAMILIES and is_refusal_answer(answer):
            raise ValueError(f"known entity routed to refusal: {record.get('id')}")
        if record.get("task_family") == "honest_unknown_general":
            raise ValueError(f"bare unknown refusal family returned: {record.get('id')}")

    identity_train = Counter(
        name
        for record in new_records
        if record["task_family"] in {
            "novel_core_entity_v5_2",
            "novel_known_entity_v5_2",
        }
        and record["split"] == "train"
        for name in KNOWN_ENTITY_PROFILES
        if name in record["question"]
    )
    missing_coverage = {
        name: identity_train[name]
        for name in KNOWN_ENTITY_PROFILES
        if identity_train[name] < (20 if name in CORE_ENTITY_NAMES else 8)
    }
    if missing_coverage:
        raise ValueError(f"insufficient known-entity train coverage: {missing_coverage}")

    identity_records = [
        record
        for record in new_records
        if record["task_family"] in {
            "novel_core_entity_v5_2",
            "novel_known_entity_v5_2",
        }
    ]
    identity_answer_counts = Counter(record["answer"] for record in identity_records)
    if max(identity_answer_counts.values()) > 8:
        raise ValueError("an identity answer repeats more than the capacity-aligned limit")
    for record in identity_records:
        matched = [name for name in KNOWN_ENTITY_PROFILES if name in record["question"]]
        if len(matched) != 1 or matched[0] not in record["answer"]:
            raise ValueError(f"identity answer does not preserve entity name: {record['id']}")

    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in new_records:
        group_splits[record["group_id"]].add(record["split"])
    leaks = {key: splits for key, splits in group_splits.items() if len(splits) > 1}
    if leaks:
        raise ValueError(f"v5.2 group split leakage: {leaks}")

    return {
        "new_records": len(new_records),
        "new_family_counts": dict(Counter(r["task_family"] for r in new_records)),
        "new_split_counts": dict(Counter(r["split"] for r in new_records)),
        "identity_train_coverage": dict(sorted(identity_train.items())),
        "identity_unique_answers": len(identity_answer_counts),
        "identity_answer_max_repeat": max(identity_answer_counts.values()),
        "hidden_prompt_overlap_records": 0,
        "known_entity_refusal_conflicts": 0,
        "bare_unknown_refusal_records": 0,
        "semantic_group_leaks": 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = generate_run_id("sft-v5-2-entity-routing")
    loggers = configure_module_loggers(
        args.report.parent / "logs",
        run_id,
        {"data": "INFO", "validation": "INFO", "checkpoint": "INFO"},
        console=True,
    )
    try:
        base_records = read_jsonl(args.base)
        base_splits = Counter(record["split"] for record in base_records)
        if base_splits != Counter(EXPECTED_BASE_SPLITS):
            raise ValueError(f"unexpected v5.1 base splits: {dict(base_splits)}")

        clean_base, removed, removal_reasons = clean_v5_1_records(base_records)
        tokenizer = BPETokenizer.load(args.tokenizer)
        raw_candidates = repair_candidates()
        for record in raw_candidates:
            if held_out_prompt_matches(record["question"]):
                raise ValueError(f"public evaluation prompt overlap: {record['id']}")
            if hidden_prompt_matches(record["question"]):
                raise ValueError(f"hidden evaluation prompt overlap: {record['id']}")

        encodable, rejected_characters, rejected_by_reason = filter_encodable_candidates(
            raw_candidates,
            tokenizer,
            {record["question"] for record in clean_base},
        )
        encodable, duplicate_count = filter_duplicate_questions(
            encodable,
            (record["question"] for record in clean_base),
        )
        rejected_by_reason = dict(rejected_by_reason)
        rejected_by_reason["canonical_duplicate_question"] = duplicate_count
        if len(encodable) != len(raw_candidates):
            raise ValueError(
                "v5.2 repair candidates must all survive encoding and deduplication; "
                f"raw={len(raw_candidates)} accepted={len(encodable)} "
                f"reasons={rejected_by_reason}"
            )

        final_records = list(clean_base) + list(encodable)
        assert_no_math_or_pollution(final_records)
        routing_quality = validate_entity_routing(final_records)
        final_splits = dict(Counter(record["split"] for record in final_records))
        summary = validate_records(final_records, tokenizer, final_splits)

        atomic_write_text(args.output, jsonl_text(final_records))
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "base_path": str(args.base),
            "base_sha256": file_sha256(args.base),
            "base_records": len(base_records),
            "base_split_counts": dict(base_splits),
            "clean_base_records": len(clean_base),
            "removed_records": len(removed),
            "removed_reasons": dict(removal_reasons),
            "removed_split_counts": dict(Counter(r["split"] for r in removed)),
            "raw_repair_candidates": len(raw_candidates),
            "accepted_repair_candidates": len(encodable),
            "rejected_repair_candidate_reasons": rejected_by_reason,
            "rejected_repair_candidate_characters": dict(rejected_characters),
            "routing_quality": routing_quality,
            "final_records": len(final_records),
            "final_split_counts": final_splits,
            "summary": summary,
            "tokenizer_path": str(args.tokenizer),
            "tokenizer_sha256": file_sha256(args.tokenizer),
            "output_path": str(args.output),
            "output_sha256": file_sha256(args.output),
            "test_records_consumed_for_training": 0,
            "purpose": (
                "Replace bare-name refusal routing with evidence-grounded unknown "
                "handling and dense, natural known-entity supervision."
            ),
        }
        atomic_write_json(args.report, report)
        loggers["data"].info(
            "built v5.2 base=%d removed=%d repairs=%d final=%d splits=%s",
            len(base_records),
            len(removed),
            len(encodable),
            len(final_records),
            final_splits,
        )
        loggers["validation"].info(
            "entity routing gates passed known_conflicts=0 bare_unknown=0 "
            "hidden_overlap=0 identity_unique_answers=%d identity_max_repeat=%d",
            routing_quality["identity_unique_answers"],
            routing_quality["identity_answer_max_repeat"],
        )
        loggers["checkpoint"].info(
            "wrote data=%s report=%s sha256=%s",
            args.output,
            args.report,
            report["output_sha256"],
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        loggers["validation"].exception("v5.2 entity-routing build failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
