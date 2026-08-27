from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any
import json

import torch

from prepare_sft_data import (
    BASE_TENSOR_PATH,
    CORPUS_PATH,
    REQUIRED_FIELDS,
    build_special_token_ids,
    configure_logger,
    load_jsonl,
    serialize_record,
    sha256_file,
)


SOURCE_DATA_PATH = Path("data/sft/sft_hq1000_v2.jsonl")
CONVERSATION_SEED_PATH = Path(
    "data/sft/sft_balanced_v3_conversation_seed.json"
)
FIXED_PROMPT_PATH = Path("data/prompt10_eval.txt")
DATASET_PATH = Path("data/sft/sft_balanced_v3.jsonl")
TENSOR_PATH = Path("data/sft/sft_balanced_v3_tensors.pt")
MILESTONE_DIR = Path("reports/milestones/003g_sft_balanced_v3")
REPORT_PATH = MILESTONE_DIR / "sft_balanced_v3_report.json"
SAMPLE_PATH = MILESTONE_DIR / "sft_balanced_v3_quality_sample30.json"

FINAL_SPLITS = {"train": 800, "val": 100, "test": 100}
FAMILY_COUNTS = {
    "grounded_fact": 200,
    "fact_verification": 150,
    "concept_identity": 150,
    "explicit_instruction": 200,
    "conversation": 150,
    "honest_unknown": 150,
}
FAMILY_SPLITS = {
    "grounded_fact": {"train": 160, "val": 30, "test": 10},
    "fact_verification": {"train": 120, "val": 15, "test": 15},
    "concept_identity": {"train": 120, "val": 15, "test": 15},
    "explicit_instruction": {"train": 160, "val": 15, "test": 25},
    "conversation": {"train": 120, "val": 15, "test": 15},
    "honest_unknown": {"train": 120, "val": 10, "test": 20},
}
UNKNOWN_ANSWER = "现有资料不足，无法确定。"


def configure_balanced_logging() -> dict[str, Any]:
    return {
        "generation": configure_logger(
            "sft.balanced.generation",
            Path("logs/sft_balanced_generation.log"),
            "SFT_BALANCED_GENERATION_LOG_LEVEL",
        ),
        "validation": configure_logger(
            "sft.balanced.validation",
            Path("logs/sft_balanced_validation.log"),
            "SFT_BALANCED_VALIDATION_LOG_LEVEL",
        ),
        "output": configure_logger(
            "sft.balanced.output",
            Path("logs/sft_balanced_output.log"),
            "SFT_BALANCED_OUTPUT_LOG_LEVEL",
        ),
    }


def stable_hash(*parts: object) -> str:
    joined = "|".join(str(part) for part in parts)
    return sha256(joined.encode("utf-8")).hexdigest()


def copy_record(record: dict[str, Any], **updates: Any) -> dict[str, Any]:
    copied = dict(record)
    copied.update(updates)
    return copied


def load_source_groups() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_records = load_jsonl(SOURCE_DATA_PATH)
    old_facts = source_records[:100]
    concept_by_label = {}
    for record in source_records[100:]:
        concept_by_label.setdefault(record["concept_label"], record)
    concepts = list(concept_by_label.values())
    concepts.sort(key=lambda record: stable_hash("balanced-v3-concept", record["topic"]))
    if len(old_facts) != 100 or len(concepts) != 150:
        raise ValueError("source dataset must contain 100 facts and 150 concepts")
    return old_facts, concepts


def build_grounded_facts(old_facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for fact in old_facts:
        common = {
            "evidence_type": "corpus_grounded",
            "task_family": "grounded_fact",
        }
        records.append(
            copy_record(
                fact,
                **common,
                generation_method="grounded_fact_original",
            )
        )
        records.append(
            copy_record(
                fact,
                **common,
                question=f"请直接回答这个问题：{fact['question']}",
                generation_method="grounded_fact_direct_instruction",
            )
        )
    return records


def false_answer_pool(
    fact: dict[str, Any], candidates: list[dict[str, Any]], count: int
) -> list[str]:
    answers = sorted(
        {item["answer"] for item in candidates if item["answer"] != fact["answer"]},
        key=lambda answer: stable_hash("balanced-v3-wrong", fact["topic"], answer),
    )
    if len(answers) < count:
        raise ValueError(f"not enough false answers for {fact['topic']}")
    return answers[:count]


def build_fact_verification(old_facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    by_split = defaultdict(list)
    for fact in old_facts:
        by_split[fact["split"]].append(fact)
        records.append(
            copy_record(
                fact,
                question=(
                    f"根据现有资料，对问题“{fact['question']}”回答"
                    f"“{fact['answer']}”是否正确？"
                ),
                answer="正确。",
                evidence_type="corpus_grounded",
                task_family="fact_verification",
                generation_method="fact_verification_positive",
                candidate_answer=fact["answer"],
                reference_answer=fact["answer"],
            )
        )

    train_facts = sorted(
        by_split["train"], key=lambda fact: stable_hash("balanced-v3-neg-train", fact["topic"])
    )[:40]
    for fact in train_facts:
        wrong_answer = false_answer_pool(fact, by_split["train"], 1)[0]
        records.append(
            copy_record(
                fact,
                question=(
                    f"根据现有资料，对问题“{fact['question']}”回答"
                    f"“{wrong_answer}”是否正确？"
                ),
                answer="不正确。",
                evidence_type="corpus_grounded",
                task_family="fact_verification",
                generation_method="fact_verification_negative",
                candidate_answer=wrong_answer,
                reference_answer=fact["answer"],
            )
        )
    for fact in by_split["test"]:
        for wrong_answer in false_answer_pool(fact, by_split["test"], 2):
            records.append(
                copy_record(
                    fact,
                    question=(
                        f"根据现有资料，对问题“{fact['question']}”回答"
                        f"“{wrong_answer}”是否正确？"
                    ),
                    answer="不正确。",
                    evidence_type="corpus_grounded",
                    task_family="fact_verification",
                    generation_method="fact_verification_negative",
                    candidate_answer=wrong_answer,
                    reference_answer=fact["answer"],
                )
            )
    return records


def build_concept_identity(concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for concept in concepts:
        label = concept["concept_label"]
        category = concept["concept_category"]
        records.append(
            copy_record(
                concept,
                question=f"在这部小说中，“{label}”是什么？",
                answer=f"“{label}”属于{category}。",
                evidence_type="corpus_grounded",
                task_family="concept_identity",
                generation_method="concept_identity_natural",
            )
        )
    return records


def build_explicit_instructions(concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    by_split = defaultdict(list)
    for concept in concepts:
        by_split[concept["split"]].append(concept)

    selected_val = sorted(
        by_split["val"],
        key=lambda item: stable_hash("balanced-v3-compare-val", item["topic"]),
    )[:5]
    selected_val_topics = {item["topic"] for item in selected_val}
    for concept in concepts:
        label = concept["concept_label"]
        if concept["topic"] in selected_val_topics:
            continue
        records.append(
            copy_record(
                concept,
                question=f"请原样重复这个名称：“{label}”。",
                answer=label,
                evidence_type="deterministic_instruction",
                task_family="explicit_instruction",
                generation_method="instruction_exact_copy",
            )
        )

    selected_train = sorted(
        by_split["train"],
        key=lambda item: stable_hash("balanced-v3-compare-train", item["topic"]),
    )[:40]
    train_labels = [item["concept_label"] for item in selected_train]
    for index, concept in enumerate(selected_train):
        label = concept["concept_label"]
        same = index % 2 == 0
        other = label if same else train_labels[(index + 1) % len(train_labels)]
        records.append(
            copy_record(
                concept,
                question=f"“{label}”和“{other}”是否完全相同？",
                answer="相同。" if same else "不相同。",
                evidence_type="deterministic_instruction",
                task_family="explicit_instruction",
                generation_method=(
                    "instruction_compare_same"
                    if same
                    else "instruction_compare_different"
                ),
                left_text=label,
                right_text=other,
            )
        )

    val_labels = [item["concept_label"] for item in selected_val]
    for index, concept in enumerate(selected_val):
        label = concept["concept_label"]
        same = index % 2 == 0
        other = label if same else val_labels[(index + 1) % len(val_labels)]
        records.append(
            copy_record(
                concept,
                question=f"“{label}”和“{other}”是否完全相同？",
                answer="相同。" if same else "不相同。",
                evidence_type="deterministic_instruction",
                task_family="explicit_instruction",
                generation_method=(
                    "instruction_compare_same"
                    if same
                    else "instruction_compare_different"
                ),
                left_text=label,
                right_text=other,
            )
        )

    test_concepts = sorted(
        by_split["test"],
        key=lambda item: stable_hash("balanced-v3-compare-test", item["topic"]),
    )[:5]
    test_labels = [item["concept_label"] for item in test_concepts]
    for index, concept in enumerate(test_concepts):
        label = concept["concept_label"]
        other = test_labels[(index + 1) % len(test_labels)]
        records.extend(
            [
                copy_record(
                    concept,
                    question=f"“{label}”和“{label}”是否完全相同？",
                    answer="相同。",
                    evidence_type="deterministic_instruction",
                    task_family="explicit_instruction",
                    generation_method="instruction_compare_same",
                    left_text=label,
                    right_text=label,
                ),
                copy_record(
                    concept,
                    question=f"“{label}”和“{other}”是否完全相同？",
                    answer="不相同。",
                    evidence_type="deterministic_instruction",
                    task_family="explicit_instruction",
                    generation_method="instruction_compare_different",
                    left_text=label,
                    right_text=other,
                ),
            ]
        )
    return records


def build_conversations() -> list[dict[str, Any]]:
    seeds = json.loads(CONVERSATION_SEED_PATH.read_text(encoding="utf-8"))
    if len(seeds) != 50:
        raise ValueError("conversation seed must contain exactly 50 intents")
    records = []
    for index, seed in enumerate(seeds):
        split = "train" if index < 40 else "val" if index < 45 else "test"
        if len(seed["questions"]) != 3:
            raise ValueError(f"conversation intent {seed['intent']} needs three questions")
        for question in seed["questions"]:
            records.append(
                {
                    "question": question,
                    "answer": seed["answer"],
                    "evidence": "人工编写并审核的通用对话行为",
                    "source_line": None,
                    "topic": f"conversation_{seed['intent']}",
                    "split": split,
                    "generation_method": "curated_conversation",
                    "evidence_type": "curated_behavior",
                    "task_family": "conversation",
                }
            )
    return records


def find_unknown_labels(corpus_text: str, known_labels: set[str]) -> list[str]:
    prefixes = "玄青紫赤白黑金银天星"
    middles = "云风雷火山海灵月阳河"
    suffixes = ("门", "宗", "谷", "城", "宫")
    labels = []
    for prefix in prefixes:
        for middle in middles:
            for suffix in suffixes:
                label = prefix + middle + suffix
                if label in corpus_text or label in known_labels:
                    continue
                labels.append(label)
    labels.sort(key=lambda label: stable_hash("balanced-v3-unknown", label))
    if len(labels) < 50:
        raise ValueError("not enough absent synthetic labels")
    return labels[:50]


def build_honest_unknown(
    corpus_text: str, known_labels: set[str]
) -> list[dict[str, Any]]:
    labels = find_unknown_labels(corpus_text, known_labels)
    templates = (
        "“{label}”的创建者是谁？",
        "“{label}”位于什么地方？",
        "“{label}”在小说中有什么作用？",
        "请介绍“{label}”的来历。",
    )
    records = []
    for index, label in enumerate(labels):
        if index < 40:
            split, variant_count = "train", 3
        elif index < 43:
            split = "val"
            variant_count = 4 if index == 42 else 3
        else:
            split = "test"
            variant_count = 2 if index == 49 else 3
        for template in templates[:variant_count]:
            records.append(
                {
                    "question": template.format(label=label),
                    "answer": UNKNOWN_ANSWER,
                    "evidence": "虚构名称未出现在授权语料中",
                    "source_line": None,
                    "topic": f"unknown_{stable_hash(label)[:12]}",
                    "split": split,
                    "generation_method": "honest_unknown_absent_entity",
                    "evidence_type": "verified_absence",
                    "task_family": "honest_unknown",
                    "concept_label": label,
                }
            )
    return records


def assign_ids(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters = Counter()
    output = []
    for record in records:
        split = record["split"]
        counters[split] += 1
        output.append(copy_record(record, id=f"{split}_v3_{counters[split]:04d}"))
    return output


def validate_text_vocabulary(
    records: list[dict[str, Any]], stoi: dict[str, int]
) -> None:
    missing_by_character = defaultdict(list)
    for record in records:
        for character in sorted(set(record["question"] + record["answer"]) - set(stoi)):
            missing_by_character[character].append(record["id"])
    if missing_by_character:
        summary = {
            character: record_ids[:5]
            for character, record_ids in sorted(missing_by_character.items())
        }
        raise ValueError(f"dataset contains OOV characters: {summary}")


def validate_records(
    records: list[dict[str, Any]],
    corpus_lines: list[str],
    corpus_text: str,
    stoi: dict[str, int],
    prepared_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(records) != 1000:
        raise ValueError(f"expected 1000 records, got {len(records)}")
    if Counter(record["split"] for record in records) != FINAL_SPLITS:
        raise ValueError("final split counts are incorrect")
    if Counter(record["task_family"] for record in records) != FAMILY_COUNTS:
        raise ValueError("task family counts are incorrect")
    for family, expected in FAMILY_SPLITS.items():
        actual = Counter(
            record["split"] for record in records if record["task_family"] == family
        )
        if actual != expected:
            raise ValueError(f"family {family} split mismatch {dict(actual)}")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("record IDs must be unique")
    if len({record["question"] for record in records}) != len(records):
        raise ValueError("questions must be unique")

    topic_splits = defaultdict(set)
    evidence_counts = Counter()
    for record in records:
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"{record.get('id')} missing required fields {missing}")
        missing_chars = sorted(set(record["question"] + record["answer"]) - set(stoi))
        if missing_chars:
            raise ValueError(f"{record['id']} contains OOV characters {missing_chars}")
        topic_splits[record["topic"]].add(record["split"])
        evidence_type = record["evidence_type"]
        evidence_counts[evidence_type] += 1
        if record["source_line"] is not None:
            line_number = record["source_line"]
            if not 1 <= line_number <= len(corpus_lines):
                raise ValueError(f"{record['id']} has invalid source line")
            if record["evidence"] not in corpus_lines[line_number - 1]:
                raise ValueError(f"{record['id']} evidence mismatch")
        elif evidence_type == "verified_absence":
            if record["concept_label"] in corpus_text:
                raise ValueError(f"{record['id']} unknown label exists in corpus")
            if record["answer"] != UNKNOWN_ANSWER:
                raise ValueError(f"{record['id']} unknown answer mismatch")
        elif evidence_type != "curated_behavior":
            raise ValueError(f"{record['id']} lacks auditable evidence")

        method = record["generation_method"]
        if method == "fact_verification_positive":
            if record["candidate_answer"] != record["reference_answer"]:
                raise ValueError(f"{record['id']} positive verification is false")
            if record["answer"] != "正确。":
                raise ValueError(f"{record['id']} positive label is incorrect")
        elif method == "fact_verification_negative":
            if record["candidate_answer"] == record["reference_answer"]:
                raise ValueError(f"{record['id']} negative verification is true")
            if record["answer"] != "不正确。":
                raise ValueError(f"{record['id']} negative label is incorrect")
        elif method == "instruction_exact_copy":
            if record["answer"] != record["concept_label"]:
                raise ValueError(f"{record['id']} copy instruction changed the text")
        elif method == "concept_identity_natural":
            expected = f"“{record['concept_label']}”属于{record['concept_category']}。"
            if record["answer"] != expected:
                raise ValueError(f"{record['id']} concept identity mismatch")
        elif method == "instruction_compare_same":
            if record["left_text"] != record["right_text"] or record["answer"] != "相同。":
                raise ValueError(f"{record['id']} same comparison mismatch")
        elif method == "instruction_compare_different":
            if record["left_text"] == record["right_text"] or record["answer"] != "不相同。":
                raise ValueError(f"{record['id']} different comparison mismatch")

    leaked_topics = [topic for topic, splits in topic_splits.items() if len(splits) > 1]
    if leaked_topics:
        raise ValueError(f"topics leak across splits: {leaked_topics[:5]}")
    lengths = [record["sequence_length"] for record in prepared_records]
    if max(lengths) > 256:
        raise ValueError("a sequence exceeds the 256-token context")
    fixed_prompts = {
        line.strip()
        for line in FIXED_PROMPT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    fixed_prompt_overlap = sorted(
        fixed_prompts & {record["question"] for record in records}
    )
    if fixed_prompt_overlap:
        raise ValueError(f"fixed evaluation prompts leak into training data: {fixed_prompt_overlap}")
    answer_counts = Counter(record["answer"] for record in records)
    most_common_answer, most_common_answer_count = answer_counts.most_common(1)[0]
    comparison_splits = {
        split: dict(
            Counter(
                record["answer"]
                for record in records
                if record["split"] == split
                and record["generation_method"].startswith("instruction_compare")
            )
        )
        for split in FINAL_SPLITS
    }
    return {
        "split_counts": dict(Counter(record["split"] for record in records)),
        "task_family_counts": dict(Counter(record["task_family"] for record in records)),
        "task_family_splits": {
            family: dict(
                Counter(
                    record["split"]
                    for record in records
                    if record["task_family"] == family
                )
            )
            for family in FAMILY_COUNTS
        },
        "generation_method_counts": dict(
            Counter(record["generation_method"] for record in records)
        ),
        "evidence_type_counts": dict(evidence_counts),
        "unique_question_count": len(records),
        "unique_topic_count": len(topic_splits),
        "topic_split_leakage": False,
        "max_sequence_length": max(lengths),
        "mean_sequence_length": sum(lengths) / len(lengths),
        "classification_share": FAMILY_COUNTS["concept_identity"] / len(records),
        "unique_answer_count": len(answer_counts),
        "most_common_answer": most_common_answer,
        "most_common_answer_count": most_common_answer_count,
        "most_common_answer_share": most_common_answer_count / len(records),
        "conversation_intent_count": len(
            {
                record["topic"]
                for record in records
                if record["task_family"] == "conversation"
            }
        ),
        "unknown_entity_count": len(
            {
                record["topic"]
                for record in records
                if record["task_family"] == "honest_unknown"
            }
        ),
        "comparison_answer_splits": comparison_splits,
        "fixed_prompt_exact_overlap": [],
    }


def quality_sample(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for record in records:
        groups[record["task_family"]].append(record)
    sample = []
    for family, items in sorted(groups.items()):
        ordered = sorted(
            items,
            key=lambda record: stable_hash("balanced-v3-sample", family, record["id"]),
        )
        sample.extend(ordered[:5])
    return sample


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    loggers = configure_balanced_logging()
    try:
        corpus_text = CORPUS_PATH.read_text(encoding="utf-8")
        corpus_lines = corpus_text.splitlines()
        base_payload = torch.load(BASE_TENSOR_PATH, map_location="cpu", weights_only=False)
        stoi = base_payload["stoi"]
        old_facts, concepts = load_source_groups()

        family_records = {
            "grounded_fact": build_grounded_facts(old_facts),
            "fact_verification": build_fact_verification(old_facts),
            "concept_identity": build_concept_identity(concepts),
            "explicit_instruction": build_explicit_instructions(concepts),
            "conversation": build_conversations(),
            "honest_unknown": build_honest_unknown(
                corpus_text, {record["concept_label"] for record in concepts}
            ),
        }
        for family, records in family_records.items():
            loggers["generation"].info(
                "built family=%s count=%d splits=%s",
                family,
                len(records),
                dict(Counter(record["split"] for record in records)),
            )
        records = assign_ids(
            [record for family in FAMILY_COUNTS for record in family_records[family]]
        )

        special_token_ids = build_special_token_ids(int(base_payload["vocab_size"]))
        validate_text_vocabulary(records, stoi)
        prepared_records = [
            serialize_record(record, stoi, special_token_ids) for record in records
        ]
        validation = validate_records(
            records, corpus_lines, corpus_text, stoi, prepared_records
        )
        loggers["validation"].info(
            "validated records=1000 train=800 val=100 test=100 classification_share=0.15"
        )

        write_jsonl(DATASET_PATH, records)
        split_records = {
            split: [record for record in prepared_records if record["split"] == split]
            for split in FINAL_SPLITS
        }
        torch.save(
            {
                "train_records": split_records["train"],
                "val_records": split_records["val"],
                "test_records": split_records["test"],
                "base_vocab_size": int(base_payload["vocab_size"]),
                "vocab_size": int(base_payload["vocab_size"]) + len(special_token_ids),
                "stoi": stoi,
                "itos": base_payload["itos"],
                "special_token_ids": special_token_ids,
                "ignore_index": -100,
            },
            TENSOR_PATH,
        )

        sample = quality_sample(records)
        MILESTONE_DIR.mkdir(parents=True, exist_ok=True)
        SAMPLE_PATH.write_text(
            json.dumps(sample, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report = {
            "milestone": "M003g",
            "dataset_version": "sft_balanced_v3",
            "record_count": len(records),
            **validation,
            "source_dataset": str(SOURCE_DATA_PATH),
            "source_dataset_sha256": sha256_file(SOURCE_DATA_PATH),
            "conversation_seed": str(CONVERSATION_SEED_PATH),
            "conversation_seed_sha256": sha256_file(CONVERSATION_SEED_PATH),
            "corpus_sha256": sha256_file(CORPUS_PATH),
            "dataset_sha256": sha256_file(DATASET_PATH),
            "quality_sample_sha256": sha256_file(SAMPLE_PATH),
            "base_vocab_size": int(base_payload["vocab_size"]),
            "extended_vocab_size": int(base_payload["vocab_size"])
            + len(special_token_ids),
            "supervised_label_count": sum(
                int((record["labels"] != -100).sum()) for record in prepared_records
            ),
            "masked_label_count": sum(
                int((record["labels"] == -100).sum()) for record in prepared_records
            ),
            "test_records_consumed": 0,
        }
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        loggers["output"].info(
            "wrote dataset=%s tensors=%s sample=%s report=%s",
            DATASET_PATH,
            TENSOR_PATH,
            SAMPLE_PATH,
            REPORT_PATH,
        )
    except Exception:
        loggers["validation"].exception("balanced SFT v3 build failed")
        raise


if __name__ == "__main__":
    main()
