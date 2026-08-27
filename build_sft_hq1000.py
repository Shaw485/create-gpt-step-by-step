from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
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


BASE_DATASET_PATH = Path("data/sft/sft_pilot50_v1.jsonl")
EXPANSION_50_PATH = Path("data/sft/sft_expansion50_v1.jsonl")
EXPANSION_900_PATH = Path("data/sft/sft_hq1000_expansion900_v2.jsonl")
DATASET_PATH = Path("data/sft/sft_hq1000_v2.jsonl")
TENSOR_PATH = Path("data/sft/sft_hq1000_v2_tensors.pt")
MILESTONE_DIR = Path("reports/milestones/003d_sft_hq1000")
REPORT_PATH = MILESTONE_DIR / "sft_hq1000_report.json"
SAMPLE_PATH = MILESTONE_DIR / "sft_hq1000_quality_sample24.json"

FINAL_SPLITS = {"train": 800, "val": 100, "test": 100}
BASE_SPLITS = {"train": 80, "val": 15, "test": 5}
NEW_SPLITS = {"train": 720, "val": 85, "test": 95}

CATEGORY_TERMS = {
    "人物": (
        "萧炎", "药老", "萧战", "薰儿", "纳兰嫣然", "云韵", "美杜莎", "小医仙",
        "海波东", "古河", "云山", "韩枫", "紫妍", "青鳞", "彩鳞", "林修崖",
        "柳擎", "苏千", "法犸", "加刑天", "雅妃", "若琳", "萧玉", "萧鼎",
        "萧厉", "夭夜", "夭月", "木辰", "纳兰桀", "纳兰肃", "凌影", "古元",
        "魂天帝", "烛坤", "净莲妖圣", "黄泉妖圣", "风尊者", "慕骨老人",
        "曹颖", "丹晨", "玄空子", "天雷子", "玄衣", "唐震", "唐火儿", "冰河",
        "天火尊者", "熊战", "雷尊者", "剑尊者",
    ),
    "斗技或功法": (
        "焚决", "八极崩", "焰分噬浪尺", "三千雷动", "佛怒火莲", "大天造化掌",
        "黄泉天怒", "黄泉指", "黄泉掌", "天火三玄变", "紫云翼", "鹰之翼",
        "帝印决", "开山印", "翻海印", "覆地印", "湮天印", "古帝印",
        "弄焰决", "六合游身尺", "风之极", "吸掌", "吹火掌",
    ),
    "异火": (
        "青莲地心火", "陨落心炎", "净莲妖火", "三千焱炎火", "骨灵冷火",
        "海心焰", "金帝焚天炎", "九龙雷罡火", "生灵之焱", "虚无吞炎",
        "万兽灵火", "八荒破灭焱", "红莲业火", "九幽金祖火", "风怒龙炎",
        "九幽风炎", "火云水焱", "玄黄炎",
    ),
    "丹药": (
        "聚气散", "血莲丹", "复灵紫丹", "融灵丹", "地灵丹", "破宗丹",
        "斗灵丹", "皇极丹", "菩提丹", "阴阳玄龙丹", "噬生丹", "回气丹",
        "筑基灵液", "洗髓寒灵液", "龙力丹", "天魂融血丹",
    ),
    "势力": (
        "云岚宗", "迦南学院", "丹塔", "魂殿", "古族", "萧族", "魂族", "炎族",
        "雷族", "药族", "石族", "灵族", "太虚古龙", "天妖凰族", "花宗",
        "焚炎谷", "风雷阁", "星陨阁", "冰河谷", "黄泉阁", "万剑阁", "四方阁",
        "黑盟", "磐门", "萧门", "毒宗", "蛇人族", "加玛皇室", "米特尔家族",
        "纳兰家族", "木家", "韩家", "曹家", "丹家", "叶家",
    ),
    "地点": (
        "乌坦城", "黑角域", "加玛帝国", "出云帝国", "魔兽山脉", "塔戈尔大沙漠",
        "中州", "圣丹城", "亡魂山脉", "天目山脉", "骸骨山脉", "古界", "天墓",
        "蛮荒古域", "黑皇城", "枫城", "和平镇", "内院", "天焚炼气塔",
    ),
    "物品或体质": (
        "玄重尺", "药鼎", "纳戒", "魔核", "厄难毒体", "碧蛇三花瞳", "斗帝血脉",
        "菩提古树", "菩提心", "陀舍古帝玉",
    ),
}

CATEGORY_QUOTAS = {
    "人物": 46,
    "斗技或功法": 20,
    "异火": 15,
    "丹药": 15,
    "势力": 26,
    "地点": 18,
    "物品或体质": 10,
}

CATEGORY_SPLIT_QUOTAS = {
    "人物": {"train": 37, "val": 5, "test": 4},
    "斗技或功法": {"train": 16, "val": 2, "test": 2},
    "异火": {"train": 12, "val": 1, "test": 2},
    "丹药": {"train": 12, "val": 2, "test": 1},
    "势力": {"train": 21, "val": 2, "test": 3},
    "地点": {"train": 14, "val": 2, "test": 2},
    "物品或体质": {"train": 8, "val": 1, "test": 1},
}

ANSWER_CATEGORIES = tuple(CATEGORY_TERMS)


@dataclass(frozen=True)
class Concept:
    label: str
    category: str
    source_line: int
    split: str
    variant_count: int


def configure_hq_logging() -> dict[str, Any]:
    return {
        "generation": configure_logger(
            "sft.hq.generation",
            Path("logs/sft_hq_generation.log"),
            "SFT_HQ_GENERATION_LOG_LEVEL",
        ),
        "validation": configure_logger(
            "sft.hq.validation",
            Path("logs/sft_hq_validation.log"),
            "SFT_HQ_VALIDATION_LOG_LEVEL",
        ),
        "output": configure_logger(
            "sft.hq.output",
            Path("logs/sft_hq_output.log"),
            "SFT_HQ_OUTPUT_LOG_LEVEL",
        ),
    }


def stable_hash(*parts: object) -> str:
    joined = "|".join(str(part) for part in parts)
    return sha256(joined.encode("utf-8")).hexdigest()


def find_distinct_source_lines(
    labels_by_category: dict[str, tuple[str, ...]],
    corpus_lines: list[str],
    excluded_lines: set[int],
) -> dict[tuple[str, str], int]:
    used_lines = set(excluded_lines)
    result = {}
    for category, labels in labels_by_category.items():
        ordered_labels = sorted(
            labels, key=lambda label: stable_hash("hq1000-label-v2", category, label)
        )
        selected = 0
        for label in ordered_labels:
            candidate_lines = [
                line_number
                for line_number, line in enumerate(corpus_lines, 1)
                if line_number not in used_lines and label in line
            ]
            if not candidate_lines:
                continue
            candidate_lines.sort(
                key=lambda line_number: stable_hash(
                    "hq1000-source-v2", category, label, line_number
                )
            )
            source_line = candidate_lines[0]
            result[(category, label)] = source_line
            used_lines.add(source_line)
            selected += 1
            if selected == CATEGORY_QUOTAS[category]:
                break
        if selected != CATEGORY_QUOTAS[category]:
            raise ValueError(
                f"category {category} only found {selected} of "
                f"{CATEGORY_QUOTAS[category]} distinct concepts"
            )
    return result


def assign_concepts(
    source_lines: dict[tuple[str, str], int]
) -> list[Concept]:
    concepts = []
    split_positions: Counter[str] = Counter()
    for category in CATEGORY_TERMS:
        category_labels = [
            label for found_category, label in source_lines if found_category == category
        ]
        category_labels.sort(
            key=lambda label: stable_hash("hq1000-concept-split-v2", category, label)
        )
        cursor = 0
        for split, count in CATEGORY_SPLIT_QUOTAS[category].items():
            split_labels = category_labels[cursor : cursor + count]
            cursor += count
            for label in split_labels:
                split_index = split_positions[split]
                split_positions[split] += 1
                if split == "train":
                    variant_count = 6
                elif split == "val":
                    variant_count = 5 if split_index < 5 else 6
                else:
                    variant_count = 7 if split_index < 5 else 6
                concepts.append(
                    Concept(
                        label=label,
                        category=category,
                        source_line=source_lines[(category, label)],
                        split=split,
                        variant_count=variant_count,
                    )
                )
    split_counts = Counter(concept.split for concept in concepts)
    if split_counts != {"train": 120, "val": 15, "test": 15}:
        raise ValueError(f"unexpected concept splits {dict(split_counts)}")
    return concepts


def category_options(category: str, label: str) -> list[str]:
    wrong = [item for item in ANSWER_CATEGORIES if item != category]
    wrong.sort(key=lambda item: stable_hash("hq1000-options-v2", label, item))
    options = [category, *wrong[:3]]
    options.sort(key=lambda item: stable_hash("hq1000-option-order-v2", label, item))
    return options


def wrong_category(category: str, label: str) -> str:
    wrong = [item for item in ANSWER_CATEGORIES if item != category]
    return min(wrong, key=lambda item: stable_hash("hq1000-wrong-v2", label, item))


def concept_variants(concept: Concept) -> list[tuple[str, str, str]]:
    label = concept.label
    category = concept.category
    wrong = wrong_category(category, label)
    options = "、".join(category_options(category, label))
    variants = [
        (f"“{label}”属于什么类别？", f"{category}。", "category_direct"),
        (
            f"请判断“{label}”属于人物、势力、地点、异火、丹药、斗技或功法，还是物品或体质。",
            f"{category}。",
            "category_full_choice",
        ),
        (f"“{label}”是{category}吗？", "是。", "category_positive"),
        (
            f"“{label}”是{wrong}吗？",
            f"不是，“{label}”属于{category}。",
            "category_negative_correction",
        ),
        (
            f"从选项中选择“{label}”的类别。选项：{options}",
            f"{category}。",
            "category_multiple_choice",
        ),
        (
            f"请用一句话说明“{label}”的类型。",
            f"“{label}”属于{category}。",
            "category_explanation",
        ),
        (
            f"在给定分类体系中，“{label}”应归入哪一类？",
            f"{category}。",
            "category_system",
        ),
    ]
    return variants[: concept.variant_count]


def build_new_records(concepts: list[Concept]) -> list[dict[str, Any]]:
    counters = dict(BASE_SPLITS)
    records = []
    for concept in concepts:
        topic = f"concept_{stable_hash(concept.category, concept.label)[:12]}"
        for question, answer, method in concept_variants(concept):
            counters[concept.split] += 1
            records.append(
                {
                    "id": f"{concept.split}_{counters[concept.split]:04d}",
                    "question": question,
                    "answer": answer,
                    "evidence": concept.label,
                    "source_line": concept.source_line,
                    "topic": topic,
                    "split": concept.split,
                    "generation_method": method,
                    "concept_label": concept.label,
                    "concept_category": concept.category,
                }
            )
    if Counter(record["split"] for record in records) != NEW_SPLITS:
        raise ValueError("new records do not match the 720/85/95 split")
    return records


def validate_dataset(
    records: list[dict[str, Any]],
    corpus_lines: list[str],
    stoi: dict[str, int],
    prepared_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(records) != 1000:
        raise ValueError(f"expected 1000 records, got {len(records)}")
    if len({record["id"] for record in records}) != 1000:
        raise ValueError("record ids must be unique")
    if len({record["question"] for record in records}) != 1000:
        raise ValueError("questions must be unique")
    split_counts = Counter(record["split"] for record in records)
    if split_counts != FINAL_SPLITS:
        raise ValueError(f"unexpected split counts {dict(split_counts)}")

    topic_splits: dict[str, set[str]] = defaultdict(set)
    concept_splits: dict[str, set[str]] = defaultdict(set)
    source_line_splits: dict[int, set[str]] = defaultdict(set)
    new_source_lines = set()
    for index, record in enumerate(records):
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"{record.get('id')} missing {missing}")
        line_number = record["source_line"]
        if not 1 <= line_number <= len(corpus_lines):
            raise ValueError(f"{record['id']} has an invalid source line")
        if record["evidence"] not in corpus_lines[line_number - 1]:
            raise ValueError(f"{record['id']} evidence mismatch")
        if set(record["question"] + record["answer"]) - set(stoi):
            raise ValueError(f"{record['id']} contains out-of-vocabulary characters")
        topic_splits[record["topic"]].add(record["split"])
        source_line_splits[line_number].add(record["split"])
        if index >= 100:
            label = record["concept_label"]
            category = record["concept_category"]
            concept_splits[label].add(record["split"])
            new_source_lines.add(line_number)
            if category not in record["answer"] and record["answer"] != "是。":
                raise ValueError(f"{record['id']} answer does not preserve category")

    leaked_topics = [topic for topic, splits in topic_splits.items() if len(splits) > 1]
    leaked_concepts = [
        label for label, splits in concept_splits.items() if len(splits) > 1
    ]
    if leaked_topics or leaked_concepts:
        raise ValueError("a topic or concept leaks across dataset splits")
    leaked_source_lines = sorted(
        line_number
        for line_number, splits in source_line_splits.items()
        if len(splits) > 1
    )
    if leaked_source_lines != [378]:
        raise ValueError(f"unexpected source-line overlap {leaked_source_lines}")
    if len(new_source_lines) != 150:
        raise ValueError("the 150 new concepts must use distinct source lines")

    lengths = [record["sequence_length"] for record in prepared_records]
    if max(lengths) > 256:
        raise ValueError("a sequence exceeds the model context window")
    return {
        "split_counts": dict(split_counts),
        "unique_question_count": 1000,
        "unique_topic_count": len(topic_splits),
        "new_concept_count": len(concept_splits),
        "new_unique_source_line_count": len(new_source_lines),
        "topic_split_leakage": False,
        "concept_split_leakage": False,
        "pre_existing_source_line_overlap": [378],
        "new_source_line_split_leakage": False,
        "max_sequence_length": max(lengths),
        "mean_sequence_length": sum(lengths) / len(lengths),
    }


def build_quality_sample(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group = record.get("generation_method", "human_style_grounded_fact_qa")
        grouped[group].append(record)
    sample = []
    for group, group_records in grouped.items():
        ordered = sorted(
            group_records,
            key=lambda record: stable_hash("hq1000-quality-sample-v2", group, record["id"]),
        )
        sample.extend(ordered[:3])
    return sample


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    loggers = configure_hq_logging()
    try:
        corpus_text = CORPUS_PATH.read_text(encoding="utf-8")
        corpus_lines = corpus_text.splitlines()
        base_payload = torch.load(BASE_TENSOR_PATH, map_location="cpu", weights_only=False)
        stoi = base_payload["stoi"]
        old_records = load_jsonl(BASE_DATASET_PATH) + load_jsonl(EXPANSION_50_PATH)
        old_records = [
            {**record, "generation_method": "human_style_grounded_fact_qa"}
            for record in old_records
        ]
        source_lines = find_distinct_source_lines(
            CATEGORY_TERMS,
            corpus_lines,
            {record["source_line"] for record in old_records},
        )
        concepts = assign_concepts(source_lines)
        new_records = build_new_records(concepts)
        final_records = old_records + new_records
        loggers["generation"].info(
            "built old_fact_qa=%d concepts=%d new_records=%d",
            len(old_records),
            len(concepts),
            len(new_records),
        )

        special_token_ids = build_special_token_ids(int(base_payload["vocab_size"]))
        prepared_records = [
            serialize_record(record, stoi, special_token_ids) for record in final_records
        ]
        validation = validate_dataset(
            final_records, corpus_lines, stoi, prepared_records
        )
        loggers["validation"].info(
            "validated records=1000 concepts=150 train=800 val=100 test=100"
        )

        write_jsonl(EXPANSION_900_PATH, new_records)
        write_jsonl(DATASET_PATH, final_records)
        sample = build_quality_sample(final_records)
        MILESTONE_DIR.mkdir(parents=True, exist_ok=True)
        SAMPLE_PATH.write_text(
            json.dumps(sample, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
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
                "vocab_size": int(base_payload["vocab_size"])
                + len(special_token_ids),
                "stoi": stoi,
                "itos": base_payload["itos"],
                "special_token_ids": special_token_ids,
                "ignore_index": -100,
            },
            TENSOR_PATH,
        )

        report = {
            "milestone": "M003d",
            "dataset_version": "sft_hq1000_v2",
            "record_count": len(final_records),
            **validation,
            "data_composition": {
                "human_style_grounded_fact_qa": 100,
                "clean_concept_classification_records": 900,
            },
            "category_concept_counts": dict(
                Counter(concept.category for concept in concepts)
            ),
            "generation_method_counts": dict(
                Counter(record["generation_method"] for record in new_records)
            ),
            "evidence_verified": True,
            "base_five_test_ids_preserved": [f"test_{index:03d}" for index in range(1, 6)],
            "base_vocab_size": int(base_payload["vocab_size"]),
            "extended_vocab_size": int(base_payload["vocab_size"])
            + len(special_token_ids),
            "supervised_label_count": sum(
                int((record["labels"] != -100).sum()) for record in prepared_records
            ),
            "masked_label_count": sum(
                int((record["labels"] == -100).sum()) for record in prepared_records
            ),
            "quality_sample_count": len(sample),
            "corpus_sha256": sha256_file(CORPUS_PATH),
            "dataset_sha256": sha256_file(DATASET_PATH),
            "expansion_sha256": sha256_file(EXPANSION_900_PATH),
            "quality_sample_sha256": sha256_file(SAMPLE_PATH),
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
        loggers["validation"].exception("hq1000 build failed")
        raise


if __name__ == "__main__":
    main()
