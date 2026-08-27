from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import json
import re

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
EXPANSION_900_PATH = Path("data/sft/sft_hq1000_expansion900_v1.jsonl")
DATASET_PATH = Path("data/sft/sft_hq1000_v1.jsonl")
TENSOR_PATH = Path("data/sft/sft_hq1000_v1_tensors.pt")
REPORT_PATH = Path(
    "reports/milestones/003d_sft_hq1000/sft_hq1000_report.json"
)
SAMPLE_PATH = Path(
    "reports/milestones/003d_sft_hq1000/sft_hq1000_quality_sample25.json"
)

FINAL_SPLITS = {"train": 800, "val": 100, "test": 100}
NEW_TASK_SPLITS = {
    "person_choice": {"train": 200, "val": 25, "test": 25},
    "term_choice": {"train": 200, "val": 25, "test": 25},
    "positive_verification": {"train": 160, "val": 20, "test": 20},
    "negative_correction": {"train": 160, "val": 15, "test": 25},
}

PEOPLE = (
    "萧炎", "药老", "萧战", "薰儿", "纳兰嫣然", "云韵", "美杜莎", "小医仙",
    "海波东", "古河", "云山", "韩枫", "紫妍", "青鳞", "彩鳞", "林修崖",
    "柳擎", "苏千", "法犸", "加刑天", "雅妃", "若琳", "萧玉", "萧鼎",
    "萧厉", "夭夜", "夭月", "木辰", "纳兰桀", "纳兰肃", "凌影", "古元",
    "魂天帝", "烛坤", "净莲妖圣", "黄泉妖圣", "风尊者", "慕骨老人",
    "曹颖", "丹晨", "玄空子", "天雷子", "玄衣", "唐震", "唐火儿", "冰河",
    "天火尊者", "熊战", "雷尊者", "剑尊者",
)

TERMS_BY_CATEGORY = {
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

ALL_TERMS = tuple(
    (category, term)
    for category, terms in TERMS_BY_CATEGORY.items()
    for term in terms
)
COMPLETE_SEGMENT_RE = re.compile(r"([^。！？；…]+)[。！？；…]+")
NOISE = (
    "本书", "章节", "更新", "推荐票", "月票", "收藏", "网址", "手机用户",
    "点击", "广告", "未完待续",
    "而前", "走出现", "事吧既然", "越的", "萧炱", "强奸", "脱光", "裸体",
)
BAD_MARKS = "<>[]{}【】《》‘’“”'\"「」『』,"
LABEL_CAPS = {
    "person_choice": 10,
    "term_choice": 8,
    "positive_verification": 8,
    "negative_correction": 8,
}


@dataclass(frozen=True)
class Segment:
    evidence: str
    source_line: int


@dataclass(frozen=True)
class Draft:
    question: str
    answer: str
    evidence: str
    source_line: int
    task_type: str
    topic: str
    target_label: str


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
    return sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def clean_segments(
    corpus_lines: list[str], excluded_lines: set[int], stoi: dict[str, int]
) -> list[Segment]:
    segments: list[Segment] = []
    character_counts = Counter("".join(corpus_lines))
    for source_line, line in enumerate(corpus_lines, 1):
        if source_line in excluded_lines:
            continue
        for match in COMPLETE_SEGMENT_RE.finditer(line):
            raw_segment = match.group(1)
            evidence = raw_segment.strip()
            if not 18 <= len(evidence) <= 70:
                continue
            if any(fragment in evidence for fragment in NOISE):
                continue
            if any(mark in evidence for mark in BAD_MARKS + ".．"):
                continue
            if any(char.isascii() and char.isalpha() for char in evidence):
                continue
            if evidence.startswith(tuple("，、：；！？")) or "，，" in evidence:
                continue
            if evidence.endswith(tuple("的地得而于向将把被从与和或在")):
                continue
            chinese_count = sum("\u3400" <= char <= "\u9fff" for char in evidence)
            if chinese_count / len(evidence) < 0.8:
                continue
            if any(
                "\u3400" <= char <= "\u9fff" and character_counts[char] < 3
                for char in evidence
            ):
                continue
            if set(evidence + "。") - set(stoi):
                continue
            segments.append(Segment(evidence=evidence, source_line=source_line))
            break
    return segments


def exact_people(evidence: str) -> list[str]:
    return [person for person in PEOPLE if person in evidence]


def exact_terms(evidence: str) -> list[tuple[str, str]]:
    return [(category, term) for category, term in ALL_TERMS if term in evidence]


def distractors(
    correct: str, pool: Iterable[str], evidence: str, seed: str, count: int = 3
) -> list[str]:
    choices = [item for item in pool if item != correct and item not in evidence]
    choices.sort(key=lambda item: stable_hash(seed, item))
    if len(choices) < count:
        raise ValueError(f"not enough distractors for {correct}")
    return choices[:count]


def ordered_options(correct: str, wrong: list[str], seed: str) -> list[str]:
    options = [correct, *wrong]
    options.sort(key=lambda item: stable_hash(seed, item))
    return options


def make_draft(
    task_type: str,
    segment: Segment,
    question: str,
    answer: str,
    label: str,
) -> Draft:
    topic = f"{task_type}_{stable_hash(segment.source_line, label)[:12]}"
    return Draft(
        question=question,
        answer=answer,
        evidence=segment.evidence,
        source_line=segment.source_line,
        task_type=task_type,
        topic=topic,
        target_label=label.split("|")[0],
    )


def continuation_candidates(segments: list[Segment], stoi: dict[str, int]) -> list[Draft]:
    drafts = []
    for segment in segments:
        commas = [
            index + 1
            for index, char in enumerate(segment.evidence)
            if char in "，、" and index + 1 >= 8 and len(segment.evidence) - index - 1 >= 8
        ]
        if not commas:
            continue
        split_point = min(commas, key=lambda index: abs(index - len(segment.evidence) // 2))
        prefix = segment.evidence[:split_point]
        answer = segment.evidence[split_point:] + "。"
        question = f"请根据原文补全后半句：{prefix}"
        if set(question + answer) - set(stoi):
            continue
        drafts.append(
            make_draft(
                "grounded_continuation", segment, question, answer, segment.evidence
            )
        )
    return drafts


def person_choice_candidates(segments: list[Segment], stoi: dict[str, int]) -> list[Draft]:
    drafts = []
    for segment in segments:
        people = exact_people(segment.evidence)
        if len(people) != 1:
            continue
        correct = people[0]
        wrong = distractors(correct, PEOPLE, segment.evidence, f"person|{segment.source_line}")
        options = ordered_options(correct, wrong, f"person-order|{segment.source_line}")
        question = (
            f"根据片段，从选项中选择明确出现的人物。片段：{segment.evidence} "
            f"选项：{'、'.join(options)}"
        )
        answer = correct + "。"
        if set(question + answer) - set(stoi):
            continue
        drafts.append(make_draft("person_choice", segment, question, answer, correct))
    return drafts


def term_choice_candidates(segments: list[Segment], stoi: dict[str, int]) -> list[Draft]:
    drafts = []
    for segment in segments:
        terms = exact_terms(segment.evidence)
        if len(terms) != 1:
            continue
        category, correct = terms[0]
        wrong = distractors(
            correct,
            TERMS_BY_CATEGORY[category],
            segment.evidence,
            f"term|{segment.source_line}",
        )
        options = ordered_options(correct, wrong, f"term-order|{segment.source_line}")
        question = (
            f"根据片段，从选项中选择明确出现的{category}。片段：{segment.evidence} "
            f"选项：{'、'.join(options)}"
        )
        answer = correct + "。"
        if set(question + answer) - set(stoi):
            continue
        drafts.append(make_draft("term_choice", segment, question, answer, correct))
    return drafts


def labeled_segments(segments: list[Segment]) -> list[tuple[Segment, str, tuple[str, ...]]]:
    labeled = []
    for segment in segments:
        people = exact_people(segment.evidence)
        terms = exact_terms(segment.evidence)
        if len(people) == 1 and not terms:
            labeled.append((segment, people[0], PEOPLE))
        elif len(terms) == 1 and not people:
            category, term = terms[0]
            labeled.append((segment, term, TERMS_BY_CATEGORY[category]))
    return labeled


def verification_candidates(
    segments: list[Segment], stoi: dict[str, int], negative: bool
) -> list[Draft]:
    task_type = "negative_correction" if negative else "positive_verification"
    drafts = []
    for segment, correct, pool in labeled_segments(segments):
        if negative:
            wrong = distractors(
                correct, pool, segment.evidence, f"negative|{segment.source_line}", count=1
            )[0]
            claim = wrong
            answer = f"不一致。片段中提到的是{correct}，不是{wrong}。"
            label = f"{correct}|{wrong}"
        else:
            claim = correct
            answer = "一致。"
            label = correct
        question = (
            f"判断说法是否与片段一致。片段：{segment.evidence} "
            f"说法：片段中提到了{claim}。"
        )
        if set(question + answer) - set(stoi):
            continue
        drafts.append(make_draft(task_type, segment, question, answer, label))
    return drafts


def select_unique_lines(
    candidates: list[Draft], count: int, used_lines: set[int], task_type: str
) -> list[Draft]:
    selected = []
    label_counts: Counter[str] = Counter()
    label_cap = LABEL_CAPS.get(task_type)
    ordered = sorted(
        candidates,
        key=lambda draft: stable_hash("hq1000-v1", task_type, draft.source_line, draft.question),
    )
    for draft in ordered:
        if draft.source_line in used_lines:
            continue
        if label_cap is not None and label_counts[draft.target_label] >= label_cap:
            continue
        selected.append(draft)
        used_lines.add(draft.source_line)
        label_counts[draft.target_label] += 1
        if len(selected) == count:
            return selected
    raise ValueError(f"{task_type} only produced {len(selected)} of {count} records")


def split_schedule(task_type: str) -> list[str]:
    schedule = [
        (split, index)
        for split, count in NEW_TASK_SPLITS[task_type].items()
        for index in range(count)
    ]
    schedule.sort(key=lambda item: stable_hash("hq1000-split-v1", task_type, *item))
    return [split for split, _ in schedule]


def materialize_records(drafts_by_task: dict[str, list[Draft]]) -> list[dict[str, Any]]:
    counters = {"train": 80, "val": 15, "test": 5}
    records = []
    for task_type in NEW_TASK_SPLITS:
        drafts = drafts_by_task[task_type]
        schedule = split_schedule(task_type)
        for draft, split in zip(drafts, schedule, strict=True):
            counters[split] += 1
            records.append(
                {
                    "id": f"{split}_{counters[split]:04d}",
                    "question": draft.question,
                    "answer": draft.answer,
                    "evidence": draft.evidence,
                    "source_line": draft.source_line,
                    "topic": draft.topic,
                    "split": split,
                    "generation_method": draft.task_type,
                    "target_label": draft.target_label,
                }
            )
    return records


def validate_dataset(
    records: list[dict[str, Any]],
    corpus_lines: list[str],
    stoi: dict[str, int],
    prepared_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(records) != 1000:
        raise ValueError(f"expected 1000 records, got {len(records)}")
    ids = [record["id"] for record in records]
    questions = [record["question"] for record in records]
    topics = [record["topic"] for record in records]
    if len(set(ids)) != 1000 or len(set(questions)) != 1000:
        raise ValueError("record ids and questions must be unique")
    if len(set(topics)) != 1000:
        raise ValueError("topics must be unique")
    split_counts = Counter(record["split"] for record in records)
    if dict(split_counts) != FINAL_SPLITS:
        raise ValueError(f"unexpected split counts {dict(split_counts)}")

    line_splits: dict[int, set[str]] = defaultdict(set)
    new_line_counts: Counter[int] = Counter()
    task_split_counts: Counter[tuple[str, str]] = Counter()
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
        line_splits[line_number].add(record["split"])
        if index >= 100:
            new_line_counts[line_number] += 1
            task_type = record["generation_method"]
            target_label = record["target_label"]
            task_split_counts[(task_type, record["split"])] += 1
            if task_type == "grounded_continuation":
                prefix = record["question"].removeprefix("请根据原文补全后半句：")
                if prefix + record["answer"].removesuffix("。") != record["evidence"]:
                    raise ValueError(f"{record['id']} has an invalid continuation target")
            elif task_type in {"person_choice", "term_choice"}:
                options = record["question"].rsplit("选项：", 1)[-1].split("、")
                if len(options) != 4 or len(set(options)) != 4:
                    raise ValueError(f"{record['id']} must contain four unique options")
                if target_label not in options or target_label not in record["evidence"]:
                    raise ValueError(f"{record['id']} choice target is not grounded")
                if record["answer"] != target_label + "。":
                    raise ValueError(f"{record['id']} choice answer does not match target")
            elif task_type == "positive_verification":
                claim = record["question"].rsplit("片段中提到了", 1)[-1].removesuffix("。")
                if claim != target_label or claim not in record["evidence"]:
                    raise ValueError(f"{record['id']} positive claim is not grounded")
                if record["answer"] != "一致。":
                    raise ValueError(f"{record['id']} has an invalid positive answer")
            elif task_type == "negative_correction":
                claim = record["question"].rsplit("片段中提到了", 1)[-1].removesuffix("。")
                if target_label not in record["evidence"] or claim in record["evidence"]:
                    raise ValueError(f"{record['id']} negative correction is not grounded")
                expected_answer = (
                    f"不一致。片段中提到的是{target_label}，不是{claim}。"
                )
                if record["answer"] != expected_answer:
                    raise ValueError(f"{record['id']} has an invalid correction answer")
    if any(count != 1 for count in new_line_counts.values()):
        raise ValueError("every new record must use a unique source line")
    leaked_lines = sorted(line for line, splits in line_splits.items() if len(splits) > 1)
    if leaked_lines != [378]:
        raise ValueError(f"unexpected cross-split source lines {leaked_lines[:10]}")
    expected_task_splits = {
        (task, split): count
        for task, split_counts_for_task in NEW_TASK_SPLITS.items()
        for split, count in split_counts_for_task.items()
    }
    if dict(task_split_counts) != expected_task_splits:
        raise ValueError(f"unexpected task split matrix {dict(task_split_counts)}")

    sequence_lengths = [record["sequence_length"] for record in prepared_records]
    if max(sequence_lengths) > 256:
        raise ValueError("a sequence exceeds the model's 256-token context")
    task_counts = Counter(
        record.get("generation_method", "human_style_grounded_fact_qa")
        for record in records
    )
    task_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records[100:]:
        task_label_counts[record["generation_method"]][record["target_label"]] += 1
    return {
        "split_counts": dict(split_counts),
        "task_counts": dict(task_counts),
        "unique_question_count": len(set(questions)),
        "unique_topic_count": len(set(topics)),
        "new_unique_source_line_count": len(new_line_counts),
        "new_source_line_split_leakage": False,
        "pre_existing_source_line_overlap": [378],
        "task_unique_target_counts": {
            task: len(counts) for task, counts in task_label_counts.items()
        },
        "task_max_examples_per_target": {
            task: max(counts.values()) for task, counts in task_label_counts.items()
        },
        "task_split_counts": {
            task: {
                split: task_split_counts[(task, split)] for split in FINAL_SPLITS
            }
            for task in NEW_TASK_SPLITS
        },
        "max_sequence_length": max(sequence_lengths),
        "mean_sequence_length": sum(sequence_lengths) / len(sequence_lengths),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def build_quality_sample(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["generation_method"]].append(record)
    sample = []
    for task_type, task_records in grouped.items():
        ordered = sorted(
            task_records,
            key=lambda record: stable_hash(
                "hq1000-quality-sample-v1", task_type, record["id"]
            ),
        )
        sample.extend(ordered[:5])
    return sample


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
        excluded_lines = {record["source_line"] for record in old_records}
        segments = clean_segments(corpus_lines, excluded_lines, stoi)
        loggers["generation"].info(
            "loaded old_records=%d clean_segments=%d excluded_lines=%d",
            len(old_records),
            len(segments),
            len(excluded_lines),
        )

        pools = {
            "grounded_continuation": continuation_candidates(segments, stoi),
            "person_choice": person_choice_candidates(segments, stoi),
            "term_choice": term_choice_candidates(segments, stoi),
            "positive_verification": verification_candidates(segments, stoi, False),
            "negative_correction": verification_candidates(segments, stoi, True),
        }
        used_lines: set[int] = set()
        selected_by_task = {}
        for task_type, split_counts in NEW_TASK_SPLITS.items():
            count = sum(split_counts.values())
            selected_by_task[task_type] = select_unique_lines(
                pools[task_type], count, used_lines, task_type
            )
            loggers["generation"].info(
                "selected task=%s count=%d candidate_pool=%d",
                task_type,
                count,
                len(pools[task_type]),
            )

        new_records = materialize_records(selected_by_task)
        final_records = old_records + new_records
        special_token_ids = build_special_token_ids(int(base_payload["vocab_size"]))
        prepared_records = [
            serialize_record(record, stoi, special_token_ids) for record in final_records
        ]
        validation = validate_dataset(
            final_records, corpus_lines, stoi, prepared_records
        )
        loggers["validation"].info(
            "validated records=1000 train=800 val=100 test=100 max_length=%d",
            validation["max_sequence_length"],
        )

        write_jsonl(EXPANSION_900_PATH, new_records)
        write_jsonl(DATASET_PATH, final_records)
        quality_sample = build_quality_sample(final_records)
        SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SAMPLE_PATH.write_text(
            json.dumps(quality_sample, ensure_ascii=False, indent=2) + "\n",
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
            "dataset_version": "sft_hq1000_v1",
            "record_count": len(final_records),
            **validation,
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
            "corpus_sha256": sha256_file(CORPUS_PATH),
            "dataset_sha256": sha256_file(DATASET_PATH),
            "expansion_sha256": sha256_file(EXPANSION_900_PATH),
            "quality_sample_count": len(quality_sample),
            "quality_sample_sha256": sha256_file(SAMPLE_PATH),
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        loggers["output"].info(
            "wrote dataset=%s expansion=%s tensors=%s sample=%s report=%s",
            DATASET_PATH,
            EXPANSION_900_PATH,
            TENSOR_PATH,
            SAMPLE_PATH,
            REPORT_PATH,
        )
    except Exception:
        loggers["validation"].exception("hq1000 build failed")
        raise


if __name__ == "__main__":
    main()
