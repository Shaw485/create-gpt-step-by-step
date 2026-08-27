from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
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
EXPANSION_400_PATH = Path("data/sft/sft_expansion400_v1.jsonl")
OUTPUT_PATH = Path("data/sft/sft_pilot500_v1_tensors.pt")
REPORT_PATH = Path(
    "reports/milestones/003c_sft_data500/sft_data500_report.json"
)
EXPECTED_FINAL_SPLITS = {"train": 400, "val": 75, "test": 25}
EXPECTED_NEW_SPLITS = {"train": 320, "val": 60, "test": 20}

SEGMENT_SPLIT_RE = re.compile(r"[。！？；…]+")
PROMPT_TEMPLATES = (
    "请补全这句原文：{prefix}",
    "请续写这句原文：{prefix}",
    "原文接下来是什么：{prefix}",
    "请把后半句补上：{prefix}",
)
DOMAIN_HINTS = tuple(
    "斗丹火炎药宗院功法技兽族帝国空间灵魂血脉能量实力强者城山塔宫谷"
    "萧云纳兰古魂蛇紫薰彩鳞青鳞医仙海波东林韩枫美杜莎天妖凰太虚龙"
)
NOISY_FRAGMENTS = (
    "本书",
    "章节",
    "更新",
    "推荐票",
    "月票",
    "收藏",
    "网址",
    "手机用户",
    "点击",
    "广告",
    "未完待续",
)
FRAGMENT_PREFIXES = (
    "去，",
    "来，",
    "了，",
    "着，",
    "而",
    "但",
    "可",
    "则",
    "便",
    "完全",
    "对于",
    "所以",
    "因此",
    "若是",
    "如果",
    "只是",
    "不过",
)


@dataclass(frozen=True)
class Candidate:
    question: str
    answer: str
    evidence: str
    source_line: int
    prompt_type: str
    topic: str


def configure_generation_logging() -> dict[str, Any]:
    return {
        "generation": configure_logger(
            "sft.generation",
            Path("logs/sft_generation.log"),
            "SFT_GENERATION_LOG_LEVEL",
        ),
        "validation": configure_logger(
            "sft.validation500",
            Path("logs/sft_validation500.log"),
            "SFT_VALIDATION500_LOG_LEVEL",
        ),
        "output": configure_logger(
            "sft.output500",
            Path("logs/sft_output500.log"),
            "SFT_OUTPUT500_LOG_LEVEL",
        ),
    }


def is_clean_evidence(evidence: str) -> bool:
    if not 20 <= len(evidence) <= 46:
        return False
    if any(fragment in evidence for fragment in NOISY_FRAGMENTS):
        return False
    if any(mark in evidence for mark in "<>[]{}【】《》‘’“”'\""):
        return False
    if ".." in evidence:
        return False
    if evidence.startswith(FRAGMENT_PREFIXES):
        return False
    if evidence.endswith(tuple("的地得而于向将把被从与和或在")):
        return False
    chinese_count = sum("\u3400" <= char <= "\u9fff" for char in evidence)
    if chinese_count / len(evidence) < 0.78:
        return False
    return any(hint in evidence for hint in DOMAIN_HINTS)


def choose_split_point(evidence: str) -> int | None:
    minimum_side = 8
    middle = len(evidence) // 2
    commas = [
        index + 1
        for index, char in enumerate(evidence)
        if char in "，、"
        and index + 1 >= minimum_side
        and len(evidence) - index - 1 >= minimum_side
    ]
    if commas:
        return min(commas, key=lambda index: abs(index - middle))
    return None


def mine_candidates(
    corpus_lines: list[str], excluded_lines: set[int], stoi: dict[str, int]
) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen_questions: set[str] = set()
    for source_line, line in enumerate(corpus_lines, 1):
        if source_line in excluded_lines:
            continue
        for raw_segment in SEGMENT_SPLIT_RE.split(line):
            evidence = raw_segment.strip(" \t\r\n‘’“”'\"")
            if not is_clean_evidence(evidence):
                continue
            split_point = choose_split_point(evidence)
            if split_point is None:
                continue
            prefix = evidence[:split_point]
            answer = evidence[split_point:] + "。"
            template_index = int(
                sha256(f"template|{source_line}|{evidence}".encode("utf-8")).hexdigest(),
                16,
            ) % len(PROMPT_TEMPLATES)
            question = PROMPT_TEMPLATES[template_index].format(prefix=prefix)
            if set(question + answer) - set(stoi):
                continue
            if question in seen_questions:
                continue
            digest = sha256(
                f"{source_line}|{question}|{answer}".encode("utf-8")
            ).hexdigest()
            candidates.append(
                Candidate(
                    question=question,
                    answer=answer,
                    evidence=evidence,
                    source_line=source_line,
                    prompt_type=f"grounded_cloze_{template_index + 1}_v1",
                    topic=f"grounded_cloze_{digest[:12]}",
                )
            )
            seen_questions.add(question)
            break
    return candidates


def select_candidates(candidates: list[Candidate], count: int) -> list[Candidate]:
    selected: list[Candidate] = []
    used_lines: set[int] = set()
    prompt_counts: Counter[str] = Counter()
    prompt_cap = count // len(PROMPT_TEMPLATES)
    ordered = sorted(
        candidates,
        key=lambda item: sha256(
            f"pilot500-v2|{item.source_line}|{item.question}".encode("utf-8")
        ).hexdigest(),
    )
    for candidate in ordered:
        if candidate.source_line in used_lines:
            continue
        if prompt_counts[candidate.prompt_type] >= prompt_cap:
            continue
        selected.append(candidate)
        used_lines.add(candidate.source_line)
        prompt_counts[candidate.prompt_type] += 1
        if len(selected) == count:
            return selected
    raise ValueError(
        f"only selected {len(selected)} of {count}; prompt counts={prompt_counts}"
    )


def assign_splits(candidates: list[Candidate]) -> list[dict[str, Any]]:
    numbered_splits = [
        (split, index)
        for split, count in EXPECTED_NEW_SPLITS.items()
        for index in range(count)
    ]
    numbered_splits.sort(
        key=lambda item: sha256(
            f"pilot500-split-v1|{item[0]}|{item[1]}".encode("utf-8")
        ).hexdigest()
    )
    split_schedule = [split for split, _ in numbered_splits]
    counters = {"train": 80, "val": 15, "test": 5}
    records: list[dict[str, Any]] = []
    for candidate, split in zip(candidates, split_schedule, strict=True):
        counters[split] += 1
        records.append(
            {
                "id": f"{split}_{counters[split]:03d}",
                "question": candidate.question,
                "answer": candidate.answer,
                "evidence": candidate.evidence,
                "source_line": candidate.source_line,
                "topic": candidate.topic,
                "split": split,
                "generation_method": candidate.prompt_type,
            }
        )
    return records


def validate_final_records(
    records: list[dict[str, Any]], corpus_lines: list[str], stoi: dict[str, int]
) -> dict[str, Any]:
    ids = [record["id"] for record in records]
    questions = [record["question"] for record in records]
    topics = [record["topic"] for record in records]
    if len(records) != 500:
        raise ValueError(f"expected 500 records, got {len(records)}")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate record ids")
    if len(set(questions)) != len(questions):
        raise ValueError("duplicate questions")
    if len(set(topics)) != len(topics):
        raise ValueError("duplicate topics")
    split_counts = Counter(record["split"] for record in records)
    if dict(split_counts) != EXPECTED_FINAL_SPLITS:
        raise ValueError(f"unexpected final splits {dict(split_counts)}")

    line_splits: dict[int, set[str]] = {}
    base_line_splits: dict[int, set[str]] = {}
    new_line_splits: dict[int, set[str]] = {}
    for record_index, record in enumerate(records):
        missing_fields = REQUIRED_FIELDS - record.keys()
        if missing_fields:
            raise ValueError(f"{record.get('id')} missing {missing_fields}")
        line_number = record["source_line"]
        if not 1 <= line_number <= len(corpus_lines):
            raise ValueError(f"{record['id']} has an invalid source line")
        if record["evidence"] not in corpus_lines[line_number - 1]:
            raise ValueError(f"{record['id']} has invalid evidence")
        if set(record["question"] + record["answer"]) - set(stoi):
            raise ValueError(f"{record['id']} contains out-of-vocabulary chars")
        line_splits.setdefault(line_number, set()).add(record["split"])
        target_map = base_line_splits if record_index < 100 else new_line_splits
        target_map.setdefault(line_number, set()).add(record["split"])

    leaked_lines = sorted(line for line, splits in line_splits.items() if len(splits) > 1)
    pre_existing_leaked_lines = sorted(
        line for line, splits in base_line_splits.items() if len(splits) > 1
    )
    new_leaked_lines = sorted(
        line for line, splits in new_line_splits.items() if len(splits) > 1
    )
    if new_leaked_lines:
        raise ValueError(f"new source lines leak across splits: {new_leaked_lines[:10]}")
    if leaked_lines != pre_existing_leaked_lines:
        raise ValueError("the expansion introduced cross-split source-line leakage")
    return {
        "split_counts": dict(split_counts),
        "unique_question_count": len(set(questions)),
        "unique_topic_count": len(set(topics)),
        "pre_existing_source_line_split_leakage": pre_existing_leaked_lines,
        "new_source_line_split_leakage": False,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    loggers = configure_generation_logging()
    try:
        corpus_text = CORPUS_PATH.read_text(encoding="utf-8")
        corpus_lines = corpus_text.splitlines()
        base_payload = torch.load(BASE_TENSOR_PATH, map_location="cpu", weights_only=False)
        stoi = base_payload["stoi"]
        existing_records = load_jsonl(BASE_DATASET_PATH) + load_jsonl(EXPANSION_50_PATH)
        excluded_lines = {record["source_line"] for record in existing_records}
        loggers["generation"].info(
            "mining corpus_lines=%d existing_records=%d excluded_lines=%d",
            len(corpus_lines),
            len(existing_records),
            len(excluded_lines),
        )

        candidates = mine_candidates(corpus_lines, excluded_lines, stoi)
        selected = select_candidates(candidates, 400)
        expansion_records = assign_splits(selected)
        final_records = existing_records + expansion_records
        validation = validate_final_records(final_records, corpus_lines, stoi)
        loggers["validation"].info(
            "validated records=500 train=400 val=75 test=25 unique_topics=500"
        )

        write_jsonl(EXPANSION_400_PATH, expansion_records)
        special_token_ids = build_special_token_ids(int(base_payload["vocab_size"]))
        prepared_records = [
            serialize_record(record, stoi, special_token_ids) for record in final_records
        ]
        split_records = {
            split: [record for record in prepared_records if record["split"] == split]
            for split in EXPECTED_FINAL_SPLITS
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
            OUTPUT_PATH,
        )

        prompt_counts = Counter(
            record["generation_method"] for record in expansion_records
        )
        lengths = [record["sequence_length"] for record in prepared_records]
        report = {
            "milestone": "M003c",
            "dataset_version": "sft_pilot500_v1",
            "record_count": len(final_records),
            **validation,
            "data_composition": {
                "human_style_grounded_fact_qa": 100,
                "deterministic_grounded_cloze_instructions": 400,
            },
            "limitation": (
                "The 400 new records primarily teach instruction format and grounded "
                "completion; they are not a substitute for 400 human-reviewed fact QAs."
            ),
            "new_record_count": len(expansion_records),
            "new_split_counts": dict(
                Counter(record["split"] for record in expansion_records)
            ),
            "generation_method_counts": dict(prompt_counts),
            "evidence_verified": True,
            "base_five_test_ids_preserved": [
                record["id"] for record in final_records if record["id"].startswith("test_00")
            ][:5],
            "base_vocab_size": int(base_payload["vocab_size"]),
            "extended_vocab_size": int(base_payload["vocab_size"])
            + len(special_token_ids),
            "min_sequence_length": min(lengths),
            "max_sequence_length": max(lengths),
            "mean_sequence_length": sum(lengths) / len(lengths),
            "masked_label_count": sum(
                int((record["labels"] == -100).sum()) for record in prepared_records
            ),
            "supervised_label_count": sum(
                int((record["labels"] != -100).sum()) for record in prepared_records
            ),
            "corpus_sha256": sha256_file(CORPUS_PATH),
            "base50_sha256": sha256_file(BASE_DATASET_PATH),
            "expansion50_sha256": sha256_file(EXPANSION_50_PATH),
            "expansion400_sha256": sha256_file(EXPANSION_400_PATH),
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        loggers["output"].info(
            "wrote expansion=%s tensors=%s report=%s",
            EXPANSION_400_PATH,
            OUTPUT_PATH,
            REPORT_PATH,
        )
    except Exception:
        loggers["validation"].exception("500-record SFT build failed")
        raise


if __name__ == "__main__":
    main()
