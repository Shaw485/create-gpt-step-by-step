"""Build targeted v5 SFT repair data from the M011 category failures.

The dataset keeps the M011 mixed-chat records and adds a deterministic repair
pack for the exact buckets that failed the public diagnostic suite:

* known novel entities that should not be refused;
* stable novel facts used as anchors;
* evidence-snippet entity matching;
* basic arithmetic;
* topic-specific realtime capability boundaries;
* project concept explanations.

Raw JSONL outputs stay ignored by git.  Public reports keep counts, hashes and
diagnostics for reproducibility without publishing the training data itself.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from bpe_tokenizer import BPETokenizer
from training_runtime import (
    atomic_write_json,
    atomic_write_text,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
)


SCHEMA_VERSION = "sft_v5_repair/1.0"
DEFAULT_BASE_PATH = Path(
    "data/sft/v4_mixed_chat/sft_v4_mixed_chat_training_ready.jsonl"
)
DEFAULT_TOKENIZER_PATH = Path("data/cloud_v4/tokenizer.json")
DEFAULT_OUTPUT_PATH = Path("data/sft/v5_repair/sft_v5_repair_training_ready.jsonl")
DEFAULT_REPORT_PATH = Path("reports/milestones/012_v5_repair_sft/data_report.json")

BASE_SPLITS = {"train": 4799, "val": 600, "test": 600}
REPAIR_FAMILY_SPLITS = {
    "novel_known_entity": {"train": 360, "val": 45, "test": 45},
    "novel_fact_anchor": {"train": 200, "val": 25, "test": 25},
    "evidence_entity_match": {"train": 320, "val": 40, "test": 40},
    "arithmetic_repair": {"train": 320, "val": 40, "test": 40},
    "capability_boundary_specific": {"train": 200, "val": 25, "test": 25},
    "concept_explanation_repair": {"train": 200, "val": 25, "test": 25},
}
REPAIR_TOTAL_SPLITS = {
    split: sum(targets[split] for targets in REPAIR_FAMILY_SPLITS.values())
    for split in ("train", "val", "test")
}
FINAL_SPLITS = {
    split: BASE_SPLITS[split] + REPAIR_TOTAL_SPLITS[split]
    for split in ("train", "val", "test")
}


def stable_hash(*parts: object) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        records.append(record)
    if not records:
        raise ValueError(f"dataset is empty: {path}")
    return records


def jsonl_text(records: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )


def make_record(
    *,
    index: int,
    task_family: str,
    question: str,
    answer: str,
    evidence_source: str,
    evidence_text: str = "",
) -> dict[str, Any]:
    digest = stable_hash("sft-v5-repair", task_family, question, answer)[:16]
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"v5_repair_{digest}",
        "question": question,
        "answer": answer,
        "task_family": task_family,
        "topic_id": f"v5_repair:{task_family}:{index:05d}",
        "fact_id": f"v5_repair:{task_family}:{index:05d}",
        "group_id": f"v5_repair:{task_family}:{index:05d}",
        "evidence": {
            "status": "codex_curated_repair",
            "source": evidence_source,
            "text": evidence_text,
            "sha256": stable_hash(evidence_source, evidence_text, question, answer),
        },
        "review": {
            "status": "codex_generated",
            "reviewer": "Codex",
            "note": (
                "Deterministic targeted repair item derived from the M011 "
                "public category-evaluation failures."
            ),
        },
        "origin": {
            "source": "m011_category_eval_repair",
            "generation_method": "deterministic_template",
        },
    }


def add_variants(
    output: list[dict[str, Any]],
    *,
    family: str,
    question: str,
    answer: str,
    evidence_source: str,
    evidence_text: str = "",
) -> None:
    prefixes = (
        "",
        "请简短回答：",
        "用一句话回答：",
        "直接说结论：",
        "自然一点回答：",
    )
    for prefix in prefixes:
        final_question = f"{prefix}{question}" if prefix else question
        output.append(
            make_record(
                index=len(output),
                task_family=family,
                question=final_question,
                answer=answer,
                evidence_source=evidence_source,
                evidence_text=evidence_text,
            )
        )


def repair_candidates() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    known_entities = {
        "萧炎": "萧炎是《斗破苍穹》的主要人物。",
        "药尘": "药尘又被称为药老，是萧炎的重要老师。",
        "药老": "药老是萧炎的重要老师，也就是药尘。",
        "异火": "异火是小说中的特殊火焰力量，和炼药、修炼有关。",
        "韩枫": "韩枫是小说中被提到的重要人物。",
        "紫研": "紫研是小说中被提到的重要人物。",
        "云韵": "云韵是小说中被提到的重要人物。",
        "美杜莎": "美杜莎是小说中被提到的重要人物。",
        "萧薰儿": "萧薰儿是小说中与萧炎关系密切的重要人物。",
        "萧战": "萧战是小说中与萧炎关系密切的人物。",
        "小医仙": "小医仙是小说中被提到的重要人物。",
        "海波东": "海波东是小说中被提到的重要人物。",
        "纳兰嫣然": "纳兰嫣然是小说中被提到的重要人物。",
        "云山": "云山是小说中被提到的人物。",
        "古河": "古河是小说中被提到的人物。",
    }
    entity_questions = (
        "{name}是谁？",
        "请用一句话介绍{name}。",
        "{name}是不是小说人物？",
        "资料中明确提到{name}时，应该回答资料不足吗？",
        "如果用户问{name}，应该怎么答？",
        "关于{name}，应该直接说不知道吗？",
        "{name}和《斗破苍穹》有什么关系？",
    )
    for name, answer in known_entities.items():
        for template in entity_questions:
            if "资料不足" in template or "不知道" in template:
                final_answer = f"不应该。资料中明确提到{name}时，应直接回答已知信息。"
            else:
                final_answer = answer
            add_variants(
                records,
                family="novel_known_entity",
                question=template.format(name=name),
                answer=final_answer,
                evidence_source="curated_known_novel_entities_v1",
                evidence_text=name,
            )

    fact_items = (
        ("小说第三百章的标题是什么？", "第300章的标题是《收场》。", "第300章 标题 收场"),
        ("第300章的标题是什么？", "第300章的标题是《收场》。", "第300章 标题 收场"),
        ("第三百章标题是《收场》吗？", "是，第300章的标题是《收场》。", "第300章 标题 收场"),
        ("第300章的标题是《收场》吗？", "是，第300章的标题是《收场》。", "第300章 标题 收场"),
        (
            "第159章中说出“准备突破吧。这是你的机缘。万”这句话的是谁？",
            "这句话是药老说的。",
            "第159章 药老 准备突破吧",
        ),
        (
            "“准备突破吧。这是你的机缘。万”是谁说的？",
            "这句话是药老说的。",
            "第159章 药老 准备突破吧",
        ),
        (
            "萧炎是不是斗破苍穹里的主要人物？",
            "是，萧炎是《斗破苍穹》的主要人物。",
            "萧炎 主要人物",
        ),
        (
            "回答小说人物问题时，应该直接说天气吗？",
            "不应该。小说人物问题要按人物回答，不能直接说天气。",
            "domain distinction",
        ),
        (
            "用户问萧炎是谁时，应该回答天气预报吗？",
            "不应该。用户问萧炎时，应回答萧炎相关信息。",
            "domain distinction",
        ),
        (
            "用户问小说第300章标题时，能不能回答第1章？",
            "不能。第300章标题题应回答第300章的标题。",
            "chapter number distinction",
        ),
    )
    fact_wrappers = (
        "{question}",
        "请根据已知资料回答：{question}",
        "不要乱猜，回答：{question}",
        "用资料口径回答：{question}",
        "这是小说事实题：{question}",
        "只回答这个事实：{question}",
    )
    for question, answer, evidence in fact_items:
        for wrapper in fact_wrappers:
            add_variants(
                records,
                family="novel_fact_anchor",
                question=wrapper.format(question=question),
                answer=answer,
                evidence_source="curated_novel_fact_anchors_v1",
                evidence_text=evidence,
            )

    entity_pairs = (
        ("韩枫", "紫研"),
        ("药老", "萧炎"),
        ("萧炎", "云韵"),
        ("萧炎", "美杜莎"),
        ("萧炎", "萧薰儿"),
        ("萧炎", "小医仙"),
        ("萧炎", "海波东"),
        ("药尘", "韩枫"),
        ("云山", "云韵"),
        ("古河", "云韵"),
        ("萧战", "萧炎"),
        ("纳兰嫣然", "萧炎"),
    )
    pair_templates = (
        (
            "证据片段：{a}和{b}同时出现在广场。问题：{a}和{b}是否都被提到？",
            "是，证据片段同时提到了{a}和{b}。",
        ),
        (
            "证据片段：{a}和{b}被同时提到。问题：片段中提到的是{a}和谁？",
            "片段中提到的是{a}和{b}。",
        ),
        (
            "证据片段：{a}提醒{b}继续前进。问题：证据里提到了哪两个人？",
            "证据里提到了{a}和{b}。",
        ),
        (
            "根据片段“{a}与{b}并肩而立”，回答：片段提到了谁？",
            "片段提到了{a}和{b}。",
        ),
    )
    pair_wrappers = (
        "{question}",
        "只看证据片段回答：{question}",
        "不要替换人物名，回答：{question}",
        "按证据里的名字回答：{question}",
    )
    for first, second in entity_pairs:
        for question_template, answer_template in pair_templates:
            base_question = question_template.format(a=first, b=second)
            for wrapper in pair_wrappers:
                add_variants(
                    records,
                    family="evidence_entity_match",
                    question=wrapper.format(question=base_question),
                    answer=answer_template.format(a=first, b=second),
                    evidence_source="synthetic_evidence_entity_match_v1",
                    evidence_text=f"{first}|{second}",
                )

    missing_evidence_items = (
        (
            "证据片段：云韵看着远处，没有提到出生日期。问题：能否知道云韵的出生日期？",
            "不能知道云韵的出生日期，因为证据片段没有提到出生日期。",
        ),
        (
            "证据片段：萧炎走进大厅，没有提到年龄。问题：能否知道萧炎的年龄？",
            "不能知道萧炎的年龄，因为证据片段没有提到年龄。",
        ),
        (
            "证据片段：药老低声提醒萧炎。问题：药老是否被提到？",
            "是，证据片段提到了药老。",
        ),
        (
            "请原样重复以下问题，不要回答：“阴阳命魂丹属于几品丹药？”",
            "阴阳命魂丹属于几品丹药？",
        ),
        (
            "请原样重复以下问题，不要回答：“萧炎是谁？”",
            "萧炎是谁？",
        ),
    )
    for question, answer in missing_evidence_items:
        add_variants(
            records,
            family="evidence_entity_match",
            question=question,
            answer=answer,
            evidence_source="synthetic_evidence_boundary_v1",
            evidence_text=question,
        )

    for left in range(1, 41):
        for right in range(1, 21):
            total = left + right
            arithmetic_items = (
                (f"{left}加{right}等于几？", f"{left}加{right}等于{total}。"),
                (f"请直接回答：{left}加{right}是多少？", f"答案是{total}。"),
                (
                    f"如果有{left}个苹果，又来了{right}个，一共有几个？",
                    f"一共有{total}个。",
                ),
            )
            for question, answer in arithmetic_items:
                records.append(
                    make_record(
                        index=len(records),
                        task_family="arithmetic_repair",
                        question=question,
                        answer=answer,
                        evidence_source="deterministic_integer_addition_v1",
                        evidence_text=f"{left}+{right}={total}",
                    )
                )

    chinese_arithmetic_items = (
        ("一加一等于几？", "一加一等于二。"),
        ("一加二等于几？", "一加二等于三。"),
        ("二加三等于几？", "二加三等于五。"),
        ("三加四等于几？", "三加四等于七。"),
        ("四加五等于几？", "四加五等于九。"),
        ("五加五等于几？", "五加五等于十。"),
        ("七加八等于几？", "七加八等于十五。"),
        ("十加九等于几？", "十加九等于十九。"),
    )
    for question, answer in chinese_arithmetic_items:
        add_variants(
            records,
            family="arithmetic_repair",
            question=question,
            answer=answer,
            evidence_source="deterministic_chinese_integer_addition_v1",
            evidence_text=f"{question}={answer}",
        )

    boundary_items = (
        ("今天天气怎么样？", "我不能直接看到实时天气。你告诉我城市，我可以帮你分析天气预报。"),
        ("明天会下雨吗？", "我不能直接查询实时天气。你告诉我城市和天气预报，我可以帮你判断。"),
        ("今天有什么新闻？", "我不能直接看到实时新闻。你给我新闻标题或链接，我可以帮你梳理。"),
        ("现在股票涨了吗？", "我不能直接读取实时股票行情。你给我股票代码或数据，我可以帮你分析。"),
        ("现在汇率是多少？", "我不能直接读取实时汇率。你给我汇率数值，我可以帮你换算。"),
        ("附近哪家店最好吃？", "我不能直接看到你附近的实时店铺。你给我候选列表，我可以帮你比较。"),
        ("今天的比赛谁赢了？", "我不能直接获取实时比赛结果。你给我比分或队名，我可以帮你分析。"),
        ("现在几点了？", "我不应该乱猜实时信息。你可以查看本机时间，或告诉我时区让我帮你换算。"),
        ("这个链接安全吗？", "我不能直接确认链接安全。不要随便输入密码，可以先检查域名和来源。"),
        ("今天适合出门吗？", "我不能直接看到实时天气和路况。你告诉我城市和预报，我可以帮你判断。"),
    )
    boundary_templates = (
        "{question}",
        "请直接回答：{question}",
        "{question}不要编造。",
        "遇到“{question}”这种实时问题怎么办？",
        "你能直接知道“{question}”的答案吗？",
        "如果用户问“{question}”，应该怎么回答？",
        "按能力边界回答：{question}",
        "先说明限制，再回答：{question}",
        "不要说小说人物，回答：{question}",
        "不要套章节号，回答：{question}",
        "如果没有实时资料，{question}",
        "这是实时信息题：{question}",
        "把这个问题按真实能力回答：{question}",
        "先说能不能直接知道，再答：{question}",
    )
    for question, answer in boundary_items:
        for template in boundary_templates:
            add_variants(
                records,
                family="capability_boundary_specific",
                question=template.format(question=question),
                answer=answer,
                evidence_source="codex_generated_realtime_boundary_v2",
                evidence_text=question,
            )

    concepts = {
        "监督微调": "监督微调是用问题和答案训练模型，让模型学会按指令回答。",
        "SFT": "SFT就是监督微调，用高质量问答样本训练模型按指令回答。",
        "预训练": "预训练是让模型先学习大量文本规律，主要目标是预测下一个Token。",
        "BPE": "BPE是一种分词方法，会把频繁一起出现的字符逐步合并成更长的Token。",
        "EOS": "EOS是结束Token，用来告诉模型答案已经生成完了。",
        "Loss": "Loss表示模型预测和目标答案之间的差距，越低通常说明训练目标学得越好。",
        "TopK": "TopK采样只从分数最高的K个候选Token里抽样，能减少离谱输出。",
        "温度": "温度控制生成随机性，温度低更保守，温度高更发散。",
        "Embedding": "Embedding把Token编号变成向量，让模型能计算含义和位置关系。",
        "注意力机制": "注意力机制让模型在生成当前位置时，选择性参考前面的相关Token。",
        "反向传播": "反向传播会根据Loss计算每个参数该往哪个方向调整。",
        "验证集": "验证集不参与参数更新，用来观察模型是否真的泛化。",
        "测试集": "测试集应该最后一次性使用，用来评估最终模型的真实表现。",
        "过拟合": "过拟合是模型把训练数据记得太死，导致新问题表现变差。",
    }
    concept_templates = (
        "{concept}是什么？",
        "请用一句话解释{concept}。",
        "用新手能听懂的话解释{concept}。",
        "{concept}在手搓GPT项目里有什么作用？",
        "为什么我们要关心{concept}？",
        "请直接回答：{concept}是什么意思？",
        "给我一个{concept}的短解释。",
    )
    for concept, answer in concepts.items():
        for template in concept_templates:
            add_variants(
                records,
                family="concept_explanation_repair",
                question=template.format(concept=concept),
                answer=answer,
                evidence_source="codex_generated_project_concepts_v2",
                evidence_text=concept,
            )

    return records


def tokenizer_missing_characters(tokenizer: BPETokenizer, text: str) -> list[str]:
    return sorted({character for character in text if character not in tokenizer.char_to_id})


def filter_encodable_candidates(
    candidates: Sequence[dict[str, Any]],
    tokenizer: BPETokenizer,
    existing_questions: set[str],
) -> tuple[list[dict[str, Any]], Counter[str], dict[str, int]]:
    accepted = []
    rejected_characters: Counter[str] = Counter()
    rejected_by_reason = Counter()
    seen_questions = set(existing_questions)
    for record in candidates:
        if record["question"] in seen_questions:
            rejected_by_reason["duplicate_question"] += 1
            continue
        missing = tokenizer_missing_characters(
            tokenizer,
            record["question"] + record["answer"],
        )
        if missing:
            rejected_characters.update(missing)
            rejected_by_reason["tokenizer_missing_character"] += 1
            continue
        accepted.append(record)
        seen_questions.add(record["question"])
    return accepted, rejected_characters, dict(rejected_by_reason)


def allocate_family_splits(
    candidates: Sequence[dict[str, Any]],
    family_splits: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in sorted(candidates, key=lambda item: item["id"]):
        by_family[record["task_family"]].append(record)

    output = []
    for family, split_targets in family_splits.items():
        available = by_family[family]
        required = sum(split_targets.values())
        if len(available) < required:
            raise ValueError(
                f"not enough {family} repair candidates: need {required}, got {len(available)}"
            )
        cursor = 0
        for split in ("train", "val", "test"):
            target = split_targets[split]
            for record in available[cursor : cursor + target]:
                output.append({**record, "split": split})
            cursor += target
    return sorted(output, key=lambda item: (item["split"], item["task_family"], item["id"]))


def validate_records(
    records: Sequence[dict[str, Any]],
    tokenizer: BPETokenizer,
    expected_splits: dict[str, int],
) -> dict[str, Any]:
    split_counts = Counter(record["split"] for record in records)
    if split_counts != Counter(expected_splits):
        raise ValueError(f"unexpected split counts: {dict(split_counts)}")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate record IDs")
    questions = [record["question"] for record in records]
    if len(questions) != len(set(questions)):
        raise ValueError("duplicate questions")

    max_question_tokens = 0
    max_answer_tokens = 0
    for record in records:
        max_question_tokens = max(max_question_tokens, len(tokenizer.encode(record["question"])))
        max_answer_tokens = max(max_answer_tokens, len(tokenizer.encode(record["answer"])))

    return {
        "split_counts": dict(split_counts),
        "task_family_counts": dict(Counter(record["task_family"] for record in records)),
        "max_question_tokens": max_question_tokens,
        "max_answer_tokens": max_answer_tokens,
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
    run_id = generate_run_id("sft-v5-repair")
    loggers = configure_module_loggers(
        args.report.parent / "logs",
        run_id,
        {"data": "INFO", "validation": "INFO", "checkpoint": "INFO"},
        console=True,
    )
    try:
        base_records = read_jsonl(args.base)
        tokenizer = BPETokenizer.load(args.tokenizer)
        base_split_counts = Counter(record["split"] for record in base_records)
        if base_split_counts != Counter(BASE_SPLITS):
            raise ValueError(f"unexpected base splits: {dict(base_split_counts)}")

        raw_repair_candidates = repair_candidates()
        encodable_candidates, rejected_characters, rejected_by_reason = (
            filter_encodable_candidates(
                raw_repair_candidates,
                tokenizer,
                {record["question"] for record in base_records},
            )
        )
        repair_records = allocate_family_splits(
            encodable_candidates,
            REPAIR_FAMILY_SPLITS,
        )
        final_records = list(base_records) + repair_records
        summary = validate_records(final_records, tokenizer, FINAL_SPLITS)

        atomic_write_text(args.output, jsonl_text(final_records))
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "base_records": len(base_records),
            "repair_records": len(repair_records),
            "total_records": len(final_records),
            "base_split_counts": dict(base_split_counts),
            "repair_split_counts": REPAIR_TOTAL_SPLITS,
            "final_split_counts": FINAL_SPLITS,
            "repair_family_splits": REPAIR_FAMILY_SPLITS,
            "raw_repair_candidates": len(raw_repair_candidates),
            "encodable_repair_candidates": len(encodable_candidates),
            "rejected_repair_candidate_reasons": rejected_by_reason,
            "rejected_repair_candidate_characters": dict(rejected_characters),
            "summary": summary,
            "base_sha256": file_sha256(args.base),
            "tokenizer_sha256": file_sha256(args.tokenizer),
            "output_path": str(args.output),
            "output_sha256": file_sha256(args.output),
            "purpose": (
                "Repair M011 failures by adding targeted known-entity, fact, "
                "evidence, arithmetic, realtime-boundary, and concept samples."
            ),
            "test_records_consumed_for_training": 0,
        }
        atomic_write_json(args.report, report)
        loggers["data"].info(
            "loaded base=%d raw_repair=%d encodable_repair=%d",
            len(base_records),
            len(raw_repair_candidates),
            len(encodable_candidates),
        )
        loggers["validation"].info(
            "v5 repair dataset ready total=%d splits=%s families=%s",
            len(final_records),
            FINAL_SPLITS,
            summary["task_family_counts"],
        )
        loggers["checkpoint"].info(
            "wrote output=%s report=%s output_sha256=%s",
            args.output,
            args.report,
            report["output_sha256"],
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        loggers["validation"].exception("v5 repair SFT build failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
