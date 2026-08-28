"""Build a v4 SFT dataset that mixes novel QA with general chat behavior."""

from __future__ import annotations

import argparse
from collections import Counter
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


SCHEMA_VERSION = "sft_v4_mixed_chat/1.0"
DEFAULT_NOVEL_PATH = Path(
    "data/sft/v4_teacher_repair/sft_v4_teacher_ai_training_ready.jsonl"
)
DEFAULT_TOKENIZER_PATH = Path("data/cloud_v4/tokenizer.json")
DEFAULT_OUTPUT_PATH = Path(
    "data/sft/v4_mixed_chat/sft_v4_mixed_chat_training_ready.jsonl"
)
DEFAULT_REPORT_PATH = Path(
    "reports/milestones/011_v4_mixed_chat_sft/data_report.json"
)
GENERAL_SPLIT_TARGETS = {"train": 2400, "val": 300, "test": 300}
EXPECTED_NOVEL_SPLITS = {"train": 2399, "val": 300, "test": 300}
GENERAL_FAMILY_TARGETS = {
    "general_chat": 450,
    "basic_reasoning": 400,
    "project_explanation": 450,
    "capability_boundary": 400,
    "study_planning": 350,
    "instruction_following": 350,
    "honest_unknown_general": 300,
    "domain_switching": 300,
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


def allocate_splits(
    candidates: Sequence[dict[str, Any]],
    targets: dict[str, int],
) -> list[dict[str, Any]]:
    total = sum(targets.values())
    if len(candidates) < total:
        raise ValueError(f"need {total} candidates, got {len(candidates)}")
    ordered = sorted(candidates, key=lambda record: record["id"])
    output = []
    cursor = 0
    for split in ("train", "val", "test"):
        for record in ordered[cursor : cursor + targets[split]]:
            output.append({**record, "split": split})
        cursor += targets[split]
    return output


def select_family_quotas(
    candidates: Sequence[dict[str, Any]],
    targets: dict[str, int],
) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {family: [] for family in targets}
    for record in candidates:
        family = record["task_family"]
        if family in by_family:
            by_family[family].append(record)
    output = []
    for family, target in targets.items():
        selected = sorted(by_family[family], key=lambda record: record["id"])[:target]
        if len(selected) < target:
            raise ValueError(
                f"not enough {family} candidates: need {target}, got {len(selected)}"
            )
        output.extend(selected)
    return sorted(output, key=lambda record: record["id"])


def make_record(
    *,
    index: int,
    task_family: str,
    question: str,
    answer: str,
    source: str,
) -> dict[str, Any]:
    digest = stable_hash("mixed-chat-v1", task_family, question, answer)[:16]
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"mixed_chat_{digest}",
        "question": question,
        "answer": answer,
        "task_family": task_family,
        "topic_id": f"mixed_chat:{task_family}:{index:04d}",
        "fact_id": f"mixed_chat:{task_family}:{index:04d}",
        "group_id": f"mixed_chat:{task_family}:{index:04d}",
        "evidence": {
            "status": "synthetic_instruction",
            "source": source,
            "sha256": stable_hash(source, question, answer),
        },
        "review": {
            "status": "codex_generated",
            "reviewer": "Codex",
            "note": "Synthetic general-chat instruction for the local teaching model.",
        },
        "origin": {
            "source": source,
            "generation_method": "deterministic_template",
        },
    }


def general_chat_candidates() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def add(task_family: str, question: str, answer: str) -> None:
        records.append(
            make_record(
                index=len(records),
                task_family=task_family,
                question=question,
                answer=answer,
                source="codex_generated_general_chat_v1",
            )
        )

    greetings = ["你好", "早上好", "晚上好", "嗨", "在吗", "下午好", "辛苦了"]
    for greeting in greetings:
        add("general_chat", greeting, "你好，我在。你想聊什么，或者想让我帮你做什么？")
        add("general_chat", f"{greeting}，你是谁？", "我是一个本地训练出来的迷你GPT实验模型。")
        add("general_chat", f"{greeting}，陪我聊两句。", "可以，我们慢慢聊。你现在最想说的是什么？")

    moods = ["有点累", "有点焦虑", "不太想学习", "今天状态不好", "想放弃", "有点烦", "没有动力"]
    for mood in moods:
        add("general_chat", f"我{mood}，怎么办？", "先把目标缩小，只做十分钟。能开始，比一次做完更重要。")
        add("general_chat", f"如果我{mood}，你会怎么鼓励我？", "你不用一下子变得很厉害，先完成一个小步骤就很好。")
        add("general_chat", f"我{mood}，你能陪我整理一下思路吗？", "可以。先说最困扰你的一个点，我们只处理这一件事。")

    small_talk_topics = ["学习", "写代码", "做视频", "训练模型", "读小说", "整理资料", "复盘项目"]
    for topic in small_talk_topics:
        add("general_chat", f"我想聊聊{topic}。", f"好，我们就从{topic}开始。你最关心哪一部分？")
        add("general_chat", f"你能帮我一起想想{topic}吗？", f"可以。我会先帮你把{topic}拆成几个小问题。")
        add("general_chat", f"关于{topic}，给我一个开场建议。", f"先用一句话说清楚你想通过{topic}解决什么问题。")

    for left in range(1, 31):
        for right in range(1, 11):
            add("basic_reasoning", f"{left}加{right}等于几？", f"{left}加{right}等于{left + right}。")
            add("basic_reasoning", f"请直接回答：{left}加{right}是多少？", f"答案是{left + right}。")
            add("basic_reasoning", f"如果有{left}个苹果，又来了{right}个，一共有几个？", f"一共有{left + right}个。")

    concepts = {
        "监督微调": "监督微调是用人工或教师模型写好的问题和答案，训练模型学会按指令回答。",
        "预训练": "预训练是让模型先学习大量文本里的语言规律，主要目标是预测下一个Token。",
        "Tokenizer": "Tokenizer负责把文字切成模型能处理的Token编号，再把编号还原成文字。",
        "BPE": "BPE是一种常见分词方法，会把频繁一起出现的字符逐步合并成更长的Token。",
        "Loss": "Loss表示模型预测和目标答案之间的差距，越低通常说明训练目标学得越好。",
        "EOS": "EOS是结束Token，用来告诉模型答案已经生成完了。",
        "TopK": "TopK采样只从分数最高的K个候选Token里抽样，能减少太离谱的输出。",
        "Temperature": "温度控制生成随机性，温度低更保守，温度高更发散。",
        "Embedding": "Embedding把Token编号变成向量，让模型能在连续空间里计算含义和位置关系。",
        "注意力机制": "注意力机制让模型在生成当前位置时，选择性参考前面的相关Token。",
        "反向传播": "反向传播会根据Loss计算每个参数该往哪个方向调整。",
        "梯度": "梯度表示参数变化会怎样影响Loss，是模型更新参数的依据。",
        "验证集": "验证集不参与参数更新，用来观察模型是否真的泛化。",
        "测试集": "测试集应该最后一次性使用，用来评估最终模型的真实表现。",
        "过拟合": "过拟合是模型把训练数据记得太死，导致新问题表现变差。",
    }
    concept_questions = [
        "{}是什么？",
        "请用一句话解释{}。",
        "我想学{}，先给我一个简单解释。",
        "{}在手搓GPT项目里有什么作用？",
        "为什么我们要关心{}？",
        "用新手能听懂的话解释{}。",
        "{}和训练模型有什么关系？",
    ]
    for concept, answer in concepts.items():
        for template in concept_questions:
            add("project_explanation", template.format(concept), answer)

    realtime_questions = [
        ("今天天气怎么样？", "我不能直接看到实时天气。你告诉我城市，我可以帮你分析天气预报。"),
        ("现在股票涨了吗？", "我不能直接读取实时行情。你给我具体代码或数据，我可以帮你分析。"),
        ("今天有什么新闻？", "我不能直接看到实时新闻。你给我新闻标题或链接，我可以帮你梳理。"),
        ("现在几点了？", "我不应该乱猜实时信息。你可以看本机时间，或者告诉我时区让我帮你换算。"),
        ("明天会下雨吗？", "我不能直接查询实时天气。你告诉我城市和天气预报，我可以帮你判断。"),
        ("附近哪家店最好吃？", "我不能直接看到你附近的实时店铺。你给我候选列表，我可以帮你比较。"),
        ("今天的比赛谁赢了？", "我不能直接获取实时比赛结果。你给我比分或队名，我可以帮你分析。"),
        ("现在汇率是多少？", "我不能直接读取实时汇率。你给我数值，我可以帮你换算。"),
        ("这个链接安全吗？", "我不能直接确认链接安全。不要随便输入密码，可以先检查域名和来源。"),
        ("今天适合出门吗？", "我不能直接看到实时天气和路况。你告诉我城市和预报，我可以帮你判断。"),
    ]
    for question, answer in realtime_questions:
        add("capability_boundary", question, answer)
        add("capability_boundary", f"请直接回答：{question}", answer)
        add("capability_boundary", f"如果我问“{question}”，你应该怎么回答？", answer)
        add("capability_boundary", f"{question}不要编造。", answer)
        add("capability_boundary", f"遇到“{question}”这种实时问题怎么办？", answer)
        add("capability_boundary", f"你能直接知道“{question}”的答案吗？", answer)

    study_subjects = [
        "Python",
        "机器学习",
        "GPT原理",
        "数学",
        "英语",
        "写作",
        "项目复盘",
        "数据清洗",
        "模型训练",
        "分词器",
        "注意力机制",
        "监督微调",
    ]
    for subject in study_subjects:
        add("study_planning", f"我应该如何学习{subject}？", f"先定一个小目标，再做一个练习，最后用自己的话总结{subject}。")
        add("study_planning", f"帮我安排今天的{subject}学习。", f"建议分三段：先复习概念，再动手练习，最后记录问题。")
        add("study_planning", f"学习{subject}时总是分心怎么办？", "把任务切小，关掉干扰，先专注二十五分钟，再休息五分钟。")
        add("study_planning", f"给我一个{subject}入门计划。", f"先学{subject}的核心概念，再做小练习，最后复盘哪里没懂。")
        add("study_planning", f"今天只学半小时{subject}，怎么安排？", "前十分钟复习，十五分钟练习，最后五分钟记录问题。")
        add("study_planning", f"我学{subject}卡住了。", "先找出卡住的具体点，再回到一个最小例子重新做。")
        add("study_planning", f"怎样判断{subject}有没有学会？", "能不用照抄、用自己的话解释，并独立完成一个小练习，就算开始学会了。")

    writing_tasks = [
        ("请写一句鼓励学习的话。", "慢一点没关系，只要每天推进一点，你就在靠近目标。"),
        ("请写一句适合放在视频开头的话。", "今天我们不调用大模型接口，而是一步一步手搓一个迷你GPT。"),
        ("请把“我不会”改得积极一点。", "我现在还不会，但我可以一步一步学会。"),
        ("请用一句话总结这个项目。", "这个项目用代码、数据和实验记录，从零理解GPT如何被训练出来。"),
        ("请给这个阶段起一个标题。", "从小说续写到会回答问题：第一次指令微调。"),
        ("请把“模型还很差”说得客观一点。", "当前模型已经学到部分格式，但问答准确性仍未达标。"),
        ("请写一句提醒自己不要只看Loss的话。", "Loss会告诉我们模型学得多像答案，但样本会告诉我们它是否真的好用。"),
        ("请用一句话说明为什么要做数据清洗。", "数据清洗能减少噪声，让模型学到更稳定、更可靠的模式。"),
        ("请把“继续训练”改成更具体的计划。", "下一步先固定评测样本，再增加聊天数据，并用验证Loss选择模型。"),
        ("请写一句视频结尾。", "这一步不是终点，而是我们第一次看见模型学会回答的轮廓。"),
        ("请把“先跑通流程”说得专业一点。", "先验证端到端训练链路，再逐步扩大数据、步数和模型容量。"),
        ("请把“模型乱答”说得客观一点。", "模型当前输出存在事实错误和任务理解偏差。"),
        ("请写一句提醒要看样本的话。", "指标能看趋势，样本能看真实体验，两者都不能少。"),
        ("请用一句话解释为什么要固定样本。", "固定样本能让不同训练阶段的输出变化可比较。"),
        ("请把“不要编”改成正式表达。", "在证据不足时，应明确说明无法确定，而不是生成未经验证的内容。"),
        ("请写一句适合标题的短句。", "让迷你GPT第一次学会听指令。"),
        ("请把“数据决定模型”扩写一句。", "模型会模仿训练数据中的高频模式，所以数据比例会直接影响回答风格。"),
        ("请用一句话说明为什么要做SFT。", "SFT让预训练模型从单纯续写文本，转向按照用户指令回答。"),
        ("请写一句关于过拟合的提醒。", "训练集Loss下降不代表模型更好，验证集和样本表现也要一起看。"),
        ("请把“继续”改成明确任务。", "继续构造混合聊天SFT数据，并进行一次安全试训。"),
        ("请写一句关于BPE的短解释。", "BPE把常见字符组合合并成Token，让同样长度的上下文装下更多文字。"),
        ("请把“模型还小”说得自然一点。", "这个模型规模很小，能学到一些模式，但离真正聊天还很远。"),
        ("请写一句关于实验记录的话。", "每次训练都要留下配置、指标、样本和结论，方便之后复盘。"),
        ("请用一句话解释为什么测试集不能乱用。", "测试集要留到最后评估，否则我们会不知不觉按测试集调模型。"),
        ("请写一句温和的失败总结。", "这次没有达到理想效果，但它清楚告诉我们下一步该改数据。"),
        ("请把“全是数字”改成分析表述。", "模型输出过度依赖章节号和数字模板，说明训练数据分布存在偏置。"),
        ("请写一句面向新手的解释。", "先别急着追求大模型效果，我们正在拆开每个零件看它怎么工作。"),
        ("请把“问答不行”说得更准确。", "模型具备部分问答格式能力，但语义理解和事实准确率仍不足。"),
        ("请写一句下一步建议。", "下一步应增加聊天与通用指令数据，再重新从预训练checkpoint微调。"),
        ("请用一句话解释为什么要重新训练。", "数据分布改变后，需要从同一预训练起点重新微调，才能公平比较效果。"),
    ]
    for question, answer in writing_tasks:
        add("instruction_following", question, answer)
        add("instruction_following", f"直接完成任务：{question}", answer)

    unknowns = [
        "云海真人",
        "青铜火莲",
        "九星猫皇",
        "玄冰学院院长",
        "天河药典",
        "蓝焰小队",
        "星河炼药塔",
        "白银斗帝",
        "风雪猫王",
        "南山火典",
        "青木灵舟",
        "赤云学院",
        "玄风石碑",
        "灵月长老",
        "北海药王",
    ]
    for unknown in unknowns:
        add("honest_unknown_general", f"{unknown}是谁？", "现有资料不足，无法确定。")
        add("honest_unknown_general", f"请介绍{unknown}。", "我没有足够可靠的信息，不能硬编。")
        add("honest_unknown_general", f"{unknown}第一次出现在哪一章？", "现有资料不足，无法确定其首次出现章节。")
        add("honest_unknown_general", f"如果资料里没有{unknown}，应该怎么答？", "应该说明资料不足，不能编造。")

    anchors = [
        "天气",
        "数学",
        "学习计划",
        "人工智能",
        "监督微调",
        "写作",
        "时间",
        "股票",
        "新闻",
        "安全",
        "情绪",
        "代码",
        "饭店",
        "比赛",
        "汇率",
        "链接",
        "心情",
        "计划",
        "概念解释",
    ]
    for anchor in anchors:
        add(
            "domain_switching",
            f"如果问题是关于{anchor}，你应该回答成小说章节吗？",
            f"不应该。关于{anchor}的问题要按{anchor}本身回答，不能硬套小说章节。",
        )
        add(
            "domain_switching",
            f"用户问{anchor}时，不要回答第几章，对吗？",
            "对。除非用户明确问小说章节，否则不要强行回答章节号。",
        )
        add(
            "domain_switching",
            f"{anchor}问题和小说章节问题应该怎么区分？",
            f"先看用户是否明确问小说、章节或人物；如果没有，就按{anchor}问题正常回答。",
        )
        add(
            "domain_switching",
            f"遇到{anchor}问题时，能不能硬说萧炎？",
            "不能。只有用户明确询问小说相关内容时，才应该回答小说人物或章节。",
        )

    expanded: list[dict[str, Any]] = []
    prefixes = [
        "",
        "请简短回答：",
        "用一句话回答：",
        "直接说结论：",
        "自然一点回答：",
        "像聊天一样回答：",
        "别提小说章节，回答：",
        "简单说：",
    ]
    for record in records:
        for prefix in prefixes:
            question = f"{prefix}{record['question']}" if prefix else record["question"]
            expanded.append(
                make_record(
                    index=len(expanded),
                    task_family=record["task_family"],
                    question=question,
                    answer=record["answer"],
                    source="codex_generated_general_chat_v1",
                )
            )
    return expanded


def tokenizer_missing_characters(tokenizer: BPETokenizer, text: str) -> list[str]:
    return sorted({character for character in text if character not in tokenizer.char_to_id})


def filter_encodable_candidates(
    candidates: Sequence[dict[str, Any]],
    tokenizer: BPETokenizer,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    accepted = []
    rejected_characters: Counter[str] = Counter()
    for record in candidates:
        missing = tokenizer_missing_characters(
            tokenizer,
            record["question"] + record["answer"],
        )
        if missing:
            rejected_characters.update(missing)
            continue
        accepted.append(record)
    return accepted, rejected_characters


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

    missing_characters: Counter[str] = Counter()
    max_question_tokens = 0
    max_answer_tokens = 0
    for record in records:
        for field in ("question", "answer"):
            try:
                token_count = len(tokenizer.encode(record[field]))
            except ValueError as error:
                for character in set(record[field]):
                    if character not in tokenizer.char_to_id:
                        missing_characters[character] += 1
                raise ValueError(
                    f"record {record['id']} has tokenizer coverage error"
                ) from error
            if field == "question":
                max_question_tokens = max(max_question_tokens, token_count)
            else:
                max_answer_tokens = max(max_answer_tokens, token_count)
    return {
        "split_counts": dict(split_counts),
        "task_family_counts": dict(Counter(record["task_family"] for record in records)),
        "max_question_tokens": max_question_tokens,
        "max_answer_tokens": max_answer_tokens,
        "missing_characters": dict(missing_characters),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--novel", type=Path, default=DEFAULT_NOVEL_PATH)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = generate_run_id("sft-v4-mixed-chat")
    loggers = configure_module_loggers(
        args.report.parent / "logs",
        run_id,
        {"data": "INFO", "validation": "INFO", "checkpoint": "INFO"},
        console=True,
    )
    try:
        novel_records = read_jsonl(args.novel)
        tokenizer = BPETokenizer.load(args.tokenizer)
        novel_split_counts = Counter(record["split"] for record in novel_records)
        if novel_split_counts != Counter(EXPECTED_NOVEL_SPLITS):
            raise ValueError(f"unexpected novel splits: {dict(novel_split_counts)}")
        loggers["data"].info(
            "loaded novel_records=%d tokenizer_vocab=%d",
            len(novel_records),
            tokenizer.vocab_size,
        )

        raw_general_candidates = general_chat_candidates()
        encodable_candidates, rejected_characters = filter_encodable_candidates(
            raw_general_candidates,
            tokenizer,
        )
        selected_general_candidates = select_family_quotas(
            encodable_candidates,
            GENERAL_FAMILY_TARGETS,
        )
        general_records = allocate_splits(
            selected_general_candidates,
            GENERAL_SPLIT_TARGETS,
        )
        expected_final_splits = {
            split: EXPECTED_NOVEL_SPLITS[split] + GENERAL_SPLIT_TARGETS[split]
            for split in ("train", "val", "test")
        }
        mixed_records = list(novel_records) + general_records
        summary = validate_records(mixed_records, tokenizer, expected_final_splits)

        atomic_write_text(args.output, jsonl_text(mixed_records))
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "novel_records": len(novel_records),
            "general_chat_records": len(general_records),
            "raw_general_candidates": len(raw_general_candidates),
            "encodable_general_candidates": len(encodable_candidates),
            "rejected_general_candidate_characters": dict(rejected_characters),
            "total_records": len(mixed_records),
            "novel_split_counts": dict(novel_split_counts),
            "general_split_counts": GENERAL_SPLIT_TARGETS,
            "final_split_counts": expected_final_splits,
            "summary": summary,
            "novel_sha256": file_sha256(args.novel),
            "tokenizer_sha256": file_sha256(args.tokenizer),
            "output_path": str(args.output),
            "output_sha256": file_sha256(args.output),
            "purpose": (
                "Reduce the chapter-number bias observed in M010 by adding "
                "general chat, basic reasoning, project explanation, boundary, "
                "study planning, instruction-following, honest unknown, and "
                "domain-switching examples."
            ),
        }
        atomic_write_json(args.report, report)
        loggers["validation"].info(
            "mixed chat dataset ready total=%d splits=%s families=%s",
            len(mixed_records),
            expected_final_splits,
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
        loggers["validation"].exception("mixed chat SFT build failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
