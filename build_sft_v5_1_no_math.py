"""Build a clean v5.1 SFT dataset with no arithmetic training questions.

M012 showed three data problems: arithmetic was outside the model's intended
scope, negative meta-prompts leaked into normal chat, and prefix-only variants
made the model repeat a few answers.  This builder starts from M011, removes
those records, then adds a compact repair pack whose answers are genuinely
diverse and whose semantic groups never cross train/val/test splits.

The builder also treats the public category prompts as held-out diagnostics.
Exact, neutral-prefix, and embedded prompt variants are rejected from every
data split. Topic-level examples remain available because this public suite
measures whether the intended capability was learned, not zero-shot knowledge.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Sequence

from bpe_tokenizer import BPETokenizer
from build_sft_v5_repair import (
    filter_encodable_candidates,
    jsonl_text,
    read_jsonl,
    validate_records,
)
from evaluate_sft_v4_categories import EVAL_ITEMS
from training_runtime import (
    atomic_write_json,
    atomic_write_text,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
)


SCHEMA_VERSION = "sft_v5_1_no_math/2.1"
DEFAULT_BASE_PATH = Path(
    "data/sft/v4_mixed_chat/sft_v4_mixed_chat_training_ready.jsonl"
)
DEFAULT_TOKENIZER_PATH = Path("data/cloud_v4/tokenizer.json")
DEFAULT_OUTPUT_PATH = Path(
    "data/sft/v5_1_no_math/sft_v5_1_no_math_training_ready.jsonl"
)
DEFAULT_REPORT_PATH = Path(
    "reports/milestones/013_v5_1_no_math_sft/data_report.json"
)

EXPECTED_BASE_SPLITS = {"train": 4799, "val": 600, "test": 600}
EXCLUDED_BASE_TASK_FAMILIES = {
    "basic_reasoning",
    "continuation_rewrite_instruction",
    "domain_switching",
}
POLLUTED_PROMPT_PREFIXES = ("别提小说章节，回答：",)
NEUTRAL_QUESTION_PREFIXES = (
    "请简短回答：",
    "用一句话回答：",
    "直接说结论：",
    "自然一点回答：",
    "像聊天一样回答：",
    "简单说：",
    "请直接回答：",
    "请友好地回答：",
    "请根据已知资料回答：",
    "用资料口径回答：",
    "这是小说事实题：",
    "只回答这个事实：",
)
NUMBER_PATTERN = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+)"
ARITHMETIC_PATTERN = re.compile(
    rf"{NUMBER_PATTERN}\s*(?:加|减|乘以?|除以?|[+×÷*])\s*{NUMBER_PATTERN}"
)
AMBIGUOUS_SYMBOL_ARITHMETIC_PATTERN = re.compile(
    rf"{NUMBER_PATTERN}\s*[-/]\s*{NUMBER_PATTERN}"
)
ARITHMETIC_RESULT_CUE_PATTERN = re.compile(r"计算|等于|结果|答案|是多少|得几")
QUANTIFIED_NUMBER_PATTERN = re.compile(
    rf"{NUMBER_PATTERN}\s*(?:个|只|本|元|颗|件|支|张|枚|辆)"
)
APPLICATION_ARITHMETIC_CUE_PATTERN = re.compile(
    r"一共|总共|合计|剩下|还剩|又有|又买|再买|拿走|用掉|吃掉"
)
APPLICATION_RESULT_QUESTION_PATTERN = re.compile(
    r"多少|几个|几只|几本|几元|几颗|几件|几支|几张|几枚|几辆"
)
MATH_TOPIC_PATTERN = re.compile(r"数学|算术|加法|减法|乘法|除法")
EXPECTED_REPAIR_FAMILIES = {
    "novel_known_entity",
    "novel_fact_anchor",
    "evidence_entity_match",
    "capability_boundary_specific",
    "concept_explanation_repair",
    "natural_conversation_repair",
    "instruction_following_repair",
}


def stable_hash(*parts: object) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonicalize_question(question: str) -> str:
    """Normalize a question and repeatedly remove harmless style prefixes."""

    normalized = unicodedata.normalize("NFKC", question).strip()
    changed = True
    while changed:
        changed = False
        for prefix in NEUTRAL_QUESTION_PREFIXES:
            normalized_prefix = unicodedata.normalize("NFKC", prefix)
            if normalized.startswith(normalized_prefix):
                normalized = normalized[len(normalized_prefix) :].strip()
                changed = True
                break
    return re.sub(r"\s+", "", normalized)


HELD_OUT_EVALUATION_QUESTIONS = frozenset(item["question"] for item in EVAL_ITEMS)
HELD_OUT_CANONICAL_QUESTIONS = frozenset(
    canonicalize_question(question) for question in HELD_OUT_EVALUATION_QUESTIONS
)


def held_out_prompt_matches(question: str) -> tuple[str, ...]:
    """Return frozen prompts copied whole into a candidate question.

    This catches wrappers such as ``今天天气怎么样？不要编造`` that are not
    equal to the public prompt but still expose that prompt verbatim.
    """

    canonical = canonicalize_question(question)
    return tuple(
        sorted(
            held_out
            for held_out in HELD_OUT_CANONICAL_QUESTIONS
            if held_out in canonical
        )
    )


def is_arithmetic_text(text: str) -> bool:
    """Return whether text matches an arithmetic task shape removed in v5.1."""

    if ARITHMETIC_PATTERN.search(text):
        return True
    if (
        AMBIGUOUS_SYMBOL_ARITHMETIC_PATTERN.search(text)
        and ARITHMETIC_RESULT_CUE_PATTERN.search(text)
    ):
        return True
    return bool(
        QUANTIFIED_NUMBER_PATTERN.search(text)
        and APPLICATION_ARITHMETIC_CUE_PATTERN.search(text)
        and APPLICATION_RESULT_QUESTION_PATTERN.search(text)
    )


def is_math_topic_text(text: str) -> bool:
    """Return whether text explicitly asks about the excluded math domain."""

    return bool(MATH_TOPIC_PATTERN.search(text))


def clean_base_records(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    """Remove math, meta-task pollution, and held-out diagnostics from M011."""

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for record in records:
        question = str(record.get("question", ""))
        answer = str(record.get("answer", ""))
        family = str(record.get("task_family", ""))
        reason = ""
        if held_out_prompt_matches(question):
            reason = "held_out_evaluation_prompt_overlap"
        elif family == "basic_reasoning" or is_arithmetic_text(question + answer):
            reason = "arithmetic"
        elif is_math_topic_text(question + answer):
            reason = "math_topic"
        elif family == "domain_switching":
            reason = "domain_switching"
        elif family == "continuation_rewrite_instruction":
            reason = "continuation_rewrite_instruction"
        elif question.startswith(POLLUTED_PROMPT_PREFIXES):
            reason = "polluted_prompt_prefix"

        if reason:
            removed.append(record)
            reasons[reason] += 1
        else:
            kept.append(record)
    return kept, removed, reasons


def make_clean_record(
    *,
    task_family: str,
    group_key: str,
    question: str,
    answer: str,
    evidence_source: str,
    evidence_text: str = "",
) -> dict[str, Any]:
    digest = stable_hash("sft-v5-1-no-math", task_family, question, answer)[:16]
    group_digest = stable_hash("sft-v5-1-group", task_family, group_key)[:16]
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"v5_1_{digest}",
        "question": question,
        "answer": answer,
        "task_family": task_family,
        "topic_id": f"v5_1:{task_family}:{group_digest}",
        "fact_id": f"v5_1:{task_family}:{group_digest}",
        "group_id": f"v5_1:{task_family}:{group_digest}",
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
                "Deterministic v5.1 repair item. Math, negative domain-switch "
                "prompts, and prefix-only answer duplication are excluded."
            ),
        },
        "origin": {
            "source": "m012_category_failure_repair",
            "generation_method": "deterministic_diverse_template",
        },
    }


def novel_entity_candidates() -> list[dict[str, Any]]:
    predicates = {
        "萧炎": "是故事的主要人物",
        "药尘": "又称药老，是萧炎的重要老师",
        "药老": "就是药尘，也是萧炎的重要老师",
        "异火": "是小说中的特殊火焰力量，和炼药、修炼有关",
        "韩枫": "是小说中与药尘有关的重要人物",
        "紫研": "是小说中的重要人物",
        "云韵": "是小说中的重要人物",
        "美杜莎": "是小说中的重要人物",
        "萧薰儿": "是与萧炎关系密切的重要人物",
        "萧战": "是萧炎的父亲",
        "小医仙": "是小说中的重要人物",
        "海波东": "是小说中的重要人物",
        "纳兰嫣然": "是小说中的重要人物",
        "云山": "是云岚宗的重要人物",
        "古河": "是小说中被称为丹王的人物",
    }
    question_templates = (
        "{name}是谁或是什么？",
        "请介绍一下{name}。",
        "《斗破苍穹》中的{name}是什么？",
        "{name}和《斗破苍穹》有什么关系？",
    )
    answer_templates = (
        "{name}{predicate}。",
        "在《斗破苍穹》中，{name}{predicate}。",
        "简单说，{name}{predicate}。",
        "资料里的{name}{predicate}。",
    )
    records = []
    for name, predicate in predicates.items():
        for question_template, answer_template in zip(
            question_templates,
            answer_templates,
            strict=True,
        ):
            records.append(
                make_clean_record(
                    task_family="novel_known_entity",
                    group_key=name,
                    question=question_template.format(name=name),
                    answer=answer_template.format(name=name, predicate=predicate),
                    evidence_source="curated_known_novel_entities_v2",
                    evidence_text=name,
                )
            )
    return records


def novel_fact_candidates() -> list[dict[str, Any]]:
    facts = (
        (
            "chapter_300_title",
            (
                "第三百章叫什么名字？",
                "请告诉我第300章的章名。",
                "《收场》对应小说中的哪一章？",
                "小说第300章是否叫《收场》？",
            ),
            (
                "第三百章的标题是《收场》。",
                "第300章的章名为《收场》。",
                "《收场》对应的是第300章。",
                "是的，小说第300章叫《收场》。",
            ),
            "第300章 标题 收场",
        ),
        (
            "chapter_159_speaker",
            (
                "第159章里提醒萧炎准备突破的人是谁？",
                "谁在第159章告诉萧炎准备突破？",
                "第159章的突破提醒来自哪个人物？",
                "药老是否在第159章提醒萧炎准备突破？",
            ),
            (
                "提醒萧炎准备突破的人是药老。",
                "第159章里，这句话由药老说出。",
                "这段突破提醒来自药老。",
                "是，药老在第159章给出了突破提醒。",
            ),
            "第159章 药老 准备突破",
        ),
        (
            "xiaoyan_main_character",
            (
                "故事主要围绕哪个人物展开？",
                "《斗破苍穹》的核心人物是谁？",
                "萧炎在故事中是不是主要人物？",
                "请指出小说的主要人物。",
            ),
            (
                "故事主要围绕萧炎展开。",
                "《斗破苍穹》的核心人物是萧炎。",
                "是，萧炎是故事的主要人物。",
                "小说的主要人物是萧炎。",
            ),
            "萧炎 主要人物",
        ),
        (
            "yaochen_alias",
            (
                "药老的本名是什么？",
                "药尘还有什么常用称呼？",
                "药老和药尘是不是同一个人物？",
                "小说里谁被称为药老？",
            ),
            (
                "药老的本名是药尘。",
                "药尘常被称为药老。",
                "是，药老和药尘指同一个人物。",
                "药尘被称为药老。",
            ),
            "药尘 药老",
        ),
    )
    records = []
    for group_key, questions, answers, evidence in facts:
        for question, answer in zip(questions, answers, strict=True):
            records.append(
                make_clean_record(
                    task_family="novel_fact_anchor",
                    group_key=group_key,
                    question=question,
                    answer=answer,
                    evidence_source="curated_novel_fact_anchors_v2",
                    evidence_text=evidence,
                )
            )
    return records


def evidence_entity_candidates() -> list[dict[str, Any]]:
    pairs = (
        ("韩枫", "紫研"),
        ("药老", "萧炎"),
        ("萧炎", "云韵"),
        ("美杜莎", "萧炎"),
        ("萧薰儿", "萧炎"),
        ("小医仙", "萧炎"),
        ("海波东", "萧炎"),
        ("药尘", "韩枫"),
        ("云山", "云韵"),
        ("古河", "云韵"),
        ("萧战", "萧炎"),
        ("纳兰嫣然", "萧炎"),
        ("紫研", "美杜莎"),
        ("云韵", "纳兰嫣然"),
        ("萧炎", "药老"),
        ("韩枫", "药尘"),
        ("萧战", "萧薰儿"),
        ("海波东", "美杜莎"),
        ("小医仙", "紫研"),
        ("古河", "云山"),
    )
    context_templates = (
        "{a}和{b}同时走进大厅。",
        "{a}提醒{b}继续前进。",
        "{a}与{b}并肩而立。",
        "{a}看见{b}走来。",
        "{a}和{b}都出现在片段里。",
    )
    question_answer_templates = (
        ("片段：{context}问题：{a}和谁一起出现？", "和{a}一起出现的是{b}。"),
        ("片段：{context}问题：第二个人是谁？", "第二个人是{b}，第一个人是{a}。"),
        ("片段：{context}问题：除了{a}还提到谁？", "除了{a}，片段还提到{b}。"),
        ("片段：{context}问题：片段中的两个人是谁？", "片段中的两个人是{a}和{b}。"),
    )
    records = []
    for index, (first, second) in enumerate(pairs):
        context = context_templates[index % len(context_templates)].format(
            a=first,
            b=second,
        )
        group_key = f"{first}|{second}"
        for question_template, answer_template in question_answer_templates:
            records.append(
                make_clean_record(
                    task_family="evidence_entity_match",
                    group_key=group_key,
                    question=question_template.format(
                        context=context,
                        a=first,
                        b=second,
                    ),
                    answer=answer_template.format(a=first, b=second),
                    evidence_source="synthetic_balanced_entity_extraction_v2",
                    evidence_text=context,
                )
            )
    return records


def capability_boundary_candidates() -> list[dict[str, Any]]:
    items = (
        ("天气", "今天天气怎么样？", "实时天气", "城市和天气预报"),
        ("降雨", "明天会下雨吗？", "实时降雨预报", "城市和预报"),
        ("新闻", "今天有什么新闻？", "实时新闻", "新闻标题或链接"),
        ("股票", "现在股票涨了吗？", "实时股票行情", "股票代码或行情数据"),
        ("汇率", "现在汇率是多少？", "实时汇率", "具体汇率数值"),
        ("店铺", "附近哪家店最好吃？", "附近店铺信息", "地点和候选店铺"),
        ("比赛", "今天的比赛谁赢了？", "实时比赛结果", "队名或比分"),
        ("时间", "现在几点了？", "本机当前时间", "时区或本机时间"),
        ("链接", "这个链接安全吗？", "链接的真实安全状态", "域名和来源信息"),
        ("出门", "今天适合出门吗？", "实时天气和路况", "城市、预报和路况"),
    )
    question_templates = (
        "{question}",
        "你能直接知道“{question}”的答案吗？",
        "遇到“{question}”这种实时问题怎么办？",
        "如果我问“{question}”，你会怎么处理？",
        "没有联网资料时怎样回答“{question}”",
    )
    answer_templates = (
        "我不能直接获取{limit}。你提供{need}后，我可以继续分析。",
        "这需要{limit}，我目前不能直接确认；可以先给我{need}。",
        "我无法凭空知道{limit}，但能根据你提供的{need}帮你判断。",
        "先说明限制：我看不到{limit}。如果有{need}，我可以协助比较。",
        "关于{topic}，我不能猜测{limit}；请提供{need}再做分析。",
    )
    records = []
    for topic, question, limit, need in items:
        for question_template, answer_template in zip(
            question_templates,
            answer_templates,
            strict=True,
        ):
            records.append(
                make_clean_record(
                    task_family="capability_boundary_specific",
                    group_key=topic,
                    question=question_template.format(question=question),
                    answer=answer_template.format(topic=topic, limit=limit, need=need),
                    evidence_source="codex_curated_realtime_boundary_v3",
                    evidence_text=topic,
                )
            )
    return records


def concept_candidates() -> list[dict[str, Any]]:
    concepts = {
        "监督微调": "用高质量问题和答案继续训练模型，让模型学习按指令回答",
        "SFT": "监督微调的英文缩写，目标是让模型学习指令和标准答案之间的关系",
        "预训练": "先用大量文本训练下一个Token预测，让模型学习语言规律",
        "BPE": "把频繁相邻的字符逐步合并成更长Token的分词方法",
        "EOS": "表示一段答案已经结束的特殊Token",
        "Loss": "衡量模型预测与目标答案差距的数值",
        "TopK": "生成时只保留分数最高的K个候选Token再采样",
        "温度": "控制生成随机性的参数，越低通常越保守",
        "Embedding": "把Token编号映射成可计算向量的表示层",
        "注意力机制": "让当前位置选择性参考前文相关Token的计算方法",
        "反向传播": "根据Loss计算梯度并指导参数更新的过程",
        "验证集": "不参与参数更新、用于观察模型泛化表现的数据",
        "过拟合": "模型过度记住训练数据而在新问题上表现变差的现象",
    }
    question_templates = (
        "{concept}是什么？",
        "请给新手解释{concept}。",
        "{concept}在这个项目里有什么作用？",
        "为什么训练模型时要关心{concept}？",
        "用一个简短例子说明{concept}。",
    )
    answer_templates = (
        "{concept}是{definition}。",
        "给新手的解释是：{concept}就是{definition}。",
        "在这个项目里，{concept}用于{definition}。",
        "需要关心{concept}，因为它关系到{definition}。",
        "可以把{concept}理解为{definition}。",
    )
    records = []
    for concept, definition in concepts.items():
        for question_template, answer_template in zip(
            question_templates,
            answer_templates,
            strict=True,
        ):
            records.append(
                make_clean_record(
                    task_family="concept_explanation_repair",
                    group_key=concept,
                    question=question_template.format(concept=concept),
                    answer=answer_template.format(
                        concept=concept,
                        definition=definition,
                    ),
                    evidence_source="codex_curated_project_concepts_v3",
                    evidence_text=concept,
                )
            )
    return records


def natural_conversation_candidates() -> list[dict[str, Any]]:
    subjects = (
        "Python",
        "英语",
        "写作",
        "阅读",
        "编程",
        "历史",
        "手搓GPT项目",
        "数据清洗",
        "模型训练",
        "代码调试",
    )
    challenges = (
        (
            "学习{subject}时总是分心怎么办？",
            "学习{subject}时可以先关掉干扰，只安排一小段专注时间，完成后再休息。",
        ),
        (
            "{subject}内容太多，我该怎么拆分？",
            "把{subject}列成几个小主题，每次只完成一个能检查结果的小任务。",
        ),
        (
            "学{subject}卡住了怎么办？",
            "先写清楚{subject}卡住的位置和预期结果，再做一个最小例子逐步检查。",
        ),
        (
            "学完{subject}很快就忘了怎么办？",
            "学完{subject}后合上资料主动复述，再隔一天做一次短复习。",
        ),
        (
            "我不知道从哪里开始学{subject}。",
            "先为{subject}定一个今天能完成的小目标，做完后再根据结果安排下一步。",
        ),
        (
            "今天怎么安排{subject}学习？",
            "先学习{subject}的一个核心内容，再练习一个例子，最后记录问题和收获。",
        ),
        (
            "学习{subject}时怎么做笔记？",
            "{subject}笔记可以只保留问题、结论、例子和暂时没理解的地方。",
        ),
        (
            "最近没有动力继续{subject}怎么办？",
            "把{subject}目标缩小到马上能完成的一步，先获得一次明确进展。",
        ),
        (
            "如何检查我的{subject}有没有进步？",
            "定期完成同一类{subject}小任务，并比较正确率、速度和独立完成程度。",
        ),
        (
            "感觉{subject}进度很慢，要不要放弃？",
            "先回看最近完成的{subject}任务并调整难度，稳定的小进步更可靠。",
        ),
    )
    records = []
    for subject in subjects:
        for index, (question_template, answer_template) in enumerate(challenges):
            records.append(
                make_clean_record(
                    task_family="natural_conversation_repair",
                    group_key=f"{subject}|{index}",
                    question=question_template.format(subject=subject),
                    answer=answer_template.format(subject=subject),
                    evidence_source="codex_curated_natural_conversation_v2",
                    evidence_text=subject,
                )
            )
    return records


def instruction_following_candidates() -> list[dict[str, Any]]:
    phrases = (
        "明白",
        "开始训练",
        "保存模型",
        "检查数据",
        "记录结果",
        "继续学习",
        "读取文件",
        "运行测试",
        "分析样本",
        "完成复盘",
        "准备好了",
        "先做验证",
        "查看日志",
        "缩小问题",
        "保持专注",
        "逐步检查",
        "整理思路",
        "确认输入",
        "核对输出",
        "修复错误",
    )
    records = []
    for phrase in phrases:
        for question in (
            f"请只回复“{phrase}”。",
            f"请原样输出这几个字：{phrase}",
        ):
            records.append(
                make_clean_record(
                    task_family="instruction_following_repair",
                    group_key=f"exact|{phrase}",
                    question=question,
                    answer=f"{phrase}。",
                    evidence_source="codex_curated_positive_instruction_v2",
                    evidence_text=phrase,
                )
            )

    yes_no_items = (
        ("验证集会直接更新模型参数吗？", "否。"),
        ("训练集会用于计算训练Loss吗？", "是。"),
        ("最终评估前应该反复查看测试集吗？", "否。"),
        ("EOS可以表示答案结束吗？", "是。"),
        ("日志里应该记录密码吗？", "否。"),
        ("修改数据后应该重新核验吗？", "是。"),
        ("模型参数会通过反向传播更新吗？", "是。"),
        ("验证Loss低就一定代表聊天好吗？", "否。"),
        ("训练前应该检查数据格式吗？", "是。"),
        ("遇到错误时应该先看报错信息吗？", "是。"),
    )
    for index, (question, answer) in enumerate(yes_no_items):
        records.append(
            make_clean_record(
                task_family="instruction_following_repair",
                group_key=f"yes_no|{index}",
                question=f"请只回答是或否：{question}",
                answer=answer,
                evidence_source="codex_curated_positive_instruction_v2",
                evidence_text=question,
            )
        )
    return records


def repair_candidates() -> list[dict[str, Any]]:
    candidates = (
        novel_entity_candidates()
        + novel_fact_candidates()
        + evidence_entity_candidates()
        + capability_boundary_candidates()
        + concept_candidates()
        + natural_conversation_candidates()
        + instruction_following_candidates()
    )
    return [
        record
        for record in candidates
        if not held_out_prompt_matches(record["question"])
    ]


def allocate_grouped_splits(
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign whole semantic groups to one split using stable ordering."""

    by_family: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in candidates:
        by_family[record["task_family"]][record["group_id"]].append(record)

    output = []
    for family in sorted(by_family):
        groups = sorted(
            by_family[family].items(),
            key=lambda item: stable_hash("split", family, item[0]),
        )
        group_count = len(groups)
        if group_count < 3:
            raise ValueError(f"need at least three semantic groups for {family}")
        val_groups = max(1, round(group_count * 0.10))
        test_groups = max(1, round(group_count * 0.10))
        train_groups = group_count - val_groups - test_groups
        if train_groups <= 0:
            raise ValueError(f"not enough train groups for {family}")
        split_names = (
            ["train"] * train_groups
            + ["val"] * val_groups
            + ["test"] * test_groups
        )
        for (_, records), split in zip(groups, split_names, strict=True):
            output.extend({**record, "split": split} for record in records)
    return sorted(output, key=lambda item: (item["split"], item["task_family"], item["id"]))


def filter_canonical_duplicates(
    candidates: Sequence[dict[str, Any]],
    forbidden_questions: set[str] | frozenset[str],
) -> tuple[list[dict[str, Any]], int]:
    """Reject normalized duplicates against base/eval and within candidates."""

    accepted = []
    seen = set(forbidden_questions)
    rejected = 0
    for record in candidates:
        canonical = canonicalize_question(record["question"])
        if canonical in seen:
            rejected += 1
            continue
        seen.add(canonical)
        accepted.append(record)
    return accepted, rejected


def validate_repair_quality(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Enforce split isolation and answer-diversity gates for repair data."""

    families = {record["task_family"] for record in records}
    if families != EXPECTED_REPAIR_FAMILIES:
        raise ValueError(f"unexpected repair families: {sorted(families)}")

    group_splits: dict[str, set[str]] = defaultdict(set)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group_splits[record["group_id"]].add(record["split"])
        by_family[record["task_family"]].append(record)
    leaking_groups = {
        group: sorted(splits)
        for group, splits in group_splits.items()
        if len(splits) != 1
    }
    if leaking_groups:
        raise ValueError(f"semantic groups cross splits: {leaking_groups}")

    family_quality = {}
    for family, family_records in sorted(by_family.items()):
        answer_counts = Counter(record["answer"] for record in family_records)
        unique_ratio = len(answer_counts) / len(family_records)
        minimum_ratio = 0.30 if family == "instruction_following_repair" else 0.75
        if unique_ratio < minimum_ratio:
            raise ValueError(
                f"answer diversity too low for {family}: {unique_ratio:.3f}"
            )
        family_quality[family] = {
            "records": len(family_records),
            "semantic_groups": len({record["group_id"] for record in family_records}),
            "unique_answers": len(answer_counts),
            "unique_answer_ratio": unique_ratio,
            "max_exact_answer_repeat": max(answer_counts.values()),
            "split_counts": dict(Counter(record["split"] for record in family_records)),
        }
    return {
        "semantic_group_leaks": 0,
        "families": family_quality,
    }


def assert_no_math_or_pollution(records: Iterable[dict[str, Any]]) -> None:
    """Fail closed if excluded task shapes or diagnostics re-enter data."""

    for record in records:
        family = str(record.get("task_family", ""))
        question = str(record.get("question", ""))
        answer = str(record.get("answer", ""))
        if family in EXCLUDED_BASE_TASK_FAMILIES | {"arithmetic_repair"}:
            raise ValueError(f"excluded task family returned: {family}")
        if is_arithmetic_text(question + answer):
            raise ValueError(f"arithmetic record returned: {record.get('id')}")
        if is_math_topic_text(question + answer):
            raise ValueError(f"math-topic record returned: {record.get('id')}")
        if question.startswith(POLLUTED_PROMPT_PREFIXES):
            raise ValueError(f"polluted prompt prefix returned: {record.get('id')}")
        matches = held_out_prompt_matches(question)
        if matches:
            raise ValueError(
                "held-out evaluation prompt returned: "
                f"{record.get('id')} matches={matches}"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = generate_run_id("sft-v5-1-no-math")
    loggers = configure_module_loggers(
        args.report.parent / "logs",
        run_id,
        {"data": "INFO", "validation": "INFO", "checkpoint": "INFO"},
        console=True,
    )
    try:
        base_records = read_jsonl(args.base)
        base_split_counts = Counter(record["split"] for record in base_records)
        if base_split_counts != Counter(EXPECTED_BASE_SPLITS):
            raise ValueError(f"unexpected base splits: {dict(base_split_counts)}")

        clean_base, removed_base, removal_reasons = clean_base_records(base_records)
        clean_base_splits = Counter(record["split"] for record in clean_base)
        clean_base_canonical_questions = {
            canonicalize_question(record["question"]) for record in clean_base
        }
        tokenizer = BPETokenizer.load(args.tokenizer)
        raw_candidates = repair_candidates()
        encodable_candidates, rejected_characters, rejected_by_reason = (
            filter_encodable_candidates(
                raw_candidates,
                tokenizer,
                {record["question"] for record in clean_base}
                | set(HELD_OUT_EVALUATION_QUESTIONS),
            )
        )
        encodable_candidates, canonical_duplicate_count = filter_canonical_duplicates(
            encodable_candidates,
            clean_base_canonical_questions | set(HELD_OUT_CANONICAL_QUESTIONS),
        )
        rejected_by_reason = dict(rejected_by_reason)
        rejected_by_reason["canonical_duplicate_question"] = canonical_duplicate_count
        repair_records = allocate_grouped_splits(encodable_candidates)
        repair_quality = validate_repair_quality(repair_records)
        final_records = list(clean_base) + repair_records
        assert_no_math_or_pollution(final_records)
        final_splits = dict(Counter(record["split"] for record in final_records))
        summary = validate_records(final_records, tokenizer, final_splits)

        atomic_write_text(args.output, jsonl_text(final_records))
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "base_records_before_cleaning": len(base_records),
            "base_records_after_cleaning": len(clean_base),
            "removed_base_records": len(removed_base),
            "removed_base_reasons": dict(removal_reasons),
            "removed_base_split_counts": dict(
                Counter(record["split"] for record in removed_base)
            ),
            "clean_base_split_counts": dict(clean_base_splits),
            "repair_records": len(repair_records),
            "repair_split_counts": dict(
                Counter(record["split"] for record in repair_records)
            ),
            "repair_quality": repair_quality,
            "total_records": len(final_records),
            "final_split_counts": final_splits,
            "raw_repair_candidates": len(raw_candidates),
            "encodable_repair_candidates": len(encodable_candidates),
            "rejected_repair_candidate_reasons": rejected_by_reason,
            "rejected_repair_candidate_characters": dict(rejected_characters),
            "summary": summary,
            "base_sha256": file_sha256(args.base),
            "tokenizer_sha256": file_sha256(args.tokenizer),
            "output_path": str(args.output),
            "output_sha256": file_sha256(args.output),
            "arithmetic_records": 0,
            "math_topic_records": 0,
            "domain_switching_records": 0,
            "continuation_rewrite_instruction_records": 0,
            "polluted_prompt_prefix_records": 0,
            "held_out_evaluation_prompt_overlap_records": 0,
            "test_records_consumed_for_training": 0,
            "purpose": (
                "Remove math and meta-prompt pollution, isolate diagnostics, "
                "and add diverse grouped repairs for useful chat behavior."
            ),
        }
        atomic_write_json(args.report, report)
        loggers["data"].info(
            "cleaned base before=%d after=%d removed=%d reasons=%s",
            len(base_records),
            len(clean_base),
            len(removed_base),
            dict(removal_reasons),
        )
        loggers["validation"].info(
            "validated no_math=true eval_prompt_overlap=0 group_leaks=0 total=%d splits=%s",
            len(final_records),
            final_splits,
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
        loggers["validation"].exception("v5.1 no-math SFT build failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
