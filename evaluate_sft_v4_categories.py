"""Evaluate the active no-math SFT checkpoint with a fixed diagnostic suite."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any, Sequence

import torch

from bpe_tokenizer import BPETokenizer
from sample_sft_v4_custom import (
    build_prompt_ids,
    markdown_escape,
    validate_checkpoint_payload_compatibility,
)
from train_pretrain_v4 import load_config
from train_sft_v4 import build_model, generate_answer, load_model_checkpoint, select_device
from training_runtime import (
    atomic_write_json,
    atomic_write_text,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
)


DEFAULT_CONFIG_PATH = Path("configs/local_m4_8m_continue_6000.json")
DEFAULT_DATA_PATH = Path("data/cloud_v4/sft_v5_1_no_math_tensors.pt")
DEFAULT_CHECKPOINT_PATH = Path(
    "runs/sft_v5_1_no_math_continue3000_cumulative5000/latest.pt"
)
DEFAULT_OUTPUT_JSON = Path(
    "reports/milestones/013_v5_1_no_math_sft/category_eval_cumulative5000_v2_greedy.json"
)
DEFAULT_OUTPUT_MD = Path(
    "reports/milestones/013_v5_1_no_math_sft/category_eval_cumulative5000_v2_greedy.md"
)
DEFAULT_MAX_NEW_TOKENS = 30
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_K = 1
DEFAULT_SEED = 20260828

NOVEL_LEAK_MARKERS = [
    "第",
    "章",
    "萧炎",
    "药老",
    "药尘",
    "韩枫",
    "紫研",
    "斗气",
    "异火",
]
UNKNOWN_REFUSAL_MARKERS = ["资料不足", "无法确定", "不能硬编", "没有足够"]
BOUNDARY_MARKERS = ["不能", "无法", "不应该", "不要乱猜"]
REALTIME_MARKERS = ["实时", "天气", "新闻", "行情", "汇率", "附近", "比赛", "时间"]

EVAL_ITEMS = [
    {
        "id": "novel_entity_xiaoyan",
        "category": "小说人物",
        "question": "萧炎是谁？",
        "metric": "known_entity",
        "required_any": ["萧炎"],
        "required_context_any": ["主要", "主角", "核心"],
        "forbidden_any": UNKNOWN_REFUSAL_MARKERS,
    },
    {
        "id": "novel_entity_yaochen",
        "category": "小说人物",
        "question": "药尘是谁？",
        "metric": "known_entity",
        "required_any": ["药尘", "药老"],
        "required_context_any": ["药老", "老师"],
        "forbidden_any": UNKNOWN_REFUSAL_MARKERS,
    },
    {
        "id": "novel_entity_yaolao",
        "category": "小说人物",
        "question": "药老是谁？",
        "metric": "known_entity",
        "required_any": ["药老", "药尘"],
        "required_context_any": ["药尘", "老师"],
        "forbidden_any": UNKNOWN_REFUSAL_MARKERS,
    },
    {
        "id": "novel_entity_xiaoyan_one_sentence",
        "category": "小说人物",
        "question": "请用一句话介绍萧炎。",
        "metric": "known_entity",
        "required_any": ["萧炎"],
        "required_context_any": ["主要", "主角", "核心"],
        "forbidden_any": UNKNOWN_REFUSAL_MARKERS + ["天气", "新闻"],
    },
    {
        "id": "novel_entity_yihuo",
        "category": "小说人物",
        "question": "异火是什么？",
        "metric": "known_entity",
        "required_any": ["异火"],
        "required_context_any": ["火焰", "力量", "炼药"],
        "forbidden_any": UNKNOWN_REFUSAL_MARKERS,
    },
    {
        "id": "novel_fact_chapter300",
        "category": "小说事实",
        "question": "小说第三百章的标题是什么？",
        "metric": "all_required",
        "required_all": ["第300章", "收场"],
        "forbidden_any": ["资料不足", "无法确定"],
    },
    {
        "id": "novel_fact_speaker_yaolao",
        "category": "小说事实",
        "question": "第159章中说出“准备突破吧。这是你的机缘。万”这句话的是谁？",
        "metric": "exact_any",
        "required_any": ["药老"],
        "forbidden_any": ["萧炎", "资料不足"],
    },
    {
        "id": "novel_fact_title_direct",
        "category": "小说事实",
        "question": "第300章的标题是《收场》吗？",
        "metric": "all_required",
        "required_all": ["是", "第300章", "收场"],
        "forbidden_any": ["不是", "无法确定"],
    },
    {
        "id": "novel_fact_no_unknown_for_known",
        "category": "小说事实",
        "question": "萧炎是不是斗破苍穹里的主要人物？",
        "metric": "all_required",
        "required_all": ["是", "萧炎"],
        "forbidden_any": UNKNOWN_REFUSAL_MARKERS,
    },
    {
        "id": "novel_fact_not_weather",
        "category": "小说事实",
        "question": "回答小说人物问题时，应该直接说天气吗？",
        "metric": "exact_any",
        "required_any": ["不应该", "不能", "不应"],
        "forbidden_any": ["天气预报"],
    },
    {
        "id": "evidence_hanfeng_ziyan",
        "category": "证据判断",
        "question": "证据片段：韩枫和紫研同时出现在广场。问题：韩枫和紫研是否都被提到？",
        "metric": "all_required",
        "required_all": ["是", "韩枫", "紫研"],
        "forbidden_any": ["苏千", "萧炎", "无法确定"],
    },
    {
        "id": "evidence_yaolao_only",
        "category": "证据判断",
        "question": "证据片段：药老低声提醒萧炎。问题：药老是否被提到？",
        "metric": "all_required",
        "required_all": ["是", "药老"],
        "forbidden_any": ["没有", "无法确定"],
    },
    {
        "id": "evidence_no_birthdate",
        "category": "证据判断",
        "question": "证据片段：云韵看着远处，没有提到出生日期。问题：能否知道云韵的出生日期？",
        "metric": "all_required",
        "required_all": ["不能", "出生日期"],
        "forbidden_any": ["能确认"],
    },
    {
        "id": "evidence_copy_question",
        "category": "证据判断",
        "question": "请原样重复以下问题，不要回答：“阴阳命魂丹属于几品丹药？”",
        "metric": "all_required",
        "required_all": ["阴阳命魂丹", "几品丹药"],
        "forbidden_any": ["第", "章"],
    },
    {
        "id": "evidence_two_names_no_extra",
        "category": "证据判断",
        "question": "证据片段：韩枫和紫研被同时提到。问题：片段中提到的是韩枫和谁？",
        "metric": "all_required",
        "required_all": ["紫研"],
        "forbidden_any": ["苏千", "萧炎"],
    },
    {
        "id": "chat_weather",
        "category": "能力边界",
        "question": "今天天气怎么样？",
        "metric": "boundary",
        "topic_required_any": ["天气", "预报"],
    },
    {
        "id": "chat_news",
        "category": "能力边界",
        "question": "今天有什么新闻？",
        "metric": "boundary",
        "topic_required_any": ["新闻"],
    },
    {
        "id": "chat_stock",
        "category": "能力边界",
        "question": "现在股票涨了吗？",
        "metric": "boundary",
        "topic_required_any": ["股票", "行情"],
    },
    {
        "id": "chat_exchange_rate",
        "category": "能力边界",
        "question": "现在汇率是多少？",
        "metric": "boundary",
        "topic_required_any": ["汇率"],
    },
    {
        "id": "chat_restaurant",
        "category": "能力边界",
        "question": "附近哪家店最好吃？",
        "metric": "boundary",
        "topic_required_any": ["附近", "店", "候选"],
    },
    {
        "id": "instruction_ack_exact",
        "category": "指令遵循",
        "question": "请只回答“收到”。",
        "metric": "exact_response",
        "accepted_responses": ["收到", "收到。"],
    },
    {
        "id": "instruction_repeat_exact",
        "category": "指令遵循",
        "question": "请原样重复这四个字：“训练模型”。",
        "metric": "exact_response",
        "accepted_responses": ["训练模型", "训练模型。"],
    },
    {
        "id": "instruction_two_advice",
        "category": "指令遵循",
        "question": "请给出两条提高学习专注力的建议。",
        "metric": "two_advice",
        "required_any": ["专注", "学习", "干扰", "休息"],
    },
    {
        "id": "instruction_test_not_train",
        "category": "指令遵循",
        "question": "请只回答“是”或“否”：测试集应该参与参数更新吗？",
        "metric": "exact_response",
        "accepted_responses": ["否", "否。"],
    },
    {
        "id": "instruction_validation_one_sentence",
        "category": "指令遵循",
        "question": "请用一句话说明验证集的作用。",
        "metric": "validation_role",
        "required_any": ["泛化", "过拟合", "评估", "表现", "参数更新"],
        "forbidden_any": NOVEL_LEAK_MARKERS,
    },
    {
        "id": "general_encourage",
        "category": "通用聊天",
        "question": "请写一句鼓励学习的话。",
        "metric": "chat_quality",
    },
    {
        "id": "general_plan_today",
        "category": "通用聊天",
        "question": "我应该如何安排今天的学习？",
        "metric": "chat_quality",
    },
    {
        "id": "general_python_focus",
        "category": "通用聊天",
        "question": "学习Python时总是分心怎么办？",
        "metric": "chat_quality",
    },
    {
        "id": "general_explain_sft",
        "category": "通用聊天",
        "question": "请用一句话解释什么是监督微调。",
        "metric": "all_required",
        "required_all": ["监督微调"],
        "forbidden_any": NOVEL_LEAK_MARKERS + ["天气"],
    },
    {
        "id": "general_explain_bpe",
        "category": "通用聊天",
        "question": "BPE是什么？",
        "metric": "all_required",
        "required_all": ["BPE"],
        "forbidden_any": ["萧炎", "第", "章"],
    },
]


def load_sft_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"vocab_size", "special_token_ids", "itos", "tokenizer_path"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"SFT payload is missing keys: {missing}")
    return payload


def contains_any(text: str, values: Sequence[str]) -> bool:
    return any(value in text for value in values)


def contains_all(text: str, values: Sequence[str]) -> bool:
    return all(value in text for value in values)


def has_forbidden(text: str, item: dict[str, Any]) -> bool:
    return contains_any(text, item.get("forbidden_any", []))


def score_item(item: dict[str, Any], answer: str, stopped_on_eos: bool) -> dict[str, Any]:
    metric = item["metric"]
    reason = ""
    passed = False

    if metric == "known_entity":
        passed = (
            contains_any(answer, item["required_any"])
            and contains_any(answer, item.get("required_context_any", item["required_any"]))
            and not has_forbidden(answer, item)
        )
        reason = "must identify the entity with a relevant role and avoid unknown refusal"
    elif metric == "exact_any":
        passed = contains_any(answer, item["required_any"]) and not has_forbidden(answer, item)
        reason = "must contain one expected phrase and avoid forbidden phrases"
    elif metric == "all_required":
        passed = contains_all(answer, item.get("required_all", [])) and not has_forbidden(answer, item)
        reason = "must contain all required phrases and avoid forbidden phrases"
    elif metric == "boundary":
        passed = (
            contains_any(answer, BOUNDARY_MARKERS)
            and contains_any(answer, item.get("topic_required_any", REALTIME_MARKERS))
            and not contains_any(answer, ["第", "章", "萧炎", "药老"])
        )
        reason = "must state topic-specific capability boundary without leaking novel patterns"
    elif metric == "exact_response":
        normalized = answer.strip()
        passed = normalized in item["accepted_responses"]
        reason = f"must exactly match one of {item['accepted_responses']}"
    elif metric == "two_advice":
        segments = [
            segment.strip()
            for segment in re.split(r"[。；\n]+", answer)
            if segment.strip()
        ]
        numbered = bool(re.search(r"(?:1[.、]|一[、.]).*(?:2[.、]|二[、.])", answer))
        passed = (
            (len(segments) >= 2 or numbered)
            and contains_any(answer, item["required_any"])
            and not contains_any(answer, NOVEL_LEAK_MARKERS)
        )
        reason = "must provide at least two topic-relevant suggestions without novel leakage"
    elif metric == "validation_role":
        segments = [
            segment.strip()
            for segment in re.split(r"[。；\n]+", answer)
            if segment.strip()
        ]
        passed = (
            "验证集" in answer
            and contains_any(answer, item["required_any"])
            and len(segments) == 1
            and not has_forbidden(answer, item)
        )
        reason = "must explain validation-set purpose in one sentence without novel leakage"
    elif metric == "chat_quality":
        passed = (
            len(answer.strip()) >= 6
            and not contains_any(answer, NOVEL_LEAK_MARKERS)
            and not contains_any(answer, ["资料不足", "无法确定", "不能硬编"])
            and (
                not item.get("required_any")
                or contains_any(answer, item["required_any"])
            )
        )
        reason = "must be useful, follow requested topic, and avoid novel/refusal leakage"
    else:
        raise ValueError(f"unsupported metric: {metric}")

    if not stopped_on_eos:
        passed = False
        reason = f"{reason}; generation must stop on EOS"

    return {
        "passed": bool(passed),
        "reason": reason,
        "metric": metric,
    }


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_category[result["category"]].append(result)
    categories = {}
    for category, items in sorted(by_category.items()):
        passed = sum(1 for item in items if item["passed"])
        categories[category] = {
            "passed": passed,
            "total": len(items),
            "accuracy": passed / len(items),
        }
    total_passed = sum(1 for result in results if result["passed"])
    return {
        "passed": total_passed,
        "total": len(results),
        "accuracy": total_passed / len(results),
        "by_category": categories,
        "eos_count": sum(1 for result in results if result["stopped_on_eos"]),
        "metric_counts": dict(Counter(result["metric"] for result in results)),
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    rows = [
        "# SFT 分类型诊断评估（v2，无数学）",
        "",
        f"Checkpoint：`{report['checkpoint']}`",
        "",
        f"Checkpoint step：`{report['checkpoint_step']}`",
        "",
        f"Checkpoint SHA-256：`{report['checkpoint_sha256']}`",
        "",
        f"采样参数：temperature `{report['temperature']}`，top-k `{report['top_k']}`，max_new_tokens `{report['max_new_tokens']}`",
        "",
        f"总分：`{summary['passed']}/{summary['total']}`，准确率 `{summary['accuracy']:.2%}`，EOS `{summary['eos_count']}/{summary['total']}`",
        "",
        "## 分类型结果",
        "",
        "| 类别 | 通过 | 总数 | 准确率 |",
        "|---|---:|---:|---:|",
    ]
    for category, item in summary["by_category"].items():
        rows.append(
            f"| {category} | {item['passed']} | {item['total']} | {item['accuracy']:.2%} |"
        )
    rows.extend(
        [
            "",
            "## 明细",
            "",
            "| # | 类别 | 输入 | 输出 | 通过 | 规则 |",
            "|---:|---|---|---|---|---|",
        ]
    )
    for index, result in enumerate(report["results"], 1):
        rows.append(
            "| {index} | {category} | {question} | {answer} | {passed} | {reason} |".format(
                index=index,
                category=markdown_escape(result["category"]),
                question=markdown_escape(result["question"]),
                answer=markdown_escape(result["generated_answer"]),
                passed="是" if result["passed"] else "否",
                reason=markdown_escape(result["score_reason"]),
            )
        )
    return "\n".join(rows) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.top_k < 0:
        raise ValueError("top_k must be non-negative")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    run_id = generate_run_id("sft-v4-category-eval")
    base_config = load_config(args.config)
    loggers = configure_module_loggers(
        args.output_json.parent / "logs",
        run_id,
        {"data": "INFO", "validation": "INFO"},
        max_bytes=int(base_config["logging"]["max_bytes"]),
        backup_count=int(base_config["logging"]["backup_count"]),
        console=bool(base_config["logging"]["console"]),
    )

    payload = load_sft_payload(args.data)
    tokenizer = BPETokenizer.load(Path(payload["tokenizer_path"]))
    device = select_device(args.device)
    model = build_model(base_config, int(payload["vocab_size"])).to(device)
    checkpoint = load_model_checkpoint(model, args.checkpoint, device)
    try:
        payload_compatibility = validate_checkpoint_payload_compatibility(
            checkpoint,
            payload,
        )
    except ValueError:
        loggers["validation"].exception(
            "checkpoint/data compatibility failed checkpoint=%s data=%s",
            args.checkpoint,
            args.data,
        )
        raise
    loggers["validation"].info(
        "checkpoint/data compatibility passed splits=%s",
        payload_compatibility["split_counts"],
    )

    results = []
    for index, item in enumerate(EVAL_ITEMS):
        prompt_ids = build_prompt_ids(
            tokenizer,
            item["question"],
            payload["special_token_ids"],
        )
        generated, stopped_on_eos = generate_answer(
            model,
            prompt_ids,
            payload["itos"],
            payload["special_token_ids"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed + index,
            device=device,
        )
        score = score_item(item, generated, stopped_on_eos)
        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "generated_answer": generated,
                "stopped_on_eos": stopped_on_eos,
                "metric": score["metric"],
                "passed": score["passed"],
                "score_reason": score["reason"],
            }
        )

    checkpoint_step = int(checkpoint.get("step", -1))
    report = {
        "schema_version": "sft-v4-category-eval/v2-no-math",
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "data": str(args.data),
        "data_sha256": file_sha256(args.data),
        "checkpoint_data_compatible": True,
        "device": str(device),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
        "test_records_consumed": 0,
        "evaluation_suite": "six_categories_no_math_v2",
        "summary": summarize(results),
        "results": results,
    }
    atomic_write_json(args.output_json, report)
    atomic_write_text(args.output_md, render_markdown(report))
    loggers["validation"].info(
        "category eval complete checkpoint_step=%d score=%d/%d output=%s",
        checkpoint_step,
        report["summary"]["passed"],
        report["summary"]["total"],
        args.output_json,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "checkpoint_step": checkpoint_step,
                "score": f"{report['summary']['passed']}/{report['summary']['total']}",
                "accuracy": report["summary"]["accuracy"],
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
