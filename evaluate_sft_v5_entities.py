"""Evaluate known-entity answers and grounded unknown refusals on frozen prompts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from bpe_tokenizer import BPETokenizer
from sample_sft_v4_custom import (
    build_prompt_ids,
    markdown_escape,
    validate_checkpoint_payload_compatibility,
)
from sft_v5_entity_spec import HIDDEN_ENTITY_EVAL_ITEMS
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
DEFAULT_DATA_PATH = Path("data/cloud_v4/sft_v5_2_2_core_routing_tensors.pt")
DEFAULT_CHECKPOINT_PATH = Path("runs/sft_v5_2_2_core_probe1000/latest.pt")
DEFAULT_OUTPUT_JSON = Path(
    "reports/milestones/014_v5_2_entity_routing/hidden_entity_eval.json"
)
DEFAULT_OUTPUT_MD = Path(
    "reports/milestones/014_v5_2_entity_routing/hidden_entity_eval.md"
)
REFUSAL_MARKERS = (
    "没有找到",
    "无法核实",
    "不能确认",
    "无法确认",
    "无法确定",
    "资料不足",
    "资料中没有",
)


def score_hidden_item(
    item: dict[str, Any],
    answer: str,
    stopped_on_eos: bool,
) -> tuple[bool, str]:
    if item["category"] == "已知实体":
        passed = (
            any(value in answer for value in item["required_any"])
            and any(value in answer for value in item["required_context_any"])
            and not any(value in answer for value in REFUSAL_MARKERS)
        )
        reason = "应识别已知实体及其关键身份，不得拒答"
    elif item["category"] == "不存在实体":
        passed = (
            (
                item["entity"] in answer
                or any(reference in answer for reference in ("该名称", "这个名称", "它"))
            )
            and any(value in answer for value in REFUSAL_MARKERS)
        )
        reason = "应点名实体并基于语料缺失作有依据的拒答"
    else:
        raise ValueError(f"unsupported hidden category: {item['category']}")
    if not stopped_on_eos:
        return False, reason + "；必须生成EOS"
    return bool(passed), reason


def summarize_hidden(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    categories: dict[str, dict[str, Any]] = {}
    for category in sorted({row["category"] for row in results}):
        rows = [row for row in results if row["category"] == category]
        passed = sum(1 for row in rows if row["passed"])
        categories[category] = {
            "passed": passed,
            "total": len(rows),
            "accuracy": passed / len(rows),
        }
    passed = sum(1 for row in results if row["passed"])
    return {
        "passed": passed,
        "total": len(results),
        "accuracy": passed / len(results),
        "eos_count": sum(1 for row in results if row["stopped_on_eos"]),
        "by_category": categories,
        "failure_counts": dict(
            Counter(row["category"] for row in results if not row["passed"])
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    rows = [
        "# SFT v5.2 隐藏实体评估",
        "",
        f"总分：`{summary['passed']}/{summary['total']}`，EOS：`{summary['eos_count']}/{summary['total']}`",
        "",
        "| 类别 | 通过 | 总数 | 准确率 |",
        "|---|---:|---:|---:|",
    ]
    for category, values in summary["by_category"].items():
        rows.append(
            f"| {category} | {values['passed']} | {values['total']} | {values['accuracy']:.2%} |"
        )
    rows.extend(
        [
            "",
            "| # | 类别 | 问题 | 输出 | 通过 |",
            "|---:|---|---|---|---|",
        ]
    )
    for index, result in enumerate(report["results"], 1):
        rows.append(
            f"| {index} | {result['category']} | {markdown_escape(result['question'])} | "
            f"{markdown_escape(result['generated_answer'])} | {'是' if result['passed'] else '否'} |"
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
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = generate_run_id("sft-v5-hidden-entity-eval")
    base_config = load_config(args.config)
    loggers = configure_module_loggers(
        args.output_json.parent / "logs",
        run_id,
        {"data": "INFO", "validation": "INFO"},
        max_bytes=int(base_config["logging"]["max_bytes"]),
        backup_count=int(base_config["logging"]["backup_count"]),
        console=bool(base_config["logging"]["console"]),
    )
    try:
        payload = torch.load(args.data, map_location="cpu", weights_only=False)
        tokenizer = BPETokenizer.load(Path(payload["tokenizer_path"]))
        device = select_device(args.device)
        model = build_model(base_config, int(payload["vocab_size"])).to(device)
        checkpoint = load_model_checkpoint(model, args.checkpoint, device)
        validate_checkpoint_payload_compatibility(checkpoint, payload)

        results: list[dict[str, Any]] = []
        for index, item in enumerate(HIDDEN_ENTITY_EVAL_ITEMS):
            prompt_ids = build_prompt_ids(
                tokenizer,
                item["question"],
                payload["special_token_ids"],
            )
            answer, stopped_on_eos = generate_answer(
                model,
                prompt_ids,
                payload["itos"],
                payload["special_token_ids"],
                max_new_tokens=40,
                temperature=0.3,
                top_k=1,
                seed=args.seed + index,
                device=device,
            )
            passed, reason = score_hidden_item(item, answer, stopped_on_eos)
            results.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "question": item["question"],
                    "generated_answer": answer,
                    "stopped_on_eos": stopped_on_eos,
                    "passed": passed,
                    "score_reason": reason,
                }
            )
        report = {
            "schema_version": "sft-v5-hidden-entity-eval/v1",
            "status": "complete",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_step": int(checkpoint.get("step", -1)),
            "data": str(args.data),
            "data_sha256": file_sha256(args.data),
            "temperature": 0.3,
            "top_k": 1,
            "max_new_tokens": 40,
            "seed": args.seed,
            "test_records_consumed": 0,
            "summary": summarize_hidden(results),
            "results": results,
        }
        atomic_write_json(args.output_json, report)
        atomic_write_text(args.output_md, render_markdown(report))
        loggers["validation"].info(
            "hidden entity eval complete score=%d/%d eos=%d/%d checkpoint_step=%d",
            report["summary"]["passed"],
            report["summary"]["total"],
            report["summary"]["eos_count"],
            report["summary"]["total"],
            report["checkpoint_step"],
        )
        print(json.dumps({
            "status": "complete",
            "score": f"{report['summary']['passed']}/{report['summary']['total']}",
            "eos": f"{report['summary']['eos_count']}/{report['summary']['total']}",
            "output_json": str(args.output_json),
            "output_md": str(args.output_md),
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        loggers["validation"].exception("hidden entity evaluation failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
