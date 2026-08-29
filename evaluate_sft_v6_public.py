"""Evaluate an SFT v6 checkpoint without reading the sealed test split."""

from __future__ import annotations

import argparse
from collections import defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any, Sequence

import torch

from bpe_tokenizer import BPETokenizer
from sample_sft_v4_custom import build_prompt_ids
from train_pretrain_v4 import load_config
from train_sft_v4 import build_model, generate_answer, load_model_checkpoint, select_device
from train_sft_v5 import evaluate_all_records
from training_runtime import (
    atomic_write_json,
    atomic_write_text,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
    resolve_module_log_levels,
)


DEFAULT_CONFIG = Path("configs/formal_pretrain_14m_bpe3000.json")
DEFAULT_DATASET = Path("data/sft/v6/sft_v6_10000.jsonl")
DEFAULT_TENSORS = Path("data/sft/v6/sft_v6_bpe_tensors.pt")
DEFAULT_CHECKPOINT = Path("runs/sft_v6_10000_step2000/checkpoints/step_02000.pt")
DEFAULT_REPORT = Path("reports/milestones/018_sft_v6_10000/public_eval_step2000.json")
DEFAULT_MARKDOWN = Path("reports/milestones/018_sft_v6_10000/public_eval_step2000.md")


def normalized_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def similarity(reference: str, generated: str) -> float:
    return SequenceMatcher(None, normalized_text(reference), normalized_text(generated)).ratio()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_balanced(records: Sequence[dict[str, Any]], per_family: int) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_family[str(record["task_family"])].append(record)
    selected: list[dict[str, Any]] = []
    for family in sorted(by_family):
        selected.extend(sorted(by_family[family], key=lambda record: str(record["id"]))[:per_family])
    return selected


def render_markdown(report: dict[str, Any]) -> str:
    rows = [
        f"# SFT v6 公开诊断：step {report['checkpoint_step']}",
        "",
        f"公开诊断 Loss：`{report['public_loss']:.6f}`",
        "",
        f"生成样本：`{report['sample_count']}`；EOS：`{report['eos_rate']:.1%}`；"
        f"完全一致：`{report['exact_match_rate']:.1%}`；平均文本相似度：`{report['mean_similarity']:.3f}`",
        "",
        "| 任务 | 问题 | 参考答案 | 模型输出 | EOS | 相似度 |",
        "|---|---|---|---|---|---:|",
    ]
    for item in report["samples"]:
        escape = lambda text: str(text).replace("|", "\\|").replace("\n", "↵")
        rows.append(
            f"| {escape(item['task_family'])} | {escape(item['question'])} | "
            f"{escape(item['reference_answer'])} | {escape(item['generated_answer'])} | "
            f"{'是' if item['stopped_on_eos'] else '否'} | {item['similarity']:.3f} |"
        )
    return "\n".join(rows) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tensors", type=Path, default=DEFAULT_TENSORS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--per-family", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.per_family <= 0 or args.max_new_tokens <= 0:
        raise ValueError("per-family and max-new-tokens must be positive")
    config = load_config(args.config)
    run_id = generate_run_id("sft-v6-public-eval")
    loggers = configure_module_loggers(
        args.report.parent / "logs",
        run_id,
        resolve_module_log_levels(
            {"data": "INFO", "validation": "INFO", "orchestrator": "INFO"}
        ),
        max_bytes=int(config["logging"]["max_bytes"]),
        backup_count=int(config["logging"]["backup_count"]),
        console=bool(config["logging"]["console"]),
    )
    try:
        payload = torch.load(args.tensors, map_location="cpu", weights_only=False)
        if "public_diagnostic_records" not in payload:
            raise ValueError("tensor payload has no public diagnostic split")
        source_records = load_jsonl(args.dataset)
        public_sources = [record for record in source_records if record["split"] == "public_diagnostic"]
        if len(public_sources) != len(payload["public_diagnostic_records"]):
            raise ValueError("public diagnostic JSONL/tensor counts differ")

        device = select_device(args.device)
        tokenizer = BPETokenizer.load(Path(payload["tokenizer_path"]))
        model = build_model(config, int(payload["vocab_size"])).to(device)
        checkpoint = load_model_checkpoint(model, args.checkpoint, device)
        expected_data_sha = checkpoint.get("extra", {}).get("data_sha256")
        if expected_data_sha != file_sha256(args.tensors):
            raise ValueError("checkpoint was not trained from the supplied tensor payload")
        public_loss = evaluate_all_records(
            model,
            payload["public_diagnostic_records"],
            int(payload["special_token_ids"]["<PAD>"]),
            8,
            device,
        )
        selected = select_balanced(public_sources, args.per_family)
        samples: list[dict[str, Any]] = []
        for index, record in enumerate(selected):
            prompt_ids = build_prompt_ids(tokenizer, record["question"], payload["special_token_ids"])
            generated, stopped_on_eos = generate_answer(
                model,
                prompt_ids,
                payload["itos"],
                payload["special_token_ids"],
                max_new_tokens=args.max_new_tokens,
                temperature=0.3,
                top_k=1,
                seed=20260829 + index,
                device=device,
            )
            reference = str(record["answer"])
            samples.append(
                {
                    "id": record["id"],
                    "task_family": record["task_family"],
                    "question": record["question"],
                    "reference_answer": reference,
                    "generated_answer": generated,
                    "stopped_on_eos": stopped_on_eos,
                    "exact_match": normalized_text(reference) == normalized_text(generated),
                    "similarity": similarity(reference, generated),
                }
            )
        report = {
            "schema_version": "sft-v6-public-eval/v1",
            "status": "complete",
            "run_id": run_id,
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": int(checkpoint.get("step", -1)),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "tensor_sha256": file_sha256(args.tensors),
            "dataset_sha256": file_sha256(args.dataset),
            "public_records": len(public_sources),
            "public_loss": public_loss,
            "sample_count": len(samples),
            "eos_rate": sum(item["stopped_on_eos"] for item in samples) / len(samples),
            "exact_match_rate": sum(item["exact_match"] for item in samples) / len(samples),
            "mean_similarity": sum(item["similarity"] for item in samples) / len(samples),
            "sealed_test_records_consumed": 0,
            "samples": samples,
        }
        atomic_write_json(args.report, report)
        atomic_write_text(args.markdown, render_markdown(report))
        loggers["validation"].info(
            "public evaluation complete step=%d loss=%.6f samples=%d eos=%.3f exact=%.3f similarity=%.3f sealed_consumed=0",
            report["checkpoint_step"],
            public_loss,
            len(samples),
            report["eos_rate"],
            report["exact_match_rate"],
            report["mean_similarity"],
        )
        loggers["orchestrator"].info("wrote report=%s markdown=%s", args.report, args.markdown)
        print(json.dumps({key: report[key] for key in (
            "status", "checkpoint_step", "public_loss", "sample_count", "eos_rate",
            "exact_match_rate", "mean_similarity", "sealed_test_records_consumed",
        )}, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        loggers["validation"].exception("SFT v6 public evaluation failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
