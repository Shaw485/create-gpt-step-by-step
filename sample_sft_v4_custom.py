"""Generate fixed custom samples from a v4 SFT checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from bpe_tokenizer import BPETokenizer
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
DEFAULT_DATA_PATH = Path("data/cloud_v4/sft_v4_mixed_chat_tensors.pt")
DEFAULT_CHECKPOINT_PATH = Path("runs/sft_v4_mixed_chat_step5000/best.pt")
DEFAULT_OUTPUT_JSON = Path(
    "reports/milestones/011_v4_mixed_chat_sft/novel_vs_general_step5000_best_samples.json"
)
DEFAULT_OUTPUT_MD = Path(
    "reports/milestones/011_v4_mixed_chat_sft/novel_vs_general_step5000_best_samples.md"
)
DEFAULT_MAX_NEW_TOKENS = 30
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 20
DEFAULT_SEED = 20260828

DEFAULT_PROMPTS = [
    {"category": "小说问题", "question": "小说第三百章的标题是什么？"},
    {"category": "小说问题", "question": "萧炎是谁？"},
    {"category": "小说问题", "question": "药尘是谁？"},
    {"category": "小说问题", "question": "异火是什么？"},
    {"category": "小说问题", "question": "请用一句话介绍萧炎。"},
    {"category": "小说问题", "question": "根据证据片段，韩枫和紫研是否都被提到？"},
    {"category": "非小说问题", "question": "今天天气怎么样？"},
    {"category": "非小说问题", "question": "请写一句鼓励学习的话。"},
    {"category": "非小说问题", "question": "一加一等于几？"},
    {"category": "非小说问题", "question": "请介绍人工智能。"},
    {"category": "非小说问题", "question": "我应该如何安排今天的学习？"},
    {"category": "非小说问题", "question": "请用一句话解释什么是监督微调。"},
]


def load_sft_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"vocab_size", "special_token_ids", "itos", "tokenizer_path"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"SFT payload is missing keys: {missing}")
    return payload


def build_prompt_ids(
    tokenizer: BPETokenizer,
    question: str,
    special_token_ids: dict[str, int],
) -> list[int]:
    return [
        int(special_token_ids["<BOS>"]),
        int(special_token_ids["<USER>"]),
        *tokenizer.encode(question),
        int(special_token_ids["<ASSISTANT>"]),
    ]


def markdown_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def render_markdown(
    *,
    title: str,
    checkpoint_step: int,
    checkpoint_sha256: str,
    samples: Sequence[dict[str, Any]],
) -> str:
    rows = [
        f"# {title}",
        "",
        f"Checkpoint step：`{checkpoint_step}`",
        "",
        f"Checkpoint SHA-256：`{checkpoint_sha256}`",
        "",
        "| # | 类别 | 输入 | 输出 | EOS |",
        "|---:|---|---|---|---|",
    ]
    for index, sample in enumerate(samples, 1):
        rows.append(
            "| {index} | {category} | {question} | {answer} | {eos} |".format(
                index=index,
                category=markdown_escape(sample["category"]),
                question=markdown_escape(sample["question"]),
                answer=markdown_escape(sample["generated_answer"]),
                eos="是" if sample["stopped_on_eos"] else "否",
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
    run_id = generate_run_id("sft-v4-custom-samples")
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

    samples = []
    for index, prompt in enumerate(DEFAULT_PROMPTS):
        prompt_ids = build_prompt_ids(
            tokenizer,
            prompt["question"],
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
        samples.append(
            {
                "category": prompt["category"],
                "question": prompt["question"],
                "generated_answer": generated,
                "stopped_on_eos": stopped_on_eos,
            }
        )

    checkpoint_step = int(checkpoint.get("step", -1))
    checkpoint_sha256 = file_sha256(args.checkpoint)
    report = {
        "schema_version": "sft-v4-custom-samples/v1",
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": checkpoint_sha256,
        "data": str(args.data),
        "data_sha256": file_sha256(args.data),
        "device": str(device),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
        "samples": samples,
    }
    atomic_write_json(args.output_json, report)
    atomic_write_text(
        args.output_md,
        render_markdown(
            title="M011 mixed chat SFT checkpoint 小说/非小说样本",
            checkpoint_step=checkpoint_step,
            checkpoint_sha256=checkpoint_sha256,
            samples=samples,
        ),
    )
    loggers["validation"].info(
        "custom samples generated checkpoint_step=%d count=%d output=%s",
        checkpoint_step,
        len(samples),
        args.output_json,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "checkpoint_step": checkpoint_step,
                "samples": len(samples),
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
