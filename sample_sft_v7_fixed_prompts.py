"""Run the frozen 16-case SFT v7 comparison suite against one checkpoint.

Only the prompt-set JSON, isolated public tensor artifact, tokenizer files and
checkpoint are read. Public source JSONL and the sealed split are never inputs.
Prompt and response bodies are written to the requested result artifacts only;
rotating structured logs contain counts, hashes and error codes.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import sample_sft_v7 as sample
from training_runtime import (
    atomic_write_json,
    atomic_write_text,
    close_module_loggers,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
    resolve_module_log_levels,
)


DEFAULT_PROMPT_SET = Path("configs/sft_v7_fixed_prompts.json")
DEFAULT_PUBLIC_TENSORS = sample.DEFAULT_PUBLIC_TENSORS
DEFAULT_CHECKPOINT = sample.DEFAULT_CHECKPOINT
DEFAULT_BASELINE_CHECKPOINT = sample.DEFAULT_BASELINE_CHECKPOINT
DEFAULT_OUTPUT_JSON = Path(
    "reports/milestones/020_sft_v7_vertical/fixed_samples.json"
)
DEFAULT_OUTPUT_MARKDOWN = Path(
    "reports/milestones/020_sft_v7_vertical/fixed_samples.md"
)
DEFAULT_LOG_DIR = Path("logs/sft_v7_fixed_samples")
EXPECTED_SCHEMA = "sft-v7-fixed-prompts/v1"
EXPECTED_CASE_COUNT = 16
GENERATION_FIELDS = ("max_new_tokens", "temperature", "top_k", "seed")
_CASE_ID = re.compile(r"[a-z0-9_]+")


class FixedPromptSamplingError(ValueError):
    """A prompt-body-safe failure identified only by a stable error code."""

    def __init__(self, code: str):
        super().__init__(f"fixed prompt sampling failed [{code}]")
        self.code = code


def _fail(code: str) -> None:
    raise FixedPromptSamplingError(code)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_generation(generation: Any) -> dict[str, int | float]:
    if not isinstance(generation, Mapping) or set(generation) != set(GENERATION_FIELDS):
        _fail("prompt_set_generation_fields")
    max_new_tokens = generation.get("max_new_tokens")
    temperature = generation.get("temperature")
    top_k = generation.get("top_k")
    seed = generation.get("seed")
    if not _is_int(max_new_tokens) or int(max_new_tokens) <= 0:
        _fail("prompt_set_max_new_tokens")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or float(temperature) <= 0
    ):
        _fail("prompt_set_temperature")
    if not _is_int(top_k) or int(top_k) < 0:
        _fail("prompt_set_top_k")
    if not _is_int(seed) or not 0 <= int(seed) < 2**63:
        _fail("prompt_set_seed")
    return {
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "seed": int(seed),
    }


def validate_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages or len(messages) % 2 == 0:
        _fail("prompt_set_turn_count")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            _fail("prompt_set_message_fields")
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected_role:
            _fail("prompt_set_role_order")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            _fail("prompt_set_empty_content")
        normalized.append({"role": expected_role, "content": content})
    return normalized


def validate_prompt_set(document: Any) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        _fail("prompt_set_root")
    if set(document) != {"schema_version", "purpose", "generation", "cases"}:
        _fail("prompt_set_fields")
    if document.get("schema_version") != EXPECTED_SCHEMA:
        _fail("prompt_set_schema")
    if not isinstance(document.get("purpose"), str) or not document["purpose"].strip():
        _fail("prompt_set_purpose")
    generation = validate_generation(document.get("generation"))
    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != EXPECTED_CASE_COUNT:
        _fail("prompt_set_case_count")
    seen_ids: set[str] = set()
    cases: list[dict[str, Any]] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping) or set(raw_case) != {
            "id",
            "category",
            "messages",
        }:
            _fail("prompt_set_case_fields")
        case_id = raw_case.get("id")
        category = raw_case.get("category")
        if (
            not isinstance(case_id, str)
            or not _CASE_ID.fullmatch(case_id)
            or case_id in seen_ids
        ):
            _fail("prompt_set_case_id")
        if not isinstance(category, str) or not category.strip():
            _fail("prompt_set_category")
        seen_ids.add(case_id)
        cases.append(
            {
                "id": case_id,
                "category": category,
                "messages": validate_messages(raw_case.get("messages")),
            }
        )
    return {
        "schema_version": EXPECTED_SCHEMA,
        "purpose": str(document["purpose"]),
        "generation": generation,
        "cases": cases,
    }


def load_prompt_set(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FixedPromptSamplingError("prompt_set_read") from error
    return validate_prompt_set(document)


def resolve_generation(
    frozen: Mapping[str, int | float], args: argparse.Namespace
) -> tuple[dict[str, int | float], list[str]]:
    resolved = dict(frozen)
    overridden: list[str] = []
    for field in GENERATION_FIELDS:
        value = getattr(args, field)
        if value is not None:
            resolved[field] = value
            overridden.append(field)
    return validate_generation(resolved), overridden


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _render_messages(messages: Sequence[Mapping[str, str]]) -> str:
    return "<br>".join(
        f"{message['role']}: {_escape_markdown(message['content'])}"
        for message in messages
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    decoding = report["generation"]
    rows = [
        "# SFT v7 固定 16 题完整采样",
        "",
        f"Checkpoint：`{report['checkpoint_step']}`（`{report['checkpoint_mode']}`）",
        f"Checkpoint SHA-256：`{report['checkpoint_sha256']}`",
        f"Prompt-set SHA-256：`{report['prompt_set_sha256']}`",
        (
            "解码："
            f"max_new_tokens={decoding['max_new_tokens']}，"
            f"temperature={decoding['temperature']}，top_k={decoding['top_k']}，"
            f"seed={decoding['seed']}"
        ),
        "",
        "| # | ID | 类别 | 完整对话输入 | 完整模型输出 | EOS | 截断 |",
        "|---:|---|---|---|---|---|---|",
    ]
    for index, result in enumerate(report["results"], 1):
        rows.append(
            f"| {index} | {_escape_markdown(result['id'])} | "
            f"{_escape_markdown(result['category'])} | "
            f"{_render_messages(result['messages'])} | "
            f"{_escape_markdown(result['generated_text'])} | "
            f"{'是' if result['stopped_on_eos'] else '否'} | "
            f"{'是' if result['truncated'] else '否'} |"
        )
    return "\n".join(rows) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-set", type=Path, default=DEFAULT_PROMPT_SET)
    parser.add_argument("--config", type=Path, default=sample.DEFAULT_CONFIG)
    parser.add_argument("--public-tensors", type=Path, default=DEFAULT_PUBLIC_TENSORS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--checkpoint-mode",
        choices=("sft-v7", "pretrain-baseline"),
        default="sft-v7",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    sample.reject_forbidden_public_fields(
        {
            "public_tensor_path": args.public_tensors,
            "checkpoint_path": args.checkpoint,
        },
        location="fixed_prompt_arguments",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.checkpoint_mode == "pretrain-baseline" and args.checkpoint == DEFAULT_CHECKPOINT:
        args.checkpoint = DEFAULT_BASELINE_CHECKPOINT
    run_id = generate_run_id("sft-v7-fixed")
    config = sample.load_config(args.config)
    loggers = configure_module_loggers(
        args.log_dir,
        run_id,
        resolve_module_log_levels(
            {"generation": "INFO", "evaluation": "INFO", "checkpoint": "INFO"}
        ),
        max_bytes=int(config["logging"]["max_bytes"]),
        backup_count=int(config["logging"]["backup_count"]),
        console=bool(config["logging"]["console"]),
    )
    try:
        validate_args(args)
        prompt_set = load_prompt_set(args.prompt_set)
        prompt_set_sha256 = file_sha256(args.prompt_set)
        generation, overridden_fields = resolve_generation(
            prompt_set["generation"], args
        )
        payload = sample.load_public_payload(args.public_tensors)
        tokenizer = sample.load_bound_tokenizer(payload)
        device = sample.select_device(args.device)
        model, checkpoint, provenance = sample.load_model_bundle(
            args.config,
            args.checkpoint,
            payload,
            device,
            args.checkpoint_mode,
        )
        checkpoint_sha256 = file_sha256(args.checkpoint)
        loggers["checkpoint"].info(
            "checkpoint validated step=%d mode=%s sha256=%s device=%s",
            int(checkpoint["step"]),
            args.checkpoint_mode,
            checkpoint_sha256,
            device,
        )
        results: list[dict[str, Any]] = []
        for index, case in enumerate(prompt_set["cases"]):
            prompt_ids = sample.build_conversation_prompt_ids(
                tokenizer,
                case["messages"],
                payload["special_token_ids"],
            )
            case_seed = int(generation["seed"]) + index
            generated = sample.generate_response(
                model,
                prompt_ids,
                tokenizer,
                payload["special_token_ids"],
                max_new_tokens=int(generation["max_new_tokens"]),
                temperature=float(generation["temperature"]),
                top_k=int(generation["top_k"]),
                seed=case_seed,
                device=device,
            )
            results.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "messages": deepcopy(case["messages"]),
                    "generated_text": generated["generated_text"],
                    "generated_token_ids": generated["generated_token_ids"],
                    "generated_tokens": generated["generated_tokens"],
                    "stopped_on_eos": generated["stopped_on_eos"],
                    "truncated": generated["truncated"],
                    "seed": case_seed,
                }
            )
            loggers["generation"].info(
                "case generated index=%d turns=%d prompt_tokens=%d output_tokens=%d "
                "eos=%s truncated=%s",
                index,
                len(case["messages"]),
                len(prompt_ids),
                generated["generated_tokens"],
                generated["stopped_on_eos"],
                generated["truncated"],
            )
        report = {
            "schema_version": "sft-v7-fixed-samples/v1",
            "status": "complete",
            "run_id": run_id,
            "prompt_set_path": str(args.prompt_set),
            "prompt_set_sha256": prompt_set_sha256,
            "prompt_set_schema": prompt_set["schema_version"],
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_step": int(checkpoint["step"]),
            "checkpoint_mode": args.checkpoint_mode,
            "checkpoint_provenance": provenance,
            "public_tensor_path": str(args.public_tensors),
            "public_tensor_sha256": file_sha256(args.public_tensors),
            "tokenizer_sha256": payload["tokenizer_sha256"],
            "bpe_token_manifest_sha256": payload["bpe_token_manifest_sha256"],
            "sft_dataset_manifest_sha256": payload[
                "sft_dataset_manifest_sha256"
            ],
            "device": str(device),
            "generation": {
                **generation,
                "overridden_fields": overridden_fields,
                "per_case_seed_rule": "generation.seed + zero_based_case_index",
                "masked_special_tokens": [
                    "<UNK>",
                    "<BOS>",
                    "<USER>",
                    "<ASSISTANT>",
                    "<PAD>",
                ],
                "eos_allowed": True,
            },
            "results": results,
        }
        atomic_write_json(args.output_json, report)
        atomic_write_text(args.output_markdown, render_markdown(report))
        loggers["evaluation"].info(
            "fixed artifacts written cases=%d eos=%d truncated=%d "
            "prompt_set_sha256=%s checkpoint_sha256=%s",
            len(results),
            sum(bool(result["stopped_on_eos"]) for result in results),
            sum(bool(result["truncated"]) for result in results),
            prompt_set_sha256,
            checkpoint_sha256,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "checkpoint_step": int(checkpoint["step"]),
                    "cases": len(results),
                    "output_json": str(args.output_json),
                    "output_markdown": str(args.output_markdown),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as error:
        error_code = str(getattr(error, "code", "unexpected_failure"))
        loggers["evaluation"].error(
            "fixed prompt sampling failed error_code=%s",
            error_code,
        )
        if isinstance(error, FixedPromptSamplingError):
            raise
        raise FixedPromptSamplingError(error_code) from error
    finally:
        close_module_loggers(loggers)


if __name__ == "__main__":
    raise SystemExit(main())
