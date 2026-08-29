"""Audit what a pure single-novel pretrained language model can actually do.

The audit deliberately evaluates a base language model as a next-token model,
not as a chatbot.  It measures a validation next-token diagnostic, fixed-prompt
continuations, mechanical degeneration, EOS behaviour, and declarative cloze
candidate ranking.  Question-answer formatting is therefore *not* a hard gate.

Logging is split into ``data``, ``checkpoint``, ``validation``, ``generation``,
``cloze``, and ``orchestrator`` JSONL streams.  Each module can be changed with
``GPT_AUDIT_LOG_LEVEL_<MODULE>`` (for example ``..._GENERATION=DEBUG`` or
``..._CLOZE=OFF``), or with repeatable ``--log-level MODULE=LEVEL`` arguments.
Logs rotate by size, carry a run ID, and redact common secret fields.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
from statistics import mean
import sys
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional

from bpe_tokenizer import BPETokenizer
from gpt_model_v4 import GPTConfig, GPTLanguageModelV4
from story_quality import longest_character_run, ngram_repetition
from train_pretrain_v4 import load_config, load_tensor
from training_runtime import (
    JsonLogFormatter,
    atomic_write_json,
    atomic_write_text,
    canonical_json_sha256,
    file_sha256,
    generate_run_id,
    load_checkpoint,
)


SCHEMA_VERSION = "pretrain-capability-audit/v1"
LOG_MODULES = (
    "data",
    "checkpoint",
    "validation",
    "generation",
    "cloze",
    "orchestrator",
)
DEFAULT_LOG_LEVELS = {module: "INFO" for module in LOG_MODULES}
DEFAULT_CLOZE_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "xiaoyan-father",
        "context": "少年萧炎的父亲是",
        "candidates": ["萧战", "药老", "海波东", "古元"],
        "correct": "萧战",
    },
    {
        "id": "yaolao-name",
        "context": "萧炎的老师药老，本名",
        "candidates": ["药尘", "萧战", "古元", "韩枫"],
        "correct": "药尘",
    },
    {
        "id": "flame-mantra",
        "context": "萧炎修炼的奇异功法名为",
        "candidates": ["焚诀", "风雷决", "弄焰诀", "天火三玄变"],
        "correct": "焚诀",
    },
    {
        "id": "inner-academy-tower",
        "context": "迦南学院内院中有一座",
        "candidates": ["天焚炼气塔", "丹塔", "魂殿", "云岚宗"],
        "correct": "天焚炼气塔",
    },
    {
        "id": "medusa-name",
        "context": "美杜莎女王后来使用的名字是",
        "candidates": ["彩鳞", "云韵", "小医仙", "雅妃"],
        "correct": "彩鳞",
    },
    {
        "id": "wutan-family",
        "context": "萧炎出生于乌坦城的",
        "candidates": ["萧家", "古族", "魂族", "纳兰家"],
        "correct": "萧家",
    },
)
DEGENERATION_THRESHOLDS = {
    "maximum_four_gram_repetition": 0.20,
    "maximum_character_run": 6,
    "minimum_unique_character_ratio": 0.10,
    "minimum_characters_for_unique_ratio": 20,
}


@dataclass(frozen=True)
class GenerationResult:
    continuation: str
    generated_token_ids: list[int]
    stop_reason: str
    eos_emitted: bool


def _parse_level(level: str) -> tuple[int, bool]:
    normalized = level.strip().upper()
    if normalized in {"OFF", "DISABLED", "NONE"}:
        return logging.CRITICAL + 1, True
    value = getattr(logging, normalized, None)
    if not isinstance(value, int):
        raise ValueError(f"unknown log level: {level!r}")
    return value, False


def resolve_log_levels(overrides: Sequence[str] = ()) -> dict[str, str]:
    """Resolve production-safe defaults, environment, then CLI overrides."""

    levels = {
        module: os.getenv(
            f"GPT_AUDIT_LOG_LEVEL_{module.upper()}", DEFAULT_LOG_LEVELS[module]
        )
        for module in LOG_MODULES
    }
    for override in overrides:
        if "=" not in override:
            raise ValueError("--log-level must use MODULE=LEVEL")
        module, level = (part.strip() for part in override.split("=", 1))
        if module not in LOG_MODULES:
            raise ValueError(
                f"unknown audit log module {module!r}; choose from {', '.join(LOG_MODULES)}"
            )
        _parse_level(level)
        levels[module] = level.upper()
    return levels


def configure_audit_loggers(
    log_dir: Path,
    run_id: str,
    levels: Mapping[str, str],
    *,
    max_bytes: int = 2 * 1024 * 1024,
    backup_count: int = 3,
    console: bool = True,
) -> dict[str, logging.Logger]:
    """Create independently filterable, rotating, redacted JSONL log streams."""

    if max_bytes <= 0 or backup_count < 0:
        raise ValueError("log max_bytes must be positive and backup_count non-negative")
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = JsonLogFormatter(run_id)
    loggers: dict[str, logging.Logger] = {}
    for module in LOG_MODULES:
        level, disabled = _parse_level(levels.get(module, "INFO"))
        logger = logging.getLogger(f"pretrain_audit.{module}")
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()
        logger.propagate = False
        logger.disabled = disabled
        logger.setLevel(level)
        file_handler = RotatingFileHandler(
            log_dir / f"{run_id}.{module}.jsonl",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        loggers[module] = logger
    return loggers


def close_audit_loggers(loggers: Mapping[str, logging.Logger]) -> None:
    for logger in loggers.values():
        for handler in logger.handlers:
            try:
                handler.flush()
            finally:
                handler.close()
        logger.handlers.clear()
        logger.disabled = False


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = (
            "mps"
            if torch.backends.mps.is_built() and torch.backends.mps.is_available()
            else "cpu"
        )
    if requested == "mps":
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device: {requested}")


def deterministic_window_starts(
    token_count: int,
    block_size: int,
    window_count: int,
) -> list[int]:
    """Select reproducible, evenly spaced next-token validation windows."""

    if block_size <= 0 or window_count <= 0:
        raise ValueError("block_size and window_count must be positive")
    maximum_start = token_count - block_size - 1
    if maximum_start < 0:
        raise ValueError("evaluation split is not longer than block_size")
    count = min(window_count, maximum_start + 1)
    if count == 1:
        return [maximum_start // 2]
    return [
        (index * maximum_start) // (count - 1)
        for index in range(count)
    ]


@torch.inference_mode()
def evaluate_held_out(
    model: GPTLanguageModelV4,
    data: torch.Tensor,
    *,
    window_count: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate fixed windows, returning token NLL and token-level perplexity."""

    if batch_size <= 0:
        raise ValueError("evaluation batch_size must be positive")
    starts = deterministic_window_starts(
        len(data), model.config.block_size, window_count
    )
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    top1_correct = 0
    try:
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            inputs = torch.stack(
                [data[start : start + model.config.block_size] for start in batch_starts]
            ).to(device)
            targets = torch.stack(
                [
                    data[start + 1 : start + model.config.block_size + 1]
                    for start in batch_starts
                ]
            ).to(device)
            logits, _ = model(inputs, targets)
            summed_nll = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            total_nll += float(summed_nll.detach().cpu())
            total_tokens += targets.numel()
            top1_correct += int(
                logits.argmax(dim=-1).eq(targets).sum().detach().cpu()
            )
    finally:
        model.train(was_training)
    loss = total_nll / total_tokens
    if not math.isfinite(loss):
        raise FloatingPointError("validation diagnostic loss is not finite")
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 80.0)),
        "top1_accuracy": top1_correct / total_tokens,
        "tokens_evaluated": total_tokens,
        "windows_evaluated": len(starts),
        "window_selection": "deterministic_evenly_spaced",
        "first_start": starts[0],
        "last_start": starts[-1],
    }


@torch.inference_mode()
def generate_continuation(
    model: GPTLanguageModelV4,
    tokenizer: BPETokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    max_characters: int,
    temperature: float,
    top_k: int,
    generator: torch.Generator,
    device: torch.device,
) -> GenerationResult:
    if max_new_tokens <= 0 or max_characters <= 0:
        raise ValueError("generation limits must be positive")
    if temperature <= 0 or top_k <= 0:
        raise ValueError("temperature and top_k must be positive")
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("generation prompt cannot be empty")
    eos_id = tokenizer.special_to_id.get("<EOS>")
    if eos_id is None:
        raise ValueError("tokenizer does not define <EOS>")
    all_ids = list(prompt_ids)
    generated: list[int] = []
    stop_reason = "max_new_tokens"
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            context = torch.tensor(
                [all_ids[-model.config.block_size :]],
                dtype=torch.long,
                device=device,
            )
            logits, _ = model(context)
            scores = logits[0, -1].float().cpu() / temperature
            # Base pretraining may use EOS at chapter boundaries, but chat-only
            # and padding sentinels must never leak into visible generations.
            for special_id in tokenizer.special_to_id.values():
                if special_id != eos_id:
                    scores[int(special_id)] = float("-inf")
            candidate_count = min(top_k, scores.numel())
            values, indices = torch.topk(scores, candidate_count)
            sampled = torch.multinomial(
                torch.softmax(values, dim=-1),
                1,
                generator=generator,
            )
            next_id = int(indices[sampled].item())
            if next_id == eos_id:
                stop_reason = "eos"
                break
            all_ids.append(next_id)
            generated.append(next_id)
            visible = tokenizer.decode(generated, skip_special_tokens=True)
            if len(visible) >= max_characters:
                stop_reason = "max_characters"
                break
    finally:
        model.train(was_training)
    continuation = tokenizer.decode(generated, skip_special_tokens=True)[:max_characters]
    return GenerationResult(
        continuation=continuation,
        generated_token_ids=generated,
        stop_reason=stop_reason,
        eos_emitted=stop_reason == "eos",
    )


def generation_diagnostics(text: str) -> dict[str, Any]:
    compact = "".join(text.split())
    unique_ratio = len(set(compact)) / len(compact) if compact else 0.0
    four_gram = ngram_repetition(compact, 4)
    character_run = longest_character_run(compact)
    flags = {
        "empty": not compact,
        "high_four_gram_repetition": four_gram
        > DEGENERATION_THRESHOLDS["maximum_four_gram_repetition"],
        "long_character_run": character_run
        > DEGENERATION_THRESHOLDS["maximum_character_run"],
        "very_low_character_diversity": (
            len(compact)
            >= DEGENERATION_THRESHOLDS["minimum_characters_for_unique_ratio"]
            and unique_ratio
            < DEGENERATION_THRESHOLDS["minimum_unique_character_ratio"]
        ),
    }
    return {
        "characters": len(text),
        "content_characters": len(compact),
        "unique_character_ratio": unique_ratio,
        "four_gram_repetition": four_gram,
        "longest_character_run": character_run,
        "degeneration_flags": flags,
        "mechanically_degenerate": any(flags.values()),
    }


def summarize_generations(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("at least one generation sample is required")
    return {
        "sample_count": len(samples),
        "mean_characters": mean(float(row["characters"]) for row in samples),
        "mean_four_gram_repetition": mean(
            float(row["four_gram_repetition"]) for row in samples
        ),
        "maximum_character_run": max(
            int(row["longest_character_run"]) for row in samples
        ),
        "eos_stop_rate": mean(bool(row["eos_emitted"]) for row in samples),
        "empty_rate": mean(
            bool(row["degeneration_flags"]["empty"]) for row in samples
        ),
        "mechanical_degeneration_rate": mean(
            bool(row["mechanically_degenerate"]) for row in samples
        ),
        "stop_reason_counts": {
            reason: sum(row["stop_reason"] == reason for row in samples)
            for reason in ("eos", "max_characters", "max_new_tokens")
        },
    }


def validate_cloze_cases(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in cases:
        case_id = str(raw.get("id", "")).strip()
        context = str(raw.get("context", "")).strip()
        candidates = [str(value).strip() for value in raw.get("candidates", [])]
        correct = str(raw.get("correct", "")).strip()
        if not case_id or case_id in seen_ids:
            raise ValueError("cloze case IDs must be non-empty and unique")
        if not context:
            raise ValueError(f"cloze case {case_id!r} has an empty context")
        if len(candidates) < 2 or len(set(candidates)) != len(candidates):
            raise ValueError(f"cloze case {case_id!r} needs distinct candidates")
        if correct not in candidates:
            raise ValueError(f"cloze case {case_id!r} correct answer is not a candidate")
        normalized.append(
            {
                **dict(raw),
                "id": case_id,
                "context": context,
                "candidates": candidates,
                "correct": correct,
            }
        )
        seen_ids.add(case_id)
    if not normalized:
        raise ValueError("at least one cloze case is required")
    return normalized


def load_cloze_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return validate_cloze_cases(DEFAULT_CLOZE_CASES)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list):
        raise ValueError("cloze file must be a JSON list or an object with a cases list")
    return validate_cloze_cases(raw_cases)


def _expected_prompt_file_bytes(prompts: Sequence[str]) -> bytes:
    return ("\n".join(prompts) + "\n").encode("utf-8")


def validate_probe_artifact_for_evaluation(
    artifact_path: Path,
    prompts_path: Path,
    token_manifest: Mapping[str, Any],
    *,
    require_formal_declarations: bool = False,
) -> dict[str, Any]:
    """Validate the immutable validation-probe artifact before formal use.

    Only validation probes are returned.  Train probes remain calibration
    metadata inside the artifact and are never scored as formal evidence here.
    """

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != "pretrain-capability-probes/v1":
        raise ValueError("--cloze must use pretrain-capability-probes/v1")
    usage = artifact.get("usage")
    if not isinstance(usage, Mapping):
        raise ValueError("probe artifact is missing usage policy")
    usage_checks = {
        "training_allowed_false": usage.get("training_allowed") is False,
        "sft_false": usage.get("sft_information_used") is False,
        "test_false": usage.get("test_split_read") is False,
    }
    if not all(usage_checks.values()):
        raise ValueError(
            "probe usage must declare training_allowed=false, "
            "sft_information_used=false, and test_split_read=false"
        )
    validation = artifact.get("validation")
    if not isinstance(validation, Mapping) or validation.get("passed") is not True:
        raise ValueError("probe artifact validation.passed must be true")
    if validation.get("failures") not in (None, []):
        raise ValueError("probe artifact contains validation failures")

    build_inputs = artifact.get("build", {}).get("inputs", {})
    for split in ("train", "val"):
        artifact_sha = build_inputs.get(split, {}).get("sha256")
        manifest_sha = token_manifest.get("splits", {}).get(split, {}).get(
            "text_sha256"
        )
        if not isinstance(manifest_sha, str) or artifact_sha != manifest_sha:
            raise ValueError(
                f"probe build.inputs.{split}.sha256 does not match token manifest"
            )

    probes = artifact.get("probes")
    cases = artifact.get("cases")
    continuation_prompts = artifact.get("continuation_prompts")
    if not isinstance(probes, list) or not isinstance(cases, list):
        raise ValueError("probe artifact must contain probes and cases lists")
    if require_formal_declarations and any(
        probe.get("source", {}).get("split") == "test"
        for probe in probes
        if isinstance(probe, Mapping)
    ):
        raise ValueError("formal probe artifact must not contain any test-split probe")
    if not isinstance(continuation_prompts, list) or not all(
        isinstance(prompt, str) and prompt for prompt in continuation_prompts
    ):
        raise ValueError("probe artifact continuation_prompts must be non-empty strings")

    validation_cloze = [
        probe
        for probe in probes
        if probe.get("probe_type") == "cloze_candidate_ranking"
        and probe.get("source", {}).get("split") == "val"
    ]
    expected_cases = [
        {
            "id": probe.get("id"),
            "context": probe.get("prompt"),
            "candidates": [
                candidate.get("text") for candidate in probe.get("candidates", [])
            ],
            "correct": probe.get("expected", {}).get("text"),
        }
        for probe in validation_cloze
    ]
    compact_cases = [
        {
            "id": case.get("id"),
            "context": case.get("context"),
            "candidates": case.get("candidates"),
            "correct": case.get("correct"),
        }
        for case in cases
        if isinstance(case, Mapping)
    ]
    if compact_cases != expected_cases:
        raise ValueError(
            "top-level cases must exactly correspond to validation cloze probes"
        )
    declared_cases_sha256 = artifact.get("evaluator_compatibility", {}).get(
        "cases_canonical_sha256"
    )
    if require_formal_declarations and not (
        isinstance(declared_cases_sha256, str)
        and len(declared_cases_sha256) == 64
        and all(character in "0123456789abcdef" for character in declared_cases_sha256)
    ):
        raise ValueError(
            "formal probe artifact requires a valid declared cases_canonical_sha256"
        )
    if declared_cases_sha256 is not None and (
        canonical_json_sha256({"cases": cases}) != declared_cases_sha256
    ):
        # The builder hashes the list itself in some artifact revisions.
        cases_payload_sha256 = hashlib.sha256(
            json.dumps(
                cases,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if cases_payload_sha256 != declared_cases_sha256:
            raise ValueError("declared cases_canonical_sha256 does not match cases")

    validation_continuations = [
        probe
        for probe in probes
        if probe.get("probe_type") == "held_out_continuation"
        and probe.get("source", {}).get("split") == "val"
    ]
    expected_prompts = [probe.get("prompt") for probe in validation_continuations]
    if continuation_prompts != expected_prompts:
        raise ValueError(
            "continuation_prompts must exactly correspond to validation probes"
        )
    expected_prompt_bytes = _expected_prompt_file_bytes(continuation_prompts)
    actual_prompt_bytes = prompts_path.read_bytes()
    expected_prompts_sha256 = hashlib.sha256(expected_prompt_bytes).hexdigest()
    actual_prompts_sha256 = hashlib.sha256(actual_prompt_bytes).hexdigest()
    declared_prompts_sha256 = (
        artifact.get("prompts_sha256")
        or artifact.get("evaluator_compatibility", {}).get("prompts_sha256")
        or artifact.get("evaluator_compatibility", {}).get(
            "prompts_content_sha256"
        )
        or artifact.get("build", {}).get("outputs", {}).get("prompts_sha256")
    )
    if require_formal_declarations and not (
        isinstance(declared_prompts_sha256, str)
        and len(declared_prompts_sha256) == 64
        and all(character in "0123456789abcdef" for character in declared_prompts_sha256)
    ):
        raise ValueError(
            "formal probe artifact requires a valid declared prompts_content_sha256"
        )
    if actual_prompt_bytes != expected_prompt_bytes:
        raise ValueError(
            "prompts file content does not exactly match artifact continuation_prompts"
        )
    if actual_prompts_sha256 != expected_prompts_sha256:
        raise ValueError("prompts_sha256 does not match continuation_prompts")
    if declared_prompts_sha256 is not None and (
        declared_prompts_sha256 != actual_prompts_sha256
    ):
        raise ValueError("declared prompts_sha256 does not match prompts file")

    enriched_cases: list[dict[str, Any]] = []
    for compact, probe in zip(cases, validation_cloze):
        entity = dict(probe.get("entity", {}))
        case_metadata = dict(compact.get("metadata", {}))
        enriched_cases.append(
            {
                **compact,
                "frequency_tier": case_metadata.get(
                    "frequency_tier", entity.get("frequency_tier", "unknown")
                ),
                "capability": probe.get("capability"),
                "probe_type": probe.get("probe_type"),
                "entity": entity,
                "source": dict(probe.get("source", {})),
                "evidence": dict(probe.get("evidence", {})),
                "expected": dict(probe.get("expected", {})),
                "candidate_metadata": [
                    dict(candidate) for candidate in probe.get("candidates", [])
                ],
                "probe_metadata": dict(probe),
            }
        )
    validate_cloze_cases(enriched_cases)
    return {
        "mode": "validated_artifact",
        "formal_status_eligible": True,
        "artifact_path": str(artifact_path),
        "artifact_sha256": file_sha256(artifact_path),
        "schema_version": artifact["schema_version"],
        "usage_checks": usage_checks,
        "validation_passed": True,
        "build_input_sha256_matches": {"train": True, "val": True},
        "prompts_path": str(prompts_path),
        "prompts_sha256": actual_prompts_sha256,
        "declared_prompts_sha256": declared_prompts_sha256,
        "declared_cases_canonical_sha256": declared_cases_sha256,
        "formal_declarations_required": require_formal_declarations,
        "cases": enriched_cases,
        "continuation_prompts": list(continuation_prompts),
        "probe_count": len(probes),
        "validation_cloze_count": len(validation_cloze),
        "validation_continuation_count": len(validation_continuations),
    }


@torch.inference_mode()
def _score_candidate_under_context(
    model: GPTLanguageModelV4,
    tokenizer: BPETokenizer,
    context: str,
    candidate: str,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    """Return candidate token IDs and their conditional log probabilities."""

    prefix_ids = tokenizer.encode(context)
    candidate_ids = tokenizer.encode(candidate)
    if not prefix_ids or not candidate_ids:
        raise ValueError("cloze context and candidate must encode to at least one token")
    if len(candidate_ids) >= model.config.block_size:
        raise ValueError("cloze candidate is too long for the model context window")
    running_ids = list(prefix_ids)
    token_log_probabilities: list[float] = []
    was_training = model.training
    model.eval()
    try:
        for token_id in candidate_ids:
            inputs = torch.tensor(
                [running_ids[-model.config.block_size :]],
                dtype=torch.long,
                device=device,
            )
            logits, _ = model(inputs)
            log_probabilities = functional.log_softmax(logits[0, -1].float(), dim=-1)
            token_log_probabilities.append(float(log_probabilities[token_id].cpu()))
            running_ids.append(token_id)
    finally:
        model.train(was_training)
    return candidate_ids, token_log_probabilities


def score_candidate(
    model: GPTLanguageModelV4,
    tokenizer: BPETokenizer,
    context: str,
    candidate: str,
    device: torch.device,
    *,
    neutral_context: str = "\n",
) -> dict[str, Any]:
    """Score one candidate with raw and neutral-prior-corrected diagnostics."""

    candidate_ids, token_log_probabilities = _score_candidate_under_context(
        model, tokenizer, context, candidate, device
    )
    neutral_ids, neutral_log_probabilities = _score_candidate_under_context(
        model, tokenizer, neutral_context, candidate, device
    )
    if candidate_ids != neutral_ids:
        raise AssertionError("candidate tokenization changed across scoring contexts")
    total = sum(token_log_probabilities)
    neutral_total = sum(neutral_log_probabilities)
    character_count = len(candidate)
    if character_count <= 0:
        raise ValueError("cloze candidate cannot be empty")
    return {
        "candidate": candidate,
        "token_count": len(candidate_ids),
        "character_count": character_count,
        "total_log_probability": total,
        "mean_token_log_probability": total / len(candidate_ids),
        "per_character_log_probability": total / character_count,
        "neutral_total_log_probability": neutral_total,
        "neutral_mean_token_log_probability": neutral_total / len(candidate_ids),
        "neutral_per_character_log_probability": neutral_total / character_count,
        "context_lift": total - neutral_total,
        "context_lift_mean_token": (total - neutral_total) / len(candidate_ids),
        "context_lift_per_character": (total - neutral_total) / character_count,
    }


RANKING_METRICS = (
    "total_log_probability",
    "mean_token_log_probability",
    "per_character_log_probability",
    "context_lift",
)


def _ranking_for_metric(
    scores: Sequence[Mapping[str, Any]],
    correct: str,
    metric: str,
) -> dict[str, Any]:
    ordered = sorted(scores, key=lambda row: float(row[metric]), reverse=True)
    correct_rank = next(
        rank
        for rank, row in enumerate(ordered, start=1)
        if row["candidate"] == correct
    )
    correct_score = next(
        float(row[metric]) for row in ordered if row["candidate"] == correct
    )
    best_incorrect = max(
        float(row[metric]) for row in ordered if row["candidate"] != correct
    )
    return {
        "predicted": ordered[0]["candidate"],
        "correct_rank": correct_rank,
        "top1_correct": correct_rank == 1,
        "reciprocal_rank": 1.0 / correct_rank,
        "correct_margin_over_best_incorrect": correct_score - best_incorrect,
        "ordered_candidates": [row["candidate"] for row in ordered],
    }


def _aggregate_rankings(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "top1_accuracy": mean(
                bool(row["rankings"][metric]["top1_correct"]) for row in rows
            ),
            "mean_reciprocal_rank": mean(
                float(row["rankings"][metric]["reciprocal_rank"]) for row in rows
            ),
            "mean_correct_margin": mean(
                float(
                    row["rankings"][metric][
                        "correct_margin_over_best_incorrect"
                    ]
                )
                for row in rows
            ),
        }
        for metric in RANKING_METRICS
    }


def evaluate_cloze(
    model: GPTLanguageModelV4,
    tokenizer: BPETokenizer,
    cases: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in validate_cloze_cases(cases):
        scores = [
            score_candidate(model, tokenizer, case["context"], candidate, device)
            for candidate in case["candidates"]
        ]
        rankings = {
            metric: _ranking_for_metric(scores, case["correct"], metric)
            for metric in RANKING_METRICS
        }
        primary = rankings["mean_token_log_probability"]
        results.append(
            {
                **case,
                "scores": scores,
                "rankings": rankings,
                "predicted": primary["predicted"],
                "correct_rank": primary["correct_rank"],
                "top1_correct": primary["top1_correct"],
                "correct_margin_over_best_incorrect": primary[
                    "correct_margin_over_best_incorrect"
                ],
            }
        )
    metrics = _aggregate_rankings(results)
    tier_names = sorted({str(row.get("frequency_tier", "unknown")) for row in results})
    by_frequency_tier = {}
    for tier in tier_names:
        tier_rows = [
            row for row in results if str(row.get("frequency_tier", "unknown")) == tier
        ]
        by_frequency_tier[tier] = {
            "case_count": len(tier_rows),
            "metrics": _aggregate_rankings(tier_rows),
        }
    primary_metrics = metrics["mean_token_log_probability"]
    return {
        "case_count": len(results),
        "neutral_context": "\n",
        "ranking_metrics": list(RANKING_METRICS),
        "formulae": {
            "total_log_probability": (
                "sum_i log P(candidate_token_i | validation_prefix, "
                "prior_candidate_tokens)"
            ),
            "mean_token_log_probability": (
                "total_log_probability / candidate_token_count"
            ),
            "per_character_log_probability": (
                "total_log_probability / candidate_unicode_character_count"
            ),
            "context_lift": (
                "total_log_probability(validation_prefix) - "
                "total_log_probability(neutral_newline_prefix)"
            ),
        },
        "metrics": metrics,
        "by_frequency_tier": by_frequency_tier,
        "primary_diagnostic_metric": "mean_token_log_probability",
        "top1_accuracy": primary_metrics["top1_accuracy"],
        "mean_reciprocal_rank": primary_metrics["mean_reciprocal_rank"],
        "mean_correct_margin": primary_metrics["mean_correct_margin"],
        "score_policy": (
            "validation-prefix ranking reports total, mean-token, per-character, "
            "and neutral-prior-corrected context lift; higher is better"
        ),
        "validation_prefix_ranking_not_entity_knowledge_hard_gate": True,
        "diagnostic_not_qa_hard_gate": True,
        "cases": results,
    }


def validate_pretraining_provenance(
    checkpoint: Mapping[str, Any],
    *,
    expected_config_sha256: str,
    expected_model_config: Mapping[str, Any],
    expected_token_manifest_sha256: str,
    require_initial_checkpoint_none: bool = False,
) -> dict[str, Any]:
    """Reject post-training checkpoints and incompatible base-model artifacts."""

    if checkpoint.get("config_sha256") != expected_config_sha256:
        raise ValueError(
            "checkpoint is not from the supplied pure-pretraining config "
            "(configuration signature mismatch)"
        )
    extra = checkpoint.get("extra")
    if not isinstance(extra, Mapping):
        raise ValueError("checkpoint has no pretraining provenance metadata")
    if extra.get("model_config") != dict(expected_model_config):
        raise ValueError("checkpoint model configuration does not match pretraining config")
    if extra.get("token_manifest_sha256") != expected_token_manifest_sha256:
        raise ValueError(
            "checkpoint token manifest does not match the current token_manifest.json"
        )
    sft_markers = sorted(
        {"payload_summary", "sampler_state", "data_sha256"}.intersection(extra)
    )
    if sft_markers:
        raise ValueError(
            "checkpoint contains post-training metadata and is not a pure "
            f"pretraining checkpoint: {', '.join(sft_markers)}"
        )
    if require_initial_checkpoint_none:
        if "initial_checkpoint" not in extra or extra.get("initial_checkpoint") is not None:
            raise ValueError(
                "formal audit requires explicit from-scratch pretraining provenance; "
                "initial_checkpoint must be present and null"
            )
    return {
        "config_signature_matches": True,
        "model_config_matches": True,
        "token_manifest_matches": True,
        "token_manifest_sha256": expected_token_manifest_sha256,
        "post_training_markers_absent": True,
        "formal_initial_checkpoint_check": (
            extra.get("initial_checkpoint") is None
            if require_initial_checkpoint_none
            else "not_required"
        ),
        "initial_checkpoint": extra.get("initial_checkpoint"),
    }


def _read_prompts(path: Path, limit: int) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    prompts = [prompt for prompt in prompts if prompt][:limit]
    if not prompts:
        raise ValueError("fixed prompt file contains no prompts")
    return prompts


def _load_and_verify_data(
    data_dir: Path,
    split: str,
    loggers: Mapping[str, logging.Logger],
) -> tuple[BPETokenizer, torch.Tensor, dict[str, Any], Path, Path]:
    manifest_path = data_dir / "token_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer_path = data_dir / "tokenizer.json"
    tokenizer_hash = file_sha256(tokenizer_path)
    if manifest.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError("tokenizer checksum does not match token manifest")
    split_meta = manifest.get("splits", {}).get(split)
    if not isinstance(split_meta, dict):
        raise ValueError(f"token manifest has no {split!r} split")
    tensor_path = data_dir / f"{split}_tokens.pt"
    tensor_hash = file_sha256(tensor_path)
    if split_meta.get("tensor_sha256") != tensor_hash:
        raise ValueError(f"{split} tensor checksum does not match token manifest")
    tokenizer = BPETokenizer.load(tokenizer_path)
    data = load_tensor(tensor_path)
    loggers["data"].info(
        "verified evaluation tensor and tokenizer",
        extra={
            "context": {
                "split": split,
                "tokens": len(data),
                "vocab_size": tokenizer.vocab_size,
                "tokenizer_sha256": tokenizer_hash,
                "tensor_sha256": tensor_hash,
            }
        },
    )
    return tokenizer, data, manifest, tokenizer_path, tensor_path


def build_markdown_report(report: Mapping[str, Any]) -> str:
    validation_diagnostic = report["validation_diagnostic"]
    summary = report["generation_summary"]
    cloze = report["cloze"]
    probe_provenance = report.get("probe_provenance", {})
    lines = [
        "# 纯预训练模型能力审计",
        "",
        "> 这是基础语言模型（Base LM）的审计，不是聊天模型考试。它学习的是小说中的下一个 Token；问答格式不属于本阶段硬门。",
        "",
        f"- Checkpoint：`{report['checkpoint']['path']}`（Step {report['checkpoint']['step']}）",
        f"- 设备：`{report['device']}`",
        f"- 参数量：{report['model']['parameter_count']:,}",
        f"- Validation diagnostic split：`{validation_diagnostic['split']}`",
        f"- Validation diagnostic Loss：{validation_diagnostic['loss']:.6f}",
        f"- Token Perplexity：{validation_diagnostic['perplexity']:.3f}",
        f"- Top-1 next-token accuracy：{validation_diagnostic['top1_accuracy']:.3%}",
        f"- Probe 来源：`{probe_provenance.get('mode', 'unknown')}`",
        f"- Probe SHA-256：`{probe_provenance.get('artifact_sha256') or 'handcrafted / none'}`",
        "",
        "## 固定提示续写",
        "",
        f"EOS 停止率为 {summary['eos_stop_rate']:.1%}；机械退化率为 {summary['mechanical_degeneration_rate']:.1%}。EOS 率只描述章节边界行为，不单独决定模型好坏。",
        "",
        "| 提示 | 续写 | 停止原因 | 4-gram 重复率 | 最长单字连写 |",
        "|---|---|---|---:|---:|",
    ]
    for sample in report["generations"]:
        prompt = sample["prompt"].replace("|", "\\|").replace("\n", "↵")
        continuation = sample["continuation"].replace("|", "\\|").replace("\n", "↵")
        lines.append(
            f"| {prompt} | {continuation} | {sample['stop_reason']} | "
            f"{sample['four_gram_repetition']:.3f} | {sample['longest_character_run']} |"
        )
    lines.extend(
        [
            "",
            "## Validation-prefix Cloze / 候选排序",
            "",
            "这里同时报告候选总对数概率、平均 Token、平均字符以及相对固定换行 neutral context 的 context lift。它只是 validation-prefix ranking，不是聊天问答准确率，也不能冒充“模型已经掌握实体知识”的硬门。",
            "",
            f"Top-1：{cloze['top1_accuracy']:.1%}；MRR：{cloze['mean_reciprocal_rank']:.3f}",
            "",
            "| 声明式前缀 | 正确候选 | 模型首选 | 正确名次 |",
            "|---|---|---|---:|",
        ]
    )
    for case in cloze["cases"]:
        lines.append(
            f"| {case['context'].replace('|', '\\|')} | {case['correct']} | "
            f"{case['predicted']} | {case['correct_rank']} |"
        )
    if cloze.get("by_frequency_tier"):
        lines.extend(
            [
                "",
                "### 按训练语料频率分层",
                "",
                "| Frequency tier | 数量 | Mean-token Top-1 | Context-lift Top-1 |",
                "|---|---:|---:|---:|",
            ]
        )
        for tier, tier_report in cloze["by_frequency_tier"].items():
            tier_metrics = tier_report["metrics"]
            lines.append(
                f"| {tier} | {tier_report['case_count']} | "
                f"{tier_metrics['mean_token_log_probability']['top1_accuracy']:.1%} | "
                f"{tier_metrics['context_lift']['top1_accuracy']:.1%} |"
            )
    lines.extend(
        [
            "",
            "## 解释与人工验收",
            "",
            "- 自动指标可发现 Validation Loss、复读、空输出、单字连写和候选排序问题，但不能独立证明文笔流畅、情节正确或实体知识已经形成。",
            "- 人工应逐条检查：语句流畅、局部连贯、承接提示、人物与世界观一致、是否大段背诵训练原文。",
            "- 本报告不以“能否回答通用问题”验收预训练；领域问答、格式遵循和边界拒答属于后续 SFT/RAG 阶段。",
            "",
            f"审计状态：**{report['status']}**",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit_outputs(
    report: Mapping[str, Any],
    json_path: Path,
    markdown_path: Path,
    logger: logging.Logger | None = None,
) -> None:
    atomic_write_json(json_path, report)
    atomic_write_text(markdown_path, build_markdown_report(report))
    if logger is not None:
        logger.info(
            "audit artifacts written",
            extra={
                "context": {
                    "json_path": str(json_path),
                    "markdown_path": str(markdown_path),
                    "status": report["status"],
                }
            },
        )


def run_audit(
    args: argparse.Namespace,
    run_id: str,
    loggers: Mapping[str, logging.Logger],
) -> dict[str, Any]:
    config = load_config(args.config)
    config_hash = canonical_json_sha256(config)
    data_dir = args.data_dir or Path(config["data_dir"])
    checkpoint_path = args.checkpoint or Path(config["run_dir"]) / "best.pt"
    device = select_device(args.device)
    tokenizer, validation_data, manifest, tokenizer_path, tensor_path = (
        _load_and_verify_data(data_dir, args.held_out_split, loggers)
    )
    token_manifest_path = data_dir / "token_manifest.json"
    token_manifest_sha256 = file_sha256(token_manifest_path)
    model_config = GPTConfig(vocab_size=tokenizer.vocab_size, **config["model"])
    model = GPTLanguageModelV4(model_config)
    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location="cpu",
        expected_config_sha256=config_hash,
    )
    provenance = validate_pretraining_provenance(
        checkpoint,
        expected_config_sha256=config_hash,
        expected_model_config=model_config.to_dict(),
        expected_token_manifest_sha256=token_manifest_sha256,
        require_initial_checkpoint_none=args.formal,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    loggers["checkpoint"].info(
        "verified pure pretraining checkpoint",
        extra={
            "context": {
                "path": str(checkpoint_path),
                "step": int(checkpoint["step"]),
                "sha256": file_sha256(checkpoint_path),
                "config_signature_matches": True,
                "token_manifest_matches": True,
                "post_training_markers_absent": True,
                "formal_from_scratch_required": args.formal,
            }
        },
    )

    validation_diagnostic = evaluate_held_out(
        model,
        validation_data,
        window_count=args.eval_windows,
        batch_size=args.eval_batch_size,
        device=device,
    )
    validation_diagnostic["split"] = args.held_out_split
    validation_diagnostic["tensor_path"] = str(tensor_path)
    validation_diagnostic["tensor_sha256"] = file_sha256(tensor_path)
    validation_diagnostic["interpretation"] = (
        "validation diagnostic, not an untouched final-test estimate"
        if args.held_out_split == "val"
        else "explicitly authorized sealed-test diagnostic"
    )
    loggers["validation"].info(
        "validation next-token diagnostic complete",
        extra={"context": validation_diagnostic},
    )

    if args.cloze is not None:
        probe_provenance = validate_probe_artifact_for_evaluation(
            args.cloze,
            args.prompts,
            manifest,
            require_formal_declarations=args.formal,
        )
        cloze_cases = probe_provenance.pop("cases")
        artifact_prompts = probe_provenance.pop("continuation_prompts")
        prompts = (
            artifact_prompts
            if args.formal
            else artifact_prompts[: args.prompt_limit]
        )
    else:
        probe_provenance = {
            "mode": "handcrafted",
            "formal_status_eligible": False,
            "artifact_path": None,
            "artifact_sha256": None,
            "note": (
                "built-in handcrafted diagnostics; results cannot receive formal status"
            ),
        }
        cloze_cases = load_cloze_cases(None)
        prompts = _read_prompts(args.prompts, args.prompt_limit)
    loggers["data"].info(
        "capability probe provenance verified",
        extra={
            "context": {
                "mode": probe_provenance["mode"],
                "formal_status_eligible": probe_provenance[
                    "formal_status_eligible"
                ],
                "artifact_sha256": probe_provenance.get("artifact_sha256"),
                "prompt_count": len(prompts),
                "cloze_count": len(cloze_cases),
            }
        },
    )
    generations: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        generator = torch.Generator().manual_seed(args.seed + 1000 + index)
        generated = generate_continuation(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            max_characters=args.max_characters,
            temperature=args.temperature,
            top_k=args.top_k,
            generator=generator,
            device=device,
        )
        measured = {
            "prompt_index": index + 1,
            "prompt": prompt,
            "continuation": generated.continuation,
            "generated_tokens": len(generated.generated_token_ids),
            "stop_reason": generated.stop_reason,
            "eos_emitted": generated.eos_emitted,
            **generation_diagnostics(generated.continuation),
        }
        generations.append(measured)
        loggers["generation"].info(
            "fixed-prompt continuation generated",
            extra={
                "context": {
                    "prompt_index": index + 1,
                    "prompt_characters": len(prompt),
                    "generated_characters": measured["characters"],
                    "stop_reason": measured["stop_reason"],
                    "mechanically_degenerate": measured["mechanically_degenerate"],
                }
            },
        )
    generation_summary = summarize_generations(generations)

    cloze = evaluate_cloze(model, tokenizer, cloze_cases, device)
    loggers["cloze"].info(
        "declarative cloze candidate ranking complete",
        extra={
            "context": {
                "case_count": cloze["case_count"],
                "top1_accuracy": cloze["top1_accuracy"],
                "mean_reciprocal_rank": cloze["mean_reciprocal_rank"],
                "validation_prefix_ranking_not_entity_knowledge_hard_gate": True,
            }
        },
    )

    mechanical_checks = {
        "pure_pretraining_provenance": all(
            provenance[key]
            for key in (
                "config_signature_matches",
                "model_config_matches",
                "token_manifest_matches",
                "post_training_markers_absent",
            )
        ),
        "finite_validation_diagnostic_loss": math.isfinite(
            float(validation_diagnostic["loss"])
        ),
        "all_fixed_prompts_generated": len(generations) == len(prompts),
        "no_empty_generation": generation_summary["empty_rate"] == 0.0,
        "no_mechanical_degeneration": (
            generation_summary["mechanical_degeneration_rate"] == 0.0
        ),
    }
    if not args.formal:
        status = "DIAGNOSTIC_ONLY_NOT_FORMAL"
    elif all(mechanical_checks.values()):
        status = "FORMAL_AUDIT_COMPLETE_MANUAL_REVIEW_REQUIRED"
    else:
        status = "FORMAL_MECHANICAL_REVIEW_REQUIRED"
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "scope": {
            "model_type": "base_causal_language_model",
            "training_domain": "single_novel",
            "base_lm_is_not_a_chat_model": True,
            "qa_format_is_not_a_pretraining_hard_gate": True,
            "automatic_metrics_are_not_semantic_quality_scores": True,
            "formal_requested": args.formal,
            "formal_status_eligible": bool(
                args.formal and probe_provenance["formal_status_eligible"]
            ),
        },
        "device": str(device),
        "config": {
            "path": str(args.config),
            "canonical_sha256": config_hash,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "step": int(checkpoint["step"]),
            "provenance": provenance,
        },
        "model": {
            "config": model_config.to_dict(),
            "parameter_count": model.parameter_count(),
        },
        "data": {
            "manifest_path": str(token_manifest_path),
            "manifest_sha256": token_manifest_sha256,
            "manifest_schema": manifest.get("schema_version"),
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_sha256": file_sha256(tokenizer_path),
        },
        "validation_diagnostic": validation_diagnostic,
        "probe_provenance": probe_provenance,
        "generation_settings": {
            "seed": args.seed,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "max_characters": args.max_characters,
            "prompt_path": str(args.prompts),
            "prompt_count": len(prompts),
            "prompt_limit_policy": (
                "all_validated_artifact_prompts"
                if args.formal
                else f"first_{args.prompt_limit}"
            ),
        },
        "degeneration_thresholds": DEGENERATION_THRESHOLDS,
        "generation_summary": generation_summary,
        "generations": generations,
        "cloze": cloze,
        "mechanical_checks": mechanical_checks,
        "interpretation": {
            "automatic_metrics_cover": [
                "validation next-token diagnostic",
                "empty output",
                "character and n-gram repetition",
                "EOS stopping behaviour",
                "validation-prefix candidate ranking",
            ],
            "manual_review_dimensions": [
                "fluency",
                "local coherence",
                "prompt continuation relevance",
                "character and world consistency",
                "verbatim memorization risk",
            ],
            "excluded_as_pretraining_hard_gates": [
                "chat question answering",
                "general world knowledge",
                "instruction following",
                "out-of-domain refusal",
                "validation-prefix ranking as proof of stored entity knowledge",
            ],
        },
        "logging": {
            "directory": str(args.log_dir),
            "modules": list(LOG_MODULES),
            "environment_override": "GPT_AUDIT_LOG_LEVEL_<MODULE>",
            "rotation_max_bytes": args.log_max_bytes,
            "rotation_backup_count": args.log_backup_count,
            "sensitive_fields_redacted": True,
            "production_default": "INFO",
        },
    }
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/formal_pretrain_14m_bpe3000.json"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--held-out-split",
        choices=("val", "test"),
        default="val",
        help="Evaluation tensor; val is a validation diagnostic, test stays sealed by default.",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicitly authorize reading the sealed test tensor.",
    )
    parser.add_argument(
        "--formal",
        action="store_true",
        help="Require a validated pretrain-capability-probes/v1 artifact and from-scratch checkpoint.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--eval-windows", type=int, default=60)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--prompts", type=Path, default=Path("data/story_prompt5_eval.txt"))
    parser.add_argument("--prompt-limit", type=int, default=5)
    parser.add_argument("--cloze", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-characters", type=int, default=180)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/pretrain_capability_audit/audit.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("reports/pretrain_capability_audit/audit.md"),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("reports/pretrain_capability_audit/logs"),
    )
    parser.add_argument(
        "--log-level",
        action="append",
        default=[],
        metavar="MODULE=LEVEL",
        help="Override one audit log module; LEVEL may be DEBUG/INFO/WARNING/ERROR/OFF.",
    )
    parser.add_argument("--log-max-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--log-backup-count", type=int, default=3)
    parser.add_argument("--no-console-log", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "eval_windows": args.eval_windows,
        "eval_batch_size": args.eval_batch_size,
        "prompt_limit": args.prompt_limit,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "max_characters": args.max_characters,
        "log_max_bytes": args.log_max_bytes,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"positive values required: {', '.join(invalid)}")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.log_backup_count < 0:
        raise ValueError("log_backup_count cannot be negative")
    if args.held_out_split == "test" and not args.allow_test:
        raise ValueError(
            "test split is sealed; pass --allow-test together with "
            "--held-out-split test to authorize access"
        )
    if args.formal and args.cloze is None:
        raise ValueError(
            "--formal requires --cloze with a validated "
            "pretrain-capability-probes/v1 artifact"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = generate_run_id("pretrain-capability-audit")
    loggers: dict[str, logging.Logger] = {}
    try:
        levels = resolve_log_levels(args.log_level)
        loggers = configure_audit_loggers(
            args.log_dir,
            run_id,
            levels,
            max_bytes=args.log_max_bytes,
            backup_count=args.log_backup_count,
            console=not args.no_console_log,
        )
        validate_args(args)
        loggers["orchestrator"].info(
            "pretraining capability audit started",
            extra={
                "context": {
                    "config": str(args.config),
                    "checkpoint": str(args.checkpoint) if args.checkpoint else "config default",
                    "device_request": args.device,
                    "held_out_split": args.held_out_split,
                    "test_access_explicitly_allowed": args.allow_test,
                    "formal_requested": args.formal,
                }
            },
        )
        report = run_audit(args, run_id, loggers)
        write_audit_outputs(
            report,
            args.output_json,
            args.output_markdown,
            loggers["orchestrator"],
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "run_id": run_id,
                    "output_json": str(args.output_json),
                    "output_markdown": str(args.output_markdown),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as error:
        if loggers:
            loggers["orchestrator"].exception(
                "pretraining capability audit failed",
                extra={
                    "context": {
                        "operation": "run_audit",
                        "error_type": type(error).__name__,
                        "config": str(args.config),
                        "checkpoint": str(args.checkpoint) if args.checkpoint else "config default",
                    }
                },
            )
        print(f"pretraining capability audit failed: {error}", file=sys.stderr)
        return 1
    finally:
        close_audit_loggers(loggers)


if __name__ == "__main__":
    raise SystemExit(main())
