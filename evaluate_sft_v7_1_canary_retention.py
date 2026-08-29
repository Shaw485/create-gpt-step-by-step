"""Evaluate pretraining-capability retention after the M021 Canary SFT run.

The audit intentionally reuses the frozen M019 *validation-only* comparison:
the same 60 fixed windows, BPC definition, and 16 continuation prompts.  It has
no CLI input for pretraining test data or any SFT public/sealed split.  Canary
lineage, tokenizer, base Step 5750, effective configuration, tensor and dataset
manifest are checksum-bound before model weights are evaluated.

Runtime logs are separate rotating JSONL streams for ``data``, ``checkpoint``,
``validation``, ``generation`` and ``orchestrator``.  Each module can be set
with ``GPT_CANARY_RETENTION_LOG_LEVEL_<MODULE>`` or ``--log-level
MODULE=LEVEL``.  Logs contain only hashes, counts and aggregate metrics; prompt
and continuation bodies are retained only in the JSON/Markdown reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import torch

from evaluate_pretrain_capabilities import (
    evaluate_held_out,
    generate_continuation,
    generation_diagnostics,
    summarize_generations,
)
from evaluate_sft_v7_retention import (
    BASE_CHECKPOINT_SHA256,
    BASE_CONFIG_CANONICAL_SHA256,
    BASE_PARAMETER_COUNT,
    BASE_STEP,
    BASE_TOKEN_MANIFEST_SHA256,
    EXPECTED_BASELINE_AUDIT_SHA256,
    EXPECTED_BASELINE_FIXED_WINDOW_LOSS,
    EXPECTED_CONFIG_MODEL,
    EXPECTED_PROMPT_COUNT,
    EXPECTED_PROMPTS_SHA256,
    EXPECTED_PROBE_SHA256,
    EXPECTED_RAW_VALIDATION_SHA256,
    EXPECTED_VALIDATION_CHARACTERS,
    EXPECTED_VALIDATION_TENSOR_SHA256,
    EXPECTED_VALIDATION_TOKENS,
    EXPECTED_VALIDATION_WINDOWS,
    EXPECTED_WINDOW_TOKENS,
    GENERATION_SETTINGS,
    TOKENIZER_SHA256,
    assert_publishable_paths,
    fixed_window_bpc,
    normalize_artifact_paths,
    portable_artifact_path,
    validate_baseline_reference,
    validate_pretraining_inputs,
)
from gpt_model_v4 import GPTConfig, GPTLanguageModelV4
from train_pretrain_v4 import load_config
from training_runtime import (
    DEFAULT_LOG_MODULES,
    atomic_write_json,
    atomic_write_text,
    canonical_json_sha256,
    close_module_loggers,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
    load_checkpoint,
)
from prepare_sft_v7_1_canary import (
    DEFAULT_EVAL as DEFAULT_CANARY_DEVELOPMENT,
    DEFAULT_TRAIN as DEFAULT_CANARY_TRAIN,
    load_and_validate_canary_manifest,
)
from train_sft_v7_1_canary import load_canary_tensor_payload


SCHEMA_VERSION = "sft-v7.1-canary-pretrain-retention/v1"
DEFAULT_CONFIG = Path("configs/formal_pretrain_14m_bpe3000.json")
DEFAULT_CHECKPOINT = Path("runs/sft_v7_1_canary/latest.pt")
DEFAULT_EFFECTIVE_CONFIG = Path("runs/sft_v7_1_canary/effective_config.json")
DEFAULT_CANARY_TENSOR = Path("data/sft/v7_1_canary/train_eval_tensors.pt")
DEFAULT_CANARY_MANIFEST = Path("data/sft/v7_1_canary/manifest.json")
DEFAULT_DATA_DIR = Path("data/scaling_a/bpe_3000")
DEFAULT_RAW_VALIDATION = Path("data/cloud_v4/val.txt")
DEFAULT_PROBES = Path("data/eval/pretrain_capability_probes.json")
DEFAULT_PROMPTS = Path("data/eval/pretrain_capability_prompts.txt")
DEFAULT_BASELINE_AUDIT = Path(
    "reports/milestones/019_pretrain_capability_audit/step_05750/audit.json"
)
DEFAULT_OUTPUT_JSON = Path(
    "reports/milestones/021_sft_v7_1_canary/pretrain_retention.json"
)
DEFAULT_OUTPUT_MARKDOWN = Path(
    "reports/milestones/021_sft_v7_1_canary/pretrain_retention.md"
)
DEFAULT_LOG_DIR = Path("logs/sft_v7_1_canary_retention")

EXPECTED_CANARY_TENSOR_SCHEMA = "sft-v7.1-canary-tensors/v1"
EXPECTED_CANARY_MANIFEST_SCHEMA = "sft-v7.1-canary-manifest/v1"
EXPECTED_TRAIN_COUNT = 64
EXPECTED_HOLDOUT_COUNT = 16
EXPECTED_BASE_CHECKPOINT_PATH = (
    "runs/formal_pretrain_14m_bpe3000_6000/checkpoints/step_05750.pt"
)
RETENTION_THRESHOLDS = {
    "maximum_relative_fixed_window_bpc_degradation": 0.05,
    "required_nonempty_continuations": 16,
    "maximum_mechanical_degeneration_rate": 0.25,
}

LOG_MODULES = (
    "data",
    "checkpoint",
    "validation",
    "generation",
    "orchestrator",
)
_ALL_LOG_MODULES = tuple(dict.fromkeys((*DEFAULT_LOG_MODULES, *LOG_MODULES)))
_LEVEL_NAMES = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "OFF"}
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


class CanaryRetentionError(ValueError):
    """A stable, log-safe retention failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str) -> None:
    raise CanaryRetentionError(code, message)


def _read_json_object(path: Path, code: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CanaryRetentionError(code, "cannot read required JSON artifact") from error
    if not isinstance(payload, dict):
        _fail(code, "JSON artifact root must be an object")
    return payload


def resolve_log_levels(overrides: Sequence[str] = ()) -> dict[str, str]:
    """Resolve production-safe, independently switchable module levels."""

    levels = {
        module: os.getenv(
            f"GPT_CANARY_RETENTION_LOG_LEVEL_{module.upper()}",
            "INFO" if module in LOG_MODULES else "OFF",
        ).upper()
        for module in _ALL_LOG_MODULES
    }
    for module, level in levels.items():
        if level not in _LEVEL_NAMES:
            raise ValueError(f"unknown log level for {module}: {level}")
    for override in overrides:
        if "=" not in override:
            raise ValueError("--log-level must use MODULE=LEVEL")
        module, level = (part.strip() for part in override.split("=", 1))
        level = level.upper()
        if module not in _ALL_LOG_MODULES:
            raise ValueError(f"unknown Canary retention log module: {module}")
        if level not in _LEVEL_NAMES:
            raise ValueError(f"unknown log level: {level}")
        levels[module] = level
    return levels


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = (
            "mps"
            if torch.backends.mps.is_built() and torch.backends.mps.is_available()
            else "cpu"
        )
    if requested == "mps":
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device: {requested}")


def _verify_sidecar(path: Path) -> str:
    sidecar = Path(f"{path}.sha256")
    if not sidecar.is_file():
        _fail("checkpoint_sidecar_missing", "checkpoint checksum sidecar is missing")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if not fields or not _HEX_SHA256.fullmatch(fields[0].lower()):
        _fail("checkpoint_sidecar_malformed", "checkpoint checksum sidecar is malformed")
    actual = file_sha256(path)
    if actual != fields[0].lower():
        _fail("checkpoint_sidecar_mismatch", "checkpoint checksum does not match sidecar")
    return actual


def _signature_payload(effective: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact fields that identify one reviewed Canary run."""

    required = ("schema_version", "model", "provenance", "training", "schedule")
    missing = [key for key in required if key not in effective]
    if missing:
        _fail("effective_config_incomplete", "effective Canary config is incomplete")
    return {key: effective[key] for key in required}


def _load_canary_tensor_metadata(path: Path) -> dict[str, Any]:
    """Load the reviewed Canary tensor bundle without inspecting text bodies."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise CanaryRetentionError(
            "canary_tensor_invalid", "cannot load Canary tensor artifact"
        ) from error
    if not isinstance(payload, Mapping):
        _fail("canary_tensor_invalid", "Canary tensor root must be a mapping")
    if payload.get("schema_version") != EXPECTED_CANARY_TENSOR_SCHEMA:
        _fail("canary_tensor_schema_mismatch", "Canary tensor schema changed")
    train_records = payload.get("train_records")
    eval_records = payload.get("eval_records")
    if not isinstance(train_records, Sequence) or len(train_records) != EXPECTED_TRAIN_COUNT:
        _fail("canary_train_count_mismatch", "Canary tensor train count changed")
    if not isinstance(eval_records, Sequence) or len(eval_records) != EXPECTED_HOLDOUT_COUNT:
        _fail("canary_holdout_count_mismatch", "Canary tensor holdout count changed")
    if payload.get("tokenizer_sha256") != TOKENIZER_SHA256:
        _fail("canary_tokenizer_sha_mismatch", "Canary tensor tokenizer changed")
    if payload.get("bpe_token_manifest_sha256") != BASE_TOKEN_MANIFEST_SHA256:
        _fail("canary_token_manifest_sha_mismatch", "Canary token manifest changed")
    required_base = payload.get("required_base_checkpoint")
    if not isinstance(required_base, Mapping):
        _fail("canary_base_binding_missing", "Canary tensor lacks base binding")
    expected_base_without_binding = {
        "path": EXPECTED_BASE_CHECKPOINT_PATH,
        "sha256": BASE_CHECKPOINT_SHA256,
        "step": BASE_STEP,
        "parameter_count": BASE_PARAMETER_COUNT,
        "config_canonical_sha256": BASE_CONFIG_CANONICAL_SHA256,
        "token_manifest_sha256": BASE_TOKEN_MANIFEST_SHA256,
    }
    expected_base = {
        **expected_base_without_binding,
        "binding_sha256": canonical_json_sha256(expected_base_without_binding),
    }
    for key, expected in expected_base.items():
        if required_base.get(key) != expected:
            _fail("canary_base_binding_mismatch", f"Canary base binding mismatch: {key}")
    return {
        "schema_version": payload["schema_version"],
        "split_counts": {
            "train": len(train_records),
            "holdout_eval": len(eval_records),
        },
        "tokenizer_sha256": payload["tokenizer_sha256"],
        "bpe_token_manifest_sha256": payload["bpe_token_manifest_sha256"],
        "canary_dataset_manifest_sha256": payload.get(
            "canary_dataset_manifest_sha256"
        ),
        "canary_dataset_identity_sha256": payload.get(
            "canary_dataset_identity_sha256"
        ),
        "artifact_binding_sha256": payload.get("artifact_binding_sha256"),
        "required_base_checkpoint": dict(required_base),
        "record_bodies_logged": False,
    }


def validate_canary_lineage(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    effective_config: Mapping[str, Any],
    effective_config_path: Path,
    canary_tensor_path: Path,
    canary_manifest_path: Path,
) -> dict[str, Any]:
    """Bind a candidate strictly to the 64-row M021 Canary training run."""

    checkpoint_sha = _verify_sidecar(checkpoint_path)
    if checkpoint.get("schema_version") != "training-checkpoint/v1":
        _fail("checkpoint_schema_mismatch", "checkpoint schema changed")
    if not isinstance(checkpoint.get("model_state_dict"), Mapping):
        _fail("checkpoint_model_state_missing", "checkpoint lacks model weights")
    step = int(checkpoint.get("step", -1))
    if step <= 0:
        _fail("checkpoint_step_invalid", "Canary checkpoint step must be positive")
    checkpoint_signature = str(checkpoint.get("config_sha256", ""))
    if not _HEX_SHA256.fullmatch(checkpoint_signature):
        _fail("checkpoint_signature_invalid", "checkpoint signature is invalid")
    extra = checkpoint.get("extra")
    if not isinstance(extra, Mapping):
        _fail("checkpoint_provenance_missing", "checkpoint lacks Canary provenance")

    if effective_config.get("schema_version") != (
        "sft-v7.1-canary-training-signature/v1"
    ):
        _fail("effective_config_schema_mismatch", "Canary signature schema changed")
    if effective_config.get("model") != EXPECTED_CONFIG_MODEL:
        _fail("model_config_mismatch", "Canary checkpoint model config changed")
    signature = canonical_json_sha256(_signature_payload(effective_config))
    declared_signature = str(effective_config.get("signature_sha256", ""))
    if signature != declared_signature or checkpoint_signature != declared_signature:
        _fail(
            "training_signature_mismatch",
            "checkpoint and effective Canary signatures do not agree",
        )
    provenance = effective_config.get("provenance")
    if not isinstance(provenance, Mapping):
        _fail("effective_provenance_missing", "effective config lacks provenance")
    for key, value in provenance.items():
        if extra.get(key) != value:
            _fail("checkpoint_provenance_mismatch", f"provenance mismatch: {key}")

    expected = {
        "stage": "sft_v7_1_canary",
        "base_checkpoint_path": EXPECTED_BASE_CHECKPOINT_PATH,
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "base_checkpoint_step": BASE_STEP,
        "base_config_canonical_sha256": BASE_CONFIG_CANONICAL_SHA256,
        "base_token_manifest_sha256": BASE_TOKEN_MANIFEST_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "holdout_records_consumed": 0,
        "public_records_consumed": 0,
        "sealed_records_consumed": 0,
    }
    for key, expected_value in expected.items():
        if extra.get(key) != expected_value:
            _fail("checkpoint_provenance_mismatch", f"provenance mismatch: {key}")

    development_required = {
        "development_unseen_wording_records": EXPECTED_HOLDOUT_COUNT,
        "development_records_consumed_for_teacher_loss": EXPECTED_HOLDOUT_COUNT,
        "development_records_used_for_checkpoint_selection": EXPECTED_HOLDOUT_COUNT,
    }
    development_field_names = set(development_required) | {
        "development_optimizer_records",
        "development_records_used_for_optimization",
    }
    present_development_fields = development_field_names.intersection(extra)
    if present_development_fields and (
        not set(development_required).issubset(extra)
        or (
            "development_records_used_for_optimization" not in extra
            and "development_optimizer_records" not in extra
        )
    ):
        _fail(
            "checkpoint_provenance_mismatch",
            "development provenance is incomplete",
        )
    for key in set(development_required).intersection(extra):
        if extra.get(key) != development_required[key]:
            _fail("checkpoint_provenance_mismatch", f"provenance mismatch: {key}")
    if "development_optimizer_records" in extra and extra.get(
        "development_optimizer_records"
    ) != 0:
        _fail(
            "checkpoint_provenance_mismatch",
            "provenance mismatch: development_optimizer_records",
        )
    if "development_records_used_for_optimization" in extra and extra.get(
        "development_records_used_for_optimization"
    ) != 0:
        _fail(
            "checkpoint_provenance_mismatch",
            "provenance mismatch: development_records_used_for_optimization",
        )

    tensor_sha = file_sha256(canary_tensor_path)
    if extra.get("canary_tensor_sha256") != tensor_sha:
        _fail("canary_tensor_sha_mismatch", "checkpoint Canary tensor SHA changed")
    if str(extra.get("canary_tensor_path", "")) != str(DEFAULT_CANARY_TENSOR):
        _fail("canary_tensor_path_mismatch", "checkpoint Canary tensor path changed")
    tensor_meta = _load_canary_tensor_metadata(canary_tensor_path)

    manifest_sha = file_sha256(canary_manifest_path)
    if extra.get("canary_dataset_manifest_sha256") != manifest_sha:
        _fail("canary_manifest_sha_mismatch", "checkpoint Canary manifest SHA changed")
    manifest = _read_json_object(canary_manifest_path, "canary_manifest_invalid")
    if manifest.get("manifest_schema_version") != EXPECTED_CANARY_MANIFEST_SCHEMA:
        _fail("canary_manifest_schema_mismatch", "Canary manifest schema changed")
    if manifest.get("status") != "frozen_canary_ready":
        _fail("canary_manifest_status_mismatch", "Canary manifest is not frozen")
    if manifest.get("split_totals") != {
        "train": EXPECTED_TRAIN_COUNT,
        "holdout_eval": EXPECTED_HOLDOUT_COUNT,
    }:
        _fail("canary_manifest_split_mismatch", "Canary manifest counts changed")
    if manifest.get("dataset_identity_sha256") != tensor_meta[
        "canary_dataset_identity_sha256"
    ]:
        _fail("canary_dataset_identity_mismatch", "Canary dataset identity changed")
    if tensor_meta["canary_dataset_manifest_sha256"] != manifest_sha:
        _fail("canary_tensor_manifest_binding_mismatch", "tensor/manifest binding changed")
    access = manifest.get("access_audit")
    if not isinstance(access, Mapping) or any(bool(value) for value in access.values()):
        _fail("canary_access_scope_mismatch", "Canary source access scope changed")

    payload_summary = extra.get("payload_summary")
    if not isinstance(payload_summary, Mapping) or payload_summary.get(
        "split_counts"
    ) != {"train": EXPECTED_TRAIN_COUNT, "holdout_eval": EXPECTED_HOLDOUT_COUNT}:
        _fail("checkpoint_split_count_mismatch", "checkpoint split counts changed")
    training = effective_config.get("training")
    schedule = effective_config.get("schedule")
    if not isinstance(training, Mapping) or not isinstance(schedule, Mapping):
        _fail("training_contract_missing", "effective config lacks training contract")
    target_steps = int(training.get("target_steps", -1))
    if target_steps <= 0 or step > target_steps:
        _fail("checkpoint_target_step_mismatch", "checkpoint step exceeds target")
    if training.get("sampler") != "deterministic_shuffled_epoch/v1":
        _fail("sampler_mismatch", "effective config sampler changed")
    if extra.get("learning_rate_schedule") != schedule:
        _fail("schedule_mismatch", "checkpoint learning-rate schedule changed")
    if not isinstance(extra.get("sampler_state"), Mapping):
        _fail("sampler_state_missing", "checkpoint lacks deterministic sampler state")

    return {
        "verified": True,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_sidecar_verified": True,
        "step": step,
        "target_steps": target_steps,
        "effective_config_path": str(effective_config_path),
        "effective_config_sha256": file_sha256(effective_config_path),
        "training_signature_sha256": declared_signature,
        "base_checkpoint_path": EXPECTED_BASE_CHECKPOINT_PATH,
        "base_checkpoint_step": BASE_STEP,
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "canary_tensor_path": str(canary_tensor_path),
        "canary_tensor_sha256": tensor_sha,
        "canary_manifest_path": str(canary_manifest_path),
        "canary_manifest_sha256": manifest_sha,
        "canary_dataset_identity_sha256": manifest["dataset_identity_sha256"],
        "training_split_counts": {"train": EXPECTED_TRAIN_COUNT},
        "holdout_eval_count": EXPECTED_HOLDOUT_COUNT,
        "holdout_records_consumed": 0,
        **development_required,
        "development_records_used_for_optimization": 0,
        "development_optimizer_records": 0,
        "development_split_role": "unseen_question_development_selection",
        "development_provenance_inferred_from_legacy_checkpoint": not bool(
            present_development_fields
        ),
        "public_records_consumed": 0,
        "sealed_records_consumed": 0,
        "pretraining_test_body_reads": 0,
        "blind_body_reads": 0,
    }


def compare_bpc(candidate_bpc: float, baseline_bpc: float) -> dict[str, Any]:
    """Compare same-window BPC using the Canary-specific five-percent gate."""

    if not all(math.isfinite(value) and value > 0 for value in (candidate_bpc, baseline_bpc)):
        raise ValueError("candidate and baseline BPC must be finite and positive")
    absolute_delta = candidate_bpc - baseline_bpc
    relative = absolute_delta / baseline_bpc
    threshold = RETENTION_THRESHOLDS[
        "maximum_relative_fixed_window_bpc_degradation"
    ]
    return {
        "candidate_bpc": candidate_bpc,
        "baseline_bpc": baseline_bpc,
        "absolute_delta_candidate_minus_baseline": absolute_delta,
        "relative_degradation_candidate_minus_baseline": relative,
        "maximum_allowed_relative_degradation": threshold,
        "passed": relative <= threshold or math.isclose(
            relative, threshold, rel_tol=0.0, abs_tol=1e-12
        ),
        "lower_is_better": True,
    }


def build_automatic_gates(
    *,
    comparison: Mapping[str, Any],
    generation_summary: Mapping[str, Any],
    generation_count: int,
) -> dict[str, Any]:
    nonempty_count = generation_count - round(
        float(generation_summary["empty_rate"]) * generation_count
    )
    gates = [
        {
            "name": "relative_fixed_window_bpc_degradation",
            "observed": comparison["relative_degradation_candidate_minus_baseline"],
            "operator": "<=",
            "threshold": RETENTION_THRESHOLDS[
                "maximum_relative_fixed_window_bpc_degradation"
            ],
            "passed": bool(comparison["passed"]),
        },
        {
            "name": "all_16_continuations_generated",
            "observed": generation_count,
            "operator": "==",
            "threshold": EXPECTED_PROMPT_COUNT,
            "passed": generation_count == EXPECTED_PROMPT_COUNT,
        },
        {
            "name": "all_16_continuations_nonempty",
            "observed": nonempty_count,
            "operator": "==",
            "threshold": RETENTION_THRESHOLDS["required_nonempty_continuations"],
            "passed": nonempty_count
            == RETENTION_THRESHOLDS["required_nonempty_continuations"],
        },
        {
            "name": "mechanical_degeneration_rate",
            "observed": generation_summary["mechanical_degeneration_rate"],
            "operator": "<=",
            "threshold": RETENTION_THRESHOLDS[
                "maximum_mechanical_degeneration_rate"
            ],
            "passed": float(generation_summary["mechanical_degeneration_rate"])
            <= RETENTION_THRESHOLDS["maximum_mechanical_degeneration_rate"],
        },
    ]
    return {
        "gates": gates,
        "passed": all(bool(gate["passed"]) for gate in gates),
        "gate_count": len(gates),
        "failed_gate_names": [gate["name"] for gate in gates if not gate["passed"]],
    }


def safe_generation_log_context(
    *, prompt_index: int, prompt: str, measured: Mapping[str, Any]
) -> dict[str, Any]:
    """Produce useful generation diagnostics without logging any text body."""

    return {
        "prompt_index": prompt_index,
        "prompt_characters": len(prompt),
        "generated_characters": measured["characters"],
        "generated_tokens": measured["generated_tokens"],
        "stop_reason": measured["stop_reason"],
        "eos_emitted": measured["eos_emitted"],
        "four_gram_repetition": measured["four_gram_repetition"],
        "mechanically_degenerate": measured["mechanically_degenerate"],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    """Render the complete samples only into the explicit report artifact."""

    report = normalize_artifact_paths(report)
    assert_publishable_paths(report)
    validation = report["validation_diagnostic"]
    comparison = report["bpc_comparison"]
    generation = report["generation_summary"]
    lines = [
        "# M021 Canary 预训练能力保持审计",
        "",
        "> 仅复用 M019 的 validation 固定窗口和 16 条 validation-origin 续写提示；预训练 test、SFT public、SFT sealed 正文读取均为 0。",
        "",
        f"- Canary checkpoint：`{report['checkpoint_lineage']['checkpoint_path']}`（Step {report['checkpoint_lineage']['step']}）",
        f"- 固定窗口 Loss：{validation['loss']:.6f}",
        f"- 固定窗口 BPC：{comparison['candidate_bpc']:.6f}",
        f"- Step 5750 基线 BPC：{comparison['baseline_bpc']:.6f}",
        f"- 相对恶化：{comparison['relative_degradation_candidate_minus_baseline']:+.2%}（硬门 ≤5%）",
        f"- 16 条非空率：{1.0 - generation['empty_rate']:.1%}",
        f"- EOS 停止：{generation['stop_reason_counts']['eos']}/16",
        f"- 长度截断：{generation['stop_reason_counts']['max_characters']}/16",
        f"- Token 上限截断：{generation['stop_reason_counts']['max_new_tokens']}/16",
        f"- 机械退化率：{generation['mechanical_degeneration_rate']:.1%}（硬门 ≤25%）",
        "",
        "## 自动硬门",
        "",
        "| 门 | 观测 | 规则 | 结果 |",
        "|---|---:|---:|---|",
    ]
    for gate in report["automatic_hard_gates"]["gates"]:
        observed = gate["observed"]
        observed_text = f"{observed:.6f}" if isinstance(observed, float) else str(observed)
        lines.append(
            f"| {gate['name']} | {observed_text} | {gate['operator']} {gate['threshold']} | "
            f"{'PASS' if gate['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## 固定 16 条续写（完整报告样本）",
            "",
            "| # | 提示 | 续写（最多 120 字） | 停止原因 | EOS | 4-gram 重复率 | 退化 |",
            "|---:|---|---|---|---|---:|---|",
        ]
    )
    for sample in report["generations"]:
        prompt = sample["prompt"].replace("|", "\\|").replace("\n", "↵")
        continuation = sample["continuation"].replace("|", "\\|").replace(
            "\n", "↵"
        )
        lines.append(
            f"| {sample['prompt_index']} | {prompt} | {continuation} | "
            f"{sample['stop_reason']} | {'是' if sample['eos_emitted'] else '否'} | "
            f"{sample['four_gram_repetition']:.3f} | "
            f"{'是' if sample['mechanically_degenerate'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- 本报告只回答 Canary SFT 是否保留预训练续写能力，不替代问答能力 Canary 门。",
            "- 日志不含提示、续写、正文或 Token ID；完整样本只保留在本报告及 JSON。",
            f"- 状态：**{report['status']}**。",
            "",
        ]
    )
    markdown = "\n".join(lines)
    assert_publishable_paths(markdown)
    return markdown


def run_retention_audit(
    args: argparse.Namespace,
    run_id: str,
    loggers: Mapping[str, logging.Logger],
) -> dict[str, Any]:
    """Run lineage, same-window BPC and fixed-prompt retention checks."""

    # Fail closed on the frozen Canary data lineage before loading model weights.
    load_and_validate_canary_manifest(
        args.canary_manifest,
        {
            "train": DEFAULT_CANARY_TRAIN,
            "holdout_eval": DEFAULT_CANARY_DEVELOPMENT,
        },
    )
    load_canary_tensor_payload(args.canary_tensor)
    config = load_config(args.config)
    device = select_device(args.device)
    effective = _read_json_object(args.effective_config, "effective_config_invalid")
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    lineage = validate_canary_lineage(
        checkpoint,
        checkpoint_path=args.checkpoint,
        effective_config=effective,
        effective_config_path=args.effective_config,
        canary_tensor_path=args.canary_tensor,
        canary_manifest_path=args.canary_manifest,
    )
    loggers["checkpoint"].info(
        "Canary lineage verified",
        extra={
            "context": {
                "checkpoint_sha256": lineage["checkpoint_sha256"],
                "step": lineage["step"],
                "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
                "canary_tensor_sha256": lineage["canary_tensor_sha256"],
                "canary_manifest_sha256": lineage["canary_manifest_sha256"],
                "development_optimizer_records": 0,
                "development_teacher_loss_records": EXPECTED_HOLDOUT_COUNT,
                "development_checkpoint_selection_records": EXPECTED_HOLDOUT_COUNT,
                "public_records_consumed": 0,
                "sealed_records_consumed": 0,
            }
        },
    )

    tokenizer, validation_data, _manifest, probe_bundle = validate_pretraining_inputs(
        config=config,
        data_dir=args.data_dir,
        raw_validation_path=args.raw_validation,
        probes_path=args.probes,
        prompts_path=args.prompts,
    )
    baseline_payload = _read_json_object(args.baseline_audit, "baseline_audit_invalid")
    baseline = validate_baseline_reference(baseline_payload, args.baseline_audit)
    prompts = list(probe_bundle["continuation_prompts"])
    if len(prompts) != EXPECTED_PROMPT_COUNT:
        _fail("prompt_count_mismatch", "fixed continuation prompt count changed")
    loggers["data"].info(
        "frozen validation-only evaluation inputs verified",
        extra={
            "context": {
                "raw_validation_sha256": EXPECTED_RAW_VALIDATION_SHA256,
                "validation_tensor_sha256": EXPECTED_VALIDATION_TENSOR_SHA256,
                "validation_tokens": EXPECTED_VALIDATION_TOKENS,
                "prompt_count": len(prompts),
                "probe_sha256": EXPECTED_PROBE_SHA256,
                "prompts_sha256": EXPECTED_PROMPTS_SHA256,
                "pretraining_test_body_reads": 0,
                "sft_public_body_reads": 0,
                "sft_sealed_body_reads": 0,
            }
        },
    )

    model_config = GPTConfig(vocab_size=tokenizer.vocab_size, **config["model"])
    model = GPTLanguageModelV4(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if model.parameter_count() != BASE_PARAMETER_COUNT:
        _fail("parameter_count_mismatch", "Canary model parameter count changed")
    model.to(device)

    validation = evaluate_held_out(
        model,
        validation_data,
        window_count=EXPECTED_VALIDATION_WINDOWS,
        batch_size=args.eval_batch_size,
        device=device,
    )
    if validation["windows_evaluated"] != EXPECTED_VALIDATION_WINDOWS:
        _fail("validation_window_count_mismatch", "fixed validation windows changed")
    if validation["tokens_evaluated"] != EXPECTED_WINDOW_TOKENS:
        _fail("validation_window_token_mismatch", "fixed validation token count changed")
    validation.update(
        {
            "split": "val",
            "tensor_path": str(args.data_dir / "val_tokens.pt"),
            "tensor_sha256": EXPECTED_VALIDATION_TENSOR_SHA256,
            "raw_validation_path": str(args.raw_validation),
            "raw_validation_sha256": EXPECTED_RAW_VALIDATION_SHA256,
            "fixed_window_bpc": fixed_window_bpc(
                validation["loss"],
                EXPECTED_VALIDATION_TOKENS,
                EXPECTED_VALIDATION_CHARACTERS,
            ),
            "bpc_definition": (
                "fixed-window mean token NLL * frozen full-validation token_count / "
                "full-validation character_count / ln(2)"
            ),
            "same_windows_as_m019": True,
        }
    )
    comparison = compare_bpc(
        validation["fixed_window_bpc"], baseline["fixed_window_bpc"]
    )
    loggers["validation"].info(
        "fixed-window retention computed",
        extra={
            "context": {
                "loss": validation["loss"],
                "fixed_window_bpc": validation["fixed_window_bpc"],
                "baseline_fixed_window_bpc": baseline["fixed_window_bpc"],
                "relative_bpc_degradation": comparison[
                    "relative_degradation_candidate_minus_baseline"
                ],
                "maximum_relative_degradation": 0.05,
                "passed": comparison["passed"],
            }
        },
    )

    generations: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        generator = torch.Generator().manual_seed(
            int(GENERATION_SETTINGS["seed"]) + 1000 + index
        )
        result = generate_continuation(
            model,
            tokenizer,
            prompt,
            max_new_tokens=int(GENERATION_SETTINGS["max_new_tokens"]),
            max_characters=int(GENERATION_SETTINGS["max_characters"]),
            temperature=float(GENERATION_SETTINGS["temperature"]),
            top_k=int(GENERATION_SETTINGS["top_k"]),
            generator=generator,
            device=device,
        )
        row = {
            "prompt_index": index + 1,
            "prompt": prompt,
            "continuation": result.continuation,
            "generated_tokens": len(result.generated_token_ids),
            "stop_reason": result.stop_reason,
            "eos_emitted": result.eos_emitted,
            **generation_diagnostics(result.continuation),
        }
        generations.append(row)
        loggers["generation"].info(
            "fixed validation continuation generated",
            extra={
                "context": safe_generation_log_context(
                    prompt_index=index + 1, prompt=prompt, measured=row
                )
            },
        )
    generation_summary = summarize_generations(generations)
    gates = build_automatic_gates(
        comparison=comparison,
        generation_summary=generation_summary,
        generation_count=len(generations),
    )
    status = "retention_passed" if gates["passed"] else "retention_failed"
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "retention_passed": gates["passed"],
        "device": str(device),
        "checkpoint_lineage": lineage,
        "model": {
            "config": model_config.to_dict(),
            "parameter_count": model.parameter_count(),
        },
        "data_scope": {
            "pretraining_validation_only": True,
            "raw_validation_path": str(args.raw_validation),
            "raw_validation_sha256": EXPECTED_RAW_VALIDATION_SHA256,
            "validation_tensor_sha256": EXPECTED_VALIDATION_TENSOR_SHA256,
            "pretraining_test_body_reads": 0,
            "sft_public_body_reads": 0,
            "sft_sealed_body_reads": 0,
            "blind_split_used_for_selection": False,
        },
        "baseline_reference": baseline,
        "validation_diagnostic": validation,
        "bpc_comparison": comparison,
        "probe_provenance": {
            key: value for key, value in probe_bundle.items() if key != "cases"
        },
        "generation_settings": {
            **GENERATION_SETTINGS,
            "prompt_path": str(args.prompts),
            "prompt_count": len(prompts),
            "same_as_m019": True,
        },
        "generation_summary": generation_summary,
        "generations": generations,
        "retention_thresholds": RETENTION_THRESHOLDS,
        "automatic_hard_gates": gates,
        "logging": {
            "directory": str(args.log_dir),
            "modules": list(LOG_MODULES),
            "environment_override": "GPT_CANARY_RETENTION_LOG_LEVEL_<MODULE>",
            "format": "rotating JSONL with UTC timestamp and run_id",
            "rotation_max_bytes": args.log_max_bytes,
            "rotation_backup_count": args.log_backup_count,
            "prompt_or_continuation_bodies_logged": False,
            "token_ids_logged": False,
            "sensitive_fields_redacted": True,
            "production_default": "INFO",
        },
    }
    normalized = normalize_artifact_paths(report)
    assert_publishable_paths(normalized)
    return normalized


def write_outputs(
    report: Mapping[str, Any], json_path: Path, markdown_path: Path
) -> dict[str, Any]:
    normalized = normalize_artifact_paths(report)
    assert_publishable_paths(normalized)
    markdown = build_markdown_report(normalized)
    atomic_write_json(json_path, normalized)
    atomic_write_text(markdown_path, markdown)
    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--effective-config", type=Path, default=DEFAULT_EFFECTIVE_CONFIG
    )
    parser.add_argument("--canary-tensor", type=Path, default=DEFAULT_CANARY_TENSOR)
    parser.add_argument(
        "--canary-manifest", type=Path, default=DEFAULT_CANARY_MANIFEST
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--raw-validation", type=Path, default=DEFAULT_RAW_VALIDATION
    )
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--baseline-audit", type=Path, default=DEFAULT_BASELINE_AUDIT
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--log-level", action="append", default=[], metavar="MODULE=LEVEL"
    )
    parser.add_argument("--log-max-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--log-backup-count", type=int, default=3)
    parser.add_argument("--no-console-log", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.eval_batch_size <= 0 or args.log_max_bytes <= 0:
        raise ValueError("eval_batch_size and log_max_bytes must be positive")
    if args.log_backup_count < 0:
        raise ValueError("log_backup_count cannot be negative")
    if args.canary_tensor.name != "train_eval_tensors.pt":
        raise ValueError("Canary tensor filename changed")
    if args.canary_manifest.name != "manifest.json":
        raise ValueError("Canary manifest filename changed")
    # No test/public/sealed CLI exists.  Also reject blind-looking aliases for
    # every evaluation body-bearing path so accidental substitution fails early.
    for label in ("raw_validation", "probes", "prompts"):
        path = Path(getattr(args, label))
        lowered = [part.lower() for part in path.parts]
        if any(
            "sealed" in part
            or "public" in part
            or part in {"test", "test.txt", "test_tokens.pt"}
            for part in lowered
        ):
            raise ValueError(f"{label} cannot point to test, public or sealed data")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = generate_run_id("sft-v7-1-canary-retention")
    loggers: dict[str, logging.Logger] = {}
    try:
        validate_args(args)
        levels = resolve_log_levels(args.log_level)
        loggers = configure_module_loggers(
            args.log_dir,
            run_id,
            levels,
            max_bytes=args.log_max_bytes,
            backup_count=args.log_backup_count,
            console=not args.no_console_log,
        )
        loggers["orchestrator"].info(
            "Canary retention audit started",
            extra={
                "context": {
                    "checkpoint": portable_artifact_path(
                        args.checkpoint, role="checkpoint"
                    ),
                    "device_request": args.device,
                    "evaluation_split": "val",
                    "fixed_window_count": EXPECTED_VALIDATION_WINDOWS,
                    "pretraining_test_body_reads": 0,
                    "sft_public_body_reads": 0,
                    "sft_sealed_body_reads": 0,
                }
            },
        )
        report = run_retention_audit(args, run_id, loggers)
        report = write_outputs(report, args.output_json, args.output_markdown)
        loggers["orchestrator"].info(
            "Canary retention artifacts written",
            extra={
                "context": {
                    "status": report["status"],
                    "hard_gates_passed": report["automatic_hard_gates"]["passed"],
                    "output_json": portable_artifact_path(
                        args.output_json, role="output-json"
                    ),
                    "output_markdown": portable_artifact_path(
                        args.output_markdown, role="output-markdown"
                    ),
                }
            },
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "checkpoint_step": report["checkpoint_lineage"]["step"],
                    "fixed_window_bpc": report["validation_diagnostic"][
                        "fixed_window_bpc"
                    ],
                    "relative_bpc_degradation": report["bpc_comparison"][
                        "relative_degradation_candidate_minus_baseline"
                    ],
                    "eos_count": report["generation_summary"][
                        "stop_reason_counts"
                    ]["eos"],
                    "truncated_count": report["generation_summary"][
                        "stop_reason_counts"
                    ]["max_characters"]
                    + report["generation_summary"]["stop_reason_counts"][
                        "max_new_tokens"
                    ],
                    "hard_gates_passed": report["automatic_hard_gates"]["passed"],
                    "output_json": portable_artifact_path(
                        args.output_json, role="output-json"
                    ),
                    "output_markdown": portable_artifact_path(
                        args.output_markdown, role="output-markdown"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as error:
        code = getattr(error, "code", "canary_retention_failed")
        if loggers:
            loggers["orchestrator"].error(
                "Canary retention audit failed",
                extra={
                    "context": {
                        "error_code": code,
                        "error_type": type(error).__name__,
                        "operation": "run_retention_audit",
                        "pretraining_test_body_reads": 0,
                        "sft_public_body_reads": 0,
                        "sft_sealed_body_reads": 0,
                    }
                },
            )
        print(
            f"Canary retention audit failed [{code}]: {type(error).__name__}",
            file=sys.stderr,
        )
        return 1
    finally:
        close_module_loggers(loggers)


if __name__ == "__main__":
    raise SystemExit(main())
