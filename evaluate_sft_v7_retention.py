"""Audit whether an SFT-v7 checkpoint retained the frozen base-LM capability.

This evaluator deliberately reuses the M019 validation-only contract:

* the exact BPE tokenizer, raw validation SHA and validation token tensor;
* the same 60 deterministic validation windows;
* the same 16 continuation prefixes and 12 validation cloze probes;
* the same generation seed, temperature, top-k and length limits.

It never exposes an option for the pretraining test tensor or the SFT sealed
split.  The SFT checkpoint is accepted only when its checksum, effective
training signature, frozen Step-5750 ancestry, train/val tensor identity and
dataset-manifest identity all agree.  Runtime logs contain only hashes, counts,
lengths and aggregate metrics.  Prompt, continuation and evidence bodies are
written to the requested report artifacts, never to logs.

Logs are separated into ``data``, ``checkpoint``, ``validation``,
``generation``, ``cloze`` and ``orchestrator`` JSONL streams.  Set an individual
module with ``GPT_SFT_RETENTION_LOG_LEVEL_<MODULE>`` or repeat
``--log-level MODULE=LEVEL``.  Logs rotate, carry a run ID and use the shared
sensitive-field redaction formatter.
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

from bpe_tokenizer import BPETokenizer
from evaluate_pretrain_capabilities import (
    evaluate_cloze,
    evaluate_held_out,
    generate_continuation,
    generation_diagnostics,
    summarize_generations,
    validate_probe_artifact_for_evaluation,
)
from gpt_model_v4 import GPTConfig, GPTLanguageModelV4
from train_pretrain_v4 import load_config, load_tensor
from train_sft_v7 import (
    BASE_CHECKPOINT_SHA256,
    BASE_CONFIG_CANONICAL_SHA256,
    BASE_PARAMETER_COUNT,
    BASE_STEP,
    BASE_TOKEN_MANIFEST_SHA256,
    EXPECTED_CONFIG_MODEL,
    PHASE_ORDER,
    TOKENIZER_SHA256,
    phase_for_next_update,
    schedule_contract,
)
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


SCHEMA_VERSION = "sft-v7-pretrain-retention/v1"
DEFAULT_CONFIG = Path("configs/formal_pretrain_14m_bpe3000.json")
DEFAULT_CHECKPOINT = Path("runs/sft_v7_vertical_2000/latest.pt")
DEFAULT_EFFECTIVE_CONFIG = Path("runs/sft_v7_vertical_2000/effective_config.json")
DEFAULT_SFT_TENSOR = Path("data/sft/v7/train_val_tensors.pt")
DEFAULT_SFT_MANIFEST = Path("data/sft/v7/manifest.json")
DEFAULT_DATA_DIR = Path("data/scaling_a/bpe_3000")
DEFAULT_RAW_VALIDATION = Path("data/cloud_v4/val.txt")
DEFAULT_PROBES = Path("data/eval/pretrain_capability_probes.json")
DEFAULT_PROMPTS = Path("data/eval/pretrain_capability_prompts.txt")
DEFAULT_BASELINE_AUDIT = Path(
    "reports/milestones/019_pretrain_capability_audit/step_05750/audit.json"
)
DEFAULT_OUTPUT_JSON = Path(
    "reports/milestones/020_sft_v7_vertical/pretrain_retention.json"
)
DEFAULT_OUTPUT_MARKDOWN = Path(
    "reports/milestones/020_sft_v7_vertical/pretrain_retention.md"
)
DEFAULT_LOG_DIR = Path("logs/sft_v7_retention")

EXPECTED_RAW_VALIDATION_SHA256 = (
    "f7ee1531a503001921b5fb767d655f04c393c42edfddf319d7eeca44655d2977"
)
EXPECTED_VALIDATION_TENSOR_SHA256 = (
    "f678203005119c241178130790ece08423717cbce0f8949926d3a79a1b18e328"
)
EXPECTED_PROBE_SHA256 = (
    "f95d594eaa8d08ef340a704c7b9103627ab949e88c1796fe3db0527cfc54f36e"
)
EXPECTED_PROMPTS_SHA256 = (
    "032fd93f12ffba3df3d4d9ad4bc1898a6a86ec416a812cd85a0c0f1928a253a7"
)
EXPECTED_BASELINE_AUDIT_SHA256 = (
    "274d6f8b2c32ed5e54987ff1ee8586b2e38fa1837b71fc25cf84ca2cfc02ab62"
)
EXPECTED_VALIDATION_CHARACTERS = 314_610
EXPECTED_VALIDATION_TOKENS = 184_003
EXPECTED_VALIDATION_WINDOWS = 60
EXPECTED_WINDOW_TOKENS = 30_720
EXPECTED_BASELINE_FIXED_WINDOW_LOSS = 4.438906606038412
EXPECTED_BASELINE_FULL_HISTORY_BPC = 3.761229497144957
EXPECTED_PROMPT_COUNT = 16
EXPECTED_CLOZE_COUNT = 12
GENERATION_SETTINGS = {
    "seed": 42,
    "temperature": 0.7,
    "top_k": 20,
    "max_new_tokens": 256,
    "max_characters": 120,
}
RETENTION_THRESHOLDS = {
    "maximum_relative_fixed_window_bpc_degradation": 0.10,
    "required_nonempty_continuations": 16,
    "maximum_mechanical_degeneration_rate": 0.25,
    "minimum_external_ai_fluency": 2.0,
    "minimum_external_ai_local_coherence": 2.0,
}

LOG_MODULES = (
    "data",
    "checkpoint",
    "validation",
    "generation",
    "cloze",
    "orchestrator",
)
_ALL_RUNTIME_LOG_MODULES = tuple(dict.fromkeys((*DEFAULT_LOG_MODULES, *LOG_MODULES)))
_LEVEL_NAMES = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "OFF"}
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
PROJECT_ROOT = Path(__file__).resolve().parent
_PATH_KEY_PATTERN = re.compile(
    r"(?:^|_)(?:path|paths|dir|directory|checkpoint|config|artifact)(?:$|_)",
    re.IGNORECASE,
)


class SFTV7RetentionError(ValueError):
    """Retention contract failure with a stable, log-safe error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str) -> None:
    raise SFTV7RetentionError(code, message)


def _read_json_object(path: Path, code: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SFTV7RetentionError(code, f"cannot read required JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        _fail(code, f"JSON artifact root must be an object: {path}")
    return payload


def _portable_role(role: str) -> str:
    normalized = re.sub(r"[^a-z0-9_-]+", "-", role.strip().lower()).strip("-")
    return normalized or "artifact"


def portable_artifact_path(
    value: str | Path,
    *,
    role: str,
    project_root: Path = PROJECT_ROOT,
) -> str:
    """Represent paths without publishing a workstation-specific absolute path."""

    raw = str(value)
    if raw.startswith("artifact://"):
        return raw
    candidate = Path(raw)
    if not candidate.is_absolute():
        return candidate.as_posix()
    resolved_root = project_root.resolve(strict=False)
    resolved_candidate = candidate.resolve(strict=False)
    try:
        return resolved_candidate.relative_to(resolved_root).as_posix()
    except ValueError:
        basename = candidate.name or "artifact"
        return f"artifact://{_portable_role(role)}/{basename}"


def normalize_artifact_paths(
    value: Any,
    *,
    role: str = "artifact",
    project_root: Path = PROJECT_ROOT,
) -> Any:
    """Recursively make report/log path values portable and publishable.

    Path objects are always normalized.  String values are normalized when
    their field name denotes a path, or when they contain the project root.
    Text bodies remain byte-for-byte unchanged.
    """

    if isinstance(value, Mapping):
        return {
            str(key): normalize_artifact_paths(
                nested,
                role=str(key),
                project_root=project_root,
            )
            for key, nested in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            normalize_artifact_paths(
                nested,
                role=role,
                project_root=project_root,
            )
            for nested in value
        ]
    if isinstance(value, Path):
        return portable_artifact_path(value, role=role, project_root=project_root)
    if isinstance(value, str):
        if value.startswith("artifact://"):
            return value
        project_prefix = project_root.resolve(strict=False).as_posix().rstrip("/")
        if value == project_prefix or value.startswith(project_prefix + "/"):
            return portable_artifact_path(value, role=role, project_root=project_root)
        if project_prefix in value:
            # Defensive handling for a path embedded in explanatory text.
            return value.replace(project_prefix + "/", "").replace(project_prefix, ".")
        if _PATH_KEY_PATTERN.search(role) and Path(value).is_absolute():
            return portable_artifact_path(value, role=role, project_root=project_root)
        if _PATH_KEY_PATTERN.search(role) and not value.startswith(("http://", "https://")):
            return Path(value).as_posix()
    return value


def assert_publishable_paths(
    value: Any,
    *,
    project_root: Path = PROJECT_ROOT,
) -> None:
    """Fail before writing if a report still exposes a local absolute path."""

    root_text = project_root.resolve(strict=False).as_posix()
    forbidden = (root_text, "/Users/")

    def visit(nested: Any) -> None:
        if isinstance(nested, Mapping):
            for item in nested.values():
                visit(item)
        elif isinstance(nested, (list, tuple)):
            for item in nested:
                visit(item)
        elif isinstance(nested, str) and any(marker in nested for marker in forbidden):
            raise ValueError("report contains a non-publishable absolute path")

    visit(value)


def resolve_log_levels(overrides: Sequence[str] = ()) -> dict[str, str]:
    """Resolve production-safe per-module log levels."""

    levels = {
        module: os.getenv(
            f"GPT_SFT_RETENTION_LOG_LEVEL_{module.upper()}",
            "INFO" if module in LOG_MODULES else "OFF",
        ).upper()
        for module in _ALL_RUNTIME_LOG_MODULES
    }
    for module, level in levels.items():
        if level not in _LEVEL_NAMES:
            raise ValueError(f"unknown log level for {module}: {level}")
    for override in overrides:
        if "=" not in override:
            raise ValueError("--log-level must use MODULE=LEVEL")
        module, level = (part.strip() for part in override.split("=", 1))
        level = level.upper()
        if module not in _ALL_RUNTIME_LOG_MODULES:
            raise ValueError(f"unknown retention log module: {module}")
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


def fixed_window_bpc(token_loss: float, token_count: int, character_count: int) -> float:
    """Convert token NLL to a fixed-window BPC using one frozen corpus ratio."""

    if not math.isfinite(token_loss) or token_loss < 0:
        raise ValueError("token_loss must be finite and non-negative")
    if token_count <= 0 or character_count <= 0:
        raise ValueError("token_count and character_count must be positive")
    return token_loss * token_count / character_count / math.log(2.0)


def compare_bpc(candidate_bpc: float, baseline_bpc: float) -> dict[str, Any]:
    """Return signed absolute and relative degradation; lower BPC is better."""

    if not all(math.isfinite(value) and value > 0 for value in (candidate_bpc, baseline_bpc)):
        raise ValueError("candidate and baseline BPC must be finite and positive")
    absolute_delta = candidate_bpc - baseline_bpc
    relative_degradation = absolute_delta / baseline_bpc
    threshold = RETENTION_THRESHOLDS[
        "maximum_relative_fixed_window_bpc_degradation"
    ]
    return {
        "candidate_bpc": candidate_bpc,
        "baseline_bpc": baseline_bpc,
        "absolute_delta_candidate_minus_baseline": absolute_delta,
        "relative_degradation_candidate_minus_baseline": relative_degradation,
        "maximum_allowed_relative_degradation": threshold,
        "passed": (
            relative_degradation <= threshold
            or math.isclose(relative_degradation, threshold, rel_tol=0.0, abs_tol=1e-12)
        ),
        "lower_is_better": True,
    }


def _signature_payload(effective: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "schema_version",
        "model",
        "provenance",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "betas",
        "gradient_clip",
        "sampling_schedule",
        "seed",
    )
    missing = [key for key in required if key not in effective]
    if missing:
        _fail("effective_config_incomplete", "effective SFT configuration is incomplete")
    return {key: effective[key] for key in required}


def _verify_checkpoint_sidecar(checkpoint_path: Path) -> str:
    sidecar = Path(f"{checkpoint_path}.sha256")
    if not sidecar.is_file():
        _fail("checkpoint_sidecar_missing", "checkpoint checksum sidecar is missing")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if not fields or not _HEX_SHA256.fullmatch(fields[0].lower()):
        _fail("checkpoint_sidecar_malformed", "checkpoint checksum sidecar is malformed")
    actual = file_sha256(checkpoint_path)
    if fields[0].lower() != actual:
        _fail("checkpoint_sidecar_mismatch", "checkpoint checksum does not match sidecar")
    return actual


def validate_sft_lineage(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    effective_config: Mapping[str, Any],
    effective_config_path: Path,
    sft_tensor_path: Path,
    sft_manifest_path: Path,
) -> dict[str, Any]:
    """Strictly bind one checkpoint to the reviewed SFT-v7 training lineage."""

    checkpoint_sha256 = _verify_checkpoint_sidecar(checkpoint_path)
    if checkpoint.get("schema_version") != "training-checkpoint/v1":
        _fail("checkpoint_schema_mismatch", "checkpoint is not a training-checkpoint/v1")
    if not isinstance(checkpoint.get("model_state_dict"), Mapping):
        _fail("checkpoint_model_state_missing", "checkpoint has no model state dictionary")
    step = int(checkpoint.get("step", -1))
    if step < 0:
        _fail("checkpoint_step_invalid", "checkpoint step is invalid")
    checkpoint_signature = str(checkpoint.get("config_sha256", ""))
    if not _HEX_SHA256.fullmatch(checkpoint_signature):
        _fail("checkpoint_signature_invalid", "checkpoint signature is invalid")
    extra = checkpoint.get("extra")
    if not isinstance(extra, Mapping):
        _fail("checkpoint_provenance_missing", "checkpoint lacks SFT provenance")

    signature = _signature_payload(effective_config)
    computed_signature = canonical_json_sha256(signature)
    declared_signature = str(effective_config.get("signature_sha256", ""))
    if computed_signature != declared_signature or checkpoint_signature != declared_signature:
        _fail(
            "training_signature_mismatch",
            "checkpoint and effective configuration signatures do not agree",
        )
    if effective_config.get("schema_version") != "sft-v7-training-signature/v2":
        _fail("training_signature_schema_mismatch", "SFT training signature schema changed")
    if effective_config.get("model") != EXPECTED_CONFIG_MODEL:
        _fail("model_config_mismatch", "SFT checkpoint does not use the frozen model")

    provenance = effective_config.get("provenance")
    if not isinstance(provenance, Mapping):
        _fail("effective_provenance_missing", "effective configuration lacks provenance")
    for key, value in provenance.items():
        if extra.get(key) != value:
            _fail("checkpoint_provenance_mismatch", f"checkpoint provenance mismatch: {key}")

    expected = {
        "stage": "sft_v7_vertical",
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "base_checkpoint_step": BASE_STEP,
        "base_config_canonical_sha256": BASE_CONFIG_CANONICAL_SHA256,
        "base_token_manifest_sha256": BASE_TOKEN_MANIFEST_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "public_records_consumed": 0,
        "sealed_records_consumed": 0,
    }
    for key, value in expected.items():
        if extra.get(key) != value:
            _fail("checkpoint_provenance_mismatch", f"checkpoint provenance mismatch: {key}")
    if Path(str(extra.get("base_checkpoint_path", ""))).as_posix() != (
        "runs/formal_pretrain_14m_bpe3000_6000/checkpoints/step_05750.pt"
    ):
        _fail("base_checkpoint_path_mismatch", "SFT ancestry does not name frozen Step 5750")

    actual_tensor_sha = file_sha256(sft_tensor_path)
    if extra.get("sft_tensor_sha256") != actual_tensor_sha:
        _fail("sft_tensor_sha_mismatch", "checkpoint training tensor SHA does not match")
    if Path(str(extra.get("sft_tensor_path", ""))).name != "train_val_tensors.pt":
        _fail("sft_tensor_path_mismatch", "checkpoint training tensor name changed")
    actual_manifest_sha = file_sha256(sft_manifest_path)
    if extra.get("sft_dataset_manifest_sha256") != actual_manifest_sha:
        _fail("sft_manifest_sha_mismatch", "checkpoint dataset manifest SHA does not match")

    manifest = _read_json_object(sft_manifest_path, "sft_manifest_invalid")
    if manifest.get("manifest_schema_version") != "sft-v7-vertical-manifest/v1":
        _fail("sft_manifest_schema_mismatch", "SFT dataset manifest schema changed")
    if manifest.get("split_totals") != {
        "train": 8000,
        "val": 800,
        "public_diagnostic": 600,
        "sealed_test": 600,
    }:
        _fail("sft_manifest_split_mismatch", "SFT dataset split totals changed")
    if manifest.get("frozen_status") != "frozen_unspent":
        _fail("sft_manifest_blind_status_mismatch", "blind dataset is not frozen and unspent")

    payload_summary = extra.get("payload_summary")
    if not isinstance(payload_summary, Mapping) or payload_summary.get("split_counts") != {
        "train": 8000,
        "val": 800,
    }:
        _fail("checkpoint_training_counts_mismatch", "checkpoint is not formal 8000/800 v7")
    schedule = effective_config.get("sampling_schedule")
    phase1_steps = int(effective_config.get("phase1_steps", -1))
    if phase1_steps <= 0 or schedule != schedule_contract(phase1_steps):
        _fail("sampling_schedule_mismatch", "checkpoint sampling schedule changed")
    if extra.get("sampling_schedule") != schedule:
        _fail("checkpoint_schedule_mismatch", "checkpoint schedule provenance changed")
    if int(extra.get("phase1_steps", -1)) != phase1_steps:
        _fail("checkpoint_phase_boundary_mismatch", "checkpoint phase boundary changed")
    expected_phase = phase_for_next_update(step, phase1_steps)
    if extra.get("current_phase") != expected_phase or expected_phase not in PHASE_ORDER:
        _fail("checkpoint_phase_mismatch", "checkpoint phase does not match completed step")
    target_steps = int(effective_config.get("target_steps", -1))
    if target_steps <= 0 or step > target_steps:
        _fail("checkpoint_target_step_mismatch", "checkpoint step exceeds declared target")
    if not isinstance(extra.get("sampler_states"), Mapping) or set(
        extra["sampler_states"]
    ) != set(PHASE_ORDER):
        _fail("sampler_state_missing", "checkpoint does not preserve both sampler states")

    return {
        "verified": True,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_sidecar_verified": True,
        "step": step,
        "effective_config_path": str(effective_config_path),
        "effective_config_sha256": file_sha256(effective_config_path),
        "training_signature_sha256": declared_signature,
        "base_checkpoint_step": BASE_STEP,
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "sft_tensor_path": str(sft_tensor_path),
        "sft_tensor_sha256": actual_tensor_sha,
        "sft_dataset_manifest_path": str(sft_manifest_path),
        "sft_dataset_manifest_sha256": actual_manifest_sha,
        "training_split_counts": {"train": 8000, "val": 800},
        "public_records_consumed": 0,
        "sealed_records_consumed": 0,
        "phase": expected_phase,
        "blind_body_reads": 0,
    }


def validate_baseline_reference(
    baseline: Mapping[str, Any], baseline_path: Path
) -> dict[str, Any]:
    """Verify and extract the exact M019 Step-5750 comparison reference."""

    if file_sha256(baseline_path) != EXPECTED_BASELINE_AUDIT_SHA256:
        _fail("baseline_audit_sha_mismatch", "M019 baseline audit file changed")
    if baseline.get("schema_version") != "pretrain-capability-audit/v1":
        _fail("baseline_audit_schema_mismatch", "M019 baseline audit schema changed")
    checkpoint = baseline.get("checkpoint")
    diagnostic = baseline.get("validation_diagnostic")
    probe = baseline.get("probe_provenance")
    generation = baseline.get("generation_settings")
    if not all(isinstance(value, Mapping) for value in (checkpoint, diagnostic, probe, generation)):
        _fail("baseline_audit_incomplete", "M019 baseline audit is incomplete")
    checks = {
        "checkpoint_sha": checkpoint.get("sha256") == BASE_CHECKPOINT_SHA256,
        "checkpoint_step": int(checkpoint.get("step", -1)) == BASE_STEP,
        "validation_split": diagnostic.get("split") == "val",
        "validation_tensor_sha": diagnostic.get("tensor_sha256")
        == EXPECTED_VALIDATION_TENSOR_SHA256,
        "window_count": int(diagnostic.get("windows_evaluated", -1))
        == EXPECTED_VALIDATION_WINDOWS,
        "window_tokens": int(diagnostic.get("tokens_evaluated", -1))
        == EXPECTED_WINDOW_TOKENS,
        "window_policy": diagnostic.get("window_selection")
        == "deterministic_evenly_spaced",
        "probe_sha": probe.get("artifact_sha256") == EXPECTED_PROBE_SHA256,
        "prompts_sha": probe.get("prompts_sha256") == EXPECTED_PROMPTS_SHA256,
        "prompt_count": int(generation.get("prompt_count", -1)) == EXPECTED_PROMPT_COUNT,
        "generation_settings": all(
            generation.get(key) == value for key, value in GENERATION_SETTINGS.items()
        ),
    }
    if not all(checks.values()):
        failed = ", ".join(key for key, passed in checks.items() if not passed)
        _fail("baseline_audit_contract_mismatch", f"M019 baseline contract changed: {failed}")
    loss = float(diagnostic.get("loss", math.nan))
    if not math.isclose(loss, EXPECTED_BASELINE_FIXED_WINDOW_LOSS, rel_tol=0, abs_tol=1e-12):
        _fail("baseline_loss_mismatch", "M019 fixed-window baseline loss changed")
    baseline_bpc = fixed_window_bpc(
        loss, EXPECTED_VALIDATION_TOKENS, EXPECTED_VALIDATION_CHARACTERS
    )
    return {
        "path": str(baseline_path),
        "sha256": EXPECTED_BASELINE_AUDIT_SHA256,
        "checkpoint_step": BASE_STEP,
        "checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "teacher_forced_fixed_window_loss": loss,
        "perplexity": float(diagnostic["perplexity"]),
        "fixed_window_bpc": baseline_bpc,
        "full_history_validation_bpc_reference_only": EXPECTED_BASELINE_FULL_HISTORY_BPC,
        "full_history_reference_not_used_for_gate": True,
        "windows": EXPECTED_VALIDATION_WINDOWS,
        "tokens_evaluated": EXPECTED_WINDOW_TOKENS,
        "probe_sha256": EXPECTED_PROBE_SHA256,
        "prompts_sha256": EXPECTED_PROMPTS_SHA256,
        "checks": checks,
    }


def validate_pretraining_inputs(
    *,
    config: Mapping[str, Any],
    data_dir: Path,
    raw_validation_path: Path,
    probes_path: Path,
    prompts_path: Path,
) -> tuple[BPETokenizer, torch.Tensor, dict[str, Any], dict[str, Any]]:
    """Load only frozen validation artifacts and validation-origin probes."""

    if canonical_json_sha256(config) != BASE_CONFIG_CANONICAL_SHA256:
        _fail("base_config_sha_mismatch", "pretraining configuration is not frozen")
    if config.get("model") != EXPECTED_CONFIG_MODEL:
        _fail("base_model_config_mismatch", "pretraining model configuration changed")
    manifest_path = data_dir / "token_manifest.json"
    tokenizer_path = data_dir / "tokenizer.json"
    validation_tensor_path = data_dir / "val_tokens.pt"
    if file_sha256(manifest_path) != BASE_TOKEN_MANIFEST_SHA256:
        _fail("token_manifest_sha_mismatch", "BPE token manifest changed")
    manifest = _read_json_object(manifest_path, "token_manifest_invalid")
    val_meta = manifest.get("splits", {}).get("val")
    if not isinstance(val_meta, Mapping):
        _fail("validation_manifest_missing", "token manifest lacks validation metadata")
    if file_sha256(tokenizer_path) != TOKENIZER_SHA256:
        _fail("tokenizer_sha_mismatch", "BPE tokenizer changed")
    if file_sha256(validation_tensor_path) != EXPECTED_VALIDATION_TENSOR_SHA256:
        _fail("validation_tensor_sha_mismatch", "validation tensor changed")
    if file_sha256(raw_validation_path) != EXPECTED_RAW_VALIDATION_SHA256:
        _fail("raw_validation_sha_mismatch", "raw validation text changed")
    if val_meta.get("text_sha256") != EXPECTED_RAW_VALIDATION_SHA256:
        _fail("validation_text_binding_mismatch", "manifest raw-validation SHA changed")
    if val_meta.get("tensor_sha256") != EXPECTED_VALIDATION_TENSOR_SHA256:
        _fail("validation_tensor_binding_mismatch", "manifest validation tensor SHA changed")
    if int(val_meta.get("characters", -1)) != EXPECTED_VALIDATION_CHARACTERS:
        _fail("validation_character_count_mismatch", "validation character count changed")
    if int(val_meta.get("tokens", -1)) != EXPECTED_VALIDATION_TOKENS:
        _fail("validation_token_count_mismatch", "validation token count changed")
    if file_sha256(probes_path) != EXPECTED_PROBE_SHA256:
        _fail("probe_sha_mismatch", "M019 validation probes changed")
    if file_sha256(prompts_path) != EXPECTED_PROMPTS_SHA256:
        _fail("prompt_sha_mismatch", "M019 continuation prompts changed")

    probe_bundle = validate_probe_artifact_for_evaluation(
        probes_path,
        prompts_path,
        manifest,
        require_formal_declarations=True,
    )
    if probe_bundle["validation_cloze_count"] != EXPECTED_CLOZE_COUNT:
        _fail("cloze_count_mismatch", "formal validation cloze count changed")
    if probe_bundle["validation_continuation_count"] != EXPECTED_PROMPT_COUNT:
        _fail("prompt_count_mismatch", "formal continuation prompt count changed")
    if any(
        case.get("source", {}).get("split") != "val"
        for case in probe_bundle["cases"]
    ):
        _fail("probe_split_mismatch", "retention cloze includes a non-validation source")
    tokenizer = BPETokenizer.load(tokenizer_path)
    validation_data = load_tensor(validation_tensor_path)
    if len(validation_data) != EXPECTED_VALIDATION_TOKENS:
        _fail("validation_tensor_length_mismatch", "validation tensor length changed")
    return tokenizer, validation_data, manifest, probe_bundle


def build_automatic_gates(
    *,
    comparison: Mapping[str, Any],
    generation_summary: Mapping[str, Any],
    generation_count: int,
) -> dict[str, Any]:
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
            "observed": generation_count
            - round(float(generation_summary["empty_rate"]) * generation_count),
            "operator": "==",
            "threshold": RETENTION_THRESHOLDS["required_nonempty_continuations"],
            "passed": float(generation_summary["empty_rate"]) == 0.0,
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
    """Return aggregate-only generation context with no prompt/output body."""

    return {
        "prompt_index": prompt_index,
        "prompt_characters": len(prompt),
        "generated_characters": measured["characters"],
        "generated_tokens": measured["generated_tokens"],
        "stop_reason": measured["stop_reason"],
        "four_gram_repetition": measured["four_gram_repetition"],
        "mechanically_degenerate": measured["mechanically_degenerate"],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    report = normalize_artifact_paths(report)
    assert_publishable_paths(report)
    validation = report["validation_diagnostic"]
    comparison = report["bpc_comparison"]
    generation = report["generation_summary"]
    cloze = report["cloze"]
    lines = [
        "# SFT v7 预训练能力保持审计",
        "",
        "> 本报告只使用 M019 的 validation 固定窗口与 validation-origin 探针；没有读取预训练 test 或 SFT sealed 正文。AI 流畅度/连贯度仍为外部待审，不能冒充真人验收。",
        "",
        f"- Checkpoint：`{report['checkpoint_lineage']['checkpoint_path']}`（SFT Step {report['checkpoint_lineage']['step']}）",
        f"- Teacher-forced fixed-window Loss：{validation['loss']:.6f}",
        f"- Token Perplexity：{validation['perplexity']:.3f}",
        f"- Candidate fixed-window BPC：{comparison['candidate_bpc']:.6f}",
        f"- Step 5750 baseline fixed-window BPC：{comparison['baseline_bpc']:.6f}",
        f"- BPC 绝对增量（候选−基座）：{comparison['absolute_delta_candidate_minus_baseline']:+.6f}",
        f"- BPC 相对恶化：{comparison['relative_degradation_candidate_minus_baseline']:+.2%}（门槛 ≤10%）",
        f"- 固定 16 条非空率：{1.0 - generation['empty_rate']:.1%}",
        f"- EOS 停止率：{generation['eos_stop_rate']:.1%}",
        f"- 机械退化率：{generation['mechanical_degeneration_rate']:.1%}（门槛 ≤25%）",
        f"- Cloze mean-token Top-1：{cloze['top1_accuracy']:.1%}（诊断项，不是发布硬门）",
        "",
        "## 自动硬门",
        "",
        "| 门 | 观测 | 规则 | 结果 |",
        "|---|---:|---:|---|",
    ]
    for gate in report["automatic_hard_gates"]["gates"]:
        observed = gate["observed"]
        threshold = gate["threshold"]
        if isinstance(observed, float):
            observed_text = f"{observed:.6f}"
        else:
            observed_text = str(observed)
        lines.append(
            f"| {gate['name']} | {observed_text} | {gate['operator']} {threshold} | "
            f"{'PASS' if gate['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## 固定续写",
            "",
            "| # | 提示 | 完整续写（最多 120 字） | EOS | 4-gram 重复率 | 退化 |",
            "|---:|---|---|---|---:|---|",
        ]
    )
    for sample in report["generations"]:
        prompt = sample["prompt"].replace("|", "\\|").replace("\n", "↵")
        continuation = sample["continuation"].replace("|", "\\|").replace("\n", "↵")
        lines.append(
            f"| {sample['prompt_index']} | {prompt} | {continuation} | "
            f"{'是' if sample['eos_emitted'] else '否'} | "
            f"{sample['four_gram_repetition']:.3f} | "
            f"{'是' if sample['mechanically_degenerate'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "## Cloze 诊断",
            "",
            "Cloze 复用 M019 的 12 条 validation-prefix 候选排序。它只用于观察 SFT 前后变化，不证明模型已可靠掌握实体事实，也不进入自动发布硬门。",
            "",
            "| Case ID | 正确候选 | 模型首选 | 正确名次 |",
            "|---|---|---|---:|",
        ]
    )
    for case in cloze["cases"]:
        lines.append(
            f"| {case['id']} | {case['correct']} | {case['predicted']} | "
            f"{case['correct_rank']} |"
        )
    lines.extend(
        [
            "",
            "## 外部待审与结论边界",
            "",
            "- AI 辅助流畅度 ≥2/5：**pending**。",
            "- AI 辅助局部连贯度 ≥2/5：**pending**。",
            "- 独立真人最终抽查：**pending**。",
            "- 自动硬门通过也不代表候选可发布；外部复核完成前 `candidate_eligible=false`。",
            "",
            f"状态：**{report['status']}**",
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
    config = load_config(args.config)
    device = select_device(args.device)
    effective_config = _read_json_object(
        args.effective_config, "effective_config_invalid"
    )
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    lineage = validate_sft_lineage(
        checkpoint,
        checkpoint_path=args.checkpoint,
        effective_config=effective_config,
        effective_config_path=args.effective_config,
        sft_tensor_path=args.sft_tensor,
        sft_manifest_path=args.sft_manifest,
    )
    loggers["checkpoint"].info(
        "SFT v7 checkpoint lineage verified",
        extra={
            "context": {
                "checkpoint_sha256": lineage["checkpoint_sha256"],
                "step": lineage["step"],
                "base_checkpoint_sha256": lineage["base_checkpoint_sha256"],
                "sft_tensor_sha256": lineage["sft_tensor_sha256"],
                "sft_dataset_manifest_sha256": lineage[
                    "sft_dataset_manifest_sha256"
                ],
                "public_records_consumed": 0,
                "sealed_records_consumed": 0,
            }
        },
    )

    tokenizer, validation_data, manifest, probe_bundle = validate_pretraining_inputs(
        config=config,
        data_dir=args.data_dir,
        raw_validation_path=args.raw_validation,
        probes_path=args.probes,
        prompts_path=args.prompts,
    )
    baseline_payload = _read_json_object(
        args.baseline_audit, "baseline_audit_invalid"
    )
    baseline = validate_baseline_reference(baseline_payload, args.baseline_audit)
    loggers["data"].info(
        "frozen validation-only artifacts verified",
        extra={
            "context": {
                "raw_validation_sha256": EXPECTED_RAW_VALIDATION_SHA256,
                "validation_tensor_sha256": EXPECTED_VALIDATION_TENSOR_SHA256,
                "validation_characters": EXPECTED_VALIDATION_CHARACTERS,
                "validation_tokens": EXPECTED_VALIDATION_TOKENS,
                "probe_sha256": EXPECTED_PROBE_SHA256,
                "prompts_sha256": EXPECTED_PROMPTS_SHA256,
                "pretraining_test_body_reads": 0,
                "sft_sealed_body_reads": 0,
            }
        },
    )

    model_config = GPTConfig(vocab_size=tokenizer.vocab_size, **config["model"])
    model = GPTLanguageModelV4(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if model.parameter_count() != BASE_PARAMETER_COUNT:
        _fail("parameter_count_mismatch", "checkpoint parameter count changed")
    model.to(device)

    validation = evaluate_held_out(
        model,
        validation_data,
        window_count=EXPECTED_VALIDATION_WINDOWS,
        batch_size=args.eval_batch_size,
        device=device,
    )
    if validation["windows_evaluated"] != EXPECTED_VALIDATION_WINDOWS:
        _fail("validation_window_count_mismatch", "fixed-window count changed")
    if validation["tokens_evaluated"] != EXPECTED_WINDOW_TOKENS:
        _fail("validation_window_token_mismatch", "fixed-window token count changed")
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
            "bpc_is_comparable_to_m019_same_window_reference": True,
            "bpc_is_not_the_full_history_training_metric": True,
        }
    )
    comparison = compare_bpc(validation["fixed_window_bpc"], baseline["fixed_window_bpc"])
    loggers["validation"].info(
        "fixed-window validation retention computed",
        extra={
            "context": {
                "loss": validation["loss"],
                "perplexity": validation["perplexity"],
                "fixed_window_bpc": validation["fixed_window_bpc"],
                "baseline_fixed_window_bpc": baseline["fixed_window_bpc"],
                "absolute_bpc_delta": comparison[
                    "absolute_delta_candidate_minus_baseline"
                ],
                "relative_bpc_degradation": comparison[
                    "relative_degradation_candidate_minus_baseline"
                ],
                "passed": comparison["passed"],
            }
        },
    )

    cloze_cases = list(probe_bundle.pop("cases"))
    prompts = list(probe_bundle.pop("continuation_prompts"))
    generations: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        generator = torch.Generator().manual_seed(
            GENERATION_SETTINGS["seed"] + 1000 + index
        )
        result = generate_continuation(
            model,
            tokenizer,
            prompt,
            max_new_tokens=GENERATION_SETTINGS["max_new_tokens"],
            max_characters=GENERATION_SETTINGS["max_characters"],
            temperature=GENERATION_SETTINGS["temperature"],
            top_k=GENERATION_SETTINGS["top_k"],
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
    cloze = evaluate_cloze(model, tokenizer, cloze_cases, device)
    loggers["cloze"].info(
        "validation cloze diagnostic complete",
        extra={
            "context": {
                "case_count": cloze["case_count"],
                "top1_accuracy": cloze["top1_accuracy"],
                "mean_reciprocal_rank": cloze["mean_reciprocal_rank"],
                "hard_gate": False,
            }
        },
    )

    gates = build_automatic_gates(
        comparison=comparison,
        generation_summary=generation_summary,
        generation_count=len(generations),
    )
    status = (
        "automatic_retention_gates_passed_external_review_pending"
        if gates["passed"]
        else "automatic_retention_gates_failed_external_review_pending"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "candidate_eligible": False,
        "candidate_ineligibility_reason": (
            "external AI-assisted fluency/coherence and independent human review pending"
        ),
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
            "token_manifest_path": str(args.data_dir / "token_manifest.json"),
            "token_manifest_sha256": BASE_TOKEN_MANIFEST_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "validation_tensor_sha256": EXPECTED_VALIDATION_TENSOR_SHA256,
            "pretraining_test_body_reads": 0,
            "sft_public_body_reads": 0,
            "sft_sealed_body_reads": 0,
            "blind_split_used_for_selection": False,
        },
        "baseline_reference": baseline,
        "validation_diagnostic": validation,
        "bpc_comparison": comparison,
        "probe_provenance": probe_bundle,
        "generation_settings": {
            **GENERATION_SETTINGS,
            "prompt_path": str(args.prompts),
            "prompt_count": len(prompts),
            "same_as_m019": True,
        },
        "generation_summary": generation_summary,
        "generations": generations,
        "cloze": cloze,
        "cloze_policy": {
            "diagnostic_only": True,
            "hard_gate": False,
            "does_not_prove_reliable_entity_knowledge": True,
        },
        "retention_thresholds": RETENTION_THRESHOLDS,
        "automatic_hard_gates": gates,
        "external_reviews": {
            "ai_assisted_fluency": {
                "status": "pending",
                "minimum_score": RETENTION_THRESHOLDS["minimum_external_ai_fluency"],
                "score": None,
            },
            "ai_assisted_local_coherence": {
                "status": "pending",
                "minimum_score": RETENTION_THRESHOLDS[
                    "minimum_external_ai_local_coherence"
                ],
                "score": None,
            },
            "independent_human_review": {
                "status": "pending",
                "score": None,
            },
            "automated_report_claims_human_review": False,
        },
        "logging": {
            "directory": str(args.log_dir),
            "modules": list(LOG_MODULES),
            "environment_override": "GPT_SFT_RETENTION_LOG_LEVEL_<MODULE>",
            "rotation_max_bytes": args.log_max_bytes,
            "rotation_backup_count": args.log_backup_count,
            "prompt_or_continuation_bodies_logged": False,
            "sensitive_fields_redacted": True,
            "production_default": "INFO",
        },
    }
    normalized_report = normalize_artifact_paths(report)
    assert_publishable_paths(normalized_report)
    return normalized_report


def write_retention_outputs(
    report: Mapping[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    """Atomically write portable JSON/Markdown retention artifacts."""

    normalized_report = normalize_artifact_paths(report)
    assert_publishable_paths(normalized_report)
    markdown = build_markdown_report(normalized_report)
    assert_publishable_paths(markdown)
    atomic_write_json(json_path, normalized_report)
    atomic_write_text(markdown_path, markdown)
    return normalized_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--effective-config", type=Path, default=DEFAULT_EFFECTIVE_CONFIG
    )
    parser.add_argument("--sft-tensor", type=Path, default=DEFAULT_SFT_TENSOR)
    parser.add_argument("--sft-manifest", type=Path, default=DEFAULT_SFT_MANIFEST)
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
        "--log-level",
        action="append",
        default=[],
        metavar="MODULE=LEVEL",
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
    # There is intentionally no test/sealed CLI input.  Prevent a relocated
    # validation artifact from being pointed at a blind body by name.
    for label in ("raw_validation", "probes", "prompts"):
        path = Path(getattr(args, label))
        lowered = [part.lower() for part in path.parts]
        if any("sealed" in part or part in {"test", "test.txt", "test_tokens.pt"} for part in lowered):
            raise ValueError(f"{label} cannot point to a test or sealed artifact")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = generate_run_id("sft-v7-retention")
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
            "SFT v7 retention audit started",
            extra={
                "context": {
                    "checkpoint": portable_artifact_path(
                        args.checkpoint, role="checkpoint"
                    ),
                    "device_request": args.device,
                    "evaluation_split": "val",
                    "fixed_window_count": EXPECTED_VALIDATION_WINDOWS,
                    "pretraining_test_body_reads": 0,
                    "sft_sealed_body_reads": 0,
                }
            },
        )
        report = run_retention_audit(args, run_id, loggers)
        report = write_retention_outputs(
            report,
            args.output_json,
            args.output_markdown,
        )
        loggers["orchestrator"].info(
            "SFT v7 retention artifacts written",
            extra={
                "context": {
                    "status": report["status"],
                    "automatic_hard_gates_passed": report["automatic_hard_gates"][
                        "passed"
                    ],
                    "candidate_eligible": False,
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
                    "automatic_hard_gates_passed": report[
                        "automatic_hard_gates"
                    ]["passed"],
                    "candidate_eligible": False,
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
        code = getattr(error, "code", "retention_audit_failed")
        if loggers:
            loggers["orchestrator"].error(
                "SFT v7 retention audit failed",
                extra={
                    "context": {
                        "error_code": code,
                        "error_type": type(error).__name__,
                        "operation": "run_retention_audit",
                        "pretraining_test_body_reads": 0,
                        "sft_sealed_body_reads": 0,
                    }
                },
            )
        print(
            f"SFT v7 retention audit failed [{code}]: {type(error).__name__}",
            file=sys.stderr,
        )
        return 1
    finally:
        close_module_loggers(loggers)


if __name__ == "__main__":
    raise SystemExit(main())
