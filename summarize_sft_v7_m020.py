"""Build a provenance-aware M020 SFT-v7 checkpoint comparison bundle.

This tool is intentionally file-only: it does not import a model, open a
checkpoint, read a tensor, or touch the SFT sealed split.  It summarizes the
smoke/formal training reports, public diagnostics, fixed-prompt samples and
pretraining-retention reports that already exist.  The 600-case public
diagnostic is required for the frozen baseline and formal 500-step milestones,
but intentionally optional for the 20-step engineering smoke; smoke still
requires its training report and frozen fixed-16 output.

Missing future artifacts are represented as ``pending`` and never replaced by
zeroes or guessed metrics.  Automatic gates and external gates remain separate;
an external ``pending`` value is never promoted to passed.  The generated
comparison is diagnostic evidence, not a release decision.

Logs are split into independently configurable ``discovery``, ``validation``,
``rendering`` and ``orchestrator`` JSONL streams.  They contain paths, hashes,
counts and aggregate states only--never prompts, generated text or novel
evidence.  Set ``GPT_M020_SUMMARY_LOG_LEVEL_<MODULE>`` or repeat
``--log-level MODULE=LEVEL``.  Rotating logs default to INFO and use the shared
credential redaction formatter.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape as xml_escape
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

from training_runtime import (
    DEFAULT_LOG_MODULES,
    atomic_write_json,
    atomic_write_text,
    close_module_loggers,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
)


SCHEMA_VERSION = "sft-v7-m020-checkpoint-comparison/v1"
TRAIN_SCHEMA = "sft-v7-train-report/v1"
PUBLIC_SCHEMA = "sft-v7-public-evaluation/v1"
FIXED_SCHEMA = "sft-v7-fixed-samples/v1"
RETENTION_SCHEMA = "sft-v7-pretrain-retention/v1"

DEFAULT_OUTPUT_DIR = Path("reports/milestones/020_sft_v7_vertical")
DEFAULT_LOG_DIR = Path("logs/sft_v7_m020_summary")
LOG_MODULES = ("discovery", "validation", "rendering", "orchestrator")
_ALL_LOG_MODULES = tuple(dict.fromkeys((*DEFAULT_LOG_MODULES, *LOG_MODULES)))
_LEVEL_NAMES = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "OFF"}
_SHA256 = re.compile(r"[0-9a-f]{64}")


class M020SummaryError(ValueError):
    """Raised when a present artifact violates its declared contract."""


@dataclass(frozen=True)
class ArtifactSpec:
    """One report input with ordered filename aliases."""

    kind: str
    paths: tuple[Path, ...]
    required: bool = True


@dataclass(frozen=True)
class CheckpointSpec:
    """Expected evidence bundle for one displayed checkpoint."""

    key: str
    display_name: str
    expected_step: int
    checkpoint_mode: str
    public: ArtifactSpec
    fixed: ArtifactSpec
    retention: ArtifactSpec | None = None


def _paths(root: Path, *names: str) -> tuple[Path, ...]:
    return tuple(root / name for name in names)


def default_train_specs(root: Path = DEFAULT_OUTPUT_DIR) -> dict[str, ArtifactSpec]:
    return {
        "smoke20": ArtifactSpec(
            "train",
            _paths(root, "smoke_train_report.json", "train_smoke20_report.json"),
        ),
        "formal2000": ArtifactSpec(
            "train",
            _paths(root, "formal_train_report.json", "train_report.json"),
        ),
    }


def default_checkpoint_specs(
    root: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[CheckpointSpec, ...]:
    """Return the frozen M020 display order and supported filename aliases."""

    def public(
        step: str, *aliases: str, required: bool = True
    ) -> ArtifactSpec:
        return ArtifactSpec(
            "public",
            _paths(root, f"public_eval_{step}.json", *aliases),
            required=required,
        )

    def fixed(step: str, *aliases: str) -> ArtifactSpec:
        return ArtifactSpec(
            "fixed",
            _paths(root, f"fixed_samples_{step}.json", *aliases),
        )

    def retention(step: str, *aliases: str, required: bool = False) -> ArtifactSpec:
        return ArtifactSpec(
            "retention",
            _paths(root, f"pretrain_retention_{step}.json", *aliases),
            required=required,
        )

    return (
        CheckpointSpec(
            key="baseline_step05750",
            display_name="Pretrain baseline Step 5750",
            expected_step=5750,
            checkpoint_mode="pretrain-baseline",
            public=public(
                "step05750",
                "public_evaluation_step05750.json",
            ),
            fixed=fixed("step05750"),
            retention=None,
        ),
        CheckpointSpec(
            key="smoke_step00020",
            display_name="SFT v7 smoke Step 20",
            expected_step=20,
            checkpoint_mode="sft-v7",
            public=public(
                "smoke20",
                "public_eval_step00020.json",
                required=False,
            ),
            fixed=fixed("smoke20", "fixed_samples_step00020.json"),
            retention=retention("smoke20", "pretrain_retention_step00020.json"),
        ),
        CheckpointSpec(
            key="sft_step00500",
            display_name="SFT v7 Step 500",
            expected_step=500,
            checkpoint_mode="sft-v7",
            public=public(
                "step00500",
                "public_eval_step500.json",
                "public_evaluation_step00500.json",
            ),
            fixed=fixed("step00500", "fixed_samples_step500.json"),
            retention=retention(
                "step00500",
                "pretrain_retention_step500.json",
            ),
        ),
        CheckpointSpec(
            key="sft_step01000",
            display_name="SFT v7 Step 1000",
            expected_step=1000,
            checkpoint_mode="sft-v7",
            public=public(
                "step01000",
                "public_eval_step1000.json",
                "public_evaluation_step01000.json",
            ),
            fixed=fixed("step01000", "fixed_samples_step1000.json"),
            retention=retention(
                "step01000",
                "pretrain_retention_step1000.json",
            ),
        ),
        CheckpointSpec(
            key="sft_step01500",
            display_name="SFT v7 Step 1500",
            expected_step=1500,
            checkpoint_mode="sft-v7",
            public=public(
                "step01500",
                "public_eval_step1500.json",
                "public_evaluation_step01500.json",
            ),
            fixed=fixed("step01500", "fixed_samples_step1500.json"),
            retention=retention(
                "step01500",
                "pretrain_retention_step1500.json",
            ),
        ),
        CheckpointSpec(
            key="sft_step02000",
            display_name="SFT v7 Step 2000",
            expected_step=2000,
            checkpoint_mode="sft-v7",
            public=public(
                "step02000",
                "public_eval_step2000.json",
                "public_evaluation_step02000.json",
            ),
            fixed=fixed("step02000", "fixed_samples_step2000.json"),
            retention=retention(
                "step02000",
                "pretrain_retention_step2000.json",
                "pretrain_retention.json",
                required=True,
            ),
        ),
    )


def resolve_log_levels(overrides: Sequence[str] = ()) -> dict[str, str]:
    levels = {
        module: os.getenv(
            f"GPT_M020_SUMMARY_LOG_LEVEL_{module.upper()}",
            "INFO" if module in LOG_MODULES else "OFF",
        ).upper()
        for module in _ALL_LOG_MODULES
    }
    for module, level in levels.items():
        if level not in _LEVEL_NAMES:
            raise M020SummaryError(f"unknown log level for {module}: {level}")
    for override in overrides:
        if "=" not in override:
            raise M020SummaryError("--log-level must use MODULE=LEVEL")
        module, level = (part.strip() for part in override.split("=", 1))
        level = level.upper()
        if module not in _ALL_LOG_MODULES:
            raise M020SummaryError(f"unknown M020 summary log module: {module}")
        if level not in _LEVEL_NAMES:
            raise M020SummaryError(f"unknown log level: {level}")
        levels[module] = level
    return levels


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise M020SummaryError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise M020SummaryError(f"{label} must be finite")
    return result


def _optional_finite(value: Any, label: str) -> float | None:
    return None if value is None else _finite(value, label)


def _read_json(path: Path, expected_schema: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise M020SummaryError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise M020SummaryError(f"JSON artifact root must be an object: {path}")
    if value.get("schema_version") != expected_schema:
        raise M020SummaryError(
            f"unexpected schema for {path}: expected {expected_schema!r}, "
            f"got {value.get('schema_version')!r}"
        )
    return value


def _reject_blind_path(path: Path) -> None:
    lowered = path.name.lower()
    if "sealed" in lowered:
        raise M020SummaryError(
            f"summary inputs cannot be sealed artifacts: {path.name}"
        )


def _discover(
    spec: ArtifactSpec,
    *,
    expected_schema: str,
    loggers: Mapping[str, logging.Logger],
) -> dict[str, Any]:
    for path in spec.paths:
        _reject_blind_path(path)
    existing = [path for path in spec.paths if path.is_file()]
    if not existing:
        state = "pending" if spec.required else "not_run_optional"
        loggers["discovery"].info(
            "artifact not present",
            extra={
                "context": {
                    "artifact_kind": spec.kind,
                    "state": state,
                    "expected_path_count": len(spec.paths),
                }
            },
        )
        return {
            "status": state,
            "required": spec.required,
            "expected_paths": [str(path) for path in spec.paths],
            "payload": None,
        }

    hashes = {path: file_sha256(path) for path in existing}
    if len(set(hashes.values())) > 1:
        raise M020SummaryError(
            f"multiple filename aliases exist with different content for {spec.kind}: "
            + ", ".join(str(path) for path in existing)
        )
    path = existing[0]
    payload = _read_json(path, expected_schema)
    loggers["discovery"].info(
        "artifact discovered",
        extra={
            "context": {
                "artifact_kind": spec.kind,
                "path": str(path),
                "sha256": hashes[path],
                "duplicate_aliases_with_identical_content": len(existing) - 1,
            }
        },
    )
    return {
        "status": "present",
        "required": spec.required,
        "path": str(path),
        "sha256": hashes[path],
        "schema_version": expected_schema,
        "payload": payload,
    }


def _public_summary(
    source: Mapping[str, Any], spec: CheckpointSpec
) -> tuple[dict[str, Any], list[str]]:
    if source["status"] != "present":
        return {key: value for key, value in source.items() if key != "payload"}, []
    payload = source["payload"]
    errors: list[str] = []
    step = int(payload.get("checkpoint_step", -1))
    mode = str(payload.get("checkpoint_mode", ""))
    if step != spec.expected_step:
        errors.append(
            f"public checkpoint step {step} does not match expected {spec.expected_step}"
        )
    if mode != spec.checkpoint_mode:
        errors.append(
            f"public checkpoint mode {mode!r} does not match {spec.checkpoint_mode!r}"
        )
    checkpoint_sha = str(payload.get("checkpoint_sha256", ""))
    if not _SHA256.fullmatch(checkpoint_sha):
        errors.append("public checkpoint SHA-256 is missing or invalid")

    teacher = payload.get("teacher_forced")
    overall = payload.get("overall")
    if not isinstance(teacher, Mapping) or not isinstance(overall, Mapping):
        raise M020SummaryError("public report lacks teacher_forced or overall metrics")
    generation = overall.get("generation_quality")
    if not isinstance(generation, Mapping):
        raise M020SummaryError("public report lacks overall generation_quality")
    automatic = payload.get("automatic_gates")
    external = payload.get("external_gates")
    if not isinstance(automatic, list) or not automatic:
        raise M020SummaryError("public report has no automatic gate list")
    if not isinstance(external, list) or not external:
        raise M020SummaryError("public report has no external gate list")
    auto_passed = sum(gate.get("passed") is True for gate in automatic if isinstance(gate, Mapping))
    auto_failed = sum(gate.get("passed") is not True for gate in automatic if isinstance(gate, Mapping))
    ext_passed = sum(gate.get("passed") is True for gate in external if isinstance(gate, Mapping))
    ext_failed = sum(gate.get("passed") is False for gate in external if isinstance(gate, Mapping))
    ext_pending = len(external) - ext_passed - ext_failed
    derived_auto = auto_failed == 0 and len(automatic) > 0
    derived_external = ext_passed == len(external) and len(external) > 0
    if bool(payload.get("automatic_gates_passed")) != derived_auto:
        errors.append("public automatic_gates_passed disagrees with individual gates")
    if bool(payload.get("external_gates_passed")) != derived_external:
        errors.append("public external_gates_passed disagrees with individual gates")
    declared_candidate = payload.get("candidate_eligible") is True
    if declared_candidate and not (derived_auto and derived_external):
        errors.append("public report claims candidate eligibility before every gate passed")

    result = {
        **{key: value for key, value in source.items() if key != "payload"},
        "checkpoint_step": step,
        "checkpoint_mode": mode,
        "checkpoint_sha256": checkpoint_sha or None,
        "teacher_forced_loss": _finite(
            teacher.get("loss"), "public teacher-forced loss"
        ),
        "teacher_forced_perplexity": _finite(
            teacher.get("perplexity"), "public teacher-forced perplexity"
        ),
        "records": int(overall.get("records", -1)),
        "generation_quality": {
            key: _optional_finite(generation.get(key), f"public {key}")
            for key in (
                "eos_rate",
                "empty_rate",
                "truncation_rate",
                "mechanical_repetition_rate",
                "meta_phrase_rate",
            )
        },
        "automatic_gate_state": "passed" if derived_auto else "failed",
        "automatic_gates": {
            "total": len(automatic),
            "passed": auto_passed,
            "failed": auto_failed,
        },
        "external_gate_state": (
            "failed" if ext_failed else "passed" if derived_external else "pending"
        ),
        "external_gates": {
            "total": len(external),
            "passed": ext_passed,
            "failed": ext_failed,
            "pending": ext_pending,
        },
        "declared_candidate_eligible": declared_candidate,
        "sft_dataset_manifest_sha256": payload.get(
            "sft_dataset_manifest_sha256"
        ),
        "public_tensor_sha256": payload.get("public_tensor_sha256"),
    }
    return result, errors


def _fixed_summary(
    source: Mapping[str, Any], spec: CheckpointSpec
) -> tuple[dict[str, Any], list[str]]:
    if source["status"] != "present":
        return {key: value for key, value in source.items() if key != "payload"}, []
    payload = source["payload"]
    errors: list[str] = []
    step = int(payload.get("checkpoint_step", -1))
    mode = str(payload.get("checkpoint_mode", ""))
    if step != spec.expected_step:
        errors.append(
            f"fixed-sample checkpoint step {step} does not match expected {spec.expected_step}"
        )
    if mode != spec.checkpoint_mode:
        errors.append(
            f"fixed-sample checkpoint mode {mode!r} does not match {spec.checkpoint_mode!r}"
        )
    checkpoint_sha = str(payload.get("checkpoint_sha256", ""))
    if not _SHA256.fullmatch(checkpoint_sha):
        errors.append("fixed-sample checkpoint SHA-256 is missing or invalid")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise M020SummaryError("fixed-sample report has no results")
    rows = [row for row in results if isinstance(row, Mapping)]
    if len(rows) != len(results):
        raise M020SummaryError("fixed-sample results must contain JSON objects")
    empty = sum(
        int(row.get("generated_tokens", -1)) <= 0
        or not str(row.get("generated_text", "")).strip()
        for row in rows
    )
    eos = sum(row.get("stopped_on_eos") is True for row in rows)
    truncated = sum(row.get("truncated") is True for row in rows)
    result = {
        **{key: value for key, value in source.items() if key != "payload"},
        "checkpoint_step": step,
        "checkpoint_mode": mode,
        "checkpoint_sha256": checkpoint_sha or None,
        "prompt_set_sha256": payload.get("prompt_set_sha256"),
        "case_count": len(rows),
        "empty_count": empty,
        "eos_count": eos,
        "eos_rate": eos / len(rows),
        "truncated_count": truncated,
        "truncation_rate": truncated / len(rows),
        "sft_dataset_manifest_sha256": payload.get(
            "sft_dataset_manifest_sha256"
        ),
        "public_tensor_sha256": payload.get("public_tensor_sha256"),
        "contains_sample_bodies": False,
    }
    return result, errors


def _retention_external_state(payload: Mapping[str, Any]) -> tuple[str, dict[str, int]]:
    reviews = payload.get("external_reviews")
    if not isinstance(reviews, Mapping) or not reviews:
        raise M020SummaryError("retention report has no external_reviews object")
    passed = failed = pending = 0
    for review in reviews.values():
        if isinstance(review, bool):
            # A boolean audit declaration is not an independent review score.
            pending += 1
            continue
        if not isinstance(review, Mapping):
            pending += 1
            continue
        status = str(review.get("status", "pending")).strip().lower()
        score = review.get("score")
        minimum = review.get("minimum_score")
        score_valid = isinstance(score, (int, float)) and not isinstance(score, bool)
        if status == "passed" and score_valid:
            if minimum is None or float(score) >= _finite(minimum, "external minimum score"):
                passed += 1
            else:
                failed += 1
        elif status in {"failed", "rejected"}:
            failed += 1
        else:
            pending += 1
    state = "failed" if failed else "passed" if pending == 0 else "pending"
    return state, {"total": passed + failed + pending, "passed": passed, "failed": failed, "pending": pending}


def _retention_summary(
    source: Mapping[str, Any], spec: CheckpointSpec
) -> tuple[dict[str, Any], list[str]]:
    if source["status"] != "present":
        return {key: value for key, value in source.items() if key != "payload"}, []
    payload = source["payload"]
    errors: list[str] = []
    lineage = payload.get("checkpoint_lineage")
    validation = payload.get("validation_diagnostic")
    comparison = payload.get("bpc_comparison")
    auto = payload.get("automatic_hard_gates")
    if not all(isinstance(value, Mapping) for value in (lineage, validation, comparison, auto)):
        raise M020SummaryError("retention report lacks lineage, validation, comparison or gates")
    step = int(lineage.get("step", -1))
    if step != spec.expected_step:
        errors.append(
            f"retention checkpoint step {step} does not match expected {spec.expected_step}"
        )
    checkpoint_sha = str(lineage.get("checkpoint_sha256", ""))
    if not _SHA256.fullmatch(checkpoint_sha):
        errors.append("retention checkpoint SHA-256 is missing or invalid")
    if int(lineage.get("public_records_consumed", -1)) != 0:
        errors.append("retention lineage reports public records consumed during training")
    if int(lineage.get("sealed_records_consumed", -1)) != 0:
        errors.append("retention lineage reports sealed records consumed")
    if int(lineage.get("blind_body_reads", -1)) != 0:
        errors.append("retention lineage reports blind body reads")
    data_scope = payload.get("data_scope")
    if not isinstance(data_scope, Mapping):
        raise M020SummaryError("retention report lacks data_scope")
    for key in (
        "pretraining_test_body_reads",
        "sft_public_body_reads",
        "sft_sealed_body_reads",
    ):
        if int(data_scope.get(key, -1)) != 0:
            errors.append(f"retention data scope violates {key}=0")
    gates = auto.get("gates")
    if not isinstance(gates, list) or not gates:
        raise M020SummaryError("retention report has no automatic gate list")
    auto_passed = sum(
        gate.get("passed") is True for gate in gates if isinstance(gate, Mapping)
    )
    derived_auto = auto_passed == len(gates)
    if bool(auto.get("passed")) != derived_auto:
        errors.append("retention automatic gate aggregate disagrees with individual gates")
    external_state, external_counts = _retention_external_state(payload)
    if payload.get("candidate_eligible") is True and external_state != "passed":
        errors.append("retention report claims candidate eligibility before external review")
    result = {
        **{key: value for key, value in source.items() if key != "payload"},
        "checkpoint_step": step,
        "checkpoint_sha256": checkpoint_sha or None,
        "fixed_window_loss": _finite(
            validation.get("loss"), "retention fixed-window loss"
        ),
        "fixed_window_bpc": _finite(
            validation.get("fixed_window_bpc"), "retention fixed-window BPC"
        ),
        "relative_bpc_degradation": _finite(
            comparison.get("relative_degradation_candidate_minus_baseline"),
            "retention relative BPC degradation",
        ),
        "automatic_gate_state": "passed" if derived_auto else "failed",
        "automatic_gates": {
            "total": len(gates),
            "passed": auto_passed,
            "failed": len(gates) - auto_passed,
        },
        "external_review_state": external_state,
        "external_reviews": external_counts,
        "declared_candidate_eligible": payload.get("candidate_eligible") is True,
        "sft_dataset_manifest_sha256": lineage.get(
            "sft_dataset_manifest_sha256"
        ),
        "sealed_body_reads": 0,
    }
    return result, errors


def _artifact_identity_errors(row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    present = []
    for key in ("public", "fixed_samples", "retention"):
        value = row.get(key)
        if isinstance(value, Mapping) and value.get("status") == "present":
            present.append(value)
    for field, label in (
        ("checkpoint_sha256", "checkpoint SHA-256"),
        ("sft_dataset_manifest_sha256", "SFT dataset-manifest SHA-256"),
        ("public_tensor_sha256", "public tensor SHA-256"),
    ):
        identities = {
            str(value.get(field)) for value in present if value.get(field)
        }
        if len(identities) > 1:
            errors.append(f"{label} differs across public/fixed/retention reports")
    return errors


def _checkpoint_status(row: Mapping[str, Any], *, baseline: bool) -> str:
    if row.get("integrity_errors"):
        return "invalid_artifact_identity"
    if row["fixed_samples"]["status"] != "present":
        return "pending_required_artifacts"
    if row["public"]["status"] != "present":
        if row["public"].get("required"):
            return "pending_required_artifacts"
        return "engineering_smoke_complete_public_not_run_optional"
    if row["public"].get("automatic_gate_state") != "passed":
        return "automatic_public_gates_failed"
    if baseline:
        return "baseline_diagnostic_external_review_pending"
    retention = row.get("retention")
    if retention and retention.get("status") == "present":
        if retention.get("automatic_gate_state") != "passed":
            return "automatic_retention_gates_failed"
        if retention.get("external_review_state") != "passed":
            return "external_review_pending_or_failed"
    elif retention and retention.get("required"):
        return "pending_required_retention"
    if row["public"].get("external_gate_state") != "passed":
        return "external_review_pending_or_failed"
    return "all_loaded_gates_passed_release_review_still_required"


def _strict_candidate_eligible(row: Mapping[str, Any], *, baseline: bool) -> bool:
    if baseline or row.get("integrity_errors"):
        return False
    public = row["public"]
    fixed = row["fixed_samples"]
    retention = row.get("retention")
    return bool(
        public.get("status") == "present"
        and fixed.get("status") == "present"
        and retention
        and retention.get("status") == "present"
        and public.get("automatic_gate_state") == "passed"
        and public.get("external_gate_state") == "passed"
        and retention.get("automatic_gate_state") == "passed"
        and retention.get("external_review_state") == "passed"
        and public.get("declared_candidate_eligible") is True
        and retention.get("declared_candidate_eligible") is True
    )


def _summarize_train_report(
    label: str,
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = {key: value for key, value in source.items() if key != "payload"}
    if source["status"] != "present":
        return base, []
    payload = source["payload"]
    history = payload.get("history")
    if not isinstance(history, list):
        raise M020SummaryError(f"training report {label} has no history list")
    if int(payload.get("public_records_consumed", -1)) != 0:
        raise M020SummaryError(f"training report {label} consumed public records")
    if int(payload.get("sealed_records_consumed", -1)) != 0:
        raise M020SummaryError(f"training report {label} consumed sealed records")
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, item in enumerate(history):
        if not isinstance(item, Mapping):
            raise M020SummaryError(f"training history {label}[{index}] is not an object")
        step = int(item.get("step", -1))
        if step < 0 or step in seen:
            raise M020SummaryError(f"training history {label} has invalid/duplicate step {step}")
        seen.add(step)
        coverage_payload = item.get("coverage")
        coverage = None
        if isinstance(coverage_payload, Mapping):
            coverage = _optional_finite(
                coverage_payload.get("coverage"), f"training {label} coverage"
            )
        rows.append(
            {
                "run_label": label,
                "step": step,
                "train_loss": _finite(item.get("train_loss"), f"{label} train loss"),
                "val_loss": _finite(item.get("val_loss"), f"{label} val loss"),
                "coverage": coverage,
                "active_phase": item.get("active_phase"),
                "source_report": source["path"],
            }
        )
    rows.sort(key=lambda row: row["step"])
    base.update(
        {
            "status_in_report": payload.get("status"),
            "target_step": int(payload.get("target_step", -1)),
            "best_val_loss": _finite(payload.get("best_val_loss"), f"{label} best val loss"),
            "history_points": len(rows),
            "public_records_consumed": 0,
            "sealed_records_consumed": 0,
        }
    )
    return base, rows


def build_loss_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "run_label",
            "step",
            "train_loss",
            "val_loss",
            "coverage",
            "active_phase",
            "source_report",
        )
    )
    for row in rows:
        writer.writerow(
            (
                row["run_label"],
                row["step"],
                row["train_loss"],
                row["val_loss"],
                "" if row["coverage"] is None else row["coverage"],
                row["active_phase"] or "",
                row["source_report"],
            )
        )
    return output.getvalue()


def build_loss_svg(rows: Sequence[Mapping[str, Any]]) -> str:
    width, height = 960, 540
    left, right, top, bottom = 90, 35, 55, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">',
        '<rect width="960" height="540" fill="#ffffff"/>',
        '<text x="480" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">M020 SFT v7 Loss</text>',
    ]
    if not rows:
        lines.extend(
            [
                '<rect x="90" y="55" width="835" height="410" fill="#fafafa" stroke="#cccccc"/>',
                '<text x="507" y="260" text-anchor="middle" font-family="sans-serif" font-size="18" fill="#666666">Pending: no completed training history</text>',
                '<text x="507" y="290" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#888888">No zero or estimated loss values were inserted.</text>',
                "</svg>",
            ]
        )
        return "\n".join(lines) + "\n"

    steps = [int(row["step"]) for row in rows]
    losses = [float(row[key]) for row in rows for key in ("train_loss", "val_loss")]
    x_min, x_max = min(steps), max(steps)
    y_min, y_max = min(losses), max(losses)
    if x_min == x_max:
        x_min, x_max = x_min - 1, x_max + 1
    if math.isclose(y_min, y_max):
        y_min, y_max = y_min - 0.5, y_max + 0.5
    padding = (y_max - y_min) * 0.08
    y_min = max(0.0, y_min - padding)
    y_max += padding

    def x_coord(step: int) -> float:
        return left + (step - x_min) / (x_max - x_min) * plot_width

    def y_coord(loss: float) -> float:
        return top + (y_max - loss) / (y_max - y_min) * plot_height

    lines.append(
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#fafafa" stroke="#cccccc"/>'
    )
    for index in range(6):
        fraction = index / 5
        y = top + fraction * plot_height
        value = y_max - fraction * (y_max - y_min)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e6e6e6"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.3f}</text>')
    for index in range(6):
        fraction = index / 5
        x = left + fraction * plot_width
        value = x_min + fraction * (x_max - x_min)
        lines.append(f'<text x="{x:.2f}" y="{top + plot_height + 25}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.0f}</text>')
    lines.extend(
        [
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="13">Optimizer step</text>',
            f'<text x="22" y="{top + plot_height / 2:.2f}" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 22 {top + plot_height / 2:.2f})">Mean NLL loss</text>',
        ]
    )
    palette = ("#1565c0", "#c62828", "#2e7d32", "#6a1b9a")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["run_label"]), []).append(row)
    legend_y = 50
    for run_index, (label, group) in enumerate(grouped.items()):
        group = sorted(group, key=lambda item: int(item["step"]))
        base_color = palette[run_index % len(palette)]
        for metric, dash in (("train_loss", ""), ("val_loss", "7,4")):
            points = " ".join(
                f'{x_coord(int(row["step"])):.2f},{y_coord(float(row[metric])):.2f}'
                for row in group
            )
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            lines.append(f'<polyline points="{points}" fill="none" stroke="{base_color}" stroke-width="2.2"{dash_attr}/>')
            for row in group:
                lines.append(f'<circle cx="{x_coord(int(row["step"])):.2f}" cy="{y_coord(float(row[metric])):.2f}" r="3" fill="{base_color}"/>')
        legend_x = 100 + run_index * 210
        lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" y2="{legend_y}" stroke="{base_color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 28}" y="{legend_y + 4}" font-family="sans-serif" font-size="11">{xml_escape(label)} (solid train, dashed val)</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _format_number(value: Any, digits: int = 4) -> str:
    return "pending" if value is None else f"{float(value):.{digits}f}"


def _gate_cell(summary: Mapping[str, Any], key: str) -> str:
    value = summary.get(key)
    if not isinstance(value, Mapping):
        return "pending"
    state = summary.get(key.replace("gates", "gate_state"), "pending")
    return f"{value.get('passed', 0)}/{value.get('total', 0)} ({state})"


def build_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# M020 SFT v7 Checkpoint Comparison",
        "",
        f"- 汇总状态：`{report['status']}`",
        f"- 必需产物待补：`{len(report['pending_required_artifacts'])}`",
        f"- 严格候选：`{', '.join(report['strict_candidate_keys']) if report['strict_candidate_keys'] else 'none'}`",
        "- 发布就绪：`false`（本工具不替代独立真人发布复核）",
        "- 外部门禁口径：只有每项明确 `passed=true` 才计为通过；`pending` 不会提升为通过。",
        "- 数据边界：本工具未打开 checkpoint、张量、原始语料或 sealed 正文。",
        "",
        "## Checkpoint 对比",
        "",
        "| Checkpoint | Step | Public Loss | Public 自动门 | Public 外部门 | Fixed EOS | Retention BPC 恶化 | Retention 自动门 | 汇总状态 |",
        "|---|---:|---:|---|---|---:|---:|---|---|",
    ]
    for row in report["checkpoints"]:
        public = row["public"]
        fixed = row["fixed_samples"]
        retention = row.get("retention") or {}
        public_auto = _gate_cell(public, "automatic_gates")
        public_external = public.get("external_gate_state", "pending")
        fixed_eos = _format_number(fixed.get("eos_rate"))
        retention_delta = _format_number(retention.get("relative_bpc_degradation"))
        retention_auto = _gate_cell(retention, "automatic_gates")
        lines.append(
            f"| {row['display_name']} | {row['expected_step']} | "
            f"{_format_number(public.get('teacher_forced_loss'), 6)} | "
            f"{public_auto} | {public_external} | {fixed_eos} | "
            f"{retention_delta} | {retention_auto} | {row['summary_status']} |"
        )
    lines.extend(
        [
            "",
            "## 待补产物",
            "",
        ]
    )
    if report["pending_required_artifacts"]:
        lines.extend(
            f"- `{item['owner']}` / `{item['kind']}`："
            + ", ".join(f"`{path}`" for path in item["expected_paths"])
            for item in report["pending_required_artifacts"]
        )
    else:
        lines.append("- 无必需文件缺失；外部评审仍按各 checkpoint 状态单独判断。")
    lines.extend(
        [
            "",
            "## 完整性问题",
            "",
        ]
    )
    integrity = [
        (row["key"], error)
        for row in report["checkpoints"]
        for error in row["integrity_errors"]
    ]
    if integrity:
        lines.extend(f"- `{key}`：{error}" for key, error in integrity)
    else:
        lines.append("- 当前已加载报告之间未发现 checkpoint 身份冲突。")
    lines.extend(
        [
            "",
            "## Loss 曲线",
            "",
            f"- CSV：`{report['outputs']['loss_csv']['path']}`",
            f"- SVG：`{report['outputs']['loss_svg']['path']}`",
            "- 缺失训练报告时 CSV 只保留表头，SVG 明确显示 pending；不会补零或插值。",
            "",
        ]
    )
    return "\n".join(lines)


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_sha256_markdown(paths: Sequence[Path], project_root: Path) -> str:
    unique = sorted({path.resolve() for path in paths if path.is_file()}, key=str)
    lines = [
        "# M020 SHA-256",
        "",
        "仅列出本次汇总实际读取或生成的非 sealed 工件；`SHA256SUMS.md` 不自哈希。",
        "",
        "| SHA-256 | 文件 |",
        "|---|---|",
    ]
    for path in unique:
        lines.append(f"| `{file_sha256(path)}` | `{_display_path(path, project_root)}` |")
    lines.append("")
    return "\n".join(lines)


def summarize_m020(
    *,
    train_specs: Mapping[str, ArtifactSpec],
    checkpoint_specs: Sequence[CheckpointSpec],
    output_dir: Path,
    project_root: Path,
    run_id: str,
    loggers: Mapping[str, logging.Logger],
) -> dict[str, Any]:
    """Read available reports and atomically build all summary artifacts."""

    schema_by_kind = {
        "train": TRAIN_SCHEMA,
        "public": PUBLIC_SCHEMA,
        "fixed": FIXED_SCHEMA,
        "retention": RETENTION_SCHEMA,
    }
    train_reports: dict[str, Any] = {}
    loss_rows: list[dict[str, Any]] = []
    loaded_paths: list[Path] = []
    pending_required: list[dict[str, Any]] = []
    for label, spec in train_specs.items():
        source = _discover(
            spec,
            expected_schema=schema_by_kind[spec.kind],
            loggers=loggers,
        )
        summary, rows = _summarize_train_report(label, source)
        train_reports[label] = summary
        loss_rows.extend(rows)
        if source["status"] == "present":
            loaded_paths.append(Path(source["path"]))
        elif source["required"]:
            pending_required.append(
                {"owner": label, "kind": "train", "expected_paths": source["expected_paths"]}
            )
    loss_rows.sort(key=lambda row: (row["run_label"], row["step"]))

    loss_csv_path = output_dir / "sft_v7_loss_curve.csv"
    loss_svg_path = output_dir / "sft_v7_loss_curve.svg"
    atomic_write_text(loss_csv_path, build_loss_csv(loss_rows))
    atomic_write_text(loss_svg_path, build_loss_svg(loss_rows))
    loggers["rendering"].info(
        "loss artifacts written",
        extra={
            "context": {
                "history_points": len(loss_rows),
                "csv_path": str(loss_csv_path),
                "svg_path": str(loss_svg_path),
            }
        },
    )

    checkpoints: list[dict[str, Any]] = []
    prompt_set_hashes: set[str] = set()
    public_tensor_hashes: set[str] = set()
    dataset_manifest_hashes: set[str] = set()
    for spec in checkpoint_specs:
        public_source = _discover(
            spec.public,
            expected_schema=PUBLIC_SCHEMA,
            loggers=loggers,
        )
        fixed_source = _discover(
            spec.fixed,
            expected_schema=FIXED_SCHEMA,
            loggers=loggers,
        )
        public_summary, public_errors = _public_summary(public_source, spec)
        fixed_summary, fixed_errors = _fixed_summary(fixed_source, spec)
        retention_summary: dict[str, Any] | None = None
        retention_errors: list[str] = []
        if spec.retention is not None:
            retention_source = _discover(
                spec.retention,
                expected_schema=RETENTION_SCHEMA,
                loggers=loggers,
            )
            retention_summary, retention_errors = _retention_summary(
                retention_source, spec
            )
            if retention_source["status"] == "present":
                loaded_paths.append(Path(retention_source["path"]))
            elif retention_source["required"]:
                pending_required.append(
                    {
                        "owner": spec.key,
                        "kind": "retention",
                        "expected_paths": retention_source["expected_paths"],
                    }
                )
        for kind, source in (("public", public_source), ("fixed", fixed_source)):
            if source["status"] == "present":
                loaded_paths.append(Path(source["path"]))
            elif source["required"]:
                pending_required.append(
                    {
                        "owner": spec.key,
                        "kind": kind,
                        "expected_paths": source["expected_paths"],
                    }
                )
        if fixed_summary.get("prompt_set_sha256"):
            prompt_set_hashes.add(str(fixed_summary["prompt_set_sha256"]))
        if public_summary.get("public_tensor_sha256"):
            public_tensor_hashes.add(str(public_summary["public_tensor_sha256"]))
        if public_summary.get("sft_dataset_manifest_sha256"):
            dataset_manifest_hashes.add(
                str(public_summary["sft_dataset_manifest_sha256"])
            )
        row: dict[str, Any] = {
            "key": spec.key,
            "display_name": spec.display_name,
            "expected_step": spec.expected_step,
            "checkpoint_mode": spec.checkpoint_mode,
            "public": public_summary,
            "fixed_samples": fixed_summary,
            "retention": retention_summary,
            "integrity_errors": [*public_errors, *fixed_errors, *retention_errors],
        }
        row["integrity_errors"].extend(_artifact_identity_errors(row))
        row["summary_status"] = _checkpoint_status(
            row, baseline=spec.checkpoint_mode == "pretrain-baseline"
        )
        row["strict_candidate_eligible"] = _strict_candidate_eligible(
            row, baseline=spec.checkpoint_mode == "pretrain-baseline"
        )
        checkpoints.append(row)

    if len(prompt_set_hashes) > 1:
        for row in checkpoints:
            if row["fixed_samples"].get("status") == "present":
                row["integrity_errors"].append(
                    "fixed prompt-set SHA-256 differs between checkpoints"
                )
                row["summary_status"] = "invalid_artifact_identity"
                row["strict_candidate_eligible"] = False
    if len(public_tensor_hashes) > 1:
        for row in checkpoints:
            if row["public"].get("status") == "present":
                row["integrity_errors"].append(
                    "public tensor SHA-256 differs between checkpoints"
                )
                row["summary_status"] = "invalid_artifact_identity"
                row["strict_candidate_eligible"] = False
    if len(dataset_manifest_hashes) > 1:
        for row in checkpoints:
            if row["public"].get("status") == "present":
                row["integrity_errors"].append(
                    "SFT dataset-manifest SHA-256 differs between checkpoints"
                )
                row["summary_status"] = "invalid_artifact_identity"
                row["strict_candidate_eligible"] = False

    strict_candidates = [
        row["key"] for row in checkpoints if row["strict_candidate_eligible"]
    ]
    integrity_error_count = sum(len(row["integrity_errors"]) for row in checkpoints)
    any_auto_failed = any(
        row["summary_status"]
        in {"automatic_public_gates_failed", "automatic_retention_gates_failed"}
        for row in checkpoints
    )
    if integrity_error_count:
        status = "invalid_loaded_artifacts"
    elif pending_required:
        status = "pending_required_artifacts"
    elif any_auto_failed:
        status = "automatic_gates_failed_external_review_pending"
    elif not strict_candidates:
        status = "automatic_diagnostics_complete_external_review_pending"
    else:
        status = "strict_candidate_found_release_review_still_required"

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "status": status,
        "scope": {
            "file_only": True,
            "model_loaded": False,
            "checkpoint_body_reads": 0,
            "tensor_body_reads": 0,
            "raw_corpus_body_reads": 0,
            "sealed_body_reads": 0,
            "sample_bodies_copied_to_summary": False,
        },
        "decision_policy": {
            "missing_metrics_are_pending_not_zero": True,
            "external_pending_is_never_passed": True,
            "automatic_and_external_gates_separate": True,
            "loss_is_auxiliary_not_release_proof": True,
            "release_decision_requires_independent_review": True,
        },
        "release_ready": False,
        "release_ready_reason": "summary tool does not replace independent release review",
        "strict_candidate_keys": strict_candidates,
        "pending_required_artifacts": pending_required,
        "integrity_error_count": integrity_error_count,
        "training_reports": train_reports,
        "loss_history_points": len(loss_rows),
        "checkpoints": checkpoints,
        "outputs": {
            "loss_csv": {
                "path": str(loss_csv_path),
                "sha256": file_sha256(loss_csv_path),
            },
            "loss_svg": {
                "path": str(loss_svg_path),
                "sha256": file_sha256(loss_svg_path),
            },
        },
        "logging": {
            "modules": list(LOG_MODULES),
            "environment_override": "GPT_M020_SUMMARY_LOG_LEVEL_<MODULE>",
            "sample_or_evidence_bodies_logged": False,
            "sensitive_fields_redacted": True,
            "production_default": "INFO",
        },
    }
    comparison_json = output_dir / "checkpoint_comparison.json"
    comparison_markdown = output_dir / "checkpoint_comparison.md"
    atomic_write_json(comparison_json, report)
    atomic_write_text(comparison_markdown, build_markdown(report))
    sha_path = output_dir / "SHA256SUMS.md"
    hash_paths = [
        *loaded_paths,
        loss_csv_path,
        loss_svg_path,
        comparison_json,
        comparison_markdown,
    ]
    atomic_write_text(sha_path, build_sha256_markdown(hash_paths, project_root))
    report["outputs"].update(
        {
            "comparison_json": {
                "path": str(comparison_json),
                "sha256": file_sha256(comparison_json),
            },
            "comparison_markdown": {
                "path": str(comparison_markdown),
                "sha256": file_sha256(comparison_markdown),
            },
            "sha256_manifest": {
                "path": str(sha_path),
                "sha256": file_sha256(sha_path),
                "self_hash_included": False,
            },
        }
    )
    loggers["validation"].info(
        "M020 artifact states validated",
        extra={
            "context": {
                "checkpoint_rows": len(checkpoints),
                "pending_required_artifacts": len(pending_required),
                "integrity_error_count": integrity_error_count,
                "strict_candidate_count": len(strict_candidates),
                "external_pending_promoted_to_passed": False,
                "sealed_body_reads": 0,
            }
        },
    )
    loggers["rendering"].info(
        "comparison artifacts written",
        extra={
            "context": {
                "comparison_json": str(comparison_json),
                "comparison_markdown": str(comparison_markdown),
                "sha256_manifest": str(sha_path),
                "source_report_count": len(loaded_paths),
            }
        },
    )
    return report


def _parse_assignment(value: str, option: str) -> tuple[str, Path]:
    if "=" not in value:
        raise M020SummaryError(f"{option} must use LABEL=PATH")
    label, raw_path = (part.strip() for part in value.split("=", 1))
    if not label or not raw_path:
        raise M020SummaryError(f"{option} must use non-empty LABEL=PATH")
    return label, Path(raw_path)


def _override_train_specs(
    defaults: Mapping[str, ArtifactSpec], values: Sequence[str]
) -> dict[str, ArtifactSpec]:
    result = dict(defaults)
    for value in values:
        label, path = _parse_assignment(value, "--train-report")
        result[label] = ArtifactSpec("train", (path,), required=True)
    return result


def _override_checkpoint_specs(
    defaults: Sequence[CheckpointSpec],
    public_values: Sequence[str],
    fixed_values: Sequence[str],
    retention_values: Sequence[str],
) -> tuple[CheckpointSpec, ...]:
    overrides: dict[str, dict[str, Path]] = {}
    for kind, values, option in (
        ("public", public_values, "--public-eval"),
        ("fixed", fixed_values, "--fixed-samples"),
        ("retention", retention_values, "--retention"),
    ):
        for value in values:
            label, path = _parse_assignment(value, option)
            overrides.setdefault(label, {})[kind] = path
    known = {spec.key for spec in defaults}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise M020SummaryError(
            "checkpoint override labels must be one of "
            + ", ".join(sorted(known))
            + f"; unknown: {', '.join(unknown)}"
        )
    result: list[CheckpointSpec] = []
    for spec in defaults:
        values = overrides.get(spec.key, {})
        public = (
            ArtifactSpec(
                "public",
                (values["public"],),
                required=spec.public.required,
            )
            if "public" in values
            else spec.public
        )
        fixed = (
            ArtifactSpec("fixed", (values["fixed"],), required=True)
            if "fixed" in values
            else spec.fixed
        )
        retention = spec.retention
        if "retention" in values:
            retention = ArtifactSpec(
                "retention",
                (values["retention"],),
                required=bool(spec.retention and spec.retention.required),
            )
        result.append(
            CheckpointSpec(
                key=spec.key,
                display_name=spec.display_name,
                expected_step=spec.expected_step,
                checkpoint_mode=spec.checkpoint_mode,
                public=public,
                fixed=fixed,
                retention=retention,
            )
        )
    return tuple(result)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--train-report", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--public-eval", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--fixed-samples", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--retention", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--log-level", action="append", default=[], metavar="MODULE=LEVEL")
    parser.add_argument("--log-max-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--log-backup-count", type=int, default=3)
    parser.add_argument("--no-console-log", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = generate_run_id("sft-v7-m020-summary")
    loggers: dict[str, logging.Logger] = {}
    try:
        if args.log_max_bytes <= 0 or args.log_backup_count < 0:
            raise M020SummaryError("log size must be positive and backup count non-negative")
        levels = resolve_log_levels(args.log_level)
        loggers = configure_module_loggers(
            args.log_dir,
            run_id,
            levels,
            max_bytes=args.log_max_bytes,
            backup_count=args.log_backup_count,
            console=not args.no_console_log,
        )
        train_specs = _override_train_specs(
            default_train_specs(args.output_dir), args.train_report
        )
        checkpoint_specs = _override_checkpoint_specs(
            default_checkpoint_specs(args.output_dir),
            args.public_eval,
            args.fixed_samples,
            args.retention,
        )
        loggers["orchestrator"].info(
            "M020 summary started",
            extra={
                "context": {
                    "train_report_slots": len(train_specs),
                    "checkpoint_slots": len(checkpoint_specs),
                    "model_loaded": False,
                    "sealed_body_reads": 0,
                }
            },
        )
        report = summarize_m020(
            train_specs=train_specs,
            checkpoint_specs=checkpoint_specs,
            output_dir=args.output_dir,
            project_root=args.project_root,
            run_id=run_id,
            loggers=loggers,
        )
        loggers["orchestrator"].info(
            "M020 summary complete",
            extra={
                "context": {
                    "status": report["status"],
                    "pending_required_artifacts": len(
                        report["pending_required_artifacts"]
                    ),
                    "strict_candidate_count": len(report["strict_candidate_keys"]),
                    "release_ready": False,
                }
            },
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "pending_required_artifacts": len(
                        report["pending_required_artifacts"]
                    ),
                    "integrity_error_count": report["integrity_error_count"],
                    "strict_candidate_keys": report["strict_candidate_keys"],
                    "release_ready": False,
                    "comparison_json": str(args.output_dir / "checkpoint_comparison.json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as error:
        if loggers:
            loggers["orchestrator"].exception(
                "M020 summary failed",
                extra={
                    "context": {
                        "error_type": type(error).__name__,
                        "operation": "summarize_m020",
                        "model_loaded": False,
                        "sealed_body_reads": 0,
                    }
                },
            )
        print(f"M020 summary failed: {error}", file=sys.stderr)
        return 1
    finally:
        close_module_loggers(loggers)


if __name__ == "__main__":
    raise SystemExit(main())
