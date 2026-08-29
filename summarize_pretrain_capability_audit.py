"""Summarize three frozen pretraining audits without reading the test split.

This is a file-only, deterministic comparison stage.  It never imports a
model, opens a checkpoint, reads a token tensor, or follows any data path from
an audit report.  In particular, the sealed test split is never read.

Metric semantics are intentionally strict:

* ``fixed_window_validation_token_loss`` is mean natural-log loss per BPE token
  from the 60-window validation diagnostic;
* ``token_bits_per_token`` is that token loss divided by ``ln(2)``;
* ``validation_bits_per_character`` (BPC) is loaded only from the frozen
  pretraining history.  Under BPE its definition is
  ``token_nats * token_count / original_character_count / ln(2)``.  It must not
  be replaced by token loss divided by ``ln(2)``.

Logging uses independently configurable rotating JSONL modules ``data``,
``validation`` and ``orchestrator``.  Logs default to
``logs/pretrain_capability_summary`` and contain timestamps, run IDs, hashes,
and non-sensitive counts, never generated novel text.  Set
``GPT_LOG_LEVEL_DATA``, ``GPT_LOG_LEVEL_VALIDATION`` or
``GPT_LOG_LEVEL_ORCHESTRATOR`` to DEBUG/INFO/WARNING/ERROR/OFF, or use the
matching CLI options.  The shared formatter redacts common credential fields.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from training_runtime import (
    atomic_write_json,
    atomic_write_text,
    close_module_loggers,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
    resolve_module_log_levels,
)


SCHEMA_VERSION = "pretrain-capability-comparison/v1"
AUDIT_SCHEMA_VERSION = "pretrain-capability-audit/v1"
EXPECTED_STEPS = (250, 5750, 6000)
CLOZE_METRICS = (
    "total_log_probability",
    "mean_token_log_probability",
    "per_character_log_probability",
    "context_lift",
)
FREQUENCY_TIERS = ("high", "low")
DEFAULT_PROTOCOL = Path("docs/pretrain_capability_audit_protocol.md")
DEFAULT_BPC_SOURCE = Path(
    "reports/milestones/016_formal_pretrain_14m/pretrain_v4_report.json"
)
DEFAULT_OUTPUT_DIR = Path("reports/milestones/019_pretrain_capability_audit")
DEFAULT_LOG_DIR = Path("logs/pretrain_capability_summary")

FROZEN_THRESHOLDS = {
    "language_base_minimum_bpc_relative_improvement": 0.20,
    "language_base_maximum_mechanical_degeneration_rate": 0.25,
    "language_base_minimum_ai_review_score": 2.0,
    "plateau_maximum_bpc_improvement": 0.01,
    "plateau_maximum_top1_improvement": 0.005,
    "plateau_maximum_context_lift_mrr_improvement": 0.05,
    "mature_generator_minimum_independent_human_score": 4.0,
}

REQUIRED_PROTOCOL_MARKERS = (
    "单本小说预训练能力审计协议 v1",
    "Step 5750 相比 Step 250 的 Validation BPC 至少改善 20%",
    "机械退化率不高于 25%",
    "平均流畅度与局部连贯度均不低于 2/5",
    "Step 6000 相比 Step 5750 的 Validation BPC 改善小于 0.01",
    "Top-1 改善小于 0.5 个百分点",
    "先验校正 Cloze MRR 改善不超过 0.05",
    "均达到 4/5",
    "`test`：继续封存",
)


class SummaryError(RuntimeError):
    """Raised for a contract, provenance, or frozen-threshold violation."""


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SummaryError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SummaryError(f"JSON root must be an object: {path}")
    return payload


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SummaryError(f"{label} must be finite")
    return result


def _probability(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if not 0.0 <= result <= 1.0:
        raise SummaryError(f"{label} must be between 0 and 1")
    return result


def _validate_protocol(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise SummaryError(f"cannot read frozen protocol {path}: {error}") from error
    missing = [marker for marker in REQUIRED_PROTOCOL_MARKERS if marker not in text]
    if missing:
        raise SummaryError(
            "frozen protocol is missing required threshold markers: "
            + "; ".join(missing)
        )
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "frozen": True,
        "thresholds": dict(FROZEN_THRESHOLDS),
    }


def _load_bpc_history(path: Path) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    rows: list[Mapping[str, Any]]
    source_format: str
    if path.suffix.lower() == ".json":
        payload = _read_json_object(path)
        if payload.get("test_evaluated") not in (False, None):
            raise SummaryError("BPC source reports test evaluation; test must remain sealed")
        raw_rows = payload.get("history")
        if not isinstance(raw_rows, list):
            raise SummaryError("BPC JSON source has no history list")
        rows = raw_rows
        source_format = "pretrain_v4_report_json_history"
    elif path.suffix.lower() == ".csv":
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except OSError as error:
            raise SummaryError(f"cannot read BPC CSV source {path}: {error}") from error
        source_format = "pretrain_v4_loss_csv"
    else:
        raise SummaryError("BPC source must be pretrain_v4_report.json or a CSV history")

    history: dict[int, dict[str, float]] = {}
    for raw in rows:
        try:
            step = int(raw["step"])
        except (KeyError, TypeError, ValueError) as error:
            raise SummaryError("BPC history row has an invalid step") from error
        if step in history:
            raise SummaryError(f"BPC history contains duplicate step {step}")
        history[step] = {
            "validation_bits_per_character": _finite_number(
                raw.get("val_bits_per_character"),
                f"step {step} val_bits_per_character",
            ),
            "training_history_validation_token_loss": _finite_number(
                raw.get("val_loss"), f"step {step} val_loss"
            ),
            "training_history_train_token_loss": _finite_number(
                raw.get("train_loss"), f"step {step} train_loss"
            ),
        }
    missing = [step for step in EXPECTED_STEPS if step not in history]
    if missing:
        raise SummaryError(f"BPC history is missing steps: {missing}")
    return history, {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "format": source_format,
        "test_read": False,
        "definition": (
            "token_nats * token_count / original_character_count / ln(2); "
            "loaded from frozen full-validation training history"
        ),
        "forbidden_substitution": (
            "fixed-window validation token loss / ln(2) is bits per BPE token, not BPC"
        ),
    }


def _cloze_metrics(report: Mapping[str, Any], step: int) -> dict[str, Any]:
    cloze = report.get("cloze")
    if not isinstance(cloze, Mapping):
        raise SummaryError(f"step {step} has no cloze object")
    if int(cloze.get("case_count", -1)) != 12:
        raise SummaryError(f"step {step} must contain exactly 12 formal cloze cases")
    metrics = cloze.get("metrics")
    tiers = cloze.get("by_frequency_tier")
    if not isinstance(metrics, Mapping) or not isinstance(tiers, Mapping):
        raise SummaryError(f"step {step} cloze metrics or frequency tiers are missing")
    output: dict[str, Any] = {
        "case_count": 12,
        "primary_metric": str(cloze.get("primary_diagnostic_metric")),
        "metrics": {},
    }
    for metric in CLOZE_METRICS:
        metric_values = metrics.get(metric)
        if not isinstance(metric_values, Mapping):
            raise SummaryError(f"step {step} is missing cloze metric {metric}")
        tier_values: dict[str, Any] = {}
        for tier in FREQUENCY_TIERS:
            tier_payload = tiers.get(tier)
            if not isinstance(tier_payload, Mapping):
                raise SummaryError(f"step {step} is missing cloze tier {tier}")
            if int(tier_payload.get("case_count", -1)) != 6:
                raise SummaryError(f"step {step} cloze tier {tier} must contain 6 cases")
            tier_metric = tier_payload.get("metrics", {}).get(metric)
            if not isinstance(tier_metric, Mapping):
                raise SummaryError(
                    f"step {step} tier {tier} is missing cloze metric {metric}"
                )
            tier_values[tier] = {
                "top1_accuracy": _probability(
                    tier_metric.get("top1_accuracy"),
                    f"step {step} {tier} {metric} top1",
                ),
                "mean_reciprocal_rank": _probability(
                    tier_metric.get("mean_reciprocal_rank"),
                    f"step {step} {tier} {metric} MRR",
                ),
            }
        output["metrics"][metric] = {
            "overall": {
                "top1_accuracy": _probability(
                    metric_values.get("top1_accuracy"),
                    f"step {step} {metric} top1",
                ),
                "mean_reciprocal_rank": _probability(
                    metric_values.get("mean_reciprocal_rank"),
                    f"step {step} {metric} MRR",
                ),
            },
            **tier_values,
        }
    if output["primary_metric"] != "mean_token_log_probability":
        raise SummaryError(f"step {step} changed the frozen primary cloze metric")
    return output


def _generation_metrics(report: Mapping[str, Any], step: int) -> dict[str, Any]:
    summary = report.get("generation_summary")
    generations = report.get("generations")
    if not isinstance(summary, Mapping) or not isinstance(generations, list):
        raise SummaryError(f"step {step} generation results are incomplete")
    if int(summary.get("sample_count", -1)) != 16 or len(generations) != 16:
        raise SummaryError(f"step {step} must contain exactly 16 fixed generations")
    unique_values = [
        _probability(
            row.get("unique_character_ratio"),
            f"step {step} generation unique_character_ratio",
        )
        for row in generations
    ]
    return {
        "sample_count": 16,
        "empty_rate": _probability(summary.get("empty_rate"), f"step {step} empty"),
        "eos_stop_rate": _probability(
            summary.get("eos_stop_rate"), f"step {step} EOS"
        ),
        "mechanical_degeneration_rate": _probability(
            summary.get("mechanical_degeneration_rate"),
            f"step {step} degeneration",
        ),
        "mean_unique_character_ratio": sum(unique_values) / len(unique_values),
        "mean_four_gram_repetition": _probability(
            summary.get("mean_four_gram_repetition"),
            f"step {step} repetition",
        ),
        "maximum_character_run": int(summary.get("maximum_character_run", -1)),
    }


def _audit_case_contract(report: Mapping[str, Any], step: int) -> str:
    cases = report.get("cloze", {}).get("cases")
    if not isinstance(cases, list):
        raise SummaryError(f"step {step} has no cloze cases list")
    compact = [
        {
            "id": row.get("id"),
            "context": row.get("context"),
            "candidates": row.get("candidates"),
            "correct": row.get("correct"),
            "frequency_tier": row.get("frequency_tier"),
        }
        for row in cases
    ]
    encoded = json.dumps(
        compact, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def _prompt_contract(report: Mapping[str, Any], step: int) -> str:
    generations = report.get("generations")
    if not isinstance(generations, list):
        raise SummaryError(f"step {step} has no generations list")
    prompts = [str(row.get("prompt", "")) for row in generations]
    if any(not prompt for prompt in prompts):
        raise SummaryError(f"step {step} contains an empty fixed prompt")
    import hashlib

    return hashlib.sha256(
        json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _summarize_audit(
    path: Path,
    expected_step: int,
    bpc_history: Mapping[int, Mapping[str, float]],
) -> tuple[dict[str, Any], dict[str, str]]:
    report = _read_json_object(path)
    if report.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise SummaryError(f"step {expected_step} audit schema is not formal v1")
    checkpoint = report.get("checkpoint")
    validation = report.get("validation_diagnostic")
    scope = report.get("scope")
    probe = report.get("probe_provenance")
    if not all(isinstance(item, Mapping) for item in (checkpoint, validation, scope, probe)):
        raise SummaryError(f"step {expected_step} audit is missing provenance sections")
    step = int(checkpoint.get("step", -1))
    if step != expected_step:
        raise SummaryError(f"expected audit step {expected_step}, found {step}")
    checkpoint_sha256 = str(checkpoint.get("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256):
        raise SummaryError(f"step {step} checkpoint SHA-256 is invalid")
    if validation.get("split") != "val":
        raise SummaryError(f"step {step} reads {validation.get('split')!r}; only val is allowed")
    if "test" in str(validation.get("tensor_path", "")).lower():
        raise SummaryError(f"step {step} validation tensor path references test")
    if not scope.get("formal_requested") or not scope.get("formal_status_eligible"):
        raise SummaryError(f"step {step} is not a formal eligible audit")
    usage_checks = probe.get("usage_checks")
    if not isinstance(usage_checks, Mapping) or usage_checks.get("test_false") is not True:
        raise SummaryError(f"step {step} probe provenance does not keep test sealed")
    if probe.get("formal_status_eligible") is not True:
        raise SummaryError(f"step {step} probe artifact is not formal eligible")

    token_loss = _finite_number(validation.get("loss"), f"step {step} token loss")
    perplexity = _finite_number(validation.get("perplexity"), f"step {step} PPL")
    token_top1 = _probability(validation.get("top1_accuracy"), f"step {step} top1")
    bpc = bpc_history[step]
    result = {
        "step": step,
        "audit_path": str(path.resolve()),
        "audit_sha256": file_sha256(path),
        "checkpoint_path_declared": str(checkpoint.get("path", "")),
        "checkpoint_sha256_declared": checkpoint_sha256,
        "audit_status": str(report.get("status", "")),
        "validation_diagnostic": {
            "split": "val",
            "windows_evaluated": int(validation.get("windows_evaluated", -1)),
            "tokens_evaluated": int(validation.get("tokens_evaluated", -1)),
            "fixed_window_validation_token_loss_nats": token_loss,
            "token_bits_per_token_value": token_loss / math.log(2.0),
            "token_perplexity": perplexity,
            "next_token_top1_accuracy": token_top1,
            "validation_bits_per_character": bpc["validation_bits_per_character"],
            "validation_bpc_source": "frozen_full_validation_training_history",
            "training_history_validation_token_loss": bpc[
                "training_history_validation_token_loss"
            ],
            "training_history_train_token_loss": bpc[
                "training_history_train_token_loss"
            ],
            "metric_separation": (
                "token_bits_per_token is fixed-window loss/ln(2); BPC is independently "
                "loaded from full-validation training history and includes token/character ratio"
            ),
        },
        "generation": _generation_metrics(report, step),
        "cloze": _cloze_metrics(report, step),
    }
    contracts = {
        "config_sha256": str(report.get("config", {}).get("canonical_sha256", "")),
        "token_manifest_sha256": str(report.get("data", {}).get("manifest_sha256", "")),
        "tokenizer_sha256": str(report.get("data", {}).get("tokenizer_sha256", "")),
        "validation_tensor_sha256": str(validation.get("tensor_sha256", "")),
        "probe_artifact_sha256": str(probe.get("artifact_sha256", "")),
        "prompts_sha256": str(probe.get("prompts_sha256", "")),
        "case_contract_sha256": _audit_case_contract(report, step),
        "prompt_contract_sha256": _prompt_contract(report, step),
    }
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in contracts.values()):
        raise SummaryError(f"step {step} has an invalid or missing comparison contract hash")
    return result, contracts


def _metric_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_validation = before["validation_diagnostic"]
    after_validation = after["validation_diagnostic"]
    before_generation = before["generation"]
    after_generation = after["generation"]
    before_bpc = before_validation["validation_bits_per_character"]
    after_bpc = after_validation["validation_bits_per_character"]
    cloze: dict[str, Any] = {}
    for metric in CLOZE_METRICS:
        cloze[metric] = {}
        for tier in ("overall", *FREQUENCY_TIERS):
            before_values = before["cloze"]["metrics"][metric][tier]
            after_values = after["cloze"]["metrics"][metric][tier]
            cloze[metric][tier] = {
                "top1_delta": (
                    after_values["top1_accuracy"] - before_values["top1_accuracy"]
                ),
                "mrr_delta": (
                    after_values["mean_reciprocal_rank"]
                    - before_values["mean_reciprocal_rank"]
                ),
            }
    return {
        "from_step": before["step"],
        "to_step": after["step"],
        "delta_direction": "to_minus_from",
        "validation": {
            "fixed_window_token_loss_delta": (
                after_validation["fixed_window_validation_token_loss_nats"]
                - before_validation["fixed_window_validation_token_loss_nats"]
            ),
            "token_bits_per_token_delta": (
                after_validation["token_bits_per_token_value"]
                - before_validation["token_bits_per_token_value"]
            ),
            "token_perplexity_delta": (
                after_validation["token_perplexity"]
                - before_validation["token_perplexity"]
            ),
            "next_token_top1_delta": (
                after_validation["next_token_top1_accuracy"]
                - before_validation["next_token_top1_accuracy"]
            ),
            "validation_bpc_delta": after_bpc - before_bpc,
            "validation_bpc_improvement": before_bpc - after_bpc,
            "validation_bpc_relative_improvement": (
                (before_bpc - after_bpc) / before_bpc
            ),
        },
        "generation": {
            key + "_delta": after_generation[key] - before_generation[key]
            for key in (
                "empty_rate",
                "eos_stop_rate",
                "mechanical_degeneration_rate",
                "mean_unique_character_ratio",
                "mean_four_gram_repetition",
            )
        },
        "cloze": cloze,
    }


def _condition(
    *,
    name: str,
    observed: float,
    operator: str,
    threshold: float,
    passed: bool,
    source: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "passed": passed,
        "source": source,
    }


def _frozen_gate_results(
    checkpoints: Mapping[int, Mapping[str, Any]],
    deltas: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    early = checkpoints[250]
    selected = checkpoints[5750]
    final = checkpoints[6000]
    early_delta = deltas["250_to_5750"]
    plateau_delta = deltas["5750_to_6000"]
    thresholds = FROZEN_THRESHOLDS
    language_conditions = [
        _condition(
            name="validation_bpc_relative_improvement",
            observed=early_delta["validation"]["validation_bpc_relative_improvement"],
            operator=">=",
            threshold=thresholds["language_base_minimum_bpc_relative_improvement"],
            passed=(
                early_delta["validation"]["validation_bpc_relative_improvement"]
                >= thresholds["language_base_minimum_bpc_relative_improvement"]
            ),
            source="frozen_full_validation_training_history",
        ),
        _condition(
            name="validation_next_token_top1_improvement",
            observed=early_delta["validation"]["next_token_top1_delta"],
            operator=">",
            threshold=0.0,
            passed=early_delta["validation"]["next_token_top1_delta"] > 0.0,
            source="fixed_60_window_validation_diagnostic",
        ),
        _condition(
            name="step5750_empty_rate",
            observed=selected["generation"]["empty_rate"],
            operator="==",
            threshold=0.0,
            passed=selected["generation"]["empty_rate"] == 0.0,
            source="fixed_16_prompt_generation",
        ),
        _condition(
            name="step5750_mechanical_degeneration_rate",
            observed=selected["generation"]["mechanical_degeneration_rate"],
            operator="<=",
            threshold=thresholds[
                "language_base_maximum_mechanical_degeneration_rate"
            ],
            passed=(
                selected["generation"]["mechanical_degeneration_rate"]
                <= thresholds["language_base_maximum_mechanical_degeneration_rate"]
            ),
            source="fixed_16_prompt_generation",
        ),
    ]
    language_automatic_passed = all(item["passed"] for item in language_conditions)

    bpc_improvement = plateau_delta["validation"]["validation_bpc_improvement"]
    top1_improvement = plateau_delta["validation"]["next_token_top1_delta"]
    context_lift_mrr_improvement = plateau_delta["cloze"]["context_lift"][
        "overall"
    ]["mrr_delta"]
    plateau_conditions = [
        _condition(
            name="step5750_to_6000_validation_bpc_improvement",
            observed=bpc_improvement,
            operator="<",
            threshold=thresholds["plateau_maximum_bpc_improvement"],
            passed=bpc_improvement < thresholds["plateau_maximum_bpc_improvement"],
            source="frozen_full_validation_training_history",
        ),
        _condition(
            name="step5750_to_6000_next_token_top1_improvement",
            observed=top1_improvement,
            operator="<",
            threshold=thresholds["plateau_maximum_top1_improvement"],
            passed=top1_improvement < thresholds["plateau_maximum_top1_improvement"],
            source="fixed_60_window_validation_diagnostic",
        ),
        _condition(
            name="step5750_to_6000_context_lift_mrr_improvement",
            observed=context_lift_mrr_improvement,
            operator="<=",
            threshold=thresholds[
                "plateau_maximum_context_lift_mrr_improvement"
            ],
            passed=(
                context_lift_mrr_improvement
                <= thresholds["plateau_maximum_context_lift_mrr_improvement"]
            ),
            source="validation_prefix_cloze_context_lift",
        ),
    ]
    plateau_numeric_passed = all(item["passed"] for item in plateau_conditions)
    mechanical_improved = (
        final["generation"]["mechanical_degeneration_rate"]
        < selected["generation"]["mechanical_degeneration_rate"]
    )
    return {
        "language_base": {
            "automatic_conditions": language_conditions,
            "automatic_conditions_passed": language_automatic_passed,
            "manual_condition": {
                "name": "ai_assisted_fluency_and_local_coherence_means",
                "required_minimum_each": thresholds[
                    "language_base_minimum_ai_review_score"
                ],
                "observed": None,
                "status": "not_scored_by_automatic_summarizer",
            },
            "final_status": (
                "manual_review_required"
                if language_automatic_passed
                else "failed_automatic_threshold"
            ),
        },
        "practical_plateau": {
            "automatic_numeric_conditions": plateau_conditions,
            "automatic_numeric_conditions_passed": plateau_numeric_passed,
            "mechanical_observation": {
                "step5750_rate": selected["generation"][
                    "mechanical_degeneration_rate"
                ],
                "step6000_rate": final["generation"][
                    "mechanical_degeneration_rate"
                ],
                "improved": mechanical_improved,
            },
            "manual_condition": {
                "name": "fixed_generation_semantic_results_show_no_consistent_improvement",
                "observed": None,
                "status": "manual_review_required_no_subjective_score_inferred",
            },
            "final_status": (
                "manual_review_required"
                if plateau_numeric_passed
                else "failed_automatic_threshold"
            ),
        },
        "mature_novel_generator": {
            "required_independent_human_score_each": thresholds[
                "mature_generator_minimum_independent_human_score"
            ],
            "observed": None,
            "status": "not_assessed_requires_independent_human_review",
        },
    }


def build_comparison(
    audit_paths: Mapping[int, Path],
    *,
    protocol_path: Path,
    bpc_source_path: Path,
    run_id: str,
    loggers: Mapping[str, logging.Logger],
) -> dict[str, Any]:
    if set(audit_paths) != set(EXPECTED_STEPS):
        raise SummaryError(f"audit paths must contain exactly steps {EXPECTED_STEPS}")
    protocol = _validate_protocol(protocol_path)
    bpc_history, bpc_source = _load_bpc_history(bpc_source_path)
    checkpoints: dict[int, dict[str, Any]] = {}
    contracts: dict[int, dict[str, str]] = {}
    for step in EXPECTED_STEPS:
        checkpoints[step], contracts[step] = _summarize_audit(
            audit_paths[step], step, bpc_history
        )
        loggers["data"].info(
            "formal audit loaded",
            extra={
                "context": {
                    "step": step,
                    "audit_path": str(audit_paths[step]),
                    "audit_sha256": checkpoints[step]["audit_sha256"],
                    "checkpoint_sha256": checkpoints[step][
                        "checkpoint_sha256_declared"
                    ],
                }
            },
        )
    contract_names = tuple(next(iter(contracts.values())).keys())
    mismatches = {
        name: {step: contracts[step][name] for step in EXPECTED_STEPS}
        for name in contract_names
        if len({contracts[step][name] for step in EXPECTED_STEPS}) != 1
    }
    if mismatches:
        raise SummaryError(f"audit comparison contracts differ: {sorted(mismatches)}")
    shared_contracts = {
        name: contracts[EXPECTED_STEPS[0]][name] for name in contract_names
    }
    deltas = {
        "250_to_5750": _metric_delta(checkpoints[250], checkpoints[5750]),
        "5750_to_6000": _metric_delta(checkpoints[5750], checkpoints[6000]),
    }
    gates = _frozen_gate_results(checkpoints, deltas)
    loggers["validation"].info(
        "frozen automatic thresholds evaluated",
        extra={
            "context": {
                "language_base_automatic_passed": gates["language_base"][
                    "automatic_conditions_passed"
                ],
                "plateau_numeric_passed": gates["practical_plateau"][
                    "automatic_numeric_conditions_passed"
                ],
                "manual_scores_inferred": False,
                "test_read": False,
            }
        },
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "scope": {
            "model_type": "single_novel_base_causal_language_model",
            "comparison_steps": list(EXPECTED_STEPS),
            "test_read": False,
            "subjective_manual_scores_inferred": False,
            "fixed_window_token_diagnostic_is_not_bpc": True,
        },
        "inputs": {
            "protocol": protocol,
            "bpc_source": bpc_source,
            "audits": {
                str(step): {
                    "path": str(audit_paths[step].resolve()),
                    "sha256": checkpoints[step]["audit_sha256"],
                }
                for step in EXPECTED_STEPS
            },
            "shared_comparison_contracts": shared_contracts,
        },
        "checkpoints": [checkpoints[step] for step in EXPECTED_STEPS],
        "deltas": deltas,
        "frozen_gates": gates,
        "interpretation_limits": {
            "automatic_outputs": (
                "validation token diagnostics, full-history BPC, mechanical generation "
                "diagnostics, and validation-prefix cloze ranking"
            ),
            "not_automatically_scored": [
                "fluency",
                "local coherence",
                "prompt continuation relevance",
                "character and world consistency",
                "mature generator status",
            ],
        },
    }


def comparison_csv_rows(comparison: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint in comparison["checkpoints"]:
        validation = checkpoint["validation_diagnostic"]
        generation = checkpoint["generation"]
        row: dict[str, Any] = {
            "step": checkpoint["step"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256_declared"],
            "fixed_window_validation_token_loss_nats": validation[
                "fixed_window_validation_token_loss_nats"
            ],
            "token_bits_per_token_value": validation["token_bits_per_token_value"],
            "validation_bits_per_character": validation[
                "validation_bits_per_character"
            ],
            "validation_bpc_source": validation["validation_bpc_source"],
            "token_perplexity": validation["token_perplexity"],
            "next_token_top1_accuracy": validation["next_token_top1_accuracy"],
            "generation_empty_rate": generation["empty_rate"],
            "generation_eos_stop_rate": generation["eos_stop_rate"],
            "generation_mechanical_degeneration_rate": generation[
                "mechanical_degeneration_rate"
            ],
            "generation_mean_unique_character_ratio": generation[
                "mean_unique_character_ratio"
            ],
            "generation_mean_four_gram_repetition": generation[
                "mean_four_gram_repetition"
            ],
        }
        for metric in CLOZE_METRICS:
            for tier in ("overall", *FREQUENCY_TIERS):
                values = checkpoint["cloze"]["metrics"][metric][tier]
                prefix = f"cloze_{metric}_{tier}"
                row[f"{prefix}_top1_accuracy"] = values["top1_accuracy"]
                row[f"{prefix}_mean_reciprocal_rank"] = values[
                    "mean_reciprocal_rank"
                ]
        rows.append(row)
    return rows


def build_csv(comparison: Mapping[str, Any]) -> str:
    rows = comparison_csv_rows(comparison)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(rows[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def build_markdown(comparison: Mapping[str, Any]) -> str:
    lines = [
        "# 单本小说预训练能力自动汇总",
        "",
        "> 本表不包含主观人工评分。固定窗口 Token Loss/PPL/Top-1 与训练历史 BPC 是不同口径。",
        "> Token bits/token = 固定窗口 Token Loss ÷ ln(2)；BPC 只引用完整 Validation 训练历史，绝不由前者冒充。",
        "",
        "| Step | Checkpoint SHA | Token Loss | bits/token | Validation BPC | Token PPL | Next-token Top-1 | Empty | EOS | Degeneration | Unique | 4-gram repeat |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in comparison["checkpoints"]:
        validation = checkpoint["validation_diagnostic"]
        generation = checkpoint["generation"]
        lines.append(
            f"| {checkpoint['step']} | `{checkpoint['checkpoint_sha256_declared'][:12]}…` | "
            f"{validation['fixed_window_validation_token_loss_nats']:.6f} | "
            f"{validation['token_bits_per_token_value']:.6f} | "
            f"{validation['validation_bits_per_character']:.6f} | "
            f"{validation['token_perplexity']:.3f} | "
            f"{validation['next_token_top1_accuracy']:.3%} | "
            f"{generation['empty_rate']:.1%} | {generation['eos_stop_rate']:.1%} | "
            f"{generation['mechanical_degeneration_rate']:.1%} | "
            f"{generation['mean_unique_character_ratio']:.3f} | "
            f"{generation['mean_four_gram_repetition']:.3f} |"
        )
    lines.extend(["", "## Cloze 四类排名", ""])
    for checkpoint in comparison["checkpoints"]:
        lines.extend(
            [
                f"### Step {checkpoint['step']}",
                "",
                "| Metric | Overall Top-1/MRR | High Top-1/MRR | Low Top-1/MRR |",
                "|---|---:|---:|---:|",
            ]
        )
        for metric in CLOZE_METRICS:
            values = checkpoint["cloze"]["metrics"][metric]
            lines.append(
                f"| {metric} | {values['overall']['top1_accuracy']:.1%} / "
                f"{values['overall']['mean_reciprocal_rank']:.3f} | "
                f"{values['high']['top1_accuracy']:.1%} / "
                f"{values['high']['mean_reciprocal_rank']:.3f} | "
                f"{values['low']['top1_accuracy']:.1%} / "
                f"{values['low']['mean_reciprocal_rank']:.3f} |"
            )
        lines.append("")
    lines.extend(["## 冻结门槛", ""])
    for gate_name, gate in comparison["frozen_gates"].items():
        lines.append(f"- `{gate_name}`：**{gate['status'] if 'status' in gate else gate['final_status']}**")
    lines.extend(
        [
            "",
            "自动汇总不会给流畅度、连贯度、承接、人物一致性或成熟生成器状态打分。",
            "",
        ]
    )
    return "\n".join(lines)


def write_comparison_outputs(
    comparison: Mapping[str, Any],
    *,
    json_path: Path,
    csv_path: Path,
    markdown_path: Path,
) -> None:
    atomic_write_text(csv_path, build_csv(comparison))
    atomic_write_text(markdown_path, build_markdown(comparison))
    atomic_write_json(json_path, comparison)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-step250", type=Path, required=True)
    parser.add_argument("--audit-step5750", type=Path, required=True)
    parser.add_argument("--audit-step6000", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--bpc-source", type=Path, default=DEFAULT_BPC_SOURCE)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "comparison.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "comparison.csv",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "comparison.md",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--data-log-level", default="INFO")
    parser.add_argument("--validation-log-level", default="INFO")
    parser.add_argument("--orchestrator-log-level", default="INFO")
    parser.add_argument("--log-max-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--log-backup-count", type=int, default=3)
    parser.add_argument("--no-console-log", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_id = args.run_id or generate_run_id("pretrain-capability-summary")
    loggers: dict[str, logging.Logger] = {}
    try:
        if args.log_max_bytes <= 0 or args.log_backup_count < 0:
            raise SummaryError("log size must be positive and backup count non-negative")
        levels = resolve_module_log_levels(
            {
                "data": args.data_log_level,
                "validation": args.validation_log_level,
                "orchestrator": args.orchestrator_log_level,
            }
        )
        loggers = configure_module_loggers(
            args.log_dir,
            run_id,
            levels,
            max_bytes=args.log_max_bytes,
            backup_count=args.log_backup_count,
            console=not args.no_console_log,
        )
        loggers["orchestrator"].info(
            "pretraining capability comparison started",
            extra={
                "context": {
                    "steps": list(EXPECTED_STEPS),
                    "protocol": str(args.protocol),
                    "bpc_source": str(args.bpc_source),
                    "test_read": False,
                }
            },
        )
        comparison = build_comparison(
            {
                250: args.audit_step250,
                5750: args.audit_step5750,
                6000: args.audit_step6000,
            },
            protocol_path=args.protocol,
            bpc_source_path=args.bpc_source,
            run_id=run_id,
            loggers=loggers,
        )
        write_comparison_outputs(
            comparison,
            json_path=args.output_json,
            csv_path=args.output_csv,
            markdown_path=args.output_markdown,
        )
        loggers["orchestrator"].info(
            "comparison artifacts written",
            extra={
                "context": {
                    "json": str(args.output_json),
                    "json_sha256": file_sha256(args.output_json),
                    "csv": str(args.output_csv),
                    "markdown": str(args.output_markdown),
                    "test_read": False,
                }
            },
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "run_id": run_id,
                    "output_json": str(args.output_json),
                    "output_csv": str(args.output_csv),
                    "output_markdown": str(args.output_markdown),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as error:
        if not loggers:
            try:
                loggers = configure_module_loggers(
                    Path(tempfile.gettempdir()) / "pretrain-summary-fallback-logs",
                    run_id,
                    {"data": "OFF", "validation": "OFF", "orchestrator": "ERROR"},
                    max_bytes=1024 * 1024,
                    backup_count=1,
                    console=True,
                )
            except Exception:
                loggers = {}
        if loggers:
            loggers["orchestrator"].exception(
                "pretraining capability comparison failed",
                extra={
                    "context": {
                        "operation": "summarize_three_formal_audits",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "test_read": False,
                        "remediation": (
                            "Verify all three audit JSON contracts, frozen protocol markers, "
                            "BPC history steps, input hashes, and output permissions; rerun "
                            "with unchanged inputs."
                        ),
                    }
                },
            )
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "remediation": (
                        "Inspect orchestrator JSONL; verify audit/protocol/BPC contracts and "
                        "output permissions."
                    ),
                },
                ensure_ascii=False,
            )
        )
        return 2
    finally:
        if loggers:
            close_module_loggers(loggers)


if __name__ == "__main__":
    raise SystemExit(main())
