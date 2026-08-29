"""Evaluate all 600 SFT v7 public diagnostics without blind-data access.

The evaluator binds the public JSONL to its public-only tensor artifact, then
binds the checkpoint to the frozen Step5750/BPE3000 lineage.  Teacher-forced
loss is computed over every public record.  Deterministic generation produces
task metrics for every record, while reports expose only a small balanced set
of public examples.  Open-expression judgments remain explicitly pending AI
assistance and independent human review; string similarity is not a hard gate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from sample_sft_v7 import (
    DEFAULT_BASELINE_CHECKPOINT,
    SFTV7SamplingError,
    build_conversation_prompt_ids,
    generate_responses_length_bucketed,
    load_bound_tokenizer,
    load_model_bundle,
    load_public_payload,
    reject_forbidden_public_fields,
)
from train_pretrain_v4 import load_config
from train_sft_v4 import collate_records, select_device
from training_runtime import (
    atomic_write_json,
    atomic_write_text,
    close_module_loggers,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
    resolve_module_log_levels,
)


DEFAULT_CONFIG = Path("configs/formal_pretrain_14m_bpe3000.json")
DEFAULT_PUBLIC_JSONL = Path("data/sft/v7/public_diagnostic.jsonl")
DEFAULT_PUBLIC_TENSORS = Path("data/sft/v7/public_diagnostic_tensors.pt")
DEFAULT_CHECKPOINT = Path("runs/sft_v7_vertical/latest.pt")
DEFAULT_REPORT = Path(
    "reports/milestones/020_sft_v7_vertical/public_evaluation.json"
)
DEFAULT_MARKDOWN = Path(
    "reports/milestones/020_sft_v7_vertical/public_evaluation.md"
)
DEFAULT_LOG_DIR = Path("logs/sft_v7_public_eval")

PUBLIC_RECORD_COUNT = 600
CANONICAL_DIMENSIONS = (
    "core_facts_and_corrections",
    "single_evidence_qa",
    "rag_evidence_composition",
    "vertical_chat_multiturn_eos",
    "novel_expression",
    "capability_boundary",
)
DIMENSION_ALIASES = {
    "core_facts_and_corrections": "core_facts_and_corrections",
    "parameter_core_fact_and_correction": "core_facts_and_corrections",
    "single_evidence_qa": "single_evidence_qa",
    "single_passage_grounded_qa": "single_evidence_qa",
    "rag_evidence_composition": "rag_evidence_composition",
    "multi_passage_rag_evidence_composition": "rag_evidence_composition",
    "vertical_chat_multiturn_eos": "vertical_chat_multiturn_eos",
    "novel_expression": "novel_expression",
    "novel_summary_rewrite_short_continuation": "novel_expression",
    "capability_boundary": "capability_boundary",
    "capability_boundary_clarification_evidence_request": "capability_boundary",
}
REFUSAL_MARKERS = (
    "无法确定",
    "不能确定",
    "无法确认",
    "不能确认",
    "资料不足",
    "证据不足",
    "信息不足",
    "需要证据",
    "需要提供",
    "需提供",
    "需要检索",
    "需检索",
    "无法回答",
    "不知道",
    "不清楚",
)
META_PHRASES = (
    "可以先记录问题",
    "作为一个ai",
    "作为ai",
    "作为语言模型",
    "根据你的要求",
    "以下是回答",
    "希望对你有所帮助",
    "如果你愿意",
)


class SFTV7PublicEvaluationError(ValueError):
    """A public-evaluation validation or scoring failure with a safe code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str) -> None:
    raise SFTV7PublicEvaluationError(code, message)


def normalized_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


NORMALIZED_EXACT_DEFINITION = (
    "Unicode text is lowercased and all non-word characters/underscores are removed; "
    "the resulting non-empty prediction must equal the normalized public reference."
)
NORMALIZED_CHAR_F1_DEFINITION = (
    "Character-multiset F1 after the same normalization: overlap is the sum of the "
    "per-character minimum counts. This is a deterministic lexical proxy, not a "
    "semantic-correctness or evidence-support judgment."
)


def normalized_exact_match(prediction: str, reference: str) -> float:
    """Return strict normalized EM; an empty normalized prediction never passes."""

    normalized_prediction = normalized_text(prediction)
    normalized_reference = normalized_text(reference)
    return float(
        bool(normalized_prediction)
        and bool(normalized_reference)
        and normalized_prediction == normalized_reference
    )


def normalized_char_multiset_f1(prediction: str, reference: str) -> float:
    """Return a public, deterministic character-multiset lexical F1 proxy."""

    normalized_prediction = normalized_text(prediction)
    normalized_reference = normalized_text(reference)
    if not normalized_prediction or not normalized_reference:
        return 0.0
    prediction_counts = Counter(normalized_prediction)
    reference_counts = Counter(normalized_reference)
    overlap = sum(
        min(count, reference_counts.get(character, 0))
        for character, count in prediction_counts.items()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(normalized_prediction)
    recall = overlap / len(normalized_reference)
    return 2.0 * precision * recall / (precision + recall)


def canonical_dimension(value: Any) -> str:
    canonical = DIMENSION_ALIASES.get(str(value))
    if canonical is None:
        _fail("unknown_public_dimension", "public record uses an unknown dimension")
    return canonical


def _validate_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        _fail("invalid_evaluation_metadata", f"evaluation field {field} must be a string list")
    return list(value)


def normalize_keypoints(value: Any) -> list[list[str]]:
    """Represent every keypoint as one or more accepted textual alternatives."""

    if value is None:
        return []
    if not isinstance(value, list):
        _fail("invalid_keypoints", "evaluation keypoints must be a list")
    groups: list[list[str]] = []
    for item in value:
        if isinstance(item, str):
            alternatives = [item]
        elif isinstance(item, list) and item and all(isinstance(term, str) for term in item):
            alternatives = list(item)
        elif isinstance(item, Mapping):
            alternatives = item.get("any_of")
            if not isinstance(alternatives, list) or not alternatives or any(
                not isinstance(term, str) for term in alternatives
            ):
                _fail("invalid_keypoints", "keypoint any_of must be a non-empty string list")
            alternatives = list(alternatives)
        else:
            _fail("invalid_keypoints", "keypoint must be text, alternatives, or any_of")
        if any(not normalized_text(term) for term in alternatives):
            _fail("invalid_keypoints", "keypoint alternatives cannot be empty")
        groups.append(alternatives)
    return groups


def validate_evaluation_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("evaluation_metadata_missing", "public record lacks evaluation metadata")
    required = {
        "metric",
        "required_terms",
        "forbidden_terms",
        "known_fact",
        "needs_evidence",
        "evidence_sufficient",
    }
    if required.difference(value):
        _fail("evaluation_metadata_incomplete", "public scoring metadata is incomplete")
    if not isinstance(value["metric"], str) or not value["metric"]:
        _fail("evaluation_metric_invalid", "public scoring metric is invalid")
    required_terms = _validate_string_list(value["required_terms"], "required_terms")
    forbidden_terms = _validate_string_list(value["forbidden_terms"], "forbidden_terms")
    for flag in ("known_fact", "needs_evidence", "evidence_sufficient"):
        if type(value[flag]) is not bool:
            _fail("evaluation_flag_invalid", f"evaluation flag {flag} must be boolean")
    keypoint_value = value.get("keypoints", value.get("required_keypoints"))
    keypoints = normalize_keypoints(keypoint_value)
    # The frozen builder represents its ``metric=keypoints`` contract through
    # required_terms.  Preserve a distinct keypoint aggregate while keeping the
    # source metadata backward-compatible with that frozen schema.
    if not keypoints and value["metric"] == "keypoints":
        keypoints = [[term] for term in required_terms]
    return {
        **deepcopy(dict(value)),
        "required_terms": required_terms,
        "forbidden_terms": forbidden_terms,
        "keypoints": keypoints,
    }


def read_public_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.name != "public_diagnostic.jsonl":
        _fail("public_jsonl_name_mismatch", "public JSONL has an unexpected name")
    reject_forbidden_public_fields({"public_jsonl_path": path})
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SFTV7PublicEvaluationError(
                    "invalid_public_jsonl", f"public JSONL line {line_number} is invalid"
                ) from error
            if not isinstance(record, dict):
                _fail("invalid_public_record", "public JSONL contains a non-object record")
            reject_forbidden_public_fields(record, location="public_record")
            if record.get("split") != "public_diagnostic":
                _fail("public_record_split_mismatch", "public JSONL record has a wrong split")
            records.append(record)
    if not records:
        _fail("public_jsonl_empty", "public JSONL contains no records")
    return records


def validate_public_pair(
    source_records: Sequence[Mapping[str, Any]],
    tensor_payload: Mapping[str, Any],
    *,
    expected_count: int = PUBLIC_RECORD_COUNT,
) -> list[dict[str, Any]]:
    """Bind JSONL bodies to public tensor IDs, dimensions, families, and metadata."""

    tensor_records = list(tensor_payload["public_records"])
    if len(source_records) != expected_count or len(tensor_records) != expected_count:
        _fail("public_record_count_mismatch", "public evaluation must use the full frozen set")
    source_ids = [str(record.get("id", "")) for record in source_records]
    tensor_ids = [str(record.get("id", "")) for record in tensor_records]
    if source_ids != tensor_ids or len(set(source_ids)) != expected_count:
        _fail("public_id_binding_mismatch", "public JSONL and tensor IDs do not bind in order")

    validated: list[dict[str, Any]] = []
    for source, tensor in zip(source_records, tensor_records):
        source_dimension = canonical_dimension(
            source.get("primary_dimension", source.get("dimension"))
        )
        tensor_dimension = canonical_dimension(
            tensor.get("primary_dimension", tensor.get("dimension"))
        )
        source_family = source.get("task_family", source.get("family"))
        tensor_family = tensor.get("task_family", tensor.get("family"))
        if source_dimension != tensor_dimension or source_family != tensor_family:
            _fail("public_record_metadata_mismatch", "public record scoring identity changed")
        source_evaluation = validate_evaluation_metadata(source.get("evaluation"))
        tensor_evaluation = validate_evaluation_metadata(tensor.get("evaluation"))
        if source_evaluation != tensor_evaluation:
            _fail("public_scoring_metadata_mismatch", "public tensor scoring metadata changed")
        messages = source.get("messages")
        if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
            _fail("public_messages_invalid", "public record messages are invalid")
        for index, message in enumerate(messages):
            role = "user" if index % 2 == 0 else "assistant"
            if not isinstance(message, Mapping) or message.get("role") != role:
                _fail("public_role_order_invalid", "public record role order is invalid")
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                _fail("public_message_empty", "public record contains empty content")
        if source.get("answer") is not None and source["answer"] != messages[-1]["content"]:
            _fail("public_answer_mismatch", "public answer and final assistant turn differ")
        validated.append(
            {
                "source": source,
                "tensor": tensor,
                "dimension": source_dimension,
                "task_family": str(source_family),
                "evaluation": source_evaluation,
            }
        )
    observed_dimensions = {item["dimension"] for item in validated}
    if expected_count == PUBLIC_RECORD_COUNT and observed_dimensions != set(CANONICAL_DIMENSIONS):
        _fail("public_dimension_coverage_mismatch", "public set does not cover all six dimensions")
    return validated


def load_and_validate_public_inputs(
    jsonl_path: Path,
    tensor_path: Path,
    *,
    expected_count: int = PUBLIC_RECORD_COUNT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = load_public_payload(tensor_path)
    source_sha = file_sha256(jsonl_path)
    expected_sha = str(payload["source_jsonl_sha256"]["public_diagnostic"])
    if source_sha != expected_sha:
        _fail("public_jsonl_sha_mismatch", "public JSONL does not match its tensor artifact")
    records = read_public_jsonl(jsonl_path)
    paired = validate_public_pair(records, payload, expected_count=expected_count)
    return paired, payload


@torch.no_grad()
def evaluate_teacher_forced_loss(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    pad_token_id: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Compute exact token-weighted CE over all supplied public records."""

    if not records or batch_size <= 0:
        _fail("teacher_forced_arguments_invalid", "teacher-forced evaluation is empty")
    was_training = bool(model.training)
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    try:
        for start in range(0, len(records), batch_size):
            inputs, labels = collate_records(records[start : start + batch_size], pad_token_id)
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits, _ = model(inputs)
            token_count = int((labels != -100).sum().detach().cpu())
            if token_count <= 0:
                _fail("teacher_forced_no_targets", "public batch has no supervised targets")
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            if not bool(torch.isfinite(loss_sum)):
                _fail("teacher_forced_nonfinite", "public teacher-forced loss is non-finite")
            total_nll += float(loss_sum.detach().cpu())
            total_tokens += token_count
    finally:
        model.train(was_training)
    loss = total_nll / total_tokens
    return {
        "records": len(records),
        "supervised_tokens": total_tokens,
        "negative_log_likelihood_sum": total_nll,
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 80 else float("inf"),
    }


def _term_present(normalized_answer: str, term: str) -> bool:
    normalized_term = normalized_text(term)
    return bool(normalized_term) and normalized_term in normalized_answer


def repeated_four_gram_ratio(text: str) -> float:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 4:
        return 0.0
    grams = [compact[index : index + 4] for index in range(len(compact) - 3)]
    return (len(grams) - len(set(grams))) / len(grams)


def score_generated_answer(
    answer: str,
    evaluation: Mapping[str, Any],
    *,
    reference_answer: str = "",
    stopped_on_eos: bool,
    truncated: bool,
    dimension: str,
) -> dict[str, Any]:
    """Score explicit metadata and degeneration signals without fuzzy hard gates."""

    metadata = validate_evaluation_metadata(evaluation)
    normalized_answer = normalized_text(answer)
    required_terms = metadata["required_terms"]
    forbidden_terms = metadata["forbidden_terms"]
    required_matches = [_term_present(normalized_answer, term) for term in required_terms]
    forbidden_matches = [_term_present(normalized_answer, term) for term in forbidden_terms]
    keypoint_matches = [
        any(_term_present(normalized_answer, alternative) for alternative in alternatives)
        for alternatives in metadata["keypoints"]
    ]
    refused = any(marker in normalized_answer for marker in map(normalized_text, REFUSAL_MARKERS))
    meta_phrase = any(marker in normalized_answer for marker in map(normalized_text, META_PHRASES))
    required_pass = all(required_matches) if required_matches else None
    forbidden_pass = not any(forbidden_matches) if forbidden_matches else None
    keypoint_pass = all(keypoint_matches) if keypoint_matches else None
    known_fact = bool(metadata["known_fact"])
    needs_evidence = bool(metadata["needs_evidence"])
    evidence_sufficient = bool(metadata["evidence_sufficient"])
    insufficient_case = needs_evidence and not evidence_sufficient
    capability_mode = str(metadata.get("capability_mode", ""))
    calibration_triplet_id = str(metadata.get("calibration_triplet_id", ""))
    # Known-core, chat and expression records may all carry evidence_sufficient=True,
    # but they are not evidence-recovery probes.  Limit this score to explicit
    # grounded/recovery routing metadata.
    sufficient_case = (
        (not known_fact)
        and evidence_sufficient
        and (capability_mode == "grounded_answer" or needs_evidence)
    )
    sufficient_answer = (
        bool(normalized_answer)
        and (not refused)
        and (required_pass is not False)
        and (keypoint_pass is not False)
    )
    boundary_recovery_case = (
        sufficient_case
        and (bool(calibration_triplet_id) or dimension == "capability_boundary")
    )
    metric = str(metadata["metric"])
    normalized_em = (
        normalized_exact_match(answer, reference_answer) if metric == "exact" else None
    )
    normalized_char_f1 = (
        normalized_char_multiset_f1(answer, reference_answer)
        if metric == "normalized_f1"
        else None
    )
    return {
        "metric": metric,
        "normalized_exact_match": normalized_em,
        "normalized_char_multiset_f1": normalized_char_f1,
        "required_terms": len(required_matches),
        "required_terms_matched": sum(required_matches),
        "required_case_pass": required_pass,
        "forbidden_terms": len(forbidden_matches),
        "forbidden_terms_hit": sum(forbidden_matches),
        "forbidden_case_pass": forbidden_pass,
        "keypoints": len(keypoint_matches),
        "keypoints_matched": sum(keypoint_matches),
        "keypoint_case_pass": keypoint_pass,
        "known_fact_case": known_fact,
        "known_fact_misrefusal": known_fact and refused,
        "refused": refused,
        "insufficient_evidence_case": insufficient_case,
        "insufficient_evidence_stopped": insufficient_case and refused,
        "sufficient_evidence_case": sufficient_case,
        "sufficient_evidence_answered": sufficient_case and sufficient_answer,
        "boundary_recovery_case": boundary_recovery_case,
        "boundary_recovered": boundary_recovery_case and sufficient_answer,
        "stopped_on_eos": bool(stopped_on_eos),
        "empty_answer": not bool(normalized_answer),
        "truncated": bool(truncated),
        "repeated_four_gram_ratio": repeated_four_gram_ratio(answer),
        "mechanical_repetition": repeated_four_gram_ratio(answer) > 0.25,
        "meta_phrase": meta_phrase,
        "open_expression_review": (
            "ai_assisted_and_independent_human_review_pending"
            if dimension == "novel_expression"
            else "not_applicable"
        ),
    }


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def summarize_scores(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required_cases = [score for score in scores if score["required_case_pass"] is not None]
    forbidden_cases = [score for score in scores if score["forbidden_case_pass"] is not None]
    keypoint_cases = [score for score in scores if score["keypoint_case_pass"] is not None]
    known_cases = [score for score in scores if score["known_fact_case"]]
    insufficient = [score for score in scores if score["insufficient_evidence_case"]]
    sufficient = [score for score in scores if score["sufficient_evidence_case"]]
    recovery = [score for score in scores if score["boundary_recovery_case"]]
    exact_cases = [
        score for score in scores if score.get("normalized_exact_match") is not None
    ]
    normalized_f1_cases = [
        score for score in scores if score.get("normalized_char_multiset_f1") is not None
    ]
    return {
        "records": len(scores),
        "required": {
            "terms": sum(int(score["required_terms"]) for score in scores),
            "matched_terms": sum(int(score["required_terms_matched"]) for score in scores),
            "term_recall": _rate(
                sum(int(score["required_terms_matched"]) for score in scores),
                sum(int(score["required_terms"]) for score in scores),
            ),
            "cases": len(required_cases),
            "passed_cases": sum(bool(score["required_case_pass"]) for score in required_cases),
            "case_pass_rate": _rate(
                sum(bool(score["required_case_pass"]) for score in required_cases),
                len(required_cases),
            ),
        },
        "forbidden": {
            "terms": sum(int(score["forbidden_terms"]) for score in scores),
            "hit_terms": sum(int(score["forbidden_terms_hit"]) for score in scores),
            "cases": len(forbidden_cases),
            "violating_cases": sum(
                not bool(score["forbidden_case_pass"]) for score in forbidden_cases
            ),
            "violation_rate": _rate(
                sum(not bool(score["forbidden_case_pass"]) for score in forbidden_cases),
                len(forbidden_cases),
            ),
        },
        "keypoints": {
            "points": sum(int(score["keypoints"]) for score in scores),
            "matched_points": sum(int(score["keypoints_matched"]) for score in scores),
            "recall": _rate(
                sum(int(score["keypoints_matched"]) for score in scores),
                sum(int(score["keypoints"]) for score in scores),
            ),
            "cases": len(keypoint_cases),
            "passed_cases": sum(bool(score["keypoint_case_pass"]) for score in keypoint_cases),
            "case_pass_rate": _rate(
                sum(bool(score["keypoint_case_pass"]) for score in keypoint_cases),
                len(keypoint_cases),
            ),
        },
        "known_fact": {
            "cases": len(known_cases),
            "misrefusals": sum(bool(score["known_fact_misrefusal"]) for score in known_cases),
            "misrefusal_rate": _rate(
                sum(bool(score["known_fact_misrefusal"]) for score in known_cases),
                len(known_cases),
            ),
        },
        "insufficient_evidence": {
            "cases": len(insufficient),
            "correct_stops": sum(
                bool(score["insufficient_evidence_stopped"]) for score in insufficient
            ),
            "correct_stop_rate": _rate(
                sum(bool(score["insufficient_evidence_stopped"]) for score in insufficient),
                len(insufficient),
            ),
        },
        "sufficient_evidence": {
            "cases": len(sufficient),
            "answered": sum(bool(score["sufficient_evidence_answered"]) for score in sufficient),
            "answer_rate": _rate(
                sum(bool(score["sufficient_evidence_answered"]) for score in sufficient),
                len(sufficient),
            ),
        },
        "boundary_recovery": {
            "cases": len(recovery),
            "recovered": sum(bool(score["boundary_recovered"]) for score in recovery),
            "recovery_rate": _rate(
                sum(bool(score["boundary_recovered"]) for score in recovery),
                len(recovery),
            ),
        },
        "reference_metrics": {
            "normalized_exact_match": {
                "definition": NORMALIZED_EXACT_DEFINITION,
                "cases": len(exact_cases),
                "mean": _rate(
                    sum(float(score["normalized_exact_match"]) for score in exact_cases),
                    len(exact_cases),
                ),
            },
            "normalized_char_multiset_f1": {
                "definition": NORMALIZED_CHAR_F1_DEFINITION,
                "cases": len(normalized_f1_cases),
                "mean": _rate(
                    sum(
                        float(score["normalized_char_multiset_f1"])
                        for score in normalized_f1_cases
                    ),
                    len(normalized_f1_cases),
                ),
                "interpretation": "lexical_proxy_only_not_semantic_or_support",
            },
        },
        "generation_quality": {
            "eos_rate": _rate(sum(bool(score["stopped_on_eos"]) for score in scores), len(scores)),
            "empty_rate": _rate(sum(bool(score["empty_answer"]) for score in scores), len(scores)),
            "truncation_rate": _rate(sum(bool(score["truncated"]) for score in scores), len(scores)),
            "mean_repeated_four_gram_ratio": _rate(
                sum(float(score["repeated_four_gram_ratio"]) for score in scores), len(scores)
            ),
            "mechanical_repetition_rate": _rate(
                sum(bool(score["mechanical_repetition"]) for score in scores), len(scores)
            ),
            "meta_phrase_rate": _rate(sum(bool(score["meta_phrase"]) for score in scores), len(scores)),
        },
    }


def _boolean_rate(
    rows: Sequence[Mapping[str, Any]], field: str
) -> float | None:
    return _rate(sum(bool(row.get(field)) for row in rows), len(rows))


def _content_proxy_pass(row: Mapping[str, Any]) -> bool:
    """Prefer an explicit keypoint result, otherwise use required-term routing."""

    if row.get("keypoint_case_pass") is not None:
        return bool(row["keypoint_case_pass"])
    return bool(row.get("required_case_pass"))


def build_gate_results(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute only frozen, deterministic public proxies and expose pending gates.

    The automatic string/behavior gates are useful candidate filters.  They do
    not substitute for the semantic, support, anti-distractor, expression, or
    retention reviews listed in ``external_gates``.
    """

    by_dimension = {
        dimension: [
            row for row in case_results if row.get("dimension") == dimension
        ]
        for dimension in CANONICAL_DIMENSIONS
    }
    core = by_dimension["core_facts_and_corrections"]
    single = by_dimension["single_evidence_qa"]
    rag = by_dimension["rag_evidence_composition"]
    chat = by_dimension["vertical_chat_multiturn_eos"]
    expression = by_dimension["novel_expression"]
    boundary = by_dimension["capability_boundary"]

    core_content = _rate(sum(_content_proxy_pass(row) for row in core), len(core))
    known_core = [row for row in core if row.get("known_fact_case")]
    known_misrefusal = _boolean_rate(known_core, "known_fact_misrefusal")

    single_positive = [row for row in single if row.get("sufficient_evidence_case")]
    single_f1_rows = [
        row
        for row in single_positive
        if row.get("normalized_char_multiset_f1") is not None
    ]
    single_f1 = _rate(
        sum(float(row["normalized_char_multiset_f1"]) for row in single_f1_rows),
        len(single_f1_rows),
    )
    single_support_proxy = _rate(
        sum(_content_proxy_pass(row) for row in single_positive),
        len(single_positive),
    )
    single_insufficient = [
        row for row in single if row.get("insufficient_evidence_case")
    ]
    single_stop = _boolean_rate(single_insufficient, "insufficient_evidence_stopped")

    rag_positive = [row for row in rag if row.get("sufficient_evidence_case")]
    rag_gold_proxy = _rate(
        sum(_content_proxy_pass(row) for row in rag_positive), len(rag_positive)
    )
    rag_support_proxy = _rate(
        sum(
            bool(row.get("required_case_pass"))
            for row in rag_positive
            if row.get("required_case_pass") is not None
        ),
        sum(row.get("required_case_pass") is not None for row in rag_positive),
    )

    chat_multiturn = [row for row in chat if row.get("task_family") == "chat_multiturn"]
    boundary_stop_rows = [
        row for row in boundary if row.get("insufficient_evidence_case")
    ]
    recovery_rows = [row for row in case_results if row.get("boundary_recovery_case")]

    def at_least(value: float | None, threshold: float) -> bool:
        return value is not None and value >= threshold

    def at_most(value: float | None, threshold: float) -> bool:
        return value is not None and value <= threshold

    automatic_gates = [
        {
            "id": "core_required_keypoint_proxy",
            "threshold": ">= 0.80",
            "value": core_content,
            "passed": at_least(core_content, 0.80),
            "source": "public core cases; keypoint pass when present, otherwise required-term pass",
        },
        {
            "id": "core_known_misrefusal",
            "threshold": "<= 0.05",
            "value": known_misrefusal,
            "passed": at_most(known_misrefusal, 0.05),
            "source": "public core cases with evaluation.known_fact=true",
        },
        {
            "id": "single_evidence_lexical_or_support_proxy",
            "threshold": "normalized_char_multiset_f1 >= 0.90 OR required/keypoint support proxy >= 0.90",
            "value": {
                "normalized_char_multiset_f1": single_f1,
                "required_keypoint_support_proxy": single_support_proxy,
            },
            "passed": at_least(single_f1, 0.90)
            or at_least(single_support_proxy, 0.90),
            "source": (
                "positive public single-evidence cases; lexical F1 is a non-semantic proxy, "
                "support proxy checks only declared required/keypoint strings"
            ),
        },
        {
            "id": "single_evidence_insufficient_stop",
            "threshold": ">= 0.90",
            "value": single_stop,
            "passed": at_least(single_stop, 0.90),
            "source": "public single-evidence cases marked needs_evidence and not sufficient",
        },
        {
            "id": "rag_gold_keypoint_proxy",
            "threshold": ">= 0.80",
            "value": rag_gold_proxy,
            "passed": at_least(rag_gold_proxy, 0.80),
            "source": "positive public RAG cases; declared keypoint/required-term pass proxy",
        },
        {
            "id": "rag_required_support_proxy",
            "threshold": ">= 0.90",
            "value": rag_support_proxy,
            "passed": at_least(rag_support_proxy, 0.90),
            "source": (
                "positive public RAG cases; required-term presence only, not citation support"
            ),
        },
        {
            "id": "chat_required_routing_proxy",
            "threshold": ">= 0.90",
            "value": _boolean_rate(
                [row for row in chat if row.get("required_case_pass") is not None],
                "required_case_pass",
            ),
            "passed": at_least(
                _boolean_rate(
                    [row for row in chat if row.get("required_case_pass") is not None],
                    "required_case_pass",
                ),
                0.90,
            ),
            "source": "public chat declared required-term routing proxy",
        },
        {
            "id": "chat_multiturn_carryover_proxy",
            "threshold": ">= 0.80",
            "value": _rate(
                sum(_content_proxy_pass(row) for row in chat_multiturn),
                len(chat_multiturn),
            ),
            "passed": at_least(
                _rate(
                    sum(_content_proxy_pass(row) for row in chat_multiturn),
                    len(chat_multiturn),
                ),
                0.80,
            ),
            "source": "public task_family=chat_multiturn required/keypoint presence proxy",
        },
        {
            "id": "chat_eos",
            "threshold": ">= 0.95",
            "value": _boolean_rate(chat, "stopped_on_eos"),
            "passed": at_least(_boolean_rate(chat, "stopped_on_eos"), 0.95),
            "source": "public chat generation EOS termination",
        },
        {
            "id": "chat_empty",
            "threshold": "= 0",
            "value": _boolean_rate(chat, "empty_answer"),
            "passed": at_most(_boolean_rate(chat, "empty_answer"), 0.0),
            "source": "public chat normalized empty-answer rate",
        },
        {
            "id": "chat_meta_phrase",
            "threshold": "<= 0.01",
            "value": _boolean_rate(chat, "meta_phrase"),
            "passed": at_most(_boolean_rate(chat, "meta_phrase"), 0.01),
            "source": "public chat frozen meta-phrase marker rate",
        },
        {
            "id": "chat_cross_task_refusal_proxy",
            "threshold": "<= 0.05",
            "value": _boolean_rate(chat, "refused"),
            "passed": at_most(_boolean_rate(chat, "refused"), 0.05),
            "source": "public in-domain chat refusal-marker rate (cross-task proxy)",
        },
        {
            "id": "expression_mechanical_repetition",
            "threshold": "<= 0.10",
            "value": _boolean_rate(expression, "mechanical_repetition"),
            "passed": at_most(
                _boolean_rate(expression, "mechanical_repetition"), 0.10
            ),
            "source": "public expression repeated-four-gram mechanical degeneration",
        },
        {
            "id": "boundary_correct_stop",
            "threshold": ">= 0.90",
            "value": _boolean_rate(
                boundary_stop_rows, "insufficient_evidence_stopped"
            ),
            "passed": at_least(
                _boolean_rate(boundary_stop_rows, "insufficient_evidence_stopped"),
                0.90,
            ),
            "source": "public capability-boundary insufficient-evidence cases",
        },
        {
            "id": "boundary_evidence_recovery",
            "threshold": ">= 0.85",
            "value": _boolean_rate(recovery_rows, "boundary_recovered"),
            "passed": at_least(
                _boolean_rate(recovery_rows, "boundary_recovered"), 0.85
            ),
            "source": (
                "public calibration triplets in explicit grounded_answer mode; empty output fails"
            ),
        },
    ]
    external_gates = [
        {
            "id": "core_alias_accuracy",
            "threshold": ">= 0.85",
            "status": "pending",
            "passed": None,
            "required_review": "independent alias-equivalence adjudication report",
        },
        {
            "id": "core_semantic_accuracy_and_unsupported_details",
            "threshold": "semantic fact accuracy >= 0.80 and unsupported detail rate <= 0.05",
            "status": "pending",
            "passed": None,
            "required_review": "human semantic fact and unsupported-detail audit",
        },
        {
            "id": "single_evidence_semantics_and_support",
            "threshold": "semantic correctness >= 0.80 and evidence support >= 0.90",
            "status": "pending",
            "passed": None,
            "required_review": "evidence-span attribution plus independent semantic review",
        },
        {
            "id": "rag_citation_support_and_distractor_resistance",
            "threshold": "citation support >= 0.90 and anti-distractor >= 0.85",
            "status": "pending",
            "passed": None,
            "required_review": "bundle-level citation/entailment and distractor adjudication",
        },
        {
            "id": "chat_format_length_semantics",
            "threshold": ">= 0.90",
            "status": "pending",
            "passed": None,
            "required_review": "format/length instruction and semantic carryover review",
        },
        {
            "id": "expression_ai_and_human_quality",
            "threshold": "fluency, local coherence, prompt following each >= 2.5/5",
            "status": "pending",
            "passed": None,
            "required_review": "AI-assisted rubric scores plus independent human review",
        },
        {
            "id": "boundary_hallucinated_fact_rate",
            "threshold": "<= 0.05",
            "status": "pending",
            "passed": None,
            "required_review": "human factual-hallucination adjudication",
        },
        {
            "id": "pretraining_retention",
            "threshold": "16/16 non-empty, mechanical <= 0.25, AI quality >= 2/5, val BPC degradation <= 0.10",
            "status": "pending",
            "passed": None,
            "required_review": "frozen pretraining retention probe report",
        },
    ]
    automatic_gates_passed = all(bool(gate["passed"]) for gate in automatic_gates)
    external_gates_passed = all(gate["passed"] is True for gate in external_gates)
    return {
        "automatic_gates": automatic_gates,
        "external_gates": external_gates,
        "automatic_gates_passed": automatic_gates_passed,
        "external_gates_passed": external_gates_passed,
        "candidate_eligible": automatic_gates_passed and external_gates_passed,
    }


def stable_case_seed(base_seed: int, record_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{record_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def render_markdown(report: Mapping[str, Any]) -> str:
    def rate(value: Any) -> str:
        return "—" if value is None else f"{float(value):.1%}"

    rows = [
        "# SFT v7 公开诊断",
        "",
        f"Checkpoint step：`{report['checkpoint_step']}`",
        "",
        f"全量公开 teacher-forced loss：`{report['teacher_forced']['loss']:.6f}`；"
        f"perplexity：`{report['teacher_forced']['perplexity']:.3f}`",
        "",
        "开放表达只提供自动退化信号；AI 质量审核与独立真人审核均为待完成，"
        "不以模糊字符串相似度作为硬门。",
        "",
        "| 维度 | 数量 | Required | Forbidden违规 | Keypoint | 已知误拒 | EOS | 空答 | 截断 | 4gram退化 | 元话术 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dimension in CANONICAL_DIMENSIONS:
        summary = report["dimensions"][dimension]
        rows.append(
            f"| {dimension} | {summary['records']} | "
            f"{rate(summary['required']['case_pass_rate'])} | "
            f"{rate(summary['forbidden']['violation_rate'])} | "
            f"{rate(summary['keypoints']['case_pass_rate'])} | "
            f"{rate(summary['known_fact']['misrefusal_rate'])} | "
            f"{rate(summary['generation_quality']['eos_rate'])} | "
            f"{rate(summary['generation_quality']['empty_rate'])} | "
            f"{rate(summary['generation_quality']['truncation_rate'])} | "
            f"{rate(summary['generation_quality']['mechanical_repetition_rate'])} | "
            f"{rate(summary['generation_quality']['meta_phrase_rate'])} |"
        )
    rows.extend(
        [
            "",
            "## 自动候选硬门",
            "",
            "这些结果是可复算的词面/行为代理；不把 EM、字符 F1 或关键词命中夸大为语义正确或证据支持。",
            "",
            "| 门 | 阈值 | 值 | 通过 | 来源 |",
            "|---|---|---|---|---|",
        ]
    )
    escape = lambda value: str(value).replace("|", "\\|").replace("\n", "<br>")
    for gate in report["automatic_gates"]:
        value = gate["value"]
        rendered_value = (
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, Mapping)
            else "—"
            if value is None
            else f"{float(value):.4f}"
        )
        rows.append(
            f"| {gate['id']} | {escape(gate['threshold'])} | {escape(rendered_value)} | "
            f"{'是' if gate['passed'] else '否'} | {escape(gate['source'])} |"
        )
    rows.extend(
        [
            "",
            f"自动门全部通过：**{'是' if report['automatic_gates_passed'] else '否'}**；"
            f"当前候选资格：**{'有' if report['candidate_eligible'] else '无'}**。",
            "",
            "## 外部待审硬门",
            "",
            "| 门 | 阈值 | 状态 | 所需评审/报告 |",
            "|---|---|---|---|",
        ]
    )
    for gate in report["external_gates"]:
        rows.append(
            f"| {gate['id']} | {escape(gate['threshold'])} | {gate['status']} | "
            f"{escape(gate['required_review'])} |"
        )
    rows.extend(
        [
            "",
            "## 指标定义",
            "",
            f"- Normalized EM：{NORMALIZED_EXACT_DEFINITION}",
            f"- Normalized character-multiset F1：{NORMALIZED_CHAR_F1_DEFINITION}",
            "",
            "## 每维少量公开样本",
            "",
            "| 维度 | ID | 问题 | 参考 | 输出 | EOS |",
            "|---|---|---|---|---|---|",
        ]
    )
    for sample in report["samples"]:
        rows.append(
            f"| {sample['dimension']} | {sample['id']} | {escape(sample['question'])} | "
            f"{escape(sample['reference_answer'])} | {escape(sample['generated_answer'])} | "
            f"{'是' if sample['stopped_on_eos'] else '否'} |"
        )
    return "\n".join(rows) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--public-jsonl", type=Path, default=DEFAULT_PUBLIC_JSONL)
    parser.add_argument("--public-tensors", type=Path, default=DEFAULT_PUBLIC_TENSORS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--checkpoint-mode",
        choices=("sft-v7", "pretrain-baseline"),
        default="sft-v7",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--samples-per-dimension", type=int, default=2)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.batch_size <= 0
        or args.generation_batch_size <= 0
        or args.max_new_tokens <= 0
    ):
        _fail("invalid_arguments", "batch size and generation length must be positive")
    if args.temperature <= 0 or args.top_k < 0:
        _fail("invalid_arguments", "temperature or top-k is invalid")
    if not 1 <= args.samples_per_dimension <= 3:
        _fail("sample_display_limit_invalid", "display at most three samples per dimension")
    reject_forbidden_public_fields(
        {
            "public_jsonl_path": args.public_jsonl,
            "public_tensor_path": args.public_tensors,
            "checkpoint_path": args.checkpoint,
        },
        location="arguments",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.checkpoint_mode == "pretrain-baseline" and args.checkpoint == DEFAULT_CHECKPOINT:
        args.checkpoint = DEFAULT_BASELINE_CHECKPOINT
    validate_args(args)
    run_id = generate_run_id("sft-v7-public-eval")
    config = load_config(args.config)
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
        paired, payload = load_and_validate_public_inputs(
            args.public_jsonl, args.public_tensors
        )
        tokenizer = load_bound_tokenizer(payload)
        device = select_device(args.device)
        model, checkpoint, provenance = load_model_bundle(
            args.config, args.checkpoint, payload, device, args.checkpoint_mode
        )
        checkpoint_sha256 = file_sha256(args.checkpoint)
        loggers["checkpoint"].info(
            "checkpoint validated step=%d sha256=%s base_sha256=%s device=%s",
            provenance["step"],
            checkpoint_sha256,
            provenance["base_checkpoint_sha256"],
            device,
        )
        teacher_forced = evaluate_teacher_forced_loss(
            model,
            [item["tensor"] for item in paired],
            int(payload["special_token_ids"]["<PAD>"]),
            args.batch_size,
            device,
        )
        loggers["evaluation"].info(
            "full public teacher-forced evaluation records=%d supervised_tokens=%d loss=%.6f",
            teacher_forced["records"],
            teacher_forced["supervised_tokens"],
            teacher_forced["loss"],
        )

        case_results: list[dict[str, Any]] = []
        samples: list[dict[str, Any]] = []
        samples_by_dimension: defaultdict[str, int] = defaultdict(int)
        prompt_ids_by_case = [
            build_conversation_prompt_ids(
                tokenizer,
                item["source"]["messages"][:-1],
                payload["special_token_ids"],
            )
            for item in paired
        ]
        case_seeds = [
            stable_case_seed(args.seed, str(item["source"]["id"]))
            for item in paired
        ]
        loggers["generation"].info(
            "public batched generation start records=%d generation_batch_size=%d "
            "distinct_prompt_lengths=%d batching_policy=%s",
            len(paired),
            args.generation_batch_size,
            len({len(prompt_ids) for prompt_ids in prompt_ids_by_case}),
            "nearby_length_sorted_left_padded_bounded_chunks",
        )
        generated_results = generate_responses_length_bucketed(
            model,
            prompt_ids_by_case,
            tokenizer,
            payload["special_token_ids"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seeds=case_seeds,
            device=device,
            generation_batch_size=args.generation_batch_size,
        )
        for index, (item, generated) in enumerate(
            zip(paired, generated_results), 1
        ):
            source = item["source"]
            messages = source["messages"]
            score = score_generated_answer(
                generated["generated_text"],
                item["evaluation"],
                reference_answer=messages[-1]["content"],
                stopped_on_eos=generated["stopped_on_eos"],
                truncated=generated["truncated"],
                dimension=item["dimension"],
            )
            case_results.append(
                {
                    "id": source["id"],
                    "dimension": item["dimension"],
                    "task_family": item["task_family"],
                    **score,
                }
            )
            if samples_by_dimension[item["dimension"]] < args.samples_per_dimension:
                samples.append(
                    {
                        "id": source["id"],
                        "dimension": item["dimension"],
                        "task_family": item["task_family"],
                        "question": messages[-2]["content"],
                        "reference_answer": messages[-1]["content"],
                        "generated_answer": generated["generated_text"],
                        "stopped_on_eos": generated["stopped_on_eos"],
                        "truncated": generated["truncated"],
                        "score": score,
                    }
                )
                samples_by_dimension[item["dimension"]] += 1
            if index % 50 == 0 or index == len(paired):
                loggers["generation"].info(
                    "public generation progress completed=%d total=%d eos=%d empty=%d truncated=%d",
                    index,
                    len(paired),
                    sum(bool(row["stopped_on_eos"]) for row in case_results),
                    sum(bool(row["empty_answer"]) for row in case_results),
                    sum(bool(row["truncated"]) for row in case_results),
                )

        dimensions = {
            dimension: summarize_scores(
                [row for row in case_results if row["dimension"] == dimension]
            )
            for dimension in CANONICAL_DIMENSIONS
        }
        overall = summarize_scores(case_results)
        gates = build_gate_results(case_results)
        report = {
            "schema_version": "sft-v7-public-evaluation/v1",
            "status": "diagnostic_complete_candidate_pending_or_ineligible",
            "run_id": run_id,
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_step": int(checkpoint["step"]),
            "checkpoint_mode": args.checkpoint_mode,
            "public_jsonl_path": str(args.public_jsonl),
            "public_jsonl_sha256": file_sha256(args.public_jsonl),
            "public_tensor_path": str(args.public_tensors),
            "public_tensor_sha256": file_sha256(args.public_tensors),
            "tokenizer_sha256": payload["tokenizer_sha256"],
            "bpe_token_manifest_sha256": payload["bpe_token_manifest_sha256"],
            "sft_dataset_manifest_sha256": payload[
                "sft_dataset_manifest_sha256"
            ],
            "device": str(device),
            "teacher_forced": teacher_forced,
            "generation_configuration": {
                "records": len(case_results),
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "generation_batch_size": args.generation_batch_size,
                "batching_policy": (
                    "nearby_length_sorted_left_padded_bounded_chunks"
                ),
                "seed": args.seed,
                "seed_policy": "sha256(base_seed, public_record_id)",
                "masked_special_tokens": [
                    "<UNK>",
                    "<BOS>",
                    "<USER>",
                    "<ASSISTANT>",
                    "<PAD>",
                ],
                "eos_allowed": True,
            },
            "overall": overall,
            "dimensions": dimensions,
            "metric_definitions": {
                "normalized_exact_match": NORMALIZED_EXACT_DEFINITION,
                "normalized_char_multiset_f1": NORMALIZED_CHAR_F1_DEFINITION,
                "scope_warning": (
                    "Both are deterministic lexical proxies; neither proves semantic "
                    "correctness, evidence support, or absence of hallucination."
                ),
            },
            **gates,
            "open_expression_review": {
                "automatic_signals": "complete",
                "ai_assisted_quality_review": "pending",
                "independent_human_review": "pending",
                "fuzzy_string_similarity_hard_gate": False,
            },
            "case_results": case_results,
            "sample_display_policy": {
                "per_dimension": args.samples_per_dimension,
                "maximum_total": len(CANONICAL_DIMENSIONS) * args.samples_per_dimension,
            },
            "samples": samples,
        }
        atomic_write_json(args.report, report)
        atomic_write_text(args.markdown, render_markdown(report))
        loggers["evaluation"].info(
            "public behavior summary records=%d eos=%.4f empty=%.4f truncated=%.4f "
            "repetition=%.4f meta=%.4f displayed_samples=%d",
            overall["records"],
            overall["generation_quality"]["eos_rate"],
            overall["generation_quality"]["empty_rate"],
            overall["generation_quality"]["truncation_rate"],
            overall["generation_quality"]["mechanical_repetition_rate"],
            overall["generation_quality"]["meta_phrase_rate"],
            len(samples),
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "checkpoint_step": report["checkpoint_step"],
                    "public_records": overall["records"],
                    "teacher_forced_loss": teacher_forced["loss"],
                    "eos_rate": overall["generation_quality"]["eos_rate"],
                    "report": str(args.report),
                    "markdown": str(args.markdown),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (SFTV7SamplingError, SFTV7PublicEvaluationError) as error:
        loggers["evaluation"].error(
            "public evaluation failed error_code=%s error_type=%s",
            error.code,
            type(error).__name__,
        )
        raise
    except Exception as error:
        loggers["evaluation"].error(
            "public evaluation failed error_code=unexpected_failure error_type=%s",
            type(error).__name__,
        )
        raise
    finally:
        close_module_loggers(loggers)


if __name__ == "__main__":
    raise SystemExit(main())
