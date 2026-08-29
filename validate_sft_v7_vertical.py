"""Validate the frozen, physically split SFT v7 vertical dataset.

Default operation opens only ``train.jsonl``, ``val.jsonl``, and
``public_diagnostic.jsonl``.  The sealed build artifact is unreachable unless
both an explicit path and ``--allow-sealed-build-validation`` are supplied.
Even in that one build-time mode, outputs contain only aggregate counts,
hashes, identifiers, and risk codes -- never record bodies.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from bpe_tokenizer import BPETokenizer
from training_runtime import (
    atomic_write_json,
    close_module_loggers,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
    resolve_module_log_levels,
)


SCHEMA_VERSION = "sft_v7_vertical/1.0"
MANIFEST_SCHEMA_VERSION = "sft-v7-vertical-manifest/v1"
REPORT_SCHEMA_VERSION = "sft-v7-vertical-validation/v1"
CONTEXT_TOKENS = 512

DEFAULT_DATA_DIR = Path("data/sft/v7")
DEFAULT_TRAIN = DEFAULT_DATA_DIR / "train.jsonl"
DEFAULT_VAL = DEFAULT_DATA_DIR / "val.jsonl"
DEFAULT_PUBLIC = DEFAULT_DATA_DIR / "public_diagnostic.jsonl"
DEFAULT_CORPUS = Path("data/cloud_v4/train.txt")
DEFAULT_TOKENIZER = Path("data/scaling_a/bpe_3000/tokenizer.json")
DEFAULT_REPORT = Path(
    "reports/milestones/020_sft_v7_vertical/validation_report.json"
)
DEFAULT_MANIFEST = Path(
    "reports/milestones/020_sft_v7_vertical/validation_manifest.json"
)
DEFAULT_LOG_DIR = Path("logs/sft_v7_vertical_validation")

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
PUBLIC_SPLIT = "public_diagnostic"
SEALED_SPLIT = "sealed_test"
DEFAULT_VALIDATION_SPLITS = (TRAIN_SPLIT, VAL_SPLIT, PUBLIC_SPLIT)
ALL_SPLITS = (*DEFAULT_VALIDATION_SPLITS, SEALED_SPLIT)

# These values are the builder/catalog wire contract.  Keep the public aliases
# used by the validator tests, but never translate records from an older
# ontology silently.
PARAMETRIC_CORE = "parameter_core_fact_and_correction"
GROUNDED_SINGLE = "single_passage_grounded_qa"
RAG_MULTI = "multi_passage_rag_evidence_composition"
VERTICAL_CHAT = "vertical_chat_multiturn_eos"
NOVEL_EXPRESSION = "novel_summary_rewrite_short_continuation"
CAPABILITY_BOUNDARY = "capability_boundary_clarification_evidence_request"

DIMENSION_TOTAL_TARGETS: Mapping[str, int] = {
    PARAMETRIC_CORE: 1_800,
    GROUNDED_SINGLE: 3_200,
    RAG_MULTI: 1_400,
    VERTICAL_CHAT: 1_800,
    NOVEL_EXPRESSION: 1_300,
    CAPABILITY_BOUNDARY: 500,
}
SPLIT_TARGETS: Mapping[str, int] = {
    TRAIN_SPLIT: 8_000,
    VAL_SPLIT: 800,
    PUBLIC_SPLIT: 600,
    SEALED_SPLIT: 600,
}
SPLIT_PERCENTAGES: Mapping[str, int] = {
    TRAIN_SPLIT: 80,
    VAL_SPLIT: 8,
    PUBLIC_SPLIT: 6,
    SEALED_SPLIT: 6,
}
DIMENSION_SPLIT_TARGETS: Mapping[str, Mapping[str, int]] = {
    dimension: {
        split: total * percent // 100
        for split, percent in SPLIT_PERCENTAGES.items()
    }
    for dimension, total in DIMENSION_TOTAL_TARGETS.items()
}

MIN_MULTITURN_BY_SPLIT: Mapping[str, int] = {
    TRAIN_SPLIT: 960,
    VAL_SPLIT: 96,
    PUBLIC_SPLIT: 72,
    SEALED_SPLIT: 72,
}
MIN_RAG_BUNDLES_BY_SPLIT: Mapping[str, int] = {
    TRAIN_SPLIT: 800,
    VAL_SPLIT: 80,
    PUBLIC_SPLIT: 60,
    SEALED_SPLIT: 60,
}
MIN_CALIBRATION_TRIPLETS_BY_SPLIT: Mapping[str, int] = {
    TRAIN_SPLIT: 100,
    VAL_SPLIT: 10,
    PUBLIC_SPLIT: 10,
    SEALED_SPLIT: 10,
}

REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "id",
        "split",
        "primary_dimension",
        "task_family",
        "semantic_group",
        "fact_id",
        "generalization_policy",
        "question",
        "answer",
        "messages",
        "evidence",
        "evaluation",
        "generation",
        "encoding_audit",
        "coverage",
        "provenance",
        "review",
    }
)

SPECIAL_TOKENS = ("<BOS>", "<USER>", "<ASSISTANT>", "<EOS>", "<PAD>")
APPROVED_REVIEW_STATUSES = frozenset(
    {"approved", "independently_verified", "ai_reviewed_and_rule_checked"}
)
EVIDENCE_STATUSES = frozenset(
    {
        "verified_train_corpus",
        "reviewed_exact_train_lines",
        "verified_exact_train_line",
        "verified_exact_train_lines",
        "insufficient_for_target",
        "evidence_not_supplied",
        "insufficient",
        "not_applicable",
        "curated_boundary",
    }
)
SUPPORT_STATUSES = frozenset({"supported", "insufficient_evidence", "not_applicable"})
CAPABILITY_MODES = frozenset(
    {"known_core", "needs_evidence", "grounded_answer", "interaction", "expression"}
)
CALIBRATION_MODES = frozenset({"known_core", "needs_evidence", "grounded_answer"})

ALLOWED_METRICS: Mapping[str, frozenset[str]] = {
    PARAMETRIC_CORE: frozenset(
        {
            "keypoints",
            "behavior",
            "entity_fact_accuracy",
            "alias_accuracy",
            "core_correction_accuracy",
            "core_fact_accuracy",
        }
    ),
    GROUNDED_SINGLE: frozenset(
        {
            "normalized_f1",
            "behavior",
            "exact_match",
            "token_f1",
            "semantic_correctness",
            "evidence_support",
            "abstention_accuracy",
            "grounded_semantic_accuracy",
        }
    ),
    RAG_MULTI: frozenset(
        {
            "keypoints",
            "behavior",
            "semantic_correctness",
            "citation_support",
            "distractor_robustness",
            "abstention_accuracy",
            "rag_evidence_composition",
        }
    ),
    VERTICAL_CHAT: frozenset(
        {
            "behavior",
            "routing_accuracy",
            "multiturn_coreference",
            "format_adherence",
            "eos",
            "conversation_quality",
        }
    ),
    NOVEL_EXPRESSION: frozenset(
        {
            "behavior",
            "exact",
            "fluency_rubric",
            "coherence_rubric",
            "prompt_relevance",
            "expression_quality",
        }
    ),
    CAPABILITY_BOUNDARY: frozenset(
        {
            "behavior",
            "boundary_accuracy",
            "evidence_recovery",
            "clarification_accuracy",
        }
    ),
}
ALLOWED_MODES_BY_DIMENSION: Mapping[str, frozenset[str]] = {
    PARAMETRIC_CORE: frozenset({"known_core"}),
    GROUNDED_SINGLE: frozenset({"grounded_answer", "needs_evidence"}),
    RAG_MULTI: frozenset({"grounded_answer", "needs_evidence"}),
    VERTICAL_CHAT: frozenset({"interaction"}),
    NOVEL_EXPRESSION: frozenset({"expression", "grounded_answer"}),
    CAPABILITY_BOUNDARY: frozenset({"needs_evidence"}),
}

META_MARKERS = (
    "可以先",
    "先先",
    "原问题是",
    "现只做局部证据核验",
    "当前证据片段",
    "正确，证据支持",
    "审核占位",
    "训练样本",
)
REFUSAL_MARKERS = (
    "无法确认",
    "资料不足",
    "不能直接回答",
    "无法凭空",
    "需要检索",
    "请提供证据",
    "请提供材料",
)
NEEDS_EVIDENCE_MARKERS = (
    "检索",
    "证据",
    "索引",
    "材料",
    "需要补充",
    "需要包含",
    "需要检索",
    "需要证据",
    "请提供证据",
    "请提供材料",
    "无法确认",
    "资料不足",
)
PROJECT_CONCEPT_MARKERS = (
    "Block Size",
    "Embedding",
    "Tokenizer",
    "Token是什么",
    "监督微调是什么",
    "模型训练",
    "注意力机制",
    "学习计划",
    "日志应该",
    "PyTorch",
)
GENERAL_ENCYCLOPEDIA_MARKERS = (
    "绿巨人",
    "爱因斯坦",
    "牛顿",
    "世界首都",
    "百科知识",
    "游乐园在哪里",
)
PUNCTUATION_PAIRS = (("（", "）"), ("(", ")"), ("《", "》"), ("【", "】"))
EVIDENCE_TERM_EQUIVALENTS = {
    # Frozen aliases are backed by reviewed occurrences in formal train.txt.
    "萧薰儿": ("萧薰儿", "薰儿"),
    "药尘": ("药尘", "药老"),
    "美杜莎": ("美杜莎", "彩鳞"),
}
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:+-]{1,160}$")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
POSITIVE_MATH_PROMPT_PATTERNS = (
    re.compile(r"(?:计算|求解|解答).{0,20}(?:算式|方程|结果|等于多少)"),
    re.compile(r"(?:加法|减法|乘法|除法|数学题|求导|积分|几何证明)"),
)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(*parts: object) -> str:
    return text_sha256("|".join(str(part) for part in parts))


def canonical_question(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s，。！？、；：“”‘’（）()《》【】\[\]：:,.!?;'\"`]+", "", normalized)


def opening_fingerprint(answer: str, *, width: int = 12) -> str:
    prefix = canonical_question(answer)[:width]
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:16]


def _safe_identifier(value: Any) -> bool:
    return isinstance(value, str) and SAFE_IDENTIFIER.fullmatch(value) is not None


def _is_positive_math_prompt(text: str) -> bool:
    """Match actual math requests without flagging novel words such as 无数."""

    return any(pattern.search(text) for pattern in POSITIVE_MATH_PROMPT_PATTERNS)


def read_jsonl_split(path: Path, expected_split: str) -> list[dict[str, Any]]:
    """Load one explicitly authorized physical split without logging its body."""

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path.name}:{line_number} is invalid JSON"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} is not a JSON object")
            if value.get("split") != expected_split:
                # Do not include user-controlled split text in the exception.
                raise ValueError(f"{path.name}:{line_number} split does not match its file")
            records.append(value)
    return records


def encoding_metrics(
    record: Mapping[str, Any], tokenizer: BPETokenizer
) -> dict[str, int]:
    """Reproduce the builder's role masking and assistant-EOS accounting."""

    sequence_length = 1  # BOS
    supervised_tokens = 0
    final_answer_tokens = 0
    assistant_turns = 0
    messages = record["messages"]
    for index, message in enumerate(messages):
        content_tokens = len(tokenizer.encode(str(message["content"])))
        sequence_length += 1 + content_tokens  # one role token
        if message["role"] == "assistant":
            sequence_length += 1  # exactly one EOS after every assistant turn
            supervised_tokens += content_tokens + 1
            assistant_turns += 1
            if index == len(messages) - 1:
                final_answer_tokens = content_tokens
    return {
        "sequence_tokens": sequence_length,
        "supervised_tokens": supervised_tokens,
        "last_answer_tokens": final_answer_tokens,
        "assistant_turns": assistant_turns,
        "eos_targets": assistant_turns,
        "masked_user_and_role_tokens": sequence_length - supervised_tokens,
    }


def sequence_metrics(record: Mapping[str, Any], tokenizer: BPETokenizer) -> tuple[int, int, int]:
    """Backward-compatible tuple view used by callers and tests."""

    metrics = encoding_metrics(record, tokenizer)
    return (
        metrics["sequence_tokens"],
        metrics["supervised_tokens"],
        metrics["last_answer_tokens"],
    )


def evidence_items(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = record.get("evidence")
    if isinstance(value, Mapping):
        value = value.get("chunks")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _combined_text(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        contents = [
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, Mapping)
        ]
        if contents:
            return "\n".join(contents)
    return f"{record.get('question', '')}\n{record.get('answer', '')}"


def _risk(
    risks: list[dict[str, str]],
    record: Mapping[str, Any],
    code: str,
    *,
    severity: str = "P0",
) -> None:
    """Record only a stable id and code; never copy record content."""

    risks.append(
        {
            "id": str(record.get("id", ""))[:128],
            "split": str(record.get("split", ""))[:32],
            "code": code,
            "severity": severity,
        }
    )


def _roles_and_eos_contract_ok(record: Mapping[str, Any], risks: list[dict[str, str]]) -> bool:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
        _risk(risks, record, "invalid_message_count")
        return False
    expected_roles = ["user" if index % 2 == 0 else "assistant" for index in range(len(messages))]
    actual_roles: list[Any] = []
    for message in messages:
        if not isinstance(message, Mapping):
            _risk(risks, record, "invalid_message_object")
            return False
        actual_roles.append(message.get("role"))
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            _risk(risks, record, "empty_message_content")
            return False
        if any(token in content for token in SPECIAL_TOKENS):
            _risk(risks, record, "literal_special_token_in_content")
    if actual_roles != expected_roles:
        _risk(risks, record, "invalid_role_alternation")
        return False
    if record.get("question") != messages[-2].get("content"):
        _risk(risks, record, "question_last_user_mismatch")
    if record.get("answer") != messages[-1].get("content"):
        _risk(risks, record, "answer_last_assistant_mismatch")
    # EOS is appended by the encoder.  Literal EOS would create two EOS tokens.
    if str(record.get("answer", "")).endswith("<EOS>"):
        _risk(risks, record, "literal_eos_before_encoder_eos")
    return True


def _validate_evaluation_contract(
    record: Mapping[str, Any],
    risks: list[dict[str, str]],
) -> tuple[str, str, str]:
    dimension = str(record.get("primary_dimension", ""))
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, Mapping):
        _risk(risks, record, "missing_evaluation_contract")
        return "", "", ""
    metric = evaluation.get("metric")
    track = record.get("generalization_policy")
    required_terms = evaluation.get("required_terms")
    forbidden_terms = evaluation.get("forbidden_terms")
    if metric not in ALLOWED_METRICS.get(dimension, frozenset()):
        _risk(risks, record, "metric_not_allowed_for_dimension")
    if not isinstance(track, str) or not track.strip():
        _risk(risks, record, "missing_evaluation_track")
        track = ""
    if not isinstance(required_terms, list) or any(
        not isinstance(term, str) or not term for term in required_terms
    ):
        _risk(risks, record, "invalid_required_terms_contract")
        required_terms = []
    if not isinstance(forbidden_terms, list) or any(
        not isinstance(term, str) or not term for term in forbidden_terms
    ):
        _risk(risks, record, "invalid_forbidden_terms_contract")
        forbidden_terms = []
    if set(required_terms).intersection(forbidden_terms):
        _risk(risks, record, "evaluation_terms_overlap")
    answer = str(record.get("answer", ""))
    if any(term not in answer for term in required_terms):
        _risk(risks, record, "required_term_missing_from_answer")
    if any(term in answer for term in forbidden_terms):
        _risk(risks, record, "forbidden_term_present_in_answer")

    known_fact = evaluation.get("known_fact")
    needs_evidence = evaluation.get("needs_evidence")
    evidence_sufficient = evaluation.get("evidence_sufficient")
    if not all(
        isinstance(value, bool)
        for value in (known_fact, needs_evidence, evidence_sufficient)
    ):
        _risk(risks, record, "invalid_evaluation_boolean_contract")
        known_fact = needs_evidence = evidence_sufficient = False
    if known_fact and (needs_evidence or not evidence_sufficient):
        _risk(risks, record, "known_fact_calibration_conflict")
    if needs_evidence and (known_fact or evidence_sufficient):
        _risk(risks, record, "needs_evidence_calibration_conflict")
    if dimension == PARAMETRIC_CORE:
        mode = "known_core"
        if (known_fact, needs_evidence, evidence_sufficient) != (True, False, True):
            _risk(risks, record, "core_evaluation_contract_mismatch")
    elif dimension in {GROUNDED_SINGLE, RAG_MULTI}:
        mode = "needs_evidence" if needs_evidence else "grounded_answer"
        allowed = {(False, False, True), (False, True, False)}
        if (known_fact, needs_evidence, evidence_sufficient) not in allowed:
            _risk(risks, record, "grounding_evaluation_contract_mismatch")
    elif dimension == VERTICAL_CHAT:
        mode = "interaction"
        if (known_fact, needs_evidence, evidence_sufficient) not in {
            (False, False, False),
            (False, False, True),
        }:
            _risk(risks, record, "interaction_evaluation_contract_mismatch")
    elif dimension == NOVEL_EXPRESSION:
        mode = "expression"
        if known_fact or needs_evidence:
            _risk(risks, record, "expression_evaluation_contract_mismatch")
    elif dimension == CAPABILITY_BOUNDARY:
        mode = "needs_evidence"
        if (known_fact, needs_evidence, evidence_sufficient) != (False, True, False):
            _risk(risks, record, "boundary_evaluation_contract_mismatch")
    else:
        mode = ""

    # Transitional duplicate fields are accepted only when they agree with the
    # authoritative builder fields above; the validator never relies on them.
    declared_mode = evaluation.get("capability_mode")
    if declared_mode is not None and declared_mode != mode:
        _risk(risks, record, "declared_capability_mode_mismatch")
    declared_track = evaluation.get("evaluation_track")
    if declared_track is not None and declared_track != track:
        _risk(risks, record, "declared_evaluation_track_mismatch")
    if mode == "known_core" and any(marker in answer for marker in REFUSAL_MARKERS):
        _risk(risks, record, "known_core_false_refusal")
    if mode == "needs_evidence" and not any(marker in answer for marker in NEEDS_EVIDENCE_MARKERS):
        _risk(risks, record, "needs_evidence_answer_does_not_request_evidence")
    acceptance_case_id = evaluation.get("acceptance_case_id", "")
    if acceptance_case_id and not _safe_identifier(acceptance_case_id):
        _risk(risks, record, "invalid_acceptance_case_id")
        acceptance_case_id = ""
    # Acceptance cases are fixed public questions, not three-mode calibration
    # groups.  Treating their IDs as triplets creates four false incompletes.
    triplet_id = evaluation.get("calibration_triplet_id", "")
    if triplet_id and not _safe_identifier(triplet_id):
        _risk(risks, record, "invalid_calibration_triplet_id")
        triplet_id = ""
    return str(mode or ""), str(track), str(triplet_id)


def _validate_evidence_contract(
    record: Mapping[str, Any],
    corpus_path: Path,
    corpus_lines: Sequence[str],
    risks: list[dict[str, str]],
) -> tuple[list[str], list[str], str]:
    dimension = str(record.get("primary_dimension", ""))
    evidence_value = record.get("evidence")
    evaluation = record.get("evaluation")
    evaluation_sufficient = (
        evaluation.get("evidence_sufficient")
        if isinstance(evaluation, Mapping)
        else None
    )
    evaluation_needs = (
        evaluation.get("needs_evidence") if isinstance(evaluation, Mapping) else None
    )

    # The authoritative v7 builder uses an evidence envelope.  During the
    # coordinated rollout one builder revision emitted the same information as
    # ``evidence`` + ``answer_support``; accept that representation only when
    # all equivalent fields are present and mutually consistent.
    support: Mapping[str, Any] | None = None
    if isinstance(evidence_value, Mapping):
        items_value = evidence_value.get("chunks")
        envelope_status = evidence_value.get("status")
        envelope_sufficient = evidence_value.get("sufficient_for_answer")
        bundle_sha256 = evidence_value.get("bundle_sha256")
    elif isinstance(evidence_value, list):
        items_value = evidence_value
        candidate = record.get("answer_support")
        if isinstance(candidate, Mapping):
            support = candidate
            envelope_status = candidate.get("evidence_status")
            support_status = candidate.get("status")
            envelope_sufficient = support_status == "supported"
            bundle_sha256 = candidate.get("bundle_sha256")
        else:
            _risk(risks, record, "missing_answer_support_contract")
            envelope_status = None
            envelope_sufficient = None
            bundle_sha256 = None
    else:
        _risk(risks, record, "invalid_evidence_envelope")
        items_value = []
        envelope_status = None
        envelope_sufficient = None
        bundle_sha256 = None

    if not isinstance(items_value, list):
        _risk(risks, record, "evidence_chunks_must_be_a_list")
        items_value = []
    items = [item for item in items_value if isinstance(item, Mapping)]
    if len(items) != len(items_value):
        _risk(risks, record, "invalid_evidence_item")
    if dimension == PARAMETRIC_CORE and not 1 <= len(items) <= 2:
        _risk(risks, record, "core_fact_requires_one_or_two_evidence_items")
    if dimension == GROUNDED_SINGLE and len(items) != 1:
        _risk(risks, record, "single_evidence_dimension_requires_one_item")
    if dimension == RAG_MULTI and not 2 <= len(items) <= 4:
        _risk(risks, record, "rag_requires_two_to_four_evidence_items")
    if dimension == VERTICAL_CHAT and len(items) not in {0, 1}:
        _risk(risks, record, "interaction_requires_zero_or_one_evidence_item")
    if dimension == NOVEL_EXPRESSION and len(items) not in {1, 2}:
        _risk(risks, record, "expression_requires_one_or_two_evidence_items")
    if dimension == CAPABILITY_BOUNDARY and len(items) > 0:
        _risk(risks, record, "dimension_should_not_carry_evidence")

    if not isinstance(envelope_sufficient, bool):
        _risk(risks, record, "invalid_evidence_sufficiency")
    elif isinstance(evaluation_sufficient, bool) and envelope_sufficient != evaluation_sufficient:
        _risk(risks, record, "evidence_sufficiency_mismatch")
    if envelope_status not in EVIDENCE_STATUSES:
        _risk(risks, record, "invalid_evidence_status")

    evidence_hashes: list[str] = []
    chunk_hashes: list[str] = []
    chapter_keys: list[str] = []
    evidence_texts: list[str] = []
    for item in items:
        item_status = item.get("status")
        if item_status is not None and item_status != "verified_train_corpus":
            _risk(risks, record, "invalid_chunk_evidence_status")
        text = item.get("text")
        sha256 = item.get("text_sha256", item.get("sha256"))
        if not isinstance(text, str) or not text:
            _risk(risks, record, "empty_evidence_text")
            continue
        if sha256 != text_sha256(text):
            _risk(risks, record, "evidence_sha256_mismatch")
        else:
            evidence_hashes.append(str(sha256))
        evidence_texts.append(text)
        source_path = item.get("source_path")
        try:
            source_matches = Path(str(source_path)).resolve() == corpus_path.resolve()
        except (OSError, RuntimeError):
            source_matches = False
        if not source_matches:
            _risk(risks, record, "evidence_source_not_formal_train")
        if item.get("source_split") not in {None, "formal_pretrain_train"}:
            _risk(risks, record, "evidence_source_split_mismatch")
        start = item.get("line_start", item.get("start_line"))
        end = item.get("line_end", item.get("end_line"))
        if not isinstance(start, int) or not isinstance(end, int) or not (
            1 <= start <= end <= len(corpus_lines)
        ):
            _risk(risks, record, "evidence_line_range_invalid")
        else:
            expected = "\n".join(corpus_lines[start - 1 : end])
            if text != expected:
                _risk(risks, record, "evidence_line_text_mismatch")
        heading_line = item.get("chapter_heading_line", item.get("heading_line"))
        chapter = item.get("chapter_number", item.get("chapter_id"))
        if heading_line is None or chapter is None or chapter == "":
            _risk(risks, record, "missing_evidence_chapter_id")
        else:
            # Chapter numbers repeat across the audited source-version groups.
            # The one-based heading line plus chapter hash identifies the
            # actual source chapter that was physically partitioned.
            chapter_hash = item.get("chapter_sha256")
            chapter_keys.append(f"{source_path}:{heading_line}:{chapter_hash}")
        chunk_sha = item.get("chunk_sha256")
        if isinstance(heading_line, int) and isinstance(start, int) and isinstance(sha256, str):
            expected_chunk_sha = stable_hash("sft-v7-chunk", heading_line, start, sha256)
            if chunk_sha != expected_chunk_sha:
                _risk(risks, record, "evidence_chunk_sha256_mismatch")
            else:
                chunk_hashes.append(chunk_sha)
        elif not isinstance(chunk_sha, str) or SHA256_HEX.fullmatch(chunk_sha) is None:
            _risk(risks, record, "invalid_evidence_chunk_sha256")

    expected_bundle = text_sha256("|".join(chunk_hashes))
    if bundle_sha256 != expected_bundle:
        _risk(risks, record, "evidence_bundle_sha256_mismatch")

    support_status = (
        "supported"
        if envelope_sufficient is True
        else "insufficient_evidence"
        if evaluation_needs is True
        else "not_applicable"
    )
    if support is not None:
        if support.get("status") not in SUPPORT_STATUSES:
            _risk(risks, record, "invalid_answer_support_status")
        elif support.get("status") != support_status:
            _risk(risks, record, "answer_support_status_mismatch")
        declared_hashes = support.get("evidence_sha256s")
        spans = support.get("supporting_spans")
        if not isinstance(declared_hashes, list) or any(
            not isinstance(value, str) for value in declared_hashes
        ):
            _risk(risks, record, "invalid_support_evidence_hashes")
            declared_hashes = []
        if not isinstance(spans, list) or any(not isinstance(span, str) for span in spans):
            _risk(risks, record, "invalid_supporting_spans")
            spans = []
        if not set(declared_hashes).issubset(set(evidence_hashes)):
            _risk(risks, record, "support_references_unknown_evidence")
        if support_status == "supported":
            if not declared_hashes or not spans:
                _risk(risks, record, "supported_answer_missing_proof")
            if any(not span or not any(span in text for text in evidence_texts) for span in spans):
                _risk(risks, record, "supporting_span_not_in_evidence")

    if support_status == "supported" and not evidence_texts:
        _risk(risks, record, "supported_answer_missing_evidence")
    required_terms = (
        evaluation.get("required_terms", []) if isinstance(evaluation, Mapping) else []
    )
    if support_status == "supported" and required_terms and not any(
        equivalent in text
        for term in required_terms
        for equivalent in EVIDENCE_TERM_EQUIVALENTS.get(term, (term,))
        for text in evidence_texts
    ):
        _risk(risks, record, "required_terms_not_grounded_in_evidence")
    metric = evaluation.get("metric") if isinstance(evaluation, Mapping) else None
    if metric in {"exact", "exact_match"} and evidence_texts and not any(
        str(record.get("answer", "")).strip() == text.strip()
        for text in evidence_texts
    ):
        _risk(risks, record, "exact_match_answer_not_in_evidence")
    return evidence_hashes, chapter_keys, support_status


def _cross_split_count(
    references: Mapping[str, Sequence[Mapping[str, str]]],
    *,
    allow_tracked_core: bool,
) -> int:
    leaks = 0
    for entries in references.values():
        splits = {entry["split"] for entry in entries}
        if len(splits) <= 1:
            continue
        allowed = allow_tracked_core and all(
            entry["dimension"] == PARAMETRIC_CORE
            and entry["track"] == "seen_fact_unseen_wording"
            for entry in entries
        )
        if not allowed:
            leaks += 1
    return leaks


def validate_records_by_split(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    corpus_path: Path,
    tokenizer: BPETokenizer | None,
    enforce_release_gates: bool = True,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Validate authorized records and return aggregate-only diagnostics."""

    corpus_lines = corpus_path.read_text(encoding="utf-8").splitlines()
    risks: list[dict[str, str]] = []
    split_counts: Counter[str] = Counter()
    dimension_counts: Counter[str] = Counter()
    dimension_split_counts: MutableMapping[str, Counter[str]] = defaultdict(Counter)
    family_counts: Counter[str] = Counter()
    prompt_template_splits: MutableMapping[str, set[str]] = defaultdict(set)
    prompt_template_hashes: MutableMapping[str, set[str]] = defaultdict(set)
    prompt_hash_splits: MutableMapping[str, set[str]] = defaultdict(set)
    answer_style_splits: MutableMapping[str, set[str]] = defaultdict(set)
    answer_style_counts: Counter[str] = Counter()
    opening_counts: Counter[str] = Counter()
    ids: list[str] = []
    exact_questions: list[str] = []
    canonical_questions: list[str] = []
    general_answers: Counter[str] = Counter()
    semantic_refs: MutableMapping[str, list[dict[str, str]]] = defaultdict(list)
    evidence_refs: MutableMapping[str, list[dict[str, str]]] = defaultdict(list)
    chapter_refs: MutableMapping[str, list[dict[str, str]]] = defaultdict(list)
    triplet_refs: MutableMapping[str, list[dict[str, str]]] = defaultdict(list)
    core_coverage: set[str] = set()
    answer_lengths: list[int] = []
    sequence_lengths: list[int] = []
    supervised_by_split: Counter[str] = Counter()
    sequence_by_split: Counter[str] = Counter()
    multiturn_by_split: Counter[str] = Counter()
    rag_bundles_by_split: Counter[str] = Counter()
    negative_grounding_by_split: Counter[str] = Counter()
    grounding_records_by_split: Counter[str] = Counter()

    math_positive = 0
    encyclopedia_positive = 0
    project_concept_positive = 0
    meta_records = 0
    unencodable = 0
    over_context = 0
    review_pending = 0
    punctuation_errors = 0

    for file_split, records in records_by_split.items():
        for record in records:
            split = str(record.get("split", ""))
            if split != file_split:
                _risk(risks, record, "record_split_file_mismatch")
            split_counts[split] += 1
            missing = REQUIRED_FIELDS.difference(record)
            if missing:
                _risk(risks, record, "missing_required_fields")
                continue
            dimension = str(record["primary_dimension"])
            family = str(record["task_family"])
            dimension_counts[dimension] += 1
            dimension_split_counts[dimension][split] += 1
            family_counts[family] += 1
            if record["schema_version"] != SCHEMA_VERSION:
                _risk(risks, record, "schema_version_mismatch")
            if dimension not in DIMENSION_TOTAL_TARGETS:
                _risk(risks, record, "unknown_primary_dimension")
            for field_name in ("id", "semantic_group", "fact_id"):
                if not _safe_identifier(record.get(field_name)):
                    _risk(risks, record, f"invalid_{field_name}")
            generation = record.get("generation")
            if not isinstance(generation, Mapping):
                _risk(risks, record, "missing_generation_contract")
                prompt_template_id = ""
                answer_style_id = ""
                prompt_template_sha = ""
            else:
                prompt_template_id = generation.get("prompt_template_id")
                answer_style_id = generation.get("answer_style_id")
                prompt_template_sha = generation.get("prompt_template_sha256")
                if not _safe_identifier(prompt_template_id):
                    _risk(risks, record, "invalid_prompt_template_id")
                    prompt_template_id = ""
                if not _safe_identifier(answer_style_id):
                    _risk(risks, record, "invalid_answer_style_id")
                    answer_style_id = ""
                if not isinstance(prompt_template_sha, str) or SHA256_HEX.fullmatch(
                    prompt_template_sha
                ) is None:
                    _risk(risks, record, "invalid_prompt_template_sha256")
                    prompt_template_sha = ""
            if record.get("prompt_template_id") not in {None, prompt_template_id}:
                _risk(risks, record, "top_level_prompt_template_mismatch")
            if record.get("answer_style_id") not in {None, answer_style_id}:
                _risk(risks, record, "top_level_answer_style_mismatch")
            ids.append(str(record["id"]))
            question = str(record["question"])
            answer = str(record["answer"])
            exact_questions.append(question)
            canonical_questions.append(canonical_question(question))
            if prompt_template_id:
                prompt_template_splits[str(prompt_template_id)].add(split)
                if prompt_template_sha:
                    prompt_template_hashes[str(prompt_template_id)].add(
                        str(prompt_template_sha)
                    )
                    prompt_hash_splits[str(prompt_template_sha)].add(split)
            if answer_style_id:
                answer_style_counts[str(answer_style_id)] += 1
                answer_style_splits[str(answer_style_id)].add(split)
            opening_counts[opening_fingerprint(answer)] += 1
            # Exact-answer reuse is a global anti-template signal.  The only
            # exception is the small, manually reviewed known-core bank where
            # several phrasings intentionally share one canonical fact answer.
            if family != "known_core_direct":
                general_answers[answer] += 1

            roles_ok = _roles_and_eos_contract_ok(record, risks)
            if roles_ok and len(record["messages"]) >= 4:
                multiturn_by_split[split] += 1
            mode, track, triplet_id = _validate_evaluation_contract(record, risks)
            if triplet_id:
                triplet_refs[triplet_id].append(
                    {"split": split, "mode": mode, "dimension": dimension, "track": track}
                )

            evidence_hashes, chapter_keys, support_status = _validate_evidence_contract(
                record, corpus_path, corpus_lines, risks
            )
            if mode in {"known_core", "grounded_answer"} and support_status != "supported":
                _risk(risks, record, "answer_support_status_conflicts_with_capability_mode")
            if mode == "needs_evidence" and support_status != "insufficient_evidence":
                _risk(risks, record, "answer_support_status_conflicts_with_capability_mode")
            expected_interaction_support = (
                "supported"
                if isinstance(record.get("evaluation"), Mapping)
                and record["evaluation"].get("evidence_sufficient") is True
                else "not_applicable"
            )
            if mode == "interaction" and support_status != expected_interaction_support:
                _risk(risks, record, "answer_support_status_conflicts_with_capability_mode")
            if mode == "expression" and support_status not in {"supported", "not_applicable"}:
                _risk(risks, record, "answer_support_status_conflicts_with_capability_mode")
            ref = {"split": split, "dimension": dimension, "track": track}
            semantic_refs[str(record["semantic_group"])].append(ref)
            for value in evidence_hashes:
                evidence_refs[value].append(ref)
            for value in chapter_keys:
                chapter_refs[value].append(ref)
            if dimension == RAG_MULTI and 2 <= len(evidence_items(record)) <= 4:
                rag_bundles_by_split[split] += 1
            if dimension in {GROUNDED_SINGLE, RAG_MULTI}:
                grounding_records_by_split[split] += 1
                if mode == "needs_evidence":
                    negative_grounding_by_split[split] += 1

            combined = _combined_text(record)
            is_boundary = dimension == CAPABILITY_BOUNDARY and mode == "needs_evidence"
            if _is_positive_math_prompt(question) and not is_boundary:
                math_positive += 1
                _risk(risks, record, "positive_math_training")
            if any(marker in combined for marker in GENERAL_ENCYCLOPEDIA_MARKERS) and not is_boundary:
                encyclopedia_positive += 1
                _risk(risks, record, "positive_general_encyclopedia_training")
            if any(marker in combined for marker in PROJECT_CONCEPT_MARKERS) and not is_boundary:
                project_concept_positive += 1
                _risk(risks, record, "positive_project_concept_training")
            if any(marker in combined for marker in META_MARKERS):
                meta_records += 1
                _risk(risks, record, "forbidden_meta_wrapper")
            for opening, closing in PUNCTUATION_PAIRS:
                if answer.count(opening) != answer.count(closing):
                    punctuation_errors += 1
                    _risk(risks, record, "unbalanced_answer_punctuation")
                    break
            review = record.get("review")
            if not isinstance(review, Mapping) or review.get("status") not in APPROVED_REVIEW_STATUSES:
                review_pending += 1
                _risk(risks, record, "review_not_approved", severity="P1")

            coverage = record.get("coverage")
            if not isinstance(coverage, Mapping):
                _risk(risks, record, "invalid_coverage_contract")
            elif dimension == PARAMETRIC_CORE:
                for value in coverage.get("entities", []):
                    if isinstance(value, str) and value:
                        core_coverage.add(value)
                for value in coverage.get("concepts", []):
                    if isinstance(value, str) and value:
                        core_coverage.add(value)

            declared_encoding = record.get("encoding_audit")
            encoding_fields = {
                "sequence_tokens",
                "supervised_tokens",
                "last_answer_tokens",
                "assistant_turns",
                "eos_targets",
                "masked_user_and_role_tokens",
            }
            if not isinstance(declared_encoding, Mapping) or any(
                not isinstance(declared_encoding.get(field), int)
                or isinstance(declared_encoding.get(field), bool)
                or declared_encoding.get(field, -1) < 0
                for field in encoding_fields
            ):
                _risk(risks, record, "invalid_encoding_audit")
            elif roles_ok:
                assistant_turns = len(record["messages"]) // 2
                if (
                    declared_encoding["assistant_turns"] != assistant_turns
                    or declared_encoding["eos_targets"] != assistant_turns
                ):
                    _risk(risks, record, "encoding_eos_target_mismatch")
                if declared_encoding["sequence_tokens"] != (
                    declared_encoding["supervised_tokens"]
                    + declared_encoding["masked_user_and_role_tokens"]
                ):
                    _risk(risks, record, "encoding_mask_accounting_mismatch")

            if tokenizer is not None and roles_ok:
                try:
                    computed_encoding = encoding_metrics(record, tokenizer)
                except Exception:
                    unencodable += 1
                    _risk(risks, record, "unencodable_record")
                else:
                    if not isinstance(declared_encoding, Mapping) or any(
                        declared_encoding.get(field) != computed_encoding[field]
                        for field in encoding_fields
                    ):
                        _risk(risks, record, "encoding_audit_mismatch")
                    sequence_length = computed_encoding["sequence_tokens"]
                    supervised_tokens = computed_encoding["supervised_tokens"]
                    answer_tokens = computed_encoding["last_answer_tokens"]
                    sequence_lengths.append(sequence_length)
                    answer_lengths.append(answer_tokens)
                    supervised_by_split[split] += supervised_tokens
                    sequence_by_split[split] += sequence_length
                    if sequence_length > CONTEXT_TOKENS:
                        over_context += 1
                        _risk(risks, record, "sequence_over_512")

    duplicate_ids = sum(count - 1 for count in Counter(ids).values() if count > 1)
    duplicate_exact = sum(
        count - 1 for count in Counter(exact_questions).values() if count > 1
    )
    duplicate_canonical = sum(
        count - 1 for count in Counter(canonical_questions).values() if count > 1
    )
    prompt_template_leaks = sum(
        1 for splits in prompt_template_splits.values() if len(splits) > 1
    )
    prompt_template_hash_conflicts = sum(
        1 for hashes in prompt_template_hashes.values() if len(hashes) > 1
    )
    prompt_hash_split_leaks = sum(
        1 for splits in prompt_hash_splits.values() if len(splits) > 1
    )
    answer_style_split_leaks = sum(
        1 for splits in answer_style_splits.values() if len(splits) > 1
    )
    semantic_leaks = _cross_split_count(semantic_refs, allow_tracked_core=True)
    evidence_leaks = _cross_split_count(evidence_refs, allow_tracked_core=True)
    chapter_leaks = _cross_split_count(chapter_refs, allow_tracked_core=True)

    complete_triplets_by_split: Counter[str] = Counter()
    incomplete_triplets = 0
    triplet_split_leaks = 0
    for entries in triplet_refs.values():
        splits = {entry["split"] for entry in entries}
        modes = {entry["mode"] for entry in entries}
        if len(splits) != 1:
            triplet_split_leaks += 1
        elif modes == CALIBRATION_MODES and len(entries) == 3:
            complete_triplets_by_split[next(iter(splits))] += 1
        else:
            incomplete_triplets += 1

    largest_opening_count = max(opening_counts.values(), default=0)
    record_count = sum(split_counts.values())
    largest_opening_share = largest_opening_count / record_count if record_count else 0.0
    maximum_general_answer_repeat = max(general_answers.values(), default=0)
    long_answers = sum(97 <= length <= 160 for length in answer_lengths)
    medium_answers = sum(33 <= length <= 96 for length in answer_lengths)
    long_answer_share = long_answers / len(answer_lengths) if answer_lengths else 0.0
    medium_answer_share = medium_answers / len(answer_lengths) if answer_lengths else 0.0

    p0_failures: list[str] = []
    p1_failures: list[str] = []
    if enforce_release_gates:
        for split in records_by_split:
            if split_counts[split] != SPLIT_TARGETS[split]:
                p0_failures.append(f"quota.split.{split}")
            for dimension in DIMENSION_TOTAL_TARGETS:
                if dimension_split_counts[dimension][split] != DIMENSION_SPLIT_TARGETS[dimension][split]:
                    p0_failures.append(f"quota.dimension.{dimension}.{split}")
        if tokenizer is None:
            p0_failures.append("tokenizer_required_for_release_validation")

    aggregate_p0_checks = (
        (duplicate_ids == 0, "duplicate_ids"),
        (duplicate_exact == 0, "duplicate_exact_questions"),
        (duplicate_canonical == 0, "duplicate_canonical_questions"),
        (prompt_template_leaks == 0, "prompt_template_split_leakage"),
        (prompt_template_hash_conflicts == 0, "prompt_template_hash_conflicts"),
        (prompt_hash_split_leaks == 0, "prompt_text_split_leakage"),
        (answer_style_split_leaks == 0, "answer_style_split_leakage"),
        (semantic_leaks == 0, "semantic_group_split_leakage"),
        (evidence_leaks == 0, "evidence_sha_split_leakage"),
        (chapter_leaks == 0, "chapter_split_leakage"),
        (triplet_split_leaks == 0, "calibration_triplet_split_leakage"),
        (incomplete_triplets == 0, "incomplete_calibration_triplets"),
        (math_positive == 0, "positive_math_training"),
        (encyclopedia_positive == 0, "positive_general_encyclopedia_training"),
        (project_concept_positive == 0, "positive_project_concept_training"),
        (meta_records == 0, "forbidden_meta_wrappers"),
        (punctuation_errors == 0, "punctuation_errors"),
        (unencodable == 0, "unencodable_records"),
        (over_context == 0, "sequences_over_512"),
    )
    for passed, name in aggregate_p0_checks:
        if not passed:
            p0_failures.append(name)
    if any(risk["severity"] == "P0" for risk in risks):
        p0_failures.append("record_level_p0_risks")

    if enforce_release_gates:
        if review_pending:
            p1_failures.append("pending_review_records")
        for split in records_by_split:
            if multiturn_by_split[split] < MIN_MULTITURN_BY_SPLIT[split]:
                p1_failures.append(f"multiturn_coverage.{split}")
            if rag_bundles_by_split[split] < MIN_RAG_BUNDLES_BY_SPLIT[split]:
                p1_failures.append(f"rag_bundle_coverage.{split}")
            if complete_triplets_by_split[split] < MIN_CALIBRATION_TRIPLETS_BY_SPLIT[split]:
                p1_failures.append(f"calibration_triplets.{split}")
            grounding = grounding_records_by_split[split]
            negative_share = negative_grounding_by_split[split] / grounding if grounding else 0.0
            if not 0.15 <= negative_share <= 0.20:
                p1_failures.append(f"negative_grounding_share.{split}")
        if not 40 <= len(core_coverage) <= 60:
            p1_failures.append("core_entity_concept_coverage")
        if tokenizer is not None:
            if long_answer_share < 0.10:
                p1_failures.append("long_answer_share")
            if medium_answer_share <= 0.50:
                p1_failures.append("medium_answer_majority")
        if largest_opening_share > 0.02:
            p1_failures.append("answer_opening_template_share")
        if maximum_general_answer_repeat > 5:
            p1_failures.append("general_exact_answer_repeat")

    risk_counts = Counter(risk["code"] for risk in risks)
    severity_counts = Counter(risk["severity"] for risk in risks)
    status = "passed" if not p0_failures and not p1_failures else "needs_revision"
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "validated_splits": list(records_by_split),
        "sealed_body_accessed": SEALED_SPLIT in records_by_split,
        "record_body_emitted": False,
        "p0_failures": sorted(set(p0_failures)),
        "p1_failures": sorted(set(p1_failures)),
        "risk_records": len(risks),
        "risk_code_counts": dict(sorted(risk_counts.items())),
        "risk_severity_counts": dict(sorted(severity_counts.items())),
        "records": record_count,
        "split_counts": dict(sorted(split_counts.items())),
        "dimension_counts": dict(sorted(dimension_counts.items())),
        "dimension_split_counts": {
            dimension: dict(sorted(counter.items()))
            for dimension, counter in sorted(dimension_split_counts.items())
        },
        "task_family_counts": dict(sorted(family_counts.items())),
        "duplicate_ids": duplicate_ids,
        "duplicate_exact_questions": duplicate_exact,
        "duplicate_canonical_questions": duplicate_canonical,
        "prompt_template_split_leaks": prompt_template_leaks,
        "prompt_template_hash_conflicts": prompt_template_hash_conflicts,
        "prompt_text_hash_split_leaks": prompt_hash_split_leaks,
        "answer_style_split_leaks": answer_style_split_leaks,
        "semantic_group_split_leaks": semantic_leaks,
        "evidence_sha_split_leaks": evidence_leaks,
        "chapter_split_leaks": chapter_leaks,
        "calibration": {
            "complete_triplets_by_split": dict(sorted(complete_triplets_by_split.items())),
            "incomplete_triplets": incomplete_triplets,
            "split_leaks": triplet_split_leaks,
        },
        "domain_alignment": {
            "positive_math_records": math_positive,
            "positive_general_encyclopedia_records": encyclopedia_positive,
            "positive_project_concept_records": project_concept_positive,
            "meta_wrapper_records": meta_records,
        },
        "quality": {
            "review_pending_records": review_pending,
            "punctuation_error_records": punctuation_errors,
            "core_entity_concept_coverage": len(core_coverage),
            "multiturn_records_by_split": dict(sorted(multiturn_by_split.items())),
            "rag_bundle_records_by_split": dict(sorted(rag_bundles_by_split.items())),
            "negative_grounding_records_by_split": dict(
                sorted(negative_grounding_by_split.items())
            ),
            "grounding_records_by_split": dict(sorted(grounding_records_by_split.items())),
            "distinct_prompt_templates": len(prompt_template_splits),
            "distinct_answer_styles": len(answer_style_counts),
            "distinct_answer_openings": len(opening_counts),
            "largest_answer_opening_count": largest_opening_count,
            "largest_answer_opening_share": largest_opening_share,
            "maximum_general_exact_answer_repeat": maximum_general_answer_repeat,
        },
        "tokenization": {
            "performed": tokenizer is not None,
            "unencodable_records": unencodable,
            "sequences_over_512": over_context,
            "min_sequence_tokens": min(sequence_lengths, default=0),
            "max_sequence_tokens": max(sequence_lengths, default=0),
            "sequence_tokens_by_split": dict(sorted(sequence_by_split.items())),
            "supervised_tokens_by_split": dict(sorted(supervised_by_split.items())),
            "answer_33_to_96_tokens": medium_answers,
            "answer_97_to_160_tokens": long_answers,
            "medium_answer_share": medium_answer_share,
            "long_answer_share": long_answer_share,
        },
    }
    return report, risks


def validate_dataset_files(
    *,
    train_path: Path,
    val_path: Path,
    public_path: Path,
    corpus_path: Path,
    tokenizer: BPETokenizer | None,
    sealed_path: Path | None = None,
    allow_sealed_build_validation: bool = False,
    enforce_release_gates: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    """Validate authorized files; reject sealed paths before filesystem access."""

    if sealed_path is not None and not allow_sealed_build_validation:
        raise PermissionError(
            "sealed_test is inaccessible by default; use the explicit build-validation flag"
        )
    if allow_sealed_build_validation and sealed_path is None:
        raise ValueError("sealed build validation requires an explicit sealed path")

    paths: dict[str, Path] = {
        TRAIN_SPLIT: train_path,
        VAL_SPLIT: val_path,
        PUBLIC_SPLIT: public_path,
    }
    if allow_sealed_build_validation:
        # This branch is the only place the sealed path enters the read set.
        paths[SEALED_SPLIT] = sealed_path  # type: ignore[assignment]

    records_by_split: dict[str, list[dict[str, Any]]] = {}
    files: dict[str, dict[str, Any]] = {}
    for split, path in paths.items():
        records = read_jsonl_split(path, split)
        records_by_split[split] = records
        files[split] = {
            "path": str(path),
            "records": len(records),
            "sha256": file_sha256(path),
            "schema_version": SCHEMA_VERSION,
        }

    report, risks = validate_records_by_split(
        records_by_split,
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        enforce_release_gates=enforce_release_gates,
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": report["status"],
        "validation_mode": (
            "build_including_sealed"
            if allow_sealed_build_validation
            else "default_nonsealed_only"
        ),
        "sealed_build_validation_explicitly_allowed": allow_sealed_build_validation,
        "record_body_emitted": False,
        "files": files,
        "aggregate": {
            "records": report["records"],
            "split_counts": report["split_counts"],
            "dimension_counts": report["dimension_counts"],
            "p0_failure_count": len(report["p0_failures"]),
            "p1_failure_count": len(report["p1_failures"]),
            "risk_code_counts": report["risk_code_counts"],
        },
        "corpus": {"path": str(corpus_path), "sha256": file_sha256(corpus_path)},
    }
    return report, manifest, risks


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--sealed", type=Path)
    parser.add_argument("--allow-sealed-build-validation", action="store_true")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--data-log-level", default="INFO")
    parser.add_argument("--validation-log-level", default="INFO")
    parser.add_argument("--sft-log-level", default="INFO")
    parser.add_argument("--preflight-log-level", default="INFO")
    parser.add_argument("--orchestrator-log-level", default="INFO")
    parser.add_argument("--no-console-log", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sealed is not None and not args.allow_sealed_build_validation:
        # Fail before tokenizer/corpus loading or logger setup to make the
        # default sealed boundary easy to reason about and test.
        raise PermissionError(
            "--sealed requires --allow-sealed-build-validation during the one build audit"
        )
    run_id = generate_run_id("sft-v7-vertical-validation")
    levels = resolve_module_log_levels(
        {
            "data": args.data_log_level,
            "validation": args.validation_log_level,
            "sft": args.sft_log_level,
            "preflight": args.preflight_log_level,
            "orchestrator": args.orchestrator_log_level,
            "pretrain": "OFF",
            "checkpoint": "OFF",
            "gpu": "OFF",
        }
    )
    loggers = configure_module_loggers(
        args.log_dir,
        run_id,
        levels,
        max_bytes=10 * 1024 * 1024,
        backup_count=5,
        console=not args.no_console_log,
    )
    try:
        tokenizer = BPETokenizer.load(args.tokenizer)
        report, manifest, risks = validate_dataset_files(
            train_path=args.train,
            val_path=args.val,
            public_path=args.public,
            sealed_path=args.sealed,
            allow_sealed_build_validation=args.allow_sealed_build_validation,
            corpus_path=args.corpus,
            tokenizer=tokenizer,
            enforce_release_gates=True,
        )
        report.update(
            {
                "run_id": run_id,
                "manifest_path": str(args.manifest),
                "tokenizer_path": str(args.tokenizer),
                "tokenizer_sha256": file_sha256(args.tokenizer),
            }
        )
        manifest.update(
            {
                "run_id": run_id,
                "tokenizer": {
                    "path": str(args.tokenizer),
                    "sha256": file_sha256(args.tokenizer),
                },
            }
        )
        atomic_write_json(args.report, report)
        atomic_write_json(args.manifest, manifest)
        loggers["data"].info(
            "authorized physical SFT splits loaded",
            extra={
                "context": {
                    "validated_splits": report["validated_splits"],
                    "split_counts": report["split_counts"],
                    "sealed_build_mode": args.allow_sealed_build_validation,
                }
            },
        )
        loggers["sft"].info(
            "evidence and interaction contracts checked",
            extra={
                "context": {
                    "risk_code_counts": report["risk_code_counts"],
                    "record_body_emitted": False,
                }
            },
        )
        loggers["validation"].info(
            "SFT v7 validation complete",
            extra={
                "context": {
                    "status": report["status"],
                    "p0_failures": report["p0_failures"],
                    "p1_failures": report["p1_failures"],
                    "risk_records": len(risks),
                }
            },
        )
        loggers["preflight"].info(
            "sealed access policy enforced",
            extra={
                "context": {
                    "sealed_accessed": report["sealed_body_accessed"],
                    "explicit_build_permission": args.allow_sealed_build_validation,
                    "record_body_emitted": False,
                }
            },
        )
        loggers["orchestrator"].info(
            "wrote aggregate validation artifacts",
            extra={
                "context": {
                    "report_path": str(args.report),
                    "manifest_path": str(args.manifest),
                    "status": report["status"],
                }
            },
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["status"] == "passed" else 2
    except Exception:
        loggers["validation"].exception("SFT v7 vertical validation failed")
        raise
    finally:
        close_module_loggers(loggers)


if __name__ == "__main__":
    raise SystemExit(main())
