"""Read-only semantic-density audit for the non-public SFT v7 train/val data.

Only ``data/sft/v7/train.jsonl`` and ``data/sft/v7/val.jsonl`` are valid data
sources.  The report deliberately contains aggregate counts, portable paths,
hashes, and risk codes only; it never emits a question, answer, evidence span,
or record identifier.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence
import unicodedata

from sft_v7_vertical_catalog import (
    CORE,
    CORE_TERMS,
    DIMENSION_TOTALS,
    KNOWN_CORE_FACTS,
)
from training_runtime import (
    atomic_write_json,
    atomic_write_text,
    close_module_loggers,
    configure_module_loggers,
    generate_run_id,
    resolve_module_log_levels,
    utc_now,
)


REPORT_SCHEMA_VERSION = "sft-v7-semantic-density-audit/v1"
ALGORITHM_VERSION = "semantic-density-train-gates/2.0"
DEFAULT_REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN = Path("data/sft/v7/train.jsonl")
DEFAULT_VAL = Path("data/sft/v7/val.jsonl")
DEFAULT_REPORT = Path(
    "reports/milestones/021_sft_v7_1_canary/semantic_density_audit.json"
)
DEFAULT_MARKDOWN = Path(
    "reports/milestones/021_sft_v7_1_canary/semantic_density_audit.md"
)
DEFAULT_LOG_DIR = Path("logs/sft_v7_semantic_density")

# These are the six openings produced by _answer_from_chunks in the frozen v7
# builder.  The quotation mark/colon is part of the prefix so a coincidental
# occurrence later in an answer is not counted.
TEMPLATE_OPENINGS: Mapping[str, str] = {
    "original_text_says": "原文写道“",
    "material_states": "材料表述为“",
    "passage_verifiable": "片段可核对：",
    "this_place_says": "这处写的是“",
    "text_contains": "文本中出现“",
    "reviewable_original_sentence": "可复查的原句是“",
}

COPY_REVIEW_THRESHOLD = 0.50
COPY_SEVERE_THRESHOLD = 0.80
TEMPLATE_SHARE_LIMIT = 0.20
BARE_CORE_COVERAGE_MINIMUM = 0.50
CORE_DENSITY_INDEX_MINIMUM = 0.75

RELATION_WORDS = (
    "老师",
    "父亲",
    "母亲",
    "儿子",
    "女儿",
    "弟子",
    "族长",
    "宗主",
    "主人",
    "身份",
    "名字",
    "别名",
    "关系",
)


class SemanticAuditError(ValueError):
    """Raised for a safe, coded audit failure."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class LoadedJsonl:
    records: tuple[dict[str, Any], ...]
    sha256: str
    schema_counts: Mapping[str, int]


def normalize_overlap_text(text: str) -> str:
    """Normalize text for overlap: NFKC, lowercase, keep alphanumeric only.

    Chinese characters are retained by :py:meth:`str.isalnum`; whitespace and
    punctuation are removed.  This makes punctuation changes irrelevant while
    preserving exact character order.  No synonym or fuzzy matching is used.
    """

    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(character for character in normalized if character.isalnum())


def longest_contiguous_common_substring_length(left: str, right: str) -> int:
    """Return the exact longest contiguous common-substring length.

    Dynamic programming uses O(min(len(left), len(right))) memory.  Inputs are
    expected to be normalized by :func:`normalize_overlap_text` first.
    """

    if not left or not right:
        return 0
    if len(right) > len(left):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    maximum = 0
    for left_character in left:
        current = [0]
        for index, right_character in enumerate(right, 1):
            value = previous[index - 1] + 1 if left_character == right_character else 0
            current.append(value)
            if value > maximum:
                maximum = value
        previous = current
    return maximum


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_under_repo(
    value: Path,
    repo_root: Path,
    *,
    must_exist: bool,
) -> Path:
    root = repo_root.resolve()
    candidate = value if value.is_absolute() else root / value
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise SemanticAuditError("path_outside_repo") from error
    if must_exist and not candidate.is_file():
        raise SemanticAuditError("source_missing")
    return candidate


def portable_repo_path(path: Path, repo_root: Path) -> str:
    """Return a repository-relative POSIX path or fail closed."""

    resolved = _resolve_under_repo(path, repo_root, must_exist=False)
    return resolved.relative_to(repo_root.resolve()).as_posix()


def implementation_metadata(repo_root: Path) -> dict[str, Any]:
    """Return content-free provenance for the exact audit implementation.

    The implementation SHA is calculated from this source file immediately
    before reports are written.  Report writes therefore cannot make the
    implementation hash drift.  Temporary test repositories may not contain
    this module, so they receive the same repository-relative logical path and
    an ``unavailable`` Git marker instead of an absolute host path.
    """

    source = Path(__file__).resolve()
    root = repo_root.resolve()
    try:
        source_path = source.relative_to(root).as_posix()
    except ValueError:
        source_path = Path(__file__).name

    git_commit = "unavailable"
    git_dirty: bool | None = None
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        git_commit = commit_result.stdout.strip()
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        git_dirty = bool(status_result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass

    return {
        "path": source_path,
        "sha256": _sha256_file(source),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "algorithm_version": ALGORITHM_VERSION,
    }


def load_jsonl(path: Path, *, expected_split: str) -> LoadedJsonl:
    """Load one authorized split, validating every line and split label."""

    if expected_split not in {"train", "val"}:
        raise SemanticAuditError("unauthorized_split")
    records: list[dict[str, Any]] = []
    schema_counts: Counter[str] = Counter()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SemanticAuditError("invalid_jsonl") from error
                if not isinstance(value, dict):
                    raise SemanticAuditError("record_not_object")
                if value.get("split") != expected_split:
                    raise SemanticAuditError("record_split_mismatch")
                identifier = value.get("id")
                if not isinstance(identifier, str) or not identifier:
                    raise SemanticAuditError("missing_record_id")
                records.append(value)
                schema_counts[str(value.get("schema_version", "missing"))] += 1
    except UnicodeDecodeError as error:
        raise SemanticAuditError("invalid_utf8") from error
    if not records:
        raise SemanticAuditError("empty_split")
    return LoadedJsonl(
        records=tuple(records),
        sha256=_sha256_file(path),
        schema_counts=dict(sorted(schema_counts.items())),
    )


def _record_supervised_tokens(record: Mapping[str, Any]) -> int:
    audit = record.get("encoding_audit")
    if not isinstance(audit, Mapping):
        raise SemanticAuditError("missing_encoding_audit")
    value = audit.get("supervised_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SemanticAuditError("invalid_supervised_tokens")
    return value


def _supporting_spans(record: Mapping[str, Any]) -> tuple[str, ...]:
    support = record.get("answer_support")
    if not isinstance(support, Mapping) or support.get("status") != "supported":
        return ()
    spans = support.get("supporting_spans")
    if not isinstance(spans, list):
        raise SemanticAuditError("invalid_supporting_spans")
    if any(not isinstance(span, str) for span in spans):
        raise SemanticAuditError("invalid_supporting_span")
    return tuple(span for span in spans if span.strip())


def evidence_copy_ratio(answer: str, spans: Sequence[str]) -> tuple[float, int, int]:
    """Calculate max normalized contiguous overlap / normalized answer length."""

    normalized_answer = normalize_overlap_text(answer)
    if not normalized_answer or not spans:
        return 0.0, 0, len(normalized_answer)
    maximum = max(
        longest_contiguous_common_substring_length(
            normalized_answer,
            normalize_overlap_text(span),
        )
        for span in spans
    )
    return maximum / len(normalized_answer), maximum, len(normalized_answer)


def _identifier_fingerprint(record: Mapping[str, Any]) -> str:
    identifier = str(record.get("id", ""))
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:16]


def _relation_screen(answer: str) -> tuple[bool, bool, bool]:
    """Return exact-self, two-node-cycle, repeated-clause heuristic flags."""

    labels = sorted(
        {term.label for term in CORE_TERMS},
        key=lambda value: (-len(value), value),
    )
    exact_self = False
    for label in labels:
        escaped = re.escape(label)
        if re.search(rf"{escaped}\s*(?:是|为|和|与|由|属于)\s*{escaped}", answer):
            exact_self = True
            break

    label_alternation = "|".join(re.escape(label) for label in labels)
    relation_alternation = "|".join(RELATION_WORDS)
    pattern = re.compile(
        rf"(?P<left>{label_alternation})\s*(?:是|为)\s*"
        rf"(?P<right>{label_alternation})\s*的\s*(?:{relation_alternation})"
    )
    pairs = {(match.group("left"), match.group("right")) for match in pattern.finditer(answer)}
    two_node_cycle = any(left != right and (right, left) in pairs for left, right in pairs)

    clauses = [
        normalize_overlap_text(clause)
        for clause in re.split(r"[。！？；\n]+", answer)
    ]
    meaningful = [clause for clause in clauses if len(clause) >= 4]
    repeated_clause = len(meaningful) != len(set(meaningful))
    return exact_self, two_node_cycle, repeated_clause


def _round_share(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def audit_semantic_density(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Produce a content-free aggregate semantic-density report."""

    if set(records_by_split) != {"train", "val"}:
        raise SemanticAuditError("source_split_set_mismatch")

    all_records: list[tuple[str, Mapping[str, Any]]] = []
    for split in ("train", "val"):
        for record in records_by_split[split]:
            if record.get("split") != split:
                raise SemanticAuditError("record_split_mismatch")
            all_records.append((split, record))
    if not all_records:
        raise SemanticAuditError("empty_population")

    record_counts: Counter[str] = Counter()
    supervised_counts: Counter[str] = Counter()
    record_counts_by_split: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
    }
    supervised_counts_by_split: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
    }
    split_record_counts: Counter[str] = Counter()
    split_supervised_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    opening_counts: dict[str, Counter[str]] = {
        opening_id: Counter() for opening_id in TEMPLATE_OPENINGS
    }
    any_template_by_split: Counter[str] = Counter()

    canonical_by_normalized: dict[str, str] = {}
    for fact in KNOWN_CORE_FACTS:
        canonical_by_normalized[normalize_overlap_text(fact.canonical_question)] = fact.fact_id
    bare_core_counts_by_split: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
    }

    copy_eligible: Counter[str] = Counter()
    copy_review: Counter[str] = Counter()
    copy_severe: Counter[str] = Counter()
    copy_ratio_sum: defaultdict[str, float] = defaultdict(float)
    copy_lcs_sum: Counter[str] = Counter()
    copy_answer_chars: Counter[str] = Counter()
    copy_eligible_by_split: Counter[str] = Counter()
    copy_review_by_split: Counter[str] = Counter()
    copy_severe_by_split: Counter[str] = Counter()

    exact_self: Counter[str] = Counter()
    two_node_cycle: Counter[str] = Counter()
    repeated_clause: Counter[str] = Counter()
    exact_self_by_split: Counter[str] = Counter()
    two_node_cycle_by_split: Counter[str] = Counter()
    repeated_clause_by_split: Counter[str] = Counter()
    heuristic_fingerprints: dict[str, list[str]] = {
        "exact_self_relation": [],
        "two_node_relation_cycle": [],
        "repeated_normalized_clause": [],
    }

    for split, record in all_records:
        dimension = str(record.get("primary_dimension", ""))
        if dimension not in DIMENSION_TOTALS:
            raise SemanticAuditError("unknown_primary_dimension")
        answer = record.get("answer")
        question = record.get("question")
        if not isinstance(answer, str) or not answer.strip():
            raise SemanticAuditError("invalid_answer")
        if not isinstance(question, str) or not question.strip():
            raise SemanticAuditError("invalid_question")
        supervised_tokens = _record_supervised_tokens(record)
        record_counts[dimension] += 1
        supervised_counts[dimension] += supervised_tokens
        record_counts_by_split[split][dimension] += 1
        supervised_counts_by_split[split][dimension] += supervised_tokens
        split_record_counts[split] += 1
        split_supervised_counts[split] += supervised_tokens
        family_counts[str(record.get("task_family", "missing"))] += 1

        matched_template = False
        for opening_id, prefix in TEMPLATE_OPENINGS.items():
            if answer.startswith(prefix):
                opening_counts[opening_id][split] += 1
                matched_template = True
        if matched_template:
            any_template_by_split[split] += 1

        core_fact_id = canonical_by_normalized.get(normalize_overlap_text(question))
        if core_fact_id is not None:
            bare_core_counts_by_split[split][core_fact_id] += 1

        spans = _supporting_spans(record)
        if spans:
            ratio, lcs_chars, answer_chars = evidence_copy_ratio(answer, spans)
            copy_eligible[dimension] += 1
            copy_eligible_by_split[split] += 1
            copy_ratio_sum[dimension] += ratio
            copy_lcs_sum[dimension] += lcs_chars
            copy_answer_chars[dimension] += answer_chars
            if ratio >= COPY_REVIEW_THRESHOLD:
                copy_review[dimension] += 1
                copy_review_by_split[split] += 1
            if ratio >= COPY_SEVERE_THRESHOLD:
                copy_severe[dimension] += 1
                copy_severe_by_split[split] += 1

        self_flag, cycle_flag, repetition_flag = _relation_screen(answer)
        fingerprint = _identifier_fingerprint(record)
        if self_flag:
            exact_self[dimension] += 1
            exact_self_by_split[split] += 1
            heuristic_fingerprints["exact_self_relation"].append(fingerprint)
        if cycle_flag:
            two_node_cycle[dimension] += 1
            two_node_cycle_by_split[split] += 1
            heuristic_fingerprints["two_node_relation_cycle"].append(fingerprint)
        if repetition_flag:
            repeated_clause[dimension] += 1
            repeated_clause_by_split[split] += 1
            heuristic_fingerprints["repeated_normalized_clause"].append(fingerprint)

    total_records = len(all_records)
    total_supervised = sum(supervised_counts.values())
    total_template_records = sum(any_template_by_split.values())
    bare_core_counts = bare_core_counts_by_split["train"] + bare_core_counts_by_split["val"]
    distinct_bare_facts = len(bare_core_counts)
    total_known_core_facts = len(KNOWN_CORE_FACTS)
    core_record_share = _round_share(record_counts[CORE], total_records)
    core_token_share = _round_share(supervised_counts[CORE], total_supervised)
    core_density_index = (
        round(core_token_share / core_record_share, 8) if core_record_share else 0.0
    )
    total_copy_eligible = sum(copy_eligible.values())
    total_copy_review = sum(copy_review.values())
    total_copy_severe = sum(copy_severe.values())

    dimension_metrics: dict[str, Any] = {}
    for dimension in DIMENSION_TOTALS:
        records = record_counts[dimension]
        tokens = supervised_counts[dimension]
        record_share = _round_share(records, total_records)
        token_share = _round_share(tokens, total_supervised)
        dimension_metrics[dimension] = {
            "records": records,
            "record_share": record_share,
            "supervised_tokens": tokens,
            "supervised_token_share": token_share,
            "relative_supervision_density_index": (
                round(token_share / record_share, 8) if record_share else 0.0
            ),
        }

    dimension_metrics_by_split: dict[str, dict[str, Any]] = {}
    for split in ("train", "val"):
        split_metrics: dict[str, Any] = {}
        split_records = split_record_counts[split]
        split_tokens = split_supervised_counts[split]
        for dimension in DIMENSION_TOTALS:
            records = record_counts_by_split[split][dimension]
            tokens = supervised_counts_by_split[split][dimension]
            record_share = _round_share(records, split_records)
            token_share = _round_share(tokens, split_tokens)
            split_metrics[dimension] = {
                "records_numerator": records,
                "records_denominator": split_records,
                "record_share": record_share,
                "supervised_tokens_numerator": tokens,
                "supervised_tokens_denominator": split_tokens,
                "supervised_token_share": token_share,
                "relative_supervision_density_index": (
                    round(token_share / record_share, 8) if record_share else 0.0
                ),
            }
        dimension_metrics_by_split[split] = split_metrics

    copy_by_dimension: dict[str, Any] = {}
    for dimension in DIMENSION_TOTALS:
        eligible = copy_eligible[dimension]
        copy_by_dimension[dimension] = {
            "eligible_supported_records": eligible,
            "mean_per_record_copy_ratio": (
                round(copy_ratio_sum[dimension] / eligible, 8) if eligible else 0.0
            ),
            "aggregate_lcs_chars_over_answer_chars": _round_share(
                copy_lcs_sum[dimension], copy_answer_chars[dimension]
            ),
            "records_at_or_above_0_50": copy_review[dimension],
            "share_at_or_above_0_50": _round_share(copy_review[dimension], eligible),
            "records_at_or_above_0_80": copy_severe[dimension],
            "share_at_or_above_0_80": _round_share(copy_severe[dimension], eligible),
        }

    findings: list[dict[str, Any]] = []

    def finding(code: str, observed: float | int, threshold: str, severity: str = "P1") -> None:
        findings.append(
            {
                "code": code,
                "severity": severity,
                "observed": observed,
                "threshold": threshold,
                "decision_scope": "train",
            }
        )

    template_share = _round_share(total_template_records, total_records)
    bare_coverage = _round_share(distinct_bare_facts, total_known_core_facts)
    copy_review_share = _round_share(total_copy_review, total_copy_eligible)
    train_template_share = _round_share(
        any_template_by_split["train"], split_record_counts["train"]
    )
    train_bare_coverage = _round_share(
        len(bare_core_counts_by_split["train"]), total_known_core_facts
    )
    train_core_density_index = dimension_metrics_by_split["train"][CORE][
        "relative_supervision_density_index"
    ]
    train_copy_review_share = _round_share(
        copy_review_by_split["train"], copy_eligible_by_split["train"]
    )
    if train_template_share > TEMPLATE_SHARE_LIMIT:
        finding("fixed_opening_template_concentration", train_template_share, "<=0.20")
    if train_bare_coverage < BARE_CORE_COVERAGE_MINIMUM:
        finding("bare_core_question_coverage_too_low", train_bare_coverage, ">=0.50")
    if train_core_density_index < CORE_DENSITY_INDEX_MINIMUM:
        finding("core_supervision_density_too_low", train_core_density_index, ">=0.75")
    if train_copy_review_share > COPY_REVIEW_THRESHOLD:
        finding("evidence_copy_concentration", train_copy_review_share, "<=0.50")
    if exact_self_by_split["train"] or two_node_cycle_by_split["train"]:
        finding(
            "self_or_cycle_heuristic_requires_review",
            exact_self_by_split["train"] + two_node_cycle_by_split["train"],
            "0 before curated canary release",
            "P2",
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "needs_revision" if findings else "passed",
        "risk_conclusion": {
            "decision_scope": "train",
            "level": "high" if any(item["severity"] == "P1" for item in findings) else (
                "medium" if findings else "low"
            ),
            "decision": (
                "rebuild_sft_v7_1_before_additional_training"
                if findings
                else "semantic_density_gates_passed"
            ),
            "findings": findings,
        },
        "population": {
            "authorized_splits": ["train", "val"],
            "records_by_split": dict(sorted(split_record_counts.items())),
            "supervised_tokens_by_split": dict(sorted(split_supervised_counts.items())),
            "total_records": total_records,
            "total_supervised_tokens": total_supervised,
            "task_family_counts": dict(sorted(family_counts.items())),
        },
        "dimension_semantic_density": dimension_metrics,
        "dimension_semantic_density_by_split": dimension_metrics_by_split,
        "template_opening_analysis": {
            "definition": "record answer starts with the complete frozen prefix",
            "template_prefixes": dict(TEMPLATE_OPENINGS),
            "counts": {
                opening_id: {
                    "train": counts["train"],
                    "val": counts["val"],
                    "total": counts["train"] + counts["val"],
                }
                for opening_id, counts in opening_counts.items()
            },
            "records_matching_any_of_six": total_template_records,
            "share_matching_any_of_six": template_share,
            "overall": {
                "numerator": total_template_records,
                "denominator": total_records,
                "share": template_share,
            },
            "by_split": {
                split: {
                    "numerator": any_template_by_split[split],
                    "denominator": split_record_counts[split],
                    "share": _round_share(
                        any_template_by_split[split], split_record_counts[split]
                    ),
                }
                for split in ("train", "val")
            },
            "decision_scope": "train",
        },
        "bare_core_question_coverage": {
            "definition": (
                "NFKC/lowercase/alphanumeric-only exact equality with one of the "
                "reviewed catalog canonical questions; no suffix or lead is allowed"
            ),
            "catalog_fact_count": total_known_core_facts,
            "matched_record_count": sum(bare_core_counts.values()),
            "distinct_catalog_facts_matched": distinct_bare_facts,
            "distinct_fact_coverage": bare_coverage,
            "match_counts_by_fact_id": dict(sorted(bare_core_counts.items())),
            "overall": {
                "matched_record_count": sum(bare_core_counts.values()),
                "distinct_catalog_facts_matched": distinct_bare_facts,
                "catalog_fact_count": total_known_core_facts,
                "distinct_fact_coverage": bare_coverage,
                "match_counts_by_fact_id": dict(sorted(bare_core_counts.items())),
            },
            "by_split": {
                split: {
                    "matched_record_count": sum(bare_core_counts_by_split[split].values()),
                    "distinct_catalog_facts_matched": len(
                        bare_core_counts_by_split[split]
                    ),
                    "catalog_fact_count": total_known_core_facts,
                    "distinct_fact_coverage": _round_share(
                        len(bare_core_counts_by_split[split]), total_known_core_facts
                    ),
                    "match_counts_by_fact_id": dict(
                        sorted(bare_core_counts_by_split[split].items())
                    ),
                }
                for split in ("train", "val")
            },
            "decision_scope": "train",
        },
        "evidence_copy_analysis": {
            "algorithm": {
                "normalization": (
                    "Unicode NFKC, lowercase, retain only str.isalnum characters; "
                    "Chinese characters remain, whitespace and punctuation are removed"
                ),
                "overlap": (
                    "exact longest contiguous common substring between normalized answer "
                    "and each answer_support.supporting_spans item; take the maximum"
                ),
                "per_record_ratio": "maximum LCS characters / normalized answer characters",
                "eligibility": (
                    "answer_support.status == supported and at least one non-empty "
                    "supporting_spans item"
                ),
                "review_threshold": COPY_REVIEW_THRESHOLD,
                "severe_threshold": COPY_SEVERE_THRESHOLD,
            },
            "eligible_supported_records": total_copy_eligible,
            "records_at_or_above_0_50": total_copy_review,
            "share_at_or_above_0_50": copy_review_share,
            "records_at_or_above_0_80": total_copy_severe,
            "share_at_or_above_0_80": _round_share(total_copy_severe, total_copy_eligible),
            "overall": {
                "numerator_at_or_above_0_50": total_copy_review,
                "denominator_eligible_supported_records": total_copy_eligible,
                "share_at_or_above_0_50": copy_review_share,
                "numerator_at_or_above_0_80": total_copy_severe,
                "share_at_or_above_0_80": _round_share(
                    total_copy_severe, total_copy_eligible
                ),
            },
            "by_split": {
                split: {
                    "numerator_at_or_above_0_50": copy_review_by_split[split],
                    "denominator_eligible_supported_records": copy_eligible_by_split[split],
                    "share_at_or_above_0_50": _round_share(
                        copy_review_by_split[split], copy_eligible_by_split[split]
                    ),
                    "numerator_at_or_above_0_80": copy_severe_by_split[split],
                    "share_at_or_above_0_80": _round_share(
                        copy_severe_by_split[split], copy_eligible_by_split[split]
                    ),
                }
                for split in ("train", "val")
            },
            "decision_scope": "train",
            "by_dimension": copy_by_dimension,
        },
        "self_reference_and_cycle_screen": {
            "scope": "heuristic initial screen only; every hit requires semantic review",
            "definitions": {
                "exact_self_relation": "same catalog entity on both sides of 是/为/和/与/由/属于",
                "two_node_relation_cycle": (
                    "both A-is-B's-relation and B-is-A's-relation occur in one answer"
                ),
                "repeated_normalized_clause": (
                    "a punctuation-delimited normalized clause of at least four characters repeats"
                ),
            },
            "exact_self_relation_records": sum(exact_self.values()),
            "two_node_relation_cycle_records": sum(two_node_cycle.values()),
            "repeated_normalized_clause_records": sum(repeated_clause.values()),
            "by_split": {
                split: {
                    "exact_self_relation_records": exact_self_by_split[split],
                    "two_node_relation_cycle_records": two_node_cycle_by_split[split],
                    "repeated_normalized_clause_records": repeated_clause_by_split[split],
                }
                for split in ("train", "val")
            },
            "decision_scope": "train",
            "by_dimension": {
                dimension: {
                    "exact_self_relation_records": exact_self[dimension],
                    "two_node_relation_cycle_records": two_node_cycle[dimension],
                    "repeated_normalized_clause_records": repeated_clause[dimension],
                }
                for dimension in DIMENSION_TOTALS
            },
            "record_id_fingerprints": {
                key: sorted(values)[:100] for key, values in heuristic_fingerprints.items()
            },
            "fingerprint_truncation_limit_per_category": 100,
        },
        "privacy": {
            "record_body_emitted": False,
            "record_identifier_emitted": False,
            "public_diagnostic_body_accessed": False,
            "sealed_test_body_accessed": False,
            "accessed_split_bodies": ["train", "val"],
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact, content-free human-readable report."""

    population = report["population"]
    template = report["template_opening_analysis"]
    bare = report["bare_core_question_coverage"]
    copy = report["evidence_copy_analysis"]
    screen = report["self_reference_and_cycle_screen"]
    lines = [
        "# M021 SFT v7 语义密度只读审计",
        "",
        f"- 状态：`{report['status']}`",
        f"- 风险级别：`{report['risk_conclusion']['level']}`",
        "- 决策门统计范围：`train`；`val` 与总体统计只用于描述，不参与 Stop/Go。",
        f"- 审计记录：{population['total_records']}",
        f"- 监督 Token：{population['total_supervised_tokens']}",
        "- 数据边界：只读取 `train` 与 `val`；未读取 Public/Sealed 正文。",
        "",
        "## 六种固定开头",
        "",
        "| 开头ID | 固定前缀 | Train | Val | 合计 |",
        "|---|---|---:|---:|---:|",
    ]
    for opening_id, prefix in template["template_prefixes"].items():
        counts = template["counts"][opening_id]
        lines.append(
            f"| `{opening_id}` | `{prefix}` | {counts['train']} | {counts['val']} | {counts['total']} |"
        )
    lines.extend(
        [
            "",
            (
                f"总体命中：{template['overall']['numerator']} / "
                f"{template['overall']['denominator']}（{template['overall']['share']:.2%}）；"
                f"Train：{template['by_split']['train']['numerator']} / "
                f"{template['by_split']['train']['denominator']}"
                f"（{template['by_split']['train']['share']:.2%}，参与门控）；"
                f"Val：{template['by_split']['val']['numerator']} / "
                f"{template['by_split']['val']['denominator']}"
                f"（{template['by_split']['val']['share']:.2%}，仅描述）。"
            ),
            "",
            "## 各维度监督密度（总体，保留旧口径）",
            "",
            "| 维度 | 记录 | 记录占比 | 监督Token | Token占比 | 相对密度 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dimension, metrics in report["dimension_semantic_density"].items():
        lines.append(
            f"| `{dimension}` | {metrics['records']} | {metrics['record_share']:.2%} | "
            f"{metrics['supervised_tokens']} | {metrics['supervised_token_share']:.2%} | "
            f"{metrics['relative_supervision_density_index']:.3f} |"
        )
    core_overall = report["dimension_semantic_density"][CORE]
    core_train = report["dimension_semantic_density_by_split"]["train"][CORE]
    core_val = report["dimension_semantic_density_by_split"]["val"][CORE]
    lines.extend(
        [
            "",
            "核心维度相对密度分范围：",
            "",
            "| 范围 | 记录（分子/分母） | 记录占比 | 监督Token（分子/分母） | Token占比 | 相对密度 | 用于门控 |",
            "|---|---:|---:|---:|---:|---:|---|",
            (
                f"| Train | {core_train['records_numerator']} / {core_train['records_denominator']} | "
                f"{core_train['record_share']:.2%} | {core_train['supervised_tokens_numerator']} / "
                f"{core_train['supervised_tokens_denominator']} | "
                f"{core_train['supervised_token_share']:.2%} | "
                f"{core_train['relative_supervision_density_index']:.3f} | 是 |"
            ),
            (
                f"| Val | {core_val['records_numerator']} / {core_val['records_denominator']} | "
                f"{core_val['record_share']:.2%} | {core_val['supervised_tokens_numerator']} / "
                f"{core_val['supervised_tokens_denominator']} | "
                f"{core_val['supervised_token_share']:.2%} | "
                f"{core_val['relative_supervision_density_index']:.3f} | 否 |"
            ),
            (
                f"| 总体 | {core_overall['records']} / {population['total_records']} | "
                f"{core_overall['record_share']:.2%} | {core_overall['supervised_tokens']} / "
                f"{population['total_supervised_tokens']} | "
                f"{core_overall['supervised_token_share']:.2%} | "
                f"{core_overall['relative_supervision_density_index']:.3f} | 否 |"
            ),
            "",
            "## 核心问题、证据复制与循环初筛",
            "",
            (
                f"- 裸核心问题总体：命中 {bare['overall']['matched_record_count']} 条，覆盖 "
                f"{bare['overall']['distinct_catalog_facts_matched']} / "
                f"{bare['overall']['catalog_fact_count']} 个事实"
                f"（{bare['overall']['distinct_fact_coverage']:.2%}，仅描述）。"
            ),
            (
                f"- 裸核心问题 Train：命中 {bare['by_split']['train']['matched_record_count']} 条，"
                f"覆盖 {bare['by_split']['train']['distinct_catalog_facts_matched']} / "
                f"{bare['by_split']['train']['catalog_fact_count']} 个事实"
                f"（{bare['by_split']['train']['distinct_fact_coverage']:.2%}，参与门控）；"
                f"Val 覆盖 {bare['by_split']['val']['distinct_catalog_facts_matched']} / "
                f"{bare['by_split']['val']['catalog_fact_count']}"
                f"（{bare['by_split']['val']['distinct_fact_coverage']:.2%}，仅描述）。"
            ),
            (
                f"- 证据复制总体：{copy['overall']['numerator_at_or_above_0_50']} / "
                f"{copy['overall']['denominator_eligible_supported_records']}"
                f"（{copy['overall']['share_at_or_above_0_50']:.2%}，仅描述）；Train："
                f"{copy['by_split']['train']['numerator_at_or_above_0_50']} / "
                f"{copy['by_split']['train']['denominator_eligible_supported_records']}"
                f"（{copy['by_split']['train']['share_at_or_above_0_50']:.2%}，参与门控）；"
                f"Val：{copy['by_split']['val']['numerator_at_or_above_0_50']} / "
                f"{copy['by_split']['val']['denominator_eligible_supported_records']}"
                f"（{copy['by_split']['val']['share_at_or_above_0_50']:.2%}，仅描述）。"
            ),
            (
                f"- 自指初筛：{screen['exact_self_relation_records']}；两节点循环："
                f"{screen['two_node_relation_cycle_records']}；重复子句："
                f"{screen['repeated_normalized_clause_records']}。"
            ),
            (
                "- 自指/循环决策只使用 Train：自指 "
                f"{screen['by_split']['train']['exact_self_relation_records']}，循环 "
                f"{screen['by_split']['train']['two_node_relation_cycle_records']}；"
                "总体结果仅是关键词启发式初筛，命中项必须再做语义审核。"
            ),
            "",
            "## 风险结论（仅由 Train 决定）",
            "",
        ]
    )
    for item in report["risk_conclusion"]["findings"]:
        lines.append(
            f"- `{item['severity']}` `{item['code']}`：范围 `{item['decision_scope']}`，"
            f"观测值 `{item['observed']}`，门槛 `{item['threshold']}`。"
        )
    lines.extend(
        [
            "",
            "当前结论：在继续增加训练步数前，先重构 SFT v7.1 的语义密度与裸问题监督。",
            "",
            "## 可复算信息",
            "",
        ]
    )
    for source in report["sources"]:
        lines.append(
            f"- `{source['split']}`：`{source['path']}`，记录 {source['records']}，SHA-256 `{source['sha256']}`。"
        )
    implementation = report["implementation"]
    dirty_marker = "dirty" if implementation["git_dirty"] else "clean"
    lines.extend(
        [
            (
                f"- 实现：`{implementation['path']}`，算法版本 "
                f"`{implementation['algorithm_version']}`，SHA-256 "
                f"`{implementation['sha256']}`，Git `{implementation['git_commit']}` "
                f"（{dirty_marker}）。"
            ),
        ]
    )
    lines.append("")
    return "\n".join(lines)


def execute_audit(
    *,
    repo_root: Path,
    train_path: Path,
    val_path: Path,
    report_path: Path,
    markdown_path: Path,
    run_id: str,
) -> dict[str, Any]:
    """Load the two authorized files, audit, and atomically write reports."""

    root = repo_root.resolve()
    train = _resolve_under_repo(train_path, root, must_exist=True)
    val = _resolve_under_repo(val_path, root, must_exist=True)
    report_file = _resolve_under_repo(report_path, root, must_exist=False)
    markdown_file = _resolve_under_repo(markdown_path, root, must_exist=False)
    loaded_train = load_jsonl(train, expected_split="train")
    loaded_val = load_jsonl(val, expected_split="val")
    implementation = implementation_metadata(root)
    report = audit_semantic_density(
        {"train": loaded_train.records, "val": loaded_val.records}
    )
    report.update(
        {
            "generated_at_utc": utc_now(),
            "run_id": run_id,
            "implementation": implementation,
            "sources": [
                {
                    "split": "train",
                    "path": portable_repo_path(train, root),
                    "records": len(loaded_train.records),
                    "sha256": loaded_train.sha256,
                    "schema_version_counts": loaded_train.schema_counts,
                },
                {
                    "split": "val",
                    "path": portable_repo_path(val, root),
                    "records": len(loaded_val.records),
                    "sha256": loaded_val.sha256,
                    "schema_version_counts": loaded_val.schema_counts,
                },
            ],
            "artifacts": {
                "json_report": portable_repo_path(report_file, root),
                "markdown_report": portable_repo_path(markdown_file, root),
            },
        }
    )
    atomic_write_json(report_file, report)
    atomic_write_text(markdown_file, render_markdown(report))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--preflight-log-level", default="INFO")
    parser.add_argument("--data-log-level", default="INFO")
    parser.add_argument("--validation-log-level", default="INFO")
    parser.add_argument("--orchestrator-log-level", default="INFO")
    parser.add_argument("--no-console-log", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.repo_root.resolve()
    log_dir = _resolve_under_repo(args.log_dir, root, must_exist=False)
    run_id = generate_run_id("sft-v7-semantic-density")
    levels = resolve_module_log_levels(
        {
            "preflight": args.preflight_log_level,
            "data": args.data_log_level,
            "validation": args.validation_log_level,
            "orchestrator": args.orchestrator_log_level,
            "pretrain": "OFF",
            "checkpoint": "OFF",
            "gpu": "OFF",
            "sft": "OFF",
        }
    )
    loggers = configure_module_loggers(
        log_dir,
        run_id,
        levels,
        max_bytes=5 * 1024 * 1024,
        backup_count=3,
        console=not args.no_console_log,
    )
    try:
        loggers["preflight"].info(
            "authorized source boundary verified",
            extra={
                "context": {
                    "authorized_splits": ["train", "val"],
                    "public_diagnostic_body_accessed": False,
                    "sealed_test_body_accessed": False,
                }
            },
        )
        report = execute_audit(
            repo_root=root,
            train_path=args.train,
            val_path=args.val,
            report_path=args.report,
            markdown_path=args.markdown,
            run_id=run_id,
        )
        loggers["data"].info(
            "authorized SFT splits loaded",
            extra={
                "context": {
                    "records_by_split": report["population"]["records_by_split"],
                    "source_sha256s": {
                        source["split"]: source["sha256"] for source in report["sources"]
                    },
                }
            },
        )
        loggers["validation"].info(
            "semantic density audit complete",
            extra={
                "context": {
                    "status": report["status"],
                    "risk_level": report["risk_conclusion"]["level"],
                    "finding_codes": [
                        item["code"] for item in report["risk_conclusion"]["findings"]
                    ],
                }
            },
        )
        loggers["orchestrator"].info(
            "aggregate reports written",
            extra={"context": report["artifacts"]},
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "risk_level": report["risk_conclusion"]["level"],
                    "total_records": report["population"]["total_records"],
                    "artifacts": report["artifacts"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        loggers["validation"].error(
            "semantic density audit failed",
            extra={
                "context": {
                    "error_code": getattr(error, "code", "unexpected_audit_failure"),
                    "error_type": type(error).__name__,
                }
            },
        )
        raise
    finally:
        close_module_loggers(loggers)


if __name__ == "__main__":
    raise SystemExit(main())
