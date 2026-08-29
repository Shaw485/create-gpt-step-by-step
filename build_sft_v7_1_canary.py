"""Build and audit the M021 64-record SFT v7.1 capacity probe.

Questions and answers come from the reviewed Canary config, authored and
cross-checked with Codex AI assistance, and are
cross-checked against ``sft_v7_vertical_catalog.KNOWN_CORE_FACTS``.  The frozen
v7 ``manifest.json`` contributes release metadata only.  Before declaring the
Canary ready, the builder also verifies the exact parent-manifest identity,
base-checkpoint bytes, tokenizer bytes and BPE token-manifest bytes.  It never
opens the v7 train, public-diagnostic, or sealed JSONL bodies.  Answers are
deliberately short so the experiment tests whether the 14.9M model can learn a
small, explicit question-answer mapping before a larger SFT redesign is
attempted.

Diagnostics are written as independently filterable rotating JSONL logs for
``data``, ``validation`` and ``orchestrator``.  Use the corresponding CLI
``--*-log-level`` option, or ``GPT_CANARY_LOG_LEVEL_<MODULE>``, to change one
module without enabling noisy global debug output.  Record bodies are never
placed in logs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from sft_v7_vertical_catalog import KNOWN_CORE_FACTS
from training_runtime import (
    atomic_write_json,
    atomic_write_text,
    canonical_json_sha256,
    close_module_loggers,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
    resolve_module_log_levels,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = Path("configs/sft_v7_1_canary_facts.json")
DEFAULT_PARENT_MANIFEST = Path("data/sft/v7/manifest.json")
DEFAULT_BASE_CHECKPOINT = Path(
    "runs/formal_pretrain_14m_bpe3000_6000/checkpoints/step_05750.pt"
)
DEFAULT_TOKENIZER = Path("data/scaling_a/bpe_3000/tokenizer.json")
DEFAULT_TOKEN_MANIFEST = Path("data/scaling_a/bpe_3000/token_manifest.json")
DEFAULT_OUTPUT_DIR = Path("data/sft/v7_1_canary")
DEFAULT_REPORT_JSON = Path(
    "reports/milestones/021_sft_v7_1_canary/canary_data_report.json"
)
DEFAULT_REPORT_MD = Path(
    "reports/milestones/021_sft_v7_1_canary/canary_data_report.md"
)
DEFAULT_LOG_DIR = Path("logs/sft_v7_1_canary_build")

CONFIG_SCHEMA = "sft-v7.1-canary-config/v1"
RECORD_SCHEMA = "sft_v7_1_canary/1.0"
MANIFEST_SCHEMA = "sft-v7.1-canary-manifest/v1"
REPORT_SCHEMA = "sft-v7.1-canary-report/v1"
OUTPUT_NAMES = {"train": "train.jsonl", "holdout_eval": "holdout_eval.jsonl"}
PRIMARY_DIMENSION = "parameter_core_fact_and_correction"
TASK_FAMILY = "canary_known_core"
BASE_CHECKPOINT_BINDING = {
    "path": "runs/formal_pretrain_14m_bpe3000_6000/checkpoints/step_05750.pt",
    "sha256": "bfe4fec5e6045d4c06d22393e7c2079fdc03897be71829c9d9dcbaf0fcaf5c1e",
    "step": 5750,
    "parameter_count": 14880745,
}
TOKENIZER_BINDING = {
    "path": "data/scaling_a/bpe_3000/tokenizer.json",
    "sha256": "e70cf3dc0ed185a6b22ab7dc08b6a850eeb59864ba161dd156c644e003862822",
    "vocab_size": 7465,
    "context_limit": 512,
}
TOKEN_MANIFEST_BINDING = {
    "path": "data/scaling_a/bpe_3000/token_manifest.json",
    "sha256": "5d10245eac86e4dbafef908cb2d915bb1effcf61ad977b4de96d8d64d30809c7",
    "schema_version": "bpe-v4/v1",
    "status": "ready",
    "vocab_size": 7465,
}
PARENT_MANIFEST_BINDING = {
    "path": "data/sft/v7/manifest.json",
    "sha256": "422c35fa130a3e6fc3f656019515fc7c1115616aa396712ea501751aeda1b9e9",
    "dataset_identity_sha256": "47fb0f0af2aaa61f3239883f22966f3b0828a8a3942c78436b07ef5f5118d133",
}
CANARY_CONFIG_BINDING = {
    "path": "configs/sft_v7_1_canary_facts.json",
    "sha256": "0e139053aa018bd25c55d4bece3e79fdfdd85416bbd3bf21b3ebcbe22b75fe44",
}

TRAIN_ROLES = (
    "bare_question",
    "natural_paraphrase_1",
    "natural_paraphrase_2",
    "direct_answer_request",
    "correction",
    "relation_direction_forward",
    "relation_direction_reverse",
    "contrastive_check",
)
HOLDOUT_ROLES = ("holdout_paraphrase_1", "holdout_paraphrase_2")

META_ANSWER_PREFIXES = (
    "原文写道",
    "原著写道",
    "文本中出现",
    "根据原文",
    "根据原著",
    "资料显示",
    "可以先记录问题",
    "可以先",
    "从原文来看",
)
REFUSAL_MARKERS = ("资料不足", "无法确定", "不能确认", "需要检索")
FORBIDDEN_TERMS = ("原文写道", "文本中出现", "可以先记录问题")


class CanaryBuildError(ValueError):
    """Raised when the reviewed Canary contract is incomplete or has drifted."""

    def __init__(self, code: str, message: str, remediation: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.remediation = remediation


def _require(condition: bool, code: str, message: str, remediation: str) -> None:
    if not condition:
        raise CanaryBuildError(code, message, remediation)


def _portable_path(path: Path, *, role: str) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return f"artifact://{role}/{resolved.name}"


def _frozen_repository_file(
    path: Path,
    *,
    expected_relative_path: str,
    role: str,
) -> Path:
    """Resolve one frozen input without permitting lookalike external files."""

    candidate = path if path.is_absolute() else REPOSITORY_ROOT / path
    resolved = candidate.resolve()
    expected = (REPOSITORY_ROOT / expected_relative_path).resolve()
    _require(
        resolved == expected,
        f"{role}_PATH_MISMATCH",
        f"The {role.lower().replace('_', ' ')} is not the frozen repository artifact.",
        f"Use {expected_relative_path} from this repository.",
    )
    _require(
        resolved.is_file(),
        f"MISSING_{role}",
        f"The frozen {role.lower().replace('_', ' ')} is missing.",
        f"Restore {expected_relative_path} before rebuilding the Canary.",
    )
    return resolved


def _verify_file_sha256(
    path: Path,
    *,
    expected_sha256: str,
    role: str,
) -> str:
    actual_sha256 = file_sha256(path)
    _require(
        actual_sha256 == expected_sha256,
        f"{role}_SHA256_MISMATCH",
        f"The frozen {role.lower().replace('_', ' ')} bytes have changed.",
        "Restore the reviewed frozen artifact; do not derive a Canary from drifted inputs.",
    )
    return actual_sha256


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_question(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold(), flags=re.UNICODE)


def _jsonl_text(records: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    )


def _load_json_object(path: Path, *, code: str) -> dict[str, Any]:
    _require(
        path.is_file(),
        f"MISSING_{code}",
        f"Required metadata file does not exist: {path.name}",
        "Restore the reviewed metadata file before rebuilding the Canary.",
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CanaryBuildError(
            f"INVALID_{code}",
            f"Cannot parse {path.name}: {error}",
            "Repair the JSON and rerun the builder.",
        ) from error
    _require(
        isinstance(payload, dict),
        f"INVALID_{code}_ROOT",
        f"{path.name} must contain one JSON object.",
        "Replace the root value with an object.",
    )
    return payload


def _validate_parent_manifest(
    payload: Mapping[str, Any],
    *,
    actual_sha256: str,
) -> None:
    _require(
        actual_sha256 == PARENT_MANIFEST_BINDING["sha256"],
        "PARENT_MANIFEST_SHA256_MISMATCH",
        "The parent v7 manifest bytes have changed.",
        "Restore the frozen M020 v7 manifest before deriving the Canary.",
    )
    _require(
        payload.get("manifest_schema_version") == "sft-v7-vertical-manifest/v1",
        "PARENT_SCHEMA_DRIFT",
        "The parent v7 manifest schema is not the reviewed release schema.",
        "Use the frozen M020 v7 manifest.",
    )
    _require(
        payload.get("frozen_status") == "frozen_unspent",
        "PARENT_NOT_FROZEN",
        "The parent v7 dataset is not in frozen_unspent state.",
        "Resolve the parent release state before deriving the capacity probe.",
    )
    _require(
        payload.get("dataset_identity_sha256")
        == PARENT_MANIFEST_BINDING["dataset_identity_sha256"],
        "PARENT_DATASET_IDENTITY_MISMATCH",
        "The parent v7 dataset identity is not the reviewed frozen identity.",
        "Restore the frozen M020 v7 manifest and do not substitute another dataset.",
    )
    parent_tokenizer = payload.get("tokenizer")
    _require(
        isinstance(parent_tokenizer, Mapping)
        and parent_tokenizer.get("path") == TOKENIZER_BINDING["path"]
        and parent_tokenizer.get("sha256") == TOKENIZER_BINDING["sha256"]
        and int(parent_tokenizer.get("context_limit", -1))
        == TOKENIZER_BINDING["context_limit"],
        "PARENT_TOKENIZER_BINDING_MISMATCH",
        "The parent manifest no longer binds the reviewed tokenizer and 512-token context.",
        "Restore the frozen M020 v7 manifest tokenizer metadata.",
    )
    known_core = payload.get("known_core")
    _require(
        isinstance(known_core, Mapping)
        and int(known_core.get("reviewed_fact_count", 0)) >= 8,
        "INSUFFICIENT_REVIEWED_FACTS",
        "The parent manifest does not declare at least eight reviewed facts.",
        "Restore the audited v7 known-core metadata.",
    )


def _validate_frozen_training_inputs(
    *,
    base_checkpoint_path: Path,
    tokenizer_path: Path,
    token_manifest_path: Path,
) -> dict[str, Any]:
    """Verify immutable model/tokenizer lineage without loading model weights."""

    base = _frozen_repository_file(
        base_checkpoint_path,
        expected_relative_path=BASE_CHECKPOINT_BINDING["path"],
        role="BASE_CHECKPOINT",
    )
    base_sha256 = _verify_file_sha256(
        base,
        expected_sha256=BASE_CHECKPOINT_BINDING["sha256"],
        role="BASE_CHECKPOINT",
    )
    _require(
        base.name == "step_05750.pt"
        and BASE_CHECKPOINT_BINDING["step"] == 5750
        and BASE_CHECKPOINT_BINDING["parameter_count"] == 14880745,
        "BASE_CHECKPOINT_METADATA_MISMATCH",
        "The frozen base step or parameter-count binding has drifted.",
        "Restore the Step 5750 / 14,880,745-parameter binding.",
    )

    tokenizer = _frozen_repository_file(
        tokenizer_path,
        expected_relative_path=TOKENIZER_BINDING["path"],
        role="TOKENIZER",
    )
    tokenizer_sha256 = _verify_file_sha256(
        tokenizer,
        expected_sha256=TOKENIZER_BINDING["sha256"],
        role="TOKENIZER",
    )
    tokenizer_payload = _load_json_object(tokenizer, code="TOKENIZER")
    _require(
        tokenizer_payload.get("tokenizer_type") == "character_seeded_bpe"
        and isinstance(tokenizer_payload.get("tokens"), list)
        and len(tokenizer_payload["tokens"]) == TOKENIZER_BINDING["vocab_size"],
        "TOKENIZER_VOCAB_MISMATCH",
        "The frozen tokenizer does not expose the reviewed 7,465-token vocabulary.",
        "Restore the reviewed BPE-3000 tokenizer.",
    )
    token_manifest = _frozen_repository_file(
        token_manifest_path,
        expected_relative_path=TOKEN_MANIFEST_BINDING["path"],
        role="TOKEN_MANIFEST",
    )
    token_manifest_sha256 = _verify_file_sha256(
        token_manifest,
        expected_sha256=TOKEN_MANIFEST_BINDING["sha256"],
        role="TOKEN_MANIFEST",
    )
    token_manifest_payload = _load_json_object(token_manifest, code="TOKEN_MANIFEST")
    _require(
        token_manifest_payload.get("schema_version")
        == TOKEN_MANIFEST_BINDING["schema_version"]
        and token_manifest_payload.get("status") == TOKEN_MANIFEST_BINDING["status"]
        and token_manifest_payload.get("tokenizer_path") == TOKENIZER_BINDING["path"]
        and token_manifest_payload.get("tokenizer_sha256") == tokenizer_sha256
        and int(token_manifest_payload.get("vocab_size", -1))
        == TOKEN_MANIFEST_BINDING["vocab_size"],
        "TOKEN_MANIFEST_IDENTITY_MISMATCH",
        "The BPE token manifest does not bind the reviewed tokenizer identity.",
        "Restore the frozen BPE-3000 token manifest.",
    )
    return {
        "base_checkpoint": {
            **BASE_CHECKPOINT_BINDING,
            "actual_sha256": base_sha256,
            "verified": True,
        },
        "tokenizer": {
            **TOKENIZER_BINDING,
            "actual_sha256": tokenizer_sha256,
            "verified": True,
        },
        "token_manifest": {
            **TOKEN_MANIFEST_BINDING,
            "actual_sha256": token_manifest_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "verified": True,
        },
    }


def load_and_validate_config(
    config_path: Path,
    parent_manifest_path: Path,
    *,
    base_checkpoint_path: Path = DEFAULT_BASE_CHECKPOINT,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
    token_manifest_path: Path = DEFAULT_TOKEN_MANIFEST,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the frozen selection without opening any v7 JSONL body."""

    reviewed_config_path = _frozen_repository_file(
        config_path,
        expected_relative_path=CANARY_CONFIG_BINDING["path"],
        role="CANARY_CONFIG",
    )
    config_sha256 = _verify_file_sha256(
        reviewed_config_path,
        expected_sha256=CANARY_CONFIG_BINDING["sha256"],
        role="CANARY_CONFIG",
    )
    parent_path = _frozen_repository_file(
        parent_manifest_path,
        expected_relative_path=PARENT_MANIFEST_BINDING["path"],
        role="PARENT_MANIFEST",
    )
    parent_sha256 = file_sha256(parent_path)
    config = _load_json_object(reviewed_config_path, code="CONFIG")
    parent_manifest = _load_json_object(parent_path, code="PARENT_MANIFEST")
    _validate_parent_manifest(parent_manifest, actual_sha256=parent_sha256)
    frozen_inputs = _validate_frozen_training_inputs(
        base_checkpoint_path=base_checkpoint_path,
        tokenizer_path=tokenizer_path,
        token_manifest_path=token_manifest_path,
    )
    _require(
        config.get("schema_version") == CONFIG_SCHEMA,
        "CONFIG_SCHEMA_DRIFT",
        "Canary config schema is not supported.",
        f"Set schema_version to {CONFIG_SCHEMA}.",
    )
    _require(
        config.get("source_catalog") == "sft_v7_vertical_catalog.KNOWN_CORE_FACTS",
        "UNREVIEWED_SOURCE_CATALOG",
        "Canary facts are not bound to the reviewed known-core catalog.",
        "Select facts only by KNOWN_CORE_FACTS fact_id.",
    )
    _require(
        config.get("eos_policy") == "append_during_encoding",
        "EOS_POLICY_DRIFT",
        "EOS must be added at encoding time, not embedded in JSONL answers.",
        "Restore eos_policy=append_during_encoding.",
    )
    facts = config.get("facts")
    _require(
        isinstance(facts, list) and len(facts) == 8,
        "FACT_COUNT",
        "The Canary must contain exactly eight reviewed facts.",
        "Select the frozen eight fact IDs and no others.",
    )

    catalog = {fact.fact_id: fact for fact in KNOWN_CORE_FACTS}
    selected: dict[str, Any] = {}
    for item in facts:
        _require(
            isinstance(item, Mapping),
            "INVALID_FACT_CONFIG",
            "Every configured fact must be a JSON object.",
            "Repair the fact entry.",
        )
        fact_id = str(item.get("fact_id", ""))
        _require(
            fact_id in catalog and fact_id not in selected,
            "UNKNOWN_OR_DUPLICATE_FACT",
            "A fact ID is unknown or selected more than once.",
            "Use eight unique fact IDs from KNOWN_CORE_FACTS.",
        )
        source_fact = catalog[fact_id]
        expected_terms = item.get("expected_required_terms")
        _require(
            isinstance(expected_terms, list)
            and tuple(str(term) for term in expected_terms) == source_fact.required_terms,
            "REQUIRED_TERMS_DRIFT",
            f"Required terms no longer match reviewed metadata for {fact_id}.",
            "Copy required_terms exactly from the reviewed known-core fact.",
        )
        answer = item.get("answer")
        _require(
            isinstance(answer, str) and 5 <= len(answer.strip()) <= 45,
            "ANSWER_LENGTH",
            f"The direct answer for {fact_id} must contain 5-45 characters.",
            "Write one short, direct sentence containing all required terms.",
        )
        answer = answer.strip()
        _require(
            all(term in answer for term in source_fact.required_terms),
            "ANSWER_REQUIRED_TERM",
            f"The direct answer for {fact_id} misses a reviewed required term.",
            "Add the missing reviewed keypoint without adding unsupported detail.",
        )
        _require(
            not answer.startswith(META_ANSWER_PREFIXES)
            and not any(marker in answer for marker in REFUSAL_MARKERS),
            "ANSWER_META_OR_REFUSAL",
            f"The direct answer for {fact_id} uses meta-language or refuses a known fact.",
            "Answer the reviewed fact directly.",
        )
        train_questions = item.get("train_questions")
        holdout_questions = item.get("holdout_questions")
        _require(
            isinstance(train_questions, Mapping)
            and tuple(train_questions.keys()) == TRAIN_ROLES,
            "TRAIN_ROLE_CONTRACT",
            f"Training roles are incomplete or reordered for {fact_id}.",
            "Provide all eight frozen semantic roles in order.",
        )
        _require(
            isinstance(holdout_questions, Mapping)
            and tuple(holdout_questions.keys()) == HOLDOUT_ROLES,
            "HOLDOUT_ROLE_CONTRACT",
            f"Holdout roles are incomplete or reordered for {fact_id}.",
            "Provide exactly two held-out paraphrase roles in order.",
        )
        for role, question in (*train_questions.items(), *holdout_questions.items()):
            _require(
                isinstance(question, str)
                and question.strip() == question
                and 4 <= len(question) <= 60
                and "<EOS>" not in question,
                "INVALID_QUESTION",
                f"Question {role} for {fact_id} is empty, padded, too long, or embeds EOS.",
                "Use a concise natural-language question; EOS is appended later.",
            )
        selected[fact_id] = source_fact

    expected_ids = {
        "xiaoyan_identity",
        "xiaozhan_identity",
        "yaochen_identity",
        "yaolao_yaochen_alias",
        "yaolao_teacher",
        "fanjue_identity",
        "yihuo_role",
        "yunlanzong_identity",
    }
    _require(
        set(selected) == expected_ids,
        "FACT_SELECTION_DRIFT",
        "The selected eight-fact capacity probe has changed.",
        "Restore the reviewed identity, relation, setting and organization fact IDs.",
    )
    return config, parent_manifest, selected, {
        "canary_config": {
            **CANARY_CONFIG_BINDING,
            "actual_sha256": config_sha256,
            "verified": True,
        },
        "parent_manifest": {
            **PARENT_MANIFEST_BINDING,
            "actual_sha256": parent_sha256,
            "actual_dataset_identity_sha256": parent_manifest.get(
                "dataset_identity_sha256"
            ),
            "verified": True,
        },
        **frozen_inputs,
    }


def _record(
    *,
    split: str,
    fact: Any,
    question: str,
    answer: str,
    prompt_role: str,
) -> dict[str, Any]:
    stable_id = _sha256_text(
        "|".join((RECORD_SCHEMA, split, fact.fact_id, prompt_role, question, answer))
    )[:20]
    is_training = split == "train"
    return {
        "schema_version": RECORD_SCHEMA,
        "id": f"canary-{split}-{stable_id}",
        "split": split,
        "fact_id": fact.fact_id,
        "primary_dimension": PRIMARY_DIMENSION,
        "task_family": TASK_FAMILY,
        "question": question,
        "answer": answer,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "prompt_role": prompt_role,
        "supervision": {
            "assistant_only_loss": True,
            "eos_appended_by_encoder": True,
            "use_for_training": is_training,
            "semantic_role": (
                "optimization_target" if is_training else "evaluation_only_unseen_paraphrase"
            ),
        },
        "source": {
            "kind": "reviewed_v7_known_core_metadata",
            "catalog": "sft_v7_vertical_catalog.KNOWN_CORE_FACTS",
            "catalog_fact_id": fact.fact_id,
            "entity": fact.entity,
            "required_terms": list(fact.required_terms),
            "evidence_line_numbers": list(fact.evidence_lines),
            "acceptance_case_id": fact.acceptance_case_id,
            "contains_evidence_body": False,
        },
        "evaluation": {
            "metric": "required_terms_all",
            "required_terms": list(fact.required_terms),
            "forbidden_terms": list(FORBIDDEN_TERMS),
            "known_fact": True,
        },
    }


def build_records(
    config: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {"train": [], "holdout_eval": []}
    for item in config["facts"]:
        fact_id = str(item["fact_id"])
        fact = selected[fact_id]
        answer = str(item["answer"])
        for role, question in item["train_questions"].items():
            records["train"].append(
                _record(
                    split="train",
                    fact=fact,
                    question=str(question),
                    answer=answer,
                    prompt_role=str(role),
                )
            )
        for role, question in item["holdout_questions"].items():
            records["holdout_eval"].append(
                _record(
                    split="holdout_eval",
                    fact=fact,
                    question=str(question),
                    answer=answer,
                    prompt_role=str(role),
                )
            )
    return records


def validate_records(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Apply all capacity-probe gates without logging question or answer text."""

    _require(
        set(records) == {"train", "holdout_eval"},
        "SPLIT_SET",
        "Canary split set is incomplete.",
        "Emit train and holdout_eval only.",
    )
    _require(
        len(records["train"]) == 64 and len(records["holdout_eval"]) == 16,
        "SPLIT_COUNTS",
        "Canary must contain 64 training and 16 held-out records.",
        "Emit eight training and two holdout wordings per fact.",
    )
    fact_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    role_counts: Counter[str] = Counter()
    exact_questions: Counter[str] = Counter()
    normalized_questions: Counter[str] = Counter()
    ids: Counter[str] = Counter()
    prefix_matches = 0
    required_term_failures = 0
    forbidden_term_failures = 0
    semantic_role_failures = 0
    eos_literal_count = 0

    for split, split_records in records.items():
        expected_training = split == "train"
        expected_roles = set(TRAIN_ROLES if expected_training else HOLDOUT_ROLES)
        for record in split_records:
            fact_id = str(record.get("fact_id", ""))
            question = str(record.get("question", ""))
            answer = str(record.get("answer", ""))
            role = str(record.get("prompt_role", ""))
            evaluation = record.get("evaluation", {})
            supervision = record.get("supervision", {})
            source = record.get("source", {})
            messages = record.get("messages", [])
            fact_split_counts[fact_id][split] += 1
            role_counts[f"{split}:{role}"] += 1
            exact_questions[question] += 1
            normalized_questions[_normalize_question(question)] += 1
            ids[str(record.get("id", ""))] += 1
            prefix_matches += int(answer.startswith(META_ANSWER_PREFIXES))
            required_term_failures += int(
                not all(str(term) in answer for term in evaluation.get("required_terms", []))
            )
            forbidden_term_failures += int(
                any(str(term) in answer for term in evaluation.get("forbidden_terms", []))
            )
            semantic_role_failures += int(
                role not in expected_roles
                or supervision.get("use_for_training") is not expected_training
                or supervision.get("assistant_only_loss") is not True
                or supervision.get("eos_appended_by_encoder") is not True
                or supervision.get("semantic_role")
                != (
                    "optimization_target"
                    if expected_training
                    else "evaluation_only_unseen_paraphrase"
                )
            )
            eos_literal_count += int("<EOS>" in question or "<EOS>" in answer)
            _require(
                record.get("schema_version") == RECORD_SCHEMA
                and record.get("split") == split,
                "RECORD_SCHEMA_OR_SPLIT",
                "A generated record has the wrong schema or split.",
                "Rebuild with the frozen record factory.",
            )
            _require(
                source.get("catalog_fact_id") == fact_id
                and source.get("kind") == "reviewed_v7_known_core_metadata"
                and source.get("contains_evidence_body") is False,
                "SOURCE_FACT_BINDING",
                "A generated record is not bound to its reviewed fact metadata.",
                "Rebuild source metadata from KNOWN_CORE_FACTS.",
            )
            _require(
                messages
                == [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                "MESSAGE_BODY_MISMATCH",
                "Question/answer fields no longer match the serialized conversation.",
                "Regenerate messages from the canonical fields.",
            )

    _require(
        len(fact_split_counts) == 8
        and all(counts["train"] == 8 and counts["holdout_eval"] == 2 for counts in fact_split_counts.values()),
        "PER_FACT_QUOTA",
        "Every fact must have eight training and two held-out paraphrases.",
        "Restore the frozen per-fact role banks.",
    )
    _require(
        max(ids.values(), default=0) == 1,
        "DUPLICATE_ID",
        "Generated record IDs are not unique.",
        "Include split, fact, role and content in the stable ID.",
    )
    _require(
        max(exact_questions.values(), default=0) == 1,
        "DUPLICATE_QUESTION",
        "An exact question is repeated.",
        "Use a distinct natural wording for every record.",
    )
    _require(
        max(normalized_questions.values(), default=0) == 1,
        "NORMALIZED_QUESTION_COLLISION",
        "Two questions collide after punctuation and whitespace normalization.",
        "Rewrite one question semantically rather than changing punctuation.",
    )
    train_questions = {str(record["question"]) for record in records["train"]}
    eval_questions = {str(record["question"]) for record in records["holdout_eval"]}
    _require(
        not train_questions.intersection(eval_questions),
        "TRAIN_HOLDOUT_OVERLAP",
        "A held-out paraphrase is present in training.",
        "Replace it with genuinely unseen wording.",
    )
    _require(
        prefix_matches == 0,
        "META_PREFIX",
        "A direct answer starts with a banned meta-template.",
        "Lead with the answer itself.",
    )
    _require(
        required_term_failures == 0 and forbidden_term_failures == 0,
        "ANSWER_KEYPOINT_GATE",
        "A reference answer misses a required term or contains forbidden meta-language.",
        "Repair the concise answer and rerun the build.",
    )
    _require(
        semantic_role_failures == 0,
        "SEMANTIC_ROLE_GATE",
        "A training or evaluation record has an ambiguous supervision role.",
        "Keep holdout use_for_training=false and train use_for_training=true.",
    )
    _require(
        eos_literal_count == 0,
        "EOS_LITERAL",
        "EOS was embedded in JSONL content.",
        "Append EOS only during BPE encoding.",
    )
    return {
        "status": "pass",
        "fact_count": len(fact_split_counts),
        "train_count": len(records["train"]),
        "holdout_eval_count": len(records["holdout_eval"]),
        "train_questions_per_fact": 8,
        "holdout_questions_per_fact": 2,
        "exact_question_duplicates": 0,
        "normalized_question_duplicates": 0,
        "train_holdout_exact_overlap": 0,
        "meta_answer_prefix_matches": prefix_matches,
        "required_term_failures": required_term_failures,
        "forbidden_term_failures": forbidden_term_failures,
        "semantic_role_failures": semantic_role_failures,
        "literal_eos_occurrences": eos_literal_count,
        "role_counts": dict(sorted(role_counts.items())),
        "fact_split_counts": {
            fact_id: dict(sorted(counts.items()))
            for fact_id, counts in sorted(fact_split_counts.items())
        },
    }


def write_release(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    config: Mapping[str, Any],
    config_path: Path,
    parent_manifest: Mapping[str, Any],
    parent_manifest_path: Path,
    output_dir: Path,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    split_files: dict[str, dict[str, Any]] = {}
    for split, filename in OUTPUT_NAMES.items():
        destination = output_dir / filename
        atomic_write_text(destination, _jsonl_text(records[split]))
        split_files[split] = {
            "path": filename,
            "count": len(records[split]),
            "sha256": file_sha256(destination),
            "schema_version": RECORD_SCHEMA,
        }
    identity = {
        split: [str(record["id"]) for record in records[split]]
        for split in OUTPUT_NAMES
    }
    selected_facts = []
    catalog = {fact.fact_id: fact for fact in KNOWN_CORE_FACTS}
    for item in config["facts"]:
        source_fact = catalog[str(item["fact_id"])]
        selected_facts.append(
            {
                "fact_id": source_fact.fact_id,
                "entity": source_fact.entity,
                "required_terms": list(source_fact.required_terms),
                "train_count": 8,
                "holdout_eval_count": 2,
            }
        )
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA,
        "record_schema_version": RECORD_SCHEMA,
        "status": "frozen_canary_ready",
        "record_count": 80,
        "split_totals": {"train": 64, "holdout_eval": 16},
        "split_files": split_files,
        "dataset_identity_sha256": canonical_json_sha256(identity),
        "config": {
            "path": _portable_path(config_path, role="canary_config"),
            "sha256": file_sha256(config_path),
            "canonical_sha256": canonical_json_sha256(config),
        },
        "source": {
            "kind": "reviewed_v7_train_known_core_metadata_only",
            "catalog": "sft_v7_vertical_catalog.KNOWN_CORE_FACTS",
            "catalog_path": "sft_v7_vertical_catalog.py",
            "parent_manifest_path": _portable_path(
                parent_manifest_path, role="parent_manifest"
            ),
            "parent_manifest_sha256": file_sha256(parent_manifest_path),
            "parent_dataset_identity_sha256": parent_manifest.get(
                "dataset_identity_sha256"
            ),
            "selected_facts": selected_facts,
        },
        "access_audit": {
            "v7_train_body_read": False,
            "v7_public_body_read": False,
            "v7_sealed_body_read": False,
            "formal_corpus_body_read": False,
        },
        "supervision_policy": {
            "assistant_only_loss": True,
            "eos_appended_by_encoder": True,
            "jsonl_contains_literal_eos": False,
            "holdout_eval_must_not_train": True,
        },
        "training_binding": {
            "base_checkpoint": dict(BASE_CHECKPOINT_BINDING),
            "tokenizer": dict(TOKENIZER_BINDING),
        },
        "quality_summary": dict(validation),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _build_report(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    facts = [
        {
            "fact_id": item["fact_id"],
            "entity": item["entity"],
            "required_terms": item["required_terms"],
            "train_count": item["train_count"],
            "holdout_eval_count": item["holdout_eval_count"],
        }
        for item in manifest["source"]["selected_facts"]
    ]
    quality = manifest["quality_summary"]
    gates = {
        "exactly_eight_reviewed_facts": len(facts) == 8,
        "exactly_64_training_records": quality["train_count"] == 64,
        "exactly_16_unseen_holdout_paraphrases": quality["holdout_eval_count"] == 16,
        "all_required_terms_present": quality["required_term_failures"] == 0,
        "zero_banned_meta_prefixes": quality["meta_answer_prefix_matches"] == 0,
        "zero_exact_or_normalized_question_duplicates": (
            quality["exact_question_duplicates"] == 0
            and quality["normalized_question_duplicates"] == 0
        ),
        "train_and_eval_roles_unambiguous": quality["semantic_role_failures"] == 0,
        "eos_deferred_to_encoding": quality["literal_eos_occurrences"] == 0,
        "public_and_sealed_bodies_untouched": (
            manifest["access_audit"]["v7_public_body_read"] is False
            and manifest["access_audit"]["v7_sealed_body_read"] is False
        ),
        "frozen_lineage_files_verified": all(
            bool(item.get("verified")) for item in verification.values()
        ),
    }
    return {
        "report_schema_version": REPORT_SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "manifest": {
            "path": _portable_path(manifest_path, role="canary_manifest"),
            "sha256": file_sha256(manifest_path),
            "dataset_identity_sha256": manifest["dataset_identity_sha256"],
        },
        "counts": {
            "facts": len(facts),
            "train": quality["train_count"],
            "holdout_eval": quality["holdout_eval_count"],
            "total": quality["train_count"] + quality["holdout_eval_count"],
        },
        "facts": facts,
        "quality_gates": gates,
        "quality_metrics": {
            key: quality[key]
            for key in (
                "exact_question_duplicates",
                "normalized_question_duplicates",
                "train_holdout_exact_overlap",
                "meta_answer_prefix_matches",
                "required_term_failures",
                "forbidden_term_failures",
                "semantic_role_failures",
                "literal_eos_occurrences",
            )
        },
        "artifact_files": manifest["split_files"],
        "source_access": manifest["access_audit"],
        "lineage_verification": dict(verification),
        "logging": {
            "directory": "logs/sft_v7_1_canary_build",
            "modules": ["data", "validation", "orchestrator"],
            "format": "rotating JSONL with UTC timestamp and run_id",
            "default_level": "INFO",
            "per_module_cli": [
                "--data-log-level",
                "--validation-log-level",
                "--orchestrator-log-level",
            ],
            "per_module_environment_prefix": "GPT_CANARY_LOG_LEVEL_",
            "record_bodies_logged": False,
            "sensitive_fields_redacted": True,
            "rotation_defaults": {"max_bytes": 1048576, "backup_count": 3},
        },
    }


def _report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# M021 SFT v7.1 Canary 数据报告",
        "",
        f"状态：**{report['status']}**",
        "",
        "## 数量",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
        f"| 已审核事实 | {report['counts']['facts']} |",
        f"| 训练问法 | {report['counts']['train']} |",
        f"| 未见改写评估 | {report['counts']['holdout_eval']} |",
        f"| 总记录 | {report['counts']['total']} |",
        "",
        "## 事实覆盖",
        "",
        "| fact_id | 实体 | required terms | Train | Holdout |",
        "|---|---|---|---:|---:|",
    ]
    for fact in report["facts"]:
        terms = "、".join(fact["required_terms"])
        lines.append(
            f"| `{fact['fact_id']}` | {fact['entity']} | {terms} | "
            f"{fact['train_count']} | {fact['holdout_eval_count']} |"
        )
    lines.extend(
        [
            "",
            "## 质量门",
            "",
            "| 质量门 | 结果 |",
            "|---|---|",
        ]
    )
    for name, passed in report["quality_gates"].items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## 数据角色",
            "",
            "`train.jsonl` 的64条记录是优化目标；`holdout_eval.jsonl` 的16条记录只用于未见改写评估，`use_for_training=false`。JSONL中不写入字面量 `<EOS>`，编码器在每个assistant答案末尾追加EOS。",
            "",
            "问题与答案来自已审核的 `configs/sft_v7_1_canary_facts.json`，由 Codex AI 辅助构造并逐项与 `KNOWN_CORE_FACTS` 交叉核对；这不是独立真人签字。父 v7 `manifest.json` 只提供冻结版本和数据集身份元数据。构建还会对 Step 5750 基座、BPE tokenizer 与 token manifest 做路径和 SHA-256 闭环校验，但不会加载权重，也不会读取v7 train、public、sealed JSONL正文或正式预训练语料正文。",
            "",
            "## 日志和独立调试",
            "",
            "日志位于 `logs/sft_v7_1_canary_build/`，data、validation、orchestrator分别写入轮转JSONL。可使用 `--data-log-level`、`--validation-log-level`、`--orchestrator-log-level` 独立调整，也可设置 `GPT_CANARY_LOG_LEVEL_DATA` 等环境变量。日志只包含数量、状态、SHA和错误码，不包含问题、答案或消息正文；敏感字段由公共日志组件自动脱敏。默认单文件1 MiB、保留3份轮转备份。",
            "",
            "调试顺序：data失败先核对配置和父manifest；validation失败查看质量门错误码；orchestrator失败查看最终状态和remediation。生产运行保持INFO，只有定位单个模块时临时启用DEBUG。",
            "",
            "## 完整性",
            "",
            f"- Dataset identity: `{report['manifest']['dataset_identity_sha256']}`",
            f"- Manifest SHA-256: `{report['manifest']['sha256']}`",
            f"- Train SHA-256: `{report['artifact_files']['train']['sha256']}`",
            f"- Holdout SHA-256: `{report['artifact_files']['holdout_eval']['sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_release(
    *,
    config_path: Path = DEFAULT_CONFIG,
    parent_manifest_path: Path = DEFAULT_PARENT_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_json_path: Path = DEFAULT_REPORT_JSON,
    report_md_path: Path = DEFAULT_REPORT_MD,
    base_checkpoint_path: Path = DEFAULT_BASE_CHECKPOINT,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
    token_manifest_path: Path = DEFAULT_TOKEN_MANIFEST,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config, parent_manifest, selected, verification = load_and_validate_config(
        config_path,
        parent_manifest_path,
        base_checkpoint_path=base_checkpoint_path,
        tokenizer_path=tokenizer_path,
        token_manifest_path=token_manifest_path,
    )
    records = build_records(config, selected)
    validation = validate_records(records)
    manifest = write_release(
        records,
        config=config,
        config_path=config_path,
        parent_manifest=parent_manifest,
        parent_manifest_path=parent_manifest_path,
        output_dir=output_dir,
        validation=validation,
    )
    manifest_path = output_dir / "manifest.json"
    report = _build_report(
        manifest,
        manifest_path=manifest_path,
        verification=verification,
    )
    atomic_write_json(report_json_path, report)
    atomic_write_text(report_md_path, _report_markdown(report))
    return manifest, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--parent-manifest", type=Path, default=DEFAULT_PARENT_MANIFEST)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--token-manifest", type=Path, default=DEFAULT_TOKEN_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--data-log-level", default="INFO")
    parser.add_argument("--validation-log-level", default="INFO")
    parser.add_argument("--orchestrator-log-level", default="INFO")
    parser.add_argument("--log-max-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--log-backups", type=int, default=3)
    parser.add_argument("--no-console-log", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_id = generate_run_id("sft-v7-1-canary-build")
    levels = resolve_module_log_levels(
        {
            "data": args.data_log_level,
            "validation": args.validation_log_level,
            "orchestrator": args.orchestrator_log_level,
        },
        env_prefix="GPT_CANARY_LOG_LEVEL",
    )
    loggers = configure_module_loggers(
        args.log_dir,
        run_id,
        levels,
        max_bytes=args.log_max_bytes,
        backup_count=args.log_backups,
        console=not args.no_console_log,
    )
    try:
        loggers["data"].info(
            "canary metadata load started",
            extra={
                "context": {
                    "source_kind": "reviewed_config_codex_ai_assisted_cross_check_with_catalog",
                    "requested_fact_count": 8,
                    "public_body_read": False,
                    "sealed_body_read": False,
                }
            },
        )
        manifest, report = build_release(
            config_path=args.config,
            parent_manifest_path=args.parent_manifest,
            output_dir=args.output_dir,
            report_json_path=args.report_json,
            report_md_path=args.report_md,
            base_checkpoint_path=args.base_checkpoint,
            tokenizer_path=args.tokenizer,
            token_manifest_path=args.token_manifest,
        )
        loggers["validation"].info(
            "canary quality gates passed",
            extra={
                "context": {
                    "fact_count": report["counts"]["facts"],
                    "train_count": report["counts"]["train"],
                    "holdout_eval_count": report["counts"]["holdout_eval"],
                    "passed_gate_count": sum(report["quality_gates"].values()),
                    "gate_count": len(report["quality_gates"]),
                }
            },
        )
        loggers["data"].info(
            "canary artifacts written",
            extra={
                "context": {
                    "dataset_identity_sha256": manifest["dataset_identity_sha256"],
                    "train_sha256": manifest["split_files"]["train"]["sha256"],
                    "holdout_sha256": manifest["split_files"]["holdout_eval"]["sha256"],
                }
            },
        )
        loggers["orchestrator"].info(
            "canary build complete",
            extra={"context": {"status": report["status"], "run_id": run_id}},
        )
        return 0
    except CanaryBuildError as error:
        loggers["orchestrator"].error(
            "canary build failed",
            extra={
                "context": {
                    "error_code": error.code,
                    "error_type": type(error).__name__,
                    "remediation": error.remediation,
                }
            },
        )
        return 1
    except Exception as error:
        loggers["orchestrator"].error(
            "canary build failed unexpectedly",
            extra={
                "context": {
                    "error_code": "UNEXPECTED_BUILD_FAILURE",
                    "error_type": type(error).__name__,
                    "remediation": "Run the module-level tests and inspect only the failing subsystem.",
                }
            },
        )
        return 1
    finally:
        close_module_loggers(loggers)


if __name__ == "__main__":
    raise SystemExit(main())
