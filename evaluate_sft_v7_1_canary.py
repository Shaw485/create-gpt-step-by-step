"""Generate and score all 80 M021 Canary questions.

``base`` mode records the frozen Step 5750 pre-SFT baseline. ``canary`` mode
validates a complete SFT v7.1 checkpoint and applies the fixed Stop/Go gates:
normalized exact answers >=95% on train and >=75% on unseen-question
development paraphrases, at least 15/16 development EOS stops, zero severe
repetitions and zero relation-
direction errors. Required terms remain a diagnostic, not the semantic gate.

Questions, references and generated answers are written only to the JSON/Markdown
report.  ``data``, ``generation``, ``validation``, ``checkpoint`` and
``orchestrator`` rotating JSONL logs contain counts, hashes, lengths and status
only. Each is independently configurable with ``--<module>-log-level`` or
``GPT_CANARY_LOG_LEVEL_<MODULE>``; use ``OFF`` to disable one module.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import unicodedata

import torch

from prepare_bpe_sft_v7 import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SPECIAL_TOKEN_IDS,
    EXPECTED_TOKENIZER_SHA256,
    REQUIRED_BASE_CHECKPOINT,
    load_and_validate_formal_tokenizer,
    read_jsonl,
)
from prepare_sft_v7_1_canary import (
    DEFAULT_DATASET_MANIFEST,
    DEFAULT_EVAL,
    DEFAULT_TOKENIZER,
    DEFAULT_TOKEN_MANIFEST,
    DEFAULT_TRAIN,
    load_and_validate_canary_manifest,
    validate_source_records,
)
from sample_sft_v7 import (
    build_conversation_prompt_ids,
    generate_responses_length_bucketed,
    validate_pretrain_baseline_checkpoint,
)
from train_pretrain_v4 import load_config
from train_sft_v4 import build_model, select_device
from train_sft_v7 import BASE_CONFIG_CANONICAL_SHA256, validate_frozen_config
from train_sft_v7_1_canary import (
    DEFAULT_BASE_CONFIG,
    DEFAULT_DATA,
    DEFAULT_INIT_CHECKPOINT,
    DEFAULT_TRAINING_CONFIG,
    EXPECTED_EVAL_COUNT,
    EXPECTED_STAGE,
    EXPECTED_TRAIN_COUNT,
    TRAINING_SIGNATURE_SCHEMA,
    load_canary_tensor_payload,
    load_training_config,
    validate_canary_tensor_payload,
)
from training_runtime import (
    atomic_write_json,
    atomic_write_text,
    canonical_json_sha256,
    close_module_loggers,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
    load_checkpoint,
    resolve_module_log_levels,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_CANARY_CHECKPOINT = Path("runs/sft_v7_1_canary/latest.pt")
DEFAULT_EFFECTIVE_CONFIG = Path("runs/sft_v7_1_canary/effective_config.json")
DEFAULT_BASE_REPORT_JSON = Path(
    "reports/milestones/021_sft_v7_1_canary/canary_baseline_generation.json"
)
DEFAULT_BASE_REPORT_MD = Path(
    "reports/milestones/021_sft_v7_1_canary/canary_baseline_generation.md"
)
DEFAULT_CANARY_REPORT_JSON = Path(
    "reports/milestones/021_sft_v7_1_canary/canary_generation_eval.json"
)
DEFAULT_CANARY_REPORT_MD = Path(
    "reports/milestones/021_sft_v7_1_canary/canary_generation_eval.md"
)
DEFAULT_LOG_DIR = Path("logs/sft_v7_1_canary_evaluate")
REPORT_SCHEMA = "sft-v7.1-canary-generation-evaluation/v1"
_HEX_SHA = re.compile(r"[0-9a-f]{64}")


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return f"artifact://canary/{resolved.name}"


def _messages_for_question(record: Mapping[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
    ):
        raise ValueError("Canary source conversation is invalid")
    return [{"role": "user", "content": str(messages[0]["content"])}]


def severe_repetition(text: str) -> bool:
    """Flag conspicuous decoding loops, not ordinary repeated entity names."""

    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False
    if re.search(r"(.)\1{5,}", compact):
        return True
    # A phrase of 2-12 characters repeated at least three consecutive times is
    # a model loop.  Requiring consecutive repeats avoids vocabulary statistics.
    return bool(re.search(r"(.{2,12})\1{2,}", compact))


def normalize_answer(text: str) -> str:
    """Normalize presentation-only differences, while preserving every word."""

    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    ).casefold()


_DIRECTION_ERRORS: dict[str, tuple[str, ...]] = {
    "xiaoyan_identity": (
        "萧炎是萧战的父亲",
        "萧战是萧炎的儿子",
    ),
    "xiaozhan_identity": (
        "萧炎是萧战的父亲",
        "萧战是萧炎的儿子",
    ),
    "yaochen_identity": (
        "萧炎是药尘的老师",
        "药尘是萧炎的弟子",
        "药尘是萧炎的学生",
    ),
    "yaolao_teacher": (
        "萧炎是药老的老师",
        "药老是萧炎的弟子",
        "药老是萧炎的学生",
    ),
    "yaolao_yaochen_alias": (
        "药老是药尘的老师",
        "药尘是药老的老师",
    ),
}


def self_relation_error_reasons(fact_id: str, text: str) -> list[str]:
    """Detect generic self-relations and reviewed direction reversals."""

    relation_text = (
        text.replace("也是", "是")
        .replace("却是", "是")
        .replace("其实是", "是")
        .replace("并且是", "是")
    )
    reasons = [
        f"self_relation:{entity}"
        for entity in ("萧炎", "萧战", "药老", "药尘", "焚决", "异火", "云岚宗")
        if f"{entity}是{entity}" in text
    ]
    reasons.extend(
        f"direction_reversal:{phrase}"
        for phrase in _DIRECTION_ERRORS.get(fact_id, ())
        if phrase in relation_text
    )
    return reasons


def score_generation(
    record: Mapping[str, Any], generated: Mapping[str, Any]
) -> dict[str, Any]:
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("Canary evaluation metadata is missing")
    required_terms = [str(term) for term in evaluation.get("required_terms", [])]
    forbidden_terms = [str(term) for term in evaluation.get("forbidden_terms", [])]
    if not required_terms or evaluation.get("metric") != "required_terms_all":
        raise ValueError("Canary required-term metric changed")
    text = str(generated.get("generated_text", ""))
    required_present = {term: term in text for term in required_terms}
    forbidden_present = {term: term in text for term in forbidden_terms}
    relation_reasons = self_relation_error_reasons(str(record.get("fact_id", "")), text)
    repetition = severe_repetition(text)
    required_all = all(required_present.values())
    forbidden_clear = not any(forbidden_present.values())
    normalized_exact = normalize_answer(text) == normalize_answer(
        str(record.get("answer", ""))
    )
    return {
        "normalized_exact_answer": normalized_exact,
        "exact_answer_pass": normalized_exact and forbidden_clear,
        "required_terms_all": required_all,
        "forbidden_terms_clear": forbidden_clear,
        "keypoint_pass": required_all and forbidden_clear,
        "required_terms_present": required_present,
        "forbidden_terms_present": forbidden_present,
        "stopped_on_eos": bool(generated.get("stopped_on_eos", False)),
        "truncated": bool(generated.get("truncated", False)),
        "severe_repetition": repetition,
        "self_relation_error": bool(relation_reasons),
        "self_relation_error_reasons": relation_reasons,
    }


def summarize_split(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(samples)
    if count <= 0:
        raise ValueError("Canary evaluation split is empty")
    counters = Counter()
    for sample in samples:
        score = sample["score"]
        for key in (
            "normalized_exact_answer",
            "exact_answer_pass",
            "required_terms_all",
            "forbidden_terms_clear",
            "keypoint_pass",
            "stopped_on_eos",
            "truncated",
            "severe_repetition",
            "self_relation_error",
        ):
            counters[key] += int(bool(score[key]))
    return {
        "count": count,
        "normalized_exact_answer_count": counters["normalized_exact_answer"],
        "normalized_exact_answer_rate": counters["normalized_exact_answer"] / count,
        "exact_answer_pass_count": counters["exact_answer_pass"],
        "exact_answer_rate": counters["exact_answer_pass"] / count,
        "required_terms_all_count": counters["required_terms_all"],
        "required_terms_all_rate": counters["required_terms_all"] / count,
        "forbidden_terms_clear_count": counters["forbidden_terms_clear"],
        "forbidden_terms_clear_rate": counters["forbidden_terms_clear"] / count,
        "keypoint_pass_count": counters["keypoint_pass"],
        "keypoint_pass_rate": counters["keypoint_pass"] / count,
        "eos_count": counters["stopped_on_eos"],
        "eos_rate": counters["stopped_on_eos"] / count,
        "truncated_count": counters["truncated"],
        "severe_repetition_count": counters["severe_repetition"],
        "self_relation_error_count": counters["self_relation_error"],
    }


def summarize_by_fact(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Expose every fact cluster so global averages cannot hide a failure."""

    fact_ids = sorted({str(sample["fact_id"]) for sample in samples})
    if len(fact_ids) != 8:
        raise ValueError("Canary evaluation must contain eight catalog fact IDs")
    result: dict[str, dict[str, Any]] = {}
    for fact_id in fact_ids:
        fact_samples = [sample for sample in samples if sample["fact_id"] == fact_id]
        train = [sample for sample in fact_samples if sample["split"] == "train"]
        holdout = [
            sample for sample in fact_samples if sample["split"] == "holdout_eval"
        ]
        if len(train) != 8 or len(holdout) != 2:
            raise ValueError("Canary per-fact train/holdout quota changed")
        result[fact_id] = {
            "train_count": 8,
            "train_exact_count": sum(
                int(sample["score"]["exact_answer_pass"]) for sample in train
            ),
            "holdout_count": 2,
            "holdout_exact_count": sum(
                int(sample["score"]["exact_answer_pass"]) for sample in holdout
            ),
            "train_required_terms_all_count_diagnostic": sum(
                int(sample["score"]["required_terms_all"]) for sample in train
            ),
            "holdout_required_terms_all_count_diagnostic": sum(
                int(sample["score"]["required_terms_all"]) for sample in holdout
            ),
        }
    return result


def evaluate_gates(
    split_metrics: Mapping[str, Mapping[str, Any]],
    gate_config: Mapping[str, Any],
    fact_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    train = split_metrics["train"]
    holdout = split_metrics["holdout_eval"]
    repetition_count = int(train["severe_repetition_count"]) + int(
        holdout["severe_repetition_count"]
    )
    relation_count = int(train["self_relation_error_count"]) + int(
        holdout["self_relation_error_count"]
    )
    train_fact_failures = [
        fact_id
        for fact_id, metrics in fact_metrics.items()
        if int(metrics["train_exact_count"])
        < int(gate_config["per_fact_train_exact_min"])
    ]
    holdout_fact_failures = [
        fact_id
        for fact_id, metrics in fact_metrics.items()
        if int(metrics["holdout_exact_count"])
        < int(gate_config["per_fact_holdout_exact_min"])
    ]
    checks = {
        "train_exact_answer_rate": float(train["exact_answer_rate"])
        >= float(gate_config["train_exact_answer_rate_min"]),
        "holdout_exact_answer_rate": float(holdout["exact_answer_rate"])
        >= float(gate_config["holdout_exact_answer_rate_min"]),
        "holdout_eos_count": int(holdout["eos_count"])
        >= int(gate_config["holdout_eos_count_min"]),
        "per_fact_train_exact_min": not train_fact_failures,
        "per_fact_holdout_exact_min": not holdout_fact_failures,
        "severe_repetition_count": repetition_count
        <= int(gate_config["severe_repetition_count_max"]),
        "self_relation_error_count": relation_count
        <= int(gate_config["self_relation_error_count_max"]),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "observed": {
            "train_exact_answer_rate": train["exact_answer_rate"],
            "holdout_exact_answer_rate": holdout["exact_answer_rate"],
            "train_required_terms_all_rate_diagnostic": train[
                "required_terms_all_rate"
            ],
            "holdout_required_terms_all_rate_diagnostic": holdout[
                "required_terms_all_rate"
            ],
            "holdout_eos_count": holdout["eos_count"],
            "per_fact_train_failures": train_fact_failures,
            "per_fact_holdout_failures": holdout_fact_failures,
            "severe_repetition_count": repetition_count,
            "self_relation_error_count": relation_count,
        },
        "thresholds": dict(gate_config),
    }


def load_and_validate_effective_config(path: Path) -> dict[str, Any]:
    try:
        effective = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Canary effective config cannot be parsed") from error
    if not isinstance(effective, dict) or effective.get("schema_version") != TRAINING_SIGNATURE_SCHEMA:
        raise ValueError("Canary effective config schema changed")
    signature = {
        key: effective[key]
        for key in ("schema_version", "model", "provenance", "training", "schedule")
    }
    expected_sha = canonical_json_sha256(signature)
    if effective.get("signature_sha256") != expected_sha:
        raise ValueError("Canary effective config signature is invalid")
    return effective


def validate_canary_checkpoint_provenance(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    *,
    tensor_path: Path,
    tensor_payload: Mapping[str, Any],
    effective_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a trained checkpoint to this exact 64/16 capacity experiment."""

    if checkpoint.get("schema_version") != "training-checkpoint/v1":
        raise ValueError("Canary checkpoint schema changed")
    if checkpoint.get("config_sha256") != effective_config["signature_sha256"]:
        raise ValueError("Canary checkpoint and effective config signatures differ")
    extra = checkpoint.get("extra")
    if not isinstance(extra, Mapping):
        raise ValueError("Canary checkpoint provenance is missing")
    expected = {
        "stage": EXPECTED_STAGE,
        "base_checkpoint_path": str(DEFAULT_INIT_CHECKPOINT),
        "base_checkpoint_sha256": REQUIRED_BASE_CHECKPOINT["sha256"],
        "base_checkpoint_step": REQUIRED_BASE_CHECKPOINT["step"],
        "base_config_canonical_sha256": BASE_CONFIG_CANONICAL_SHA256,
        "base_token_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "canary_tensor_sha256": file_sha256(tensor_path),
        "canary_dataset_manifest_sha256": tensor_payload[
            "canary_dataset_manifest_sha256"
        ],
        "canary_dataset_identity_sha256": tensor_payload[
            "canary_dataset_identity_sha256"
        ],
        "optimization_train_records": EXPECTED_TRAIN_COUNT,
        "optimization_holdout_records": 0,
        "holdout_records_consumed": 0,
        "teacher_loss_holdout_records": EXPECTED_EVAL_COUNT,
        "public_records_consumed": 0,
        "sealed_records_consumed": 0,
    }
    for key, value in expected.items():
        if extra.get(key) != value:
            raise ValueError(f"Canary checkpoint provenance mismatch: {key}")
    development_required = {
        "development_unseen_wording_records": EXPECTED_EVAL_COUNT,
        "development_records_consumed_for_teacher_loss": EXPECTED_EVAL_COUNT,
        "development_records_used_for_checkpoint_selection": EXPECTED_EVAL_COUNT,
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
        raise ValueError("Canary checkpoint development provenance is incomplete")
    for key in set(development_required).intersection(extra):
        if extra.get(key) != development_required[key]:
            raise ValueError(f"Canary checkpoint provenance mismatch: {key}")
    if "development_optimizer_records" in extra and extra.get(
        "development_optimizer_records"
    ) != 0:
        raise ValueError(
            "Canary checkpoint provenance mismatch: development_optimizer_records"
        )
    if "development_records_used_for_optimization" in extra and extra.get(
        "development_records_used_for_optimization"
    ) != 0:
        raise ValueError(
            "Canary checkpoint provenance mismatch: "
            "development_records_used_for_optimization"
        )
    if str(extra.get("canary_tensor_path", "")) != str(DEFAULT_DATA):
        raise ValueError("Canary checkpoint tensor path changed")
    summary = extra.get("payload_summary")
    if not isinstance(summary, Mapping) or summary.get("split_counts") != {
        "train": 64,
        "holdout_eval": 16,
    }:
        raise ValueError("Canary checkpoint split counts changed")
    if not isinstance(extra.get("sampler_state"), Mapping):
        raise ValueError("Canary checkpoint sampler state is missing")
    if int(checkpoint.get("step", -1)) <= 0:
        raise ValueError("Canary checkpoint has no optimizer updates")
    if not _HEX_SHA.fullmatch(file_sha256(checkpoint_path)):
        raise ValueError("Canary checkpoint SHA is invalid")
    return {
        "checkpoint_mode": "canary",
        "step": int(checkpoint["step"]),
        "base_checkpoint_sha256": str(extra["base_checkpoint_sha256"]),
        "training_tensor_sha256": str(extra["canary_tensor_sha256"]),
        "training_split_counts": dict(summary["split_counts"]),
        **development_required,
        "development_records_used_for_optimization": 0,
        "development_optimizer_records": 0,
        "development_split_role": "unseen_question_development_selection",
        "development_provenance_inferred_from_legacy_checkpoint": not bool(
            present_development_fields
        ),
    }


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def render_markdown(report: Mapping[str, Any]) -> str:
    train = report["split_metrics"]["train"]
    holdout = report["split_metrics"]["holdout_eval"]
    rows = [
        "# M021 SFT v7.1 Canary 生成评估",
        "",
        f"模式：`{report['checkpoint_mode']}`；checkpoint step：`{report['checkpoint_step']}`；"
        f"质量门：**{report['gates']['status'].upper()}**",
        "",
        "| 指标 | Train (64) | 未见问法 Dev/Selection (16) |",
        "|---|---:|---:|",
        f"| 严格答案匹配（门控） | {train['exact_answer_pass_count']} ({train['exact_answer_rate']:.1%}) | "
        f"{holdout['exact_answer_pass_count']} ({holdout['exact_answer_rate']:.1%}) |",
        f"| 关键点通过 | {train['keypoint_pass_count']} ({train['keypoint_pass_rate']:.1%}) | "
        f"{holdout['keypoint_pass_count']} ({holdout['keypoint_pass_rate']:.1%}) |",
        f"| EOS 停止 | {train['eos_count']} ({train['eos_rate']:.1%}) | "
        f"{holdout['eos_count']} ({holdout['eos_rate']:.1%}) |",
        f"| 严重重复 | {train['severe_repetition_count']} | {holdout['severe_repetition_count']} |",
        f"| 关系方向错误 | {train['self_relation_error_count']} | {holdout['self_relation_error_count']} |",
        "",
        "## 质量门",
        "",
        "| Gate | 结果 |",
        "|---|---|",
    ]
    for name, passed in report["gates"]["checks"].items():
        rows.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    rows.extend(
        [
            "",
            "## 每个 catalog fact_id",
            "",
            "| fact_id | Train exact | Holdout exact | Train required terms（诊断） | Holdout required terms（诊断） |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for fact_id, metrics in report["fact_metrics"].items():
        rows.append(
            f"| `{fact_id}` | {metrics['train_exact_count']}/8 | "
            f"{metrics['holdout_exact_count']}/2 | "
            f"{metrics['train_required_terms_all_count_diagnostic']}/8 | "
            f"{metrics['holdout_required_terms_all_count_diagnostic']}/2 |"
        )
    rows.extend(
        [
            "",
            "## 全部样本",
            "",
            "| Split | fact_id | 问题 | 参考答案 | 生成答案 | Exact | Keypoint | EOS | 重复 | 关系错误 |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for sample in report["samples"]:
        score = sample["score"]
        rows.append(
            f"| {sample['split']} | `{sample['fact_id']}` | "
            f"{_escape_markdown(sample['question'])} | "
            f"{_escape_markdown(sample['expected_answer'])} | "
            f"{_escape_markdown(sample['generated_answer'])} | "
            f"{'PASS' if score['exact_answer_pass'] else 'FAIL'} | "
            f"{'PASS' if score['keypoint_pass'] else 'FAIL'} | "
            f"{'是' if score['stopped_on_eos'] else '否'} | "
            f"{'是' if score['severe_repetition'] else '否'} | "
            f"{'是' if score['self_relation_error'] else '否'} |"
        )
    rows.extend(
        [
            "",
            "## 日志与复现",
            "",
            "data、generation、validation、checkpoint、orchestrator 分别写入轮转 JSONL。"
            "用 `--generation-log-level DEBUG` 或 `GPT_CANARY_LOG_LEVEL_GENERATION=DEBUG` "
            "只调试生成模块；日志不保存问题、答案、生成文本或 Token ID。生产保持 INFO，"
            "默认单文件 1 MiB、保留 3 份备份。完整文本仅保存在本报告和对应 JSON 中。",
            "",
        ]
    )
    return "\n".join(rows)


def _add_log_arguments(parser: argparse.ArgumentParser, modules: Sequence[str]) -> None:
    for module in modules:
        parser.add_argument(
            f"--{module.replace('_', '-')}-log-level",
            choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "OFF"),
            default=None,
            help=f"independently set the {module} rotating JSONL log level",
        )
    parser.add_argument("--no-console-log", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-mode", choices=("base", "canary"), default="canary")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--effective-config", type=Path, default=DEFAULT_EFFECTIVE_CONFIG)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--holdout-eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--token-manifest", type=Path, default=DEFAULT_TOKEN_MANIFEST)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--generation-batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--disable-kv-cache", action="store_true")
    _add_log_arguments(
        parser, ("data", "generation", "validation", "checkpoint", "orchestrator")
    )
    return parser.parse_args(argv)


def _log_levels(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    modules: Sequence[str],
) -> dict[str, str]:
    configured = dict(config["logging"].get("module_levels", {}))
    levels = resolve_module_log_levels(
        {module: str(configured.get(module, "INFO")) for module in modules},
        env_prefix="GPT_CANARY_LOG_LEVEL",
    )
    for module in modules:
        override = getattr(args, f"{module}_log_level")
        if override is not None:
            levels[module] = override
    return levels


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    training_config = load_training_config(args.training_config)
    generation_config = dict(training_config["generation"])
    for name in ("max_new_tokens", "temperature", "top_k", "generation_batch_size", "seed"):
        value = getattr(args, name)
        if value is not None:
            generation_config[name] = value
    if (
        int(generation_config["max_new_tokens"]) <= 0
        or float(generation_config["temperature"]) <= 0
        or int(generation_config["top_k"]) < 0
        or int(generation_config["generation_batch_size"]) <= 0
    ):
        raise ValueError("Canary generation configuration is invalid")
    checkpoint_path = args.checkpoint or (
        DEFAULT_INIT_CHECKPOINT if args.checkpoint_mode == "base" else DEFAULT_CANARY_CHECKPOINT
    )
    output_json = args.output_json or (
        DEFAULT_BASE_REPORT_JSON
        if args.checkpoint_mode == "base"
        else DEFAULT_CANARY_REPORT_JSON
    )
    output_md = args.output_md or (
        DEFAULT_BASE_REPORT_MD if args.checkpoint_mode == "base" else DEFAULT_CANARY_REPORT_MD
    )
    base_config = load_config(args.base_config)
    validate_frozen_config(base_config)
    modules = ("data", "generation", "validation", "checkpoint", "orchestrator")
    run_id = generate_run_id("sft-v7-1-canary-eval")
    loggers = configure_module_loggers(
        args.log_dir,
        run_id,
        _log_levels(args, training_config, modules),
        max_bytes=int(training_config["logging"]["max_bytes"]),
        backup_count=int(training_config["logging"]["backup_count"]),
        console=(
            bool(training_config["logging"]["console"])
            and not args.no_console_log
        ),
    )
    try:
        manifest_identity = load_and_validate_canary_manifest(
            args.dataset_manifest,
            {"train": args.train, "holdout_eval": args.holdout_eval},
        )
        train_records = read_jsonl(args.train, "train")
        eval_records = read_jsonl(args.holdout_eval, "holdout_eval")
        source_summary = validate_source_records(train_records, eval_records)
        tokenizer, special_ids, tokenizer_identity = load_and_validate_formal_tokenizer(
            args.tokenizer, args.token_manifest
        )
        loggers["data"].info(
            "Canary evaluation sources validated",
            extra={
                "context": {
                    "train_count": len(train_records),
                    "holdout_eval_count": len(eval_records),
                    "fact_count": source_summary["fact_count"],
                    "dataset_manifest_sha256": manifest_identity[
                        "canary_manifest_sha256"
                    ],
                    "tokenizer_sha256": tokenizer_identity["tokenizer_sha256"],
                }
            },
        )
        device = select_device(args.device)
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        tensor_sha: str | None = None
        effective_sha: str | None = None
        if args.checkpoint_mode == "base":
            provenance = validate_pretrain_baseline_checkpoint(
                checkpoint, checkpoint_path
            )
            provenance["checkpoint_mode"] = "base"
        else:
            tensor_payload = load_canary_tensor_payload(args.data)
            validate_canary_tensor_payload(tensor_payload, int(base_config["model"]["block_size"]))
            if tensor_payload["canary_dataset_manifest_sha256"] != manifest_identity[
                "canary_manifest_sha256"
            ]:
                raise ValueError("Canary generation sources and training tensor differ")
            effective = load_and_validate_effective_config(args.effective_config)
            provenance = validate_canary_checkpoint_provenance(
                checkpoint,
                checkpoint_path,
                tensor_path=args.data,
                tensor_payload=tensor_payload,
                effective_config=effective,
            )
            tensor_sha = file_sha256(args.data)
            effective_sha = file_sha256(args.effective_config)
        model = build_model(base_config, tokenizer.vocab_size).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        checkpoint_sha = file_sha256(checkpoint_path)
        loggers["checkpoint"].info(
            "Canary evaluation checkpoint validated",
            extra={
                "context": {
                    "mode": args.checkpoint_mode,
                    "step": int(checkpoint["step"]),
                    "checkpoint_sha256": checkpoint_sha,
                    "base_checkpoint_sha256": provenance["base_checkpoint_sha256"],
                    "device": str(device),
                }
            },
        )

        all_records = list(train_records) + list(eval_records)
        prompt_ids = [
            build_conversation_prompt_ids(
                tokenizer, _messages_for_question(record), special_ids
            )
            for record in all_records
        ]
        seeds = [int(generation_config["seed"]) + index for index in range(len(all_records))]
        generated = generate_responses_length_bucketed(
            model,
            prompt_ids,
            tokenizer,
            special_ids,
            max_new_tokens=int(generation_config["max_new_tokens"]),
            temperature=float(generation_config["temperature"]),
            top_k=int(generation_config["top_k"]),
            seeds=seeds,
            device=device,
            generation_batch_size=int(generation_config["generation_batch_size"]),
            use_kv_cache=not args.disable_kv_cache,
        )
        samples: list[dict[str, Any]] = []
        for record, generation in zip(all_records, generated):
            score = score_generation(record, generation)
            samples.append(
                {
                    "id": record["id"],
                    "split": record["split"],
                    "fact_id": record["fact_id"],
                    "prompt_role": record["prompt_role"],
                    "question": record["question"],
                    "expected_answer": record["answer"],
                    "generated_answer": generation["generated_text"],
                    "generated_tokens": generation["generated_tokens"],
                    "score": score,
                }
            )
        loggers["generation"].info(
            "Canary batch generation complete",
            extra={
                "context": {
                    "sample_count": len(samples),
                    "maximum_prompt_tokens": max(len(ids) for ids in prompt_ids),
                    "total_generated_tokens": sum(
                        int(sample["generated_tokens"]) for sample in samples
                    ),
                    "eos_count": sum(
                        int(sample["score"]["stopped_on_eos"]) for sample in samples
                    ),
                    "generation_batch_size": int(
                        generation_config["generation_batch_size"]
                    ),
                }
            },
        )
        split_metrics = {
            split: summarize_split(
                [sample for sample in samples if sample["split"] == split]
            )
            for split in ("train", "holdout_eval")
        }
        fact_metrics = summarize_by_fact(samples)
        gates = evaluate_gates(
            split_metrics, training_config["gates"], fact_metrics
        )
        decision = (
            "baseline_recorded"
            if args.checkpoint_mode == "base"
            else ("GO_capacity_demonstrated" if gates["status"] == "pass" else "STOP_debug_pipeline_or_capacity")
        )
        report = {
            "report_schema_version": REPORT_SCHEMA,
            "status": "complete",
            "decision": decision,
            "run_id": run_id,
            "checkpoint_mode": args.checkpoint_mode,
            "checkpoint_path": _portable(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_step": int(checkpoint["step"]),
            "checkpoint_provenance": provenance,
            "base_config_path": _portable(args.base_config),
            "base_config_canonical_sha256": BASE_CONFIG_CANONICAL_SHA256,
            "training_tensor_path": _portable(args.data) if tensor_sha else None,
            "training_tensor_sha256": tensor_sha,
            "effective_config_path": _portable(args.effective_config)
            if effective_sha
            else None,
            "effective_config_sha256": effective_sha,
            "canary_dataset_manifest_path": _portable(args.dataset_manifest),
            "canary_dataset_manifest_sha256": manifest_identity[
                "canary_manifest_sha256"
            ],
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
            "bpe_token_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "generation": {
                **generation_config,
                "kv_cache": not args.disable_kv_cache,
                "masked_special_tokens": [
                    "<UNK>",
                    "<BOS>",
                    "<USER>",
                    "<ASSISTANT>",
                    "<PAD>",
                ],
            },
            "split_metrics": split_metrics,
            "fact_metrics": fact_metrics,
            "gates": gates,
            "samples": samples,
            "source_access": {
                "canary_train_records": 64,
                "canary_development_selection_records": 16,
                "canary_development_optimizer_records": 0,
                "formal_v7_diagnostic_body_records": 0,
                "formal_v7_blind_body_records": 0,
            },
            "logging": {
                "directory": _portable(args.log_dir),
                "modules": list(modules),
                "format": "rotating JSONL with UTC timestamp and run_id",
                "record_bodies_logged": False,
                "generated_text_logged": False,
                "token_ids_logged": False,
                "max_bytes": int(training_config["logging"]["max_bytes"]),
                "backup_count": int(training_config["logging"]["backup_count"]),
            },
        }
        atomic_write_json(output_json, report)
        atomic_write_text(output_md, render_markdown(report))
        loggers["validation"].info(
            "Canary generation gates evaluated",
            extra={
                "context": {
                    "gate_status": gates["status"],
                    "decision": decision,
                    **gates["observed"],
                }
            },
        )
        loggers["orchestrator"].info(
            "Canary evaluation complete",
            extra={
                "context": {
                    "mode": args.checkpoint_mode,
                    "decision": decision,
                    "output_json": _portable(output_json),
                    "output_markdown": _portable(output_md),
                }
            },
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "decision": decision,
                    "checkpoint_step": report["checkpoint_step"],
                    "split_metrics": split_metrics,
                    "gates": gates,
                    "report": _portable(output_json),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as error:
        loggers["validation"].error(
            "Canary generation evaluation failed",
            extra={
                "context": {
                    "error_type": type(error).__name__,
                    "error_code": getattr(error, "code", "unexpected_failure"),
                    "mode": args.checkpoint_mode,
                }
            },
        )
        raise
    finally:
        close_module_loggers(loggers)


if __name__ == "__main__":
    raise SystemExit(main())
