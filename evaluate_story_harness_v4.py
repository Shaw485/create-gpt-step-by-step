"""Compare fixed novel-continuation samples across v4 pretraining checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any

from training_runtime import (
    atomic_write_json,
    atomic_write_text,
    configure_module_loggers,
    generate_run_id,
)


HAN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
AUTOMATIC_GATES = {
    "minimum_mean_characters": 90,
    "minimum_mean_han_ratio": 0.60,
    "maximum_mean_han_ratio": 0.95,
    "maximum_four_gram_repetition": 0.08,
    "maximum_character_run": 5,
    "maximum_train_overlap": 30,
}


def ngram_repetition(text: str, size: int = 4) -> float:
    """Return the fraction of repeated character n-gram occurrences."""
    if len(text) < size:
        return 0.0
    ngrams = [text[index : index + size] for index in range(len(text) - size + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def longest_character_run(text: str) -> int:
    """Measure obvious degeneration such as repeated '试试试试'."""
    if not text:
        return 0
    longest = current = 1
    for previous, current_character in zip(text, text[1:]):
        current = current + 1 if current_character == previous else 1
        longest = max(longest, current)
    return longest


def longest_corpus_overlap(text: str, corpus: str, minimum: int = 8) -> int:
    """Find the longest generated substring copied verbatim from the train corpus."""
    compact = text.strip()
    if len(compact) < minimum:
        return 0

    def contains_overlap(size: int) -> bool:
        return any(
            compact[start : start + size] in corpus
            for start in range(0, len(compact) - size + 1)
        )

    best = 0
    lower = minimum
    upper = len(compact)
    while lower <= upper:
        middle = (lower + upper) // 2
        if contains_overlap(middle):
            best = middle
            lower = middle + 1
        else:
            upper = middle - 1
    return best


def sample_metrics(text: str, corpus: str) -> dict[str, float | int]:
    visible = text.replace("\n", "")
    return {
        "characters": len(text),
        "han_ratio": (len(HAN_PATTERN.findall(visible)) / len(visible) if visible else 0.0),
        "four_gram_repetition": ngram_repetition(visible, 4),
        "longest_character_run": longest_character_run(visible),
        "longest_train_overlap": longest_corpus_overlap(visible, corpus),
        "paragraphs": len([part for part in text.split("\n\n") if part.strip()]),
    }


def apply_automatic_gates(summary: dict[str, Any], prompt_count: int) -> dict[str, Any]:
    """Apply hard safety/degeneration gates without inventing a quality score."""
    checks = {
        "all_prompts_present": summary["sample_count"] == prompt_count,
        "generation_length": summary["mean_characters"] >= AUTOMATIC_GATES[
            "minimum_mean_characters"
        ],
        "han_ratio": AUTOMATIC_GATES["minimum_mean_han_ratio"]
        <= summary["mean_han_ratio"]
        <= AUTOMATIC_GATES["maximum_mean_han_ratio"],
        "four_gram_repetition": summary["mean_four_gram_repetition"]
        <= AUTOMATIC_GATES["maximum_four_gram_repetition"],
        "character_run": summary["maximum_character_run"]
        <= AUTOMATIC_GATES["maximum_character_run"],
        "train_overlap": summary["maximum_train_overlap"]
        <= AUTOMATIC_GATES["maximum_train_overlap"],
    }
    return {
        "status": "PASS" if all(checks.values()) else "REVIEW",
        "checks": checks,
    }


def _load_records(
    baseline_path: Path,
    history_path: Path,
    selected_evaluation_path: Path | None,
    allowed_prompts: set[str],
) -> list[dict[str, Any]]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    records = [
        {"step": int(baseline["checkpoint_step"]), **sample}
        for sample in baseline["samples"]
        if sample["prompt"] in allowed_prompts
    ]
    history = json.loads(history_path.read_text(encoding="utf-8"))
    for milestone in history.get("history", []):
        for sample in milestone["samples"]:
            if sample["prompt"] in allowed_prompts:
                records.append({"step": int(milestone["step"]), **sample})
    # The independently reloaded selected checkpoint is the release candidate.
    # When it shares a step with a periodic training sample, prefer this sample.
    if selected_evaluation_path and selected_evaluation_path.is_file():
        selected = json.loads(selected_evaluation_path.read_text(encoding="utf-8"))
        records.extend(
            {"step": int(selected["checkpoint_step"]), **sample}
            for sample in selected["samples"]
            if sample["prompt"] in allowed_prompts
        )
    deduplicated = {
        (record["step"], record["prompt"]): record for record in records
    }
    return [deduplicated[key] for key in sorted(deduplicated)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("runs/pretrain_v4_m4_continue6000/story_baseline_step2600.json"),
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("runs/pretrain_v4_m4_continue6000/samples.json"),
    )
    parser.add_argument(
        "--selected-evaluation",
        type=Path,
        default=Path(
            "runs/pretrain_v4_m4_continue6000/selected_model_evaluation.json"
        ),
    )
    parser.add_argument("--corpus", type=Path, default=Path("data/cloud_v4/train.txt"))
    parser.add_argument("--prompts", type=Path, default=Path("data/story_prompt5_eval.txt"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/story_harness_v4")
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = generate_run_id("story-harness-v4")
    loggers = configure_module_loggers(
        args.output_dir / "logs",
        run_id,
        {"validation": "INFO", "orchestrator": "INFO"},
        max_bytes=1_048_576,
        backup_count=3,
        console=True,
    )
    prompts = {
        line.strip()
        for line in args.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    corpus = args.corpus.read_text(encoding="utf-8")
    records = _load_records(
        args.baseline,
        args.history,
        args.selected_evaluation,
        prompts,
    )
    if not records:
        raise ValueError("no fixed story-prompt samples were found")

    measured = []
    for record in records:
        measured.append({**record, **sample_metrics(record["continuation"], corpus)})
    steps = sorted({record["step"] for record in measured})
    summary = []
    for step in steps:
        rows = [record for record in measured if record["step"] == step]
        entry = {
            "step": step,
            "sample_count": len(rows),
            "mean_characters": mean(row["characters"] for row in rows),
            "mean_han_ratio": mean(row["han_ratio"] for row in rows),
            "mean_four_gram_repetition": mean(
                row["four_gram_repetition"] for row in rows
            ),
            "maximum_character_run": max(
                row["longest_character_run"] for row in rows
            ),
            "maximum_train_overlap": max(
                row["longest_train_overlap"] for row in rows
            ),
        }
        entry["automatic_gates"] = apply_automatic_gates(entry, len(prompts))
        summary.append(entry)

    report = {
        "schema_version": "story-evaluation-harness-v4/v1",
        "run_id": run_id,
        "protocol": {
            "prompt_count": len(prompts),
            "automatic_metrics_are_diagnostic_not_semantic_quality_scores": True,
            "manual_dimensions": ["fluency", "coherence", "prompt_relevance"],
            "automatic_gates": AUTOMATIC_GATES,
            "selection_policy": (
                "validation loss shortlists checkpoints; automatic gates can veto; "
                "manual semantic review chooses among passing candidates"
            ),
        },
        "summary": summary,
        "samples": measured,
    }
    atomic_write_json(args.output_dir / "story_harness_report.json", report)

    summary_csv = args.output_dir / "story_harness_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

    table = [
        "# v4 小说续写固定评测",
        "",
        "自动指标只能发现重复、字符异常和疑似原文复现，不能代替人工判断语义。",
        "",
        "| Step | 开头 | 续写 | 4-gram重复率 | 最长单字连写 | 最长训练集重合 |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for record in measured:
        prompt = record["prompt"].replace("|", "\\|")
        continuation = record["continuation"].replace("\n", "↵").replace("|", "\\|")
        table.append(
            f"| {record['step']} | {prompt} | {continuation} | "
            f"{record['four_gram_repetition']:.3f} | "
            f"{record['longest_character_run']} | {record['longest_train_overlap']} |"
        )
    atomic_write_text(args.output_dir / "story_harness_samples.md", "\n".join(table) + "\n")

    manual_path = args.output_dir / "manual_review_template.csv"
    with manual_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "step",
            "prompt",
            "fluency_1_to_5",
            "coherence_1_to_5",
            "prompt_relevance_1_to_5",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in measured:
            writer.writerow({"step": record["step"], "prompt": record["prompt"]})

    loggers["validation"].info(
        "fixed story samples measured",
        extra={"context": {"steps": steps, "sample_count": len(measured)}},
    )
    loggers["orchestrator"].info(
        "story evaluation harness complete",
        extra={"context": {"output_dir": str(args.output_dir)}},
    )
    print(json.dumps({"steps": steps, "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
