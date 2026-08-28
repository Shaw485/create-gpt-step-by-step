"""Compare fixed novel-continuation samples across v4 pretraining checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from story_quality import (
    AUTOMATIC_GATES,
    apply_automatic_gates,
    longest_character_run,
    longest_corpus_overlap,
    ngram_repetition,
    sample_metrics,
    summarize_samples,
)
from training_runtime import (
    atomic_write_json,
    atomic_write_text,
    configure_module_loggers,
    generate_run_id,
)


def _load_records(
    baseline_path: Path | None,
    history_path: Path,
    selected_evaluation_path: Path | None,
    allowed_prompts: set[str],
) -> list[dict[str, Any]]:
    records = []
    if baseline_path is not None:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        records.extend(
            {"step": int(baseline["checkpoint_step"]), **sample}
            for sample in baseline["samples"]
            if sample["prompt"] in allowed_prompts
        )
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
        "--omit-baseline",
        action="store_true",
        help="Analyze only the supplied training history and selected checkpoint.",
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
        None if args.omit_baseline else args.baseline,
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
        aggregated = summarize_samples(rows, corpus, prompt_count=len(prompts))
        entry = {"step": step, **aggregated["summary"]}
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
