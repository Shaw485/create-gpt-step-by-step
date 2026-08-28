"""Plot the sealed-test Stage A tokenizer and model scaling experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/create-gpt-scaling-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training_runtime import configure_module_loggers, generate_run_id


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def model_curve_rows(model_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in model_summary["ranked_experiments"]:
        report_path = Path(experiment["run_dir"]) / "report.json"
        report = load_json(report_path)
        for point in report["history"]:
            rows.append(
                {
                    "experiment": experiment["experiment"],
                    "parameter_count": experiment["parameter_count"],
                    "step": point["step"],
                    "train_bpc": point["train_bits_per_character"],
                    "validation_bpc": point["val_bits_per_character"],
                }
            )
    return rows


def write_curve_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def normalize_svg(path: Path) -> None:
    """Remove generator-only trailing whitespace for clean version control."""
    content = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in content.splitlines()) + "\n",
        encoding="utf-8",
    )


def create_figure(
    tokenizer_summary: dict[str, Any],
    model_summary: dict[str, Any],
    curve_rows: Sequence[dict[str, Any]],
) -> plt.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    tokenizer_rows = sorted(
        tokenizer_summary["ranked_experiments"],
        key=lambda row: row["requested_merges"],
    )
    merges = [row["requested_merges"] for row in tokenizer_rows]
    bpc = [row["best_validation_bits_per_character"] for row in tokenizer_rows]
    axes[0].plot(merges, bpc, marker="o", linewidth=2)
    axes[0].set_title("Tokenizer screen (4M, 500 steps)")
    axes[0].set_xlabel("BPE merges")
    axes[0].set_ylabel("Best validation BPC (lower is better)")
    axes[0].grid(alpha=0.25)

    for experiment in sorted(
        model_summary["ranked_experiments"],
        key=lambda row: row["parameter_count"],
    ):
        points = [
            row for row in curve_rows
            if row["experiment"] == experiment["experiment"]
        ]
        label = f"{experiment['parameter_count'] / 1_000_000:.1f}M"
        axes[1].plot(
            [row["step"] for row in points],
            [row["validation_bpc"] for row in points],
            marker="o",
            markersize=3,
            label=label,
        )
    axes[1].set_title("Model scaling (BPE 3000)")
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Validation BPC")
    axes[1].grid(alpha=0.25)
    axes[1].legend(title="Parameters")

    model_rows = sorted(
        model_summary["ranked_experiments"],
        key=lambda row: row["parameter_count"],
    )
    axes[2].plot(
        [row["parameter_count"] / 1_000_000 for row in model_rows],
        [row["best_validation_bits_per_character"] for row in model_rows],
        marker="o",
        linewidth=2,
    )
    for row in model_rows:
        axes[2].annotate(
            f"{row['elapsed_seconds'] / 60:.1f} min",
            (
                row["parameter_count"] / 1_000_000,
                row["best_validation_bits_per_character"],
            ),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=8,
        )
    axes[2].set_title("Quality vs model size")
    axes[2].set_xlabel("Parameters (millions)")
    axes[2].set_ylabel("Best validation BPC")
    axes[2].grid(alpha=0.25)

    figure.suptitle("Stage A: tokenizer and model scaling (test split sealed)")
    figure.tight_layout()
    return figure


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-summary",
        type=Path,
        default=Path(
            "reports/milestones/015_scaling_stage_a/tokenizer_screen_summary.json"
        ),
    )
    parser.add_argument(
        "--model-summary",
        type=Path,
        default=Path(
            "reports/milestones/015_scaling_stage_a/model_screen_summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/milestones/015_scaling_stage_a"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = generate_run_id("scaling-stage-a-plot")
    loggers = configure_module_loggers(
        args.output_dir / "logs",
        run_id,
        {"data": "INFO", "orchestrator": "INFO"},
        max_bytes=5 * 1024 * 1024,
        backup_count=3,
        console=True,
    )
    try:
        tokenizer_summary = load_json(args.tokenizer_summary)
        model_summary = load_json(args.model_summary)
        if (
            tokenizer_summary.get("test_policy")
            != "test split not evaluated during scaling selection"
            or model_summary.get("test_policy")
            != "test split not evaluated during scaling selection"
        ):
            raise ValueError("plot inputs do not confirm the sealed-test policy")
        rows = model_curve_rows(model_summary)
        if not rows:
            raise ValueError("model summary has no curve points")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_curve_csv(args.output_dir / "model_bpc_curves.csv", rows)
        figure = create_figure(tokenizer_summary, model_summary, rows)
        figure.savefig(args.output_dir / "scaling_stage_a.png", dpi=180)
        svg_path = args.output_dir / "scaling_stage_a.svg"
        figure.savefig(svg_path)
        plt.close(figure)
        normalize_svg(svg_path)
        loggers["data"].info(
            "scaling chart written",
            extra={
                "context": {
                    "curve_rows": len(rows),
                    "output_dir": args.output_dir,
                }
            },
        )
        return 0
    except Exception:
        loggers["orchestrator"].exception("scaling chart generation failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
