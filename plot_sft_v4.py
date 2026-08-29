"""Plot v4 SFT loss history for milestone reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

MATPLOTLIB_CONFIG_DIR = Path(tempfile.gettempdir()) / "create-gpt-sft-v4-matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training_runtime import configure_module_loggers, generate_run_id


DEFAULT_REPORT_PATH = Path(
    "reports/milestones/010_v4_sft_step500/sft_v4_step500_report.json"
)


def validate_loss_history(history: Sequence[dict[str, Any]]) -> None:
    if not history:
        raise ValueError("loss history is empty")
    steps = [int(row["step"]) for row in history]
    if steps != sorted(steps):
        raise ValueError("loss history steps must be increasing")
    if len(set(steps)) != len(steps):
        raise ValueError("loss history steps must be unique")
    for row in history:
        for key in ("train_loss", "val_loss"):
            if float(row[key]) <= 0:
                raise ValueError(f"{key} must be positive")


def plot_loss_history(
    history: Sequence[dict[str, Any]],
    png_path: Path,
    svg_path: Path,
    title: str,
) -> dict[str, Any]:
    validate_loss_history(history)
    best_row = min(history, key=lambda row: float(row["val_loss"]))
    steps = [int(row["step"]) for row in history]
    train_losses = [float(row["train_loss"]) for row in history]
    val_losses = [float(row["val_loss"]) for row in history]

    figure, axis = plt.subplots(figsize=(9, 5.4))
    axis.plot(steps, train_losses, marker="o", label="Training loss")
    axis.plot(steps, val_losses, marker="s", label="Validation loss")
    axis.scatter(
        [int(best_row["step"])],
        [float(best_row["val_loss"])],
        color="black",
        zorder=5,
        label=f"Best validation: step {best_row['step']}",
    )
    axis.annotate(
        f"{float(best_row['val_loss']):.4f}",
        (int(best_row["step"]), float(best_row["val_loss"])),
        xytext=(10, 12),
        textcoords="offset points",
    )
    axis.set_title(title)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Cross-entropy loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=180)
    figure.savefig(svg_path)
    plt.close(figure)
    return {
        "best_step": int(best_row["step"]),
        "best_val_loss": float(best_row["val_loss"]),
        "png_path": str(png_path),
        "svg_path": str(svg_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--svg", type=Path)
    parser.add_argument("--title", default="v4 SFT loss by training step")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = generate_run_id("sft-v4-plot")
    loggers = configure_module_loggers(
        args.report.parent / "logs",
        run_id,
        {"validation": "INFO"},
        console=True,
    )
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        png_path = args.png or args.report.with_name(args.report.stem + "_loss_curve.png")
        svg_path = args.svg or args.report.with_name(args.report.stem + "_loss_curve.svg")
        history = report.get("loss_history", report.get("history"))
        if history is None:
            raise ValueError("report contains neither loss_history nor history")
        result = plot_loss_history(
            history,
            png_path,
            svg_path,
            args.title,
        )
        loggers["validation"].info(
            "sft v4 chart generated best_step=%d best_val_loss=%.6f png=%s svg=%s",
            result["best_step"],
            result["best_val_loss"],
            result["png_path"],
            result["svg_path"],
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        loggers["validation"].exception("SFT v4 chart generation failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
