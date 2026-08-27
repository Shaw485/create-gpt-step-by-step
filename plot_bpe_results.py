"""Create reproducible BPE pretraining and SFT loss charts."""

import csv
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/create-gpt-bpe-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRETRAIN_REPORT = Path("reports/milestones/004_bpe_pretrain/bpe_pretrain_report.json")
SFT_REPORT = Path("reports/milestones/005_bpe_sft/bpe_sft_step800_report.json")
OUTPUT_DIR = Path("reports/milestones/005_bpe_sft")


def configure_logger() -> logging.Logger:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bpe.chart")
    logger.handlers.clear()
    logger.propagate = False
    level = getattr(logging, os.getenv("BPE_CHART_LOG_LEVEL", "INFO").upper(), logging.INFO)
    logger.setLevel(level)
    handler = RotatingFileHandler(
        "logs/bpe_chart.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    if os.getenv("BPE_CHART_CONSOLE_LOG", "1") == "1":
        console = logging.StreamHandler()
        console.setFormatter(handler.formatter)
        logger.addHandler(console)
    return logger


def main() -> None:
    logger = configure_logger()
    try:
        pretrain = json.loads(PRETRAIN_REPORT.read_text(encoding="utf-8"))
        sft = json.loads(SFT_REPORT.read_text(encoding="utf-8"))
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for axis, history, title in (
            (axes[0], pretrain["loss_history"], "BPE pretraining"),
            (axes[1], sft["loss_history"], "BPE supervised fine-tuning"),
        ):
            steps = [row["step"] for row in history]
            axis.plot(steps, [row["train_loss"] for row in history], label="Train", marker="o", markersize=3)
            axis.plot(steps, [row["val_loss"] for row in history], label="Validation", marker="s", markersize=3)
            best = min(history, key=lambda row: row["val_loss"])
            axis.scatter([best["step"]], [best["val_loss"]], color="red", zorder=5, label=f"Best val @ {best['step']}")
            axis.set_title(title)
            axis.set_xlabel("Step")
            axis.set_ylabel("Cross-entropy loss")
            axis.grid(alpha=0.25)
            axis.legend()
        fig.tight_layout()
        png_path = OUTPUT_DIR / "bpe_loss_curves.png"
        svg_path = OUTPUT_DIR / "bpe_loss_curves.svg"
        fig.savefig(png_path, dpi=180)
        fig.savefig(svg_path)
        plt.close(fig)

        csv_path = OUTPUT_DIR / "bpe_loss_history.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["stage", "step", "train_loss", "val_loss"])
            writer.writeheader()
            for stage, history in (("pretrain", pretrain["loss_history"]), ("sft", sft["loss_history"])):
                for row in history:
                    writer.writerow({"stage": stage, **row})
        logger.info("charts written png=%s svg=%s csv=%s", png_path, svg_path, csv_path)
    except Exception:
        logger.exception("BPE chart generation failed")
        raise


if __name__ == "__main__":
    main()
