from pathlib import Path
import csv
import json
import logging
import os
import tempfile

from logging.handlers import RotatingFileHandler

MATPLOTLIB_CONFIG_DIR = Path(tempfile.gettempdir()) / "create-gpt-sft-matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prepare_sft_data import parse_log_level


REPORT_DIR = Path("reports/milestones/003f_sft_hq1000_step800")
TRAIN_REPORT_PATH = REPORT_DIR / "sft_hq1000_step800_report.json"
CSV_PATH = REPORT_DIR / "sft_hq1000_loss.csv"
PNG_PATH = REPORT_DIR / "sft_hq1000_loss_curve.png"
SVG_PATH = REPORT_DIR / "sft_hq1000_loss_curve.svg"


def configure_logger() -> logging.Logger:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sft.hq.chart")
    logger.handlers.clear()
    logger.setLevel(
        parse_log_level(os.getenv("SFT_CHART_LOG_LEVEL", "INFO"))
    )
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler = RotatingFileHandler(
        "logs/sft_hq_chart.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if os.getenv("SFT_CHART_CONSOLE_LOG", "1") == "1":
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    logger.propagate = False
    return logger


def validate_history(history: list[dict]) -> None:
    if not history:
        raise ValueError("loss history is empty")
    steps = [int(row["step"]) for row in history]
    if steps != sorted(steps) or len(steps) != len(set(steps)):
        raise ValueError("loss steps must be unique and increasing")


def main() -> None:
    logger = configure_logger()
    try:
        report = json.loads(TRAIN_REPORT_PATH.read_text(encoding="utf-8"))
        history = report["loss_history"]
        validate_history(history)
        best_row = min(history, key=lambda row: float(row["val_loss"]))

        with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=["step", "train_loss", "val_loss"]
            )
            writer.writeheader()
            writer.writerows(history)

        steps = [row["step"] for row in history]
        train_losses = [row["train_loss"] for row in history]
        val_losses = [row["val_loss"] for row in history]
        figure, axis = plt.subplots(figsize=(9, 5.4))
        axis.plot(steps, train_losses, marker="o", label="Training loss")
        axis.plot(steps, val_losses, marker="s", label="Validation loss")
        axis.scatter(
            [best_row["step"]],
            [best_row["val_loss"]],
            color="black",
            zorder=5,
            label=f"Best validation: step {best_row['step']}",
        )
        axis.annotate(
            f"{best_row['val_loss']:.4f}",
            (best_row["step"], best_row["val_loss"]),
            xytext=(10, 12),
            textcoords="offset points",
        )
        axis.set_title("SFT HQ1000 loss by training step")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Cross-entropy loss")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(PNG_PATH, dpi=180)
        figure.savefig(SVG_PATH)
        plt.close(figure)
        logger.info(
            "chart saved best_step=%d best_val=%.6f png=%s svg=%s csv=%s",
            best_row["step"],
            best_row["val_loss"],
            PNG_PATH,
            SVG_PATH,
            CSV_PATH,
        )
    except Exception:
        logger.exception("SFT loss chart generation failed")
        raise


if __name__ == "__main__":
    main()
