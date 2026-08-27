import csv
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import tempfile

MATPLOTLIB_CONFIG_DIR = Path(tempfile.gettempdir()) / "create-gpt-matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import train_gpt_stage3 as training


CHECKPOINT_PATH = Path(
    os.getenv("EVAL_CHECKPOINT_PATH", "checkpoints/gpt_stage5_best.pt")
)
TENSOR_PATH = Path(
    os.getenv("EVAL_TENSOR_PATH", "data/clean/doupo_stage3_tensors.pt")
)
HISTORY_PATH = Path(
    os.getenv("EVAL_HISTORY_PATH", "checkpoints/gpt_stage5_history.json")
)
PROMPT_PATH = Path(os.getenv("EVAL_PROMPT_PATH", "data/prompt10_eval.txt"))
REPORT_PATH = Path(os.getenv("EVAL_REPORT_PATH", "reports/stage5_metrics.json"))
TABLE_PATH = Path(os.getenv("EVAL_TABLE_PATH", "reports/stage5_loss_table.csv"))
CHART_PNG_PATH = Path(
    os.getenv("EVAL_CHART_PNG_PATH", "reports/stage5_loss_curve.png")
)
CHART_SVG_PATH = Path(
    os.getenv("EVAL_CHART_SVG_PATH", "reports/stage5_loss_curve.svg")
)

EVAL_BATCHES = int(os.getenv("EVAL_BATCHES", "100"))
MAX_NEW_TOKENS = int(os.getenv("EVAL_MAX_NEW_TOKENS", "30"))
TEMPERATURE = float(os.getenv("EVAL_TEMPERATURE", "0.8"))
TOP_K = int(os.getenv("EVAL_TOP_K", "20"))
SEED = int(os.getenv("EVAL_SEED", "42"))


def configure_logging() -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)

    for name in ("eval.data", "eval.metrics", "eval.generation", "eval.chart"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False
        logger.addHandler(console)
        file_handler = RotatingFileHandler(
            f"logs/{name.replace('.', '_')}.log",
            maxBytes=1_000_000,
            backupCount=3,
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def count_topk_correct(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> int:
    if k <= 0 or k > logits.shape[-1]:
        raise ValueError("k must be between 1 and vocabulary size")
    predictions = torch.topk(logits, k=k, dim=-1).indices
    matches = predictions.eq(targets.unsqueeze(-1)).any(dim=-1)
    return int(matches.sum().item())


@torch.no_grad()
def evaluate_model(
    model: training.GPTLanguageModel,
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_batches: int,
) -> dict[str, float | int]:
    if eval_batches <= 0:
        raise ValueError("eval_batches must be positive")

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    top1_correct = 0
    top5_correct = 0

    for _ in range(eval_batches):
        inputs, targets = training.get_batch(data, batch_size, block_size)
        logits, loss = model(inputs, targets)
        token_count = targets.numel()
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count
        top1_correct += count_topk_correct(logits, targets, 1)
        top5_correct += count_topk_correct(logits, targets, 5)

    if was_training:
        model.train()

    return {
        "loss": total_loss / total_tokens,
        "top1_accuracy": top1_correct / total_tokens,
        "top5_accuracy": top5_correct / total_tokens,
        "tokens_evaluated": total_tokens,
        "batches_evaluated": eval_batches,
    }


def ngram_repetition_rate(text: str, n: int = 2) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    ngrams = [text[index : index + n] for index in range(len(text) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def adjacent_repetition_rate(text: str) -> float:
    if len(text) < 2:
        return 0.0
    repeated = sum(left == right for left, right in zip(text, text[1:]))
    return repeated / (len(text) - 1)


def normalize_for_repetition(text: str) -> str:
    return "".join(text.split())


def generate_samples(
    model: training.GPTLanguageModel,
    prompts: list[str],
    stoi: dict[str, int],
    itos: dict[int, str],
    block_size: int,
) -> list[dict]:
    results = []
    for index, prompt in enumerate(prompts, start=1):
        training.set_global_seed(SEED + index, deterministic=True)
        prompt_ids = training.encode_prompt(prompt, stoi)
        continuation = training.generate_from_prompt(
            model=model,
            prompt_ids=prompt_ids,
            itos=itos,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            block_size=block_size,
        )
        repetition_text = normalize_for_repetition(continuation)
        results.append(
            {
                "prompt_index": index,
                "prompt": prompt,
                "continuation": continuation,
                "raw_character_count": len(continuation),
                "content_character_count": len(repetition_text),
                "adjacent_repetition_rate": adjacent_repetition_rate(repetition_text),
                "bigram_repetition_rate": ngram_repetition_rate(repetition_text, 2),
                "trigram_repetition_rate": ngram_repetition_rate(repetition_text, 3),
            }
        )
    return results


def load_history(path: Path) -> list[dict]:
    history = json.loads(path.read_text(encoding="utf-8"))
    if not history:
        raise ValueError("loss history is empty")
    return sorted(history, key=lambda item: int(item["step"]))


def write_history_table(history: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("step", "train_loss", "val_loss", "gap"),
        )
        writer.writeheader()
        for item in history:
            writer.writerow(
                {
                    "step": item["step"],
                    "train_loss": item["train_loss"],
                    "val_loss": item["val_loss"],
                    "gap": item["val_loss"] - item["train_loss"],
                }
            )


def plot_loss_history(
    history: list[dict],
    png_path: Path,
    svg_path: Path,
) -> dict:
    if not history:
        raise ValueError("loss history is empty")

    steps = [int(item["step"]) for item in history]
    train_losses = [float(item["train_loss"]) for item in history]
    val_losses = [float(item["val_loss"]) for item in history]
    best = min(history, key=lambda item: float(item["val_loss"]))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    ax.plot(
        steps,
        train_losses,
        color="#4C72B0",
        linewidth=2.2,
        linestyle="-",
        label="Training loss",
    )
    ax.plot(
        steps,
        val_losses,
        color="#DD8452",
        linewidth=2.2,
        linestyle="--",
        label="Validation loss",
    )
    ax.scatter(
        [best["step"]],
        [best["val_loss"]],
        color="#C44E52",
        edgecolor="white",
        linewidth=1.2,
        s=80,
        zorder=5,
        label="Best validation point",
    )
    ax.annotate(
        f"Best: {float(best['val_loss']):.3f} at step {int(best['step']):,}",
        xy=(best["step"], best["val_loss"]),
        xytext=(-190, 32),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=10,
    )
    ax.set_title(
        f"Stage 5 validation loss reached {float(best['val_loss']):.3f} by step {int(best['step']):,}",
        fontweight="bold",
        fontsize=15,
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend(frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", alpha=0.28)
    fig.text(
        0.01,
        0.01,
        f"Source: {HISTORY_PATH} | {len(history)} evaluation points",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return best


def main() -> None:
    configure_logging()
    data_logger = logging.getLogger("eval.data")
    metrics_logger = logging.getLogger("eval.metrics")
    generation_logger = logging.getLogger("eval.generation")
    chart_logger = logging.getLogger("eval.chart")

    try:
        training.set_global_seed(SEED, deterministic=True)
        payload = torch.load(TENSOR_PATH, weights_only=False)
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=training.DEVICE,
            weights_only=False,
        )
        meta = checkpoint["meta"]
        data_logger.info(
            "loaded checkpoint=%s step=%d tensor=%s device=%s",
            CHECKPOINT_PATH,
            int(meta["step"]),
            TENSOR_PATH,
            training.DEVICE,
        )

        model = training.GPTLanguageModel(
            vocab_size=int(meta["vocab_size"]),
            embedding_size=int(meta["embedding_dim"]),
            num_heads=int(meta["num_heads"]),
            context_size=int(meta["block_size"]),
            num_layers=int(meta["num_layers"]),
        ).to(training.DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])

        validation_metrics = evaluate_model(
            model=model,
            data=payload["val_data"].to(training.DEVICE),
            batch_size=int(meta["batch_size"]),
            block_size=int(meta["block_size"]),
            eval_batches=EVAL_BATCHES,
        )
        metrics_logger.info("validation metrics=%s", validation_metrics)

        prompts = training.load_prompt_inputs(str(PROMPT_PATH))
        samples = generate_samples(
            model=model,
            prompts=prompts,
            stoi=payload["stoi"],
            itos=payload["itos"],
            block_size=int(meta["block_size"]),
        )
        mean_adjacent = sum(
            item["adjacent_repetition_rate"] for item in samples
        ) / len(samples)
        mean_bigram = sum(item["bigram_repetition_rate"] for item in samples) / len(samples)
        mean_trigram = sum(item["trigram_repetition_rate"] for item in samples) / len(samples)
        repetition_metrics = {
            "mean_adjacent_repetition_rate": mean_adjacent,
            "mean_bigram_repetition_rate": mean_bigram,
            "mean_trigram_repetition_rate": mean_trigram,
        }
        generation_logger.info("repetition metrics=%s", repetition_metrics)

        history = load_history(HISTORY_PATH)
        write_history_table(history, TABLE_PATH)
        best_history_entry = plot_loss_history(
            history,
            CHART_PNG_PATH,
            CHART_SVG_PATH,
        )
        chart_logger.info(
            "chart saved png=%s svg=%s best_step=%d best_val_loss=%.6f",
            CHART_PNG_PATH,
            CHART_SVG_PATH,
            int(best_history_entry["step"]),
            float(best_history_entry["val_loss"]),
        )

        report = {
            "checkpoint": str(CHECKPOINT_PATH),
            "checkpoint_step": int(meta["step"]),
            "device": str(training.DEVICE),
            "evaluation": validation_metrics,
            "generation_settings": {
                "prompt_count": len(prompts),
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
                "top_k": TOP_K,
                "seed": SEED,
            },
            "repetition": repetition_metrics,
            "repetition_note": "Whitespace is removed before repetition rates are calculated.",
            "samples": samples,
            "history": {
                "evaluation_points": len(history),
                "best_entry": best_history_entry,
                "table_path": str(TABLE_PATH),
                "chart_png_path": str(CHART_PNG_PATH),
                "chart_svg_path": str(CHART_SVG_PATH),
            },
            "chart_alt_text": (
                "Line chart of Stage 5 training and validation cross-entropy loss "
                "from step 0 to 10000. Both fall sharply early and then decline "
                "more gradually; the lowest recorded validation loss is at step 10000."
            ),
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metrics_logger.info("report saved=%s", REPORT_PATH)

        print(json.dumps({
            "evaluation": validation_metrics,
            "repetition": repetition_metrics,
            "best_history_entry": best_history_entry,
        }, ensure_ascii=False, indent=2))
    except Exception:
        metrics_logger.exception("stage5 evaluation failed")
        raise


if __name__ == "__main__":
    main()
