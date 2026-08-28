"""Archive v4 pretraining metrics and plots for the project video."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "create-gpt-v4-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training_runtime import atomic_write_text, file_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/pretrain_v4_m4"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/milestones/006_v4_local_pretrain"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.run_dir / "report.json").read_text(encoding="utf-8"))
    evaluation = json.loads(
        (args.run_dir / "selected_model_evaluation.json").read_text(encoding="utf-8")
    )
    samples = json.loads((args.run_dir / "samples.json").read_text(encoding="utf-8"))

    copied = {
        "pretrain_v4_report.json": args.run_dir / "report.json",
        "selected_model_evaluation.json": args.run_dir / "selected_model_evaluation.json",
        "sample_history.json": args.run_dir / "samples.json",
        "effective_config.json": args.run_dir / "effective_config.json",
        "corpus_manifest.json": Path("data/cloud_v4/corpus_manifest.json"),
        "token_manifest.json": Path("data/cloud_v4/token_manifest.json"),
        "chapter_pair_review.json": Path(
            "data/clean/v4/reports/chapter_pair_review.json"
        ),
        "missing_chapters_audit.json": Path(
            "data/clean/v4/reports/missing_chapters_audit.json"
        ),
    }
    for target_name, source in copied.items():
        if not source.is_file():
            raise FileNotFoundError(f"required archive source is missing: {source}")
        shutil.copy2(source, args.output_dir / target_name)

    csv_path = args.output_dir / "pretrain_v4_loss.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "train_loss", "val_loss", "learning_rate", "elapsed_seconds"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(report["history"])

    steps = [entry["step"] for entry in report["history"]]
    train_loss = [entry["train_loss"] for entry in report["history"]]
    val_loss = [entry["val_loss"] for entry in report["history"]]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(steps, train_loss, label="Train loss", linewidth=2)
    axis.plot(steps, val_loss, label="Validation loss", linewidth=2)
    axis.scatter(
        [evaluation["checkpoint_step"]],
        [evaluation["checkpoint_validation_loss"]],
        color="#d62728",
        zorder=3,
        label=f"Selected step {evaluation['checkpoint_step']}",
    )
    axis.set_title("v4 local pretraining: 8.1M GPT on Apple M4")
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Cross-entropy loss (BPE token)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "pretrain_v4_loss_curve.png", dpi=180)
    svg_path = args.output_dir / "pretrain_v4_loss_curve.svg"
    figure.savefig(svg_path)
    plt.close(figure)
    # Matplotlib emits harmless trailing spaces in SVG path data; normalize them
    # so repository whitespace checks remain useful for handwritten files.
    normalized_svg = "\n".join(
        line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
    )
    atomic_write_text(svg_path, normalized_svg + "\n")

    # Produce a compact table containing every fixed prompt at every 500-step sample.
    sample_rows = ["| Step | 问题 | 最多30字续写 |", "|---:|---|---|"]
    for milestone in samples["history"]:
        for item in milestone["samples"]:
            continuation = item["continuation"].replace("\n", "↵").replace("|", "\\|")
            prompt = item["prompt"].replace("|", "\\|")
            sample_rows.append(f"| {milestone['step']} | {prompt} | {continuation} |")
    sample_rows.extend(["", "## 验证集选出的最佳模型（Step 2600）", ""])
    sample_rows.extend(["| 问题 | 最多30字续写 |", "|---|---|"])
    for item in evaluation["samples"]:
        continuation = item["continuation"].replace("\n", "↵").replace("|", "\\|")
        prompt = item["prompt"].replace("|", "\\|")
        sample_rows.append(f"| {prompt} | {continuation} |")
    atomic_write_text(args.output_dir / "fixed_prompt_samples.md", "\n".join(sample_rows) + "\n")

    checksum_names = sorted(
        path.name
        for path in args.output_dir.iterdir()
        if path.is_file() and path.name not in {"SHA256SUMS.md", "README.md"}
    )
    checksum_rows = ["# M006 SHA-256", "", "| 文件 | SHA-256 |", "|---|---|"]
    for name in checksum_names:
        checksum_rows.append(f"| `{name}` | `{file_sha256(args.output_dir / name)}` |")
    checksum_rows.extend(
        [
            f"| `runs/pretrain_v4_m4/best.pt` | `{file_sha256(args.run_dir / 'best.pt')}` |",
            f"| `runs/pretrain_v4_m4/latest.pt` | `{file_sha256(args.run_dir / 'latest.pt')}` |",
        ]
    )
    atomic_write_text(args.output_dir / "SHA256SUMS.md", "\n".join(checksum_rows) + "\n")
    print(f"Archived {len(copied)} reports, {len(report['history'])} loss points, and fixed samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
