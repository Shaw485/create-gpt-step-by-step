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
    parser.add_argument(
        "--evaluation",
        type=Path,
        help="Selected checkpoint evaluation; defaults to RUN_DIR/selected_model_evaluation.json.",
    )
    parser.add_argument(
        "--milestone-name",
        default="M006",
        help="Label written into the checksum archive, for example M007.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.run_dir / "report.json").read_text(encoding="utf-8"))
    evaluation_path = args.evaluation or args.run_dir / "selected_model_evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    samples = json.loads((args.run_dir / "samples.json").read_text(encoding="utf-8"))
    effective_config = json.loads(
        (args.run_dir / "effective_config.json").read_text(encoding="utf-8")
    )
    data_dir = Path(effective_config["data_dir"])

    copied = {
        "pretrain_v4_report.json": args.run_dir / "report.json",
        "selected_model_evaluation.json": evaluation_path,
        "sample_history.json": args.run_dir / "samples.json",
        "effective_config.json": args.run_dir / "effective_config.json",
        "corpus_manifest.json": Path("data/cloud_v4/corpus_manifest.json"),
        "token_manifest.json": data_dir / "token_manifest.json",
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
            fieldnames=[
                "step",
                "train_loss",
                "val_loss",
                "train_bits_per_character",
                "val_bits_per_character",
                "learning_rate",
                "elapsed_seconds",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(report["history"])

    steps = [entry["step"] for entry in report["history"]]
    train_loss = [entry["train_loss"] for entry in report["history"]]
    val_loss = [entry["val_loss"] for entry in report["history"]]
    has_bpc = all("val_bits_per_character" in entry for entry in report["history"])
    if has_bpc:
        train_values = [entry["train_bits_per_character"] for entry in report["history"]]
        val_values = [entry["val_bits_per_character"] for entry in report["history"]]
        selected_value = evaluation["checkpoint_validation_bits_per_character"]
        y_label = "Bits per character"
    else:
        train_values = train_loss
        val_values = val_loss
        selected_value = evaluation["checkpoint_validation_loss"]
        y_label = "Cross-entropy loss (BPE token)"
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(steps, train_values, label="Train", linewidth=2)
    axis.plot(steps, val_values, label="Validation", linewidth=2)
    axis.scatter(
        [evaluation["checkpoint_step"]],
        [selected_value],
        color="#d62728",
        zorder=3,
        label=f"Selected step {evaluation['checkpoint_step']}",
    )
    parameter_millions = report["parameter_count"] / 1_000_000
    axis.set_title(
        f"v4 local pretraining: {parameter_millions:.2f}M GPT on {report['device'].upper()}"
    )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel(y_label)
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

    sample_max_characters = int(
        effective_config["training"]
        .get("sample_max_characters", 30)
    )
    sample_column = f"最多{sample_max_characters}字续写"

    # Produce a compact table containing every fixed prompt at every saved sample step.
    sample_rows = [f"| Step | 提示词 | {sample_column} |", "|---:|---|---|"]
    for milestone in samples["history"]:
        for item in milestone["samples"]:
            continuation = item["continuation"].replace("\n", "↵").replace("|", "\\|")
            prompt = item["prompt"].replace("|", "\\|")
            sample_rows.append(f"| {milestone['step']} | {prompt} | {continuation} |")
    selected_step = evaluation["checkpoint_step"]
    sample_rows.extend(["", f"## 验证集选出的最佳模型（Step {selected_step}）", ""])
    sample_rows.extend([f"| 提示词 | {sample_column} |", "|---|---|"])
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
    checksum_rows = [
        f"# {args.milestone_name} SHA-256",
        "",
        "| 文件 | SHA-256 |",
        "|---|---|",
    ]
    for name in checksum_names:
        checksum_rows.append(f"| `{name}` | `{file_sha256(args.output_dir / name)}` |")
    selected_checkpoint = Path(evaluation["checkpoint"])
    checksum_rows.extend(
        [
            f"| `{selected_checkpoint}` | `{file_sha256(selected_checkpoint)}` |",
            f"| `{args.run_dir}/latest.pt` | `{file_sha256(args.run_dir / 'latest.pt')}` |",
        ]
    )
    atomic_write_text(args.output_dir / "SHA256SUMS.md", "\n".join(checksum_rows) + "\n")
    print(f"Archived {len(copied)} reports, {len(report['history'])} loss points, and fixed samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
