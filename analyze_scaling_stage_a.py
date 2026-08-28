"""Aggregate Stage A scaling runs using tokenizer-comparable validation BPC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from training_runtime import (
    atomic_write_json,
    atomic_write_text,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
)


def load_experiment(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "report.json"
    config_path = run_dir / "effective_config.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_dir = Path(config["data_dir"])
    manifest_path = data_dir / "token_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if report.get("test_evaluated") is not False or report.get("test_loss") is not None:
        raise ValueError(f"scaling run consumed test data: {run_dir}")
    if report["final_step"] <= 0 or not report["history"]:
        raise ValueError(f"incomplete scaling run: {run_dir}")
    best_entry = min(report["history"], key=lambda row: float(row["val_loss"]))
    train_split = manifest["splits"]["train"]
    model_config = config["model"]
    elapsed_seconds = float(report["stage_elapsed_seconds"])
    return {
        "experiment": run_dir.name,
        "run_dir": str(run_dir),
        "report_sha256": file_sha256(report_path),
        "config_sha256": file_sha256(config_path),
        "token_manifest_sha256": file_sha256(manifest_path),
        "requested_merges": int(manifest["requested_merges"]),
        "vocab_size": int(manifest["vocab_size"]),
        "train_tokens": int(train_split["tokens"]),
        "train_characters": int(train_split["characters"]),
        "characters_per_token": float(train_split["characters_per_token"]),
        "parameter_count": int(report["parameter_count"]),
        "embedding_size": int(model_config["embedding_size"]),
        "num_layers": int(model_config["num_layers"]),
        "num_heads": int(model_config["num_heads"]),
        "block_size": int(model_config["block_size"]),
        "final_step": int(report["final_step"]),
        "training_token_exposures": int(report["training_token_exposures"]),
        "training_tokens_per_parameter": float(
            report["training_tokens_per_parameter"]
        ),
        "best_step": int(report["best_step"]),
        "best_validation_loss": float(report["best_validation_loss"]),
        "best_validation_bits_per_character": float(
            report["best_validation_bits_per_character"]
        ),
        "best_entry": best_entry,
        "elapsed_seconds": elapsed_seconds,
        "training_tokens_per_second": (
            int(report["training_token_exposures"]) / elapsed_seconds
        ),
        "test_evaluated": False,
    }


def rank_experiments(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["best_validation_bits_per_character"]),
            int(row["parameter_count"]),
        ),
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage A 模型规模与Tokenizer实验",
        "",
        "排序指标为验证集 Bits Per Character；test在本阶段保持封存。",
        "",
        "| 排名 | 实验 | 参数 | BPE merges | 词表 | 字符/Token | 训练曝光Token | 最佳Step | 验证Loss | 验证BPC | 用时 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(report["ranked_experiments"], 1):
        lines.append(
            f"| {index} | {row['experiment']} | {row['parameter_count']:,} | "
            f"{row['requested_merges']} | {row['vocab_size']} | "
            f"{row['characters_per_token']:.4f} | "
            f"{row['training_token_exposures']:,} | {row['best_step']} | "
            f"{row['best_validation_loss']:.4f} | "
            f"{row['best_validation_bits_per_character']:.4f} | "
            f"{row['elapsed_seconds'] / 60:.1f}分钟 |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/milestones/015_scaling_stage_a/summary.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/milestones/015_scaling_stage_a/summary.md"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = generate_run_id("scaling-stage-a-analysis")
    loggers = configure_module_loggers(
        args.output_json.parent / "logs",
        run_id,
        {"data": "INFO", "validation": "INFO", "orchestrator": "INFO"},
        max_bytes=10 * 1024 * 1024,
        backup_count=5,
        console=True,
    )
    try:
        rows = [load_experiment(path) for path in args.run_dirs]
        names = [row["experiment"] for row in rows]
        if len(set(names)) != len(names):
            raise ValueError("duplicate scaling experiment names")
        ranked = rank_experiments(rows)
        report = {
            "schema_version": "scaling-stage-a/v1",
            "status": "complete",
            "selection_metric": "minimum validation bits per character",
            "test_policy": "test split not evaluated during scaling selection",
            "experiment_count": len(rows),
            "winner": ranked[0]["experiment"],
            "ranked_experiments": ranked,
        }
        atomic_write_json(args.output_json, report)
        atomic_write_text(args.output_md, render_markdown(report))
        loggers["validation"].info(
            "scaling analysis complete experiments=%d winner=%s best_bpc=%.6f",
            len(rows),
            ranked[0]["experiment"],
            ranked[0]["best_validation_bits_per_character"],
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        loggers["orchestrator"].exception("scaling analysis failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
