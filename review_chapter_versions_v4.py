"""Create a reproducible, metadata-only review of the 113 version pairs."""

from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from prepare_corpus_v4 import (
    atomic_write_json,
    atomic_write_text,
    parse_complete_chapters,
    remove_unsafe_control_characters,
    repair_known_chapter_headings,
    sha256_text,
    source_range_text,
)


REVIEWER = "codex-assisted-review-2026-08-28"
CROSS_PAIR_RANGE = (5543, 5623)


def configure_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("corpus_v4.review")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler = RotatingFileHandler(
        log_dir / "corpus_v4_review.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def confidence_for(pair: dict[str, Any]) -> str:
    coverage = float(pair.get("shared_coverage") or 0.0)
    line_ratio = float(pair.get("matching_line_ratio") or 0.0)
    if coverage >= 0.80 and line_ratio >= 0.60:
        return "high"
    if coverage >= 0.50 and line_ratio >= 0.40:
        return "medium"
    return "reviewed-low"


def build_review(
    source_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source, _ = remove_unsafe_control_characters(
        source_path.read_text(encoding="utf-8")
    )
    repaired, _ = repair_known_chapter_headings(source)
    _, chapters = parse_complete_chapters(repaired)
    chapters_by_id = {chapter.section_id: chapter for chapter in chapters}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    resolutions: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    decision_counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}

    for pair in manifest["pairs"]:
        if pair["same_chapter_number"]:
            recommended = pair["recommended_keep_start_line"]
            if recommended == pair["first"]["start_line"]:
                decision = "keep_first"
                kept = pair["first"]
                removed = pair["second"]
            elif recommended == pair["second"]["start_line"]:
                decision = "keep_second"
                kept = pair["second"]
                removed = pair["first"]
            else:
                raise ValueError(
                    f"pair {pair['pair_id']} has no valid recommended side"
                )
            confidence = confidence_for(pair)
            resolution = {
                "pair_id": pair["pair_id"],
                "decision": decision,
                "reviewer": REVIEWER,
                "note": (
                    "同一章节的两个候选版本；按完整度、正文长度和噪声质量分选择"
                    "审计建议版本。"
                ),
            }
            review_row = {
                "pair_id": pair["pair_id"],
                "chapter_number": kept["chapter_number"],
                "decision": decision,
                "kept_start_line": kept["start_line"],
                "removed_start_line": removed["start_line"],
                "confidence": confidence,
                "shared_coverage": pair["shared_coverage"],
                "matching_line_ratio": pair["matching_line_ratio"],
            }
        else:
            first = chapters_by_id[pair["first"]["section_id"]]
            start_line, end_line = CROSS_PAIR_RANGE
            range_text = source_range_text(first, start_line, end_line)
            resolution = {
                "pair_id": pair["pair_id"],
                "decision": "remove_range",
                "side": "first",
                "start_line": start_line,
                "end_line": end_line,
                "range_sha256": sha256_text(range_text),
                "reviewer": REVIEWER,
                "note": (
                    "第70章候选末尾嵌入了第71章旧稿；只移除嵌入范围，保留第70章"
                    "以及随后完整的第71章。"
                ),
            }
            confidence = "high"
            review_row = {
                "pair_id": pair["pair_id"],
                "chapter_number": [
                    pair["first"]["chapter_number"],
                    pair["second"]["chapter_number"],
                ],
                "decision": "remove_range",
                "removed_source_lines": [start_line, end_line],
                "range_sha256": resolution["range_sha256"],
                "confidence": confidence,
                "shared_coverage": pair["shared_coverage"],
                "matching_line_ratio": pair["matching_line_ratio"],
            }

        resolutions.append(resolution)
        review_rows.append(review_row)
        decision_counts[resolution["decision"]] = (
            decision_counts.get(resolution["decision"], 0) + 1
        )
        confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1

    resolution_payload = {
        "schema_version": "1.0",
        "source_audit_sha256": manifest["source_audit_sha256"],
        "reviewer": REVIEWER,
        "resolutions": resolutions,
    }
    report = {
        "schema_version": "1.0",
        "status": "complete",
        "pair_count": len(review_rows),
        "same_chapter_pair_count": sum(
            pair["same_chapter_number"] is True for pair in manifest["pairs"]
        ),
        "cross_chapter_pair_count": sum(
            pair["same_chapter_number"] is False for pair in manifest["pairs"]
        ),
        "decision_counts": decision_counts,
        "confidence_counts": confidence_counts,
        "method": (
            "同章候选逐组采用已有正文重合度、行匹配率、完整度、正文长度和噪声"
            "评分建议；跨章异常按源文件行号与 SHA-256 定点处理。"
        ),
        "reviews": review_rows,
    }
    return resolution_payload, report


def report_markdown(report: dict[str, Any]) -> str:
    decisions = ", ".join(
        f"{key}={value}" for key, value in report["decision_counts"].items()
    )
    confidences = ", ".join(
        f"{key}={value}" for key, value in report["confidence_counts"].items()
    )
    return (
        "# v4 章节版本审核\n\n"
        f"- 状态：{report['status']}\n"
        f"- 审核组数：{report['pair_count']}\n"
        f"- 同章版本组：{report['same_chapter_pair_count']}\n"
        f"- 跨章异常组：{report['cross_chapter_pair_count']}\n"
        f"- 决策：{decisions}\n"
        f"- 置信度：{confidences}\n\n"
        "特殊处理：第 70 章候选末尾混入第 71 章旧稿，仅删除经 SHA-256 "
        "锁定的第 5543–5623 行；完整第 70、71 章均保留。\n\n"
        "报告只保存章节号、源行号、评分指标和哈希，不复制小说正文。\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/clean/doupo_stage3.txt"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/clean/v4/reports/chapter_version_review_manifest.json"),
    )
    parser.add_argument(
        "--resolution",
        type=Path,
        default=Path("data/clean/v4/chapter_version_resolutions.json"),
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("data/clean/v4/reports")
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    args = parser.parse_args()
    logger = configure_logger(args.log_dir)
    try:
        resolutions, report = build_review(args.source, args.manifest)
        atomic_write_json(args.resolution, resolutions)
        atomic_write_json(args.report_dir / "chapter_pair_review.json", report)
        atomic_write_text(
            args.report_dir / "chapter_pair_review.md",
            report_markdown(report),
        )
    except Exception:
        logger.exception("chapter pair review failed")
        raise
    logger.info(
        "chapter pair review complete pairs=%d decisions=%s",
        report["pair_count"],
        report["decision_counts"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
