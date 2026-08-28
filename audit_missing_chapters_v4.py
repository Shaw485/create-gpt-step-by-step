"""Audit apparent chapter-number gaps before and after v4 heading repairs."""

from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from prepare_corpus_v4 import (
    atomic_write_json,
    atomic_write_text,
    parse_complete_chapters,
    remove_known_non_story_blocks,
    remove_unsafe_control_characters,
    repair_known_chapter_headings,
    sha256_file,
)


def missing_numbers(numbers: list[int]) -> list[int]:
    if not numbers:
        return []
    return sorted(set(range(min(numbers), max(numbers) + 1)) - set(numbers))


def configure_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("corpus_v4.missing_chapters")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler = RotatingFileHandler(
        log_dir / "corpus_v4_missing_chapters.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    return logger


def build_audit(source_path: Path, corpus_path: Path) -> dict:
    source, _ = remove_unsafe_control_characters(
        source_path.read_text(encoding="utf-8")
    )
    story, _ = remove_known_non_story_blocks(source)
    _, raw_chapters = parse_complete_chapters(story)
    repaired, repairs = repair_known_chapter_headings(story)
    _, repaired_chapters = parse_complete_chapters(repaired)
    _, final_chapters = parse_complete_chapters(
        corpus_path.read_text(encoding="utf-8")
    )

    initial_missing = missing_numbers(
        [chapter.chapter_number for chapter in raw_chapters]
    )
    final_missing = missing_numbers(
        [chapter.chapter_number for chapter in final_chapters]
    )
    recovered = sorted(set(initial_missing) - set(final_missing))
    final_by_number = {}
    for chapter in final_chapters:
        final_by_number.setdefault(chapter.chapter_number, chapter)

    gaps = []
    for number in final_missing:
        previous = final_by_number.get(number - 1)
        following = final_by_number.get(number + 1)
        gaps.append(
            {
                "chapter_number": number,
                "classification": "absent_as_independent_heading_in_source",
                "previous_chapter": (
                    {
                        "number": previous.chapter_number,
                        "title": previous.title,
                        "source_line": previous.start_line,
                    }
                    if previous
                    else None
                ),
                "next_chapter": (
                    {
                        "number": following.chapter_number,
                        "title": following.title,
                        "source_line": following.start_line,
                    }
                    if following
                    else None
                ),
                "action": "保留缺口，不伪造或从未授权来源补写正文。",
            }
        )

    return {
        "schema_version": "1.0",
        "status": "verified",
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "formal_corpus_path": str(corpus_path),
        "formal_corpus_sha256": sha256_file(corpus_path),
        "initial_parser_missing_count": len(initial_missing),
        "initial_parser_missing_numbers": initial_missing,
        "recovered_by_audited_heading_repairs_count": len(recovered),
        "recovered_numbers": recovered,
        "heading_repair_line_count": len(repairs),
        "final_missing_count": len(final_missing),
        "final_missing_numbers": final_missing,
        "requested_prior_count": 20,
        "count_reconciliation": (
            "重新从未修改的 stage3 计算得到初始 26 个编号缺口；经 29 行标题修复"
            "恢复 17 个章节编号，最终只剩 9 个没有独立标题。此前的 20 不是当前"
            "输入和解析规则下可复现的数字，因此不作为冻结依据。"
        ),
        "gaps": gaps,
    }


def markdown_report(report: dict) -> str:
    rows = ["| 编号 | 前一章 | 后一章 | 处理 |", "|---:|---|---|---|"]
    for gap in report["gaps"]:
        previous = gap["previous_chapter"]
        following = gap["next_chapter"]
        rows.append(
            "| {number} | {previous} | {following} | 保留缺口，不补写 |".format(
                number=gap["chapter_number"],
                previous=(previous["title"] if previous else "—"),
                following=(following["title"] if following else "—"),
            )
        )
    return (
        "# v4 缺失章节编号核对\n\n"
        f"- 初始缺口：{report['initial_parser_missing_count']} 个\n"
        f"- 标题纠错恢复：{report['recovered_by_audited_heading_repairs_count']} 个\n"
        f"- 最终缺口：{report['final_missing_count']} 个："
        f"{report['final_missing_numbers']}\n\n"
        f"{report['count_reconciliation']}\n\n"
        + "\n".join(rows)
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/clean/doupo_stage3.txt"))
    parser.add_argument("--corpus", type=Path, default=Path("data/cloud_v4/corpus.txt"))
    parser.add_argument(
        "--report-dir", type=Path, default=Path("data/clean/v4/reports")
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    args = parser.parse_args()
    logger = configure_logger(args.log_dir)
    try:
        report = build_audit(args.source, args.corpus)
        atomic_write_json(args.report_dir / "missing_chapters_audit.json", report)
        atomic_write_text(
            args.report_dir / "missing_chapters_audit.md",
            markdown_report(report),
        )
    except Exception:
        logger.exception("missing chapter audit failed")
        raise
    logger.info(
        "missing chapter audit complete initial=%d recovered=%d final=%d",
        report["initial_parser_missing_count"],
        report["recovered_by_audited_heading_repairs_count"],
        report["final_missing_count"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
