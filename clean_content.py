from pathlib import Path
from collections import Counter
import hashlib
import json
import logging
import os
import re
from logging.handlers import RotatingFileHandler

from audit_corpus import AD_RULES, classify_candidate


INPUT_PATH = Path("data/clean/doupo_stage1.txt")
OUTPUT_PATH = Path("data/clean/doupo_stage2.txt")
REPORT_PATH = Path("data/clean/stage2_report.json")
LOG_PATH = Path("logs/clean_content.log")
MINIMUM_RETAINED_RATIO = 0.98

INLINE_RULES = {
    "mobile_reading": re.compile(r"手机阅读斗破苍穹："),
    "tiantian_watermark": re.compile(
        r"\*{2}首发，天天中文网\s*\*{2}|天天中文网首发|"
        r"\^\^百度搜，天天中文网阅读本书最新章节\s*\*{2}"
    ),
    "badu_watermark": re.compile(
        r"复制可耻八度吧首发|八度吧手打复制请说明转自八度吧|"
        r"八度吧手打首发|八度吧首发|看书选八度吧|百度搜索八度吧|"
        r"（更新最快八度吧|（八度吧）|\(八度吧\)|八度吧"
    ),
    "update_banner": re.compile(
        r"为了方便访问,请牢记天天中文网您的支持是我们最大的动力！|"
        r"更新最快"
    ),
    "source_marker": re.compile(r"/本书由\.整理/"),
    "txt80_suffix": re.compile(
        r"\(未完待续，如欲知后事如何，请登陆www\.txt80\.com，"
        r"章节更多，支持作者，支持正版阅读！\)"
    ),
    "bom_marker": re.compile("\ufeff"),
}


def configure_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger("content_cleaner")
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.propagate = False

    io_level = os.getenv("CONTENT_IO_LOG_LEVEL", "INFO").upper()
    transform_level = os.getenv("CONTENT_TRANSFORM_LOG_LEVEL", "INFO").upper()
    logging.getLogger("content_cleaner.io").setLevel(io_level)
    logging.getLogger("content_cleaner.transform").setLevel(transform_level)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def matched_ad_rules(line):
    stripped = line.strip()
    return [
        name for name, pattern in AD_RULES.items() if pattern.search(stripped)
    ]


def clean_lines(lines):
    transform_logger = logging.getLogger("content_cleaner.transform")
    output_lines = []
    whole_line_rule_counts = Counter()
    inline_replacement_counts = Counter()
    unresolved_candidates = []

    for line_number, line in enumerate(lines, start=1):
        matched_rules = matched_ad_rules(line)
        if matched_rules and classify_candidate(line.strip(), matched_rules) == "whole_line":
            whole_line_rule_counts.update(matched_rules)
            continue

        cleaned_line = line
        for rule_name, pattern in INLINE_RULES.items():
            cleaned_line, replacement_count = pattern.subn("", cleaned_line)
            if replacement_count:
                inline_replacement_counts[rule_name] += replacement_count

        remaining_rules = matched_ad_rules(cleaned_line)
        if remaining_rules:
            unresolved_candidates.append(
                {
                    "original_line": line_number,
                    "rules": remaining_rules,
                    "preview": cleaned_line.strip()[:180],
                }
            )

        output_lines.append(cleaned_line)

    transform_logger.debug(
        "transformation details input_lines=%d output_lines=%d",
        len(lines),
        len(output_lines),
    )
    transform_logger.info(
        "content cleaned whole_lines_removed=%d inline_replacements=%d unresolved=%d",
        len(lines) - len(output_lines),
        sum(inline_replacement_counts.values()),
        len(unresolved_candidates),
    )

    statistics = {
        "input_lines": len(lines),
        "output_lines": len(output_lines),
        "whole_lines_removed": len(lines) - len(output_lines),
        "whole_line_rule_counts": dict(whole_line_rule_counts),
        "inline_replacement_counts": dict(inline_replacement_counts),
        "unresolved_candidate_count": len(unresolved_candidates),
        "unresolved_candidates": unresolved_candidates,
    }
    return output_lines, statistics


def main():
    configure_logging()
    io_logger = logging.getLogger("content_cleaner.io")

    try:
        io_logger.info("reading input path=%s", INPUT_PATH)
        input_text = INPUT_PATH.read_text(encoding="utf-8")
        had_trailing_newline = input_text.endswith("\n")
        input_lines = input_text.splitlines()

        output_lines, statistics = clean_lines(input_lines)
        output_text = "\n".join(output_lines)
        if had_trailing_newline:
            output_text += "\n"

        retained_ratio = len(output_text) / len(input_text)
        if retained_ratio < MINIMUM_RETAINED_RATIO:
            raise RuntimeError(
                f"cleaning retained only {retained_ratio:.4%} of the input"
            )

        OUTPUT_PATH.write_text(output_text, encoding="utf-8")
        report = {
            "input_path": str(INPUT_PATH),
            "output_path": str(OUTPUT_PATH),
            "input_sha256": sha256_text(input_text),
            "output_sha256": sha256_text(output_text),
            "input_characters": len(input_text),
            "output_characters": len(output_text),
            "characters_removed": len(input_text) - len(output_text),
            "retained_ratio": retained_ratio,
            **statistics,
        }
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        io_logger.info(
            "wrote output path=%s characters=%d report=%s",
            OUTPUT_PATH,
            len(output_text),
            REPORT_PATH,
        )
    except Exception:
        io_logger.exception("content cleaning failed input=%s", INPUT_PATH)
        raise


if __name__ == "__main__":
    main()
