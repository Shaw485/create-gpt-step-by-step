from pathlib import Path
from collections import Counter, defaultdict
import json
import logging
import os
import re
from logging.handlers import RotatingFileHandler


INPUT_PATH = Path(
    os.getenv("CORPUS_AUDIT_INPUT", "data/clean/doupo_stage1.txt")
)
REPORT_PATH = Path(
    os.getenv("CORPUS_AUDIT_REPORT", "data/clean/content_audit.json")
)
LOG_PATH = Path("logs/audit_corpus.log")

AD_RULES = {
    "source_notice": re.compile(r"八零电子书|txt80\.com|本书由.*整理"),
    "download_promo": re.compile(r"TXT免费下载|无弹窗|注册会员|手机阅读斗破苍穹"),
    "site_navigation": re.compile(r"按回车.*返回书目|进入下一章"),
    "update_promo": re.compile(r"更新最快|请牢记.*中文网"),
    "reading_site": re.compile(r"再读中文网|天天中文网|八度吧"),
    "ebook_ad": re.compile(r"Bambook|BBQ电子书|上海书展.*签售"),
    "website": re.compile(r"https?://|www\."),
    "page_chrome": re.compile(
        r"^\ufeff+$|^当前位置:|^《》$|^\(\);$|^1\(\);2\(\);3\(\);$|"
        r"^\|(?:\s*\|)+$|^天蚕土豆作品集|^分册作品-斗破苍穹$|"
        r"^正文更新换月票.*玄幻魔法$"
    ),
}

CHAPTER_PATTERN = re.compile(
    r"^(?:正文\s*)?第([零〇一二两三四五六七八九十百千万0-9]+)章"
)


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

    root_logger = logging.getLogger("corpus_audit")
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.propagate = False

    io_level = os.getenv("AUDIT_IO_LOG_LEVEL", "INFO").upper()
    rules_level = os.getenv("AUDIT_RULES_LOG_LEVEL", "INFO").upper()
    logging.getLogger("corpus_audit.io").setLevel(io_level)
    logging.getLogger("corpus_audit.rules").setLevel(rules_level)


def chinese_number_to_int(text):
    if text.isdigit():
        return int(text)

    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    units = {"十": 10, "百": 100, "千": 1000}
    total = 0
    current_digit = 0

    for character in text:
        if character in digits:
            current_digit = digits[character]
        elif character in units:
            total += (current_digit or 1) * units[character]
            current_digit = 0
        elif character == "万":
            total = (total + current_digit) * 10000
            current_digit = 0

    return total + current_digit


def normalize_heading(line):
    heading = line.strip()
    if heading.startswith("正文"):
        heading = heading[2:].strip()
    return re.sub(r"\s+", "", heading)


def classify_candidate(line, matched_rules):
    if line.startswith("声明:"):
        return "whole_line"
    if "ebook_ad" in matched_rules:
        return "whole_line"
    if "page_chrome" in matched_rules:
        return "whole_line"
    if line == "更新最快":
        return "whole_line"

    whole_line_prefixes = (
        "分册作品-",
        "再读中文网最新改版",
        "小提示：",
        "1();您的支持",
        "《分册作品-",
        "⒊更新最快为了方便访问",
        "更新最快为了方便访问",
    )
    if line.lstrip("\ufeff").startswith(whole_line_prefixes):
        return "whole_line"
    return "embedded"


def audit_lines(lines):
    rules_logger = logging.getLogger("corpus_audit.rules")
    candidates = []
    rule_counts = Counter()
    heading_locations = defaultdict(list)
    chapter_numbers = set()

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        matched_rules = [
            name for name, pattern in AD_RULES.items() if pattern.search(stripped)
        ]

        if matched_rules:
            rule_counts.update(matched_rules)
            candidates.append(
                {
                    "line": line_number,
                    "rules": matched_rules,
                    "kind": classify_candidate(stripped, matched_rules),
                    "preview": stripped[:180],
                }
            )

        chapter_match = CHAPTER_PATTERN.match(stripped)
        if chapter_match:
            normalized_heading = normalize_heading(stripped)
            heading_locations[normalized_heading].append(line_number)
            chapter_numbers.add(chinese_number_to_int(chapter_match.group(1)))

    duplicate_headings = {
        heading: locations
        for heading, locations in heading_locations.items()
        if len(locations) > 1
    }
    max_chapter = max(chapter_numbers)
    missing_chapters = [
        number for number in range(1, max_chapter + 1) if number not in chapter_numbers
    ]

    rules_logger.debug(
        "audit details candidate_lines=%d heading_lines=%d unique_headings=%d",
        len(candidates),
        sum(len(locations) for locations in heading_locations.values()),
        len(heading_locations),
    )
    rules_logger.info(
        "audit complete candidate_lines=%d duplicate_heading_groups=%d missing_chapters=%d",
        len(candidates),
        len(duplicate_headings),
        len(missing_chapters),
    )

    return {
        "input_path": str(INPUT_PATH),
        "total_lines": len(lines),
        "candidate_line_count": len(candidates),
        "candidate_kind_counts": dict(
            Counter(candidate["kind"] for candidate in candidates)
        ),
        "rule_counts": dict(rule_counts),
        "chapter_heading_line_count": sum(
            len(locations) for locations in heading_locations.values()
        ),
        "unique_normalized_heading_count": len(heading_locations),
        "duplicate_heading_group_count": len(duplicate_headings),
        "duplicate_heading_extra_line_count": sum(
            len(locations) - 1 for locations in duplicate_headings.values()
        ),
        "max_chapter_number": max_chapter,
        "missing_chapter_numbers": missing_chapters,
        "candidates": candidates,
        "duplicate_headings": duplicate_headings,
    }


def main():
    configure_logging()
    io_logger = logging.getLogger("corpus_audit.io")

    try:
        io_logger.info("reading input path=%s", INPUT_PATH)
        text = INPUT_PATH.read_text(encoding="utf-8")
        lines = text.splitlines()
        io_logger.debug("input loaded characters=%d lines=%d", len(text), len(lines))

        report = audit_lines(lines)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        io_logger.info("wrote audit report path=%s", REPORT_PATH)
    except Exception:
        io_logger.exception("corpus audit failed input=%s", INPUT_PATH)
        raise


if __name__ == "__main__":
    main()
