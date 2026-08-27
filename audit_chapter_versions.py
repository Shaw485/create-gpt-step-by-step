from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations
import bisect
import json
import logging
import os
import re
from logging.handlers import RotatingFileHandler

from audit_corpus import CHAPTER_PATTERN, chinese_number_to_int
from audit_duplicates import normalize_line


INPUT_PATH = Path(
    os.getenv("CHAPTER_VERSION_INPUT", "data/clean/doupo_stage2.txt")
)
DUPLICATE_REPORT_PATH = Path(
    os.getenv("CHAPTER_VERSION_DUPLICATE_REPORT", "data/clean/duplicate_audit_stage2.json")
)
REPORT_PATH = Path(
    os.getenv("CHAPTER_VERSION_REPORT", "data/clean/chapter_version_audit.json")
)
LOG_PATH = Path("logs/audit_chapter_versions.log")
SEPARATOR = "------------"
MIN_MATCHED_CHARACTERS = 500
MIN_SHARED_COVERAGE = 0.40
HIGH_CONFIDENCE_COVERAGE = 0.98

NOISE_PATTERN = re.compile(
    r"未完待续|推荐票|月票|收藏订阅|第一更|第二更|第三更|第四更|"
    r"手打|首发|百度搜|新书|请假|更新不会"
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

    root_logger = logging.getLogger("chapter_versions")
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.propagate = False

    io_level = os.getenv("CHAPTER_IO_LOG_LEVEL", "INFO").upper()
    analysis_level = os.getenv("CHAPTER_ANALYSIS_LOG_LEVEL", "INFO").upper()
    logging.getLogger("chapter_versions.io").setLevel(io_level)
    logging.getLogger("chapter_versions.analysis").setLevel(analysis_level)


def find_boundaries(lines):
    boundaries = []

    for separator_index, line in enumerate(lines):
        if line.strip() != SEPARATOR:
            continue

        heading_index = separator_index + 1
        while heading_index < len(lines) and not lines[heading_index].strip():
            heading_index += 1

        if heading_index >= len(lines):
            continue
        heading_match = CHAPTER_PATTERN.match(lines[heading_index].strip())
        if heading_match:
            boundaries.append(
                {
                    "separator_index": separator_index,
                    "heading_index": heading_index,
                    "chapter_number": chinese_number_to_int(heading_match.group(1)),
                }
            )

    return boundaries


def build_sections(lines):
    boundaries = find_boundaries(lines)
    sections = []

    for position, boundary in enumerate(boundaries):
        heading_index = boundary["heading_index"]
        end_index = (
            boundaries[position + 1]["separator_index"]
            if position + 1 < len(boundaries)
            else len(lines)
        )
        body_lines = []
        internal_heading_count = 0

        for line in lines[heading_index + 1:end_index]:
            normalized = normalize_line(line)
            if not normalized or normalized == SEPARATOR:
                continue
            if CHAPTER_PATTERN.match(normalized):
                internal_heading_count += 1
                continue
            body_lines.append(normalized)

        body_characters = sum(len(line) for line in body_lines)
        noise_hits = sum(len(NOISE_PATTERN.findall(line)) for line in body_lines)
        quality_score = body_characters - noise_hits * 150 - internal_heading_count * 100

        sections.append(
            {
                "chapter_number": boundary["chapter_number"],
                "title": lines[heading_index].strip(),
                "start_line": heading_index + 1,
                "end_line": end_index,
                "body_lines": body_lines,
                "body_characters": body_characters,
                "noise_hits": noise_hits,
                "internal_heading_count": internal_heading_count,
                "quality_score": quality_score,
            }
        )

    return sections


def section_for_line(sections, start_lines, line_number):
    section_index = bisect.bisect_right(start_lines, line_number) - 1
    if section_index < 0:
        return None
    section = sections[section_index]
    if line_number > section["end_line"]:
        return None
    return section_index


def candidate_pairs(sections, duplicate_blocks):
    pairs = set()
    sections_by_number = defaultdict(list)

    for section_index, section in enumerate(sections):
        sections_by_number[section["chapter_number"]].append(section_index)

    for section_indices in sections_by_number.values():
        if len(section_indices) > 1:
            pairs.update(combinations(section_indices, 2))

    start_lines = [section["start_line"] for section in sections]
    for block in duplicate_blocks:
        first_section = section_for_line(
            sections,
            start_lines,
            block["first_start_line"],
        )
        second_section = section_for_line(
            sections,
            start_lines,
            block["second_start_line"],
        )
        if first_section is None or second_section is None:
            continue
        if first_section != second_section:
            pairs.add(tuple(sorted((first_section, second_section))))

    return sorted(pairs)


def compare_sections(first, second):
    matcher = SequenceMatcher(
        None,
        first["body_lines"],
        second["body_lines"],
        autojunk=False,
    )
    matching_blocks = matcher.get_matching_blocks()
    matched_characters = sum(
        sum(len(first["body_lines"][block.a + offset]) for offset in range(block.size))
        for block in matching_blocks
        if block.size
    )
    shorter_body = min(first["body_characters"], second["body_characters"])
    shared_coverage = matched_characters / shorter_body if shorter_body else 0.0

    return {
        "matched_characters": matched_characters,
        "shared_coverage": shared_coverage,
        "matching_line_ratio": matcher.ratio(),
    }


def public_section(section):
    return {
        key: value
        for key, value in section.items()
        if key != "body_lines"
    }


def audit_versions(sections, pairs):
    version_pairs = []

    for first_index, second_index in pairs:
        first = sections[first_index]
        second = sections[second_index]
        similarity = compare_sections(first, second)

        if similarity["matched_characters"] < MIN_MATCHED_CHARACTERS:
            continue
        if similarity["shared_coverage"] < MIN_SHARED_COVERAGE:
            continue

        keep = max(
            (first, second),
            key=lambda section: (
                section["quality_score"],
                section["body_characters"],
                -section["start_line"],
            ),
        )
        remove = second if keep is first else first
        confidence = (
            "high"
            if (
                first["chapter_number"] == second["chapter_number"]
                and similarity["shared_coverage"] >= HIGH_CONFIDENCE_COVERAGE
            )
            else "review"
        )

        version_pairs.append(
            {
                "confidence": confidence,
                "same_chapter_number": (
                    first["chapter_number"] == second["chapter_number"]
                ),
                "first": public_section(first),
                "second": public_section(second),
                **similarity,
                "recommended_keep_start_line": keep["start_line"],
                "recommended_remove_start_line": remove["start_line"],
            }
        )

    version_pairs.sort(
        key=lambda item: (item["shared_coverage"], item["matched_characters"]),
        reverse=True,
    )
    return version_pairs


def main():
    configure_logging()
    io_logger = logging.getLogger("chapter_versions.io")
    analysis_logger = logging.getLogger("chapter_versions.analysis")

    try:
        io_logger.info("reading corpus path=%s", INPUT_PATH)
        text = INPUT_PATH.read_text(encoding="utf-8")
        lines = text.splitlines()
        duplicate_report = json.loads(
            DUPLICATE_REPORT_PATH.read_text(encoding="utf-8")
        )

        sections = build_sections(lines)
        pairs = candidate_pairs(sections, duplicate_report["duplicate_blocks"])
        version_pairs = audit_versions(sections, pairs)
        high_confidence_count = sum(
            pair["confidence"] == "high" for pair in version_pairs
        )

        report = {
            "input_path": str(INPUT_PATH),
            "chapter_section_count": len(sections),
            "candidate_pair_count": len(pairs),
            "similar_version_pair_count": len(version_pairs),
            "high_confidence_pair_count": high_confidence_count,
            "review_pair_count": len(version_pairs) - high_confidence_count,
            "version_pairs": version_pairs,
        }
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        analysis_logger.debug(
            "highest shared coverage=%.6f",
            version_pairs[0]["shared_coverage"] if version_pairs else 0.0,
        )
        analysis_logger.info(
            "version audit complete sections=%d candidates=%d similar=%d high=%d",
            len(sections),
            len(pairs),
            len(version_pairs),
            high_confidence_count,
        )
        io_logger.info("wrote version report path=%s", REPORT_PATH)
    except Exception:
        io_logger.exception("chapter version audit failed input=%s", INPUT_PATH)
        raise


if __name__ == "__main__":
    main()
