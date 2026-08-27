from pathlib import Path
from collections import defaultdict
from itertools import combinations
import hashlib
import json
import logging
import os
import re
from logging.handlers import RotatingFileHandler


INPUT_PATH = Path(
    os.getenv("DUPLICATE_AUDIT_INPUT", "data/clean/doupo_stage1.txt")
)
REPORT_PATH = Path(
    os.getenv("DUPLICATE_AUDIT_REPORT", "data/clean/duplicate_audit.json")
)
LOG_PATH = Path("logs/audit_duplicates.log")

CHAPTER_PATTERN = re.compile(
    r"^(?:正文\s*)?第[零〇一二两三四五六七八九十百千万0-9]+章"
)
SEPARATOR = "------------"
MIN_BLOCK_LINES = 5
MIN_BLOCK_CHARACTERS = 500
MAX_SEED_OCCURRENCES = 10


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

    root_logger = logging.getLogger("duplicate_audit")
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.propagate = False

    io_level = os.getenv("DUPLICATE_IO_LOG_LEVEL", "INFO").upper()
    analysis_level = os.getenv("DUPLICATE_ANALYSIS_LOG_LEVEL", "INFO").upper()
    logging.getLogger("duplicate_audit.io").setLevel(io_level)
    logging.getLogger("duplicate_audit.analysis").setLevel(analysis_level)


def normalize_line(line):
    return " ".join(line.split())


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_chapter_sections(lines):
    chapter_starts = []

    for index, line in enumerate(lines):
        if line.strip() != SEPARATOR:
            continue

        next_index = index + 1
        while next_index < len(lines) and not lines[next_index].strip():
            next_index += 1

        if next_index < len(lines) and CHAPTER_PATTERN.match(lines[next_index].strip()):
            chapter_starts.append(next_index)

    sections = []
    for position, start_index in enumerate(chapter_starts):
        end_index = (
            chapter_starts[position + 1]
            if position + 1 < len(chapter_starts)
            else len(lines)
        )
        body_lines = [
            normalize_line(line)
            for line in lines[start_index + 1:end_index]
            if normalize_line(line)
        ]
        body_text = "\n".join(body_lines)
        if not body_text:
            continue

        sections.append(
            {
                "title": lines[start_index].strip(),
                "start_line": start_index + 1,
                "end_line": end_index,
                "body_characters": len(body_text),
                "body_sha256": sha256_text(body_text),
            }
        )

    return sections


def group_duplicate_chapters(sections):
    sections_by_hash = defaultdict(list)
    for section in sections:
        sections_by_hash[section["body_sha256"]].append(section)

    return [
        group
        for group in sections_by_hash.values()
        if len(group) > 1
    ]


def build_nonempty_sequence(lines):
    sequence = []
    original_line_numbers = []

    for line_number, line in enumerate(lines, start=1):
        normalized = normalize_line(line)
        if normalized:
            sequence.append(normalized)
            original_line_numbers.append(line_number)

    return sequence, original_line_numbers


def find_duplicate_blocks(lines):
    sequence, original_line_numbers = build_nonempty_sequence(lines)
    seed_locations = defaultdict(list)

    for index, text in enumerate(sequence):
        if len(text) >= 40:
            seed_locations[text].append(index)

    blocks = []
    seen_pairs = set()

    for locations in seed_locations.values():
        if not 2 <= len(locations) <= MAX_SEED_OCCURRENCES:
            continue

        for first_index, second_index in combinations(locations, 2):
            if first_index > 0 and second_index > 0:
                if sequence[first_index - 1] == sequence[second_index - 1]:
                    continue

            matching_lines = 0
            while (
                first_index + matching_lines < len(sequence)
                and second_index + matching_lines < len(sequence)
                and sequence[first_index + matching_lines]
                == sequence[second_index + matching_lines]
            ):
                matching_lines += 1

            if matching_lines < MIN_BLOCK_LINES:
                continue
            if first_index + matching_lines > second_index:
                continue

            character_count = sum(
                len(sequence[index])
                for index in range(first_index, first_index + matching_lines)
            )
            if character_count < MIN_BLOCK_CHARACTERS:
                continue

            pair_key = (first_index, second_index, matching_lines)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            block_text = "\n".join(
                sequence[first_index:first_index + matching_lines]
            )
            blocks.append(
                {
                    "first_start_line": original_line_numbers[first_index],
                    "first_end_line": original_line_numbers[
                        first_index + matching_lines - 1
                    ],
                    "second_start_line": original_line_numbers[second_index],
                    "second_end_line": original_line_numbers[
                        second_index + matching_lines - 1
                    ],
                    "nonempty_line_count": matching_lines,
                    "character_count": character_count,
                    "content_sha256": sha256_text(block_text),
                }
            )

    blocks.sort(key=lambda block: block["character_count"], reverse=True)
    return blocks


def merge_later_duplicate_ranges(blocks, lines):
    ranges = sorted(
        (block["second_start_line"], block["second_end_line"])
        for block in blocks
    )
    merged_ranges = []

    for start_line, end_line in ranges:
        if not merged_ranges or start_line > merged_ranges[-1][1] + 1:
            merged_ranges.append([start_line, end_line])
        else:
            merged_ranges[-1][1] = max(merged_ranges[-1][1], end_line)

    range_reports = []
    for start_line, end_line in merged_ranges:
        redundant_characters = sum(
            len(normalize_line(line))
            for line in lines[start_line - 1:end_line]
        )
        range_reports.append(
            {
                "start_line": start_line,
                "end_line": end_line,
                "physical_line_count": end_line - start_line + 1,
                "normalized_character_count": redundant_characters,
            }
        )

    return range_reports


def main():
    configure_logging()
    io_logger = logging.getLogger("duplicate_audit.io")
    analysis_logger = logging.getLogger("duplicate_audit.analysis")

    try:
        io_logger.info("reading input path=%s", INPUT_PATH)
        text = INPUT_PATH.read_text(encoding="utf-8")
        lines = text.splitlines()
        io_logger.debug("input loaded characters=%d lines=%d", len(text), len(lines))

        sections = find_chapter_sections(lines)
        duplicate_chapter_groups = group_duplicate_chapters(sections)
        duplicate_blocks = find_duplicate_blocks(lines)
        later_duplicate_ranges = merge_later_duplicate_ranges(duplicate_blocks, lines)

        report = {
            "input_path": str(INPUT_PATH),
            "chapter_section_count": len(sections),
            "duplicate_chapter_group_count": len(duplicate_chapter_groups),
            "duplicate_chapter_extra_count": sum(
                len(group) - 1 for group in duplicate_chapter_groups
            ),
            "duplicate_block_count": len(duplicate_blocks),
            "merged_later_duplicate_range_count": len(later_duplicate_ranges),
            "estimated_redundant_characters": sum(
                item["normalized_character_count"]
                for item in later_duplicate_ranges
            ),
            "duplicate_chapter_groups": duplicate_chapter_groups,
            "duplicate_blocks": duplicate_blocks,
            "later_duplicate_ranges": later_duplicate_ranges,
        }
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        analysis_logger.debug(
            "largest duplicate block characters=%d",
            duplicate_blocks[0]["character_count"] if duplicate_blocks else 0,
        )
        analysis_logger.info(
            "duplicate audit complete sections=%d chapter_groups=%d blocks=%d ranges=%d",
            len(sections),
            len(duplicate_chapter_groups),
            len(duplicate_blocks),
            len(later_duplicate_ranges),
        )
        io_logger.info("wrote duplicate report path=%s", REPORT_PATH)
    except Exception:
        io_logger.exception("duplicate audit failed input=%s", INPUT_PATH)
        raise


if __name__ == "__main__":
    main()
