from collections import defaultdict
from pathlib import Path
import hashlib
import json
import logging
import os
from logging.handlers import RotatingFileHandler

from audit_chapter_versions import CHAPTER_PATTERN

INPUT_PATH = Path(
    os.getenv("DUPLICATE_VERSION_INPUT", "data/clean/doupo_stage2.txt")
)
VERSION_AUDIT_PATH = Path(
    os.getenv("DUPLICATE_VERSION_AUDIT", "data/clean/chapter_version_audit.json")
)
OUTPUT_PATH = Path(
    os.getenv("DUPLICATE_VERSION_OUTPUT", "data/clean/doupo_stage3.txt")
)
REPORT_PATH = Path(
    os.getenv("DUPLICATE_VERSION_REPORT", "data/clean/stage3_report.json")
)
LOG_PATH = Path("logs/clean_duplicate_versions.log")
SEPARATOR = "------------"
CONFIDENCE_TARGET = "high"


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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

    root_logger = logging.getLogger("duplicate_cleaner")
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.propagate = False

    io_level = os.getenv("DUPLICATE_CLEAN_IO_LOG_LEVEL", "INFO").upper()
    transform_level = os.getenv("DUPLICATE_CLEAN_TRANSFORM_LOG_LEVEL", "INFO").upper()
    logging.getLogger("duplicate_cleaner.io").setLevel(io_level)
    logging.getLogger("duplicate_cleaner.transform").setLevel(transform_level)


def line_strip_n(line):
    return line.rstrip("\r\n")


def count_lines(text):
    if not text:
        return 0
    return text.count("\n") + 1


def find_previous_separator(lines, heading_index):
    index = heading_index - 1
    while index >= 0:
        if line_strip_n(lines[index]).strip() == SEPARATOR:
            return index
        index -= 1
    return heading_index


def normalize_remove_candidates(version_pairs):
    candidates = [pair for pair in version_pairs if pair.get("confidence") == CONFIDENCE_TARGET]
    if not candidates:
        return []

    grouped = defaultdict(list)
    for pair in candidates:
        grouped[pair["recommended_remove_start_line"]].append(pair)

    normalized = []
    for remove_start_line, grouped_pairs in grouped.items():
        if len(grouped_pairs) == 1:
            pair = grouped_pairs[0]
            normalized.append(pair)
            continue

        # Keep the pair with the shortest section to maximize dedup coverage
        best_pair = min(
            grouped_pairs,
            key=lambda candidate: min(
                candidate["first"]["body_characters"],
                candidate["second"]["body_characters"],
            ),
        )
        normalized.append(best_pair)

    normalized.sort(key=lambda item: item["recommended_remove_start_line"])
    return normalized


def build_remove_ranges(lines, high_pairs):
    ranges = []
    used_start_lines = set()

    for pair in high_pairs:
        remove_start_line = pair["recommended_remove_start_line"]
        if remove_start_line in used_start_lines:
            continue
        used_start_lines.add(remove_start_line)

        if pair["first"]["start_line"] == remove_start_line:
            remove_section = pair["first"]
            keep_start_line = pair["recommended_keep_start_line"]
        elif pair["second"]["start_line"] == remove_start_line:
            remove_section = pair["second"]
            keep_start_line = pair["recommended_keep_start_line"]
        else:
            raise ValueError(
                "remove line does not match first/second section",
                {
                    "remove_start_line": remove_start_line,
                    "first_start_line": pair["first"]["start_line"],
                    "second_start_line": pair["second"]["start_line"],
                },
            )

        heading_index = remove_section["start_line"] - 1
        if heading_index < 0 or heading_index >= len(lines):
            raise ValueError("remove heading line out of range", remove_section)

        expected_title = remove_section["title"].strip()
        actual_title = line_strip_n(lines[heading_index]).strip()
        if actual_title != expected_title:
            # If the line shifted due to upstream edits, keep search bounded
            found_heading = None
            for seek in range(-5, 6):
                candidate_index = heading_index + seek
                if 0 <= candidate_index < len(lines):
                    if line_strip_n(lines[candidate_index]).strip() == expected_title:
                        found_heading = candidate_index
                        break
            if found_heading is None:
                raise ValueError(
                    "expected heading not found",
                    {
                        "expected": expected_title,
                        "actual": actual_title,
                        "remove_start_line": remove_start_line,
                    },
                )
            heading_index = found_heading

        start_index = find_previous_separator(lines, heading_index)
        end_index = remove_section["end_line"]
        if end_index < 0 or end_index > len(lines):
            raise ValueError(
                "invalid section end line",
                {"remove_section": remove_section},
            )

        start_line_one_based = start_index + 1
        end_line_inclusive_one_based = end_index
        if end_line_inclusive_one_based < start_line_one_based:
            raise ValueError(
                "invalid removal window",
                {
                    "section_start_line": start_line_one_based,
                    "section_end_line": end_line_inclusive_one_based,
                },
            )

        if not CHAPTER_PATTERN.match(expected_title):
            raise ValueError(
                "candidate heading is not chapter pattern",
                {"heading": expected_title, "remove_start_line": remove_start_line},
            )

        ranges.append(
            {
                "chapter_number": remove_section["chapter_number"],
                "title": expected_title,
                "confidence": pair.get("confidence"),
                "shared_coverage": pair.get("shared_coverage"),
                "matched_characters": pair.get("matched_characters"),
                "recommended_keep_start_line": keep_start_line,
                "reported_remove_start_line": remove_start_line,
                "reported_remove_end_line": remove_section["end_line"],
                "remove_section_start_line": start_line_one_based,
                "remove_section_end_line_inclusive": end_index,
                "remove_section_end_exclusive": end_index,
                "line_range": [start_index, end_index],
                "exclusive_end": end_index,
            }
        )

    return sorted(ranges, key=lambda item: item["line_range"][0])


def merge_overlapping_ranges(raw_ranges):
    merged = []
    for item in raw_ranges:
        start = item["line_range"][0]
        end_exclusive = item["line_range"][1]
        if not merged:
            merged.append({**item, "line_range": [start, end_exclusive]})
            continue

        prev = merged[-1]
        prev_start, prev_end = prev["line_range"]
        if start < prev_end:
            raise ValueError(
                "overlapping removal ranges detected; refusing to auto-delete",
                {
                    "previous": {"start": prev_start, "end": prev_end},
                    "current": {"start": start, "end": end_exclusive},
                },
            )
        merged.append({**item, "line_range": [start, end_exclusive]})

    return merged


def remove_ranges(lines, ranges):
    drop_set = [False] * len(lines)
    removed_segments = []

    for item in ranges:
        start, end_exclusive = item["line_range"]
        for idx in range(start, end_exclusive):
            drop_set[idx] = True
        removed_segments.append(lines[start:end_exclusive])

    kept_lines = [line for idx, line in enumerate(lines) if not drop_set[idx]]
    kept_text = "".join(kept_lines)

    removed_text = "".join(
        line
        for idx, line in enumerate(lines)
        if drop_set[idx]
    )

    removed_line_count = sum(item["line_range"][1] - item["line_range"][0] for item in ranges)
    removed_characters = len(removed_text)
    removed_sections = [
        {
            "chapter_number": item["chapter_number"],
            "title": item["title"],
            "line_start": item["line_range"][0],
            "line_end_exclusive": item["line_range"][1],
            "line_count": item["line_range"][1] - item["line_range"][0],
            "matched_characters": item["matched_characters"],
            "shared_coverage": item["shared_coverage"],
            "recommended_keep_start_line": item["recommended_keep_start_line"],
        }
        for item in ranges
    ]

    return kept_text, removed_line_count, removed_characters, removed_sections


def main():
    configure_logging()
    io_logger = logging.getLogger("duplicate_cleaner.io")
    transform_logger = logging.getLogger("duplicate_cleaner.transform")

    try:
        io_logger.info("reading input=%s", INPUT_PATH)
        input_text = INPUT_PATH.read_text(encoding="utf-8")
        input_lines = input_text.splitlines(keepends=True)
        io_logger.debug("loaded input lines=%d characters=%d", len(input_lines), len(input_text))

        audit_payload = json.loads(VERSION_AUDIT_PATH.read_text(encoding="utf-8"))
        version_pairs = audit_payload.get("version_pairs", [])
        io_logger.info(
            "version pairs in audit=%d high_confidence=%d",
            len(version_pairs),
            sum(1 for pair in version_pairs if pair.get("confidence") == CONFIDENCE_TARGET),
        )

        high_pairs = normalize_remove_candidates(version_pairs)
        for pair in high_pairs:
            if pair["recommended_remove_start_line"] == pair["recommended_keep_start_line"]:
                raise ValueError(
                    "keep and remove lines are identical",
                    {
                        "line": pair["recommended_remove_start_line"],
                        "pair": pair,
                    },
                )

        raw_ranges = build_remove_ranges(input_lines, high_pairs)
        ranges = merge_overlapping_ranges(raw_ranges)

        kept_text, removed_lines, removed_characters, removed_sections = remove_ranges(
            input_lines,
            ranges,
        )

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(kept_text, encoding="utf-8")
        retained_ratio = len(kept_text) / len(input_text) if input_text else 1.0
        report = {
            "input_path": str(INPUT_PATH),
            "version_audit_path": str(VERSION_AUDIT_PATH),
            "output_path": str(OUTPUT_PATH),
            "input_sha256": sha256_text(input_text),
            "output_sha256": sha256_text(kept_text),
            "input_lines": len(input_lines),
            "output_lines": count_lines(kept_text),
            "input_characters": len(input_text),
            "output_characters": len(kept_text),
            "characters_removed": len(input_text) - len(kept_text),
            "retained_ratio": retained_ratio,
            "input_section_count": audit_payload.get("chapter_section_count"),
            "audit_confidence_target": CONFIDENCE_TARGET,
            "high_confidence_pairs": len(high_pairs),
            "removed_sections": removed_sections,
            "removed_line_count": removed_lines,
            "removed_characters": removed_characters,
            "range_count": len(ranges),
            "removed_lines": [
                {"start": item["line_range"][0] + 1, "end_exclusive": item["line_range"][1]}
                for item in ranges
            ],
        }
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        transform_logger.info(
            "removed_sections=%d removed_lines=%d removed_characters=%d retained_ratio=%.6f",
            len(ranges),
            removed_lines,
            removed_characters,
            retained_ratio,
        )
        io_logger.info(
            "wrote output=%s report=%s",
            OUTPUT_PATH,
            REPORT_PATH,
        )
    except Exception:
        io_logger.exception("duplicate version cleaning failed input=%s", INPUT_PATH)
        raise


if __name__ == "__main__":
    main()
