"""Prepare a review-gated, chapter-level v4 corpus from stage3.

The stage3 source is always read-only.  Until every ambiguous chapter-version
pair has an explicit human decision, this program writes preview artifacts and
returns exit code 2.  It publishes formal cloud inputs only after the review
gate and all integrity checks pass.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import sys
import tempfile
import unicodedata
from typing import Any, Iterable

from audit_chapter_versions import find_boundaries
from audit_corpus import CHAPTER_PATTERN, classify_candidate, normalize_heading
from clean_content import INLINE_RULES, matched_ad_rules


DEFAULT_INPUT = Path("data/clean/doupo_stage3.txt")
DEFAULT_VERSION_AUDIT = Path("data/clean/chapter_version_audit_stage3.json")
DEFAULT_WORK_DIR = Path("data/clean/v4")
DEFAULT_CLOUD_DIR = Path("data/cloud_v4")
DEFAULT_LOG_DIR = Path("logs")
SPLIT_RATIOS = {"train": 0.90, "val": 0.05, "test": 0.05}
REVIEW_DECISIONS = {
    "pending",
    "keep_both",
    "keep_first",
    "keep_second",
    "remove_range",
}
SAFE_CONTROL_WHITESPACE = {"\n", "\t"}
SCHEMA_VERSION = "1.0"

CHINESE_CHAPTER_NUMBER = r"[零〇一二两三四五六七八九十百千万0-9]+"
KNOWN_MISSING_HEADING_PARTS = {
    "第八十七 下杀手": "第八十七章 下杀手",
    "地两百四十二章 石漠城的变故": "第二百四十二章 石漠城的变故",
    "地两百四十三章 击杀大斗师！": "第二百四十三章 击杀大斗师！",
    "第两百七十四 章 米特尔拍卖场，故人": "第两百七十四章 米特尔拍卖场，故人",
    "第两百七十四 章 米特尔拍卖场，故人【二合一！】": "第两百七十四章 米特尔拍卖场，故人【二合一！】",
    "第三百三十七章 一招": "第三百七十七章 一招",
    "第三百三十八章  杀鸡儆猴": "第三百七十八章 杀鸡儆猴",
    "第三百三十九章  黑夜中的对碰": "第三百七十九章 黑夜中的对碰",
    "第三百四十章 初次交锋": "第三百八十章 初次交锋",
    "第三百三十八章 扑朔迷离": "第三百八十一章 扑朔迷离",
    "第三百三十九章  劲敌": "第三百八十二章 劲敌",
    "四百七十章  比试【第三更！】": "第四百七十章 比试【第三更！】",
    "第五百五十六章  残卷焚决【第二更！】": "第五百六十六章 残卷焚决【第二更！】",
    "第五百五十六章 残卷焚决": "第五百六十六章 残卷焚决",
    "第五百五十六章  残卷焚决": "第五百六十六章 残卷焚决",
    "第一千两百八十五章   龙皇，紫研": "第一千两百八十四章 龙皇，紫研",
    "第一千两百八十五章 龙皇，紫研": "第一千两百八十四章 龙皇，紫研",
    "第一千两百八十五章龙皇，紫研": "第一千两百八十四章 龙皇，紫研",
    "第一千三百三十四章 现身": "第一千三百三十五章 现身",
    "第一千三百三十五章现身": "第一千三百三十五章 现身",
    "第一千三百四十五章   斩杀": "第一千三百四十四章 斩杀",
    "第一千三百四十五章 斩杀": "第一千三百四十四章 斩杀",
    "第一千三百四十五  离开天墓": "第一千三百四十五章 离开天墓",
    "第一千四三十四章   古龙一族情势": "第一千四百三十四章 古龙一族情势",
    "第一千四三十四章 古龙一族情势": "第一千四百三十四章 古龙一族情势",
    "第一千一百四十五章 　　妖圣精血": "第一千四百四十五章 妖圣精血",
    "第一千一百四十五章 妖圣精血": "第一千四百四十五章 妖圣精血",
}
OCR_DI_PATTERN = re.compile(
    rf"^地({CHINESE_CHAPTER_NUMBER}章(?:\s*).+)$"
)
SPACE_BEFORE_CHAPTER_PATTERN = re.compile(
    rf"^第({CHINESE_CHAPTER_NUMBER})\s+章\s*(.+)$"
)
AUTHOR_NOISE_BLOCK_PREFIXES = (
    "更新换月票",
    "四章已更",
    "三章已更",
    "收到起点的锦书",
    "恭喜斗破成为起点",
    "努力到现在。",
    "愿望终达成。",
    "两年三个月，五百三十万。",
    "新书已发",
    "今天晚上七点半",
    "感言。",
    "新世界",
    "新书大主宰已发。",
)
METADATA_BLOCK_TITLES = {"正文", "VIP卷"}


class CorpusV4Error(RuntimeError):
    """A preparation error with a machine-actionable reason."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        remediation: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.remediation = remediation
        self.context = context or {}

    def as_reason(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "remediation": self.remediation,
            "context": self.context,
        }


@dataclass
class Chapter:
    section_id: str
    section_index: int
    chapter_number: int
    title: str
    start_line: int
    end_line: int
    title_offset: int
    source_text: str
    working_text: str = ""
    cleaned_text: str = ""

    @property
    def source_sha256(self) -> str:
        return sha256_text(self.source_text)

    @property
    def cleaned_sha256(self) -> str:
        return sha256_text(self.cleaned_text)

    @property
    def range_start_line(self) -> int:
        return self.start_line - self.title_offset


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        keep, merge = sorted((first_root, second_root))
        self.parent[merge] = keep


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )


def _log_level(environment_name: str, default: str = "INFO") -> int:
    configured = os.getenv(environment_name, default).upper()
    if configured == "OFF":
        return logging.CRITICAL + 1
    level = getattr(logging, configured, None)
    if not isinstance(level, int):
        raise CorpusV4Error(
            "INVALID_LOG_LEVEL",
            f"{environment_name} has unsupported value {configured!r}",
            remediation="Use DEBUG, INFO, WARNING, ERROR, CRITICAL, or OFF.",
        )
    return level


def configure_logging(log_dir: Path) -> dict[str, logging.Logger]:
    """Configure independently filterable data, corpus, and split logs."""

    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    categories = {
        "data": "CORPUS_V4_DATA_LOG_LEVEL",
        "corpus": "CORPUS_V4_CORPUS_LOG_LEVEL",
        "split": "CORPUS_V4_SPLIT_LOG_LEVEL",
    }
    loggers: dict[str, logging.Logger] = {}

    for category, environment_name in categories.items():
        logger = logging.getLogger(f"corpus_v4.{category}")
        for existing_handler in logger.handlers:
            existing_handler.close()
        logger.handlers.clear()
        logger.setLevel(_log_level(environment_name))
        logger.propagate = False

        file_handler = RotatingFileHandler(
            log_dir / f"corpus_v4_{category}.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        loggers[category] = logger

    return loggers


def remove_unsafe_control_characters(text: str) -> tuple[str, dict[str, int]]:
    counts: Counter[str] = Counter()
    output: list[str] = []
    for character in text:
        if (
            unicodedata.category(character) == "Cc"
            and character not in SAFE_CONTROL_WHITESPACE
        ):
            counts[f"U+{ord(character):04X}"] += 1
            continue
        output.append(character)
    return "".join(output), dict(sorted(counts.items()))


def repair_known_chapter_headings(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Repair only manually audited heading typos while preserving line numbers."""

    output: list[str] = []
    repairs: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
        content = line.rstrip("\r\n")
        ending = line[len(content) :]
        leading = content[: len(content) - len(content.lstrip())]
        stripped = content.strip()
        replacement = KNOWN_MISSING_HEADING_PARTS.get(stripped)
        if replacement is None:
            output.append(line)
            continue
        output.append(f"{leading}{replacement}{ending}")
        repairs.append(
            {
                "line": line_number,
                "before": stripped,
                "after": replacement,
            }
        )
    return "".join(output), repairs


def remove_known_non_story_blocks(
    text: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Blank separator-backed metadata/author posts while preserving line numbers."""

    lines = text.splitlines(keepends=True)
    separators = [
        index for index, line in enumerate(lines) if line.strip() == "------------"
    ]
    removals: list[dict[str, Any]] = []
    drop_indices: set[int] = set()
    for position, separator_index in enumerate(separators):
        next_separator = (
            separators[position + 1]
            if position + 1 < len(separators)
            else len(lines)
        )
        first_content_index = next(
            (
                index
                for index in range(separator_index + 1, next_separator)
                if lines[index].strip()
            ),
            None,
        )
        if first_content_index is None:
            continue
        title = lines[first_content_index].strip()
        is_metadata = title in METADATA_BLOCK_TITLES
        is_author_post = any(
            title.startswith(prefix) for prefix in AUTHOR_NOISE_BLOCK_PREFIXES
        )
        if not (is_metadata or is_author_post):
            continue
        drop_indices.update(range(separator_index, next_separator))
        removals.append(
            {
                "start_line": separator_index + 1,
                "end_line": next_separator,
                "title": title,
                "kind": "metadata" if is_metadata else "author_post",
            }
        )

    output: list[str] = []
    for index, line in enumerate(lines):
        if index not in drop_indices:
            output.append(line)
            continue
        if line.endswith("\r\n"):
            output.append("\r\n")
        elif line.endswith("\n"):
            output.append("\n")
        else:
            output.append("")
    return "".join(output), removals


def parse_complete_chapters(text: str) -> tuple[str, list[Chapter]]:
    lines = text.splitlines(keepends=True)
    plain_lines = [line.rstrip("\r\n") for line in lines]
    boundaries = find_boundaries(plain_lines)
    if not boundaries:
        raise CorpusV4Error(
            "NO_CHAPTERS_PARSED",
            "No separator-backed chapter headings were found.",
            remediation="Verify stage3 formatting and CHAPTER_PATTERN before continuing.",
        )

    chapters: list[Chapter] = []
    for index, boundary in enumerate(boundaries):
        range_start = boundary["separator_index"]
        range_end = (
            boundaries[index + 1]["separator_index"]
            if index + 1 < len(boundaries)
            else len(lines)
        )
        heading_index = boundary["heading_index"]
        title = plain_lines[heading_index].strip()
        chapter_text = "".join(lines[range_start:range_end])
        chapters.append(
            Chapter(
                section_id=f"section-{index + 1:04d}-line-{heading_index + 1}",
                section_index=index,
                chapter_number=boundary["chapter_number"],
                title=title,
                start_line=heading_index + 1,
                end_line=range_end,
                title_offset=heading_index - range_start,
                source_text=chapter_text,
            )
        )

    preamble = "".join(lines[: boundaries[0]["separator_index"]])
    return preamble, chapters


def remove_adjacent_duplicate_titles(
    lines: list[str],
    title_offset: int,
) -> tuple[list[str], dict[str, int]]:
    if not 0 <= title_offset < len(lines):
        raise CorpusV4Error(
            "INVALID_TITLE_OFFSET",
            "The parsed chapter title offset is outside its section.",
            remediation="Inspect the chapter boundary parser and stage3 formatting.",
            context={"title_offset": title_offset, "line_count": len(lines)},
        )

    primary_title = lines[title_offset].strip()
    if not CHAPTER_PATTERN.match(primary_title):
        raise CorpusV4Error(
            "INVALID_PRIMARY_TITLE",
            "A parsed chapter does not begin with a valid chapter title.",
            remediation="Inspect the reported source line before freezing v4.",
            context={"title": primary_title},
        )

    seen = {normalize_heading(primary_title)}
    remove_indices: set[int] = set()
    removed_characters = 0

    for index in range(title_offset + 1, len(lines)):
        stripped = lines[index].strip()
        if not stripped:
            continue
        if not CHAPTER_PATTERN.match(stripped):
            break
        normalized = normalize_heading(stripped)
        if normalized in seen:
            remove_indices.add(index)
            removed_characters += len(lines[index])
        else:
            seen.add(normalized)

    output = [line for index, line in enumerate(lines) if index not in remove_indices]
    return output, {
        "adjacent_duplicate_title_lines_removed": len(remove_indices),
        "adjacent_duplicate_title_characters_removed": removed_characters,
    }


def clean_known_site_noise(lines: list[str]) -> tuple[list[str], dict[str, Any]]:
    output: list[str] = []
    whole_line_counts: Counter[str] = Counter()
    inline_counts: Counter[str] = Counter()
    unresolved: list[dict[str, Any]] = []
    removed_characters = 0
    whole_lines_removed = 0

    for line_index, line in enumerate(lines):
        original_length = len(line)
        matched_rules = matched_ad_rules(line)
        if matched_rules and classify_candidate(line.strip(), matched_rules) == "whole_line":
            whole_line_counts.update(matched_rules)
            removed_characters += original_length
            whole_lines_removed += 1
            continue

        cleaned_line = line
        for rule_name, pattern in INLINE_RULES.items():
            cleaned_line, replacement_count = pattern.subn("", cleaned_line)
            if replacement_count:
                inline_counts[rule_name] += replacement_count

        removed_characters += original_length - len(cleaned_line)
        remaining_rules = matched_ad_rules(cleaned_line)
        if remaining_rules:
            unresolved.append(
                {
                    "section_line_offset": line_index,
                    "rules": remaining_rules,
                }
            )
        output.append(cleaned_line)

    return output, {
        "known_site_noise_whole_lines_removed": whole_lines_removed,
        "known_site_noise_whole_line_rule_counts": dict(whole_line_counts),
        "known_site_noise_inline_replacements": sum(inline_counts.values()),
        "known_site_noise_inline_rule_counts": dict(inline_counts),
        "known_site_noise_characters_removed": removed_characters,
        "unresolved_site_noise_count": len(unresolved),
        "unresolved_site_noise": unresolved,
    }


def _add_numeric_statistics(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, int):
            target[key] = target.get(key, 0) + value
        elif isinstance(value, dict) and all(
            isinstance(nested, int) for nested in value.values()
        ):
            combined = Counter(target.get(key, {}))
            combined.update(value)
            target[key] = dict(combined)
        elif isinstance(value, list):
            target.setdefault(key, []).extend(value)


def clean_chapter(chapter: Chapter) -> dict[str, Any]:
    source = chapter.working_text or chapter.source_text
    lines = source.splitlines(keepends=True)
    deduplicated_lines, title_stats = remove_adjacent_duplicate_titles(
        lines,
        chapter.title_offset,
    )
    cleaned_lines, noise_stats = clean_known_site_noise(deduplicated_lines)
    chapter.cleaned_text = "".join(cleaned_lines)
    return {**title_stats, **noise_stats}


def source_range_text(chapter: Chapter, start_line: int, end_line: int) -> str:
    """Return an inclusive original-line range within one parsed section."""

    lines = chapter.source_text.splitlines(keepends=True)
    start_offset = start_line - chapter.range_start_line
    end_offset = end_line - chapter.range_start_line + 1
    if start_offset < 0 or end_offset > len(lines) or start_offset >= end_offset:
        raise CorpusV4Error(
            "REMOVE_RANGE_OUTSIDE_SECTION",
            "A requested local removal range is outside its chapter section.",
            remediation="Use source lines contained in the selected manifest side.",
            context={
                "section_id": chapter.section_id,
                "section_range": [chapter.range_start_line, chapter.end_line],
                "requested_range": [start_line, end_line],
            },
        )
    return "".join(lines[start_offset:end_offset])


def apply_local_remove_ranges(
    chapter: Chapter,
    ranges: list[dict[str, Any]],
) -> dict[str, int]:
    """Apply SHA-pinned, human-approved ranges without changing source_text."""

    lines = chapter.source_text.splitlines(keepends=True)
    drop = [False] * len(lines)
    removed_characters = 0
    removed_lines = 0
    for item in sorted(ranges, key=lambda value: value["start_line"]):
        start_offset = item["start_line"] - chapter.range_start_line
        end_offset = item["end_line"] - chapter.range_start_line + 1
        if any(drop[start_offset:end_offset]):
            raise CorpusV4Error(
                "OVERLAPPING_REMOVE_RANGES",
                "Two approved local removal ranges overlap.",
                remediation="Merge or correct the overlapping manual ranges.",
                context={"section_id": chapter.section_id},
            )
        for index in range(start_offset, end_offset):
            drop[index] = True
            removed_characters += len(lines[index])
            removed_lines += 1
    chapter.working_text = "".join(
        line for index, line in enumerate(lines) if not drop[index]
    )
    return {
        "manual_review_range_lines_removed": removed_lines,
        "manual_review_range_characters_removed": removed_characters,
    }


def make_pair_id(audit_sha256: str, pair: dict[str, Any]) -> str:
    identity = "|".join(
        (
            audit_sha256,
            str(pair["first"]["start_line"]),
            str(pair["second"]["start_line"]),
            str(pair.get("matched_characters", "")),
        )
    )
    return f"pair-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"


def build_review_manifest(
    audit_payload: dict[str, Any],
    audit_sha256: str,
    chapters_by_start: dict[int, Chapter],
) -> tuple[dict[str, Any], dict[str, tuple[str, str]]]:
    entries: list[dict[str, Any]] = []
    pair_sections: dict[str, tuple[str, str]] = {}

    for pair in audit_payload.get("version_pairs", []):
        first_line = pair["first"]["start_line"]
        second_line = pair["second"]["start_line"]
        if first_line not in chapters_by_start or second_line not in chapters_by_start:
            raise CorpusV4Error(
                "AUDIT_SECTION_NOT_FOUND",
                "A version-audit pair references a section absent from stage3.",
                remediation="Regenerate chapter_version_audit_stage3.json from the same stage3 file.",
                context={"first_start_line": first_line, "second_start_line": second_line},
            )
        first_chapter = chapters_by_start[first_line]
        second_chapter = chapters_by_start[second_line]
        pair_id = make_pair_id(audit_sha256, pair)
        pair_sections[pair_id] = (first_chapter.section_id, second_chapter.section_id)

        def public_side(chapter: Chapter, side: dict[str, Any]) -> dict[str, Any]:
            return {
                "section_id": chapter.section_id,
                "chapter_number": chapter.chapter_number,
                "title": chapter.title,
                "start_line": chapter.start_line,
                "end_line": chapter.end_line,
                "source_characters": len(chapter.source_text),
                "source_sha256": chapter.source_sha256,
                "audit_body_characters": side.get("body_characters"),
                "audit_quality_score": side.get("quality_score"),
            }

        entries.append(
            {
                "pair_id": pair_id,
                "review_status": "pending",
                "confidence": pair.get("confidence"),
                "same_chapter_number": pair.get("same_chapter_number"),
                "matched_characters": pair.get("matched_characters"),
                "shared_coverage": pair.get("shared_coverage"),
                "matching_line_ratio": pair.get("matching_line_ratio"),
                "recommended_keep_start_line": pair.get("recommended_keep_start_line"),
                "recommended_remove_start_line": pair.get("recommended_remove_start_line"),
                "first": public_side(first_chapter, pair["first"]),
                "second": public_side(second_chapter, pair["second"]),
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_audit_sha256": audit_sha256,
        "pair_count": len(entries),
        "allowed_decisions": sorted(REVIEW_DECISIONS - {"pending"}),
        "instructions": (
            "Record one explicit decision and reviewer for every pair in "
            "chapter_version_resolutions.json; do not edit this generated manifest."
        ),
        "pairs": entries,
    }
    return manifest, pair_sections


def write_resolution_template_if_missing(
    path: Path,
    manifest: dict[str, Any],
) -> None:
    if path.exists():
        return
    atomic_write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "source_audit_sha256": manifest["source_audit_sha256"],
            "instructions": (
                "Choose keep_both, keep_first, keep_second, or remove_range and add "
                "a reviewer. remove_range also requires side, start_line, end_line, "
                "and range_sha256. Pending entries keep the corpus frozen."
            ),
            "resolutions": [
                {
                    "pair_id": pair["pair_id"],
                    "decision": "pending",
                    "reviewer": "",
                    "note": "",
                }
                for pair in manifest["pairs"]
            ],
        },
    )


def load_and_validate_resolutions(
    path: Path,
    manifest: dict[str, Any],
    chapters_by_id: dict[str, Chapter],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CorpusV4Error(
            "INVALID_RESOLUTION_FILE",
            "The chapter-version resolution file cannot be read as JSON.",
            remediation="Repair chapter_version_resolutions.json, then rerun preparation.",
            context={"path": str(path), "error_type": type(error).__name__},
        ) from error

    if payload.get("source_audit_sha256") != manifest["source_audit_sha256"]:
        raise CorpusV4Error(
            "STALE_RESOLUTION_FILE",
            "The resolution file belongs to a different version-audit input.",
            remediation="Review the newly generated manifest and rebuild the resolution file.",
            context={
                "expected": manifest["source_audit_sha256"],
                "actual": payload.get("source_audit_sha256"),
            },
        )

    expected_ids = {pair["pair_id"] for pair in manifest["pairs"]}
    resolutions: dict[str, dict[str, Any]] = {}
    for item in payload.get("resolutions", []):
        pair_id = item.get("pair_id")
        if pair_id in resolutions:
            raise CorpusV4Error(
                "DUPLICATE_RESOLUTION",
                f"Resolution {pair_id!r} appears more than once.",
                remediation="Keep exactly one resolution row per manifest pair.",
            )
        if pair_id not in expected_ids:
            raise CorpusV4Error(
                "UNKNOWN_RESOLUTION_PAIR",
                f"Resolution {pair_id!r} is not present in the generated manifest.",
                remediation="Remove stale rows and use pair IDs from the current manifest.",
            )
        decision = item.get("decision", "pending")
        if decision not in REVIEW_DECISIONS:
            raise CorpusV4Error(
                "INVALID_REVIEW_DECISION",
                f"Resolution {pair_id!r} has invalid decision {decision!r}.",
                remediation=(
                    "Use pending, keep_both, keep_first, keep_second, or remove_range."
                ),
            )
        if decision != "pending" and not str(item.get("reviewer", "")).strip():
            raise CorpusV4Error(
                "MISSING_REVIEWER",
                f"Resolution {pair_id!r} has a decision but no reviewer.",
                remediation="Record who made each manual review decision.",
            )
        if decision == "remove_range":
            manifest_pair = next(
                pair for pair in manifest["pairs"] if pair["pair_id"] == pair_id
            )
            if manifest_pair.get("same_chapter_number") is not False:
                raise CorpusV4Error(
                    "REMOVE_RANGE_NOT_CROSS_CHAPTER",
                    "remove_range is reserved for a reviewed cross-chapter merge.",
                    remediation="Use keep_first/keep_second for ordinary same-chapter versions.",
                    context={"pair_id": pair_id},
                )
            side = item.get("side")
            if side not in {"first", "second"}:
                raise CorpusV4Error(
                    "INVALID_REMOVE_RANGE_SIDE",
                    "A remove_range decision must identify first or second.",
                    remediation="Set side to the manifest side containing the bad range.",
                    context={"pair_id": pair_id},
                )
            section_id = manifest_pair[side]["section_id"]
            chapter = chapters_by_id[section_id]
            try:
                start_line = int(item["start_line"])
                end_line = int(item["end_line"])
            except (KeyError, TypeError, ValueError) as error:
                raise CorpusV4Error(
                    "INVALID_REMOVE_RANGE_LINES",
                    "A remove_range decision needs integer start_line and end_line.",
                    remediation="Copy the inclusive source line range from the review report.",
                    context={"pair_id": pair_id},
                ) from error
            selected_text = source_range_text(chapter, start_line, end_line)
            expected_range_sha256 = sha256_text(selected_text)
            if item.get("range_sha256") != expected_range_sha256:
                raise CorpusV4Error(
                    "REMOVE_RANGE_HASH_MISMATCH",
                    "The approved line range no longer matches its recorded SHA-256.",
                    remediation="Re-audit this source range and record its current SHA-256.",
                    context={"pair_id": pair_id},
                )
        resolutions[pair_id] = item

    unresolved = sorted(
        pair_id
        for pair_id in expected_ids
        if resolutions.get(pair_id, {}).get("decision", "pending") == "pending"
    )
    return resolutions, unresolved


def reviewed_actions(
    resolutions: dict[str, dict[str, Any]],
    pair_sections: dict[str, tuple[str, str]],
    unresolved: list[str],
    manifest: dict[str, Any],
) -> tuple[set[str], dict[str, list[dict[str, Any]]]]:
    if unresolved:
        return set(), {}

    remove: set[str] = set()
    explicitly_keep: set[str] = set()
    local_ranges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    manifest_by_id = {pair["pair_id"]: pair for pair in manifest["pairs"]}
    for pair_id, (first_id, second_id) in pair_sections.items():
        decision = resolutions[pair_id]["decision"]
        if decision == "keep_both":
            explicitly_keep.update((first_id, second_id))
        elif decision == "keep_first":
            explicitly_keep.add(first_id)
            remove.add(second_id)
        elif decision == "keep_second":
            explicitly_keep.add(second_id)
            remove.add(first_id)
        elif decision == "remove_range":
            side = resolutions[pair_id]["side"]
            section_id = manifest_by_id[pair_id][side]["section_id"]
            explicitly_keep.update((first_id, second_id))
            local_ranges[section_id].append(
                {
                    "start_line": int(resolutions[pair_id]["start_line"]),
                    "end_line": int(resolutions[pair_id]["end_line"]),
                    "range_sha256": resolutions[pair_id]["range_sha256"],
                }
            )

    conflicts = sorted(remove & explicitly_keep)
    if conflicts:
        raise CorpusV4Error(
            "CONFLICTING_REVIEW_DECISIONS",
            "Manual decisions both keep and remove the same chapter section.",
            remediation="Resolve the connected version-pair decisions consistently.",
            context={"section_ids": conflicts},
        )
    return remove, dict(local_ranges)


def build_version_groups(
    chapters: list[Chapter],
    pair_sections: dict[str, tuple[str, str]],
) -> list[list[Chapter]]:
    chapters_by_id = {chapter.section_id: chapter for chapter in chapters}
    union_find = UnionFind(chapters_by_id)
    by_chapter_number: dict[int, list[Chapter]] = defaultdict(list)
    for chapter in chapters:
        by_chapter_number[chapter.chapter_number].append(chapter)

    for same_number in by_chapter_number.values():
        for chapter in same_number[1:]:
            union_find.union(same_number[0].section_id, chapter.section_id)

    for first_id, second_id in pair_sections.values():
        if first_id in chapters_by_id and second_id in chapters_by_id:
            union_find.union(first_id, second_id)

    grouped: dict[str, list[Chapter]] = defaultdict(list)
    for chapter in chapters:
        grouped[union_find.find(chapter.section_id)].append(chapter)
    return [
        sorted(group, key=lambda chapter: chapter.section_index)
        for _, group in sorted(grouped.items())
    ]


def split_groups(
    groups: list[list[Chapter]],
    *,
    source_sha256: str,
    seed: int,
) -> tuple[dict[str, list[Chapter]], dict[str, str]]:
    if len(groups) < 3:
        raise CorpusV4Error(
            "INSUFFICIENT_SPLIT_GROUPS",
            "At least three independent chapter groups are required for 90/5/5 splits.",
            remediation="Provide more complete chapters or revise the grouping audit.",
            context={"group_count": len(groups)},
        )

    def group_key(group: list[Chapter]) -> str:
        identity = ",".join(chapter.section_id for chapter in group)
        return hashlib.sha256(
            f"{source_sha256}|{seed}|{identity}".encode("utf-8")
        ).hexdigest()

    ordered_groups = sorted(groups, key=group_key)
    total_characters = sum(
        len(chapter.cleaned_text) for group in ordered_groups for chapter in group
    )
    train_target = total_characters * SPLIT_RATIOS["train"]
    validation_boundary = total_characters * (
        SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"]
    )
    cumulative = 0
    group_split: dict[str, str] = {}

    for index, group in enumerate(ordered_groups):
        remaining_after = len(ordered_groups) - index - 1
        if remaining_after < 2:
            split = "val" if remaining_after == 1 else "test"
        elif cumulative < train_target:
            split = "train"
        elif cumulative < validation_boundary:
            split = "val"
        else:
            split = "test"
        for chapter in group:
            group_split[chapter.section_id] = split
        cumulative += sum(len(chapter.cleaned_text) for chapter in group)

    split_chapters = {
        split: sorted(
            [chapter for group in groups for chapter in group if group_split[chapter.section_id] == split],
            key=lambda chapter: chapter.section_index,
        )
        for split in SPLIT_RATIOS
    }
    if any(not split_chapters[split] for split in SPLIT_RATIOS):
        raise CorpusV4Error(
            "EMPTY_DATA_SPLIT",
            "The deterministic chapter grouping produced an empty split.",
            remediation="Inspect group sizes or provide more chapter groups.",
            context={split: len(items) for split, items in split_chapters.items()},
        )
    return split_chapters, group_split


def assert_no_group_leakage(
    groups: list[list[Chapter]],
    group_split: dict[str, str],
) -> None:
    leaking_groups = []
    for group in groups:
        splits = {group_split[chapter.section_id] for chapter in group}
        if len(splits) != 1:
            leaking_groups.append([chapter.section_id for chapter in group])
    if leaking_groups:
        raise CorpusV4Error(
            "CHAPTER_VERSION_SPLIT_LEAKAGE",
            "One or more same-chapter/version groups cross data splits.",
            remediation="Fix grouped splitting before using any v4 output.",
            context={"leaking_groups": leaking_groups[:20]},
        )


def join_chapters(chapters: list[Chapter]) -> str:
    return "".join(chapter.cleaned_text for chapter in chapters)


def _split_statistics(
    split_chapters: dict[str, list[Chapter]],
    split_texts: dict[str, str],
    total_characters: int,
) -> dict[str, Any]:
    return {
        split: {
            "chapter_section_count": len(split_chapters[split]),
            "unique_chapter_number_count": len(
                {chapter.chapter_number for chapter in split_chapters[split]}
            ),
            "characters": len(split_texts[split]),
            "character_ratio": (
                len(split_texts[split]) / total_characters if total_characters else 0.0
            ),
            "sha256": sha256_text(split_texts[split]),
            "first_source_line": min(chapter.start_line for chapter in split_chapters[split]),
            "last_source_line": max(chapter.end_line for chapter in split_chapters[split]),
        }
        for split in SPLIT_RATIOS
    }


def _failure_reason_for_unresolved(
    unresolved: list[str],
    resolution_path: Path,
) -> dict[str, Any]:
    return {
        "code": "UNRESOLVED_CHAPTER_VERSION_REVIEWS",
        "message": f"{len(unresolved)} chapter-version pairs still require human decisions.",
        "remediation": (
            f"Edit {resolution_path}: choose keep_both, keep_first, keep_second, or "
            "a SHA-pinned remove_range "
            "and record a reviewer for every pair, then rerun."
        ),
        "context": {
            "unresolved_count": len(unresolved),
            "unresolved_pair_ids": unresolved,
        },
    }


def write_cloud_manifest_with_sidecar(
    cloud_dir: Path,
    payload: dict[str, Any],
) -> None:
    manifest_path = cloud_dir / "corpus_manifest.json"
    atomic_write_json(manifest_path, payload)
    manifest_sha256 = sha256_file(manifest_path)
    atomic_write_text(
        Path(f"{manifest_path}.sha256"),
        f"{manifest_sha256}  {manifest_path.name}\n",
    )


def prepare_corpus_v4(
    *,
    input_path: Path = DEFAULT_INPUT,
    version_audit_path: Path = DEFAULT_VERSION_AUDIT,
    work_dir: Path = DEFAULT_WORK_DIR,
    cloud_dir: Path = DEFAULT_CLOUD_DIR,
    seed: int = 42,
    loggers: dict[str, logging.Logger] | None = None,
) -> dict[str, Any]:
    loggers = loggers or {
        name: logging.getLogger(f"corpus_v4.{name}")
        for name in ("data", "corpus", "split")
    }
    data_logger = loggers["data"]
    corpus_logger = loggers["corpus"]
    split_logger = loggers["split"]

    reports_dir = work_dir / "reports"
    preview_dir = work_dir / "preview"
    manifest_path = reports_dir / "chapter_version_review_manifest.json"
    resolution_path = work_dir / "chapter_version_resolutions.json"
    report_path = reports_dir / "corpus_v4_report.json"
    status_path = reports_dir / "freeze_status.json"

    try:
        source_bytes = input_path.read_bytes()
        source_text = source_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise CorpusV4Error(
            "INPUT_READ_FAILED",
            "The stage3 input cannot be read as UTF-8.",
            remediation="Restore the expected stage3 file and verify its encoding.",
            context={"path": str(input_path), "error_type": type(error).__name__},
        ) from error
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    try:
        audit_bytes = version_audit_path.read_bytes()
        audit_payload = json.loads(audit_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusV4Error(
            "VERSION_AUDIT_READ_FAILED",
            "The stage3 chapter-version audit cannot be read as UTF-8 JSON.",
            remediation="Regenerate chapter_version_audit_stage3.json from stage3.",
            context={
                "path": str(version_audit_path),
                "error_type": type(error).__name__,
            },
        ) from error
    audit_sha256 = hashlib.sha256(audit_bytes).hexdigest()
    data_logger.info(
        "loaded stage3 path=%s characters=%d sha256=%s",
        input_path,
        len(source_text),
        source_sha256,
    )

    control_cleaned_text, control_counts = remove_unsafe_control_characters(source_text)
    story_text, non_story_removals = remove_known_non_story_blocks(
        control_cleaned_text
    )
    _, baseline_chapters = parse_complete_chapters(story_text)
    expected_section_count = audit_payload.get("chapter_section_count")
    if expected_section_count is not None and expected_section_count != len(baseline_chapters):
        raise CorpusV4Error(
            "AUDIT_SECTION_COUNT_MISMATCH",
            "The version audit and stage3 chapter parser disagree on section count.",
            remediation="Regenerate the stage3 version audit from the unchanged stage3 input.",
            context={"audit": expected_section_count, "parsed": len(baseline_chapters)},
        )

    repaired_text, heading_repairs = repair_known_chapter_headings(story_text)
    preamble, chapters = parse_complete_chapters(repaired_text)

    aggregate_cleaning: dict[str, Any] = {
        "control_characters_removed": sum(control_counts.values()),
        "control_character_counts": control_counts,
        "known_heading_lines_repaired": len(heading_repairs),
        "known_heading_repairs": heading_repairs,
        "known_non_story_blocks_removed": len(non_story_removals),
        "known_non_story_block_characters_removed": sum(
            len(control_cleaned_text.splitlines(keepends=True)[line_index - 1])
            for item in non_story_removals
            for line_index in range(item["start_line"], item["end_line"] + 1)
        ),
        "known_non_story_blocks": non_story_removals,
    }
    preamble_lines, preamble_stats = clean_known_site_noise(
        preamble.splitlines(keepends=True)
    )
    cleaned_preamble = "".join(preamble_lines)
    _add_numeric_statistics(aggregate_cleaning, preamble_stats)

    chapters_by_start = {chapter.start_line: chapter for chapter in chapters}
    if len(chapters_by_start) != len(chapters):
        raise CorpusV4Error(
            "DUPLICATE_SECTION_START_LINE",
            "Two parsed chapter sections share a source start line.",
            remediation="Inspect stage3 boundary parsing before continuing.",
        )
    review_manifest, pair_sections = build_review_manifest(
        audit_payload,
        audit_sha256,
        chapters_by_start,
    )
    atomic_write_json(manifest_path, review_manifest)
    write_resolution_template_if_missing(resolution_path, review_manifest)
    resolutions, unresolved = load_and_validate_resolutions(
        resolution_path,
        review_manifest,
        {chapter.section_id: chapter for chapter in chapters},
    )
    removals, local_ranges = reviewed_actions(
        resolutions,
        pair_sections,
        unresolved,
        review_manifest,
    )
    for section_id, ranges in local_ranges.items():
        chapter = next(item for item in chapters if item.section_id == section_id)
        _add_numeric_statistics(
            aggregate_cleaning,
            apply_local_remove_ranges(chapter, ranges),
        )
    for chapter in chapters:
        _add_numeric_statistics(aggregate_cleaning, clean_chapter(chapter))
    corpus_logger.info(
        "safe cleaning complete sections=%d controls=%d heading_repairs=%d "
        "manual_range_lines=%d duplicate_titles=%d site_noise_lines=%d",
        len(chapters),
        aggregate_cleaning["control_characters_removed"],
        aggregate_cleaning["known_heading_lines_repaired"],
        aggregate_cleaning.get("manual_review_range_lines_removed", 0),
        aggregate_cleaning.get("adjacent_duplicate_title_lines_removed", 0),
        aggregate_cleaning.get("known_site_noise_whole_lines_removed", 0),
    )
    selected_chapters = [
        chapter for chapter in chapters if chapter.section_id not in removals
    ]

    groups = build_version_groups(selected_chapters, pair_sections)
    split_chapters, group_split = split_groups(
        groups,
        source_sha256=source_sha256,
        seed=seed,
    )
    assert_no_group_leakage(groups, group_split)
    split_texts = {
        split: join_chapters(split_chapters[split]) for split in SPLIT_RATIOS
    }
    preview_corpus = cleaned_preamble + join_chapters(selected_chapters)
    selected_characters = sum(len(text) for text in split_texts.values())
    split_stats = _split_statistics(
        split_chapters,
        split_texts,
        selected_characters,
    )

    for split, text in split_texts.items():
        atomic_write_text(preview_dir / f"{split}.txt", text)
    atomic_write_text(preview_dir / "corpus.txt", preview_corpus)
    split_logger.info(
        "deterministic grouped split complete seed=%d groups=%d train_chars=%d val_chars=%d test_chars=%d",
        seed,
        len(groups),
        len(split_texts["train"]),
        len(split_texts["val"]),
        len(split_texts["test"]),
    )

    failure_reasons: list[dict[str, Any]] = []
    if unresolved:
        failure_reasons.append(
            _failure_reason_for_unresolved(unresolved, resolution_path)
        )
    if aggregate_cleaning.get("unresolved_site_noise_count", 0):
        failure_reasons.append(
            {
                "code": "UNRESOLVED_SITE_NOISE",
                "message": (
                    f"{aggregate_cleaning['unresolved_site_noise_count']} possible site-noise "
                    "lines were preserved because automatic removal was not safe."
                ),
                "remediation": "Review the reported offsets and add narrowly tested rules.",
            }
        )

    ready = not failure_reasons
    status = "ready" if ready else "freeze_not_ready"
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "ready": ready,
        "source": {
            "path": str(input_path),
            "bytes": len(source_bytes),
            "characters": len(source_text),
            "lines": len(source_text.splitlines()),
            "sha256": source_sha256,
        },
        "version_audit": {
            "path": str(version_audit_path),
            "sha256": audit_sha256,
            "pair_count": review_manifest["pair_count"],
            "resolved_pair_count": review_manifest["pair_count"] - len(unresolved),
            "unresolved_pair_count": len(unresolved),
            "review_manifest_path": str(manifest_path),
            "resolution_path": str(resolution_path),
            "reviewed_sections_removed": len(removals),
            "reviewed_local_ranges_applied": sum(
                len(ranges) for ranges in local_ranges.values()
            ),
            "partial_decisions_applied_while_unresolved": False,
        },
        "cleaning": aggregate_cleaning,
        "chapters": {
            "parsed_section_count": len(chapters),
            "selected_section_count": len(selected_chapters),
            "version_group_count": len(groups),
            "unique_chapter_number_count": len(
                {chapter.chapter_number for chapter in selected_chapters}
            ),
            "preamble_characters": len(cleaned_preamble),
            "selected_chapter_characters": selected_characters,
            "preview_corpus_characters": len(preview_corpus),
            "preview_corpus_sha256": sha256_text(preview_corpus),
        },
        "split": {
            "method": "sha256-seeded version-group shuffle with whole-group boundaries",
            "seed": seed,
            "target_ratios": SPLIT_RATIOS,
            "same_chapter_versions_cross_split": False,
            "splits": split_stats,
        },
        "preview_paths": {
            "corpus": str(preview_dir / "corpus.txt"),
            **{split: str(preview_dir / f"{split}.txt") for split in SPLIT_RATIOS},
        },
        "cloud_paths": (
            {
                "corpus": str(cloud_dir / "corpus.txt"),
                **{
                    split: str(cloud_dir / f"{split}.txt")
                    for split in SPLIT_RATIOS
                },
            }
            if ready
            else {}
        ),
        "failure_reasons": failure_reasons,
    }

    if ready:
        atomic_write_text(cloud_dir / "corpus.txt", preview_corpus)
        for split, text in split_texts.items():
            atomic_write_text(cloud_dir / f"{split}.txt", text)
        artifacts = [
            {
                "path": str(cloud_dir / "corpus.txt"),
                "sha256": sha256_text(preview_corpus),
                "size_bytes": (cloud_dir / "corpus.txt").stat().st_size,
            },
            *[
            {
                "path": str(cloud_dir / f"{split}.txt"),
                "sha256": split_stats[split]["sha256"],
                "size_bytes": (cloud_dir / f"{split}.txt").stat().st_size,
            }
            for split in SPLIT_RATIOS
            ],
        ]
        cloud_manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "ready",
            "source_sha256": source_sha256,
            "version_audit_sha256": audit_sha256,
            "corpus_report_path": str(report_path),
            "split_method": report["split"]["method"],
            "split_seed": seed,
            "artifacts": artifacts,
            "splits": {
                split: {
                    "path": str(cloud_dir / f"{split}.txt"),
                    **split_stats[split],
                }
                for split in SPLIT_RATIOS
            },
        }
        write_cloud_manifest_with_sidecar(cloud_dir, cloud_manifest)
    else:
        # A blocker manifest invalidates any stale ready manifest without publishing
        # new formal training text. Cloud preflight must reject status != ready.
        write_cloud_manifest_with_sidecar(
            cloud_dir,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "freeze_not_ready",
                "source_sha256": source_sha256,
                "artifacts": [],
                "preview_report_path": str(report_path),
                "failure_reasons": failure_reasons,
            },
        )

    atomic_write_json(report_path, report)
    freeze_status = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "ready": ready,
        "source_sha256": source_sha256,
        "report_path": str(report_path),
        "failure_reasons": failure_reasons,
    }
    atomic_write_json(status_path, freeze_status)
    data_logger.info(
        "v4 preparation status=%s report=%s failure_count=%d",
        status,
        report_path,
        len(failure_reasons),
    )
    return report


def write_preparation_failure(
    work_dir: Path,
    error: CorpusV4Error,
    data_logger: logging.Logger,
) -> None:
    status_path = work_dir / "reports" / "freeze_status.json"
    atomic_write_json(
        status_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "preparation_failed",
            "ready": False,
            "failure_reasons": [error.as_reason()],
        },
    )
    data_logger.error(
        "v4 preparation failed code=%s remediation=%s",
        error.code,
        error.remediation,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--version-audit", type=Path, default=DEFAULT_VERSION_AUDIT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--cloud-dir", type=Path, default=DEFAULT_CLOUD_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-not-ready",
        action="store_true",
        help="Return zero after writing a frozen preview; status remains freeze_not_ready.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        loggers = configure_logging(args.log_dir)
        report = prepare_corpus_v4(
            input_path=args.input,
            version_audit_path=args.version_audit,
            work_dir=args.work_dir,
            cloud_dir=args.cloud_dir,
            seed=args.seed,
            loggers=loggers,
        )
    except CorpusV4Error as error:
        data_logger = logging.getLogger("corpus_v4.data")
        write_preparation_failure(args.work_dir, error, data_logger)
        return 1
    except Exception:
        logging.getLogger("corpus_v4.data").exception(
            "unexpected v4 preparation failure"
        )
        raise

    if not report["ready"] and not args.allow_not_ready:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
