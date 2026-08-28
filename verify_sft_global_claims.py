"""Verify SFT global and chapter-wide claims against the frozen v4 corpus.

This tool never grants human approval. It can verify literal first-occurrence
and first-cooccurrence claims, compute chapter-local name counts, and isolate
semantic absence/focus claims that still require a person to review them.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from build_sft_v4 import (
    atomic_write_text,
    configure_sft_v4_logging,
    read_jsonl,
    sha256_file,
)
from prepare_corpus_v4 import Chapter, parse_complete_chapters
from repair_teacher_sft_v4 import (
    CHAPTER_REFERENCE_PATTERN,
    chinese_number_to_int,
)


GLOBAL_FLAGS = {
    "aggregation_claim_requires_full_chapter_review",
    "fuzzy_chapter_rebind_requires_review",
    "global_claim_requires_index_review",
}
FIRST_OCCURRENCE_SUBCATEGORIES = {"concept_debut", "first_appearance"}
PERSON_SUBCATEGORIES = {
    "appearance_order",
    "character_state",
    "co_appearance",
    "first_cooccurrence",
    "interaction",
    "kinship",
    "realm_state",
    "speaker_attribution",
}
NON_PERSON_TERMS = {
    "老师",
    "弟子",
    "表姐",
    "父亲",
    "母亲",
    "女儿",
    "孙女",
    "斗者",
    "斗师",
    "大斗师",
    "斗灵",
    "斗王",
    "斗皇",
    "斗宗",
    "斗尊",
    "斗圣",
}


def chapter_body(chapter: Chapter) -> str:
    lines = chapter.source_text.splitlines()
    return "\n".join(lines[chapter.title_offset + 1 :])


class ChapterTextIndex:
    """Literal full-book and per-version chapter index."""

    def __init__(self, chapters: Sequence[Chapter]) -> None:
        versions: dict[int, list[str]] = defaultdict(list)
        ordered_versions: list[tuple[int, str]] = []
        for chapter in sorted(chapters, key=lambda item: item.section_index):
            body = chapter_body(chapter)
            versions[chapter.chapter_number].append(body)
            ordered_versions.append((chapter.chapter_number, body))
        self.versions = dict(versions)
        self.chapter_numbers = sorted(versions)
        self.ordered_versions = ordered_versions

    def first_chapter(self, entity: str) -> int | None:
        for chapter_number, text in self.ordered_versions:
            if entity in text:
                return chapter_number
        return None

    def first_cooccurrence(self, entities: Sequence[str]) -> int | None:
        if len(entities) < 2:
            return None
        first, second = entities[:2]
        for chapter_number, text in self.ordered_versions:
            if first in text and second in text:
                return chapter_number
        return None

    def literal_counts(
        self, chapter_number: int, entities: Iterable[str]
    ) -> list[dict[str, int]]:
        results: list[dict[str, int]] = []
        entity_list = sorted(set(entities))
        for version_index, text in enumerate(self.versions.get(chapter_number, [])):
            counts = Counter(
                {
                    entity: text.count(entity)
                    for entity in entity_list
                    if entity and text.count(entity) > 0
                }
            )
            row = {"version_index": version_index}
            row.update(dict(counts))
            results.append(row)
        return results


def extract_chapter_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for match in CHAPTER_REFERENCE_PATTERN.finditer(text):
        try:
            numbers.append(chinese_number_to_int(match.group(1)))
        except (TypeError, ValueError):
            continue
    return numbers


def person_lexicon(records: Sequence[dict[str, Any]]) -> set[str]:
    entities: set[str] = set()
    for record in records:
        origin = record["origin"]
        source_entities = [str(value).strip() for value in origin.get("source_entities", [])]
        subcategory = origin.get("source_subcategory")
        if subcategory in PERSON_SUBCATEGORIES:
            limit = 2 if subcategory not in {"speaker_attribution", "character_state", "realm_state"} else 1
            entities.update(source_entities[:limit])
    return {
        entity
        for entity in entities
        if len(entity) >= 2
        and entity not in NON_PERSON_TERMS
        and not any(term in entity for term in ("阶别", "强者", "人物"))
    }


def count_support(
    index: ChapterTextIndex,
    chapter_number: int,
    target: str,
    excluded: set[str],
    lexicon: set[str],
) -> dict[str, Any]:
    version_results = []
    support_values: list[bool] = []
    for row in index.literal_counts(chapter_number, lexicon - excluded):
        counts = {key: value for key, value in row.items() if key != "version_index"}
        maximum = max(counts.values(), default=0)
        leaders = sorted(key for key, value in counts.items() if value == maximum)
        supported = maximum > 0 and target in leaders
        support_values.append(supported)
        version_results.append(
            {
                "version_index": row["version_index"],
                "target_count": counts.get(target, 0),
                "maximum_count": maximum,
                "leaders": leaders[:10],
                "supported": supported,
            }
        )
    if support_values and all(support_values):
        status = "literal_count_supported_all_versions"
    elif support_values and not any(support_values):
        status = "literal_count_contradicted_all_versions"
    else:
        status = "literal_count_mixed_or_missing_versions"
    return {"status": status, "versions": version_results}


def verify_record(
    record: dict[str, Any],
    index: ChapterTextIndex,
    people: set[str],
) -> dict[str, Any]:
    origin = record["origin"]
    subcategory = str(origin["source_subcategory"])
    entities = [str(value) for value in origin.get("source_entities", [])]
    claimed_chapter = int(origin["source_chapter_number"])
    base = {
        "id": record["id"],
        "split": record["split"],
        "subcategory": subcategory,
        "claimed_chapter": claimed_chapter,
        "entities": entities,
    }

    if subcategory in FIRST_OCCURRENCE_SUBCATEGORIES:
        actual = index.first_chapter(entities[0]) if entities else None
        return {
            **base,
            "method": "literal_full_book_first_occurrence",
            "status": "index_verified" if actual == claimed_chapter else "index_contradicted",
            "computed_chapter": actual,
        }

    if subcategory == "first_cooccurrence":
        actual = index.first_cooccurrence(entities)
        return {
            **base,
            "method": "literal_same_chapter_first_cooccurrence",
            "status": "index_verified" if actual == claimed_chapter else "index_contradicted",
            "computed_chapter": actual,
        }

    if subcategory == "appearance_order" and len(entities) >= 2:
        firsts = {entity: index.first_chapter(entity) for entity in entities[:2]}
        known = {entity: value for entity, value in firsts.items() if value is not None}
        earlier = min(known, key=known.get) if len(known) == 2 and len(set(known.values())) == 2 else None
        answer_supports = bool(
            earlier
            and (
                f"{earlier}更早" in record["answer"]
                or (f"{earlier}比" in record["answer"] and "更早" in record["answer"])
            )
        )
        chapter_numbers = extract_chapter_numbers(record["answer"])
        chapters_consistent = not chapter_numbers or set(chapter_numbers) == set(known.values())
        if earlier and answer_supports and chapters_consistent:
            status = "index_verified"
        elif earlier and answer_supports:
            status = "index_order_verified_chapters_stale"
        else:
            status = "index_contradicted"
        return {
            **base,
            "method": "literal_full_book_order",
            "status": status,
            "computed_first_chapters": firsts,
            "computed_earlier_entity": earlier,
            "answer_chapters": chapter_numbers,
        }

    if subcategory == "false_premise":
        if entities and ("首次登场" in record["question"] or "最早" in record["answer"]):
            actual = index.first_chapter(entities[0])
            return {
                **base,
                "method": "literal_full_book_false_premise",
                "status": "index_verified" if actual == claimed_chapter else "index_contradicted",
                "computed_chapter": actual,
            }
        return {
            **base,
            "method": "unsupported_non_chapter_false_premise",
            "status": "manual_review_required",
        }

    if subcategory == "co_appearance" and len(entities) >= 2:
        support = count_support(
            index,
            claimed_chapter,
            target=entities[1],
            excluded={entities[0]},
            lexicon=people,
        )
        return {
            **base,
            "method": "chapter_literal_person_name_counts",
            "status": support["status"],
            "target": entities[1],
            "count_details": support["versions"],
        }

    if subcategory == "chapter_focus" and entities:
        support = count_support(
            index,
            claimed_chapter,
            target=entities[0],
            excluded=set(),
            lexicon=people,
        )
        return {
            **base,
            "method": "chapter_literal_person_name_counts_semantic_focus_pending",
            "status": "manual_review_required",
            "target": entities[0],
            "count_support": support,
        }

    if subcategory == "unanswerable":
        return {
            **base,
            "method": "semantic_absence_cannot_be_proven_by_literal_index_alone",
            "status": "manual_review_required",
        }

    return {**base, "method": "unsupported", "status": "manual_review_required"}


def run_verification(
    dataset_path: Path,
    corpus_path: Path,
    report_path: Path,
    log_dir: Path,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    loggers = configure_sft_v4_logging(log_dir)
    records = read_jsonl(dataset_path)
    corpus_text = corpus_path.read_text(encoding="utf-8")
    _, chapters = parse_complete_chapters(corpus_text)
    index = ChapterTextIndex(chapters)
    people = person_lexicon(records)
    selected = [
        record
        for record in records
        if record["split"] != "train"
        and set(record["origin"].get("repair_flags", [])) & GLOBAL_FLAGS
    ]
    loggers["data"].info(
        "global claim verification loaded records=%d selected=%d chapters=%d unique_chapters=%d",
        len(records),
        len(selected),
        len(chapters),
        len(index.chapter_numbers),
    )
    results = [verify_record(record, index, people) for record in selected]
    statuses = Counter(result["status"] for result in results)
    methods = Counter(result["method"] for result in results)
    status_by_subcategory: dict[str, Counter[str]] = defaultdict(Counter)
    for result in results:
        status_by_subcategory[result["subcategory"]][result["status"]] += 1
    report = {
        "schema_version": "sft_v4_global_claim_verification/1.0",
        "status": "needs_review",
        "human_approval_was_inferred": False,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "corpus_path": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
        "chapter_section_count": len(chapters),
        "unique_chapter_number_count": len(index.chapter_numbers),
        "person_lexicon_size": len(people),
        "selected_record_count": len(selected),
        "status_counts": dict(sorted(statuses.items())),
        "method_counts": dict(sorted(methods.items())),
        "status_by_subcategory": {
            subcategory: dict(sorted(counts.items()))
            for subcategory, counts in sorted(status_by_subcategory.items())
        },
        "results": results,
        "limitations": [
            "Literal-name counts do not prove narrative focus or screen time.",
            "Aliases, pronouns, OCR variants, and implicit references require human review.",
            "Duplicate retained chapter versions are checked separately.",
            "No review approval is written by this tool.",
        ],
    }
    atomic_write_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    if summary_path is not None:
        summary = {key: value for key, value in report.items() if key != "results"}
        summary["full_report_path"] = str(report_path)
        summary["full_report_sha256"] = sha256_file(report_path)
        atomic_write_text(
            summary_path,
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    loggers["validation"].info(
        "global claim verification complete selected=%d statuses=%s report=%s",
        len(selected),
        dict(statuses),
        report_path,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/sft/v4_teacher_repair/sft_v4_teacher_candidates.jsonl"),
    )
    parser.add_argument("--corpus", type=Path, default=Path("data/cloud_v4/corpus.txt"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/sft/v4_teacher_repair/global_claim_verification.json"),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("data/sft/v4_teacher_repair/global_verification_logs"),
    )
    parser.add_argument(
        "--summary-report",
        type=Path,
        default=Path(
            "reports/milestones/008_sft_v4_teacher_repair/"
            "global_claim_verification_summary.json"
        ),
        help="Aggregate-only report safe to archive in Git.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_verification(
        args.dataset,
        args.corpus,
        args.report,
        args.log_dir,
        args.summary_report,
    )
    print(
        json.dumps(
            {
                "selected_record_count": report["selected_record_count"],
                "status_counts": report["status_counts"],
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
