"""Reusable automatic diagnostics for fixed-prompt novel generations.

These metrics detect mechanical failure modes such as repetition and verbatim
training-corpus overlap.  They deliberately do not pretend to measure semantic
quality; fluency, coherence, and prompt relevance still require human review.
"""

from __future__ import annotations

import re
from statistics import mean
from typing import Any, Sequence


HAN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
AUTOMATIC_GATES = {
    "minimum_mean_characters": 90,
    "minimum_mean_han_ratio": 0.60,
    "maximum_mean_han_ratio": 0.95,
    "maximum_four_gram_repetition": 0.08,
    "maximum_character_run": 5,
    "maximum_train_overlap": 30,
}


def ngram_repetition(text: str, size: int = 4) -> float:
    """Return the fraction of repeated character n-gram occurrences."""
    if len(text) < size:
        return 0.0
    ngrams = [text[index : index + size] for index in range(len(text) - size + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def longest_character_run(text: str) -> int:
    """Measure obvious degeneration such as repeated '试试试试'."""
    if not text:
        return 0
    longest = current = 1
    for previous, current_character in zip(text, text[1:]):
        current = current + 1 if current_character == previous else 1
        longest = max(longest, current)
    return longest


def longest_corpus_overlap(text: str, corpus: str, minimum: int = 8) -> int:
    """Find the longest generated substring copied verbatim from the train corpus."""
    compact = text.strip()
    if len(compact) < minimum:
        return 0

    def contains_overlap(size: int) -> bool:
        return any(
            compact[start : start + size] in corpus
            for start in range(0, len(compact) - size + 1)
        )

    best = 0
    lower = minimum
    upper = len(compact)
    while lower <= upper:
        middle = (lower + upper) // 2
        if contains_overlap(middle):
            best = middle
            lower = middle + 1
        else:
            upper = middle - 1
    return best


def sample_metrics(text: str, corpus: str) -> dict[str, float | int]:
    """Measure one continuation without assigning a semantic quality score."""
    visible = text.replace("\n", "")
    return {
        "characters": len(text),
        "han_ratio": (len(HAN_PATTERN.findall(visible)) / len(visible) if visible else 0.0),
        "four_gram_repetition": ngram_repetition(visible, 4),
        "longest_character_run": longest_character_run(visible),
        "longest_train_overlap": longest_corpus_overlap(visible, corpus),
        "paragraphs": len([part for part in text.split("\n\n") if part.strip()]),
    }


def apply_automatic_gates(summary: dict[str, Any], prompt_count: int) -> dict[str, Any]:
    """Apply hard safety/degeneration gates without inventing a quality score."""
    checks = {
        "all_prompts_present": summary["sample_count"] == prompt_count,
        "generation_length": summary["mean_characters"]
        >= AUTOMATIC_GATES["minimum_mean_characters"],
        "han_ratio": AUTOMATIC_GATES["minimum_mean_han_ratio"]
        <= summary["mean_han_ratio"]
        <= AUTOMATIC_GATES["maximum_mean_han_ratio"],
        "four_gram_repetition": summary["mean_four_gram_repetition"]
        <= AUTOMATIC_GATES["maximum_four_gram_repetition"],
        "character_run": summary["maximum_character_run"]
        <= AUTOMATIC_GATES["maximum_character_run"],
        "train_overlap": summary["maximum_train_overlap"]
        <= AUTOMATIC_GATES["maximum_train_overlap"],
    }
    return {
        "status": "PASS" if all(checks.values()) else "REVIEW",
        "checks": checks,
    }


def summarize_samples(
    samples: Sequence[dict[str, str]],
    corpus: str,
    *,
    prompt_count: int | None = None,
) -> dict[str, Any]:
    """Aggregate fixed-prompt diagnostics for one checkpoint."""
    measured = [
        {**sample, **sample_metrics(sample["continuation"], corpus)}
        for sample in samples
    ]
    if not measured:
        raise ValueError("at least one generated sample is required")
    expected = len(samples) if prompt_count is None else int(prompt_count)
    summary = {
        "sample_count": len(measured),
        "mean_characters": mean(row["characters"] for row in measured),
        "mean_han_ratio": mean(row["han_ratio"] for row in measured),
        "mean_four_gram_repetition": mean(
            row["four_gram_repetition"] for row in measured
        ),
        "maximum_character_run": max(row["longest_character_run"] for row in measured),
        "maximum_train_overlap": max(row["longest_train_overlap"] for row in measured),
    }
    summary["automatic_gates"] = apply_automatic_gates(summary, expected)
    return {"summary": summary, "samples": measured}
