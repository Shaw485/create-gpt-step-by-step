"""Build the frozen 10,000-record novel-vertical SFT v7 release.

The builder consumes only ``data/cloud_v4/train.txt`` and the frozen BPE
tokenizer.  It never reads SFT v6.  Four physical JSONL files are produced;
``manifest.json`` is written last and is the only completion marker.  The
sealed file is available to this builder for construction and hashing only --
no sample, prompt, answer, evidence text, or token ID is emitted to reports or
logs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

from bpe_tokenizer import BPETokenizer
from prepare_corpus_v4 import Chapter, parse_complete_chapters
from sft_v7_vertical_catalog import (
    ANSWER_STYLE_BANKS,
    BANNED_TEXT_MARKERS,
    BOUNDARY,
    CHAT,
    CORE,
    CORE_TERMS,
    DIRECT_CORE_QUESTION_SUFFIXES,
    DIRECT_CORE_SPLIT_LEADS,
    DIMENSION_SPLIT_QUOTAS,
    DIMENSION_TOTALS,
    EVIDENCE,
    EXPRESSION,
    FROZEN_SEED,
    KNOWN_CORE_FACTS,
    MANIFEST_SCHEMA_VERSION,
    MINIMUM_MULTITURN_RECORDS,
    MINIMUM_RAG_RECORDS,
    NEGATIVE_SHARE_DENOMINATOR,
    NEGATIVE_SHARE_NUMERATOR,
    PROMPT_BANKS,
    RAG,
    SCHEMA_VERSION,
    SPLITS,
    SPLIT_TOTALS,
    TERM_BY_LABEL,
    UNSAFE_SOURCE_MARKERS,
    answer_style_id,
    prompt_template,
)
from training_runtime import (
    atomic_write_json,
    atomic_write_text,
    canonical_json_sha256,
    close_module_loggers,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
    resolve_module_log_levels,
)


DEFAULT_CORPUS = Path("data/cloud_v4/train.txt")
DEFAULT_TOKENIZER = Path("data/scaling_a/bpe_3000/tokenizer.json")
DEFAULT_OUTPUT_DIR = Path("data/sft/v7")
DEFAULT_LOG_DIR = Path("logs/sft_v7_vertical_build")
REPOSITORY_ROOT = Path(__file__).resolve().parent
SPECIAL_TOKENS = ("<BOS>", "<USER>", "<ASSISTANT>", "<EOS>", "<PAD>")
OUTPUT_NAMES = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "public_diagnostic": "public_diagnostic.jsonl",
    "sealed_test": "sealed_test.jsonl",
}

STRUCTURAL_PAIRS = (("（", "）"), ("(", ")"), ("《", "》"), ("【", "】"))
QUOTE_STRIP = str.maketrans({character: "" for character in "“”‘’《》【】（）()"})
POSITIVE_MATH_PROMPT_PATTERNS = (
    re.compile(r"(?:计算|求解|解答).{0,20}(?:算式|方程|结果|等于多少)"),
    re.compile(r"(?:加法|减法|乘法|除法|数学题|求导|积分|几何证明)"),
)


class VerticalBuildError(RuntimeError):
    """A content-safe, actionable build failure."""

    def __init__(self, code: str, message: str, remediation: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.remediation = remediation


def _portable_artifact_path(path: Path, *, role: str) -> str:
    """Return a host-independent path suitable for a published manifest.

    Repository inputs use POSIX repository-relative paths.  A deliberately
    supplied external input is represented by a logical URI containing only
    its role and basename; its SHA-256 remains the authoritative identity.
    This prevents developer home-directory names from leaking into releases.
    """

    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return f"artifact://{role}/{resolved.name}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_hash(*parts: object) -> str:
    return _sha256_text("|".join(str(part) for part in parts))


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(encoded)


def _compact(text: str, limit: int = 128) -> str:
    value = re.sub(r"\s+", " ", text).strip().translate(QUOTE_STRIP)
    value = value.rstrip("，。！？；：,.!?;: ")
    if len(value) > limit:
        value = value[:limit].rstrip("，。！？；：,.!?;: ")
    return value


def _balanced(text: str) -> bool:
    return all(text.count(opening) == text.count(closing) for opening, closing in STRUCTURAL_PAIRS)


def _is_positive_math_prompt(text: str) -> bool:
    """Detect an actual math-learning request, not novel words such as '无数'."""

    return any(pattern.search(text) for pattern in POSITIVE_MATH_PROMPT_PATTERNS)


@dataclass(frozen=True)
class EvidenceChunk:
    """One exact, recomputable corpus line."""

    split: str
    chapter_number: int
    chapter_title: str
    chapter_heading_line: int
    chapter_sha256: str
    line_number: int
    text: str
    clean_text: str
    terms: tuple[str, ...]
    following_line_number: int = 0
    following_text: str = ""

    @property
    def text_sha256(self) -> str:
        return _sha256_text(self.text)

    @property
    def chunk_sha256(self) -> str:
        return _stable_hash(
            "sft-v7-chunk",
            self.chapter_heading_line,
            self.line_number,
            self.text_sha256,
        )

    def payload(self, corpus_path: str) -> dict[str, Any]:
        return {
            "source_path": corpus_path,
            "source_split": "formal_pretrain_train",
            "chapter_number": self.chapter_number,
            "chapter_title": self.chapter_title,
            "chapter_heading_line": self.chapter_heading_line,
            "chapter_sha256": self.chapter_sha256,
            "line_start": self.line_number,
            "line_end": self.line_number,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "chunk_sha256": self.chunk_sha256,
        }

    def following_chunk(self) -> "EvidenceChunk | None":
        if not self.following_line_number or not self.following_text:
            return None
        return EvidenceChunk(
            split=self.split,
            chapter_number=self.chapter_number,
            chapter_title=self.chapter_title,
            chapter_heading_line=self.chapter_heading_line,
            chapter_sha256=self.chapter_sha256,
            line_number=self.following_line_number,
            text=self.following_text,
            clean_text=self.following_text.strip(),
            terms=_matched_terms(self.following_text),
        )


def _chapter_partition(
    chapters: Sequence[Chapter],
    seed: int,
) -> dict[str, list[Chapter]]:
    if len(chapters) < 8:
        raise VerticalBuildError(
            "INSUFFICIENT_CHAPTERS",
            "At least eight chapters are required for four-way physical isolation.",
            "Provide the complete formal training corpus.",
        )
    ordered = sorted(
        chapters,
        key=lambda chapter: _stable_hash(
            "sft-v7-chapter-partition",
            seed,
            chapter.section_id,
            chapter.source_sha256,
        ),
    )
    counts = {
        "train": int(len(ordered) * 0.80),
        "val": max(1, int(len(ordered) * 0.08)),
        "public_diagnostic": max(1, int(len(ordered) * 0.06)),
    }
    counts["sealed_test"] = len(ordered) - sum(counts.values())
    if counts["sealed_test"] <= 0:
        raise VerticalBuildError(
            "EMPTY_SEALED_PARTITION",
            "Chapter partitioning left no sealed chapters.",
            "Use a larger corpus or revise the fixed partition ratios.",
        )
    result: dict[str, list[Chapter]] = {}
    cursor = 0
    for split in SPLITS:
        count = counts[split]
        result[split] = ordered[cursor : cursor + count]
        cursor += count
    return result


def _matched_terms(text: str) -> tuple[str, ...]:
    matches = []
    for term in CORE_TERMS:
        if term.label in text or any(alias in text for alias in term.aliases):
            matches.append(term.label)
    return tuple(sorted(set(matches), key=lambda value: (-len(value), value)))


def extract_evidence_chunks(
    corpus_text: str,
    *,
    seed: int = FROZEN_SEED,
) -> tuple[dict[str, list[EvidenceChunk]], dict[str, list[Chapter]]]:
    """Parse, partition, and extract unique exact corpus lines."""

    preamble, chapters = parse_complete_chapters(corpus_text)
    if preamble.strip():
        raise VerticalBuildError(
            "UNEXPECTED_CORPUS_PREAMBLE",
            "The formal corpus contains a non-empty preamble.",
            "Re-run the frozen v4 corpus integrity checks.",
        )
    partitions = _chapter_partition(chapters, seed)
    reviewed_lines = {
        line_number
        for fact in KNOWN_CORE_FACTS
        for line_number in fact.evidence_lines
    }
    protected_headings = {
        chapter.start_line
        for chapter in chapters
        if any(
            chapter.range_start_line <= line_number <= chapter.end_line
            for line_number in reviewed_lines
        )
    }
    text_counts: Counter[str] = Counter()
    candidates: dict[str, list[EvidenceChunk]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        for chapter in partitions[split]:
            # These chapters back cross-split, explicitly tracked core cards.
            # Keeping them out of all local pools prevents their remaining
            # lines from creating non-core chapter leakage.
            if chapter.start_line in protected_headings:
                continue
            lines = chapter.source_text.splitlines()
            for local_index, raw_line in enumerate(lines):
                if local_index <= chapter.title_offset:
                    continue
                clean = raw_line.strip()
                if not 32 <= len(clean) <= 220:
                    continue
                if any(marker in clean for marker in UNSAFE_SOURCE_MARKERS):
                    continue
                if any(marker in clean for marker in BANNED_TEXT_MARKERS):
                    continue
                if not _balanced(clean):
                    continue
                terms = _matched_terms(clean)
                if not terms:
                    continue
                following_index = local_index + 1
                while following_index < len(lines) and not lines[following_index].strip():
                    following_index += 1
                following_text = ""
                following_line_number = 0
                if following_index < len(lines):
                    candidate = lines[following_index]
                    candidate_clean = candidate.strip()
                    if (
                        20 <= len(candidate_clean) <= 180
                        and not any(marker in candidate_clean for marker in UNSAFE_SOURCE_MARKERS)
                        and not any(marker in candidate_clean for marker in BANNED_TEXT_MARKERS)
                        and _balanced(candidate_clean)
                    ):
                        following_text = candidate
                        following_line_number = chapter.range_start_line + following_index
                identity = re.sub(r"[\W_]+", "", clean, flags=re.UNICODE).lower()
                text_counts[identity] += 1
                candidates[split].append(
                    EvidenceChunk(
                        split=split,
                        chapter_number=chapter.chapter_number,
                        chapter_title=chapter.title,
                        chapter_heading_line=chapter.start_line,
                        chapter_sha256=chapter.source_sha256,
                        line_number=chapter.range_start_line + local_index,
                        text=raw_line,
                        clean_text=clean,
                        terms=terms,
                        following_line_number=following_line_number,
                        following_text=following_text,
                    )
                )
    output: dict[str, list[EvidenceChunk]] = {}
    for split, values in candidates.items():
        unique = [
            chunk
            for chunk in values
            if text_counts[
                re.sub(r"[\W_]+", "", chunk.clean_text, flags=re.UNICODE).lower()
            ]
            == 1
        ]
        output[split] = sorted(
            unique,
            key=lambda chunk: _stable_hash(
                "sft-v7-evidence-order", seed, split, chunk.chunk_sha256
            ),
        )
        if len(output[split]) < 50:
            raise VerticalBuildError(
                "INSUFFICIENT_SPLIT_EVIDENCE",
                f"Split {split} has too few safe unique evidence lines.",
                "Review corpus partitioning or safe-line extraction thresholds.",
            )
    return output, partitions


class SplitPool:
    """Deterministic, entity-balanced selection inside one physical split."""

    def __init__(self, split: str, chunks: Sequence[EvidenceChunk], seed: int) -> None:
        if not chunks:
            raise VerticalBuildError(
                "EMPTY_EVIDENCE_POOL",
                f"Split {split} has no evidence chunks.",
                "Extract corpus evidence before building records.",
            )
        self.split = split
        self.seed = seed
        self.chunks = tuple(chunks)
        self.continuation_chunks = tuple(
            chunk for chunk in self.chunks if chunk.following_chunk() is not None
        )
        grouped: dict[str, list[EvidenceChunk]] = defaultdict(list)
        for chunk in chunks:
            for term in chunk.terms:
                grouped[term].append(chunk)
        self.by_term = {
            term: tuple(
                sorted(
                    values,
                    key=lambda chunk: _stable_hash(
                        "sft-v7-term-order", seed, split, term, chunk.chunk_sha256
                    ),
                )
            )
            for term, values in grouped.items()
        }
        self.terms = tuple(sorted(self.by_term))
        if not self.terms:
            raise VerticalBuildError(
                "EMPTY_ENTITY_POOL",
                f"Split {split} has no catalog terms.",
                "Expand the verified corpus term catalog.",
            )

    def term_chunk(self, index: int, salt: str) -> tuple[str, EvidenceChunk]:
        term_offset = int(_stable_hash(self.seed, self.split, salt)[:8], 16)
        term = self.terms[(index + term_offset) % len(self.terms)]
        values = self.by_term[term]
        cycle = index // len(self.terms)
        chunk_offset = int(_stable_hash(salt, term)[:8], 16)
        return term, values[(cycle + chunk_offset) % len(values)]

    def alternate_term(self, excluded: str, index: int, salt: str) -> str:
        candidates = [term for term in self.terms if term != excluded]
        if not candidates:
            raise VerticalBuildError(
                "NO_DISTRACTOR_ENTITY",
                f"Split {self.split} lacks a second entity.",
                "Provide evidence for at least two catalog entities per split.",
            )
        offset = int(_stable_hash(salt, excluded)[:8], 16)
        return candidates[(index + offset) % len(candidates)]

    def unique_chunk(self, index: int, salt: str) -> tuple[str, EvidenceChunk]:
        offset = int(_stable_hash("sft-v7-unique", self.seed, self.split, salt)[:8], 16)
        chunk = self.chunks[(index + offset) % len(self.chunks)]
        return chunk.terms[0], chunk

    def bundle(
        self,
        index: int,
        salt: str,
        *,
        size: int,
        same_term: bool,
    ) -> tuple[str, list[EvidenceChunk]]:
        term, first = self.term_chunk(index, salt)
        if same_term and len(self.by_term[term]) < size:
            eligible_terms = sorted(
                candidate
                for candidate, values in self.by_term.items()
                if len(values) >= size
            )
            if not eligible_terms:
                raise VerticalBuildError(
                    "INSUFFICIENT_SAME_ENTITY_EVIDENCE",
                    f"Split {self.split} cannot form a {size}-chunk same-entity bundle.",
                    "Expand the verified source pool for multi-passage RAG.",
                )
            offset = int(_stable_hash(salt, index, size)[:8], 16)
            term = eligible_terms[offset % len(eligible_terms)]
            term_values = self.by_term[term]
            first = term_values[(index + offset) % len(term_values)]
        selected = [first]
        if same_term:
            candidates = list(self.by_term[term])
        else:
            candidates = list(self.chunks)
        candidate_offset = int(
            _stable_hash("sft-v7-bundle", self.seed, self.split, salt, index)[:8],
            16,
        ) % len(candidates)
        for delta in range(len(candidates)):
            chunk = candidates[(candidate_offset + delta) % len(candidates)]
            if chunk.chunk_sha256 in {item.chunk_sha256 for item in selected}:
                continue
            if not same_term and term in chunk.terms:
                continue
            selected.append(chunk)
            if len(selected) == size:
                break
        if len(selected) < size:
            for chunk in self.chunks:
                if chunk.chunk_sha256 not in {item.chunk_sha256 for item in selected}:
                    selected.append(chunk)
                    if len(selected) == size:
                        break
        if len(selected) != size:
            raise VerticalBuildError(
                "INSUFFICIENT_BUNDLE_EVIDENCE",
                f"Split {self.split} cannot form a {size}-chunk bundle.",
                "Increase the number of unique evidence lines.",
            )
        return term, selected

    def continuation_pair(self, index: int, salt: str) -> tuple[str, EvidenceChunk, EvidenceChunk]:
        if not self.continuation_chunks:
            raise VerticalBuildError(
                "NO_CONTINUATION_PAIR",
                f"Split {self.split} has no safe adjacent continuation pair.",
                "Review source-line extraction and continuation thresholds.",
            )
        offset = int(
            _stable_hash("sft-v7-continuation", self.seed, self.split, salt)[:8],
            16,
        )
        chunk = self.continuation_chunks[(index + offset) % len(self.continuation_chunks)]
        following = chunk.following_chunk()
        if following is None:  # Defensive: the tuple above is the frozen eligibility gate.
            raise VerticalBuildError(
                "CONTINUATION_ELIGIBILITY_DRIFT",
                f"Split {self.split} continuation eligibility changed during generation.",
                "Re-extract evidence and retry from immutable inputs.",
            )
        return chunk.terms[0], chunk, following


def _bundle_text(chunks: Sequence[EvidenceChunk], limit: int = 150) -> str:
    return "\n".join(
        f"[{index}] {_compact(chunk.clean_text, limit)}"
        for index, chunk in enumerate(chunks, 1)
    )


def _answer_from_chunks(
    entity: str,
    chunks: Sequence[EvidenceChunk],
    style: int,
    *,
    long_form: bool = False,
    scope_note: bool = False,
) -> str:
    if len(chunks) > 1:
        snippet_limit = 74 if long_form else 50
    else:
        snippet_limit = 88 if long_form else 62
    snippets = [_compact(chunk.clean_text, snippet_limit) for chunk in chunks]
    first = snippets[0]
    openings = (
        f"原文写道“{first}”，其中涉及{entity}。",
        f"材料表述为“{first}”，这里写到{entity}。",
        f"片段可核对：“{first}”。相关对象是{entity}。",
        f"这处写的是“{first}”，内容与{entity}有关。",
        f"文本中出现“{first}”，并提到{entity}。",
        f"可复查的原句是“{first}”，其中包含{entity}。",
    )
    answer = openings[style % len(openings)]
    if len(snippets) > 1:
        connectors = (
            "另一段还写到",
            "第二项依据是",
            "相关材料同时提到",
            "合并阅读时还应注意",
            "另一个可核验片段是",
            "其余证据还包括",
        )
        for offset, snippet in enumerate(snippets[1:], 1):
            answer += f"{connectors[(style + offset) % len(connectors)]}：{snippet}。"
    if long_form and len(chunks) > 1:
        syntheses = (
            f"合起来看，不同片段都提供了与{entity}有关的局部信息。",
            f"综合这些表述，可以更完整地理解{entity}在当前材料中的情况。",
            f"这些证据从不同位置补充了{entity}相关内容。",
            f"多段文字共同构成了对{entity}的局部描写。",
            f"把各段并读，{entity}相关线索会更清楚。",
            f"上述材料分别呈现了{entity}在不同局部情境中的信息。",
        )
        answer += syntheses[style % len(syntheses)]
    if scope_note:
        scope = (
            "回答限定在给定文字能够支持的范围内。",
            "片段之外的关系仍需另行查证。",
            "这里不把相邻情节混入当前结论。",
            "未在材料中出现的细节不作推断。",
            "以上说明只对应当前可见证据。",
            "更完整的时间线需要继续检索原著。",
        )
        answer += scope[style % len(scope)]
    return answer


def _insufficient_answer(entity: str, target: str, style: int, anchor: str = "") -> str:
    variants = (
        f"这组材料只明确写到{entity}，没有建立{target}与它的关系；当前证据不能支持该结论。",
        f"不能由这些片段确认{target}就是{entity}。材料缺少直接联系，需要补充相关原文后再判断。",
        f"现有证据涉及{entity}，却没有给出{target}的身份依据，因此应停止在证据边界内。",
        f"关于{target}的断言没有被当前文字支持。若要继续判断，需要包含{target}的有效片段。",
        f"这些内容不足以把{target}和{entity}联系起来；能够确认的只有材料已经明写的部分。",
        f"证据包没有完成对{target}的指认，直接下结论会超出文本，应补充可核验材料。",
    )
    answer = variants[style % len(variants)]
    if anchor:
        anchor_closings = (
            "本次可核对的原文开头是",
            "材料中实际出现的文字包括",
            "当前片段能够直接定位到",
            "用于核验的局部表述是",
            "证据包里真正可见的是",
            "原文中可复查的一处写道",
        )
        answer += f"{anchor_closings[style % len(anchor_closings)]}“{anchor}”。"
    return answer


def _correction_answer(entity: str, chunk: EvidenceChunk, style: int) -> str:
    snippet = _compact(chunk.clean_text, 82)
    variants = (
        f"这个说法不成立。第{chunk.chapter_number}章明确写到{entity}：{snippet}。",
        f"应当纠正为：第{chunk.chapter_number}章确实出现了{entity}，原文依据是{snippet}。",
        f"原著并非没有提及{entity}。第{chunk.chapter_number}章的相关表述为{snippet}。",
        f"核对结果是该判断错误；第{chunk.chapter_number}章写有{entity}，内容是{snippet}。",
        f"第{chunk.chapter_number}章与{entity}有关，证据可直接定位到这句：{snippet}。",
        f"不能说这一章没有{entity}，因为原文已经明确写道：{snippet}。",
    )
    return variants[style % len(variants)]


def _summary_answer(entity: str, chunk: EvidenceChunk, style: int) -> str:
    snippet = _compact(chunk.clean_text, 92)
    variants = (
        f"片段围绕{entity}展开，主要内容是：{snippet}。",
        f"这段文字写到{entity}，局部要点可概括为：{snippet}。",
        f"简要来说，材料中的{entity}与以下情境相关：{snippet}。",
        f"片段的核心信息涉及{entity}，可压缩为：{snippet}。",
        f"此处提到{entity}，并呈现了这样的局部内容：{snippet}。",
        f"就给定范围看，{entity}相关情节可概括成：{snippet}。",
    )
    return variants[style % len(variants)]


def _rewrite_answer(entity: str, chunk: EvidenceChunk, style: int) -> str:
    snippet = _compact(chunk.clean_text, 96)
    variants = (
        f"改写：这段内容提到{entity}，并说明{snippet}。",
        f"简明重述：{entity}出现在当前情境中，原有信息是{snippet}。",
        f"可以改写为：文本写到{entity}，相关情况为{snippet}。",
        f"直白地说，这里涉及{entity}，具体内容是{snippet}。",
        f"紧凑表达：材料围绕{entity}写了{snippet}。",
        f"保持事实不变可写成：{entity}在片段中与{snippet}有关。",
    )
    return variants[style % len(variants)]


def _encoding_audit(
    messages: Sequence[Mapping[str, str]],
    tokenizer: Any,
) -> dict[str, int]:
    missing = [token for token in SPECIAL_TOKENS if token not in tokenizer.special_to_id]
    if missing:
        raise VerticalBuildError(
            "MISSING_SPECIAL_TOKENS",
            "The tokenizer lacks required conversation special tokens.",
            "Add BOS, USER, ASSISTANT, EOS and PAD before building SFT data.",
        )
    sequence_tokens = 1
    supervised_tokens = 0
    assistant_turns = 0
    answer_tokens = 0
    for index, message in enumerate(messages):
        expected = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected:
            raise VerticalBuildError(
                "ROLE_ALTERNATION",
                "A generated conversation has invalid role alternation.",
                "Inspect the task-family record factory.",
            )
        content = str(message.get("content", ""))
        if not content.strip():
            raise VerticalBuildError(
                "EMPTY_MESSAGE",
                "A generated conversation contains an empty message.",
                "Inspect the task-family record factory.",
            )
        try:
            length = len(tokenizer.encode(content))
        except Exception as error:
            raise VerticalBuildError(
                "UNENCODABLE_TEXT",
                "A generated record cannot be encoded by the frozen tokenizer.",
                "Remove unsupported source characters or rebuild the tokenizer deliberately.",
            ) from error
        sequence_tokens += 1 + length
        if expected == "assistant":
            sequence_tokens += 1
            supervised_tokens += length + 1
            assistant_turns += 1
            answer_tokens = length
    if sequence_tokens > 512:
        raise VerticalBuildError(
            "SEQUENCE_OVER_512",
            "A generated conversation exceeds the frozen 512-token context.",
            "Shorten the evidence display or response while retaining exact provenance.",
        )
    return {
        "sequence_tokens": sequence_tokens,
        "supervised_tokens": supervised_tokens,
        "last_answer_tokens": answer_tokens,
        "assistant_turns": assistant_turns,
        "eos_targets": assistant_turns,
        "masked_user_and_role_tokens": sequence_tokens - supervised_tokens,
    }


def _record(
    *,
    split: str,
    dimension: str,
    family: str,
    record_index: int,
    messages: list[dict[str, str]],
    chunks: Sequence[EvidenceChunk],
    corpus_path: str,
    corpus_sha256: str,
    tokenizer: Any,
    prompt_id: str,
    prompt_text: str,
    style_id: str,
    entity: str,
    semantic_group: str,
    fact_id: str,
    generalization: str,
    evidence_status: str,
    negative_type: str | None,
    metric: str,
    required_terms: Sequence[str],
    forbidden_terms: Sequence[str],
    known_fact: bool,
    needs_evidence: bool,
    evidence_sufficient: bool,
    acceptance_case_id: str = "",
    calibration_triplet_id: str = "",
    seed: int = FROZEN_SEED,
) -> dict[str, Any]:
    question = messages[-2]["content"]
    answer = messages[-1]["content"]
    combined = "\n".join(message["content"] for message in messages)
    if any(marker in combined for marker in BANNED_TEXT_MARKERS):
        raise VerticalBuildError(
            "BANNED_TEMPLATE_TEXT",
            f"Generated {split}/{family}/{record_index} contains a frozen banned marker.",
            "Replace the responsible prompt or answer style.",
        )
    if _is_positive_math_prompt(question):
        raise VerticalBuildError(
            "MATH_RECORD",
            f"Generated {split}/{family}/{record_index} was classified as positive mathematics training.",
            "Replace it with a novel-domain task or a non-mathematical boundary case.",
        )
    encoding = _encoding_audit(messages, tokenizer)
    chunk_payloads = [
        {
            "status": "verified_train_corpus",
            "source_path": corpus_path,
            "source_split": "formal_pretrain_train",
            "chapter_id": f"chapter-{chunk.chapter_number}",
            "chapter_number": chunk.chapter_number,
            "chapter_title": chunk.chapter_title,
            "heading_line": chunk.chapter_heading_line,
            "chapter_sha256": chunk.chapter_sha256,
            "start_line": chunk.line_number,
            "end_line": chunk.line_number,
            "text": chunk.text,
            "sha256": chunk.text_sha256,
            "chunk_sha256": chunk.chunk_sha256,
        }
        for chunk in chunks
    ]
    bundle_sha = _sha256_text("|".join(chunk.chunk_sha256 for chunk in chunks))
    digest = _stable_hash(
        "sft-v7-record",
        split,
        dimension,
        family,
        record_index,
        question,
        answer,
        bundle_sha,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"v7_{digest[:24]}",
        "split": split,
        "primary_dimension": dimension,
        "task_family": family,
        "semantic_group": semantic_group,
        "fact_id": fact_id,
        "generalization_policy": generalization,
        "question": question,
        "answer": answer,
        "messages": messages,
        "evidence": chunk_payloads,
        "answer_support": {
            "status": (
                "supported"
                if evidence_sufficient
                else "insufficient_evidence"
                if needs_evidence
                else "not_applicable"
            ),
            "evidence_sha256s": (
                [chunk.text_sha256 for chunk in chunks] if evidence_sufficient else []
            ),
            "supporting_spans": (
                [chunk.text.strip()[:48] for chunk in chunks]
                if evidence_sufficient
                else []
            ),
            "evidence_status": evidence_status,
            "negative_type": negative_type,
            "bundle_sha256": bundle_sha,
        },
        "coverage": {
            "entities": [entity] if entity else [],
            "concepts": [],
        },
        "evaluation": {
            "metric": metric,
            "required_terms": sorted(set(required_terms)),
            "forbidden_terms": sorted(set(forbidden_terms)),
            "known_fact": known_fact,
            "needs_evidence": needs_evidence,
            "evidence_sufficient": evidence_sufficient,
            "acceptance_case_id": acceptance_case_id,
            "calibration_triplet_id": calibration_triplet_id,
            "capability_mode": (
                "known_core"
                if known_fact
                else "needs_evidence"
                if needs_evidence
                else "interaction"
                if dimension == CHAT
                else "expression"
                if dimension == EXPRESSION
                else "grounded_answer"
            ),
            "evaluation_track": generalization,
        },
        "prompt_template_id": prompt_id,
        "answer_style_id": style_id,
        "generation": {
            "prompt_template_id": prompt_id,
            "prompt_template_sha256": _sha256_text(prompt_text),
            "answer_style_id": style_id,
        },
        "encoding_audit": encoding,
        "provenance": {
            "generation_method": "deterministic_corpus_grounded_builder",
            "corpus_path": corpus_path,
            "corpus_sha256": corpus_sha256,
            "seed": seed,
        },
        "review": {
            "status": "ai_reviewed_and_rule_checked",
            "evidence_recomputed": True,
        },
    }


DIRECT_CORE_TOTAL = 900
DIRECT_CORE_SPLIT_QUOTAS = {
    "train": 720,
    "val": 72,
    "public_diagnostic": 54,
    "sealed_test": 54,
}
CALIBRATION_TRIPLETS = {
    "train": 100,
    "val": 10,
    "public_diagnostic": 10,
    "sealed_test": 10,
}
EVIDENCE_NEGATIVE_QUOTAS = {
    "train": 448,
    "val": 45,
    "public_diagnostic": 34,
    "sealed_test": 33,
}
RAG_NEGATIVE_QUOTAS = {
    "train": 196,
    "val": 20,
    "public_diagnostic": 15,
    "sealed_test": 14,
}
EXPRESSION_FAMILY_SPLIT_QUOTAS = {
    split: {
        "summary": DIMENSION_SPLIT_QUOTAS[EXPRESSION][split] * 5 // 13,
        "rewrite": DIMENSION_SPLIT_QUOTAS[EXPRESSION][split] * 5 // 13,
        "continuation": DIMENSION_SPLIT_QUOTAS[EXPRESSION][split] * 3 // 13,
    }
    for split in SPLITS
}
BOUNDARY_QUALIFIERS = (
    "回答时请说明核验条件。",
    "不要把估计当成精确结论。",
    "请区分记忆与可复算结果。",
    "若材料不够，请明确缺少什么。",
    "只在能够核验时给出精确答案。",
    "请说明是否需要检索原著。",
    "不要凭印象补全全书信息。",
    "请给出稳妥而自然的回应。",
)
KNOWN_ANSWER_TAILS = (
    "这是原文能够直接支持的核心信息。",
    "这项身份或关系有明确原著依据。",
    "回答限定在已经核验的基本事实内。",
    "这里没有加入证据之外的经历或评价。",
    "对新读者而言，核心关系可据此理解。",
    "其余细节需要结合相应情节另行说明。",
    "这能回答人物或设定的基本定位。",
    "以上结论来自已经定位的原著文字。",
)


def _find_chapter_for_line(chapters: Sequence[Chapter], line_number: int) -> Chapter:
    for chapter in chapters:
        if chapter.range_start_line <= line_number <= chapter.end_line:
            return chapter
    raise VerticalBuildError(
        "KNOWN_FACT_LINE_OUTSIDE_CHAPTER",
        f"Reviewed line {line_number} is outside parsed chapter ranges.",
        "Re-run the v4 corpus audit and update the reviewed fact catalog.",
    )


def load_known_core_evidence(
    corpus_text: str,
    chapters: Sequence[Chapter],
    split: str,
) -> dict[str, list[EvidenceChunk]]:
    """Recompute every reviewed core card from immutable corpus lines."""

    lines = corpus_text.splitlines()
    result: dict[str, list[EvidenceChunk]] = {}
    for fact in KNOWN_CORE_FACTS:
        if len(fact.evidence_lines) != len(fact.evidence_needles):
            raise VerticalBuildError(
                "KNOWN_FACT_CATALOG_SHAPE",
                f"Known fact {fact.fact_id} has mismatched evidence metadata.",
                "Pair each reviewed line with one required-substring tuple.",
            )
        chunks: list[EvidenceChunk] = []
        for line_number, needles in zip(fact.evidence_lines, fact.evidence_needles):
            if not 1 <= line_number <= len(lines):
                raise VerticalBuildError(
                    "KNOWN_FACT_LINE_RANGE",
                    f"Known fact {fact.fact_id} references an invalid source line.",
                    "Recompute the reviewed line number against formal train.txt.",
                )
            source_line = lines[line_number - 1]
            if any(needle not in source_line for needle in needles):
                raise VerticalBuildError(
                    "KNOWN_FACT_EVIDENCE_DRIFT",
                    f"Known fact {fact.fact_id} no longer matches its reviewed source line.",
                    "Stop the release and independently re-review the formal corpus evidence.",
                )
            chapter = _find_chapter_for_line(chapters, line_number)
            chunks.append(
                EvidenceChunk(
                    split=split,
                    chapter_number=chapter.chapter_number,
                    chapter_title=chapter.title,
                    chapter_heading_line=chapter.start_line,
                    chapter_sha256=chapter.source_sha256,
                    line_number=line_number,
                    text=source_line,
                    clean_text=source_line.strip(),
                    terms=_matched_terms(source_line),
                )
            )
        result[fact.fact_id] = chunks
    return result


def _render_template(
    family: str,
    split: str,
    index: int,
    **values: object,
) -> tuple[str, str, str]:
    prompt_id, template = prompt_template(family, split, index)
    try:
        rendered = template.format(**values)
    except (KeyError, IndexError, ValueError) as error:
        raise VerticalBuildError(
            "PROMPT_RENDER_FAILURE",
            f"Prompt family {family} could not be rendered.",
            "Ensure every frozen placeholder is supplied by the family factory.",
        ) from error
    return prompt_id, template, rendered


def _target_not_in_chunks(
    pool: SplitPool,
    chunks: Sequence[EvidenceChunk],
    index: int,
    salt: str,
) -> str:
    body = "\n".join(chunk.clean_text for chunk in chunks)
    candidates = sorted(
        pool.terms,
        key=lambda term: _stable_hash("sft-v7-target", pool.seed, pool.split, salt, index, term),
    )
    for candidate in candidates:
        term = TERM_BY_LABEL[candidate]
        if candidate not in body and not any(alias in body for alias in term.aliases):
            return candidate
    raise VerticalBuildError(
        "NO_CLEAN_NEGATIVE_TARGET",
        f"Split {pool.split} cannot select an absent target entity.",
        "Expand the entity catalog or reduce evidence-bundle density.",
    )


def _known_answer(fact: Any, style: int) -> str:
    return f"{fact.answer}{KNOWN_ANSWER_TAILS[style % len(KNOWN_ANSWER_TAILS)]}"


def _boundary_answer(entity: str, request_kind: int, style: int) -> str:
    requested = (
        "首次出现的精确章节",
        "全书出现总数",
        "完整且无遗漏的时间线",
    )[request_kind % 3]
    variants = (
        f"仅凭参数记忆不能可靠给出{entity}的{requested}。这类全局结论需要检索全书并保留可复算证据。",
        f"要准确回答{entity}的{requested}，需要原著索引或逐章材料；没有证据时不应猜测一个数值或位置。",
        f"当前没有足够证据核验{entity}的{requested}。请提供检索结果，我再依据材料作答。",
        f"这不是适合凭印象下结论的问题。关于{entity}的{requested}，需要先检索原著并核对章节记录。",
        f"无法在缺少全书证据时确认{entity}的{requested}。有了实体索引或相关材料后才能给出可靠回答。",
        f"关于{entity}的{requested}，应以可复算检索为准。现在证据不足，因此不能把推测当成答案。",
    )
    return variants[style % len(variants)]


def _calibration_id(split: str, index: int) -> str:
    return f"v7-cal-{split}-{index:04d}"


def _build_direct_core_records(
    *,
    split: str,
    count: int,
    corpus_path: str,
    corpus_sha256: str,
    tokenizer: Any,
    known_evidence: Mapping[str, Sequence[EvidenceChunk]],
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(count):
        fact_offset = index % len(KNOWN_CORE_FACTS)
        fact = KNOWN_CORE_FACTS[fact_offset]
        cycle = index // len(KNOWN_CORE_FACTS)
        suffix_index = cycle % len(DIRECT_CORE_QUESTION_SUFFIXES)
        suffix = DIRECT_CORE_QUESTION_SUFFIXES[suffix_index]
        secondary_suffix = ""
        secondary_index = -1
        if cycle >= len(DIRECT_CORE_QUESTION_SUFFIXES):
            group = cycle // len(DIRECT_CORE_QUESTION_SUFFIXES)
            secondary_index = (
                suffix_index + group
            ) % len(DIRECT_CORE_QUESTION_SUFFIXES)
            secondary_suffix = DIRECT_CORE_QUESTION_SUFFIXES[secondary_index]
        lead = DIRECT_CORE_SPLIT_LEADS[split]
        if split == "public_diagnostic" and fact.acceptance_case_id and cycle == 0:
            question = fact.canonical_question
            template = "{question}"
            prompt_id = f"{split}:known_core_direct:acceptance"
        else:
            question = lead.format(question=fact.canonical_question)
            question = f"{question} {suffix}"
            template = f"{lead} {suffix}"
            prompt_id = f"{split}:known_core_direct:p{suffix_index}"
            if secondary_suffix:
                question = f"{question} {secondary_suffix}"
                template = f"{template} {secondary_suffix}"
                prompt_id = f"{prompt_id}-p{secondary_index}"
        style = index % len(KNOWN_ANSWER_TAILS)
        answer = _known_answer(fact, style)
        records.append(
            _record(
                split=split,
                dimension=CORE,
                family="known_core_direct",
                record_index=index,
                messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                chunks=known_evidence[fact.fact_id],
                corpus_path=corpus_path,
                corpus_sha256=corpus_sha256,
                tokenizer=tokenizer,
                prompt_id=prompt_id,
                prompt_text=template,
                style_id=f"{split}:known_core_direct:a{style}",
                entity=fact.entity,
                semantic_group=f"known-core:{fact.fact_id}",
                fact_id=fact.fact_id,
                generalization="seen_fact_unseen_wording",
                evidence_status="reviewed_exact_train_lines",
                negative_type=None,
                metric="keypoints",
                required_terms=fact.required_terms,
                forbidden_terms=("资料不足", "无法确定"),
                known_fact=True,
                needs_evidence=False,
                evidence_sufficient=True,
                acceptance_case_id=(
                    fact.acceptance_case_id
                    if split == "public_diagnostic" and cycle == 0
                    else ""
                ),
                seed=seed,
            )
        )
    return records


def _build_local_core_records(
    *,
    split: str,
    count: int,
    pool: SplitPool,
    corpus_path: str,
    corpus_sha256: str,
    tokenizer: Any,
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    base_question_counts: Counter[str] = Counter()
    triplet_count = CALIBRATION_TRIPLETS[split]
    fact_count = (count * 2) // 3
    for index in range(count):
        calibration = index < triplet_count
        salt = "calibration" if calibration else "local-core"
        entity, chunk = pool.term_chunk(index, salt)
        family = "core_fact" if index < fact_count else "core_correction"
        prompt_id, template, question = _render_template(
            family,
            split,
            index,
            chapter_number=chunk.chapter_number,
            entity=entity,
        )
        occurrence = base_question_counts[question]
        base_question_counts[question] += 1
        if occurrence:
            qualifier = DIRECT_CORE_QUESTION_SUFFIXES[
                (occurrence - 1) % len(DIRECT_CORE_QUESTION_SUFFIXES)
            ]
            question = f"{question} {qualifier}"
            template = f"{template} {qualifier}"
            prompt_id = f"{prompt_id}:q{(occurrence - 1) % len(DIRECT_CORE_QUESTION_SUFFIXES)}"
        style = index % 6
        answer = (
            _answer_from_chunks(entity, [chunk], style)
            if family == "core_fact"
            else _correction_answer(entity, chunk, style)
        )
        triplet_id = _calibration_id(split, index) if calibration else ""
        records.append(
            _record(
                split=split,
                dimension=CORE,
                family=family,
                record_index=index,
                messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                chunks=[chunk],
                corpus_path=corpus_path,
                corpus_sha256=corpus_sha256,
                tokenizer=tokenizer,
                prompt_id=prompt_id,
                prompt_text=template,
                style_id=answer_style_id(family, split, index),
                entity=entity,
                semantic_group=f"local-core:{chunk.chunk_sha256}",
                fact_id=f"local:{chunk.chunk_sha256}",
                generalization="split_isolated_local_fact",
                evidence_status="verified_exact_train_line",
                negative_type=None,
                metric="behavior" if family == "core_correction" else "keypoints",
                required_terms=(entity,),
                forbidden_terms=("资料不足", "无法确定"),
                known_fact=True,
                needs_evidence=False,
                evidence_sufficient=True,
                calibration_triplet_id=triplet_id,
                seed=seed,
            )
        )
    return records


def _build_evidence_records(
    *,
    split: str,
    count: int,
    pool: SplitPool,
    corpus_path: str,
    corpus_sha256: str,
    tokenizer: Any,
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    used_chunks: set[str] = set()
    negative_start = count - EVIDENCE_NEGATIVE_QUOTAS[split]
    triplet_count = CALIBRATION_TRIPLETS[split]
    for index in range(count):
        calibration = index < triplet_count
        negative = index >= negative_start
        salt = "calibration" if calibration else "single-evidence"
        if calibration:
            entity, chunk = pool.term_chunk(index, salt)
        else:
            for attempt in range(len(pool.chunks)):
                entity, chunk = pool.unique_chunk(index + attempt, salt)
                if chunk.chunk_sha256 not in used_chunks:
                    break
            else:
                raise VerticalBuildError(
                    "EVIDENCE_POOL_EXHAUSTED",
                    f"Split {split} lacks enough unique single-passage evidence.",
                    "Expand the verified source pool or reduce the frozen quota.",
                )
        used_chunks.add(chunk.chunk_sha256)
        quote = _compact(chunk.clean_text, 220)
        style = index % 6
        if negative:
            family = "passage_insufficient"
            target = _target_not_in_chunks(pool, [chunk], index, family)
            answer = _insufficient_answer(
                entity,
                target,
                style,
                _compact(chunk.clean_text, 56),
            )
        else:
            family = "passage_answer"
            target = ""
            answer = _answer_from_chunks(
                entity,
                [chunk],
                style,
                long_form=index % 10 == 0,
                scope_note=index % 10 == 0,
            )
        prompt_id, template, question = _render_template(
            family,
            split,
            index,
            quote=quote,
            entity=entity,
            target=target,
        )
        triplet_id = _calibration_id(split, index) if calibration else ""
        records.append(
            _record(
                split=split,
                dimension=EVIDENCE,
                family=family,
                record_index=index,
                messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                chunks=[chunk],
                corpus_path=corpus_path,
                corpus_sha256=corpus_sha256,
                tokenizer=tokenizer,
                prompt_id=prompt_id,
                prompt_text=template,
                style_id=answer_style_id(family, split, index),
                entity=entity,
                semantic_group=f"grounded:{chunk.chunk_sha256}",
                fact_id=f"grounded:{chunk.chunk_sha256}",
                generalization="split_isolated_evidence",
                evidence_status=(
                    "insufficient_for_target" if negative else "verified_exact_train_line"
                ),
                negative_type="absent_entity_distractor" if negative else None,
                metric="behavior" if negative else "normalized_f1",
                required_terms=() if negative else (entity,),
                forbidden_terms=(),
                known_fact=False,
                needs_evidence=negative,
                evidence_sufficient=not negative,
                calibration_triplet_id=triplet_id,
                seed=seed,
            )
        )
    return records


def _build_rag_records(
    *,
    split: str,
    count: int,
    pool: SplitPool,
    corpus_path: str,
    corpus_sha256: str,
    tokenizer: Any,
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    negative_start = count - RAG_NEGATIVE_QUOTAS[split]
    for index in range(count):
        negative = index >= negative_start
        size = 2 + index % 3
        entity, chunks = pool.bundle(
            index,
            "rag-negative" if negative else "rag-positive",
            size=size,
            same_term=not negative,
        )
        style = index % 6
        if negative:
            family = "rag_insufficient"
            target = _target_not_in_chunks(pool, chunks, index, family)
            answer = _insufficient_answer(
                entity,
                target,
                style,
                _compact(chunks[0].clean_text, 56),
            )
        else:
            family = "rag_compose"
            target = ""
            answer = _answer_from_chunks(
                entity,
                chunks,
                style,
                long_form=size == 2,
                scope_note=index % 8 == 0,
            )
        bundle = _bundle_text(chunks, limit=64)
        prompt_id, template, question = _render_template(
            family,
            split,
            index,
            bundle=bundle,
            entity=entity,
            target=target,
        )
        records.append(
            _record(
                split=split,
                dimension=RAG,
                family=family,
                record_index=index,
                messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                chunks=chunks,
                corpus_path=corpus_path,
                corpus_sha256=corpus_sha256,
                tokenizer=tokenizer,
                prompt_id=prompt_id,
                prompt_text=template,
                style_id=answer_style_id(family, split, index),
                entity=entity,
                semantic_group=f"rag:{_sha256_text('|'.join(c.chunk_sha256 for c in chunks))}",
                fact_id=f"rag:{chunks[0].chunk_sha256}",
                generalization="split_isolated_rag_bundle",
                evidence_status=(
                    "insufficient_for_target" if negative else "verified_exact_train_lines"
                ),
                negative_type="irrelevant_bundle_target" if negative else None,
                metric="behavior" if negative else "keypoints",
                required_terms=() if negative else (entity,),
                forbidden_terms=(),
                known_fact=False,
                needs_evidence=negative,
                evidence_sufficient=not negative,
                seed=seed,
            )
        )
    return records


def _build_chat_records(
    *,
    split: str,
    count: int,
    pool: SplitPool,
    corpus_path: str,
    corpus_sha256: str,
    tokenizer: Any,
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    multiturn_count = count * 2 // 3
    for index in range(count):
        entity, chunk = pool.unique_chunk(index, "vertical-chat")
        quote = _compact(chunk.clean_text, 200)
        style = index % 6
        if index < multiturn_count:
            family = "chat_first"
            prompt_id, template, first_question = _render_template(
                family,
                split,
                index,
                quote=quote,
                entity=entity,
            )
            first_answer = _answer_from_chunks(entity, [chunk], style)
            followup_id, followup_template, followup_base = _render_template(
                "chat_followup",
                split,
                index,
            )
            locator = _compact(chunk.clean_text, 72)
            followup = (
                f"{followup_base}\n请对照第{chunk.chapter_number}章中"
                f"“{locator}”这一处。"
            )
            answer_snippet = _compact(chunk.clean_text, 72)
            followup_answers = (
                f"原文可直接核对：“{answer_snippet}”。这处内容涉及{entity}。",
                f"与问题对应的文字是：“{answer_snippet}”。其中写到了{entity}。",
                f"片段中的依据很清楚：“{answer_snippet}”。这里谈到的是{entity}。",
                f"支撑刚才说明的是这句：“{answer_snippet}”。对象为{entity}。",
                f"从文本中可以定位到：“{answer_snippet}”。它包含{entity}的信息。",
                f"本段能够复查的表述为：“{answer_snippet}”。相关对象是{entity}。",
            )
            answer = followup_answers[style % len(followup_answers)]
            messages = [
                {"role": "user", "content": first_question},
                {"role": "assistant", "content": first_answer},
                {"role": "user", "content": followup},
                {"role": "assistant", "content": answer},
            ]
            prompt_id = f"{prompt_id}+{followup_id}"
            template = f"{template}\n---FOLLOWUP---\n{followup_template}\n{{locator}}"
        else:
            family = "chat_single"
            prompt_id, template, question = _render_template(
                family,
                split,
                index,
                quote=quote,
                entity=entity,
            )
            answer = _answer_from_chunks(entity, [chunk], style)
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        records.append(
            _record(
                split=split,
                dimension=CHAT,
                family="chat_multiturn" if index < multiturn_count else family,
                record_index=index,
                messages=messages,
                chunks=[chunk],
                corpus_path=corpus_path,
                corpus_sha256=corpus_sha256,
                tokenizer=tokenizer,
                prompt_id=prompt_id,
                prompt_text=template,
                style_id=answer_style_id(family, split, index),
                entity=entity,
                semantic_group=f"chat:{chunk.chunk_sha256}",
                fact_id=f"interaction:{chunk.chunk_sha256}",
                generalization="split_isolated_interaction",
                evidence_status="verified_exact_train_line",
                negative_type=None,
                metric="behavior",
                required_terms=(entity,),
                forbidden_terms=("可以先",),
                known_fact=False,
                needs_evidence=False,
                evidence_sufficient=True,
                seed=seed,
            )
        )
    return records


def _build_expression_records(
    *,
    split: str,
    pool: SplitPool,
    corpus_path: str,
    corpus_sha256: str,
    tokenizer: Any,
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    quotas = EXPRESSION_FAMILY_SPLIT_QUOTAS[split]
    index = 0
    for family in ("summary", "rewrite", "continuation"):
        for family_index in range(quotas[family]):
            if family == "continuation":
                entity, chunk, following = pool.continuation_pair(
                    family_index,
                    "expression-continuation",
                )
                answer = following.text.strip()
                chunks = [chunk, following]
            else:
                entity, chunk = pool.unique_chunk(family_index, f"expression-{family}")
                following = None
                answer = (
                    _summary_answer(entity, chunk, family_index)
                    if family == "summary"
                    else _rewrite_answer(entity, chunk, family_index)
                )
                chunks = [chunk]
            quote = _compact(chunk.clean_text, 220)
            prompt_id, template, question = _render_template(
                family,
                split,
                family_index,
                quote=quote,
                entity=entity,
            )
            records.append(
                _record(
                    split=split,
                    dimension=EXPRESSION,
                    family=family,
                    record_index=index,
                    messages=[
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    chunks=chunks,
                    corpus_path=corpus_path,
                    corpus_sha256=corpus_sha256,
                    tokenizer=tokenizer,
                    prompt_id=prompt_id,
                    prompt_text=template,
                    style_id=answer_style_id(family, split, family_index),
                    entity=entity,
                    semantic_group=f"expression:{family}:{chunk.chunk_sha256}",
                    fact_id=f"expression:{chunk.chunk_sha256}",
                    generalization="split_isolated_expression",
                    evidence_status="verified_exact_train_line",
                    negative_type=None,
                    metric="exact" if family == "continuation" else "behavior",
                    required_terms=() if family == "continuation" else (entity,),
                    forbidden_terms=(),
                    known_fact=False,
                    needs_evidence=False,
                    evidence_sufficient=True,
                    seed=seed,
                )
            )
            index += 1
    return records


def _build_boundary_records(
    *,
    split: str,
    count: int,
    pool: SplitPool,
    corpus_path: str,
    corpus_sha256: str,
    tokenizer: Any,
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    triplet_count = CALIBRATION_TRIPLETS[split]
    for index in range(count):
        calibration = index < triplet_count
        salt = "calibration" if calibration else "capability-boundary"
        entity, chunk = pool.term_chunk(index, salt)
        family = "boundary_need_evidence"
        base_id, base_template, base_question = _render_template(
            family,
            split,
            index,
            entity=entity,
        )
        qualifier_index = (index // max(1, len(pool.terms))) % len(BOUNDARY_QUALIFIERS)
        qualifier = BOUNDARY_QUALIFIERS[qualifier_index]
        question = f"{base_question} {qualifier}"
        request_kind = index % 3
        style = index % 6
        answer = _boundary_answer(entity, request_kind, style)
        records.append(
            _record(
                split=split,
                dimension=BOUNDARY,
                family=family,
                record_index=index,
                messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                chunks=[],
                corpus_path=corpus_path,
                corpus_sha256=corpus_sha256,
                tokenizer=tokenizer,
                prompt_id=f"{base_id}:q{qualifier_index}",
                prompt_text=f"{base_template} {qualifier}",
                style_id=answer_style_id(family, split, index),
                entity=entity,
                semantic_group=(
                    f"boundary:{split}:"
                    f"{_stable_hash('boundary-entity', entity)[:16]}:{index}"
                ),
                fact_id=f"global-claim:{chunk.chunk_sha256}",
                generalization="split_isolated_boundary",
                evidence_status="evidence_not_supplied",
                negative_type="global_claim_requires_index",
                metric="behavior",
                required_terms=(entity,),
                forbidden_terms=(),
                known_fact=False,
                needs_evidence=True,
                evidence_sufficient=False,
                calibration_triplet_id=(
                    _calibration_id(split, index) if calibration else ""
                ),
                seed=seed,
            )
        )
    return records


def build_records_from_evidence(
    *,
    evidence_by_split: Mapping[str, Sequence[EvidenceChunk]],
    chapters: Sequence[Chapter],
    corpus_text: str,
    corpus_path: str,
    corpus_sha256: str,
    tokenizer: Any,
    seed: int = FROZEN_SEED,
    validate_records: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Build the exact frozen quotas from already parsed source material."""

    output: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        pool = SplitPool(split, evidence_by_split[split], seed)
        known_evidence = load_known_core_evidence(corpus_text, chapters, split)
        direct_count = DIRECT_CORE_SPLIT_QUOTAS[split]
        core_quota = DIMENSION_SPLIT_QUOTAS[CORE][split]
        records = _build_direct_core_records(
            split=split,
            count=direct_count,
            corpus_path=corpus_path,
            corpus_sha256=corpus_sha256,
            tokenizer=tokenizer,
            known_evidence=known_evidence,
            seed=seed,
        )
        records.extend(
            _build_local_core_records(
                split=split,
                count=core_quota - direct_count,
                pool=pool,
                corpus_path=corpus_path,
                corpus_sha256=corpus_sha256,
                tokenizer=tokenizer,
                seed=seed,
            )
        )
        records.extend(
            _build_evidence_records(
                split=split,
                count=DIMENSION_SPLIT_QUOTAS[EVIDENCE][split],
                pool=pool,
                corpus_path=corpus_path,
                corpus_sha256=corpus_sha256,
                tokenizer=tokenizer,
                seed=seed,
            )
        )
        records.extend(
            _build_rag_records(
                split=split,
                count=DIMENSION_SPLIT_QUOTAS[RAG][split],
                pool=pool,
                corpus_path=corpus_path,
                corpus_sha256=corpus_sha256,
                tokenizer=tokenizer,
                seed=seed,
            )
        )
        records.extend(
            _build_chat_records(
                split=split,
                count=DIMENSION_SPLIT_QUOTAS[CHAT][split],
                pool=pool,
                corpus_path=corpus_path,
                corpus_sha256=corpus_sha256,
                tokenizer=tokenizer,
                seed=seed,
            )
        )
        records.extend(
            _build_expression_records(
                split=split,
                pool=pool,
                corpus_path=corpus_path,
                corpus_sha256=corpus_sha256,
                tokenizer=tokenizer,
                seed=seed,
            )
        )
        records.extend(
            _build_boundary_records(
                split=split,
                count=DIMENSION_SPLIT_QUOTAS[BOUNDARY][split],
                pool=pool,
                corpus_path=corpus_path,
                corpus_sha256=corpus_sha256,
                tokenizer=tokenizer,
                seed=seed,
            )
        )
        output[split] = sorted(
            records,
            key=lambda record: _stable_hash(
                "sft-v7-record-order", seed, split, record["id"]
            ),
        )
    if validate_records:
        validate_release_records(
            output,
            corpus_text=corpus_text,
            corpus_path=corpus_path,
            tokenizer=tokenizer,
        )
    return output


def _normalized_question(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE)


def _opening_sentence(text: str) -> str:
    match = re.search(r"[。！？]", text)
    return text[: match.end()] if match else text


def _require(condition: bool, code: str, message: str, remediation: str) -> None:
    if not condition:
        raise VerticalBuildError(code, message, remediation)


def validate_release_records(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    corpus_text: str,
    corpus_path: str,
    tokenizer: Any,
) -> dict[str, Any]:
    """Enforce frozen release gates without opening any external test data."""

    _require(
        set(records_by_split) == set(SPLITS),
        "SPLIT_SET_MISMATCH",
        "The generated release does not contain exactly four physical splits.",
        "Build train, val, public_diagnostic and sealed_test together.",
    )
    corpus_lines = corpus_text.splitlines()
    _, chapters = parse_complete_chapters(corpus_text)
    chapter_hashes = {chapter.start_line: chapter.source_sha256 for chapter in chapters}
    ids: Counter[str] = Counter()
    exact_questions: Counter[str] = Counter()
    normalized_questions: Counter[str] = Counter()
    question_sources: dict[str, list[str]] = defaultdict(list)
    normalized_question_sources: dict[str, list[str]] = defaultdict(list)
    general_answers: Counter[str] = Counter()
    general_answer_sources: dict[str, list[str]] = defaultdict(list)
    openings: Counter[str] = Counter()
    fixed_12_windows: Counter[str] = Counter()
    dimension_counts: Counter[str] = Counter()
    split_dimension_counts: dict[str, Counter[str]] = {
        split: Counter() for split in SPLITS
    }
    template_splits: dict[str, set[str]] = defaultdict(set)
    template_hash_splits: dict[str, set[str]] = defaultdict(set)
    style_splits: dict[str, set[str]] = defaultdict(set)
    semantic_refs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    evidence_refs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    chapter_refs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    triplet_modes: dict[str, set[str]] = defaultdict(set)
    triplet_splits: dict[str, set[str]] = defaultdict(set)
    core_entities: set[str] = set()
    answer_lengths: list[int] = []
    multiturn = 0
    rag_records = 0
    grounding_totals: Counter[str] = Counter()
    grounding_negatives: Counter[str] = Counter()
    acceptance: dict[str, Mapping[str, Any]] = {}
    known_core_exposures = 0
    known_core_fact_ids: set[str] = set()

    for split in SPLITS:
        records = records_by_split[split]
        _require(
            len(records) == SPLIT_TOTALS[split],
            "SPLIT_QUOTA_MISMATCH",
            f"Split {split} has {len(records)} records instead of {SPLIT_TOTALS[split]}.",
            "Inspect the split-level family factories and frozen quota table.",
        )
        for record in records:
            _require(
                record.get("schema_version") == SCHEMA_VERSION,
                "SCHEMA_VERSION_MISMATCH",
                "A generated record has the wrong schema version.",
                "Use only the frozen v7 catalog constants.",
            )
            _require(
                record.get("split") == split,
                "RECORD_SPLIT_MISMATCH",
                "A record was assigned to the wrong physical file.",
                "Keep split construction and serialization keyed by the same name.",
            )
            dimension = str(record.get("primary_dimension", ""))
            _require(
                dimension in DIMENSION_TOTALS,
                "UNKNOWN_DIMENSION",
                "A generated record uses an unknown primary dimension.",
                "Use one of the six frozen catalog dimension constants.",
            )
            dimension_counts[dimension] += 1
            split_dimension_counts[split][dimension] += 1
            identifier = str(record.get("id", ""))
            question = str(record.get("question", ""))
            answer = str(record.get("answer", ""))
            _require(
                bool(identifier and question.strip() and answer.strip()),
                "EMPTY_REQUIRED_TEXT",
                "A record has an empty id, question or answer.",
                "Repair the responsible family factory.",
            )
            _require(
                all(
                    re.fullmatch(r"[A-Za-z0-9_.:+-]{1,160}", str(record.get(field, "")))
                    for field in ("id", "semantic_group", "fact_id")
                ),
                "UNSAFE_RECORD_IDENTIFIER",
                "A record id, semantic group, or fact id is not log-safe.",
                "Use a stable ASCII hash instead of embedding source text or entity names.",
            )
            ids[identifier] += 1
            exact_questions[question] += 1
            normalized = _normalized_question(question)
            normalized_questions[normalized] += 1
            safe_source = f"{split}/{record.get('task_family')}/{identifier}"
            question_sources[question].append(safe_source)
            normalized_question_sources[normalized].append(safe_source)
            if record.get("task_family") == "known_core_direct":
                known_core_exposures += 1
                known_core_fact_ids.add(str(record.get("fact_id", "")))
            else:
                general_answers[answer] += 1
                general_answer_sources[answer].append(safe_source)
            openings[_opening_sentence(answer)] += 1
            compact_answer = re.sub(r"\s+", "", answer)
            for window in {
                compact_answer[offset : offset + 12]
                for offset in range(max(0, len(compact_answer) - 11))
            }:
                fixed_12_windows[window] += 1
            combined = "\n".join(
                str(message.get("content", "")) for message in record["messages"]
            )
            _require(
                not any(marker in combined for marker in BANNED_TEXT_MARKERS),
                "BANNED_TEXT_FOUND",
                "A generated record contains a frozen meta or project marker.",
                "Replace the responsible natural-language template.",
            )
            _require(
                not _is_positive_math_prompt(question),
                "POSITIVE_MATH_FOUND",
                "A generated record was classified as mathematics training.",
                "Keep positive data inside the novel domain.",
            )
            messages = record.get("messages")
            _require(
                isinstance(messages, list) and len(messages) >= 2 and len(messages) % 2 == 0,
                "MESSAGE_CONTRACT",
                "A record has an invalid conversation length.",
                "Generate alternating user and assistant turns.",
            )
            expected_roles = ["user" if index % 2 == 0 else "assistant" for index in range(len(messages))]
            _require(
                [message.get("role") for message in messages] == expected_roles,
                "ROLE_ALTERNATION",
                "A record has invalid role alternation.",
                "Start with user and alternate through the final assistant turn.",
            )
            _require(
                messages[-2]["content"] == question and messages[-1]["content"] == answer,
                "LAST_TURN_MISMATCH",
                "Top-level question or answer does not match the final conversation turn.",
                "Derive top-level fields from the final user/assistant pair.",
            )
            audit = _encoding_audit(messages, tokenizer)
            _require(
                audit == record.get("encoding_audit"),
                "ENCODING_AUDIT_DRIFT",
                "Stored encoding diagnostics do not recompute.",
                "Rebuild the dataset with the frozen tokenizer.",
            )
            _require(
                audit["sequence_tokens"] <= 512 and audit["eos_targets"] == audit["assistant_turns"],
                "ENCODING_OR_EOS_GATE",
                "A record exceeds context or has inconsistent EOS supervision.",
                "Shorten the record and keep one encoder-appended EOS per assistant turn.",
            )
            answer_lengths.append(audit["last_answer_tokens"])
            if len(messages) >= 4:
                multiturn += 1
            if dimension == RAG:
                rag_records += 1

            prompt_id = str(record.get("prompt_template_id", ""))
            style_id = str(record.get("answer_style_id", ""))
            prompt_hash = str(record.get("generation", {}).get("prompt_template_sha256", ""))
            _require(
                prompt_id.startswith(f"{split}:") and style_id.startswith(f"{split}:"),
                "SPLIT_TEMPLATE_ID_CONTRACT",
                "A prompt or answer-style id is not namespaced to its physical split.",
                "Use the split-specific frozen prompt and style banks.",
            )
            template_splits[prompt_id].add(split)
            template_hash_splits[prompt_hash].add(split)
            style_splits[style_id].add(split)

            evaluation = record.get("evaluation", {})
            required_terms = evaluation.get("required_terms", [])
            forbidden_terms = evaluation.get("forbidden_terms", [])
            _require(
                evaluation.get("metric") in {"exact", "normalized_f1", "keypoints", "behavior"},
                "INVALID_EVALUATION_METRIC",
                "A record lacks a supported public evaluation metric.",
                "Use exact, normalized_f1, keypoints or behavior.",
            )
            _require(
                all(term in answer for term in required_terms),
                "REQUIRED_TERM_MISSING",
                "A generated answer misses one of its declared evaluation keypoints.",
                "Correct the answer or its required-term contract.",
            )
            _require(
                all(term not in answer for term in forbidden_terms),
                "FORBIDDEN_TERM_PRESENT",
                "A generated answer contains a declared forbidden term.",
                "Correct the answer or its forbidden-term contract.",
            )
            if evaluation.get("known_fact"):
                _require(
                    not any(marker in answer for marker in ("资料不足", "无法确定", "不能确认", "需要检索")),
                    "KNOWN_CORE_FALSE_REFUSAL",
                    "A known-core answer incorrectly refuses an evidence-backed fact.",
                    "Answer reviewed core facts directly.",
                )
            triplet_id = str(evaluation.get("calibration_triplet_id", ""))
            if triplet_id:
                triplet_modes[triplet_id].add(str(evaluation.get("capability_mode", "")))
                triplet_splits[triplet_id].add(split)
            acceptance_id = str(evaluation.get("acceptance_case_id", ""))
            if acceptance_id:
                acceptance[acceptance_id] = record

            evidence = record.get("evidence")
            _require(
                isinstance(evidence, list),
                "EVIDENCE_NOT_LIST",
                "Evidence must be a list of exact source spans.",
                "Emit one evidence object per contiguous source line.",
            )
            for item in evidence:
                start = item.get("start_line")
                end = item.get("end_line")
                text = item.get("text")
                _require(
                    isinstance(start, int) and start == end and 1 <= start <= len(corpus_lines),
                    "EVIDENCE_LINE_RANGE",
                    "An evidence item has an invalid exact line range.",
                    "Recompute its one-based location in formal train.txt.",
                )
                _require(
                    text == corpus_lines[start - 1] and item.get("sha256") == _sha256_text(text),
                    "EVIDENCE_TEXT_OR_HASH_DRIFT",
                    "Evidence text or SHA no longer matches formal train.txt.",
                    "Stop and rebuild from the audited corpus.",
                )
                _require(
                    item.get("source_path") == corpus_path,
                    "EVIDENCE_SOURCE_PATH",
                    "Evidence does not point to the formal pretraining train corpus.",
                    "Bind every source span to the resolved formal train.txt path.",
                )
                heading = item.get("heading_line")
                _require(
                    heading in chapter_hashes and item.get("chapter_sha256") == chapter_hashes[heading],
                    "EVIDENCE_CHAPTER_HASH",
                    "Evidence chapter metadata does not recompute.",
                    "Reparse formal chapters and rebuild the evidence record.",
                )
                evidence_refs[str(item["sha256"])].append(record)
                chapter_refs[str(heading)].append(record)
            semantic_refs[str(record.get("semantic_group", ""))].append(record)
            if dimension == CORE:
                core_entities.update(record.get("coverage", {}).get("entities", []))
            if dimension in {EVIDENCE, RAG}:
                grounding_totals[dimension] += 1
                if evaluation.get("needs_evidence"):
                    grounding_negatives[dimension] += 1

    for dimension, target in DIMENSION_TOTALS.items():
        _require(
            dimension_counts[dimension] == target,
            "DIMENSION_TOTAL_MISMATCH",
            f"Dimension {dimension} has {dimension_counts[dimension]} records instead of {target}.",
            "Inspect the dimension factory quotas.",
        )
        for split in SPLITS:
            expected = DIMENSION_SPLIT_QUOTAS[dimension][split]
            _require(
                split_dimension_counts[split][dimension] == expected,
                "DIMENSION_SPLIT_QUOTA_MISMATCH",
                f"Dimension {dimension} in {split} does not match its frozen quota.",
                "Apply the 80/8/6/6 quota table exactly.",
            )

    _require(max(ids.values(), default=0) == 1, "DUPLICATE_ID", "Generated record ids are not unique.", "Include split, family, index and content in the stable id digest.")
    duplicate_question_sources = next(
        (sources for sources in question_sources.values() if len(sources) > 1),
        [],
    )
    _require(max(exact_questions.values(), default=0) == 1, "DUPLICATE_QUESTION", f"Generated questions are not exactly unique; sources={duplicate_question_sources[:4]}.", "Increase natural split-specific wording diversity.")
    duplicate_normalized_sources = next(
        (sources for sources in normalized_question_sources.values() if len(sources) > 1),
        [],
    )
    _require(max(normalized_questions.values(), default=0) == 1, "DUPLICATE_NORMALIZED_QUESTION", f"Generated questions collide after normalization; sources={duplicate_normalized_sources[:4]}.", "Increase semantic wording diversity without adding record-number wrappers.")
    repeated_answer_sources = next(
        (sources for sources in general_answer_sources.values() if len(sources) > 5),
        [],
    )
    _require(max(general_answers.values(), default=0) <= 5, "ANSWER_REPEAT_LIMIT", f"One non-core complete answer appears more than five times; sources={repeated_answer_sources[:6]}.", "Vary evidence-backed answer realization naturally; reviewed core exposure is tracked separately.")
    total = sum(len(records_by_split[split]) for split in SPLITS)
    _require(max(openings.values(), default=0) / total <= 0.02, "OPENING_SHARE_LIMIT", "A fixed answer opening exceeds two percent.", "Diversify natural first sentences across entities and evidence.")
    top_fixed_12, top_fixed_12_count = max(
        fixed_12_windows.items(),
        key=lambda item: item[1],
        default=("", 0),
    )
    largest_fixed_12_share = top_fixed_12_count / total
    top_fixed_12_families: Counter[str] = Counter()
    for records in records_by_split.values():
        for record in records:
            compact_answer = re.sub(r"\s+", "", str(record["answer"]))
            if top_fixed_12 and top_fixed_12 in compact_answer:
                top_fixed_12_families[str(record["task_family"])] += 1
    _require(largest_fixed_12_share <= 0.02, "FIXED_12_PHRASE_SHARE", f"A fixed 12-character answer phrase exceeds two percent; phrase_sha256={_sha256_text(top_fixed_12)}, count={top_fixed_12_count}, families={dict(top_fixed_12_families)}.", "Remove repeated scope disclaimers and diversify natural answer realization.")
    _require(all(len(splits) == 1 for splits in template_splits.values()), "PROMPT_ID_SPLIT_LEAK", "A prompt template id is reused across physical splits.", "Namespace all prompt ids by split.")
    _require(all(len(splits) == 1 for splits in template_hash_splits.values()), "PROMPT_TEXT_SPLIT_LEAK", "Unrendered prompt wording is reused across physical splits.", "Use distinct wording banks for every split.")
    _require(all(len(splits) == 1 for splits in style_splits.values()), "ANSWER_STYLE_SPLIT_LEAK", "An answer-style id is reused across physical splits.", "Namespace all style ids by split.")

    def cross_split_allowed(references: Mapping[str, Sequence[Mapping[str, Any]]]) -> bool:
        for grouped in references.values():
            grouped_splits = {str(record["split"]) for record in grouped}
            if len(grouped_splits) <= 1:
                continue
            if not all(
                record["primary_dimension"] == CORE
                and record["generalization_policy"] == "seen_fact_unseen_wording"
                for record in grouped
            ):
                return False
        return True

    _require(cross_split_allowed(semantic_refs), "SEMANTIC_SPLIT_LEAK", "A non-core semantic group crosses physical splits.", "Partition non-core semantics by chapter before generation.")
    _require(cross_split_allowed(evidence_refs), "EVIDENCE_SPLIT_LEAK", "A non-core evidence hash crosses physical splits.", "Reserve reviewed cross-split lines for tracked core cards only.")
    _require(cross_split_allowed(chapter_refs), "CHAPTER_SPLIT_LEAK", "A non-core chapter crosses physical splits.", "Exclude reviewed core chapters from local evidence pools.")
    _require(40 <= len(core_entities) <= 60, "CORE_COVERAGE", "Core coverage is outside the frozen 40-60 range.", "Balance local core cards across the 50-item ontology.")
    _require(multiturn >= MINIMUM_MULTITURN_RECORDS, "MULTITURN_QUOTA", "Fewer than 1,200 records are true multi-turn conversations.", "Keep two thirds of vertical-chat records multi-turn.")
    _require(rag_records >= MINIMUM_RAG_RECORDS, "RAG_QUOTA", "Fewer than 1,000 records contain multi-passage RAG tasks.", "Preserve all frozen RAG records with two to four chunks.")
    for dimension in (EVIDENCE, RAG):
        share = grounding_negatives[dimension] / grounding_totals[dimension]
        _require(0.15 <= share <= 0.20, "NEGATIVE_SHARE", f"Dimension {dimension} has an invalid grounding-negative share.", "Keep insufficient or distractor examples between 15% and 20%.")

    required_acceptance = {
        fact.acceptance_case_id: fact.canonical_question
        for fact in KNOWN_CORE_FACTS
        if fact.acceptance_case_id
    }
    _require(set(acceptance) == set(required_acceptance), "KNOWN_CORE_ACCEPTANCE_SET", "The four fixed known-core acceptance cases are incomplete.", "Emit all reviewed direct public questions exactly once.")
    for acceptance_id, expected_question in required_acceptance.items():
        record = acceptance[acceptance_id]
        _require(record["split"] == "public_diagnostic" and record["question"] == expected_question, "KNOWN_CORE_ACCEPTANCE_WORDING", "A fixed known-core acceptance question changed or left public diagnostic.", "Restore the canonical direct question and public split assignment.")

    expected_triplets = sum(CALIBRATION_TRIPLETS.values())
    complete_triplets = 0
    for triplet_id, modes in triplet_modes.items():
        _require(len(triplet_splits[triplet_id]) == 1, "TRIPLET_SPLIT_LEAK", "A calibration triplet crosses physical splits.", "Construct every three-mode calibration group inside one split.")
        _require(modes == {"known_core", "needs_evidence", "grounded_answer"}, "INCOMPLETE_CALIBRATION_TRIPLET", "A calibration triplet is missing a boundary mode.", "Emit known, needs-evidence and evidence-recovery cases together.")
        complete_triplets += 1
    _require(complete_triplets == expected_triplets, "CALIBRATION_TRIPLET_COUNT", "The frozen calibration-triplet count is incomplete.", "Keep 100/10/10/10 triplets across the four splits.")

    medium = sum(33 <= value <= 96 for value in answer_lengths)
    long = sum(97 <= value <= 160 for value in answer_lengths)
    _require(medium / total >= 0.50, "MEDIUM_ANSWER_SHARE", "Fewer than half of answers fall in the 33-96 BPE target band.", "Lengthen short answers with evidence-bounded explanation.")
    _require(long / total >= 0.10, "LONG_ANSWER_SHARE", f"Fewer than ten percent of answers fall in the 97-160 BPE target band; count={long}, total={total}, share={long / total:.4f}.", "Add substantive multi-evidence synthesis without copying full passages.")
    return {
        "record_count": total,
        "split_counts": {split: len(records_by_split[split]) for split in SPLITS},
        "dimension_counts": dict(dimension_counts),
        "core_coverage": len(core_entities),
        "multiturn_records": multiturn,
        "rag_records": rag_records,
        "negative_shares": {
            dimension: grounding_negatives[dimension] / grounding_totals[dimension]
            for dimension in (EVIDENCE, RAG)
        },
        "medium_answer_share": medium / total,
        "long_answer_share": long / total,
        "known_core_false_refusals": 0,
        "fixed_acceptance_cases": len(acceptance),
        "complete_calibration_triplets": complete_triplets,
        "known_core_fact_count": len(known_core_fact_ids),
        "known_core_exposures": known_core_exposures,
        "largest_fixed_12_phrase_share": largest_fixed_12_share,
    }


def _jsonl_text(records: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )


def write_release(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    output_dir: Path,
    corpus_path: Path,
    corpus_sha256: str,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    seed: int,
) -> dict[str, Any]:
    """Write four split artifacts, then write the manifest completion marker."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    split_files: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        destination = output_dir / OUTPUT_NAMES[split]
        atomic_write_text(destination, _jsonl_text(records_by_split[split]))
        split_files[split] = {
            "path": destination.name,
            "count": len(records_by_split[split]),
            "schema_version": SCHEMA_VERSION,
            "sha256": file_sha256(destination),
        }
    identity_payload = {
        split: [record["id"] for record in records_by_split[split]]
        for split in SPLITS
    }
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "record_schema_version": SCHEMA_VERSION,
        "frozen_status": "frozen_unspent",
        "seed": seed,
        "record_count": sum(item["count"] for item in split_files.values()),
        "split_files": split_files,
        "split_totals": SPLIT_TOTALS,
        "dimension_totals": DIMENSION_TOTALS,
        "dimension_split_quotas": DIMENSION_SPLIT_QUOTAS,
        "known_core": {
            "reviewed_fact_count": len(KNOWN_CORE_FACTS),
            "exposure_total": DIRECT_CORE_TOTAL,
            "split_exposures": DIRECT_CORE_SPLIT_QUOTAS,
            "fixed_acceptance_case_ids": sorted(
                fact.acceptance_case_id
                for fact in KNOWN_CORE_FACTS
                if fact.acceptance_case_id
            ),
            "false_refusal_gate": 0,
        },
        "dataset_identity_sha256": canonical_json_sha256(identity_payload),
        "source": {
            "path": _portable_artifact_path(corpus_path, role="source"),
            "sha256": corpus_sha256,
            "allowed_split": "formal_pretrain_train",
        },
        "tokenizer": {
            "path": _portable_artifact_path(tokenizer_path, role="tokenizer"),
            "sha256": tokenizer_sha256,
            "context_limit": 512,
            "eos_appended_by_encoder": True,
        },
        "sealed_policy": {
            "body_in_manifest": False,
            "body_in_logs": False,
            "post_freeze_access": "explicit_one_time_final_evaluator_only",
        },
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def build_release(
    *,
    corpus_path: Path = DEFAULT_CORPUS,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = FROZEN_SEED,
    loggers: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build, validate and atomically freeze the v7 release."""

    data_logger = loggers.get("data") if loggers else None
    validation_logger = loggers.get("validation") if loggers else None
    sft_logger = loggers.get("sft") if loggers else None
    try:
        corpus_text = corpus_path.read_text(encoding="utf-8")
        tokenizer = BPETokenizer.load(tokenizer_path)
    except Exception as error:
        raise VerticalBuildError(
            "INPUT_LOAD_FAILURE",
            "The formal corpus or frozen tokenizer could not be loaded.",
            "Verify both paths, UTF-8 corpus encoding and tokenizer JSON integrity.",
        ) from error
    corpus_sha = file_sha256(corpus_path)
    tokenizer_sha = file_sha256(tokenizer_path)
    if data_logger:
        data_logger.info(
            "formal build inputs loaded",
            extra={"context": {"corpus_sha256": corpus_sha, "tokenizer_sha256": tokenizer_sha, "seed": seed}},
        )
    evidence_by_split, partitions = extract_evidence_chunks(corpus_text, seed=seed)
    if data_logger:
        data_logger.info(
            "split evidence pools prepared",
            extra={"context": {"chapter_counts": {split: len(partitions[split]) for split in SPLITS}, "evidence_counts": {split: len(evidence_by_split[split]) for split in SPLITS}}},
        )
    records = build_records_from_evidence(
        evidence_by_split=evidence_by_split,
        chapters=[chapter for split in SPLITS for chapter in partitions[split]],
        corpus_text=corpus_text,
        corpus_path=str(corpus_path.resolve()),
        corpus_sha256=corpus_sha,
        tokenizer=tokenizer,
        seed=seed,
        validate_records=False,
    )
    summary = validate_release_records(
        records,
        corpus_text=corpus_text,
        corpus_path=str(corpus_path.resolve()),
        tokenizer=tokenizer,
    )
    if validation_logger:
        validation_logger.info("release gates passed", extra={"context": summary})
    manifest = write_release(
        records,
        output_dir=output_dir,
        corpus_path=corpus_path,
        corpus_sha256=corpus_sha,
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=tokenizer_sha,
        seed=seed,
    )
    if sft_logger:
        sft_logger.info(
            "four-split v7 release frozen",
            extra={"context": {"record_count": manifest["record_count"], "dataset_identity_sha256": manifest["dataset_identity_sha256"], "split_file_sha256s": {split: item["sha256"] for split, item in manifest["split_files"].items()}}},
        )
    return manifest, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--data-log-level", default="INFO")
    parser.add_argument("--validation-log-level", default="INFO")
    parser.add_argument("--sft-log-level", default="INFO")
    parser.add_argument("--orchestrator-log-level", default="INFO")
    parser.add_argument("--log-max-bytes", type=int, default=5 * 1024 * 1024)
    parser.add_argument("--log-backups", type=int, default=3)
    parser.add_argument("--no-console-log", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = generate_run_id("sft-v7-build")
    levels = resolve_module_log_levels(
        {
            "data": args.data_log_level,
            "validation": args.validation_log_level,
            "sft": args.sft_log_level,
            "orchestrator": args.orchestrator_log_level,
        },
        env_prefix="SFT_V7_BUILD_LOG_LEVEL",
    )
    loggers = configure_module_loggers(
        args.log_dir,
        run_id,
        levels,
        max_bytes=args.log_max_bytes,
        backup_count=args.log_backups,
        console=not args.no_console_log,
    )
    try:
        build_release(
            corpus_path=args.corpus,
            tokenizer_path=args.tokenizer,
            output_dir=args.output_dir,
            seed=args.seed,
            loggers=loggers,
        )
    except VerticalBuildError as error:
        loggers["orchestrator"].error(
            "v7 build failed",
            extra={"context": {"error_code": error.code, "message": error.message, "remediation": error.remediation}},
        )
        return 2
    except Exception as error:
        loggers["orchestrator"].error(
            "v7 build failed unexpectedly",
            extra={"context": {"error_code": "UNEXPECTED_BUILD_FAILURE", "error_type": type(error).__name__, "remediation": "Inspect the named module log, fix the cause, and rerun from formal inputs."}},
        )
        return 2
    finally:
        close_module_loggers(loggers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
