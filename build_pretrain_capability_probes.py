"""Build reproducible, corpus-only probes for pretraining capability audits.

The generated artifact is evaluation-only: it must never be mixed into
pretraining or SFT.  Every expected answer is copied from the frozen formal
``train.txt`` or ``val.txt`` and is accompanied by an exact
SHA-256 and zero-based, end-exclusive character offsets.  The builder does not
import or inspect any SFT dataset, catalogue, checkpoint, or model output.
The formal test split remains sealed: its path may be named by the manifest but
the builder never opens, hashes, parses, or samples it.

Entity discovery, mention counts, chapter counts and high/low tiers use only
``train.txt``; ``val.txt`` supplies held-out contexts but cannot influence any
frequency statistic.  The top-level ``cases`` array is the formal validation
set accepted by ``evaluate_pretrain_capabilities.py`` while
``calibration_cases`` retains train-only diagnostics.  The artifact records a
canonical cases hash and the byte-exact prompts-file hash.  Prompts are written
first and the JSON artifact last, so JSON is the completion marker.

Diagnostics are split into independently configurable JSONL modules:

* ``data`` records input verification and corpus-derived entity statistics;
* ``validation`` records leakage and provenance gate outcomes;
* ``orchestrator`` records build start, completion, and failures.

Logs default to ``logs/pretrain_capability_probes/<run_id>.<module>.jsonl``.
Use ``--data-log-level``, ``--validation-log-level`` and
``--orchestrator-log-level`` (or ``GPT_LOG_LEVEL_DATA``,
``GPT_LOG_LEVEL_VALIDATION`` and ``GPT_LOG_LEVEL_ORCHESTRATOR``) to filter one
area without enabling global debug noise.  ``OFF`` disables a module.  Files
rotate according to ``--log-max-bytes`` and ``--log-backup-count`` and may be
exported as ordinary JSON Lines.  Reproduce a build with the recorded manifest
hash, input hashes, seed and CLI counts.  No corpus excerpts are written to
logs; the shared logger also redacts common credential fields.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Mapping, Sequence

from prepare_corpus_v4 import Chapter, parse_complete_chapters
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


SCHEMA_VERSION = "pretrain-capability-probes/v1"
DEFAULT_MANIFEST = Path("data/cloud_v4/corpus_manifest.json")
DEFAULT_OUTPUT = Path("data/eval/pretrain_capability_probes.json")
DEFAULT_PROMPTS_OUTPUT = Path("data/eval/pretrain_capability_prompts.txt")
DEFAULT_LOG_DIR = Path("logs/pretrain_capability_probes")
DEFAULT_SEED = 20260829
SPLITS = ("train", "val")
FORMAL_SPLITS = ("train", "val", "test")
HELD_OUT_SPLITS = {"val"}

# These are language-shape rules, not a novel knowledge catalogue.  A string
# must first occur as a speaker attribution in the frozen corpus; this list only
# rejects obvious adverbs and accepts name/title-like forms.
COMMON_SURNAME_INITIALS = set(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华"
    "金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花"
    "方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐"
    "于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹"
    "狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席"
    "季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌"
    "霍虞万支柯管卢莫经房裘缪干解应宗丁宣邓郁单杭洪包诸左"
    "石崔吉龚程嵇邢滑裴陆荣翁荀羊惠甄曲封芮储靳汲邴糜松井"
    "段富巫乌焦巴弓牧隗山谷车侯宓蓬全班仰秋仲伊宫宁仇栾暴"
    "甘钭厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲台"
    "从鄂索咸籍赖卓蔺屠蒙池乔阴胥能苍双闻莘党翟谭贡劳逄姬"
    "申扶堵冉宰郦雍郤璩桑桂濮牛寿通边扈燕冀浦尚农温别庄晏"
    "柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿"
    "满弘匡国文寇广禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融"
    "冷訾辛阚那简饶空曾毋沙乜养鞠须丰巢关蒯相查后荆红游竺"
    "权逯盖益桓公药魂烛海纳雅紫薰青墨法加"
)
PERSON_TITLE_SUFFIXES = (
    "长老",
    "尊者",
    "宗主",
    "阁主",
    "族长",
    "院长",
    "殿主",
    "门主",
    "盟主",
    "谷主",
    "城主",
    "先生",
    "小姐",
    "公子",
    "前辈",
    "大人",
    "药老",
    "医仙",
)
GENERIC_NAME_FORMS = {
    "一道",
    "不知",
    "旋即",
    "轻声",
    "沉声",
    "冷笑",
    "苦笑",
    "微笑",
    "喃喃",
    "无数",
    "突然",
    "忽然",
    "皱眉",
    "开口",
    "大笑",
    "低声",
    "叹息",
    "当下",
    "急忙",
    "方才",
    "嘿嘿",
    "含笑",
    "淡笑",
    "冷喝",
    "怒喝",
    "惊声",
    "恭声",
    "解释",
    "大长老",
}
GENERIC_NAME_FRAGMENTS = (
    "不知",
    "也是",
    "却是",
    "忽然",
    "突然",
    "连忙",
    "急忙",
    "方才",
    "再度",
    "微微",
    "轻轻",
    "淡淡",
    "沉吟",
    "笑着",
    "叹息",
    "嫣然",
    "嘿嘿",
    "拱手",
    "撇嘴",
    "咧嘴",
    "嘀咕",
    "喃喃",
    "愕然",
    "一位",
)
GENERIC_REFERENCE_SUFFIXES = ("黑袍人", "黑衣人", "女人", "老者", "青年", "少女")
INVALID_NAME_CHARACTERS = set("的一是知着劲力声")
INVALID_NAME_PREFIXES = (
    "那",
    "这",
    "其",
    "对着",
    "周围",
    "应该",
    "都是",
    "终于",
    "仰天",
    "强猛",
    "空间",
    "和善",
    "阴",
    "一位",
    "随着",
    "听得",
)
INVALID_NAME_SUFFIXES = (
    "笑",
    "微",
    "轻",
    "淡",
    "干",
    "偏头",
    "皱眉",
    "含",
    "叹",
    "追",
    "提醒",
    "安慰",
    "玩",
    "嘀咕",
    "沉吟",
    "笑着",
    "连忙",
    "急忙",
    "方才",
    "拱手",
    "撇嘴",
    "咧嘴",
    "愕然",
    "嘿嘿",
    "嫣然",
    "喝",
)
SPEECH_MODIFIERS = (
    "笑吟吟的",
    "若有所思的",
    "不置可否的",
    "微微一笑",
    "沉吟了一下",
    "沉吟片刻",
    "淡淡的",
    "缓缓的",
    "平静的",
    "冷冷的",
    "轻声",
    "低声",
    "沉声",
    "厉声",
    "冷笑",
    "苦笑",
    "微笑",
    "含笑",
    "淡笑",
    "怒声",
    "冷声",
    "柔声",
    "恭声",
    "惊声",
    "微微",
    "淡淡",
    "缓缓",
    "轻轻",
)
SPEECH_VERBS = (
    "开口说道",
    "开口笑道",
    "开口问道",
    "开口道",
    "低声说道",
    "沉声说道",
    "轻声说道",
    "说道",
    "笑道",
    "问道",
    "喝道",
    "答道",
    "怒道",
    "吼道",
    "道",
)
_MODIFIER_PATTERN = "|".join(
    re.escape(value) for value in sorted(SPEECH_MODIFIERS, key=len, reverse=True)
)
_VERB_PATTERN = "|".join(
    re.escape(value) for value in sorted(SPEECH_VERBS, key=len, reverse=True)
)
SPEAKER_ATTRIBUTION_PATTERN = re.compile(
    rf"(?:^|[“”。，！？；：\s])(?P<name>[\u4e00-\u9fff]{{2,4}}?)"
    rf"(?:(?:{_MODIFIER_PATTERN}))?(?:{_VERB_PATTERN})"
)


class ProbeBuildError(RuntimeError):
    """Raised when a probe cannot satisfy its reproducibility or leakage gate."""


@dataclass(frozen=True)
class ParagraphSpan:
    split: str
    source_path: Path
    source_sha256: str
    chapter: Chapter
    chapter_split_start: int
    text: str
    chapter_char_start: int
    split_char_start: int
    split_char_end: int
    source_line: int


@dataclass(frozen=True)
class EntityStat:
    text: str
    entity_type: str
    train_attribution_count: int
    train_count: int
    train_chapter_count: int
    frequency_tier: str = ""


def _stable_digest(*parts: object) -> str:
    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_input_path(manifest_path: Path, value: str) -> Path:
    supplied = Path(value)
    if supplied.is_absolute():
        return supplied
    project_root = manifest_path.resolve().parents[2]
    candidates = (Path.cwd() / supplied, project_root / supplied, manifest_path.parent / supplied)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return (project_root / supplied).resolve()


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProbeBuildError(f"cannot read corpus manifest {manifest_path}: {error}") from error
    if payload.get("status") != "ready":
        raise ProbeBuildError("formal corpus manifest is not marked ready")
    split_payload = payload.get("splits")
    if not isinstance(split_payload, dict) or set(FORMAL_SPLITS).difference(split_payload):
        raise ProbeBuildError("corpus manifest must define train, val and test splits")
    return payload


def _verify_and_load_splits(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    data_logger: logging.Logger,
) -> tuple[dict[str, str], dict[str, Path]]:
    texts: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for split in SPLITS:
        metadata = manifest["splits"][split]
        source_path = _resolve_input_path(manifest_path, str(metadata["path"]))
        if not source_path.is_file():
            raise ProbeBuildError(f"missing formal {split} split: {source_path}")
        actual_sha256 = file_sha256(source_path)
        expected_sha256 = str(metadata["sha256"])
        if actual_sha256 != expected_sha256:
            raise ProbeBuildError(
                f"formal {split} split SHA-256 mismatch: expected {expected_sha256}, "
                f"calculated {actual_sha256}"
            )
        text = source_path.read_text(encoding="utf-8")
        texts[split] = text
        paths[split] = source_path
        data_logger.info(
            "formal split checksum verified",
            extra={
                "context": {
                    "split": split,
                    "path": str(source_path),
                    "sha256": actual_sha256,
                    "characters": len(text),
                }
            },
        )
    return texts, paths


def _chapter_positions(text: str, chapters: Sequence[Chapter]) -> dict[str, int]:
    positions: dict[str, int] = {}
    cursor = 0
    for chapter in chapters:
        position = text.find(chapter.source_text, cursor)
        if position < 0:
            raise ProbeBuildError(
                f"cannot locate parsed chapter {chapter.section_id} in its formal split"
            )
        positions[chapter.section_id] = position
        cursor = position + len(chapter.source_text)
    return positions


def _paragraph_spans(
    split: str,
    text: str,
    source_path: Path,
    source_sha256: str,
    *,
    minimum_characters: int = 48,
    maximum_characters: int = 360,
) -> list[ParagraphSpan]:
    _, chapters = parse_complete_chapters(text)
    chapter_positions = _chapter_positions(text, chapters)
    spans: list[ParagraphSpan] = []
    for chapter in chapters:
        chapter_split_start = chapter_positions[chapter.section_id]
        local_cursor = 0
        for local_line_index, raw_line in enumerate(
            chapter.source_text.splitlines(keepends=True)
        ):
            content_without_newline = raw_line.rstrip("\r\n")
            clean = content_without_newline.strip()
            leading = len(content_without_newline) - len(content_without_newline.lstrip())
            chapter_start = local_cursor + leading
            split_start = chapter_split_start + chapter_start
            local_cursor += len(raw_line)
            if not minimum_characters <= len(clean) <= maximum_characters:
                continue
            if clean == chapter.title or set(clean) == {"-"}:
                continue
            if "http://" in clean or "https://" in clean:
                continue
            split_end = split_start + len(clean)
            if text[split_start:split_end] != clean:
                raise ProbeBuildError(
                    f"paragraph offset verification failed in {split} at {split_start}"
                )
            spans.append(
                ParagraphSpan(
                    split=split,
                    source_path=source_path,
                    source_sha256=source_sha256,
                    chapter=chapter,
                    chapter_split_start=chapter_split_start,
                    text=clean,
                    chapter_char_start=chapter_start,
                    split_char_start=split_start,
                    split_char_end=split_end,
                    source_line=chapter.range_start_line + local_line_index,
                )
            )
    return spans


def _looks_like_person_name(name: str) -> bool:
    if not 2 <= len(name) <= 4 or name in GENERIC_NAME_FORMS:
        return False
    if any(fragment in name for fragment in GENERIC_NAME_FRAGMENTS):
        return False
    if name.endswith(GENERIC_REFERENCE_SUFFIXES):
        return False
    if any(character in name for character in INVALID_NAME_CHARACTERS):
        return False
    if name.startswith(INVALID_NAME_PREFIXES) or name.endswith(INVALID_NAME_SUFFIXES):
        return False
    return name[0] in COMMON_SURNAME_INITIALS or name.endswith(PERSON_TITLE_SUFFIXES)


def extract_corpus_entities(
    texts: Mapping[str, str],
    paragraphs: Mapping[str, Sequence[ParagraphSpan]],
    *,
    minimum_occurrences: int = 20,
    minimum_attributions: int = 2,
) -> list[EntityStat]:
    """Infer person-like entities and every frequency statistic from train only."""
    if "train" not in texts or "train" not in paragraphs:
        raise ProbeBuildError("entity statistics require the formal train split")
    train_text = texts["train"]
    _, train_chapters = parse_complete_chapters(train_text)
    attribution_counts: Counter[str] = Counter()
    attribution_counts.update(
        match.group("name") for match in SPEAKER_ATTRIBUTION_PATTERN.finditer(train_text)
    )
    shape_candidates = {
        name
        for name, count in attribution_counts.items()
        if count >= minimum_attributions and _looks_like_person_name(name)
    }
    # Drop every extended form whose shorter prefix is independently observed as
    # a speaker.  This removes action-tailed false positives such as "萧炎偏头"
    # without requiring any book-specific name list.
    candidates = {
        name
        for name in shape_candidates
        if not any(
            name.startswith(prefix)
            and len(name) > len(prefix)
            and prefix in shape_candidates
            for prefix in (name[:2], name[:3])
        )
    }
    result: list[EntityStat] = []
    for name in sorted(candidates):
        train_count = train_text.count(name)
        if train_count < minimum_occurrences:
            continue
        chapter_ids = {
            chapter.section_id
            for chapter in train_chapters
            if name in chapter.source_text
        }
        result.append(
            EntityStat(
                text=name,
                entity_type="person_speaker",
                train_attribution_count=attribution_counts[name],
                train_count=train_count,
                train_chapter_count=len(chapter_ids),
            )
        )
    return sorted(result, key=lambda item: (-item.train_count, item.text))


def stratify_entities(
    entities: Sequence[EntityStat],
    *,
    minimum_tier_size: int = 4,
    maximum_tier_size: int = 24,
) -> tuple[list[EntityStat], list[EntityStat]]:
    """Return disjoint top- and bottom-frequency person entity strata."""
    if len(entities) < minimum_tier_size * 2:
        raise ProbeBuildError(
            f"need at least {minimum_tier_size * 2} corpus-derived person entities, "
            f"found {len(entities)}"
        )
    tier_size = min(maximum_tier_size, max(minimum_tier_size, len(entities) // 3))
    high_raw = list(entities[:tier_size])
    low_raw = list(entities[-tier_size:])
    high = [EntityStat(**{**item.__dict__, "frequency_tier": "high"}) for item in high_raw]
    low = [EntityStat(**{**item.__dict__, "frequency_tier": "low"}) for item in low_raw]
    if {item.text for item in high}.intersection(item.text for item in low):
        raise ProbeBuildError("high- and low-frequency entity strata overlap")
    return high, low


def _source_payload(span: ParagraphSpan) -> dict[str, Any]:
    return {
        "split": span.split,
        "role": "held_out" if span.split in HELD_OUT_SPLITS else "calibration",
        "path": str(span.source_path),
        "file_sha256": span.source_sha256,
        "chapter_section_id": span.chapter.section_id,
        "chapter_number": span.chapter.chapter_number,
        "chapter_title": span.chapter.title,
        "chapter_sha256": span.chapter.source_sha256,
        "source_line": span.source_line,
    }


def _evidence_payload(
    span: ParagraphSpan,
    evidence_text: str,
    *,
    paragraph_relative_start: int = 0,
) -> dict[str, Any]:
    split_start = span.split_char_start + paragraph_relative_start
    chapter_start = span.chapter_char_start + paragraph_relative_start
    return {
        "text": evidence_text,
        "sha256": _sha256_text(evidence_text),
        "offset_unit": "utf8_decoded_unicode_codepoint",
        "offset_interval": "zero_based_end_exclusive",
        "split_char_start": split_start,
        "split_char_end": split_start + len(evidence_text),
        "chapter_char_start": chapter_start,
        "chapter_char_end": chapter_start + len(evidence_text),
    }


def _continuation_candidate(span: ParagraphSpan, seed: int) -> dict[str, Any] | None:
    text = span.text
    if len(text) < 100:
        return None
    target_minimum = 24
    desired_boundary = min(112, len(text) - target_minimum)
    punctuation_positions = [
        index + 1
        for index, character in enumerate(text)
        if character in "，。！？；…" and 56 <= index + 1 <= len(text) - target_minimum
    ]
    if punctuation_positions:
        boundary = min(punctuation_positions, key=lambda value: (abs(value - desired_boundary), value))
    else:
        boundary = desired_boundary
    target_limit = min(len(text), boundary + 72)
    ending_positions = [
        index + 1
        for index, character in enumerate(text[boundary:target_limit], start=boundary)
        if character in "。！？；…" and index + 1 >= boundary + target_minimum
    ]
    target_end = ending_positions[0] if ending_positions else target_limit
    prompt_start = max(0, boundary - 128)
    prompt = text[prompt_start:boundary]
    target = text[boundary:target_end]
    if len(prompt) < 48 or len(target) < target_minimum or target in prompt:
        return None
    evidence = prompt + target
    digest = _stable_digest(seed, "continuation", span.split, span.source_sha256, span.split_char_start + prompt_start)
    return {
        "id": f"pretrain_{digest[:20]}",
        "probe_type": "held_out_continuation",
        "capability": "novel_next_text_prediction",
        "prompt": prompt,
        "expected": {"continuation": target},
        "source": _source_payload(span),
        "evidence": _evidence_payload(
            span,
            evidence,
            paragraph_relative_start=prompt_start,
        ),
    }


def _pick_distractors(
    correct: EntityStat,
    tier: Sequence[EntityStat],
    prompt: str,
    *,
    count: int,
    seed: int,
    identity: str,
) -> tuple[list[EntityStat], dict[str, Any]]:
    eligible = [
        item
        for item in tier
        if item.text != correct.text and item.text not in prompt
    ]

    def match_rank(item: EntityStat) -> tuple[object, ...]:
        length_delta = abs(len(item.text) - len(correct.text))
        ratio = max(item.train_count, correct.train_count) / max(
            min(item.train_count, correct.train_count), 1
        )
        return (
            0 if length_delta == 0 and ratio <= 2.0 else 1,
            length_delta,
            0 if ratio <= 2.0 else 1,
            abs(item.train_count - correct.train_count) / max(correct.train_count, 1),
            _stable_digest(seed, identity, item.text),
        )

    eligible.sort(key=match_rank)
    selected = eligible[:count]
    per_distractor = []
    for item in selected:
        ratio = max(item.train_count, correct.train_count) / max(
            min(item.train_count, correct.train_count), 1
        )
        per_distractor.append(
            {
                "text": item.text,
                "character_length": len(item.text),
                "character_length_delta": abs(len(item.text) - len(correct.text)),
                "train_count": item.train_count,
                "train_count_ratio": ratio,
                "within_narrow_train_frequency_band": ratio <= 2.0,
            }
        )
    quality = {
        "strategy": (
            "same_entity_type_and_frequency_tier_then_minimum_character_length_delta_"
            "then_minimum_relative_train_frequency_gap"
        ),
        "correct_character_length": len(correct.text),
        "correct_train_count": correct.train_count,
        "exact_character_length_matches": sum(
            item["character_length_delta"] == 0 for item in per_distractor
        ),
        "narrow_train_frequency_matches": sum(
            item["within_narrow_train_frequency_band"] for item in per_distractor
        ),
        "narrow_train_frequency_ratio_max": 2.0,
        "tokenizer_used_for_matching": False,
        "token_length_correction": (
            "Character length is only a build-time proxy. The evaluator must correct "
            "token-length effects with total log-probability, mean token log-probability, "
            "rank, and reciprocal-rank metrics under the formal protocol."
        ),
        "distractors": per_distractor,
    }
    return selected, quality


def _cloze_candidates_for_span(
    span: ParagraphSpan,
    tiers: Mapping[str, Sequence[EntityStat]],
    *,
    candidate_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for tier_name, tier_entities in tiers.items():
        for entity in tier_entities:
            occurrences = [match.start() for match in re.finditer(re.escape(entity.text), span.text)]
            if len(occurrences) != 1:
                continue
            answer_offset = occurrences[0]
            context_start = max(0, answer_offset - 128)
            prompt = span.text[context_start:answer_offset]
            if len(prompt) < 16 or entity.text in prompt:
                continue
            identity = f"{span.split}:{span.split_char_start}:{entity.text}"
            distractors, matching_quality = _pick_distractors(
                entity,
                tier_entities,
                prompt,
                count=candidate_count - 1,
                seed=seed,
                identity=identity,
            )
            if len(distractors) != candidate_count - 1:
                continue
            candidates = [entity, *distractors]
            random.Random(int(_stable_digest(seed, identity, "candidate-order")[:16], 16)).shuffle(candidates)
            answer_index = next(
                index for index, candidate in enumerate(candidates) if candidate.text == entity.text
            )
            digest = _stable_digest(seed, "cloze", identity)
            output.append(
                {
                    "id": f"pretrain_{digest[:20]}",
                    "probe_type": "cloze_candidate_ranking",
                    "capability": "corpus_entity_prediction",
                    "prompt": prompt,
                    "candidates": [
                        {
                            "text": candidate.text,
                            "entity_type": candidate.entity_type,
                            "frequency_tier": candidate.frequency_tier,
                            "train_count": candidate.train_count,
                        }
                        for candidate in candidates
                    ],
                    "expected": {
                        "text": entity.text,
                        "candidate_index": answer_index,
                    },
                    "entity": {
                        "text": entity.text,
                        "entity_type": entity.entity_type,
                        "frequency_tier": tier_name,
                        "train_count": entity.train_count,
                        "train_attribution_count": entity.train_attribution_count,
                        "train_chapter_count": entity.train_chapter_count,
                    },
                    "distractor_matching": matching_quality,
                    "source": _source_payload(span),
                    "evidence": {
                        **_evidence_payload(
                            span,
                            prompt + entity.text,
                            paragraph_relative_start=context_start,
                        ),
                        "answer_char_start": len(prompt),
                        "answer_char_end": len(prompt) + len(entity.text),
                    },
                }
            )
    return output


def _select_diverse(
    candidates: Sequence[dict[str, Any]],
    count: int,
    *,
    seed: int,
    identity: str,
    unique_entity: bool,
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: _stable_digest(seed, identity, item["id"]),
    )
    selected: list[dict[str, Any]] = []
    used_chapters: set[str] = set()
    used_entities: set[str] = set()
    for require_new_chapter, require_new_entity in ((True, unique_entity), (False, unique_entity), (False, False)):
        for item in ordered:
            if item in selected:
                continue
            chapter_key = f"{item['source']['split']}:{item['source']['chapter_section_id']}"
            entity_text = item.get("entity", {}).get("text", "")
            if require_new_chapter and chapter_key in used_chapters:
                continue
            if require_new_entity and entity_text in used_entities:
                continue
            selected.append(item)
            used_chapters.add(chapter_key)
            if entity_text:
                used_entities.add(entity_text)
            if len(selected) == count:
                return selected
    return selected


def _evaluator_case(probe: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve evaluator's four required fields plus auditable metadata."""
    return {
        "id": probe["id"],
        "context": probe["prompt"],
        "candidates": [candidate["text"] for candidate in probe["candidates"]],
        "correct": probe["expected"]["text"],
        "metadata": {
            "source": probe["source"],
            "source_split": probe["source"]["split"],
            "chapter_sha256": probe["source"]["chapter_sha256"],
            "frequency_tier": probe["entity"]["frequency_tier"],
            "train_count": probe["entity"]["train_count"],
            "train_chapter_count": probe["entity"]["train_chapter_count"],
            "evidence": probe["evidence"],
            "distractor_matching": probe["distractor_matching"],
        },
    }


def validate_probe_artifact(
    artifact: Mapping[str, Any],
    source_texts: Mapping[str, str],
) -> dict[str, Any]:
    """Recompute every evidence slice and enforce prompt-answer separation."""
    failures: list[str] = []
    ids: set[str] = set()
    counts: Counter[str] = Counter()
    for probe in artifact.get("probes", []):
        probe_id = str(probe.get("id", ""))
        if not probe_id or probe_id in ids:
            failures.append(f"duplicate or empty probe id: {probe_id!r}")
        ids.add(probe_id)
        probe_type = str(probe.get("probe_type"))
        split = str(probe.get("source", {}).get("split"))
        counts[f"{probe_type}:{split}"] += 1
        if split not in source_texts:
            failures.append(f"{probe_id}: unknown source split {split!r}")
            continue
        evidence = probe.get("evidence", {})
        start = evidence.get("split_char_start")
        end = evidence.get("split_char_end")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end:
            failures.append(f"{probe_id}: invalid evidence offsets")
            continue
        exact = source_texts[split][start:end]
        if exact != evidence.get("text"):
            failures.append(f"{probe_id}: evidence text does not match source offsets")
        if _sha256_text(exact) != evidence.get("sha256"):
            failures.append(f"{probe_id}: evidence SHA-256 mismatch")
        prompt = str(probe.get("prompt", ""))
        if probe_type == "held_out_continuation":
            answer = str(probe.get("expected", {}).get("continuation", ""))
            if not prompt or not answer or prompt + answer != exact:
                failures.append(f"{probe_id}: continuation is not an exact prompt suffix")
            if answer and answer in prompt:
                failures.append(f"{probe_id}: continuation answer leaked into prompt")
        elif probe_type == "cloze_candidate_ranking":
            answer = str(probe.get("expected", {}).get("text", ""))
            if answer and answer in prompt:
                failures.append(f"{probe_id}: cloze answer leaked into context prompt")
            if prompt + answer != exact:
                failures.append(f"{probe_id}: cloze answer is not the exact next source text")
            candidates = probe.get("candidates", [])
            if not candidates or not all(
                candidate.get("entity_type") == probe.get("entity", {}).get("entity_type")
                for candidate in candidates
            ):
                failures.append(f"{probe_id}: distractors are not type matched")
            answer_index = probe.get("expected", {}).get("candidate_index")
            if not isinstance(answer_index, int) or not 0 <= answer_index < len(candidates):
                failures.append(f"{probe_id}: invalid expected candidate index")
            elif candidates[answer_index].get("text") != answer:
                failures.append(f"{probe_id}: candidate index does not identify the answer")
        else:
            failures.append(f"{probe_id}: forbidden or unknown probe type {probe_type!r}")
    for collection_name, expected_split in (
        ("cases", "val"),
        ("calibration_cases", "train"),
    ):
        cases = artifact.get(collection_name, [])
        if not isinstance(cases, list) or not cases:
            failures.append(f"{collection_name}: expected a non-empty list")
            continue
        for case in cases:
            case_id = str(case.get("id", ""))
            metadata = case.get("metadata", {})
            if metadata.get("source_split") != expected_split:
                failures.append(
                    f"{collection_name}:{case_id}: source split must be {expected_split}"
                )
            if not isinstance(metadata.get("train_count"), int):
                failures.append(f"{collection_name}:{case_id}: train_count is missing")
            chapter_sha256 = str(metadata.get("chapter_sha256", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", chapter_sha256):
                failures.append(f"{collection_name}:{case_id}: invalid chapter SHA-256")
            case_evidence = metadata.get("evidence", {})
            evidence_text = str(case_evidence.get("text", ""))
            if _sha256_text(evidence_text) != case_evidence.get("sha256"):
                failures.append(f"{collection_name}:{case_id}: evidence SHA-256 mismatch")
    return {
        "passed": not failures,
        "probe_count": len(ids),
        "counts": dict(sorted(counts.items())),
        "failures": failures,
    }


def build_probe_artifact(
    manifest_path: Path,
    *,
    seed: int = DEFAULT_SEED,
    continuation_per_split: int = 16,
    cloze_per_tier_per_split: int = 6,
    candidate_count: int = 4,
    minimum_entity_occurrences: int = 20,
    minimum_entity_attributions: int = 2,
    run_id: str,
    loggers: Mapping[str, logging.Logger],
) -> dict[str, Any]:
    if continuation_per_split <= 0 or cloze_per_tier_per_split <= 0:
        raise ProbeBuildError("probe counts must be positive")
    if candidate_count < 3:
        raise ProbeBuildError("candidate_count must be at least 3")
    manifest = _load_manifest(manifest_path)
    source_texts, source_paths = _verify_and_load_splits(
        manifest_path,
        manifest,
        loggers["data"],
    )
    paragraphs: dict[str, list[ParagraphSpan]] = {}
    for split in SPLITS:
        paragraphs[split] = _paragraph_spans(
            split,
            source_texts[split],
            source_paths[split],
            str(manifest["splits"][split]["sha256"]),
        )
    entities = extract_corpus_entities(
        source_texts,
        paragraphs,
        minimum_occurrences=minimum_entity_occurrences,
        minimum_attributions=minimum_entity_attributions,
    )
    minimum_tier_size = max(candidate_count, 4)
    high, low = stratify_entities(entities, minimum_tier_size=minimum_tier_size)
    tiers = {"high": high, "low": low}
    loggers["data"].info(
        "corpus-derived frequency strata built",
        extra={
            "context": {
                "qualified_entities": len(entities),
                "high_tier_entities": len(high),
                "low_tier_entities": len(low),
                "statistics_scope": "train_only",
                "high_minimum_train_count": min(item.train_count for item in high),
                "low_maximum_train_count": max(item.train_count for item in low),
            }
        },
    )

    probes: list[dict[str, Any]] = []
    requested: dict[str, int] = {}
    actual: dict[str, int] = {}
    for split in SPLITS:
        continuation_candidates = [
            candidate
            for span in paragraphs[split]
            if (candidate := _continuation_candidate(span, seed)) is not None
        ]
        selected_continuations = _select_diverse(
            continuation_candidates,
            continuation_per_split,
            seed=seed,
            identity=f"continuation:{split}",
            unique_entity=False,
        )
        key = f"held_out_continuation:{split}"
        requested[key] = continuation_per_split
        actual[key] = len(selected_continuations)
        probes.extend(selected_continuations)

        cloze_candidates = [
            candidate
            for span in paragraphs[split]
            for candidate in _cloze_candidates_for_span(
                span,
                tiers,
                candidate_count=candidate_count,
                seed=seed,
            )
        ]
        for tier_name in ("high", "low"):
            tier_candidates = [
                candidate
                for candidate in cloze_candidates
                if candidate["entity"]["frequency_tier"] == tier_name
            ]
            selected_cloze = _select_diverse(
                tier_candidates,
                cloze_per_tier_per_split,
                seed=seed,
                identity=f"cloze:{split}:{tier_name}",
                unique_entity=True,
            )
            key = f"cloze_candidate_ranking:{split}:{tier_name}"
            requested[key] = cloze_per_tier_per_split
            actual[key] = len(selected_cloze)
            probes.extend(selected_cloze)

    shortfalls = {
        key: {"requested": requested[key], "actual": actual[key]}
        for key in requested
        if actual[key] != requested[key]
    }
    if shortfalls:
        raise ProbeBuildError(f"insufficient eligible corpus spans: {shortfalls}")
    probes.sort(key=lambda item: (item["source"]["split"], item["probe_type"], item["id"]))
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "usage": {
            "purpose": "pretraining_capability_evaluation_only",
            "training_allowed": False,
            "sft_information_used": False,
            "knowledge_scope": "frozen_formal_novel_corpus_only",
            "forbidden_probe_domains": ["general_encyclopedia", "mathematics"],
            "test_split_read": False,
        },
        "build": {
            "run_id": run_id,
            "builder": Path(__file__).name,
            "seed": seed,
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": file_sha256(manifest_path),
            "inputs": {
                split: {
                    "path": str(source_paths[split]),
                    "sha256": str(manifest["splits"][split]["sha256"]),
                    "characters": len(source_texts[split]),
                }
                for split in SPLITS
            },
            "sealed_inputs": {
                "test": {
                    "path_from_manifest": str(manifest["splits"]["test"]["path"]),
                    "sha256_from_manifest": str(manifest["splits"]["test"]["sha256"]),
                    "read": False,
                }
            },
            "selection": {
                "continuation_per_split": continuation_per_split,
                "cloze_per_tier_per_split": cloze_per_tier_per_split,
                "candidate_count": candidate_count,
                "minimum_entity_occurrences": minimum_entity_occurrences,
                "minimum_entity_attributions": minimum_entity_attributions,
                "requested": requested,
                "actual": actual,
            },
        },
        "entity_strata": {
            "derivation": (
                "formal train-only speaker-attribution pattern plus train-only occurrence "
                "and chapter counts; validation text never affects the catalogue or tiers"
            ),
            "statistics_scope": "train_only",
            "entity_type": "person_speaker",
            "high": [item.__dict__ for item in high],
            "low": [item.__dict__ for item in low],
        },
        "probes": probes,
        "cases": [
            _evaluator_case(probe)
            for probe in probes
            if probe["probe_type"] == "cloze_candidate_ranking"
            and probe["source"]["split"] == "val"
        ],
        "calibration_cases": [
            _evaluator_case(probe)
            for probe in probes
            if probe["probe_type"] == "cloze_candidate_ranking"
            and probe["source"]["split"] == "train"
        ],
        "continuation_prompts": [
            probe["prompt"]
            for probe in probes
            if probe["probe_type"] == "held_out_continuation"
            and probe["source"]["split"] == "val"
        ],
    }
    validation = validate_probe_artifact(artifact, source_texts)
    artifact["validation"] = validation
    if not validation["passed"]:
        loggers["validation"].error(
            "probe leakage or provenance validation failed",
            extra={"context": {"failures": validation["failures"]}},
        )
        raise ProbeBuildError(
            f"probe validation failed with {len(validation['failures'])} error(s)"
        )
    loggers["validation"].info(
        "all probe evidence and leakage gates passed",
        extra={
            "context": {
                "probe_count": validation["probe_count"],
                "counts": validation["counts"],
            }
        },
    )
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=DEFAULT_PROMPTS_OUTPUT,
        help="Write held-out validation continuation prompts, one per line.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--continuation-per-split", type=int, default=16)
    parser.add_argument("--cloze-per-tier-per-split", type=int, default=6)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--minimum-entity-occurrences", type=int, default=20)
    parser.add_argument("--minimum-entity-attributions", type=int, default=2)
    parser.add_argument("--run-id")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--data-log-level", default="INFO")
    parser.add_argument("--validation-log-level", default="INFO")
    parser.add_argument("--orchestrator-log-level", default="INFO")
    parser.add_argument("--log-max-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--log-backup-count", type=int, default=3)
    parser.add_argument("--no-console-log", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_id = args.run_id or generate_run_id("pretrain-probes")
    loggers: dict[str, logging.Logger] = {}
    json_completion_marker_written = False
    try:
        levels = resolve_module_log_levels(
            {
                "data": args.data_log_level,
                "validation": args.validation_log_level,
                "orchestrator": args.orchestrator_log_level,
            }
        )
        loggers = configure_module_loggers(
            args.log_dir,
            run_id,
            levels,
            max_bytes=args.log_max_bytes,
            backup_count=args.log_backup_count,
            console=not args.no_console_log,
        )
        loggers["orchestrator"].info(
            "pretraining capability probe build started",
            extra={
                "context": {
                    "manifest": str(args.manifest),
                    "output": str(args.output),
                    "seed": args.seed,
                }
            },
        )
        artifact = build_probe_artifact(
            args.manifest,
            seed=args.seed,
            continuation_per_split=args.continuation_per_split,
            cloze_per_tier_per_split=args.cloze_per_tier_per_split,
            candidate_count=args.candidate_count,
            minimum_entity_occurrences=args.minimum_entity_occurrences,
            minimum_entity_attributions=args.minimum_entity_attributions,
            run_id=run_id,
            loggers=loggers,
        )
        prompts_content = "\n".join(artifact["continuation_prompts"]) + "\n"
        prompt_count = len(artifact["continuation_prompts"])
        prompts_content_sha256 = _sha256_text(prompts_content)
        cases_canonical_sha256 = canonical_json_sha256({"cases": artifact["cases"]})
        artifact["evaluator_compatibility"] = {
            "cloze_json": str(args.output),
            "cloze_contract": (
                "top-level cases[{id,context,candidates,correct,metadata}]; metadata "
                "contains held-out provenance and is ignored by legacy loaders"
            ),
            "cases_canonical_sha256": cases_canonical_sha256,
            "prompts_txt": str(args.prompts_output),
            "prompts_contract": "one held-out validation prompt per line",
            "prompt_count": prompt_count,
            "prompts_content_sha256": prompts_content_sha256,
            "write_order": (
                "prompts txt is atomically written and hash-verified first; JSON is the "
                "final completion marker"
            ),
            "example_command": (
                ".venv/bin/python evaluate_pretrain_capabilities.py "
                f"--held-out-split val --prompts {args.prompts_output} "
                f"--prompt-limit {prompt_count} --cloze {args.output} --formal"
            ),
        }
        if args.output.resolve() == args.prompts_output.resolve():
            raise ProbeBuildError("JSON output and prompts output must be different paths")
        if args.output.exists():
            if not args.output.is_file() and not args.output.is_symlink():
                raise ProbeBuildError(f"JSON output path is not a file: {args.output}")
            args.output.unlink()
        atomic_write_text(
            args.prompts_output,
            prompts_content,
        )
        if file_sha256(args.prompts_output) != prompts_content_sha256:
            raise ProbeBuildError(
                "prompts output checksum verification failed; check filesystem integrity"
            )
        # JSON is deliberately written last and therefore acts as the completed
        # two-artifact build marker.
        atomic_write_json(args.output, artifact)
        json_completion_marker_written = True
        loggers["orchestrator"].info(
            "pretraining capability probe artifact written",
            extra={
                "context": {
                    "output": str(args.output),
                    "prompts_output": str(args.prompts_output),
                    "prompt_count": prompt_count,
                    "prompts_content_sha256": prompts_content_sha256,
                    "cases_canonical_sha256": cases_canonical_sha256,
                    "probe_count": len(artifact["probes"]),
                    "output_sha256": file_sha256(args.output),
                }
            },
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "run_id": run_id,
                    "output": str(args.output),
                    "probe_count": len(artifact["probes"]),
                    "validation": artifact["validation"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as error:
        if not loggers:
            fallback_log_dir = (
                Path(tempfile.gettempdir()) / "pretrain-capability-probes-fallback-logs"
            )
            try:
                loggers = configure_module_loggers(
                    fallback_log_dir,
                    run_id,
                    {
                        "data": "OFF",
                        "validation": "OFF",
                        "orchestrator": "ERROR",
                    },
                    max_bytes=1024 * 1024,
                    backup_count=1,
                    console=True,
                )
            except Exception:
                loggers = {}
        if loggers:
            loggers["orchestrator"].exception(
                "pretraining capability probe build failed",
                extra={
                    "context": {
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "requested_log_dir": str(args.log_dir),
                        "remediation": (
                            "Verify manifest/input checksums, output-directory permissions, "
                            "free disk space, and the module log for the failing stage; rerun "
                            "with the same seed after correction."
                        ),
                        "json_completion_marker_written": json_completion_marker_written,
                    }
                },
            )
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "remediation": (
                        "Inspect the orchestrator JSONL log, verify input checksums and "
                        "output permissions, then rerun with the same seed."
                    ),
                },
                ensure_ascii=False,
            )
        )
        return 2
    finally:
        if loggers:
            try:
                close_module_loggers(loggers)
            except Exception as close_error:
                print(
                    json.dumps(
                        {
                            "status": "logging_cleanup_failed",
                            "error_type": type(close_error).__name__,
                            "remediation": "Inspect and close stale log file handles.",
                        },
                        ensure_ascii=False,
                    )
                )


if __name__ == "__main__":
    raise SystemExit(main())
