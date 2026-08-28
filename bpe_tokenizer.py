"""A small, handwritten character-seeded BPE tokenizer.

The implementation is intentionally dependency-free so the project demonstrates
what BPE does instead of hiding the algorithm behind a tokenizer library.
"""

from __future__ import annotations

from collections import defaultdict
import heapq
import json
from pathlib import Path
from typing import Iterable


class BPETokenizer:
    """Encode text by repeatedly applying learned adjacent-token merges."""

    def __init__(
        self,
        tokens: list[str],
        merges: list[tuple[int, int, int]],
        special_tokens: list[str] | None = None,
    ):
        self.tokens = list(tokens)
        self.merges = [tuple(map(int, merge)) for merge in merges]
        self.special_tokens = list(special_tokens or [])
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special_tokens must be unique")
        if any(token not in self.tokens for token in self.special_tokens):
            raise ValueError("every special token must exist in tokens")
        self.special_to_id = {
            token: self.tokens.index(token) for token in self.special_tokens
        }
        self.char_to_id = {
            token: token_id
            for token_id, token in enumerate(self.tokens)
            if len(token) == 1
        }
        self.merge_rules = {
            (left, right): (rank, new_id)
            for rank, (left, right, new_id) in enumerate(self.merges)
        }

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        """Encode text using a heap so the lowest-ranked valid merge wins."""
        if not text:
            return []

        missing = sorted({char for char in text if char not in self.char_to_id})
        if missing:
            raise ValueError(f"text has characters outside the BPE vocabulary: {missing[:10]}")

        token_ids = [self.char_to_id[char] for char in text]
        length = len(token_ids)
        previous = [index - 1 for index in range(length)]
        following = [index + 1 for index in range(length)]
        following[-1] = -1
        alive = [True] * length
        candidates: list[tuple[int, int, int, int]] = []

        def offer(left_index: int) -> None:
            if left_index < 0 or not alive[left_index]:
                return
            right_index = following[left_index]
            if right_index < 0 or not alive[right_index]:
                return
            rule = self.merge_rules.get((token_ids[left_index], token_ids[right_index]))
            if rule is not None:
                rank, new_id = rule
                heapq.heappush(candidates, (rank, left_index, right_index, new_id))

        for index in range(length - 1):
            offer(index)

        while candidates:
            rank, left_index, right_index, new_id = heapq.heappop(candidates)
            if (
                not alive[left_index]
                or not alive[right_index]
                or following[left_index] != right_index
            ):
                continue
            current_rule = self.merge_rules.get(
                (token_ids[left_index], token_ids[right_index])
            )
            if current_rule != (rank, new_id):
                continue

            token_ids[left_index] = new_id
            alive[right_index] = False
            next_index = following[right_index]
            following[left_index] = next_index
            if next_index >= 0:
                previous[next_index] = left_index

            offer(previous[left_index])
            offer(left_index)

        output: list[int] = []
        index = 0
        while index >= 0:
            if alive[index]:
                output.append(token_ids[index])
            index = following[index]
        return output

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        pieces: list[str] = []
        for raw_id in token_ids:
            token_id = int(raw_id)
            if token_id < 0 or token_id >= len(self.tokens):
                raise ValueError(f"token id outside vocabulary: {token_id}")
            if skip_special_tokens and self.tokens[token_id] in self.special_to_id:
                continue
            pieces.append(self.tokens[token_id])
        return "".join(pieces)

    def with_special_tokens(self, special_tokens: list[str]) -> "BPETokenizer":
        """Return a copy with reserved IDs appended after learned BPE tokens."""

        overlap = sorted(set(special_tokens) & set(self.tokens))
        if overlap:
            raise ValueError(f"special tokens collide with learned tokens: {overlap}")
        return BPETokenizer(
            tokens=self.tokens + list(special_tokens),
            merges=self.merges,
            special_tokens=list(special_tokens),
        )

    def to_dict(self) -> dict:
        return {
            "tokenizer_type": "character_seeded_bpe",
            "version": 2 if self.special_tokens else 1,
            "tokens": self.tokens,
            "merges": [list(merge) for merge in self.merges],
            "special_tokens": self.special_tokens,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BPETokenizer":
        return cls(
            payload["tokens"],
            payload["merges"],
            payload.get("special_tokens", []),
        )

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def learn_bpe(
    sequences: list[str],
    base_tokens: list[str],
    num_merges: int,
    min_frequency: int = 2,
    forbidden_text: str = "\n",
    progress_callback=None,
) -> BPETokenizer:
    """Learn BPE merges with linked lists and lazily updated pair counts."""
    if num_merges < 0:
        raise ValueError("num_merges must be non-negative")
    char_to_id = {token: index for index, token in enumerate(base_tokens)}
    if any(len(token) != 1 for token in base_tokens):
        raise ValueError("base_tokens must contain single characters only")

    token_ids: list[int] = []
    previous: list[int] = []
    following: list[int] = []
    alive: list[bool] = []
    sequence_starts: list[int] = []

    for sequence in sequences:
        if not sequence:
            continue
        start = len(token_ids)
        sequence_starts.append(start)
        for char in sequence:
            if char not in char_to_id:
                raise ValueError(f"training character outside base vocabulary: {char!r}")
            index = len(token_ids)
            token_ids.append(char_to_id[char])
            previous.append(index - 1 if index > start else -1)
            following.append(index + 1)
            alive.append(True)
        following[-1] = -1

    occurrences: dict[tuple[int, int], set[int]] = defaultdict(set)
    heap: list[tuple[int, int, int]] = []
    tokens = list(base_tokens)

    def pair_at(left_index: int) -> tuple[int, int] | None:
        if left_index < 0 or not alive[left_index]:
            return None
        right_index = following[left_index]
        if right_index < 0 or not alive[right_index]:
            return None
        return token_ids[left_index], token_ids[right_index]

    def is_allowed(pair: tuple[int, int]) -> bool:
        return forbidden_text not in tokens[pair[0]] and forbidden_text not in tokens[pair[1]]

    for start in sequence_starts:
        index = start
        while following[index] >= 0:
            pair = pair_at(index)
            if pair is not None and is_allowed(pair):
                occurrences[pair].add(index)
            index = following[index]

    for pair, positions in occurrences.items():
        heapq.heappush(heap, (-len(positions), pair[0], pair[1]))

    merges: list[tuple[int, int, int]] = []
    existing_text = set(tokens)
    while len(merges) < num_merges and heap:
        negative_count, left_id, right_id = heapq.heappop(heap)
        pair = (left_id, right_id)
        positions = occurrences.get(pair)
        if not positions:
            continue
        current_count = len(positions)
        if -negative_count != current_count:
            heapq.heappush(heap, (-current_count, left_id, right_id))
            continue
        if current_count < min_frequency:
            break

        merged_text = tokens[left_id] + tokens[right_id]
        if merged_text in existing_text:
            occurrences[pair].clear()
            continue

        new_id = len(tokens)
        tokens.append(merged_text)
        existing_text.add(merged_text)
        merges.append((left_id, right_id, new_id))
        changed_pairs: set[tuple[int, int]] = {pair}

        for left_index in sorted(list(positions)):
            right_index = following[left_index] if alive[left_index] else -1
            if (
                right_index < 0
                or not alive[right_index]
                or token_ids[left_index] != left_id
                or token_ids[right_index] != right_id
            ):
                positions.discard(left_index)
                continue

            before_index = previous[left_index]
            after_index = following[right_index]
            for old_left in (before_index, left_index, right_index):
                old_pair = pair_at(old_left)
                if old_pair is not None:
                    occurrences[old_pair].discard(old_left)
                    changed_pairs.add(old_pair)

            token_ids[left_index] = new_id
            alive[right_index] = False
            following[left_index] = after_index
            if after_index >= 0:
                previous[after_index] = left_index

            for new_left in (before_index, left_index):
                new_pair = pair_at(new_left)
                if new_pair is not None and is_allowed(new_pair):
                    occurrences[new_pair].add(new_left)
                    changed_pairs.add(new_pair)

        for changed_pair in changed_pairs:
            count = len(occurrences.get(changed_pair, ()))
            if count >= min_frequency:
                heapq.heappush(
                    heap,
                    (-count, changed_pair[0], changed_pair[1]),
                )

        if progress_callback is not None:
            progress_callback(
                len(merges), num_merges, current_count, merged_text, len(tokens)
            )

    return BPETokenizer(tokens=tokens, merges=merges)
