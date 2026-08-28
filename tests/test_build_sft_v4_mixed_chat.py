import unittest

from bpe_tokenizer import BPETokenizer
from build_sft_v4_mixed_chat import (
    allocate_splits,
    filter_encodable_candidates,
    general_chat_candidates,
    select_family_quotas,
)


class BuildSftV4MixedChatTests(unittest.TestCase):
    def test_allocate_splits_uses_exact_targets(self):
        candidates = [{"id": f"r{index}"} for index in range(4)]
        records = allocate_splits(candidates, {"train": 2, "val": 1, "test": 1})
        self.assertEqual(
            [record["split"] for record in records],
            ["train", "train", "val", "test"],
        )

    def test_select_family_quotas_rejects_insufficient_family(self):
        candidates = [{"id": "a", "task_family": "general_chat"}]
        with self.assertRaisesRegex(ValueError, "not enough"):
            select_family_quotas(candidates, {"general_chat": 2})

    def test_filter_encodable_candidates_tracks_rejected_characters(self):
        tokenizer = BPETokenizer(tokens=["你", "好"], merges=[])
        candidates = [
            {"question": "你好", "answer": "你好"},
            {"question": "你好", "answer": "坏"},
        ]
        accepted, rejected = filter_encodable_candidates(candidates, tokenizer)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected["坏"], 1)

    def test_general_chat_candidates_have_unique_ids(self):
        candidates = general_chat_candidates()
        ids = [record["id"] for record in candidates]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
