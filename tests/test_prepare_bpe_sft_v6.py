from __future__ import annotations

import unittest

from bpe_tokenizer import learn_bpe
from prepare_bpe_sft_v6 import SPECIAL_TOKENS, require_special_ids, serialize_messages


class PrepareBpeSftV6Tests(unittest.TestCase):
    def setUp(self):
        text = "一问一答二问二答"
        self.tokenizer = learn_bpe([text * 4], sorted(set(text)), num_merges=2)
        self.tokenizer = self.tokenizer.with_special_tokens(list(SPECIAL_TOKENS))
        self.special_ids = require_special_ids(self.tokenizer)

    def test_all_assistant_turns_and_only_assistant_turns_are_supervised(self):
        record = {
            "id": "multi",
            "primary_dimension": "chat",
            "task_family": "multi",
            "split": "train",
            "messages": [
                {"role": "user", "content": "一问"},
                {"role": "assistant", "content": "一答"},
                {"role": "user", "content": "二问"},
                {"role": "assistant", "content": "二答"},
            ],
        }

        encoded = serialize_messages(record, self.tokenizer, self.special_ids)
        supervised = encoded["labels"][encoded["labels"] != -100].tolist()
        eos_id = self.special_ids["<EOS>"]
        answer_ids = [token_id for token_id in supervised if token_id != eos_id]

        self.assertEqual(self.tokenizer.decode(answer_ids), "一答二答")
        self.assertEqual(supervised.count(eos_id), 2)
        self.assertEqual(encoded["assistant_turns"], 2)

    def test_invalid_role_order_is_rejected(self):
        record = {
            "id": "bad",
            "primary_dimension": "chat",
            "task_family": "bad",
            "split": "train",
            "messages": [
                {"role": "assistant", "content": "一答"},
                {"role": "user", "content": "一问"},
            ],
        }

        with self.assertRaisesRegex(ValueError, "invalid role order"):
            serialize_messages(record, self.tokenizer, self.special_ids)


if __name__ == "__main__":
    unittest.main()
