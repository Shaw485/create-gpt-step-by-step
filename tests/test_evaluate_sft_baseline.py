import unittest

import torch

from evaluate_sft_baseline import (
    build_model_from_meta,
    encode_chat_prompt,
    expand_pretrained_model,
    verify_pretrained_weights_copied,
)


class SftBaselineTests(unittest.TestCase):
    def setUp(self):
        self.meta = {
            "vocab_size": 7,
            "embedding_dim": 8,
            "num_heads": 2,
            "block_size": 8,
            "num_layers": 1,
        }
        model = build_model_from_meta(self.meta, self.meta["vocab_size"])
        self.checkpoint = {
            "meta": self.meta,
            "model_state_dict": model.state_dict(),
        }

    def test_expansion_preserves_every_pretrained_parameter(self):
        expanded = expand_pretrained_model(self.checkpoint, 12)
        verify_pretrained_weights_copied(expanded, self.checkpoint)
        self.assertEqual(expanded.token_embedding.weight.shape, (12, 8))
        self.assertEqual(expanded.head.weight.shape, (12, 8))
        self.assertEqual(expanded.head.bias.shape, (12,))

    def test_rejects_non_expanding_vocabulary(self):
        with self.assertRaisesRegex(ValueError, "must be larger"):
            expand_pretrained_model(self.checkpoint, 7)

    def test_chat_prompt_uses_special_tokens_as_whole_ids(self):
        stoi = {"问": 0, "题": 1, "？": 2}
        special = {
            "<BOS>": 3,
            "<USER>": 4,
            "<ASSISTANT>": 5,
            "<EOS>": 6,
            "<PAD>": 7,
        }
        ids = encode_chat_prompt("问题？", stoi, special)
        self.assertEqual(ids, [3, 4, 0, 1, 2, 5])

    def test_chat_prompt_rejects_unknown_character(self):
        special = {"<BOS>": 1, "<USER>": 2, "<ASSISTANT>": 3}
        with self.assertRaisesRegex(ValueError, "out-of-vocabulary"):
            encode_chat_prompt("未知", {}, special)


if __name__ == "__main__":
    unittest.main()
