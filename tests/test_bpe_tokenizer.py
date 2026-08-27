import tempfile
import unittest
from pathlib import Path

from bpe_tokenizer import BPETokenizer, learn_bpe


def base_tokens(text: str) -> list[str]:
    return sorted(set(text))


class BPETokenizerTests(unittest.TestCase):
    def test_round_trip_after_learning_merges(self):
        text = "小猫看雨。小猫看雨。小猫看雨。"
        tokenizer = learn_bpe([text], base_tokens(text), num_merges=5)
        ids = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(ids), text)
        self.assertLess(len(ids), len(text))

    def test_newline_is_never_part_of_a_merged_token(self):
        text = "甲乙\n甲乙\n甲乙\n"
        tokenizer = learn_bpe([text], base_tokens(text), num_merges=10)
        self.assertTrue(
            all("\n" not in token for token in tokenizer.tokens if len(token) > 1)
        )
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_save_and_load_preserves_encoding(self):
        text = "天地玄黄天地玄黄"
        tokenizer = learn_bpe([text], base_tokens(text), num_merges=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            loaded = BPETokenizer.load(path)
        self.assertEqual(loaded.encode(text), tokenizer.encode(text))
        self.assertEqual(loaded.decode(loaded.encode(text)), text)

    def test_unknown_character_is_actionable(self):
        tokenizer = learn_bpe(["甲乙甲乙"], ["甲", "乙"], num_merges=2)
        with self.assertRaisesRegex(ValueError, "outside the BPE vocabulary"):
            tokenizer.encode("甲丙")

    def test_learning_is_deterministic(self):
        text = "天地天地玄黄天地天地玄黄"
        first = learn_bpe([text], base_tokens(text), num_merges=5)
        second = learn_bpe([text], base_tokens(text), num_merges=5)
        self.assertEqual(first.to_dict(), second.to_dict())


if __name__ == "__main__":
    unittest.main()
