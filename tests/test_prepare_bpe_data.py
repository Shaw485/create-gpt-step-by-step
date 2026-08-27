import unittest

from prepare_bpe_data import evenly_sample_sequences


class PrepareBPEDataTests(unittest.TestCase):
    def test_even_sampling_is_deterministic_and_bounded(self):
        text = "0123456789" * 100
        first = evenly_sample_sequences(text, target_chars=200, chunk_count=10)
        second = evenly_sample_sequences(text, target_chars=200, chunk_count=10)
        self.assertEqual(first, second)
        self.assertEqual(sum(map(len, first)), 200)
        self.assertTrue(first[0].startswith("0"))

    def test_large_target_returns_complete_text(self):
        self.assertEqual(evenly_sample_sequences("abc", 10, 2), ["abc"])


if __name__ == "__main__":
    unittest.main()
