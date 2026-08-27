import unittest

import torch

from bpe_tokenizer import learn_bpe
from train_bpe_pretrain import generate_continuation


class TrainBPEPretrainTests(unittest.TestCase):
    def test_checkpoint_style_tokenizer_has_smaller_prompt(self):
        text = "今天天气今天天气今天天气"
        tokenizer = learn_bpe([text], sorted(set(text)), num_merges=4)
        self.assertLess(len(tokenizer.encode("今天天气")), len("今天天气"))


if __name__ == "__main__":
    unittest.main()
