import unittest

from bpe_tokenizer import learn_bpe
from prepare_bpe_sft import serialize_record


class PrepareBPESFTTests(unittest.TestCase):
    def test_only_answer_and_eos_are_supervised(self):
        text = "问题回答"
        tokenizer = learn_bpe([text * 4], sorted(set(text)), num_merges=2)
        special_ids = {
            token: tokenizer.vocab_size + index
            for index, token in enumerate(("<BOS>", "<USER>", "<ASSISTANT>", "<EOS>", "<PAD>"))
        }
        record = {
            "id": "x", "topic": "x", "task_family": "x", "split": "train",
            "question": "问题", "answer": "回答",
        }
        output = serialize_record(record, tokenizer, special_ids)
        supervised = output["labels"][output["labels"] != -100].tolist()
        self.assertEqual(supervised[-1], special_ids["<EOS>"])
        self.assertEqual(tokenizer.decode(supervised[:-1]), "回答")


if __name__ == "__main__":
    unittest.main()
