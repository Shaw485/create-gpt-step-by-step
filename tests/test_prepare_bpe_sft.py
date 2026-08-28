import unittest

from bpe_tokenizer import learn_bpe
from prepare_bpe_sft import SPECIAL_TOKENS, serialize_record, special_token_ids


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

    def test_v4_topic_id_is_preserved(self):
        text = "问题回答"
        tokenizer = learn_bpe([text * 4], sorted(set(text)), num_merges=2)
        tokenizer, special_ids = special_token_ids(tokenizer)
        record = {
            "id": "x",
            "topic_id": "topic:x",
            "fact_id": "fact:x",
            "task_family": "direct_fact",
            "split": "train",
            "question": "问题",
            "answer": "回答",
        }
        output = serialize_record(record, tokenizer, special_ids)
        self.assertEqual(output["topic"], "topic:x")
        self.assertEqual(output["topic_id"], "topic:x")
        self.assertEqual(output["fact_id"], "fact:x")

    def test_special_token_ids_reuses_existing_specials(self):
        text = "问题回答"
        tokenizer = learn_bpe([text * 4], sorted(set(text)), num_merges=2)
        tokenizer = tokenizer.with_special_tokens(list(SPECIAL_TOKENS))
        same_tokenizer, ids = special_token_ids(tokenizer)
        self.assertEqual(same_tokenizer.vocab_size, tokenizer.vocab_size)
        self.assertEqual(ids["<EOS>"], tokenizer.special_to_id["<EOS>"])


if __name__ == "__main__":
    unittest.main()
