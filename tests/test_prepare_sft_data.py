import unittest

from prepare_sft_data import (
    SPECIAL_TOKENS,
    build_special_token_ids,
    parse_log_level,
    serialize_record,
    validate_records,
)
import logging


class PrepareSftDataTests(unittest.TestCase):
    def setUp(self):
        self.stoi = {char: index for index, char in enumerate("问题答案。？")}
        self.special_ids = build_special_token_ids(len(self.stoi))
        self.record = {
            "id": "train_001",
            "question": "问题？",
            "answer": "答案。",
            "evidence": "答案。",
            "source_line": 1,
            "topic": "fact_one",
            "split": "train",
        }

    def test_special_tokens_are_appended_to_base_vocab(self):
        self.assertEqual(self.special_ids["<BOS>"], len(self.stoi))
        self.assertEqual(len(self.special_ids), len(SPECIAL_TOKENS))

    def test_log_categories_can_be_disabled(self):
        self.assertGreater(parse_log_level("OFF"), logging.CRITICAL)

    def test_only_answer_and_eos_have_supervised_labels(self):
        prepared = serialize_record(self.record, self.stoi, self.special_ids)
        labels = prepared["labels"].tolist()
        assistant_index = prepared["assistant_index"]

        self.assertTrue(all(label == -100 for label in labels[:assistant_index]))
        self.assertEqual(labels[assistant_index], self.stoi["答"])
        self.assertEqual(labels[-1], self.special_ids["<EOS>"])

    def test_validation_accepts_exact_evidence(self):
        validate_records([self.record], "答案。", self.stoi)

    def test_validation_rejects_wrong_source_line(self):
        broken = {**self.record, "source_line": 2}
        with self.assertRaisesRegex(ValueError, "invalid source_line"):
            validate_records([broken], "答案。", self.stoi)

    def test_validation_rejects_topic_leakage(self):
        val_record = {
            **self.record,
            "id": "val_001",
            "question": "答案？",
            "split": "val",
        }
        with self.assertRaisesRegex(ValueError, "topics leak"):
            validate_records([self.record, val_record], "答案。", self.stoi)


if __name__ == "__main__":
    unittest.main()
