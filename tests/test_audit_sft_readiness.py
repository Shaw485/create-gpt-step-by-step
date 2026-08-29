import unittest

from bpe_tokenizer import BPETokenizer
from audit_sft_readiness import (
    audit_records,
    canonical_text,
    canonical_prompt,
    chapter_order_mismatch,
    length_summary,
    parse_chinese_integer,
    values_crossing_splits,
)


def record(record_id, split, question, answer, group_id):
    return {
        "id": record_id,
        "question": question,
        "answer": answer,
        "task_family": "general_chat",
        "split": split,
        "topic_id": group_id,
        "fact_id": group_id,
        "group_id": group_id,
        "review": {"status": "approved"},
        "evidence": {"status": "curated", "source": "unit-test"},
    }


class AuditSftReadinessTests(unittest.TestCase):
    def test_canonical_text_removes_spacing_and_punctuation(self):
        self.assertEqual(canonical_text(" 今 天？ A！ "), "今天a")

    def test_canonical_prompt_removes_style_prefix(self):
        self.assertEqual(canonical_prompt("用一句话回答：今天天气？"), "今天天气")

    def test_parse_chinese_integer(self):
        self.assertEqual(parse_chinese_integer("一千六百零八"), 1608)
        self.assertEqual(parse_chinese_integer("三十二"), 32)

    def test_chapter_order_mismatch(self):
        item = record("a", "train", "小说第一百六十七章的下一章叫什么？", "第162章。", "a")
        item["origin"] = {
            "source_subcategory": "chapter_order",
            "source_chapter_number": 162,
        }
        self.assertTrue(chapter_order_mismatch(item))

    def test_values_crossing_splits_detects_a_shared_group(self):
        records = [
            record("a", "train", "甲？", "答甲。", "shared"),
            record("b", "val", "乙？", "答乙。", "shared"),
            record("c", "test", "丙？", "答丙。", "test-only"),
        ]
        result = values_crossing_splits(records, lambda item: item["group_id"])
        self.assertEqual(result["value_count"], 1)
        self.assertEqual(result["samples"][0]["splits"], ["train", "val"])

    def test_audit_reports_token_lengths_and_exact_question_leak(self):
        records = [
            record("a", "train", "甲？", "答甲。", "train-a"),
            record("b", "val", "甲？", "答乙。", "val-b"),
        ]
        characters = sorted(set("甲？答。乙"))
        tokenizer = BPETokenizer(characters, [])
        result = audit_records(records, tokenizer)
        self.assertEqual(result["tokenization"]["unencodable_records"], 0)
        self.assertEqual(result["split_isolation"]["exact_question"]["value_count"], 1)
        self.assertEqual(result["tokenization"]["sequences_over_512"], 0)

    def test_length_summary_has_expected_median_and_maximum(self):
        summary = length_summary([1, 2, 3, 4])
        self.assertEqual(summary["p50"], 2.5)
        self.assertEqual(summary["max"], 4)


if __name__ == "__main__":
    unittest.main()
