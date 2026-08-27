import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from build_sft_hq1000 import (
    ANSWER_CATEGORIES,
    CATEGORY_QUOTAS,
    FINAL_SPLITS,
    category_options,
    wrong_category,
)


DATASET_PATH = Path("data/sft/sft_hq1000_v2.jsonl")


class BuildSftHq1000Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = [
            json.loads(line)
            for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        ]
        cls.new_records = cls.records[100:]

    def test_final_counts_and_fixed_tests(self):
        self.assertEqual(len(self.records), 1000)
        self.assertEqual(Counter(record["split"] for record in self.records), FINAL_SPLITS)
        record_ids = {record["id"] for record in self.records}
        for index in range(1, 6):
            self.assertIn(f"test_{index:03d}", record_ids)

    def test_questions_are_unique_and_concepts_do_not_leak(self):
        self.assertEqual(len({record["question"] for record in self.records}), 1000)
        concept_splits = defaultdict(set)
        for record in self.new_records:
            concept_splits[record["concept_label"]].add(record["split"])
        self.assertEqual(len(concept_splits), 150)
        self.assertTrue(all(len(splits) == 1 for splits in concept_splits.values()))

    def test_category_quotas_and_variant_counts(self):
        concept_categories = {}
        variants = Counter()
        for record in self.new_records:
            label = record["concept_label"]
            concept_categories[label] = record["concept_category"]
            variants[label] += 1
        self.assertEqual(Counter(concept_categories.values()), CATEGORY_QUOTAS)
        self.assertTrue(all(count in {5, 6, 7} for count in variants.values()))
        self.assertEqual(sum(variants.values()), 900)

    def test_every_answer_matches_the_curated_category(self):
        for record in self.new_records:
            label = record["concept_label"]
            category = record["concept_category"]
            method = record["generation_method"]
            if method == "category_positive":
                self.assertEqual(record["answer"], "是。")
            elif method == "category_negative_correction":
                self.assertIn(f"“{label}”属于{category}", record["answer"])
            else:
                self.assertIn(category, record["answer"])

    def test_options_are_deterministic_unique_and_include_answer(self):
        for category in ANSWER_CATEGORIES:
            first = category_options(category, "固定概念")
            second = category_options(category, "固定概念")
            self.assertEqual(first, second)
            self.assertEqual(len(first), 4)
            self.assertEqual(len(set(first)), 4)
            self.assertIn(category, first)

    def test_wrong_category_never_matches_the_answer(self):
        for category in ANSWER_CATEGORIES:
            self.assertNotEqual(wrong_category(category, "固定概念"), category)


if __name__ == "__main__":
    unittest.main()
