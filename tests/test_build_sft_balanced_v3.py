import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import torch

from build_sft_balanced_v3 import (
    FAMILY_COUNTS,
    FAMILY_SPLITS,
    FINAL_SPLITS,
    UNKNOWN_ANSWER,
)
from prepare_sft_data import CORPUS_PATH


DATASET_PATH = Path("data/sft/sft_balanced_v3.jsonl")
TENSOR_PATH = Path("data/sft/sft_balanced_v3_tensors.pt")


class BuildSftBalancedV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = [
            json.loads(line)
            for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        ]
        cls.corpus_text = CORPUS_PATH.read_text(encoding="utf-8")

    def test_total_family_and_split_counts(self):
        self.assertEqual(len(self.records), 1000)
        self.assertEqual(Counter(row["split"] for row in self.records), FINAL_SPLITS)
        self.assertEqual(
            Counter(row["task_family"] for row in self.records), FAMILY_COUNTS
        )
        for family, expected in FAMILY_SPLITS.items():
            actual = Counter(
                row["split"]
                for row in self.records
                if row["task_family"] == family
            )
            self.assertEqual(actual, expected)

    def test_questions_and_ids_are_unique_and_topics_do_not_leak(self):
        self.assertEqual(len({row["id"] for row in self.records}), 1000)
        self.assertEqual(len({row["question"] for row in self.records}), 1000)
        topic_splits = defaultdict(set)
        for row in self.records:
            topic_splits[row["topic"]].add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in topic_splits.values()))

    def test_classification_is_only_fifteen_percent(self):
        classification = [
            row for row in self.records if row["task_family"] == "concept_identity"
        ]
        self.assertEqual(len(classification), 150)
        self.assertEqual(len(classification) / len(self.records), 0.15)

    def test_verification_labels_follow_reference_answers(self):
        verification = [
            row for row in self.records if row["task_family"] == "fact_verification"
        ]
        for row in verification:
            if row["generation_method"] == "fact_verification_positive":
                self.assertEqual(row["candidate_answer"], row["reference_answer"])
                self.assertEqual(row["answer"], "正确。")
            else:
                self.assertNotEqual(row["candidate_answer"], row["reference_answer"])
                self.assertEqual(row["answer"], "不正确。")

    def test_unknown_entities_are_absent_and_use_honest_answer(self):
        unknown = [
            row for row in self.records if row["task_family"] == "honest_unknown"
        ]
        self.assertEqual(len({row["topic"] for row in unknown}), 50)
        for row in unknown:
            self.assertNotIn(row["concept_label"], self.corpus_text)
            self.assertEqual(row["answer"], UNKNOWN_ANSWER)

    def test_copy_instructions_preserve_exact_text(self):
        copies = [
            row
            for row in self.records
            if row["generation_method"] == "instruction_exact_copy"
        ]
        self.assertEqual(len(copies), 145)
        for row in copies:
            self.assertEqual(row["answer"], row["concept_label"])

    def test_comparison_training_labels_are_balanced(self):
        comparisons = [
            row
            for row in self.records
            if row["generation_method"]
            in {"instruction_compare_same", "instruction_compare_different"}
        ]
        train_answers = Counter(
            row["answer"] for row in comparisons if row["split"] == "train"
        )
        test_answers = Counter(
            row["answer"] for row in comparisons if row["split"] == "test"
        )
        val_answers = Counter(
            row["answer"] for row in comparisons if row["split"] == "val"
        )
        self.assertEqual(train_answers, {"相同。": 20, "不相同。": 20})
        self.assertEqual(val_answers, {"相同。": 3, "不相同。": 2})
        self.assertEqual(test_answers, {"相同。": 5, "不相同。": 5})

    def test_tensor_payload_preserves_split_and_test_is_not_consumed(self):
        payload = torch.load(TENSOR_PATH, map_location="cpu", weights_only=False)
        self.assertEqual(len(payload["train_records"]), 800)
        self.assertEqual(len(payload["val_records"]), 100)
        self.assertEqual(len(payload["test_records"]), 100)
        self.assertEqual(payload["vocab_size"], 4483)


if __name__ == "__main__":
    unittest.main()
