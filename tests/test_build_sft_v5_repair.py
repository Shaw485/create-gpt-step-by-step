import unittest
from collections import Counter

from build_sft_v5_repair import (
    FINAL_SPLITS,
    REPAIR_FAMILY_SPLITS,
    allocate_family_splits,
    filter_encodable_candidates,
    repair_candidates,
    validate_records,
)


class FakeTokenizer:
    def __init__(self):
        self.char_to_id = {chr(index): index for index in range(0x110000)}

    def encode(self, text):
        return list(range(len(text)))


class BuildSftV5RepairTest(unittest.TestCase):
    def test_repair_family_targets_sum_to_expected_final_splits(self):
        repair_totals = Counter()
        for split_targets in REPAIR_FAMILY_SPLITS.values():
            repair_totals.update(split_targets)
        self.assertEqual(repair_totals["train"], 1600)
        self.assertEqual(repair_totals["val"], 200)
        self.assertEqual(repair_totals["test"], 200)
        self.assertEqual(FINAL_SPLITS, {"train": 6399, "val": 800, "test": 800})

    def test_candidates_include_positive_known_entity_answers(self):
        candidates = repair_candidates()
        answers = {
            record["question"]: record["answer"]
            for record in candidates
            if record["task_family"] == "novel_known_entity"
        }
        self.assertIn("萧炎是谁？", answers)
        self.assertIn("萧炎", answers["萧炎是谁？"])
        self.assertNotIn("资料不足", answers["萧炎是谁？"])

    def test_boundary_answers_are_topic_specific(self):
        candidates = repair_candidates()
        answers = {
            record["question"]: record["answer"]
            for record in candidates
            if record["task_family"] == "capability_boundary_specific"
        }
        self.assertIn("现在股票涨了吗？", answers)
        self.assertIn("股票", answers["现在股票涨了吗？"])
        self.assertNotIn("天气预报", answers["现在股票涨了吗？"])
        self.assertIn("现在汇率是多少？", answers)
        self.assertIn("汇率", answers["现在汇率是多少？"])

    def test_arithmetic_answers_are_exact_for_diagnostic_shapes(self):
        candidates = repair_candidates()
        answers = {
            record["question"]: record["answer"]
            for record in candidates
            if record["task_family"] == "arithmetic_repair"
        }
        self.assertEqual(answers["一加一等于几？"], "一加一等于二。")
        self.assertEqual(answers["2加3等于几？"], "2加3等于5。")
        self.assertEqual(answers["请直接回答：10加9是多少？"], "答案是19。")

    def test_filter_rejects_duplicate_questions_before_allocation(self):
        tokenizer = FakeTokenizer()
        candidates = repair_candidates()
        accepted, _missing, rejected = filter_encodable_candidates(
            candidates,
            tokenizer,
            {"萧炎是谁？"},
        )
        self.assertEqual(rejected["duplicate_question"], 1)
        self.assertNotIn("萧炎是谁？", {record["question"] for record in accepted})

    def test_allocate_family_splits_and_validate(self):
        tokenizer = FakeTokenizer()
        candidates, _missing, _rejected = filter_encodable_candidates(
            repair_candidates(),
            tokenizer,
            set(),
        )
        selected = allocate_family_splits(candidates, REPAIR_FAMILY_SPLITS)
        selected_splits = Counter(record["split"] for record in selected)
        self.assertEqual(selected_splits, Counter({"train": 1600, "val": 200, "test": 200}))
        summary = validate_records(
            [
                {
                    "id": f"base_{index}",
                    "question": f"base question {index}",
                    "answer": "base answer",
                    "task_family": "base",
                    "split": "train" if index < 4799 else "val" if index < 5399 else "test",
                }
                for index in range(5999)
            ]
            + selected,
            tokenizer,
            FINAL_SPLITS,
        )
        self.assertEqual(summary["split_counts"], FINAL_SPLITS)


if __name__ == "__main__":
    unittest.main()
