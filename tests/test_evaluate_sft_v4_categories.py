import unittest

from evaluate_sft_v4_categories import (
    EVAL_ITEMS,
    score_item,
    score_math_exact,
    summarize,
)


class EvaluateSftV4CategoriesTests(unittest.TestCase):
    def test_eval_suite_has_expected_category_sizes(self):
        counts = {}
        for item in EVAL_ITEMS:
            counts[item["category"]] = counts.get(item["category"], 0) + 1

        self.assertEqual(
            counts,
            {
                "小说人物": 5,
                "小说事实": 5,
                "证据判断": 5,
                "能力边界": 5,
                "基础数学": 5,
                "通用聊天": 5,
            },
        )

    def test_math_exact_rejects_extra_numbers(self):
        passed, reason = score_math_exact("2加3等于21。", 2)

        self.assertFalse(passed)
        self.assertIn("expected only 2", reason)

    def test_math_exact_accepts_single_expected_number(self):
        passed, _ = score_math_exact("答案是19。", 19)

        self.assertTrue(passed)

    def test_boundary_rejects_novel_leakage(self):
        item = {"metric": "boundary", "topic_required_any": ["天气"]}

        score = score_item(item, "我不能直接看到实时天气，第300章。", True)

        self.assertFalse(score["passed"])

    def test_boundary_requires_question_topic(self):
        item = {"metric": "boundary", "topic_required_any": ["股票", "行情"]}

        score = score_item(item, "我不能直接看到实时天气。", True)

        self.assertFalse(score["passed"])

    def test_chat_quality_rejects_unknown_refusal(self):
        item = {"metric": "chat_quality"}

        score = score_item(item, "现有资料不足，无法确定。", True)

        self.assertFalse(score["passed"])

    def test_summary_groups_by_category(self):
        summary = summarize(
            [
                {
                    "category": "基础数学",
                    "passed": True,
                    "metric": "math_exact",
                    "stopped_on_eos": True,
                },
                {
                    "category": "基础数学",
                    "passed": False,
                    "metric": "math_exact",
                    "stopped_on_eos": False,
                },
            ]
        )

        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["eos_count"], 1)
        self.assertEqual(summary["by_category"]["基础数学"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
