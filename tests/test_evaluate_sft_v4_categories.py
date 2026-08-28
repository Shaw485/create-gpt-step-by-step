import unittest

from build_sft_v5_1_no_math import is_arithmetic_text, is_math_topic_text
from evaluate_sft_v4_categories import (
    EVAL_ITEMS,
    score_item,
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
                "指令遵循": 5,
                "通用聊天": 5,
            },
        )

    def test_eval_suite_contains_no_math_category_or_metric(self):
        self.assertNotIn("基础数学", {item["category"] for item in EVAL_ITEMS})
        self.assertNotIn("math_exact", {item["metric"] for item in EVAL_ITEMS})
        for item in EVAL_ITEMS:
            self.assertFalse(is_arithmetic_text(item["question"]))
            self.assertFalse(is_math_topic_text(item["question"]))

    def test_exact_response_rejects_extra_text(self):
        item = {
            "metric": "exact_response",
            "accepted_responses": ["收到", "收到。"],
        }

        score = score_item(item, "收到，我来回答。", True)

        self.assertFalse(score["passed"])

    def test_exact_response_accepts_expected_text(self):
        item = {
            "metric": "exact_response",
            "accepted_responses": ["收到", "收到。"],
        }

        score = score_item(item, "收到。", True)

        self.assertTrue(score["passed"])

    def test_two_advice_requires_two_suggestions(self):
        item = {
            "metric": "two_advice",
            "required_any": ["专注", "休息"],
        }

        one = score_item(item, "先保持专注。", True)
        two = score_item(item, "先关闭干扰保持专注。然后短暂休息。", True)

        self.assertFalse(one["passed"])
        self.assertTrue(two["passed"])

    def test_validation_role_rejects_name_only(self):
        item = {
            "metric": "validation_role",
            "required_any": ["泛化", "评估"],
            "forbidden_any": [],
        }

        name_only = score_item(item, "验证集。", True)
        useful = score_item(item, "验证集用于评估模型的泛化表现。", True)

        self.assertFalse(name_only["passed"])
        self.assertTrue(useful["passed"])

    def test_boundary_rejects_novel_leakage(self):
        item = {"metric": "boundary", "topic_required_any": ["天气"]}

        score = score_item(item, "我不能直接看到实时天气，第300章。", True)

        self.assertFalse(score["passed"])

    def test_boundary_requires_question_topic(self):
        item = {"metric": "boundary", "topic_required_any": ["股票", "行情"]}

        score = score_item(item, "我不能直接看到实时天气。", True)

        self.assertFalse(score["passed"])

    def test_chapter_title_rejects_wrong_chapter_number(self):
        item = {
            "metric": "all_required",
            "required_all": ["是", "第300章", "收场"],
            "forbidden_any": [],
        }

        score = score_item(item, "是，第30000章的标题是《收场》。", True)

        self.assertFalse(score["passed"])

    def test_chat_quality_rejects_unknown_refusal(self):
        item = {"metric": "chat_quality"}

        score = score_item(item, "现有资料不足，无法确定。", True)

        self.assertFalse(score["passed"])

    def test_every_metric_requires_eos(self):
        item = {
            "metric": "all_required",
            "required_all": ["萧炎"],
            "forbidden_any": [],
        }

        score = score_item(item, "萧炎", False)

        self.assertFalse(score["passed"])
        self.assertIn("EOS", score["reason"])

    def test_summary_groups_by_category(self):
        summary = summarize(
            [
                {
                    "category": "指令遵循",
                    "passed": True,
                    "metric": "exact_response",
                    "stopped_on_eos": True,
                },
                {
                    "category": "指令遵循",
                    "passed": False,
                    "metric": "exact_response",
                    "stopped_on_eos": False,
                },
            ]
        )

        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["eos_count"], 1)
        self.assertEqual(summary["by_category"]["指令遵循"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
