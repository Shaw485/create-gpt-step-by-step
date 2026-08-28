import unittest

from evaluate_sft_v5_entities import score_hidden_item, summarize_hidden


class EvaluateSftV5EntitiesTests(unittest.TestCase):
    def test_known_entity_requires_identity_and_eos(self):
        item = {
            "category": "已知实体",
            "required_any": ("萧炎",),
            "required_context_any": ("主角", "主要"),
        }

        self.assertTrue(score_hidden_item(item, "萧炎是小说主角。", True)[0])
        self.assertFalse(score_hidden_item(item, "资料不足。", True)[0])
        self.assertFalse(score_hidden_item(item, "萧炎是小说主角。", False)[0])

    def test_unknown_requires_entity_specific_grounded_refusal(self):
        item = {"category": "不存在实体", "entity": "九星猫皇"}

        self.assertTrue(
            score_hidden_item(
                item,
                "当前语料没有找到九星猫皇，因此不能确认。",
                True,
            )[0]
        )
        self.assertTrue(
            score_hidden_item(
                item,
                "当前语料没有找到该名称，因此无法确认。",
                True,
            )[0]
        )
        self.assertFalse(score_hidden_item(item, "现有资料不足。", True)[0])

    def test_summary_counts_categories_and_eos(self):
        rows = [
            {"category": "已知实体", "passed": True, "stopped_on_eos": True},
            {"category": "不存在实体", "passed": False, "stopped_on_eos": True},
        ]

        summary = summarize_hidden(rows)

        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["eos_count"], 2)


if __name__ == "__main__":
    unittest.main()
