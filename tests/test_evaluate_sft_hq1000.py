import unittest

from evaluate_sft_hq1000 import escape_table, score_answer, summarize_by_field


class EvaluateSftHq1000Tests(unittest.TestCase):
    def test_score_answer_exact_match(self):
        result = score_answer("萧炎。", "萧炎。")
        self.assertTrue(result["exact_match"])
        self.assertTrue(result["contains_gold"])

    def test_score_answer_accepts_gold_without_final_punctuation(self):
        result = score_answer("答案是萧炎。", "萧炎。")
        self.assertFalse(result["exact_match"])
        self.assertTrue(result["contains_gold"])

    def test_one_character_gold_does_not_match_inside_opposite_answer(self):
        result = score_answer("不是。", "是。")
        self.assertFalse(result["exact_match"])
        self.assertFalse(result["contains_gold"])

    def test_summary_groups_results(self):
        rows = [
            {
                "kind": "fact",
                "exact_match": True,
                "contains_gold": True,
                "stopped_on_eos": True,
            },
            {
                "kind": "fact",
                "exact_match": False,
                "contains_gold": False,
                "stopped_on_eos": True,
            },
        ]
        summary = summarize_by_field(rows, "kind")
        self.assertEqual(
            summary["fact"],
            {"count": 2, "exact_match": 1, "contains_gold": 1, "stopped_on_eos": 2},
        )

    def test_markdown_table_escapes_content(self):
        self.assertEqual(escape_table("甲|乙\n丙"), "甲\\|乙↵丙")


if __name__ == "__main__":
    unittest.main()
