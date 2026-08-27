import unittest

from evaluate_sft_balanced_v3 import escape_table, summarize_results


class EvaluateSftBalancedV3Tests(unittest.TestCase):
    def test_summary_reports_each_family_independently(self):
        rows = [
            {
                "family": "fact",
                "exact_match": True,
                "contains_gold": True,
                "stopped_on_eos": True,
            },
            {
                "family": "chat",
                "exact_match": False,
                "contains_gold": False,
                "stopped_on_eos": True,
            },
        ]
        summary = summarize_results(rows, "family")
        self.assertEqual(summary["fact"]["exact_match"], 1)
        self.assertEqual(summary["chat"]["exact_match"], 0)

    def test_overall_summary_has_all_rows(self):
        rows = [
            {"exact_match": True, "contains_gold": True, "stopped_on_eos": True},
            {"exact_match": False, "contains_gold": True, "stopped_on_eos": False},
        ]
        summary = summarize_results(rows)["overall"]
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["contains_gold"], 2)

    def test_table_escape_handles_separator_and_newline(self):
        self.assertEqual(escape_table("甲|乙\n丙"), "甲\\|乙↵丙")


if __name__ == "__main__":
    unittest.main()
