from __future__ import annotations

import unittest

from analyze_scaling_stage_a import rank_experiments


class AnalyzeScalingStageATests(unittest.TestCase):
    def test_ranking_uses_bpc_before_parameter_count(self):
        rows = [
            {
                "experiment": "large",
                "best_validation_bits_per_character": 4.0,
                "parameter_count": 14_000_000,
            },
            {
                "experiment": "small",
                "best_validation_bits_per_character": 4.2,
                "parameter_count": 4_000_000,
            },
        ]

        ranked = rank_experiments(rows)

        self.assertEqual(ranked[0]["experiment"], "large")

    def test_parameter_count_breaks_equal_bpc_ties(self):
        rows = [
            {
                "experiment": "large",
                "best_validation_bits_per_character": 4.0,
                "parameter_count": 14_000_000,
            },
            {
                "experiment": "small",
                "best_validation_bits_per_character": 4.0,
                "parameter_count": 4_000_000,
            },
        ]

        ranked = rank_experiments(rows)

        self.assertEqual(ranked[0]["experiment"], "small")


if __name__ == "__main__":
    unittest.main()
