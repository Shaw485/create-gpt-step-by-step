import unittest
from pathlib import Path

from plot_sft_balanced_v3 import REPORT_DIR


class PlotSftBalancedV3Tests(unittest.TestCase):
    def test_outputs_stay_inside_balanced_milestone(self):
        self.assertEqual(
            REPORT_DIR,
            Path("reports/milestones/003i_sft_balanced_v3_step800"),
        )


if __name__ == "__main__":
    unittest.main()
