import unittest

from plot_sft_hq1000 import validate_history


class PlotSftHq1000Tests(unittest.TestCase):
    def test_history_requires_rows(self):
        with self.assertRaises(ValueError):
            validate_history([])

    def test_history_requires_unique_increasing_steps(self):
        with self.assertRaises(ValueError):
            validate_history([{"step": 10}, {"step": 0}])

    def test_valid_history_passes(self):
        validate_history([{"step": 0}, {"step": 100}])


if __name__ == "__main__":
    unittest.main()
