from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from plot_scaling_stage_a import model_curve_rows, write_curve_csv


class PlotScalingStageATests(unittest.TestCase):
    def test_curve_rows_and_csv_preserve_validation_bpc(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "model4m"
            run_dir.mkdir()
            (run_dir / "report.json").write_text(
                json.dumps(
                    {
                        "history": [
                            {
                                "step": 100,
                                "train_bits_per_character": 5.5,
                                "val_bits_per_character": 5.6,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            summary = {
                "ranked_experiments": [
                    {
                        "experiment": "model4m",
                        "run_dir": str(run_dir),
                        "parameter_count": 4_000_000,
                    }
                ]
            }

            rows = model_curve_rows(summary)
            csv_path = Path(temp_dir) / "curves.csv"
            write_curve_csv(csv_path, rows)

            self.assertEqual(rows[0]["validation_bpc"], 5.6)
            self.assertIn("model4m,4000000,100,5.5,5.6", csv_path.read_text())
            self.assertNotIn(b"\r\n", csv_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
