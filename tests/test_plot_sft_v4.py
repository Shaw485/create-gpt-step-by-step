from pathlib import Path
import tempfile
import unittest

from plot_sft_v4 import plot_loss_history, validate_loss_history


class PlotSftV4Tests(unittest.TestCase):
    def test_plot_loss_history_writes_png_and_svg(self):
        history = [
            {"step": 0, "train_loss": 3.0, "val_loss": 3.2},
            {"step": 100, "train_loss": 2.0, "val_loss": 1.9},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result = plot_loss_history(
                history,
                output_dir / "loss.png",
                output_dir / "loss.svg",
                "test",
            )
            self.assertEqual(result["best_step"], 100)
            self.assertTrue((output_dir / "loss.png").is_file())
            self.assertTrue((output_dir / "loss.svg").is_file())

    def test_validate_loss_history_rejects_duplicate_steps(self):
        history = [
            {"step": 0, "train_loss": 3.0, "val_loss": 3.2},
            {"step": 0, "train_loss": 2.0, "val_loss": 1.9},
        ]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_loss_history(history)


if __name__ == "__main__":
    unittest.main()
