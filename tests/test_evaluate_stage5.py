from pathlib import Path
import tempfile
import unittest

import torch

from evaluate_stage5 import (
    adjacent_repetition_rate,
    count_topk_correct,
    ngram_repetition_rate,
    normalize_for_repetition,
    plot_loss_history,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_counts_top1_and_top2_predictions(self):
        logits = torch.tensor([[[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]])
        targets = torch.tensor([[0, 2]])

        self.assertEqual(count_topk_correct(logits, targets, 1), 1)
        self.assertEqual(count_topk_correct(logits, targets, 2), 2)

    def test_repetition_rates(self):
        self.assertAlmostEqual(adjacent_repetition_rate("哈哈哈"), 1.0)
        self.assertAlmostEqual(ngram_repetition_rate("哈哈哈哈", 2), 2 / 3)
        self.assertEqual(ngram_repetition_rate("猫", 2), 0.0)
        self.assertEqual(normalize_for_repetition("哈 \n  哈"), "哈哈")

    def test_rejects_invalid_topk(self):
        logits = torch.zeros((1, 1, 3))
        targets = torch.zeros((1, 1), dtype=torch.long)

        with self.assertRaises(ValueError):
            count_topk_correct(logits, targets, 4)

    def test_plot_writes_png_and_svg(self):
        history = [
            {"step": 0, "train_loss": 4.0, "val_loss": 4.1},
            {"step": 10, "train_loss": 3.0, "val_loss": 3.2},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            png_path = Path(temp_dir) / "curve.png"
            svg_path = Path(temp_dir) / "curve.svg"
            best = plot_loss_history(history, png_path, svg_path)

            self.assertEqual(best["step"], 10)
            self.assertTrue(png_path.exists())
            self.assertTrue(svg_path.exists())
            self.assertGreater(png_path.stat().st_size, 0)

    def test_plot_rejects_empty_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                plot_loss_history(
                    [],
                    Path(temp_dir) / "curve.png",
                    Path(temp_dir) / "curve.svg",
                )


if __name__ == "__main__":
    unittest.main()
