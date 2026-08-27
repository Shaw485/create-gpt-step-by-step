import logging
from pathlib import Path
import tempfile
import unittest

import torch

from train_gpt_stage3 import save_best_checkpoint_if_improved


class BestCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(2, 2)
        self.logger = logging.getLogger("test.best_checkpoint")
        self.history = []

    def test_first_result_creates_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "best.pt"
            best_loss, saved = save_best_checkpoint_if_improved(
                self.model, path, 100, 4.0, float("inf"), 10, self.history, self.logger
            )

            self.assertTrue(saved)
            self.assertEqual(best_loss, 4.0)
            self.assertTrue(path.exists())

    def test_worse_result_does_not_overwrite_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "best.pt"
            save_best_checkpoint_if_improved(
                self.model, path, 100, 4.0, float("inf"), 10, self.history, self.logger
            )
            best_loss, saved = save_best_checkpoint_if_improved(
                self.model, path, 200, 4.2, 4.0, 10, self.history, self.logger
            )
            payload = torch.load(path, weights_only=False)

            self.assertFalse(saved)
            self.assertEqual(best_loss, 4.0)
            self.assertEqual(payload["meta"]["step"], 100)

    def test_better_result_replaces_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "best.pt"
            save_best_checkpoint_if_improved(
                self.model, path, 100, 4.0, float("inf"), 10, self.history, self.logger
            )
            best_loss, saved = save_best_checkpoint_if_improved(
                self.model, path, 200, 3.8, 4.0, 10, self.history, self.logger
            )
            payload = torch.load(path, weights_only=False)

            self.assertTrue(saved)
            self.assertEqual(best_loss, 3.8)
            self.assertEqual(payload["meta"]["step"], 200)
            self.assertEqual(payload["meta"]["best_val_loss"], 3.8)


if __name__ == "__main__":
    unittest.main()
