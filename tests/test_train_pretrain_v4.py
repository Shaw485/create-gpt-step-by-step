import unittest

from train_pretrain_v4 import learning_rate


class ContinuedLearningRateTests(unittest.TestCase):
    def test_second_phase_schedule_starts_at_new_maximum_and_ends_at_minimum(self):
        settings = {
            "learning_rate": 5e-5,
            "minimum_learning_rate": 1e-5,
            "warmup_steps": 0,
            "schedule_start_step": 2600,
            "max_steps": 6000,
        }
        self.assertAlmostEqual(learning_rate(2600, settings), 5e-5)
        self.assertAlmostEqual(learning_rate(6000, settings), 1e-5)

    def test_original_schedule_remains_backward_compatible(self):
        settings = {
            "learning_rate": 3e-4,
            "minimum_learning_rate": 3e-5,
            "warmup_steps": 100,
            "max_steps": 3000,
        }
        self.assertAlmostEqual(learning_rate(99, settings), 3e-4)
        self.assertAlmostEqual(learning_rate(3000, settings), 3e-5)


if __name__ == "__main__":
    unittest.main()
