import unittest

from train_pretrain_v4 import (
    bits_per_character,
    early_stopping_metric,
    learning_rate,
)


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

    def test_bits_per_character_normalizes_different_token_counts(self):
        one_token_per_character = bits_per_character(1.0, 100, 100)
        half_as_many_tokens = bits_per_character(2.0, 50, 100)

        self.assertAlmostEqual(one_token_per_character, half_as_many_tokens)

    def test_formal_training_can_select_on_validation_bpc(self):
        name, value = early_stopping_metric(
            {"val_loss": 4.0, "val_bits_per_character": 3.25},
            {"early_stopping_metric": "val_bits_per_character"},
        )

        self.assertEqual(name, "val_bits_per_character")
        self.assertEqual(value, 3.25)

    def test_old_configs_still_select_on_validation_loss(self):
        name, value = early_stopping_metric(
            {"val_loss": 4.0, "val_bits_per_character": 3.25},
            {},
        )

        self.assertEqual(name, "val_loss")
        self.assertEqual(value, 4.0)


if __name__ == "__main__":
    unittest.main()
