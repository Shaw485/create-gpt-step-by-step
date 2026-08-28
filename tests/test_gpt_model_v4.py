import unittest

import torch

from gpt_model_v4 import GPTConfig, GPTLanguageModelV4


class GPTConfigTests(unittest.TestCase):
    def test_rejects_head_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            GPTConfig(vocab_size=100, embedding_size=30, num_heads=8).validate()

    def test_rejects_invalid_dropout(self):
        with self.assertRaisesRegex(ValueError, "dropout"):
            GPTConfig(vocab_size=100, dropout=1.0).validate()


class GPTLanguageModelV4Tests(unittest.TestCase):
    def test_planned_model_has_exact_parameter_count(self):
        model = GPTLanguageModelV4(GPTConfig(vocab_size=6484))
        self.assertEqual(model.parameter_count(), 8_240_980)

    def test_input_and_output_weights_share_storage(self):
        model = GPTLanguageModelV4(GPTConfig(vocab_size=128))
        self.assertEqual(
            model.token_embedding.weight.data_ptr(),
            model.output.weight.data_ptr(),
        )

    def test_forward_and_supervised_loss(self):
        config = GPTConfig(
            vocab_size=64,
            block_size=16,
            embedding_size=32,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
        model = GPTLanguageModelV4(config)
        inputs = torch.randint(0, config.vocab_size, (2, 8))
        labels = inputs.clone()
        labels[:, :3] = -100
        logits, loss = model(inputs, labels)
        self.assertEqual(logits.shape, (2, 8, config.vocab_size))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_rejects_sequence_longer_than_context(self):
        config = GPTConfig(
            vocab_size=32,
            block_size=4,
            embedding_size=16,
            num_layers=1,
            num_heads=4,
        )
        model = GPTLanguageModelV4(config)
        with self.assertRaisesRegex(ValueError, "context"):
            model(torch.zeros((1, 5), dtype=torch.long))

    def test_future_token_cannot_change_earlier_logits(self):
        config = GPTConfig(
            vocab_size=32,
            block_size=8,
            embedding_size=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )
        model = GPTLanguageModelV4(config).eval()
        first = torch.tensor([[1, 2, 3, 4]])
        second = torch.tensor([[1, 2, 3, 9]])
        first_logits, _ = model(first)
        second_logits, _ = model(second)
        self.assertTrue(torch.equal(first_logits[:, :3], second_logits[:, :3]))


if __name__ == "__main__":
    unittest.main()
