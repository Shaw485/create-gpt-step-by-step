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

    def test_inference_cache_initial_logits_exactly_match_regular_forward(self):
        torch.manual_seed(20260829)
        config = GPTConfig(
            vocab_size=32,
            block_size=8,
            embedding_size=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
        model = GPTLanguageModelV4(config).eval()
        inputs = torch.randint(0, config.vocab_size, (3, 5))

        regular_logits, _ = model(inputs)
        cached_logits, cache = model.forward_inference(inputs)

        self.assertTrue(torch.equal(cached_logits, regular_logits))
        self.assertEqual(cache.sequence_length, 5)
        self.assertEqual(cache.batch_size, 3)
        self.assertEqual(len(cache.layers), config.num_layers)

    def test_incremental_inference_logits_match_full_context(self):
        torch.manual_seed(20260829)
        config = GPTConfig(
            vocab_size=32,
            block_size=8,
            embedding_size=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
        model = GPTLanguageModelV4(config).eval()
        inputs = torch.randint(0, config.vocab_size, (2, 7))

        _, cache = model.forward_inference(inputs[:, :4])
        incremental_logits = None
        for position in range(4, inputs.shape[1]):
            incremental_logits, cache = model.forward_inference(
                inputs[:, position : position + 1],
                cache,
            )
            full_logits, _ = model(inputs[:, : position + 1])
            torch.testing.assert_close(
                incremental_logits[:, -1],
                full_logits[:, -1],
                rtol=1e-5,
                atol=1e-6,
            )
        self.assertIsNotNone(incremental_logits)
        self.assertEqual(cache.sequence_length, 7)

    def test_inference_cache_selects_active_eos_rows(self):
        config = GPTConfig(
            vocab_size=32,
            block_size=8,
            embedding_size=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
        model = GPTLanguageModelV4(config).eval()
        inputs = torch.randint(0, config.vocab_size, (3, 4))
        _, cache = model.forward_inference(inputs)

        selected = cache.select_rows([2, 0])

        self.assertEqual(selected.batch_size, 2)
        self.assertEqual(selected.sequence_length, 4)
        for (key, value), (selected_key, selected_value) in zip(
            cache.layers,
            selected.layers,
        ):
            self.assertTrue(torch.equal(selected_key, key[[2, 0]]))
            self.assertTrue(torch.equal(selected_value, value[[2, 0]]))

    def test_left_padded_batch_logits_match_individual_unpadded_rows(self):
        torch.manual_seed(20260829)
        config = GPTConfig(
            vocab_size=32,
            block_size=8,
            embedding_size=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
        model = GPTLanguageModelV4(config).eval()
        short = torch.tensor([3, 4, 5])
        long = torch.tensor([6, 7, 8, 9, 10])
        padded = torch.tensor(
            [
                [0, 0, 3, 4, 5],
                [6, 7, 8, 9, 10],
            ]
        )
        valid_mask = torch.tensor(
            [
                [False, False, True, True, True],
                [True, True, True, True, True],
            ]
        )

        batched_logits, cache = model.forward_inference(
            padded,
            attention_mask=valid_mask,
        )
        short_logits, _ = model(short[None, :])
        long_logits, _ = model(long[None, :])

        torch.testing.assert_close(
            batched_logits[0, -1], short_logits[0, -1], rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            batched_logits[1, -1], long_logits[0, -1], rtol=1e-5, atol=1e-6
        )
        self.assertTrue(torch.equal(cache.key_valid_mask, valid_mask))
        self.assertTrue(torch.isfinite(batched_logits).all())

    def test_left_padded_cache_increment_matches_each_full_context(self):
        torch.manual_seed(20260829)
        config = GPTConfig(
            vocab_size=32,
            block_size=8,
            embedding_size=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
        model = GPTLanguageModelV4(config).eval()
        padded = torch.tensor(
            [
                [0, 0, 3, 4, 5],
                [6, 7, 8, 9, 10],
            ]
        )
        valid_mask = torch.tensor(
            [
                [False, False, True, True, True],
                [True, True, True, True, True],
            ]
        )
        _, cache = model.forward_inference(padded, attention_mask=valid_mask)

        incremental_logits, cache = model.forward_inference(
            torch.tensor([[11], [12]]),
            cache,
        )
        first_full, _ = model(torch.tensor([[3, 4, 5, 11]]))
        second_full, _ = model(torch.tensor([[6, 7, 8, 9, 10, 12]]))

        torch.testing.assert_close(
            incremental_logits[0, -1], first_full[0, -1], rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            incremental_logits[1, -1], second_full[0, -1], rtol=1e-5, atol=1e-6
        )
        self.assertEqual(cache.sequence_length, 6)
        self.assertEqual(cache.key_valid_mask.sum(dim=1).tolist(), [4, 6])

    def test_inference_cache_refuses_to_cross_context_boundary(self):
        config = GPTConfig(
            vocab_size=32,
            block_size=4,
            embedding_size=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )
        model = GPTLanguageModelV4(config).eval()
        _, cache = model.forward_inference(torch.tensor([[1, 2, 3, 4]]))

        with self.assertRaisesRegex(ValueError, "context window"):
            model.forward_inference(torch.tensor([[5]]), cache)

    def test_inference_path_requires_eval_and_does_not_change_forward_signature(self):
        config = GPTConfig(
            vocab_size=32,
            block_size=4,
            embedding_size=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )
        model = GPTLanguageModelV4(config)
        with self.assertRaisesRegex(RuntimeError, "model.eval"):
            model.forward_inference(torch.tensor([[1]]))

        logits, loss = model(torch.tensor([[1, 2]]), torch.tensor([[2, 3]]))
        self.assertEqual(logits.shape, (1, 2, config.vocab_size))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
