import unittest

import torch

from train_gpt_stage3 import GPTLanguageModel
from train_sft_stage1 import (
    collate_records,
    load_training_splits,
    supervised_loss,
)


class TrainSftStage1Tests(unittest.TestCase):
    def make_record(self, record_id, inputs, labels):
        return {
            "id": record_id,
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def test_padding_never_contributes_to_loss(self):
        records = [
            self.make_record("a", [1, 2, 3], [-100, 2, 3]),
            self.make_record("b", [1, 2], [-100, 2]),
        ]
        inputs, labels = collate_records(records, pad_token_id=9)
        self.assertEqual(inputs.tolist(), [[1, 2, 3], [1, 2, 9]])
        self.assertEqual(labels.tolist(), [[-100, 2, 3], [-100, 2, -100]])

    def test_loader_does_not_return_test_records(self):
        payload = {
            "train_records": [self.make_record("train", [1], [1])],
            "val_records": [self.make_record("val", [1], [1])],
            "test_records": [self.make_record("secret_test", [1], [1])],
        }
        train_records, val_records = load_training_splits(payload)
        returned_ids = {record["id"] for record in train_records + val_records}
        self.assertNotIn("secret_test", returned_ids)

    def test_supervised_loss_ignores_minus_100_labels(self):
        model = GPTLanguageModel(10, 8, 2, 4, 1)
        inputs = torch.tensor([[1, 2, 3]], dtype=torch.long)
        labels = torch.tensor([[-100, -100, 4]], dtype=torch.long)
        loss = supervised_loss(model, inputs, labels)
        self.assertTrue(torch.isfinite(loss))

    def test_backward_produces_gradients(self):
        model = GPTLanguageModel(10, 8, 2, 4, 1)
        inputs = torch.tensor([[1, 2, 3]], dtype=torch.long)
        labels = torch.tensor([[-100, 3, 4]], dtype=torch.long)
        loss = supervised_loss(model, inputs, labels)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
