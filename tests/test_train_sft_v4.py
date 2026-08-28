from __future__ import annotations

import unittest

import torch

from gpt_model_v4 import GPTConfig, GPTLanguageModelV4
from train_sft_v4 import (
    collate_records,
    decode_ids,
    generate_answer,
    validate_sft_payload,
)


def make_record(record_id: str, split: str, ids: list[int], labels: list[int]) -> dict:
    return {
        "id": record_id,
        "split": split,
        "task_family": "direct_fact",
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "assistant_index": 2,
        "sequence_length": len(ids) + 1,
    }


class TrainSftV4Tests(unittest.TestCase):
    def test_collate_records_masks_padding_labels(self):
        records = [
            make_record("a", "train", [1, 2, 3], [-100, 2, 3]),
            make_record("b", "train", [1, 2], [-100, 2]),
        ]
        inputs, labels = collate_records(records, pad_token_id=9)
        self.assertEqual(inputs.tolist(), [[1, 2, 3], [1, 2, 9]])
        self.assertEqual(labels.tolist(), [[-100, 2, 3], [-100, 2, -100]])

    def test_validate_sft_payload_rejects_split_leakage(self):
        payload = {
            "train_records": [make_record("same", "train", [1], [1])],
            "val_records": [make_record("same", "val", [1], [1])],
            "test_records": [make_record("test", "test", [1], [1])],
        }
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_sft_payload(payload, block_size=8)

    def test_decode_ids_accepts_string_or_integer_keys(self):
        self.assertEqual(decode_ids({1: "你", "2": "好"}, [1, 2]), "你好")

    def test_generate_answer_stops_on_eos(self):
        model = GPTLanguageModelV4(
            GPTConfig(
                vocab_size=8,
                block_size=8,
                embedding_size=8,
                num_layers=1,
                num_heads=2,
                dropout=0.0,
                tie_embeddings=False,
            )
        )
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.zero_()
            model.output.bias[5] = 100.0
        answer, stopped = generate_answer(
            model,
            [1, 2, 3],
            {4: "答", 5: "<EOS>"},
            {"<BOS>": 0, "<USER>": 1, "<ASSISTANT>": 2, "<EOS>": 5, "<PAD>": 6},
            max_new_tokens=4,
            temperature=1.0,
            top_k=0,
            seed=42,
            device=torch.device("cpu"),
        )
        self.assertEqual(answer, "")
        self.assertTrue(stopped)


if __name__ == "__main__":
    unittest.main()
