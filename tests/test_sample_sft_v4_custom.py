import unittest

import torch

from bpe_tokenizer import BPETokenizer
from build_sft_v5_1_no_math import is_arithmetic_text, is_math_topic_text
from sample_sft_v4_custom import (
    DEFAULT_PROMPTS,
    build_prompt_ids,
    markdown_escape,
    render_markdown,
    validate_checkpoint_payload_compatibility,
)


class SampleSftV4CustomTests(unittest.TestCase):
    @staticmethod
    def _record(task_family: str) -> dict:
        return {"task_family": task_family, "input_ids": torch.tensor([1])}

    def test_build_prompt_ids_wraps_question_with_chat_tokens(self):
        tokenizer = BPETokenizer(tokens=["你", "好"], merges=[])
        special_token_ids = {
            "<BOS>": 10,
            "<USER>": 11,
            "<ASSISTANT>": 12,
        }

        prompt_ids = build_prompt_ids(tokenizer, "你好", special_token_ids)

        self.assertEqual(prompt_ids, [10, 11, 0, 1, 12])

    def test_markdown_escape_keeps_table_shape(self):
        self.assertEqual(markdown_escape("a|b\nc"), "a\\|b<br>c")

    def test_render_markdown_uses_generic_checkpoint_title(self):
        text = render_markdown(
            title="checkpoint samples",
            checkpoint_step=5000,
            checkpoint_sha256="abc",
            samples=[
                {
                    "category": "非小说问题",
                    "question": "你好",
                    "generated_answer": "你好。",
                    "stopped_on_eos": True,
                }
            ],
        )

        self.assertIn("# checkpoint samples", text)
        self.assertIn("Checkpoint step：`5000`", text)
        self.assertNotIn("step5000 best", text)

    def test_default_prompts_do_not_include_math_questions(self):
        questions = [item["question"] for item in DEFAULT_PROMPTS]

        self.assertNotIn("一加一等于几？", questions)
        self.assertIn("请只回答“收到”。", questions)
        self.assertTrue(all(not is_arithmetic_text(text) for text in questions))
        self.assertTrue(all(not is_math_topic_text(text) for text in questions))

    def test_checkpoint_payload_compatibility_accepts_matching_dataset(self):
        payload = {
            "train_records": [self._record("chat")],
            "val_records": [self._record("fact")],
            "test_records": [self._record("fact")],
        }
        checkpoint = {
            "extra": {
                "payload_summary": {
                    "split_counts": {"train": 1, "val": 1, "test": 1},
                    "task_family_counts": {"chat": 1, "fact": 2},
                }
            }
        }

        actual = validate_checkpoint_payload_compatibility(checkpoint, payload)

        self.assertEqual(actual["split_counts"], {"train": 1, "val": 1, "test": 1})

    def test_checkpoint_payload_compatibility_rejects_old_dataset(self):
        payload = {
            "train_records": [self._record("chat")],
            "val_records": [self._record("fact")],
            "test_records": [self._record("fact")],
        }
        checkpoint = {
            "extra": {
                "payload_summary": {
                    "split_counts": {"train": 2, "val": 1, "test": 1},
                    "task_family_counts": {"chat": 2, "fact": 2},
                }
            }
        }

        with self.assertRaisesRegex(ValueError, "checkpoint and SFT payload"):
            validate_checkpoint_payload_compatibility(checkpoint, payload)


if __name__ == "__main__":
    unittest.main()
