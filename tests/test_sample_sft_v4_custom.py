import unittest

from bpe_tokenizer import BPETokenizer
from sample_sft_v4_custom import build_prompt_ids, markdown_escape, render_markdown


class SampleSftV4CustomTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
