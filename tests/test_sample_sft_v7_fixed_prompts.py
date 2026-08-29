from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import sample_sft_v7_fixed_prompts as fixed


class FixedPromptSamplingTests(unittest.TestCase):
    def setUp(self):
        self.prompt_set_path = Path("configs/sft_v7_fixed_prompts.json")
        self.prompt_set = fixed.load_prompt_set(self.prompt_set_path)

    def test_frozen_prompt_set_has_16_unique_valid_conversations(self):
        self.assertEqual(len(self.prompt_set["cases"]), 16)
        self.assertEqual(
            len({case["id"] for case in self.prompt_set["cases"]}), 16
        )
        for case in self.prompt_set["cases"]:
            self.assertEqual(case["messages"][-1]["role"], "user")
            self.assertEqual(
                [message["role"] for message in case["messages"]],
                [
                    "user" if index % 2 == 0 else "assistant"
                    for index in range(len(case["messages"]))
                ],
            )
        self.assertEqual(
            self.prompt_set["generation"],
            {
                "max_new_tokens": 128,
                "temperature": 0.3,
                "top_k": 1,
                "seed": 20260829,
            },
        )

    def test_validation_rejects_duplicate_role_and_generation_drift_safely(self):
        duplicate = deepcopy(self.prompt_set)
        duplicate["cases"][1]["id"] = duplicate["cases"][0]["id"]
        with self.assertRaisesRegex(fixed.FixedPromptSamplingError, "case_id"):
            fixed.validate_prompt_set(duplicate)

        wrong_role = deepcopy(self.prompt_set)
        secret_body = wrong_role["cases"][0]["messages"][0]["content"]
        wrong_role["cases"][0]["messages"][0]["role"] = "assistant"
        with self.assertRaises(fixed.FixedPromptSamplingError) as caught:
            fixed.validate_prompt_set(wrong_role)
        self.assertEqual(caught.exception.code, "prompt_set_role_order")
        self.assertNotIn(secret_body, str(caught.exception))

        extra_decode_field = deepcopy(self.prompt_set)
        extra_decode_field["generation"]["min_p"] = 0.1
        with self.assertRaisesRegex(
            fixed.FixedPromptSamplingError, "generation_fields"
        ):
            fixed.validate_prompt_set(extra_decode_field)

    def test_cli_defaults_to_frozen_decoding_and_allows_explicit_overrides(self):
        defaults = fixed.parse_args([])
        generation, overridden = fixed.resolve_generation(
            self.prompt_set["generation"], defaults
        )
        self.assertEqual(generation, self.prompt_set["generation"])
        self.assertEqual(overridden, [])

        args = fixed.parse_args(
            [
                "--max-new-tokens",
                "64",
                "--temperature",
                "0.5",
                "--top-k",
                "3",
                "--seed",
                "17",
            ]
        )
        generation, overridden = fixed.resolve_generation(
            self.prompt_set["generation"], args
        )
        self.assertEqual(
            generation,
            {"max_new_tokens": 64, "temperature": 0.5, "top_k": 3, "seed": 17},
        )
        self.assertEqual(overridden, list(fixed.GENERATION_FIELDS))

    def test_render_markdown_keeps_complete_conversation_and_output(self):
        report = {
            "checkpoint_step": 500,
            "checkpoint_mode": "sft-v7",
            "checkpoint_sha256": "a" * 64,
            "prompt_set_sha256": "b" * 64,
            "generation": self.prompt_set["generation"],
            "results": [
                {
                    "id": "full_case",
                    "category": "vertical|chat",
                    "messages": [
                        {"role": "user", "content": "第一行\n第二行"}
                    ],
                    "generated_text": "完整输出|没有省略\n第二行",
                    "stopped_on_eos": True,
                    "truncated": False,
                }
            ],
        }
        markdown = fixed.render_markdown(report)
        self.assertIn("第一行<br>第二行", markdown)
        self.assertIn("完整输出\\|没有省略<br>第二行", markdown)
        self.assertNotIn("...", markdown)

    def test_success_writes_all_cases_and_logs_no_prompt_or_response_body(self):
        payload = {
            "special_token_ids": {
                "<UNK>": 1,
                "<BOS>": 2,
                "<USER>": 3,
                "<ASSISTANT>": 4,
                "<EOS>": 5,
                "<PAD>": 6,
            },
            "tokenizer_sha256": "a" * 64,
            "bpe_token_manifest_sha256": "b" * 64,
            "sft_dataset_manifest_sha256": "c" * 64,
        }
        generated = {
            "generated_text": "这是完整的模型输出，不应进入日志。",
            "generated_token_ids": [7, 8, 9],
            "generated_tokens": 3,
            "stopped_on_eos": True,
            "truncated": False,
        }
        config = {
            "logging": {"max_bytes": 4096, "backup_count": 1, "console": False}
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_json = root / "fixed.json"
            output_markdown = root / "fixed.md"
            log_dir = root / "logs"
            argv = [
                "--prompt-set",
                str(self.prompt_set_path),
                "--public-tensors",
                str(root / "public_diagnostic_tensors.pt"),
                "--checkpoint",
                str(root / "step_00500.pt"),
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_markdown),
                "--log-dir",
                str(log_dir),
                "--device",
                "cpu",
            ]
            with (
                mock.patch.object(fixed.sample, "load_config", return_value=config),
                mock.patch.object(
                    fixed.sample, "load_public_payload", return_value=payload
                ) as load_public,
                mock.patch.object(
                    fixed.sample, "load_bound_tokenizer", return_value=object()
                ),
                mock.patch.object(
                    fixed.sample, "select_device", return_value=torch.device("cpu")
                ),
                mock.patch.object(
                    fixed.sample,
                    "load_model_bundle",
                    return_value=(
                        object(),
                        {"step": 500},
                        {"step": 500, "base_checkpoint_sha256": "d" * 64},
                    ),
                ),
                mock.patch.object(
                    fixed.sample,
                    "build_conversation_prompt_ids",
                    return_value=[2, 3, 4],
                ),
                mock.patch.object(
                    fixed.sample, "generate_response", return_value=generated
                ) as generate,
                mock.patch.object(fixed, "file_sha256", return_value="e" * 64),
            ):
                self.assertEqual(fixed.main(argv), 0)

            load_public.assert_called_once_with(root / "public_diagnostic_tensors.pt")
            self.assertEqual(generate.call_count, 16)
            report = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(report["checkpoint_step"], 500)
            self.assertEqual(report["prompt_set_sha256"], "e" * 64)
            self.assertEqual(len(report["results"]), 16)
            self.assertEqual(
                report["results"][0]["generated_text"], generated["generated_text"]
            )
            self.assertIn(generated["generated_text"], output_markdown.read_text())
            log_text = "".join(
                path.read_text(encoding="utf-8")
                for path in log_dir.glob("*.jsonl")
            )
            self.assertNotIn("萧炎是谁", log_text)
            self.assertNotIn(generated["generated_text"], log_text)
            self.assertIn('"module": "cloud.generation"', log_text)
            self.assertIn('"module": "cloud.evaluation"', log_text)
            self.assertIn('"module": "cloud.checkpoint"', log_text)

    def test_failure_log_and_exception_expose_only_error_code(self):
        config = {
            "logging": {"max_bytes": 4096, "backup_count": 1, "console": False}
        }
        secret = "不应出现在日志或异常中的提示正文"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_path = root / "prompts.json"
            prompt_path.write_text(secret, encoding="utf-8")
            log_dir = root / "logs"
            with (
                mock.patch.object(fixed.sample, "load_config", return_value=config),
                mock.patch.object(
                    fixed,
                    "load_prompt_set",
                    side_effect=RuntimeError(secret),
                ),
            ):
                with self.assertRaises(fixed.FixedPromptSamplingError) as caught:
                    fixed.main(
                        [
                            "--prompt-set",
                            str(prompt_path),
                            "--log-dir",
                            str(log_dir),
                        ]
                    )
            self.assertEqual(caught.exception.code, "unexpected_failure")
            self.assertNotIn(secret, str(caught.exception))
            log_text = "".join(
                path.read_text(encoding="utf-8")
                for path in log_dir.glob("*.jsonl")
            )
            self.assertIn("error_code=unexpected_failure", log_text)
            self.assertNotIn(secret, log_text)


if __name__ == "__main__":
    unittest.main()
