from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

from bpe_tokenizer import BPETokenizer
from gpt_model_v4 import GPTConfig, GPTLanguageModelV4
from prepare_bpe_sft_v7 import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SPECIAL_TOKEN_IDS,
    EXPECTED_TOKENIZER_SHA256,
    EXPECTED_VOCAB_SIZE,
    REQUIRED_BASE_CHECKPOINT,
)
import sample_sft_v7 as sample
from training_runtime import canonical_json_sha256


def tiny_tokenizer() -> BPETokenizer:
    base = ["你", "好", "问", "答", "甲", "乙", "。"]
    specials = ["<UNK>", "<BOS>", "<USER>", "<ASSISTANT>", "<EOS>", "<PAD>"]
    return BPETokenizer(base + specials, [], specials)


class MaskProbeModel(torch.nn.Module):
    def __init__(self, vocab_size: int, unk_id: int, answer_id: int, eos_id: int):
        super().__init__()
        self.config = SimpleNamespace(block_size=64)
        self.vocab_size = vocab_size
        self.unk_id = unk_id
        self.answer_id = answer_id
        self.eos_id = eos_id

    def forward(self, inputs: torch.Tensor):
        logits = torch.full((*inputs.shape, self.vocab_size), -20.0, device=inputs.device)
        for row in range(inputs.shape[0]):
            if int(inputs[row, -1]) == self.answer_id:
                logits[row, -1, self.eos_id] = 12.0
            else:
                logits[row, -1, self.unk_id] = 15.0
                logits[row, -1, self.answer_id] = 14.0
        return logits, None


class RandomProbeModel(torch.nn.Module):
    def __init__(self, vocab_size: int, first_id: int, second_id: int):
        super().__init__()
        self.config = SimpleNamespace(block_size=64)
        self.vocab_size = vocab_size
        self.first_id = first_id
        self.second_id = second_id

    def forward(self, inputs: torch.Tensor):
        logits = torch.full((*inputs.shape, self.vocab_size), -20.0, device=inputs.device)
        logits[:, -1, self.first_id] = 1.0
        logits[:, -1, self.second_id] = 1.0
        return logits, None


class EarlyExitPromptModel(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        first_id: int,
        second_id: int,
        immediate_eos_id: int,
        eos_id: int,
    ):
        super().__init__()
        self.config = SimpleNamespace(block_size=64)
        self.vocab_size = vocab_size
        self.first_id = first_id
        self.second_id = second_id
        self.immediate_eos_id = immediate_eos_id
        self.eos_id = eos_id

    def forward(self, inputs: torch.Tensor):
        logits = torch.full((*inputs.shape, self.vocab_size), -20.0, device=inputs.device)
        for row in range(inputs.shape[0]):
            tokens = set(int(token) for token in inputs[row])
            last = int(inputs[row, -1])
            if self.immediate_eos_id in tokens or last in {self.first_id, self.second_id}:
                next_id = self.eos_id
            elif self.first_id in tokens:
                next_id = self.first_id
            else:
                next_id = self.second_id
            logits[row, -1, next_id] = 20.0
        return logits, None


class CountingGPT(GPTLanguageModelV4):
    def __init__(self, config: GPTConfig):
        super().__init__(config)
        self.inference_input_lengths: list[int] = []
        self.inference_input_shapes: list[tuple[int, int]] = []
        self.inference_used_attention_mask: list[bool] = []

    def forward_inference(
        self,
        token_ids: torch.Tensor,
        cache=None,
        *,
        attention_mask=None,
    ):
        self.inference_input_lengths.append(int(token_ids.shape[1]))
        self.inference_input_shapes.append(tuple(token_ids.shape))
        self.inference_used_attention_mask.append(attention_mask is not None)
        return super().forward_inference(
            token_ids,
            cache,
            attention_mask=attention_mask,
        )


class ProbeCache:
    def __init__(self, tokens: torch.Tensor, owner):
        self.tokens = tokens
        self.owner = owner

    def select_rows(self, rows):
        self.owner.cache_row_selections.append(list(rows))
        indices = torch.as_tensor(rows, dtype=torch.long, device=self.tokens.device)
        return ProbeCache(self.tokens.index_select(0, indices), self.owner)


class CacheAwareEarlyExitPromptModel(EarlyExitPromptModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_row_selections: list[list[int]] = []

    @property
    def supports_left_padded_inference(self):
        return True

    def forward_inference(
        self,
        inputs: torch.Tensor,
        cache=None,
        *,
        attention_mask=None,
    ):
        tokens = inputs if cache is None else torch.cat((cache.tokens, inputs), dim=1)
        logits, _ = self.forward(tokens)
        return logits, ProbeCache(tokens, self)


class SampleSftV7Tests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = tiny_tokenizer()
        self.special = self.tokenizer.special_to_id

    @staticmethod
    def formal_public_payload() -> dict:
        required_base = dict(REQUIRED_BASE_CHECKPOINT)
        required_base["binding_sha256"] = canonical_json_sha256(REQUIRED_BASE_CHECKPOINT)
        payload = {
            "schema_version": "sft-v7-public-tensors/v1",
            "public_records": [
                {
                    "id": "public-1",
                    "primary_dimension": "core_facts_and_corrections",
                    "task_family": "core",
                    "split": "public_diagnostic",
                    "input_ids": torch.tensor([1]),
                    "labels": torch.tensor([2]),
                    "assistant_spans": [(1, 2)],
                    "assistant_turns": 1,
                    "sequence_length": 2,
                    "evaluation": {"metric": "terms"},
                }
            ],
            "vocab_size": EXPECTED_VOCAB_SIZE,
            "stoi": {},
            "itos": {},
            "special_token_ids": dict(EXPECTED_SPECIAL_TOKEN_IDS),
            "ignore_index": -100,
            "tokenizer_path": "data/scaling_a/bpe_3000/tokenizer.json",
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
            "bpe_token_manifest_path": "data/scaling_a/bpe_3000/token_manifest.json",
            "bpe_token_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "sft_dataset_manifest_sha256": "d" * 64,
            "source_jsonl_paths": {
                "public_diagnostic": "data/sft/v7/public_diagnostic.jsonl"
            },
            "source_jsonl_sha256": {"public_diagnostic": "a" * 64},
            "required_base_checkpoint": required_base,
        }
        binding = {
            "schema_version": payload["schema_version"],
            "source_jsonl_sha256": payload["source_jsonl_sha256"],
            "tokenizer_sha256": payload["tokenizer_sha256"],
            "bpe_token_manifest_sha256": payload["bpe_token_manifest_sha256"],
            "sft_dataset_manifest_sha256": payload["sft_dataset_manifest_sha256"],
            "required_base_checkpoint": payload["required_base_checkpoint"],
        }
        payload["artifact_binding_sha256"] = canonical_json_sha256(binding)
        return payload

    @staticmethod
    def checkpoint(payload: dict) -> dict:
        return {
            "schema_version": "training-checkpoint/v1",
            "config_sha256": "b" * 64,
            "step": 20,
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "extra": {
                "stage": "sft_v7_vertical",
                "base_checkpoint_path": REQUIRED_BASE_CHECKPOINT["path"],
                "base_checkpoint_sha256": REQUIRED_BASE_CHECKPOINT["sha256"],
                "base_checkpoint_step": REQUIRED_BASE_CHECKPOINT["step"],
                "base_config_canonical_sha256": REQUIRED_BASE_CHECKPOINT[
                    "config_canonical_sha256"
                ],
                "base_token_manifest_sha256": REQUIRED_BASE_CHECKPOINT[
                    "token_manifest_sha256"
                ],
                "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
                "sft_dataset_manifest_sha256": payload[
                    "sft_dataset_manifest_sha256"
                ],
                "sft_tensor_path": "data/sft/v7/train_val_tensors.pt",
                "sft_tensor_sha256": "c" * 64,
                "payload_summary": {"split_counts": {"train": 8000, "val": 800}},
                "public_records_consumed": 0,
                "sealed_records_consumed": 0,
            },
        }

    def test_multiturn_prompt_serialization_is_exact(self):
        messages = [
            {"role": "user", "content": "你问"},
            {"role": "assistant", "content": "我答"},
            {"role": "user", "content": "你好"},
        ]
        # Add the one character used only in this test to a local tokenizer.
        tokenizer = BPETokenizer(
            ["你", "问", "我", "答", "好"]
            + ["<UNK>", "<BOS>", "<USER>", "<ASSISTANT>", "<EOS>", "<PAD>"],
            [],
            ["<UNK>", "<BOS>", "<USER>", "<ASSISTANT>", "<EOS>", "<PAD>"],
        )
        ids = sample.build_conversation_prompt_ids(
            tokenizer, messages, tokenizer.special_to_id
        )

        self.assertEqual(
            ids,
            [
                tokenizer.special_to_id["<BOS>"],
                tokenizer.special_to_id["<USER>"],
                0,
                1,
                tokenizer.special_to_id["<ASSISTANT>"],
                2,
                3,
                tokenizer.special_to_id["<EOS>"],
                tokenizer.special_to_id["<USER>"],
                0,
                4,
                tokenizer.special_to_id["<ASSISTANT>"],
            ],
        )

    def test_generation_masks_unk_and_all_control_tokens_except_eos(self):
        answer_id = self.tokenizer.char_to_id["答"]
        model = MaskProbeModel(
            self.tokenizer.vocab_size,
            self.special["<UNK>"],
            answer_id,
            self.special["<EOS>"],
        )
        prompt = sample.build_conversation_prompt_ids(
            self.tokenizer,
            [{"role": "user", "content": "你问"}],
            self.special,
        )
        result = sample.generate_response(
            model,
            prompt,
            self.tokenizer,
            self.special,
            max_new_tokens=4,
            temperature=1.0,
            top_k=1,
            seed=7,
            device=torch.device("cpu"),
        )

        self.assertEqual(result["generated_text"], "答")
        self.assertTrue(result["stopped_on_eos"])
        self.assertNotIn(self.special["<UNK>"], result["generated_token_ids"])
        self.assertEqual(
            set(sample.forbidden_generation_token_ids(self.special)),
            {
                self.special["<UNK>"],
                self.special["<BOS>"],
                self.special["<USER>"],
                self.special["<ASSISTANT>"],
                self.special["<PAD>"],
            },
        )

    def test_fixed_seed_repeats_the_same_sampling_path(self):
        model = RandomProbeModel(
            self.tokenizer.vocab_size,
            self.tokenizer.char_to_id["甲"],
            self.tokenizer.char_to_id["乙"],
        )
        prompt = sample.build_conversation_prompt_ids(
            self.tokenizer,
            [{"role": "user", "content": "你问"}],
            self.special,
        )
        kwargs = dict(
            model=model,
            prompt_ids=prompt,
            tokenizer=self.tokenizer,
            special_token_ids=self.special,
            max_new_tokens=12,
            temperature=1.0,
            top_k=2,
            seed=1234,
            device=torch.device("cpu"),
        )
        first = sample.generate_response(**kwargs)
        second = sample.generate_response(**kwargs)

        self.assertEqual(first["generated_token_ids"], second["generated_token_ids"])
        self.assertEqual(first["generated_text"], second["generated_text"])

    def test_same_length_batch_matches_single_top_k_one(self):
        answer_id = self.tokenizer.char_to_id["答"]
        model = MaskProbeModel(
            self.tokenizer.vocab_size,
            self.special["<UNK>"],
            answer_id,
            self.special["<EOS>"],
        )
        prompts = [
            sample.build_conversation_prompt_ids(
                self.tokenizer,
                [{"role": "user", "content": text}],
                self.special,
            )
            for text in ("你问", "你好", "甲乙")
        ]
        seeds = [7, 11, 13]
        kwargs = dict(
            model=model,
            tokenizer=self.tokenizer,
            special_token_ids=self.special,
            max_new_tokens=4,
            temperature=0.7,
            top_k=1,
            device=torch.device("cpu"),
        )
        singles = [
            sample.generate_response(prompt_ids=prompt, seed=seed, **kwargs)
            for prompt, seed in zip(prompts, seeds)
        ]
        batched = sample.generate_responses_same_length_batch(
            prompt_ids_batch=prompts,
            seeds=seeds,
            **kwargs,
        )

        self.assertEqual(batched, singles)

    def test_same_length_batch_matches_single_stochastic_fixed_seeds(self):
        model = RandomProbeModel(
            self.tokenizer.vocab_size,
            self.tokenizer.char_to_id["甲"],
            self.tokenizer.char_to_id["乙"],
        )
        prompt = sample.build_conversation_prompt_ids(
            self.tokenizer,
            [{"role": "user", "content": "你问"}],
            self.special,
        )
        prompts = [prompt, list(prompt), list(prompt)]
        seeds = [1234, 9876, 2026]
        kwargs = dict(
            model=model,
            tokenizer=self.tokenizer,
            special_token_ids=self.special,
            max_new_tokens=12,
            temperature=0.9,
            top_k=2,
            device=torch.device("cpu"),
        )
        singles = [
            sample.generate_response(prompt_ids=item, seed=seed, **kwargs)
            for item, seed in zip(prompts, seeds)
        ]
        batched = sample.generate_responses_same_length_batch(
            prompt_ids_batch=prompts,
            seeds=seeds,
            **kwargs,
        )

        self.assertEqual(batched, singles)

    def _bias_only_counting_gpt(self, *, block_size: int = 6) -> CountingGPT:
        model = CountingGPT(
            GPTConfig(
                vocab_size=self.tokenizer.vocab_size,
                block_size=block_size,
                embedding_size=16,
                num_layers=2,
                num_heads=4,
                dropout=0.0,
            )
        )
        with torch.no_grad():
            # Tied output/token weights are zeroed so cached and uncached scores
            # are intentionally identical; the test isolates decoding policy.
            model.output.weight.zero_()
            model.output.bias.fill_(-20.0)
            model.output.bias[self.tokenizer.char_to_id["甲"]] = 2.0
            model.output.bias[self.tokenizer.char_to_id["乙"]] = 1.0
        return model

    def test_cached_single_generation_matches_legacy_across_sliding_window(self):
        prompt = sample.build_conversation_prompt_ids(
            self.tokenizer,
            [{"role": "user", "content": "你问"}],
            self.special,
        )
        self.assertEqual(len(prompt), 5)
        for top_k in (1, 2):
            with self.subTest(top_k=top_k):
                model = self._bias_only_counting_gpt(block_size=6)
                kwargs = dict(
                    model=model,
                    prompt_ids=prompt,
                    tokenizer=self.tokenizer,
                    special_token_ids=self.special,
                    max_new_tokens=5,
                    temperature=0.8,
                    top_k=top_k,
                    seed=20260829,
                    device=torch.device("cpu"),
                )
                legacy = sample.generate_response(**kwargs, use_kv_cache=False)
                model.inference_input_lengths.clear()
                cached = sample.generate_response(**kwargs, use_kv_cache=True)

                self.assertEqual(cached, legacy)
                # Fill the final free slot incrementally, then rebuild the full
                # six-token cropped window on every subsequent sliding step.
                self.assertEqual(model.inference_input_lengths, [5, 1, 6, 6, 6])

    def test_cached_batch_matches_legacy_singles_for_top_k_one_and_two(self):
        prompts = [
            sample.build_conversation_prompt_ids(
                self.tokenizer,
                [{"role": "user", "content": text}],
                self.special,
            )
            for text in ("你问", "甲乙", "你好")
        ]
        seeds = [7, 11, 13]
        for top_k in (1, 2):
            with self.subTest(top_k=top_k):
                model = self._bias_only_counting_gpt(block_size=8)
                kwargs = dict(
                    model=model,
                    tokenizer=self.tokenizer,
                    special_token_ids=self.special,
                    max_new_tokens=6,
                    temperature=0.9,
                    top_k=top_k,
                    device=torch.device("cpu"),
                )
                legacy = [
                    sample.generate_response(
                        prompt_ids=prompt,
                        seed=seed,
                        use_kv_cache=False,
                        **kwargs,
                    )
                    for prompt, seed in zip(prompts, seeds)
                ]
                cached_batch = sample.generate_responses_same_length_batch(
                    prompt_ids_batch=prompts,
                    seeds=seeds,
                    use_kv_cache=True,
                    **kwargs,
                )

                self.assertEqual(cached_batch, legacy)

    def test_cross_length_batches_match_legacy_singles_top_k_one_and_two(self):
        prompts = [
            sample.build_conversation_prompt_ids(
                self.tokenizer,
                [{"role": "user", "content": text}],
                self.special,
            )
            for text in ("你", "你问", "甲乙。")
        ]
        self.assertEqual([len(prompt) for prompt in prompts], [4, 5, 6])
        seeds = [7, 11, 13]
        for top_k in (1, 2):
            with self.subTest(top_k=top_k):
                model = self._bias_only_counting_gpt(block_size=8)
                kwargs = dict(
                    model=model,
                    tokenizer=self.tokenizer,
                    special_token_ids=self.special,
                    max_new_tokens=5,
                    temperature=0.9,
                    top_k=top_k,
                    device=torch.device("cpu"),
                )
                legacy = [
                    sample.generate_response(
                        prompt_ids=prompt,
                        seed=seed,
                        use_kv_cache=False,
                        **kwargs,
                    )
                    for prompt, seed in zip(prompts, seeds)
                ]
                model.inference_input_lengths.clear()
                model.inference_input_shapes.clear()
                model.inference_used_attention_mask.clear()
                batched = sample.generate_responses_batched(
                    prompt_ids_batch=prompts,
                    seeds=seeds,
                    generation_batch_size=3,
                    use_kv_cache=True,
                    **kwargs,
                )

                self.assertEqual(batched, legacy)
                self.assertEqual(model.inference_input_shapes[0], (3, 6))
                self.assertTrue(model.inference_used_attention_mask[0])
                self.assertIn((3, 1), model.inference_input_shapes)
                # The longest row reaches block_size first, forcing a safe
                # full-window rebuild for the whole heterogeneous chunk.
                self.assertIn((3, 8), model.inference_input_shapes)

    def test_cached_batch_removes_eos_rows_without_reordering_survivors(self):
        model = CacheAwareEarlyExitPromptModel(
            self.tokenizer.vocab_size,
            self.tokenizer.char_to_id["甲"],
            self.tokenizer.char_to_id["乙"],
            self.tokenizer.char_to_id["好"],
            self.special["<EOS>"],
        )
        prompts = [
            sample.build_conversation_prompt_ids(
                self.tokenizer,
                [{"role": "user", "content": text}],
                self.special,
            )
            for text in ("好好", "甲甲", "乙乙")
        ]
        seeds = [1, 2, 3]
        kwargs = dict(
            model=model,
            tokenizer=self.tokenizer,
            special_token_ids=self.special,
            max_new_tokens=4,
            temperature=1.0,
            top_k=1,
            device=torch.device("cpu"),
        )
        legacy = [
            sample.generate_response(
                prompt_ids=prompt,
                seed=seed,
                use_kv_cache=False,
                **kwargs,
            )
            for prompt, seed in zip(prompts, seeds)
        ]
        cached_batch = sample.generate_responses_same_length_batch(
            prompt_ids_batch=prompts,
            seeds=seeds,
            use_kv_cache=True,
            **kwargs,
        )

        self.assertEqual(cached_batch, legacy)
        self.assertEqual(model.cache_row_selections, [[1, 2]])
        self.assertEqual(
            [result["generated_text"] for result in cached_batch],
            ["", "甲", "乙"],
        )
        self.assertTrue(all(result["stopped_on_eos"] for result in cached_batch))

    def test_cross_length_cached_batch_removes_eos_rows(self):
        model = CacheAwareEarlyExitPromptModel(
            self.tokenizer.vocab_size,
            self.tokenizer.char_to_id["甲"],
            self.tokenizer.char_to_id["乙"],
            self.tokenizer.char_to_id["好"],
            self.special["<EOS>"],
        )
        prompts = [
            sample.build_conversation_prompt_ids(
                self.tokenizer,
                [{"role": "user", "content": text}],
                self.special,
            )
            for text in ("好", "甲甲", "乙乙乙")
        ]
        seeds = [1, 2, 3]
        kwargs = dict(
            model=model,
            tokenizer=self.tokenizer,
            special_token_ids=self.special,
            max_new_tokens=4,
            temperature=1.0,
            top_k=1,
            device=torch.device("cpu"),
        )
        legacy = [
            sample.generate_response(
                prompt_ids=prompt,
                seed=seed,
                use_kv_cache=False,
                **kwargs,
            )
            for prompt, seed in zip(prompts, seeds)
        ]
        batched = sample.generate_responses_batched(
            prompt_ids_batch=prompts,
            seeds=seeds,
            generation_batch_size=3,
            use_kv_cache=True,
            **kwargs,
        )

        self.assertEqual(batched, legacy)
        self.assertEqual(model.cache_row_selections, [[1, 2]])
        self.assertEqual(
            [result["generated_text"] for result in batched],
            ["", "甲", "乙"],
        )
        self.assertTrue(all(result["stopped_on_eos"] for result in batched))

    def test_cache_rebuilds_if_eos_removes_the_row_that_set_padded_width(self):
        model = self._bias_only_counting_gpt(block_size=6).eval()
        decode_session = sample._InferenceDecodeSession(
            model,
            torch.device("cpu"),
            use_kv_cache=True,
            padding_token_id=self.special["<PAD>"],
        )
        short_context = [
            self.tokenizer.char_to_id["甲"],
            self.tokenizer.char_to_id["乙"],
        ]
        long_context = short_context * 3

        decode_session.next_token_scores([short_context, long_context])
        decode_session.retain_rows([0])
        decode_session.next_token_scores(
            [short_context + [self.tokenizer.char_to_id["甲"]]]
        )

        self.assertEqual(model.inference_input_shapes, [(2, 6), (1, 3)])
        self.assertEqual(model.inference_used_attention_mask, [True, False])

    def test_length_bucketing_preserves_order_and_removes_early_eos_rows(self):
        model = EarlyExitPromptModel(
            self.tokenizer.vocab_size,
            self.tokenizer.char_to_id["甲"],
            self.tokenizer.char_to_id["乙"],
            self.tokenizer.char_to_id["好"],
            self.special["<EOS>"],
        )
        prompts = [
            sample.build_conversation_prompt_ids(
                self.tokenizer,
                [{"role": "user", "content": text}],
                self.special,
            )
            for text in ("乙乙", "好", "甲")
        ]
        seeds = [1, 2, 3]
        results = sample.generate_responses_length_bucketed(
            model,
            prompts,
            self.tokenizer,
            self.special,
            max_new_tokens=4,
            temperature=1.0,
            top_k=1,
            seeds=seeds,
            device=torch.device("cpu"),
            generation_batch_size=8,
        )
        singles = [
            sample.generate_response(
                model,
                prompt,
                self.tokenizer,
                self.special,
                max_new_tokens=4,
                temperature=1.0,
                top_k=1,
                seed=seed,
                device=torch.device("cpu"),
            )
            for prompt, seed in zip(prompts, seeds)
        ]

        self.assertEqual(results, singles)
        self.assertEqual([result["generated_text"] for result in results], ["乙", "", "甲"])
        self.assertTrue(all(result["stopped_on_eos"] for result in results))

    def test_public_payload_rejects_nonpublic_fields_and_paths(self):
        payload = self.formal_public_payload()
        sample.validate_public_payload(payload, expected_count=1)

        for forbidden_key in ("sealed_test_records", "test_records"):
            changed = dict(payload)
            changed[forbidden_key] = []
            with self.assertRaisesRegex(sample.SFTV7SamplingError, "forbidden field"):
                sample.validate_public_payload(changed, expected_count=1)

        changed = dict(payload)
        changed["source_jsonl_paths"] = {
            "public_diagnostic": "data/sealed_test/public_diagnostic.jsonl"
        }
        with self.assertRaisesRegex(sample.SFTV7SamplingError, "forbidden path"):
            sample.validate_public_payload(changed, expected_count=1)

        changed = deepcopy(payload)
        changed["public_records"][0]["evaluation"]["reference_answer"] = "正文"
        with self.assertRaisesRegex(sample.SFTV7SamplingError, "plaintext field"):
            sample.validate_public_payload(changed, expected_count=1)

    def test_checkpoint_requires_step5750_lineage_and_zero_public_consumption(self):
        payload = self.formal_public_payload()
        checkpoint = self.checkpoint(payload)
        summary = sample.validate_checkpoint_provenance(checkpoint, payload)
        self.assertEqual(summary["step"], 20)
        self.assertEqual(summary["training_split_counts"], {"train": 8000, "val": 800})

        changed = self.checkpoint(payload)
        changed["extra"]["base_checkpoint_step"] = 5500
        with self.assertRaisesRegex(sample.SFTV7SamplingError, "base_checkpoint_step"):
            sample.validate_checkpoint_provenance(changed, payload)

        changed = self.checkpoint(payload)
        changed["extra"]["public_records_consumed"] = 1
        with self.assertRaisesRegex(sample.SFTV7SamplingError, "public records"):
            sample.validate_checkpoint_provenance(changed, payload)

    def test_config_and_baseline_modes_are_strict(self):
        from train_pretrain_v4 import load_config

        config = load_config(Path("configs/formal_pretrain_14m_bpe3000.json"))
        sample.validate_model_config(config)
        changed = deepcopy(config)
        changed["model"]["dropout"] = 0.2
        with self.assertRaisesRegex(sample.SFTV7SamplingError, "Step 5750"):
            sample.validate_model_config(changed)
        changed = deepcopy(config)
        changed["model"]["layer_norm_epsilon"] = 1e-6
        with self.assertRaisesRegex(sample.SFTV7SamplingError, "Step 5750"):
            sample.validate_model_config(changed)

        baseline = {
            "schema_version": "training-checkpoint/v1",
            "step": REQUIRED_BASE_CHECKPOINT["step"],
            "config_sha256": REQUIRED_BASE_CHECKPOINT["config_canonical_sha256"],
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "extra": {
                "initial_checkpoint": None,
                "parameter_count": REQUIRED_BASE_CHECKPOINT["parameter_count"],
                "token_manifest_sha256": REQUIRED_BASE_CHECKPOINT[
                    "token_manifest_sha256"
                ],
                "model_config": {
                    "vocab_size": 7465,
                    "block_size": 512,
                    "embedding_size": 320,
                    "num_layers": 10,
                    "num_heads": 8,
                    "ffn_multiplier": 4,
                    "dropout": 0.1,
                    "layer_norm_epsilon": 1e-5,
                    "initialization_std": 0.02,
                    "tie_embeddings": True,
                },
            },
        }
        with mock.patch.object(
            sample, "file_sha256", return_value=REQUIRED_BASE_CHECKPOINT["sha256"]
        ):
            result = sample.validate_pretrain_baseline_checkpoint(
                baseline, Path(REQUIRED_BASE_CHECKPOINT["path"])
            )
            self.assertEqual(result["checkpoint_mode"], "pretrain-baseline")
            baseline["extra"]["sft_dataset_manifest_sha256"] = "d" * 64
            with self.assertRaisesRegex(sample.SFTV7SamplingError, "SFT provenance"):
                sample.validate_pretrain_baseline_checkpoint(
                    baseline, Path(REQUIRED_BASE_CHECKPOINT["path"])
                )

    def test_invalid_role_order_and_overlong_prompt_fail_without_body_echo(self):
        secret_body = "你问"
        with self.assertRaises(sample.SFTV7SamplingError) as caught:
            sample.build_conversation_prompt_ids(
                self.tokenizer,
                [{"role": "assistant", "content": secret_body}],
                self.special,
            )
        self.assertNotIn(secret_body, str(caught.exception))

        with self.assertRaisesRegex(sample.SFTV7SamplingError, "512-token"):
            sample.build_conversation_prompt_ids(
                self.tokenizer,
                [{"role": "user", "content": "你" * 600}],
                self.special,
            )

    def test_cli_success_logs_counts_but_not_prompt_or_generated_body(self):
        secret_prompt = "你问"
        model = MaskProbeModel(
            self.tokenizer.vocab_size,
            self.special["<UNK>"],
            self.tokenizer.char_to_id["答"],
            self.special["<EOS>"],
        )
        payload = {
            "special_token_ids": self.special,
            "tokenizer_sha256": "a" * 64,
            "bpe_token_manifest_sha256": "b" * 64,
            "sft_dataset_manifest_sha256": "d" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_json = root / "samples.json"
            output_md = root / "samples.md"
            logs = root / "logs"
            argv = [
                "--prompt",
                secret_prompt,
                "--public-tensors",
                str(root / "public_diagnostic_tensors.pt"),
                "--checkpoint",
                str(root / "checkpoint.pt"),
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_md),
                "--log-dir",
                str(logs),
                "--device",
                "cpu",
                "--top-k",
                "1",
            ]
            config = {
                "logging": {"max_bytes": 4096, "backup_count": 1, "console": False}
            }
            with (
                mock.patch.object(sample, "load_config", return_value=config),
                mock.patch.object(sample, "load_public_payload", return_value=payload),
                mock.patch.object(sample, "load_bound_tokenizer", return_value=self.tokenizer),
                mock.patch.object(sample, "select_device", return_value=torch.device("cpu")),
                mock.patch.object(
                    sample,
                    "load_model_bundle",
                    return_value=(
                        model,
                        {"step": 20},
                        {"step": 20, "base_checkpoint_sha256": "b" * 64},
                    ),
                ),
                mock.patch.object(sample, "file_sha256", return_value="d" * 64),
            ):
                self.assertEqual(sample.main(argv), 0)

            report = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(report["samples"][0]["messages"][0]["content"], secret_prompt)
            self.assertEqual(report["samples"][0]["generated_text"], "答")
            log_text = "".join(path.read_text(encoding="utf-8") for path in logs.glob("*.jsonl"))
            self.assertNotIn(secret_prompt, log_text)
            self.assertNotIn("答", log_text)
            self.assertIn('"module": "cloud.generation"', log_text)
            self.assertIn('"module": "cloud.checkpoint"', log_text)

    def test_cli_failure_log_has_safe_code_and_no_prompt(self):
        secret_prompt = "你问"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root / "logs"
            argv = [
                "--prompt",
                secret_prompt,
                "--public-tensors",
                str(root / "public_diagnostic_tensors.pt"),
                "--checkpoint",
                str(root / "checkpoint.pt"),
                "--log-dir",
                str(logs),
            ]
            config = {
                "logging": {"max_bytes": 4096, "backup_count": 1, "console": False}
            }
            with (
                mock.patch.object(sample, "load_config", return_value=config),
                mock.patch.object(
                    sample,
                    "load_public_payload",
                    side_effect=sample.SFTV7SamplingError("public_scope", "safe failure"),
                ),
            ):
                with self.assertRaises(sample.SFTV7SamplingError):
                    sample.main(argv)
            log_text = "".join(path.read_text(encoding="utf-8") for path in logs.glob("*.jsonl"))
            self.assertIn("error_code=public_scope", log_text)
            self.assertNotIn(secret_prompt, log_text)


if __name__ == "__main__":
    unittest.main()
