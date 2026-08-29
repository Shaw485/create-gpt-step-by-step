from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from bpe_tokenizer import BPETokenizer
from build_sft_v7_1_canary import CANARY_CONFIG_BINDING
from prepare_bpe_sft_v7 import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SPECIAL_TOKEN_IDS,
    EXPECTED_TOKENIZER_SHA256,
    REQUIRED_BASE_CHECKPOINT,
)
from prepare_sft_v7_1_canary import (
    DATASET_MANIFEST_SCHEMA,
    DATASET_RECORD_SCHEMA,
    EXPECTED_DIMENSION,
    EXPECTED_FAMILY,
    TENSOR_SCHEMA,
    CanaryEncodingError,
    load_and_validate_canary_manifest,
    parse_args,
    prepare_canary_payload,
    reject_restricted_keys,
    validate_source_records,
)
from training_runtime import file_sha256


def tiny_tokenizer() -> BPETokenizer:
    tokens = list("问答事实零一二三四五六七八九01234567？。")
    specials = list(EXPECTED_SPECIAL_TOKEN_IDS)
    return BPETokenizer(tokens + specials, [], specials)


def source_record(split: str, fact_index: int, wording_index: int) -> dict:
    identifier = f"{split}-{fact_index}-{wording_index}"
    question = f"问事实{fact_index}？"
    answer = f"答事实{fact_index}。"
    return {
        "schema_version": DATASET_RECORD_SCHEMA,
        "id": identifier,
        "split": split,
        "fact_id": f"fact-{fact_index}",
        "primary_dimension": EXPECTED_DIMENSION,
        "task_family": EXPECTED_FAMILY,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "supervision": {
            "assistant_only_loss": True,
            "eos_appended_by_encoder": True,
            "use_for_training": split == "train",
        },
        "evaluation": {
            "metric": "required_terms_all",
            "required_terms": ["答"],
            "forbidden_terms": ["零"],
            "known_fact": True,
        },
    }


def all_source_records() -> tuple[list[dict], list[dict]]:
    train = [source_record("train", fact, wording) for fact in range(8) for wording in range(8)]
    evaluation = [
        source_record("holdout_eval", fact, wording)
        for fact in range(8)
        for wording in range(2)
    ]
    return train, evaluation


class PrepareSftV71CanaryTests(unittest.TestCase):
    def test_payload_has_only_train_and_eval_records_and_supervises_eos(self):
        train, evaluation = all_source_records()
        tokenizer = tiny_tokenizer()
        special = tokenizer.special_to_id
        payload, report = prepare_canary_payload(
            train,
            evaluation,
            tokenizer,
            special,
            tokenizer_path=Path("tokenizer.json"),
            token_manifest_path=Path("token_manifest.json"),
            tokenizer_sha256=EXPECTED_TOKENIZER_SHA256,
            token_manifest_sha256=EXPECTED_MANIFEST_SHA256,
            dataset_manifest_path=Path("manifest.json"),
            dataset_manifest_sha256="a" * 64,
            dataset_identity_sha256="b" * 64,
            train_path=Path("train.jsonl"),
            eval_path=Path("holdout_eval.jsonl"),
            source_hashes={"train": "c" * 64, "holdout_eval": "d" * 64},
        )

        self.assertEqual(payload["schema_version"], TENSOR_SCHEMA)
        self.assertEqual(
            {key for key in payload if key.endswith("_records")},
            {"train_records", "eval_records"},
        )
        self.assertEqual(len(payload["train_records"]), 64)
        self.assertEqual(len(payload["eval_records"]), 16)
        self.assertTrue(report["eos_appended_and_supervised"])
        for record in payload["train_records"] + payload["eval_records"]:
            supervised = record["labels"][record["labels"] != -100]
            self.assertEqual(int(supervised[-1]), special["<EOS>"])
            self.assertNotIn("messages", record)
        reject_restricted_keys(payload)

    def test_source_contract_keeps_holdout_out_of_training(self):
        train, evaluation = all_source_records()
        summary = validate_source_records(train, evaluation)
        self.assertEqual(summary["split_counts"], {"train": 64, "holdout_eval": 16})

        broken = dict(evaluation[0])
        broken["supervision"] = dict(broken["supervision"])
        broken["supervision"]["use_for_training"] = True
        evaluation[0] = broken
        with self.assertRaisesRegex(CanaryEncodingError, "supervision role"):
            validate_source_records(train, evaluation)

    def test_manifest_binds_exact_two_canary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            eval_path = root / "holdout_eval.jsonl"
            train_path.write_text("{}\n" * 64, encoding="utf-8")
            eval_path.write_text("{}\n" * 16, encoding="utf-8")
            manifest = {
                "manifest_schema_version": DATASET_MANIFEST_SCHEMA,
                "record_schema_version": DATASET_RECORD_SCHEMA,
                "status": "frozen_canary_ready",
                "config": dict(CANARY_CONFIG_BINDING),
                "split_totals": {"train": 64, "holdout_eval": 16},
                "split_files": {
                    "train": {
                        "path": "train.jsonl",
                        "count": 64,
                        "sha256": file_sha256(train_path),
                        "schema_version": DATASET_RECORD_SCHEMA,
                    },
                    "holdout_eval": {
                        "path": "holdout_eval.jsonl",
                        "count": 16,
                        "sha256": file_sha256(eval_path),
                        "schema_version": DATASET_RECORD_SCHEMA,
                    },
                },
                "dataset_identity_sha256": "e" * 64,
                "training_binding": {
                    "base_checkpoint": {
                        "path": REQUIRED_BASE_CHECKPOINT["path"],
                        "sha256": REQUIRED_BASE_CHECKPOINT["sha256"],
                        "step": REQUIRED_BASE_CHECKPOINT["step"],
                        "parameter_count": REQUIRED_BASE_CHECKPOINT["parameter_count"],
                    },
                    "tokenizer": {
                        "sha256": EXPECTED_TOKENIZER_SHA256,
                        "vocab_size": 7465,
                        "context_limit": 512,
                    },
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = load_and_validate_canary_manifest(
                manifest_path,
                {"train": train_path, "holdout_eval": eval_path},
                enforce_frozen_binding=False,
            )
            self.assertEqual(result["source_jsonl_sha256"]["train"], file_sha256(train_path))

            train_path.write_text("{}\n" * 63, encoding="utf-8")
            with self.assertRaises(CanaryEncodingError) as caught:
                load_and_validate_canary_manifest(
                    manifest_path,
                    {"train": train_path, "holdout_eval": eval_path},
                    enforce_frozen_binding=False,
                )
            self.assertIn(caught.exception.code, {"manifest_sha_mismatch", "jsonl_count_mismatch"})

    def test_restricted_scope_key_is_rejected_without_echoing_body(self):
        with self.assertRaises(CanaryEncodingError) as caught:
            reject_restricted_keys({"public_records": ["sensitive body"]})
        self.assertEqual(caught.exception.code, "restricted_scope_key")
        self.assertNotIn("sensitive body", str(caught.exception))

    def test_log_modules_have_independent_cli_overrides(self):
        args = parse_args(
            [
                "--data-log-level",
                "DEBUG",
                "--encoding-log-level",
                "OFF",
                "--validation-log-level",
                "ERROR",
            ]
        )
        self.assertEqual(args.data_log_level, "DEBUG")
        self.assertEqual(args.encoding_log_level, "OFF")
        self.assertEqual(args.validation_log_level, "ERROR")


if __name__ == "__main__":
    unittest.main()
