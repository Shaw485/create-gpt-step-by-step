from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

import torch

from bpe_tokenizer import BPETokenizer
from prepare_bpe_sft_v7 import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SPECIAL_TOKEN_IDS,
    EXPECTED_TOKENIZER_SHA256,
    IGNORE_INDEX,
    SFTV7EncodingError,
    load_and_validate_formal_tokenizer,
    load_and_validate_dataset_manifest,
    main,
    prepare_payloads,
    require_formal_special_ids,
    serialize_messages,
)
from training_runtime import file_sha256


REPOSITORY = Path(__file__).resolve().parents[1]
FORMAL_TOKENIZER = REPOSITORY / "data/scaling_a/bpe_3000/tokenizer.json"
FORMAL_MANIFEST = REPOSITORY / "data/scaling_a/bpe_3000/token_manifest.json"
FORBIDDEN_SCOPE_TOKEN = re.compile(r"(?:^|[^a-z])(sealed|test)(?:$|[^a-z])")


def record(
    identifier: str,
    split: str,
    messages: list[dict[str, str]],
    *,
    evaluation: dict | None = None,
) -> dict:
    value = {
        "id": identifier,
        "split": split,
        "primary_dimension": "novel_domain",
        "task_family": "novel_qa",
        "messages": messages,
        "question": messages[-2]["content"],
        "answer": messages[-1]["content"],
    }
    if evaluation is not None:
        value["evaluation"] = evaluation
    return value


def expected_encoding(
    tokenizer: BPETokenizer,
    special_ids: dict[str, int],
    messages: list[dict[str, str]],
) -> tuple[list[int], list[int]]:
    sequence = [special_ids["<BOS>"]]
    target_mask = [False]
    for message in messages:
        if message["role"] == "user":
            content_ids = tokenizer.encode(message["content"])
            sequence.extend([special_ids["<USER>"], *content_ids])
            target_mask.extend([False] * (len(content_ids) + 1))
        else:
            content_ids = tokenizer.encode(message["content"])
            sequence.extend([special_ids["<ASSISTANT>"], *content_ids, special_ids["<EOS>"]])
            target_mask.extend([False, *([True] * len(content_ids)), True])
    labels = sequence[1:]
    labels = [token_id if supervised else IGNORE_INDEX for token_id, supervised in zip(labels, target_mask[1:])]
    return sequence[:-1], labels


def assert_no_forbidden_scope(test_case: unittest.TestCase, value) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            test_case.assertIsNone(FORBIDDEN_SCOPE_TOKEN.search(str(key).lower()))
            assert_no_forbidden_scope(test_case, item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_no_forbidden_scope(test_case, item)
    elif isinstance(value, str):
        test_case.assertIsNone(FORBIDDEN_SCOPE_TOKEN.search(value.lower()))


class PrepareBpeSftV7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer, cls.special_ids, cls.identity = load_and_validate_formal_tokenizer(
            FORMAL_TOKENIZER,
            FORMAL_MANIFEST,
        )

    def test_formal_tokenizer_has_exact_six_special_ids(self):
        self.assertEqual(self.special_ids, EXPECTED_SPECIAL_TOKEN_IDS)
        self.assertEqual(self.identity["tokenizer_sha256"], EXPECTED_TOKENIZER_SHA256)
        self.assertEqual(self.identity["manifest_sha256"], EXPECTED_MANIFEST_SHA256)

        small = BPETokenizer(tokens=["甲", "乙"], merges=[]).with_special_tokens(
            list(EXPECTED_SPECIAL_TOKEN_IDS)
        )
        with self.assertRaisesRegex(SFTV7EncodingError, "special-token IDs"):
            require_formal_special_ids(small)

    def test_dataset_manifest_binds_routine_files_without_opening_blind_body(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "train": root / "train.jsonl",
                "val": root / "val.jsonl",
                "public_diagnostic": root / "public_diagnostic.jsonl",
            }
            counts = {"train": 8000, "val": 800, "public_diagnostic": 600}
            for split, path in paths.items():
                path.write_text("{}\n" * counts[split], encoding="utf-8")
            manifest = {
                "manifest_schema_version": "sft-v7-vertical-manifest/v1",
                "record_schema_version": "sft_v7_vertical/1.0",
                "split_files": {
                    split: {
                        "path": path.name,
                        "count": counts[split],
                        "sha256": file_sha256(path),
                    }
                    for split, path in paths.items()
                },
            }
            manifest["split_files"]["sealed_test"] = {
                "path": "must_not_be_opened.jsonl",
                "count": 600,
                "sha256": "f" * 64,
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            identity = load_and_validate_dataset_manifest(manifest_path, paths)
            self.assertEqual(
                identity["sft_dataset_manifest_sha256"], file_sha256(manifest_path)
            )

            paths["val"].write_text("{}\n" * 799, encoding="utf-8")
            with self.assertRaises(SFTV7EncodingError) as caught:
                load_and_validate_dataset_manifest(manifest_path, paths)
            self.assertIn(caught.exception.code, {"dataset_jsonl_count_mismatch", "dataset_manifest_sha_mismatch"})

    def test_single_turn_labels_only_assistant_text_and_eos(self):
        messages = [
            {"role": "user", "content": "萧炎是谁？"},
            {"role": "assistant", "content": "萧炎是主要人物。"},
        ]
        encoded = serialize_messages(
            record("single-1", "train", messages),
            self.tokenizer,
            self.special_ids,
        )
        expected_inputs, expected_labels = expected_encoding(
            self.tokenizer,
            self.special_ids,
            messages,
        )

        self.assertEqual(encoded["input_ids"].tolist(), expected_inputs)
        self.assertEqual(encoded["labels"].tolist(), expected_labels)
        supervised = encoded["labels"][encoded["labels"] != IGNORE_INDEX].tolist()
        self.assertEqual(supervised[-1], self.special_ids["<EOS>"])
        self.assertEqual(
            self.tokenizer.decode(supervised[:-1]),
            messages[-1]["content"],
        )
        self.assertNotIn("question", encoded)
        self.assertNotIn("answer", encoded)
        self.assertNotIn("messages", encoded)

    def test_multiturn_labels_match_every_position_and_each_eos(self):
        messages = [
            {"role": "user", "content": "药老是谁？"},
            {"role": "assistant", "content": "药老即药尘。"},
            {"role": "user", "content": "他和萧炎是什么关系？"},
            {"role": "assistant", "content": "他是萧炎的重要老师。"},
        ]
        encoded = serialize_messages(
            record("multi-1", "val", messages),
            self.tokenizer,
            self.special_ids,
        )
        expected_inputs, expected_labels = expected_encoding(
            self.tokenizer,
            self.special_ids,
            messages,
        )

        self.assertEqual(encoded["input_ids"].tolist(), expected_inputs)
        self.assertEqual(encoded["labels"].tolist(), expected_labels)
        supervised = encoded["labels"][encoded["labels"] != IGNORE_INDEX].tolist()
        self.assertEqual(supervised.count(self.special_ids["<EOS>"]), 2)
        self.assertEqual(encoded["assistant_turns"], 2)
        answer_ids = [token_id for token_id in supervised if token_id != self.special_ids["<EOS>"]]
        self.assertEqual(
            self.tokenizer.decode(answer_ids),
            "药老即药尘。他是萧炎的重要老师。",
        )
        for role_token in ("<BOS>", "<USER>", "<ASSISTANT>", "<PAD>", "<UNK>"):
            self.assertNotIn(self.special_ids[role_token], supervised)

    def test_public_record_retains_only_non_plaintext_evaluation_metadata(self):
        messages = [
            {"role": "user", "content": "萧炎是谁？"},
            {"role": "assistant", "content": "萧炎是主要人物。"},
        ]
        evaluation = {
            "metric": "required_terms",
            "required_any": ["萧炎", "主要人物"],
            "forbidden_any": ["资料不足"],
            "weight": 1.0,
        }
        encoded = serialize_messages(
            record(
                "public-1",
                "public_diagnostic",
                messages,
                evaluation=evaluation,
            ),
            self.tokenizer,
            self.special_ids,
            retain_evaluation=True,
        )

        self.assertEqual(encoded["evaluation"], evaluation)
        self.assertEqual(encoded["primary_dimension"], "novel_domain")
        self.assertEqual(encoded["task_family"], "novel_qa")
        self.assertNotIn("question", encoded)
        self.assertNotIn("answer", encoded)
        self.assertNotIn("messages", encoded)

        bad = record(
            "public-2",
            "public_diagnostic",
            messages,
            evaluation={"metric": "exact", "reference_answer": "正文"},
        )
        with self.assertRaisesRegex(SFTV7EncodingError, "plaintext field"):
            serialize_messages(
                bad,
                self.tokenizer,
                self.special_ids,
                retain_evaluation=True,
            )

    def test_oov_content_is_rejected_without_echoing_body(self):
        messages = [
            {"role": "user", "content": "萧炎𠮷"},
            {"role": "assistant", "content": "萧炎。"},
        ]
        with self.assertRaises(SFTV7EncodingError) as caught:
            serialize_messages(
                record("oov-1", "train", messages),
                self.tokenizer,
                self.special_ids,
            )
        self.assertEqual(caught.exception.code, "unencodable_content")
        self.assertNotIn("𠮷", str(caught.exception))

    def test_wrong_role_order_is_rejected(self):
        messages = [
            {"role": "assistant", "content": "萧炎。"},
            {"role": "user", "content": "萧炎是谁？"},
        ]
        with self.assertRaises(SFTV7EncodingError) as caught:
            serialize_messages(
                record("roles-1", "train", messages),
                self.tokenizer,
                self.special_ids,
            )
        self.assertEqual(caught.exception.code, "invalid_role_order")

    def test_sequence_over_512_is_rejected(self):
        messages = [
            {"role": "user", "content": "萧\n" * 600},
            {"role": "assistant", "content": "萧炎。"},
        ]
        with self.assertRaises(SFTV7EncodingError) as caught:
            serialize_messages(
                record("long-1", "train", messages),
                self.tokenizer,
                self.special_ids,
            )
        self.assertEqual(caught.exception.code, "sequence_too_long")

    def test_manifest_and_tokenizer_hash_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            altered_manifest = root / "token_manifest.json"
            altered_manifest.write_bytes(FORMAL_MANIFEST.read_bytes() + b"\n")
            with self.assertRaises(SFTV7EncodingError) as manifest_error:
                load_and_validate_formal_tokenizer(FORMAL_TOKENIZER, altered_manifest)
            self.assertEqual(manifest_error.exception.code, "manifest_sha256_mismatch")

            altered_tokenizer = root / "tokenizer.json"
            altered_tokenizer.write_bytes(FORMAL_TOKENIZER.read_bytes() + b"\n")
            with self.assertRaises(SFTV7EncodingError) as tokenizer_error:
                load_and_validate_formal_tokenizer(altered_tokenizer, FORMAL_MANIFEST)
            self.assertEqual(tokenizer_error.exception.code, "tokenizer_sha256_mismatch")

    def test_manifest_split_paths_are_never_opened(self):
        manifest = json.loads(FORMAL_MANIFEST.read_text(encoding="utf-8"))
        forbidden_paths = {
            (REPOSITORY / manifest["splits"]["test"]["text_path"]).resolve(),
            (REPOSITORY / manifest["splits"]["test"]["tensor_path"]).resolve(),
        }
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            candidate = path.resolve()
            if candidate in forbidden_paths:
                raise AssertionError("encoder opened a blind-evaluation split path")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open):
            tokenizer, special_ids, identity = load_and_validate_formal_tokenizer(
                FORMAL_TOKENIZER,
                FORMAL_MANIFEST,
            )
        self.assertEqual(tokenizer.vocab_size, 7465)
        self.assertEqual(special_ids, EXPECTED_SPECIAL_TOKEN_IDS)
        self.assertEqual(identity["manifest_sha256"], EXPECTED_MANIFEST_SHA256)

    def test_payloads_are_isolated_and_bind_all_source_hashes(self):
        train = record(
            "train-1",
            "train",
            [
                {"role": "user", "content": "萧炎是谁？"},
                {"role": "assistant", "content": "萧炎是主要人物。"},
            ],
        )
        val = record(
            "val-1",
            "val",
            [
                {"role": "user", "content": "药老是谁？"},
                {"role": "assistant", "content": "药老即药尘。"},
            ],
        )
        public = record(
            "public-1",
            "public_diagnostic",
            [
                {"role": "user", "content": "异火是什么？"},
                {"role": "assistant", "content": "异火是一类特殊火焰。"},
            ],
            evaluation={"metric": "required_terms", "required_any": ["异火", "火焰"]},
        )
        train_path = Path("/tmp/sft-v7/train.jsonl")
        val_path = Path("/tmp/sft-v7/val.jsonl")
        public_path = Path("/tmp/sft-v7/public_diagnostic.jsonl")
        train_payload, public_payload, report = prepare_payloads(
            [train],
            [val],
            [public],
            self.tokenizer,
            self.special_ids,
            tokenizer_path=FORMAL_TOKENIZER,
            tokenizer_sha256=EXPECTED_TOKENIZER_SHA256,
            token_manifest_path=FORMAL_MANIFEST,
            token_manifest_sha256=EXPECTED_MANIFEST_SHA256,
            sft_dataset_manifest_sha256="4" * 64,
            train_path=train_path,
            train_sha256="1" * 64,
            val_path=val_path,
            val_sha256="2" * 64,
            public_path=public_path,
            public_sha256="3" * 64,
        )

        self.assertEqual(train_payload["schema_version"], "sft-v7-train-val-tensors/v1")
        self.assertEqual(public_payload["schema_version"], "sft-v7-public-tensors/v1")
        self.assertEqual(set(train_payload["source_jsonl_paths"]), {"train", "val"})
        self.assertEqual(set(public_payload["source_jsonl_paths"]), {"public_diagnostic"})
        self.assertNotIn("public_records", train_payload)
        self.assertNotIn("train_records", public_payload)
        self.assertNotIn("val_records", public_payload)
        self.assertEqual(public_payload["public_records"][0]["evaluation"]["metric"], "required_terms")
        self.assertEqual(train_payload["required_base_checkpoint"]["step"], 5750)
        self.assertEqual(
            train_payload["required_base_checkpoint"]["token_manifest_sha256"],
            EXPECTED_MANIFEST_SHA256,
        )
        self.assertEqual(report["split_counts"], {"train": 1, "val": 1, "public_diagnostic": 1})
        assert_no_forbidden_scope(self, train_payload)
        assert_no_forbidden_scope(self, public_payload)
        assert_no_forbidden_scope(self, report)

    def test_cli_writes_independent_artifacts_and_structured_success_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path, val_path, public_path = self._write_three_inputs(root)
            train_output = root / "train_val_tensors.pt"
            public_output = root / "public_diagnostic_tensors.pt"
            report_path = root / "tensor_report.json"
            log_dir = root / "logs"
            arguments = self._main_arguments(
                train_path,
                val_path,
                public_path,
                train_output,
                public_output,
                report_path,
                log_dir,
            )
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with mock.patch(
                    "prepare_bpe_sft_v7.load_and_validate_dataset_manifest",
                    return_value={"sft_dataset_manifest_sha256": "4" * 64},
                ):
                    result = main(arguments)

            self.assertEqual(result, 0)
            train_payload = torch.load(train_output, map_location="cpu", weights_only=False)
            public_payload = torch.load(public_output, map_location="cpu", weights_only=False)
            self.assertEqual(train_payload["schema_version"], "sft-v7-train-val-tensors/v1")
            self.assertEqual(public_payload["schema_version"], "sft-v7-public-tensors/v1")
            self.assertTrue(Path(f"{train_output}.sha256").is_file())
            self.assertTrue(Path(f"{public_output}.sha256").is_file())
            self.assertEqual(
                Path(f"{train_output}.sha256").read_text().split()[0],
                file_sha256(train_output),
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["train_val_output_sha256"], file_sha256(train_output))
            self.assertEqual(report["public_output_sha256"], file_sha256(public_output))
            assert_no_forbidden_scope(self, train_payload)
            assert_no_forbidden_scope(self, public_payload)

            log_text = "".join(path.read_text(encoding="utf-8") for path in log_dir.glob("*.jsonl"))
            events = [json.loads(line) for line in log_text.splitlines() if line.strip()]
            self.assertTrue(events)
            self.assertTrue(all(event["run_id"].startswith("sft-v7-tensor-") for event in events))
            self.assertIn("cloud.data", {event["module"] for event in events})
            self.assertIn("cloud.validation", {event["module"] for event in events})
            self.assertIn("cloud.sft", {event["module"] for event in events})
            self.assertIn("cloud.orchestrator", {event["module"] for event in events})
            self.assertNotIn("萧炎是谁", log_text)
            self.assertNotIn("药老即药尘", log_text)

    def test_cli_failure_log_has_code_but_not_record_body(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path, val_path, public_path = self._write_three_inputs(root)
            broken = record(
                "broken-1",
                "train",
                [
                    {"role": "assistant", "content": "绝密正文萧炎。"},
                    {"role": "user", "content": "绝密正文是什么？"},
                ],
            )
            train_path.write_text(json.dumps(broken, ensure_ascii=False) + "\n", encoding="utf-8")
            log_dir = root / "logs"
            arguments = self._main_arguments(
                train_path,
                val_path,
                public_path,
                root / "train_val_tensors.pt",
                root / "public_diagnostic_tensors.pt",
                root / "tensor_report.json",
                log_dir,
            )
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SFTV7EncodingError):
                    with mock.patch(
                        "prepare_bpe_sft_v7.load_and_validate_dataset_manifest",
                        return_value={"sft_dataset_manifest_sha256": "4" * 64},
                    ):
                        main(arguments)

            validation_logs = list(log_dir.glob("*.validation.jsonl"))
            self.assertEqual(len(validation_logs), 1)
            log_text = validation_logs[0].read_text(encoding="utf-8")
            self.assertIn("invalid_role_order", log_text)
            self.assertIn("SFTV7EncodingError", log_text)
            self.assertNotIn("绝密正文", log_text)

    def test_cli_has_no_blind_split_argument_and_rejects_wrong_input_name(self):
        from prepare_bpe_sft_v7 import parse_args, validate_paths

        parsed = parse_args([])
        self.assertNotIn("sealed", vars(parsed))
        self.assertNotIn("test", vars(parsed))
        parsed.train = Path("wrong.jsonl")
        with self.assertRaises(SFTV7EncodingError) as caught:
            validate_paths(parsed)
        self.assertEqual(caught.exception.code, "unexpected_input_name")

    @staticmethod
    def _write_three_inputs(root: Path) -> tuple[Path, Path, Path]:
        train_path = root / "train.jsonl"
        val_path = root / "val.jsonl"
        public_path = root / "public_diagnostic.jsonl"
        train_value = record(
            "train-main-1",
            "train",
            [
                {"role": "user", "content": "萧炎是谁？"},
                {"role": "assistant", "content": "萧炎是主要人物。"},
            ],
        )
        val_value = record(
            "val-main-1",
            "val",
            [
                {"role": "user", "content": "药老是谁？"},
                {"role": "assistant", "content": "药老即药尘。"},
            ],
        )
        public_value = record(
            "public-main-1",
            "public_diagnostic",
            [
                {"role": "user", "content": "异火是什么？"},
                {"role": "assistant", "content": "异火是一类特殊火焰。"},
            ],
            evaluation={"metric": "required_terms", "required_any": ["异火", "火焰"]},
        )
        for path, value in (
            (train_path, train_value),
            (val_path, val_value),
            (public_path, public_value),
        ):
            path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
        return train_path, val_path, public_path

    @staticmethod
    def _main_arguments(
        train_path: Path,
        val_path: Path,
        public_path: Path,
        train_output: Path,
        public_output: Path,
        report_path: Path,
        log_dir: Path,
    ) -> list[str]:
        return [
            "--train",
            str(train_path),
            "--val",
            str(val_path),
            "--public-diagnostic",
            str(public_path),
            "--tokenizer",
            str(FORMAL_TOKENIZER),
            "--token-manifest",
            str(FORMAL_MANIFEST),
            "--train-val-output",
            str(train_output),
            "--public-output",
            str(public_output),
            "--report",
            str(report_path),
            "--log-dir",
            str(log_dir),
        ]


if __name__ == "__main__":
    unittest.main()
