from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from build_sft_v7_1_canary import (
    BASE_CHECKPOINT_BINDING,
    DEFAULT_BASE_CHECKPOINT,
    DEFAULT_TOKENIZER,
    DEFAULT_TOKEN_MANIFEST,
    FORBIDDEN_TERMS,
    HOLDOUT_ROLES,
    MANIFEST_SCHEMA,
    META_ANSWER_PREFIXES,
    OUTPUT_NAMES,
    PRIMARY_DIMENSION,
    PARENT_MANIFEST_BINDING,
    RECORD_SCHEMA,
    TASK_FAMILY,
    TOKEN_MANIFEST_BINDING,
    TOKENIZER_BINDING,
    TRAIN_ROLES,
    build_release,
    file_sha256,
    _validate_frozen_training_inputs,
    _validate_parent_manifest,
    CanaryBuildError,
    main,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "configs/sft_v7_1_canary_facts.json"
PARENT_MANIFEST = REPOSITORY / "data/sft/v7/manifest.json"
EXPECTED_CANARY_MANIFEST_SHA256 = (
    "68908fdabe4f8ae470f6bcd4ec6d11b59304829119835af901df7bf9888ef50d"
)
EXPECTED_TRAIN_SHA256 = (
    "e5f0f90b26f9dbacb68017bbaa4243a41ccdacb4ac60484677964061fd4d008a"
)
EXPECTED_HOLDOUT_SHA256 = (
    "fe8a72efcd8e3f179d61ca8e4b2de2b3c775dbd9841d25692fe6743f3548c64d"
)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _all_keys(value) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_all_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_all_keys(item))
    return keys


class SFTV71CanaryBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="sft-v7-1-canary-test-")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build(self, suffix: str = "one"):
        output = self.root / suffix / "data"
        report_json = self.root / suffix / "report.json"
        report_md = self.root / suffix / "report.md"
        manifest, report = build_release(
            config_path=CONFIG,
            parent_manifest_path=PARENT_MANIFEST,
            output_dir=output,
            report_json_path=report_json,
            report_md_path=report_md,
        )
        return output, report_json, report_md, manifest, report

    def test_builds_exact_64_train_and_16_holdout_records(self) -> None:
        output, _, _, manifest, report = self._build()
        train = _read_jsonl(output / OUTPUT_NAMES["train"])
        holdout = _read_jsonl(output / OUTPUT_NAMES["holdout_eval"])
        self.assertEqual(len(train), 64)
        self.assertEqual(len(holdout), 16)
        self.assertEqual(manifest["manifest_schema_version"], MANIFEST_SCHEMA)
        self.assertEqual(manifest["record_schema_version"], RECORD_SCHEMA)
        self.assertEqual(manifest["split_totals"], {"train": 64, "holdout_eval": 16})
        self.assertEqual(report["status"], "pass")
        self.assertTrue(all(report["quality_gates"].values()))
        self.assertEqual(
            report["lineage_verification"]["parent_manifest"]["sha256"],
            PARENT_MANIFEST_BINDING["sha256"],
        )
        self.assertEqual(
            report["lineage_verification"]["base_checkpoint"]["sha256"],
            BASE_CHECKPOINT_BINDING["sha256"],
        )
        self.assertEqual(
            report["lineage_verification"]["tokenizer"]["sha256"],
            TOKENIZER_BINDING["sha256"],
        )
        self.assertEqual(
            report["lineage_verification"]["token_manifest"]["sha256"],
            TOKEN_MANIFEST_BINDING["sha256"],
        )

        train_by_fact = Counter(record["fact_id"] for record in train)
        holdout_by_fact = Counter(record["fact_id"] for record in holdout)
        self.assertEqual(len(train_by_fact), 8)
        self.assertTrue(all(count == 8 for count in train_by_fact.values()))
        self.assertEqual(set(holdout_by_fact), set(train_by_fact))
        self.assertTrue(all(count == 2 for count in holdout_by_fact.values()))

    def test_records_have_clear_roles_and_direct_required_term_answers(self) -> None:
        output, _, _, _, _ = self._build()
        train = _read_jsonl(output / OUTPUT_NAMES["train"])
        holdout = _read_jsonl(output / OUTPUT_NAMES["holdout_eval"])
        self.assertEqual(Counter(record["prompt_role"] for record in train), Counter({role: 8 for role in TRAIN_ROLES}))
        self.assertEqual(Counter(record["prompt_role"] for record in holdout), Counter({role: 8 for role in HOLDOUT_ROLES}))
        for split, records, should_train in (
            ("train", train, True),
            ("holdout_eval", holdout, False),
        ):
            for record in records:
                self.assertEqual(record["schema_version"], RECORD_SCHEMA)
                self.assertEqual(record["split"], split)
                self.assertEqual(record["primary_dimension"], PRIMARY_DIMENSION)
                self.assertEqual(record["task_family"], TASK_FAMILY)
                self.assertIs(record["supervision"]["use_for_training"], should_train)
                self.assertTrue(record["supervision"]["assistant_only_loss"])
                self.assertTrue(record["supervision"]["eos_appended_by_encoder"])
                self.assertNotIn("<EOS>", record["question"])
                self.assertNotIn("<EOS>", record["answer"])
                self.assertFalse(record["answer"].startswith(META_ANSWER_PREFIXES))
                self.assertTrue(
                    all(
                        term in record["answer"]
                        for term in record["evaluation"]["required_terms"]
                    )
                )
                self.assertFalse(
                    any(term in record["answer"] for term in FORBIDDEN_TERMS)
                )
                self.assertEqual(record["source"]["catalog_fact_id"], record["fact_id"])
                self.assertFalse(record["source"]["contains_evidence_body"])
                self.assertEqual(
                    record["messages"],
                    [
                        {"role": "user", "content": record["question"]},
                        {"role": "assistant", "content": record["answer"]},
                    ],
                )

    def test_questions_are_unique_and_holdout_wording_is_unseen(self) -> None:
        output, _, _, _, _ = self._build()
        train = _read_jsonl(output / OUTPUT_NAMES["train"])
        holdout = _read_jsonl(output / OUTPUT_NAMES["holdout_eval"])
        train_questions = {record["question"] for record in train}
        holdout_questions = {record["question"] for record in holdout}
        self.assertEqual(len(train_questions), 64)
        self.assertEqual(len(holdout_questions), 16)
        self.assertFalse(train_questions.intersection(holdout_questions))
        self.assertEqual(len({record["id"] for record in (*train, *holdout)}), 80)

    def test_manifest_binds_sources_model_tokenizer_and_artifact_hashes(self) -> None:
        output, _, _, manifest, _ = self._build()
        manifest_text = json.dumps(manifest, ensure_ascii=False)
        self.assertNotIn("/Users/", manifest_text)
        self.assertNotIn("\\Users\\", manifest_text)
        self.assertEqual(manifest["source"]["parent_manifest_path"], "data/sft/v7/manifest.json")
        self.assertEqual(manifest["training_binding"]["base_checkpoint"], BASE_CHECKPOINT_BINDING)
        self.assertEqual(manifest["training_binding"]["tokenizer"], TOKENIZER_BINDING)
        self.assertEqual(
            file_sha256(output / "manifest.json"),
            EXPECTED_CANARY_MANIFEST_SHA256,
        )
        self.assertEqual(file_sha256(output / "train.jsonl"), EXPECTED_TRAIN_SHA256)
        self.assertEqual(
            file_sha256(output / "holdout_eval.jsonl"), EXPECTED_HOLDOUT_SHA256
        )
        for split, filename in OUTPUT_NAMES.items():
            metadata = manifest["split_files"][split]
            self.assertEqual(metadata["path"], filename)
            self.assertEqual(metadata["sha256"], file_sha256(output / filename))
        self.assertEqual(
            manifest["access_audit"],
            {
                "v7_train_body_read": False,
                "v7_public_body_read": False,
                "v7_sealed_body_read": False,
                "formal_corpus_body_read": False,
            },
        )

    def test_rebuild_is_byte_deterministic(self) -> None:
        output_one, report_one, md_one, manifest_one, _ = self._build("one")
        output_two, report_two, md_two, manifest_two, _ = self._build("two")
        self.assertEqual(manifest_one, manifest_two)
        for filename in (*OUTPUT_NAMES.values(), "manifest.json"):
            self.assertEqual(
                file_sha256(output_one / filename),
                file_sha256(output_two / filename),
            )
        self.assertEqual(file_sha256(report_one), file_sha256(report_two))
        self.assertEqual(file_sha256(md_one), file_sha256(md_two))

    def test_rejects_fake_parent_manifest_sha_and_dataset_identity(self) -> None:
        parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
        with self.assertRaises(CanaryBuildError) as sha_error:
            _validate_parent_manifest(parent, actual_sha256="0" * 64)
        self.assertEqual(
            sha_error.exception.code, "PARENT_MANIFEST_SHA256_MISMATCH"
        )

        parent["dataset_identity_sha256"] = "1" * 64
        with self.assertRaises(CanaryBuildError) as identity_error:
            _validate_parent_manifest(
                parent,
                actual_sha256=PARENT_MANIFEST_BINDING["sha256"],
            )
        self.assertEqual(
            identity_error.exception.code, "PARENT_DATASET_IDENTITY_MISMATCH"
        )

        parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
        parent["tokenizer"]["context_limit"] = 256
        with self.assertRaises(CanaryBuildError) as context_error:
            _validate_parent_manifest(
                parent,
                actual_sha256=PARENT_MANIFEST_BINDING["sha256"],
            )
        self.assertEqual(
            context_error.exception.code, "PARENT_TOKENIZER_BINDING_MISMATCH"
        )

    def test_rejects_fake_base_tokenizer_and_token_manifest_bytes(self) -> None:
        real_file_sha256 = file_sha256
        cases = (
            (
                (REPOSITORY / DEFAULT_BASE_CHECKPOINT).resolve(),
                "BASE_CHECKPOINT_SHA256_MISMATCH",
            ),
            (
                (REPOSITORY / DEFAULT_TOKENIZER).resolve(),
                "TOKENIZER_SHA256_MISMATCH",
            ),
            (
                (REPOSITORY / DEFAULT_TOKEN_MANIFEST).resolve(),
                "TOKEN_MANIFEST_SHA256_MISMATCH",
            ),
        )
        for drifted_path, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                def drift_one(path, *, _target=drifted_path):
                    if Path(path).resolve() == _target:
                        return "f" * 64
                    return real_file_sha256(path)

                with mock.patch(
                    "build_sft_v7_1_canary.file_sha256", side_effect=drift_one
                ):
                    with self.assertRaises(CanaryBuildError) as caught:
                        _validate_frozen_training_inputs(
                            base_checkpoint_path=DEFAULT_BASE_CHECKPOINT,
                            tokenizer_path=DEFAULT_TOKENIZER,
                            token_manifest_path=DEFAULT_TOKEN_MANIFEST,
                        )
                self.assertEqual(caught.exception.code, expected_code)

    def test_rejects_external_lookalike_frozen_file_path(self) -> None:
        fake_base = self.root / "step_05750.pt"
        fake_base.write_bytes(b"not the frozen checkpoint")
        with self.assertRaises(CanaryBuildError) as caught:
            _validate_frozen_training_inputs(
                base_checkpoint_path=fake_base,
                tokenizer_path=DEFAULT_TOKENIZER,
                token_manifest_path=DEFAULT_TOKEN_MANIFEST,
            )
        self.assertEqual(caught.exception.code, "BASE_CHECKPOINT_PATH_MISMATCH")

    def test_builder_never_reads_v7_jsonl_bodies(self) -> None:
        original_read_text = Path.read_text

        def guarded_read_text(path, *args, **kwargs):
            value = Path(path)
            if value.suffix == ".jsonl" and value.parent.name == "v7":
                self.fail(f"builder attempted to read forbidden body: {value.name}")
            return original_read_text(value, *args, **kwargs)

        with mock.patch.object(Path, "read_text", guarded_read_text):
            self._build()

    def test_success_and_failure_logs_are_modular_aggregate_only(self) -> None:
        success_root = self.root / "success"
        exit_code = main(
            [
                "--config",
                str(CONFIG),
                "--parent-manifest",
                str(PARENT_MANIFEST),
                "--output-dir",
                str(success_root / "data"),
                "--report-json",
                str(success_root / "report.json"),
                "--report-md",
                str(success_root / "report.md"),
                "--log-dir",
                str(success_root / "logs"),
                "--log-max-bytes",
                "4096",
                "--log-backups",
                "1",
                "--no-console-log",
            ]
        )
        self.assertEqual(exit_code, 0)
        log_paths = list((success_root / "logs").glob("*.jsonl"))
        logged_modules = {path.name.rsplit(".", 2)[-2] for path in log_paths}
        self.assertTrue({"data", "validation", "orchestrator"}.issubset(logged_modules))
        forbidden_log_keys = {
            "answer",
            "content",
            "messages",
            "question",
            "text",
            "evidence_line_numbers",
        }
        combined = ""
        for path in log_paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                payload = json.loads(line)
                self.assertFalse(_all_keys(payload).intersection(forbidden_log_keys))
                self.assertIn("timestamp", payload)
                self.assertIn("run_id", payload)
                combined += line
        self.assertNotIn("萧炎是谁", combined)
        self.assertNotIn("药老是萧炎", combined)

        invalid_config = self.root / "invalid_config.json"
        config_payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        config_payload["facts"][0]["expected_required_terms"] = ["错误词"]
        invalid_config.write_text(
            json.dumps(config_payload, ensure_ascii=False), encoding="utf-8"
        )
        failure_root = self.root / "failure"
        failure_code = main(
            [
                "--config",
                str(invalid_config),
                "--parent-manifest",
                str(PARENT_MANIFEST),
                "--output-dir",
                str(failure_root / "data"),
                "--report-json",
                str(failure_root / "report.json"),
                "--report-md",
                str(failure_root / "report.md"),
                "--log-dir",
                str(failure_root / "logs"),
                "--no-console-log",
            ]
        )
        self.assertEqual(failure_code, 1)
        failure_logs = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (failure_root / "logs").glob("*.orchestrator.jsonl")
        )
        self.assertIn("CANARY_CONFIG_PATH_MISMATCH", failure_logs)
        self.assertIn("remediation", failure_logs)
        self.assertIn("CanaryBuildError", failure_logs)
        self.assertNotIn("Traceback", failure_logs)
        self.assertNotIn(str(invalid_config), failure_logs)
        self.assertNotIn("萧炎是谁", failure_logs)

        unexpected_root = self.root / "unexpected"
        private_detail = "do not leak /Users/example/private.json"
        with mock.patch(
            "build_sft_v7_1_canary.build_release",
            side_effect=RuntimeError(private_detail),
        ):
            unexpected_code = main(
                [
                    "--log-dir",
                    str(unexpected_root / "logs"),
                    "--no-console-log",
                ]
            )
        self.assertEqual(unexpected_code, 1)
        unexpected_logs = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (unexpected_root / "logs").glob("*.orchestrator.jsonl")
        )
        self.assertIn("UNEXPECTED_BUILD_FAILURE", unexpected_logs)
        self.assertIn("RuntimeError", unexpected_logs)
        self.assertIn("remediation", unexpected_logs)
        self.assertNotIn(private_detail, unexpected_logs)
        self.assertNotIn("/Users/", unexpected_logs)
        self.assertNotIn("Traceback", unexpected_logs)


if __name__ == "__main__":
    unittest.main()
