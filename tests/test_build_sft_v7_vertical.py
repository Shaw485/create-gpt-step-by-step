from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from build_sft_v7_vertical import (
    CALIBRATION_TRIPLETS,
    DIRECT_CORE_SPLIT_QUOTAS,
    DIMENSION_SPLIT_QUOTAS,
    DIMENSION_TOTALS,
    KNOWN_CORE_FACTS,
    OUTPUT_NAMES,
    SCHEMA_VERSION,
    SPLITS,
    SPLIT_TOTALS,
    _portable_artifact_path,
    build_release,
    main,
)
from training_runtime import file_sha256


REPOSITORY = Path(__file__).resolve().parents[1]
FORMAL_CORPUS = REPOSITORY / "data/cloud_v4/train.txt"
FORMAL_TOKENIZER = REPOSITORY / "data/scaling_a/bpe_3000/tokenizer.json"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _all_mapping_keys(value) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_all_mapping_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_all_mapping_keys(item))
    return keys


class SFTV7VerticalBuildIntegrationTests(unittest.TestCase):
    """One real build plus one independent rebuild covers the frozen release."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="sft-v7-builder-test-")
        cls.temp_root = Path(cls.temporary.name)
        cls.output_one = cls.temp_root / "release-one"
        cls.output_two = cls.temp_root / "release-two"
        cls.log_dir = cls.temp_root / "logs"
        exit_code = main(
            [
                "--corpus",
                str(FORMAL_CORPUS),
                "--tokenizer",
                str(FORMAL_TOKENIZER),
                "--output-dir",
                str(cls.output_one),
                "--log-dir",
                str(cls.log_dir),
                "--log-max-bytes",
                "8192",
                "--log-backups",
                "2",
                "--no-console-log",
            ]
        )
        if exit_code != 0:
            diagnostics = "\n".join(
                path.read_text(encoding="utf-8")
                for path in cls.log_dir.glob("*.orchestrator.jsonl")
            )
            raise RuntimeError(f"real v7 integration build failed: {diagnostics}")
        cls.manifest_one = json.loads(
            (cls.output_one / "manifest.json").read_text(encoding="utf-8")
        )
        cls.manifest_two, cls.summary_two = build_release(
            corpus_path=FORMAL_CORPUS,
            tokenizer_path=FORMAL_TOKENIZER,
            output_dir=cls.output_two,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_exact_quotas_schema_and_quality_aggregates(self) -> None:
        manifest = self.manifest_one
        self.assertEqual(manifest["record_schema_version"], SCHEMA_VERSION)
        self.assertEqual(manifest["record_count"], 10_000)
        self.assertEqual(manifest["split_totals"], SPLIT_TOTALS)
        self.assertEqual(manifest["dimension_totals"], DIMENSION_TOTALS)
        self.assertEqual(
            manifest["dimension_split_quotas"],
            DIMENSION_SPLIT_QUOTAS,
        )
        self.assertEqual(
            {split: manifest["split_files"][split]["count"] for split in SPLITS},
            SPLIT_TOTALS,
        )
        self.assertEqual(manifest["known_core"]["reviewed_fact_count"], 18)
        self.assertEqual(manifest["known_core"]["exposure_total"], 900)
        self.assertEqual(
            manifest["known_core"]["split_exposures"],
            DIRECT_CORE_SPLIT_QUOTAS,
        )

        summary = self.summary_two
        self.assertEqual(summary["record_count"], 10_000)
        self.assertEqual(summary["core_coverage"], 50)
        self.assertEqual(summary["multiturn_records"], 1_200)
        self.assertEqual(summary["rag_records"], 1_400)
        self.assertEqual(
            summary["complete_calibration_triplets"],
            sum(CALIBRATION_TRIPLETS.values()),
        )
        self.assertEqual(summary["known_core_false_refusals"], 0)
        self.assertGreaterEqual(summary["medium_answer_share"], 0.50)
        self.assertGreaterEqual(summary["long_answer_share"], 0.10)
        self.assertLessEqual(summary["largest_fixed_12_phrase_share"], 0.02)
        self.assertTrue(
            all(0.15 <= share <= 0.20 for share in summary["negative_shares"].values())
        )

    def test_full_rebuild_is_byte_deterministic(self) -> None:
        self.assertEqual(
            self.manifest_one["dataset_identity_sha256"],
            self.manifest_two["dataset_identity_sha256"],
        )
        for split in SPLITS:
            first = self.manifest_one["split_files"][split]
            second = self.manifest_two["split_files"][split]
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(first["count"], second["count"])
            self.assertEqual(
                file_sha256(self.output_one / OUTPUT_NAMES[split]),
                first["sha256"],
            )
        self.assertEqual(
            file_sha256(self.output_one / "manifest.json"),
            file_sha256(self.output_two / "manifest.json"),
        )

    def test_manifest_paths_are_portable_and_do_not_leak_home_directories(self) -> None:
        manifest_text = json.dumps(self.manifest_one, ensure_ascii=False)
        self.assertNotIn("/Users/", manifest_text)
        self.assertNotIn("\\Users\\", manifest_text)
        self.assertEqual(
            self.manifest_one["source"]["path"],
            "data/cloud_v4/train.txt",
        )
        self.assertEqual(
            self.manifest_one["tokenizer"]["path"],
            "data/scaling_a/bpe_3000/tokenizer.json",
        )
        for metadata in self.manifest_one["split_files"].values():
            self.assertFalse(Path(metadata["path"]).is_absolute())

    def test_public_known_core_questions_answer_directly_with_exact_evidence(self) -> None:
        public_records = _read_jsonl(
            self.output_one / OUTPUT_NAMES["public_diagnostic"]
        )
        by_acceptance_id = {
            record["evaluation"]["acceptance_case_id"]: record
            for record in public_records
            if record["evaluation"].get("acceptance_case_id")
        }
        expected = {
            fact.acceptance_case_id: fact
            for fact in KNOWN_CORE_FACTS
            if fact.acceptance_case_id
        }
        self.assertEqual(set(by_acceptance_id), set(expected))
        corpus_lines = FORMAL_CORPUS.read_text(encoding="utf-8").splitlines()
        refusal_markers = ("资料不足", "无法确定", "不能确认", "需要检索")
        for acceptance_id, fact in expected.items():
            record = by_acceptance_id[acceptance_id]
            self.assertEqual(record["question"], fact.canonical_question)
            self.assertTrue(record["evaluation"]["known_fact"])
            self.assertTrue(record["evaluation"]["evidence_sufficient"])
            self.assertFalse(any(marker in record["answer"] for marker in refusal_markers))
            self.assertTrue(
                all(term in record["answer"] for term in fact.required_terms)
            )
            self.assertEqual(
                [item["start_line"] for item in record["evidence"]],
                list(fact.evidence_lines),
            )
            for item in record["evidence"]:
                source_text = corpus_lines[item["start_line"] - 1]
                self.assertEqual(item["text"], source_text)
                self.assertEqual(
                    item["sha256"],
                    hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
                )

    def test_module_logs_are_aggregate_only_and_contain_no_record_body(self) -> None:
        log_paths = list(self.log_dir.glob("*.jsonl"))
        module_names = {path.name.rsplit(".", 2)[-2] for path in log_paths}
        self.assertTrue({"data", "validation", "sft", "orchestrator"}.issubset(module_names))
        forbidden_keys = {
            "question",
            "answer",
            "messages",
            "text",
            "input_ids",
            "labels",
            "token_ids",
        }
        combined_log = ""
        for path in log_paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                payload = json.loads(line)
                self.assertTrue(forbidden_keys.isdisjoint(_all_mapping_keys(payload)))
                combined_log += line
        self.assertNotIn("萧炎是谁？", combined_log)
        self.assertNotIn("萧家现任族长", combined_log)
        self.assertNotIn("<ASSISTANT>", combined_log)
        self.assertNotIn("Bearer integration-secret", combined_log)


class SFTV7VerticalBuildFailureTests(unittest.TestCase):
    def test_missing_input_logs_actionable_failure_without_body_or_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sft-v7-failure-") as directory:
            root = Path(directory)
            output_dir = root / "release"
            log_dir = root / "logs"
            exit_code = main(
                [
                    "--corpus",
                    str(root / "missing-corpus.txt"),
                    "--tokenizer",
                    str(FORMAL_TOKENIZER),
                    "--output-dir",
                    str(output_dir),
                    "--log-dir",
                    str(log_dir),
                    "--no-console-log",
                ]
            )
            self.assertEqual(exit_code, 2)
            self.assertFalse((output_dir / "manifest.json").exists())
            events = [
                json.loads(line)
                for path in log_dir.glob("*.orchestrator.jsonl")
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len(events), 1)
            context = events[0]["context"]
            self.assertEqual(context["error_code"], "INPUT_LOAD_FAILURE")
            self.assertIn("Verify both paths", context["remediation"])
            self.assertTrue(
                {"question", "answer", "messages", "text"}.isdisjoint(
                    _all_mapping_keys(events[0])
                )
            )


class SFTV7ManifestPathUnitTests(unittest.TestCase):
    def test_published_manifest_contains_no_host_absolute_paths(self) -> None:
        manifest_path = REPOSITORY / "data/sft/v7/manifest.json"
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
        self.assertNotIn("/Users/", manifest_text)
        self.assertNotIn("\\Users\\", manifest_text)
        self.assertEqual(manifest["source"]["path"], "data/cloud_v4/train.txt")
        self.assertEqual(
            manifest["tokenizer"]["path"],
            "data/scaling_a/bpe_3000/tokenizer.json",
        )
        self.assertTrue(
            all(
                not Path(metadata["path"]).is_absolute()
                for metadata in manifest["split_files"].values()
            )
        )

    def test_repository_input_becomes_repository_relative(self) -> None:
        self.assertEqual(
            _portable_artifact_path(FORMAL_CORPUS, role="source"),
            "data/cloud_v4/train.txt",
        )

    def test_external_input_becomes_logical_uri_without_host_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sft-v7-external-input-") as directory:
            external = Path(directory) / "train.txt"
            portable = _portable_artifact_path(external, role="source")
        self.assertEqual(portable, "artifact://source/train.txt")
        self.assertNotIn("/Users/", portable)
        self.assertFalse(Path(portable).is_absolute())


if __name__ == "__main__":
    unittest.main()
