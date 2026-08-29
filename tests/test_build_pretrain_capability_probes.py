import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from build_pretrain_capability_probes import (
    SCHEMA_VERSION,
    main,
    validate_probe_artifact,
)
from training_runtime import canonical_json_sha256


NAMES = ("萧炎", "药老", "海波东", "苏千", "云山", "雅妃", "古河", "萧鼎")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_split(chapter_number: int, split_label: str) -> str:
    lines = ["------------", "", f"第{chapter_number}章 {split_label}试炼", ""]
    for index, name in enumerate(NAMES):
        lines.append(
            "    石室里的火光缓慢摇动，众人已经在长桌旁安静等待了许久，"
            f"{name}笑道：“先检查眼前留下的痕迹，再决定接下来往哪条山路走。”"
            f"说完以后，第{index + 1}盏灯也随风轻轻晃了一下，映出墙壁上的细密纹路和旧日刻痕。"
            "远处的脚步声逐渐停下，石门之外重新恢复了长久的寂静。"
        )
        lines.append("")
    return "\n".join(lines) + "\n"


class PretrainCapabilityProbeBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.train_path = self.root / "train.txt"
        self.val_path = self.root / "val.txt"
        self.train_text = make_split(1, "训练")
        self.val_text = make_split(2, "验证")
        self.train_path.write_text(self.train_text, encoding="utf-8")
        self.val_path.write_text(self.val_text, encoding="utf-8")
        # The test path intentionally does not exist.  A successful build proves
        # that the sealed split was neither opened nor hashed.
        self.test_path = self.root / "sealed-test-must-not-be-read.txt"
        self.manifest_path = self.root / "corpus_manifest.json"
        manifest = {
            "schema_version": "1.0",
            "status": "ready",
            "splits": {
                "train": {
                    "path": str(self.train_path),
                    "sha256": sha256_text(self.train_text),
                },
                "val": {
                    "path": str(self.val_path),
                    "sha256": sha256_text(self.val_text),
                },
                "test": {
                    "path": str(self.test_path),
                    "sha256": "f" * 64,
                },
            },
        }
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False),
            encoding="utf-8",
        )
        self.output_path = self.root / "probes.json"
        self.prompts_path = self.root / "prompts.txt"
        self.log_dir = self.root / "logs"

    def tearDown(self):
        self.temporary.cleanup()

    def run_builder(self, *, output_path=None, prompts_path=None):
        return main(
            [
                "--manifest",
                str(self.manifest_path),
                "--output",
                str(output_path or self.output_path),
                "--prompts-output",
                str(prompts_path or self.prompts_path),
                "--seed",
                "17",
                "--continuation-per-split",
                "1",
                "--cloze-per-tier-per-split",
                "1",
                "--candidate-count",
                "3",
                "--minimum-entity-occurrences",
                "1",
                "--minimum-entity-attributions",
                "1",
                "--run-id",
                "probe-test-run",
                "--log-dir",
                str(self.log_dir),
                "--no-console-log",
            ]
        )

    def test_build_is_corpus_only_reproducible_and_keeps_test_sealed(self):
        self.assertEqual(self.run_builder(), 0)
        artifact = json.loads(self.output_path.read_text(encoding="utf-8"))

        self.assertEqual(artifact["schema_version"], SCHEMA_VERSION)
        self.assertFalse(artifact["usage"]["training_allowed"])
        self.assertFalse(artifact["usage"]["sft_information_used"])
        self.assertFalse(artifact["usage"]["test_split_read"])
        self.assertEqual(set(artifact["build"]["inputs"]), {"train", "val"})
        self.assertFalse(artifact["build"]["sealed_inputs"]["test"]["read"])
        self.assertFalse(self.test_path.exists())
        self.assertTrue(artifact["validation"]["passed"])
        self.assertEqual(len(artifact["probes"]), 6)
        self.assertEqual(
            {probe["probe_type"] for probe in artifact["probes"]},
            {"held_out_continuation", "cloze_candidate_ranking"},
        )

        source_texts = {"train": self.train_text, "val": self.val_text}
        for probe in artifact["probes"]:
            evidence = probe["evidence"]
            exact = source_texts[probe["source"]["split"]][
                evidence["split_char_start"] : evidence["split_char_end"]
            ]
            self.assertEqual(exact, evidence["text"])
            self.assertEqual(sha256_text(exact), evidence["sha256"])
            if probe["probe_type"] == "held_out_continuation":
                answer = probe["expected"]["continuation"]
            else:
                answer = probe["expected"]["text"]
                self.assertEqual(probe["prompt"] + answer, exact)
                self.assertNotIn(answer, probe["prompt"])
                self.assertEqual(
                    {candidate["entity_type"] for candidate in probe["candidates"]},
                    {"person_speaker"},
                )
            self.assertNotIn(answer, probe["prompt"])

        second_output = self.root / "probes-second.json"
        second_prompts = self.root / "prompts-second.txt"
        self.assertEqual(
            self.run_builder(output_path=second_output, prompts_path=second_prompts),
            0,
        )
        second = json.loads(second_output.read_text(encoding="utf-8"))
        self.assertEqual(artifact["probes"], second["probes"])
        self.assertEqual(artifact["cases"], second["cases"])
        self.assertEqual(
            self.prompts_path.read_text(encoding="utf-8"),
            second_prompts.read_text(encoding="utf-8"),
        )

    def test_outputs_are_directly_compatible_with_evaluator_contracts(self):
        self.assertEqual(self.run_builder(), 0)
        artifact = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(len(artifact["cases"]), 2)
        for case in artifact["cases"]:
            self.assertEqual(
                set(case),
                {"id", "context", "candidates", "correct", "metadata"},
            )
            self.assertTrue(case["context"])
            self.assertGreaterEqual(len(case["candidates"]), 2)
            self.assertEqual(len(case["candidates"]), len(set(case["candidates"])))
            self.assertIn(case["correct"], case["candidates"])
            self.assertNotIn(case["correct"], case["context"])
            metadata = case["metadata"]
            self.assertEqual(metadata["source_split"], "val")
            self.assertEqual(metadata["source"]["split"], "val")
            self.assertIn(metadata["frequency_tier"], {"high", "low"})
            self.assertIsInstance(metadata["train_count"], int)
            self.assertGreater(metadata["train_count"], 0)
            self.assertRegex(metadata["chapter_sha256"], r"^[0-9a-f]{64}$")
            evidence = metadata["evidence"]
            self.assertEqual(sha256_text(evidence["text"]), evidence["sha256"])
            matching = metadata["distractor_matching"]
            self.assertFalse(matching["tokenizer_used_for_matching"])
            self.assertIn("evaluator", matching["token_length_correction"].lower())
        self.assertEqual(len(artifact["calibration_cases"]), 2)
        self.assertTrue(
            all(
                case["metadata"]["source_split"] == "train"
                for case in artifact["calibration_cases"]
            )
        )
        prompts = self.prompts_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(prompts), 1)
        self.assertTrue(prompts[0])
        self.assertNotIn("\n", prompts[0])
        compatibility = artifact["evaluator_compatibility"]
        self.assertEqual(compatibility["prompt_count"], 1)
        self.assertEqual(
            compatibility["prompts_content_sha256"],
            sha256_text(self.prompts_path.read_text(encoding="utf-8")),
        )
        self.assertEqual(
            compatibility["cases_canonical_sha256"],
            canonical_json_sha256({"cases": artifact["cases"]}),
        )
        command = compatibility["example_command"]
        self.assertIn("--held-out-split val", command)
        self.assertIn("--prompt-limit 1", command)
        self.assertIn("--formal", command)

    def test_entity_frequency_and_tiers_are_derived_from_train_only(self):
        self.assertEqual(self.run_builder(), 0)
        baseline = json.loads(self.output_path.read_text(encoding="utf-8"))
        baseline_strata = baseline["entity_strata"]
        self.assertEqual(baseline_strata["statistics_scope"], "train_only")

        # Inflate one name only in validation.  It is deliberately appended as a
        # very long line so it cannot become a new probe paragraph either.
        self.val_text += ("萧鼎" * 500) + "\n"
        self.val_path.write_text(self.val_text, encoding="utf-8")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["splits"]["val"]["sha256"] = sha256_text(self.val_text)
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False),
            encoding="utf-8",
        )
        second_output = self.root / "val-inflated-probes.json"
        second_prompts = self.root / "val-inflated-prompts.txt"
        self.assertEqual(
            self.run_builder(output_path=second_output, prompts_path=second_prompts),
            0,
        )
        inflated = json.loads(second_output.read_text(encoding="utf-8"))
        self.assertEqual(baseline_strata, inflated["entity_strata"])
        for tier in ("high", "low"):
            for entity in inflated["entity_strata"][tier]:
                self.assertIn("train_count", entity)
                self.assertIn("train_chapter_count", entity)
                self.assertNotIn("corpus_count", entity)

    def test_validation_reports_answer_leakage_and_offset_corruption(self):
        self.assertEqual(self.run_builder(), 0)
        artifact = json.loads(self.output_path.read_text(encoding="utf-8"))
        tampered = copy.deepcopy(artifact)
        cloze = next(
            probe
            for probe in tampered["probes"]
            if probe["probe_type"] == "cloze_candidate_ranking"
        )
        cloze["prompt"] += cloze["expected"]["text"]
        continuation = next(
            probe
            for probe in tampered["probes"]
            if probe["probe_type"] == "held_out_continuation"
        )
        continuation["evidence"]["split_char_start"] += 1

        result = validate_probe_artifact(
            tampered,
            {"train": self.train_text, "val": self.val_text},
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("leaked" in failure for failure in result["failures"]))
        self.assertTrue(any("offset" in failure for failure in result["failures"]))

    def test_logs_are_module_scoped_structured_and_do_not_copy_corpus(self):
        self.assertEqual(self.run_builder(), 0)
        for module in ("data", "validation", "orchestrator"):
            path = self.log_dir / f"probe-test-run.{module}.jsonl"
            self.assertTrue(path.is_file())
            events = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertTrue(events)
            self.assertTrue(all(event["run_id"] == "probe-test-run" for event in events))
            self.assertTrue(all(event["module"] == f"cloud.{module}" for event in events))
            self.assertNotIn(NAMES[0], path.read_text(encoding="utf-8"))

    def test_unexpected_output_failure_is_actionable_logged_and_has_no_json_marker(self):
        failed_output = self.root / "must-not-exist.json"
        # Passing a directory where the prompts file should be forces an
        # unexpected IsADirectoryError after probe construction.
        self.assertEqual(
            self.run_builder(output_path=failed_output, prompts_path=self.root),
            2,
        )
        self.assertFalse(failed_output.exists())
        log_path = self.log_dir / "probe-test-run.orchestrator.jsonl"
        events = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        failure = next(
            event
            for event in reversed(events)
            if event["level"] == "ERROR"
        )
        self.assertIn("failed", failure["message"])
        self.assertTrue(failure["context"]["error_type"])
        self.assertIn("permissions", failure["context"]["remediation"])
        self.assertFalse(failure["context"]["json_completion_marker_written"])


if __name__ == "__main__":
    unittest.main()
