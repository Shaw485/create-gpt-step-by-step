from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_sft_v4
from build_sft_v4 import (
    SCHEMA_VERSION,
    TASK_FAMILY_QUOTAS,
    SftV4ReleaseBlocked,
    SftV4ValidationError,
    build_chapter_index,
    build_pipeline,
    make_candidate,
    quality_gate,
    read_jsonl,
    release_records,
    schema_document,
    sha256_file,
)
from validate_sft_v4 import validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data/sft/sft_balanced_v3.jsonl"
CORPUS_PATH = PROJECT_ROOT / "data/clean/doupo_stage3.txt"


def synthetic_release_records() -> tuple[list[dict], list[str], str]:
    corpus_lines = ["第一章 测试", "证据"]
    corpus_hash = sha256("第一章 测试\n证据\n".encode("utf-8")).hexdigest()
    verified_evidence = {
        "status": "verified_corpus",
        "text": "证据",
        "corpus_sha256": corpus_hash,
        "chapter": {"title": "第一章 测试", "heading_line": 1},
        "span": {
            "start_line": 2,
            "end_line": 2,
            "start_character": 0,
            "end_character": 2,
        },
        "sha256": sha256("证据".encode("utf-8")).hexdigest(),
    }
    missing_evidence = {
        "status": "missing",
        "text": "",
        "corpus_sha256": None,
        "chapter": None,
        "span": None,
        "sha256": None,
    }
    family_sequence = []
    for family, count in TASK_FAMILY_QUOTAS.items():
        family_sequence.extend([family] * count)
    records = []
    for index, family in enumerate(family_sequence):
        topic_index = index // 2
        record = make_candidate(
            question=f"测试问题{index}？",
            answer=f"唯一答案{index}。",
            task_family=family,
            topic_id=f"topic:{topic_index}",
            fact_id=f"fact:{topic_index}",
            origin={"kind": "test_fixture"},
            evidence=verified_evidence if index < 2100 else missing_evidence,
            review=(
                {
                    "status": "approved",
                    "reviewer": "fixture-reviewer",
                    "reviewed_at": "2026-08-28T00:00:00+08:00",
                    "notes": "Synthetic unit-test approval only.",
                }
                if index >= 2400
                else None
            ),
        )
        if index < 2400:
            record["split"] = "train"
        elif index < 2700:
            record["split"] = "val"
        else:
            record["split"] = "test"
        records.append(record)
    return records, corpus_lines, corpus_hash


class SftV4PipelineTests(unittest.TestCase):
    def test_chapter_index_prefers_separator_backed_canonical_heading(self):
        lines = [
            "------------",
            "",
            "第一百八十六章 青鳞",
            "",
            "    第一百九十二章 青鳞",
            "正文。",
        ]
        self.assertEqual(
            build_chapter_index(lines),
            [(3, "第一百八十六章 青鳞")],
        )

    def test_schema_contract_has_all_quality_targets(self):
        schema = schema_document()
        self.assertEqual(schema["properties"]["schema_version"]["const"], SCHEMA_VERSION)
        self.assertEqual(schema["quality_contract"]["record_count"], 3000)
        self.assertEqual(schema["quality_contract"]["minimum_topic_count"], 1200)
        self.assertEqual(
            set(schema["properties"]["task_family"]["enum"]),
            set(TASK_FAMILY_QUOTAS),
        )

    def test_real_v3_import_is_deterministic_and_reports_honest_gaps(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"SFT_V4_CONSOLE_LOG": "0"}
        ):
            root = Path(directory)
            output = root / "candidate"
            logs = root / "logs"
            first = build_pipeline(
                source_path=SOURCE_PATH,
                corpus_path=CORPUS_PATH,
                candidate_dir=output,
                log_dir=logs,
            )
            first_bytes = (output / "sft_v4_candidates.jsonl").read_bytes()
            second = build_pipeline(
                source_path=SOURCE_PATH,
                corpus_path=CORPUS_PATH,
                candidate_dir=output,
                log_dir=logs,
            )
            self.assertEqual(
                first_bytes, (output / "sft_v4_candidates.jsonl").read_bytes()
            )
            self.assertEqual(first["accepted_candidate_count"], 700)
            self.assertEqual(first["gaps"]["records_missing"], 2300)
            self.assertEqual(first["gaps"]["topics_missing"], 850)
            self.assertFalse(first["release_ready"])
            self.assertEqual(first["actual"], second["actual"])
            self.assertEqual(
                first["actual"]["maximum_questions_per_fact"], 2
            )
            self.assertEqual(first["leakage"], {"topics": [], "chapters": [], "groups": []})
            self.assertGreaterEqual(first["actual"]["verified_evidence_share"], 0.70)
            candidates = read_jsonl(output / "sft_v4_candidates.jsonl")
            behavioral = [
                row
                for row in candidates
                if row["evidence"]["status"] == "legacy_behavior_claim_unreviewed"
            ]
            self.assertTrue(behavioral)
            self.assertTrue(
                all(row["review"]["status"] == "pending" for row in candidates)
            )
            for module in ("data", "build", "validation"):
                log_path = logs / f"sft_v4_{module}.log"
                self.assertTrue(log_path.exists())
                self.assertGreater(log_path.stat().st_size, 0)

    def test_complete_synthetic_release_passes_every_gate(self):
        records, corpus_lines, corpus_hash = synthetic_release_records()
        report = quality_gate(records, corpus_lines, corpus_hash)
        self.assertTrue(report["release_ready"])
        self.assertEqual(report["failed_gates"], [])
        self.assertEqual(report["actual"]["record_count"], 3000)
        self.assertEqual(report["actual"]["topic_count"], 1500)
        self.assertEqual(report["actual"]["verified_evidence_share"], 0.70)

    def test_tampered_evidence_hash_is_rejected(self):
        records, corpus_lines, corpus_hash = synthetic_release_records()
        records[0]["evidence"] = dict(records[0]["evidence"])
        records[0]["evidence"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(SftV4ValidationError, "evidence hash mismatch"):
            quality_gate(records, corpus_lines, corpus_hash)

    def test_invalid_import_is_logged_in_validation_module(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"SFT_V4_CONSOLE_LOG": "0"}
        ):
            root = Path(directory)
            bad_import = root / "bad.jsonl"
            bad_import.write_text(
                json.dumps(
                    {
                        "question": "错误证据在哪里？",
                        "answer": "不知道。",
                        "task_family": "direct_fact",
                        "topic_id": "bad-evidence",
                        "source_line": 13,
                        "evidence": "这一段并不在指定行中",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            logs = root / "logs"
            with self.assertRaisesRegex(SftV4ValidationError, "evidence is absent"):
                build_pipeline(
                    source_path=SOURCE_PATH,
                    corpus_path=CORPUS_PATH,
                    import_paths=[bad_import],
                    candidate_dir=root / "output",
                    log_dir=logs,
                )
            validation_log = (logs / "sft_v4_validation.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("SFT v4 pipeline failed", validation_log)
            self.assertIn("evidence is absent", validation_log)

    def test_pending_evaluation_review_blocks_release(self):
        records, corpus_lines, corpus_hash = synthetic_release_records()
        evaluation = next(row for row in records if row["split"] == "val")
        evaluation["review"] = {
            "status": "pending",
            "reviewer": None,
            "reviewed_at": None,
            "notes": "not reviewed",
        }
        report = quality_gate(records, corpus_lines, corpus_hash)
        self.assertFalse(report["release_ready"])
        self.assertIn("val_test_human_review", report["failed_gates"])
        with self.assertRaises(SftV4ReleaseBlocked):
            release_records(records, report, Path("unused-corpus.txt"))

    def test_release_manifest_matches_cloud_preflight_contract(self):
        records, corpus_lines, corpus_hash = synthetic_release_records()
        report = quality_gate(records, corpus_lines, corpus_hash)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.txt"
            corpus_path.write_text("第一章 测试\n证据\n", encoding="utf-8")
            release_paths = {
                split: root / f"sft_{split}.jsonl"
                for split in ("train", "val", "test")
            }
            manifest_path = root / "sft_manifest.json"
            with patch.object(build_sft_v4, "RELEASE_DIR", root), patch.object(
                build_sft_v4, "RELEASE_PATHS", release_paths
            ), patch.object(
                build_sft_v4, "RELEASE_MANIFEST_PATH", manifest_path
            ):
                manifest = release_records(records, report, corpus_path)
            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(len(manifest["artifacts"]), 3)
            for artifact in manifest["artifacts"]:
                path = Path(artifact["path"])
                self.assertEqual(artifact["sha256"], sha256_file(path))
                self.assertEqual(artifact["size_bytes"], path.stat().st_size)
            sidecar = manifest_path.with_suffix(".json.sha256")
            self.assertEqual(sidecar.read_text().strip(), sha256_file(manifest_path))

    def test_validator_writes_report_on_non_release_candidate(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"SFT_V4_CONSOLE_LOG": "0"}
        ):
            root = Path(directory)
            candidate_dir = root / "candidate"
            build_pipeline(
                source_path=SOURCE_PATH,
                corpus_path=CORPUS_PATH,
                candidate_dir=candidate_dir,
                log_dir=root / "build-logs",
            )
            dataset_path = candidate_dir / "sft_v4_candidates.jsonl"
            report_path = root / "validation.json"
            report = validate_dataset(
                dataset_path=dataset_path,
                corpus_path=CORPUS_PATH,
                report_path=report_path,
                log_dir=root / "validation-logs",
            )
            self.assertFalse(report["release_ready"])
            self.assertEqual(
                json.loads(report_path.read_text(encoding="utf-8"))["dataset_sha256"],
                sha256_file(dataset_path),
            )


if __name__ == "__main__":
    unittest.main()
