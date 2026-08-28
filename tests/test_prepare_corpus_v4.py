import hashlib
import json
import logging
from pathlib import Path
import tempfile
import unittest

from prepare_corpus_v4 import (
    configure_logging,
    main,
    prepare_corpus_v4,
    remove_known_non_story_blocks,
    repair_known_chapter_headings,
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def chapter(number, title, body, *, duplicate_title=False):
    heading = f"第{number}章 {title}"
    duplicate = f"\n{heading}\n" if duplicate_title else ""
    return f"------------\n\n{heading}\n{duplicate}\n{body}\n"


class PrepareCorpusV4Tests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_path = self.root / "doupo_stage3.txt"
        self.audit_path = self.root / "chapter_version_audit_stage3.json"
        self.work_dir = self.root / "clean" / "v4"
        self.cloud_dir = self.root / "cloud_v4"
        self.log_dir = self.root / "logs"

        text = "书名\n\n" + "".join(
            (
                chapter(1, "开端", "正文甲\x7f。", duplicate_title=True),
                chapter(2, "相遇", "正文乙。"),
                chapter(2, "相遇修订", "正文乙的另一个版本。"),
                chapter(3, "出发", "正文丙。"),
                chapter(4, "修炼", "声明:本书由八零电子书(www.txt80.com)整理。\n正文丁。"),
                chapter(5, "归来", "正文戊。"),
                chapter(6, "尾声", "正文己。"),
            )
        )
        self.input_path.write_text(text, encoding="utf-8")

        lines = text.splitlines()
        heading_lines = {
            line.strip(): index
            for index, line in enumerate(lines, start=1)
            if line.startswith("第")
        }
        first_line = heading_lines["第2章 相遇"]
        second_line = heading_lines["第2章 相遇修订"]
        audit = {
            "input_path": str(self.input_path),
            "chapter_section_count": 7,
            "similar_version_pair_count": 1,
            "review_pair_count": 1,
            "version_pairs": [
                {
                    "confidence": "review",
                    "same_chapter_number": True,
                    "matched_characters": 4,
                    "shared_coverage": 0.5,
                    "matching_line_ratio": 0.5,
                    "recommended_keep_start_line": second_line,
                    "recommended_remove_start_line": first_line,
                    "first": {
                        "chapter_number": 2,
                        "title": "第2章 相遇",
                        "start_line": first_line,
                        "end_line": first_line + 3,
                        "body_characters": 4,
                        "quality_score": 4,
                    },
                    "second": {
                        "chapter_number": 2,
                        "title": "第2章 相遇修订",
                        "start_line": second_line,
                        "end_line": second_line + 3,
                        "body_characters": 8,
                        "quality_score": 8,
                    },
                }
            ],
        }
        self.audit_path.write_text(
            json.dumps(audit, ensure_ascii=False),
            encoding="utf-8",
        )
        self.source_before = self.input_path.read_bytes()

    def tearDown(self):
        for logger_name in ("data", "corpus", "split"):
            logger = logging.getLogger(f"corpus_v4.{logger_name}")
            for handler in logger.handlers:
                handler.close()
            logger.handlers.clear()
        self.temporary_directory.cleanup()

    def run_prepare(self):
        loggers = configure_logging(self.log_dir)
        return prepare_corpus_v4(
            input_path=self.input_path,
            version_audit_path=self.audit_path,
            work_dir=self.work_dir,
            cloud_dir=self.cloud_dir,
            seed=42,
            loggers=loggers,
        )

    def test_unresolved_review_freezes_without_deleting_ambiguous_versions(self):
        report = self.run_prepare()

        self.assertEqual(report["status"], "freeze_not_ready")
        self.assertFalse(report["ready"])
        self.assertEqual(report["version_audit"]["unresolved_pair_count"], 1)
        self.assertEqual(report["version_audit"]["reviewed_sections_removed"], 0)
        self.assertEqual(self.source_before, self.input_path.read_bytes())
        self.assertFalse((self.cloud_dir / "train.txt").exists())
        blocker_manifest = json.loads(
            (self.cloud_dir / "corpus_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(blocker_manifest["status"], "freeze_not_ready")
        self.assertEqual(blocker_manifest["artifacts"], [])

        preview = (self.work_dir / "preview" / "corpus.txt").read_text(encoding="utf-8")
        self.assertIn("第2章 相遇", preview)
        self.assertIn("第2章 相遇修订", preview)
        self.assertNotIn("\x7f", preview)
        self.assertEqual(preview.count("第1章 开端"), 1)
        self.assertNotIn("八零电子书", preview)
        self.assertEqual(report["cleaning"]["control_characters_removed"], 1)
        self.assertEqual(
            report["cleaning"]["adjacent_duplicate_title_lines_removed"],
            1,
        )
        self.assertGreaterEqual(
            report["cleaning"]["known_site_noise_whole_lines_removed"],
            1,
        )
        self.assertFalse(report["split"]["same_chapter_versions_cross_split"])

    def test_explicit_review_publishes_deterministic_cloud_splits(self):
        first_report = self.run_prepare()
        resolution_path = self.work_dir / "chapter_version_resolutions.json"
        resolutions = json.loads(resolution_path.read_text(encoding="utf-8"))
        resolutions["resolutions"][0].update(
            {"decision": "keep_second", "reviewer": "unit-test", "note": "longer"}
        )
        resolution_path.write_text(
            json.dumps(resolutions, ensure_ascii=False),
            encoding="utf-8",
        )

        report = self.run_prepare()

        self.assertEqual(first_report["status"], "freeze_not_ready")
        self.assertEqual(report["status"], "ready")
        self.assertTrue(report["ready"])
        self.assertEqual(report["version_audit"]["reviewed_sections_removed"], 1)
        self.assertEqual(self.source_before, self.input_path.read_bytes())
        for split in ("train", "val", "test"):
            output_path = self.cloud_dir / f"{split}.txt"
            self.assertTrue(output_path.exists())
            self.assertEqual(
                sha256_bytes(output_path.read_bytes()),
                report["split"]["splits"][split]["sha256"],
            )

        manifest = json.loads(
            (self.cloud_dir / "corpus_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["status"], "ready")
        self.assertEqual(len(manifest["artifacts"]), 4)
        self.assertTrue((self.cloud_dir / "corpus.txt").exists())
        self.assertTrue((self.cloud_dir / "corpus_manifest.json.sha256").exists())
        published = "".join(
            (self.cloud_dir / f"{split}.txt").read_text(encoding="utf-8")
            for split in ("train", "val", "test")
        )
        self.assertNotIn("第2章 相遇\n", published)
        self.assertIn("第2章 相遇修订", published)

        hashes_before = {
            split: sha256_bytes((self.cloud_dir / f"{split}.txt").read_bytes())
            for split in ("train", "val", "test")
        }
        second_report = self.run_prepare()
        hashes_after = {
            split: sha256_bytes((self.cloud_dir / f"{split}.txt").read_bytes())
            for split in ("train", "val", "test")
        }
        self.assertEqual(hashes_before, hashes_after)
        self.assertEqual(report["split"], second_report["split"])

    def test_command_returns_two_for_review_gate_and_writes_separate_logs(self):
        exit_code = main(
            [
                "--input",
                str(self.input_path),
                "--version-audit",
                str(self.audit_path),
                "--work-dir",
                str(self.work_dir),
                "--cloud-dir",
                str(self.cloud_dir),
                "--log-dir",
                str(self.log_dir),
            ]
        )

        self.assertEqual(exit_code, 2)
        for category in ("data", "corpus", "split"):
            log_path = self.log_dir / f"corpus_v4_{category}.log"
            self.assertTrue(log_path.exists())
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn(f"corpus_v4.{category}", log_text)
            self.assertNotIn("正文甲", log_text)

        status = json.loads(
            (self.work_dir / "reports" / "freeze_status.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            status["failure_reasons"][0]["code"],
            "UNRESOLVED_CHAPTER_VERSION_REVIEWS",
        )

    def test_invalid_audit_fails_with_actionable_status(self):
        audit = json.loads(self.audit_path.read_text(encoding="utf-8"))
        audit["chapter_section_count"] = 999
        self.audit_path.write_text(json.dumps(audit), encoding="utf-8")

        exit_code = main(
            [
                "--input",
                str(self.input_path),
                "--version-audit",
                str(self.audit_path),
                "--work-dir",
                str(self.work_dir),
                "--cloud-dir",
                str(self.cloud_dir),
                "--log-dir",
                str(self.log_dir),
            ]
        )

        self.assertEqual(exit_code, 1)
        status = json.loads(
            (self.work_dir / "reports" / "freeze_status.json").read_text(
                encoding="utf-8"
            )
        )
        reason = status["failure_reasons"][0]
        self.assertEqual(reason["code"], "AUDIT_SECTION_COUNT_MISMATCH")
        self.assertIn("Regenerate", reason["remediation"])
        failure_log = (self.log_dir / "corpus_v4_data.log").read_text(
            encoding="utf-8"
        )
        self.assertIn("AUDIT_SECTION_COUNT_MISMATCH", failure_log)

    def test_known_heading_repairs_and_non_story_removal_preserve_line_numbers(self):
        text = (
            "------------\n\n正文\n\n"
            "------------\n\n第八十七 下杀手\n\n正文甲。\n"
        )
        story, removals = remove_known_non_story_blocks(text)
        repaired, repairs = repair_known_chapter_headings(story)
        self.assertEqual(len(text.splitlines()), len(repaired.splitlines()))
        self.assertEqual(len(removals), 1)
        self.assertEqual(len(repairs), 1)
        self.assertNotIn("\n正文\n", repaired)
        self.assertIn("第八十七章 下杀手", repaired)


if __name__ == "__main__":
    unittest.main()
