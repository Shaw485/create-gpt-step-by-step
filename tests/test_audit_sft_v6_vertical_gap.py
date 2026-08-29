import json
from pathlib import Path
import tempfile
import unittest

from audit_sft_v6_vertical_gap import (
    AUDITED_RECORD_LIMIT,
    audit_v6_vertical_gap,
    read_nonsealed_prefix,
    read_prefix_from_lines,
)


def v6_record(index: int, *, family: str = "curated_core_novel_identity") -> dict:
    answer = f"萧炎是小说中的核心人物{index}。"
    return {
        "schema_version": "sft_v6/1.0",
        "id": f"v6_{index}",
        "split": "train",
        "primary_dimension": "novel_entities_facts_relations_timeline",
        "task_family": family,
        "question": f"萧炎是谁{index}？",
        "answer": answer,
        "messages": [
            {"role": "user", "content": f"萧炎是谁{index}？"},
            {"role": "assistant", "content": answer},
        ],
        "evidence": {"status": "curated_project_fact"},
    }


class SentinelLines:
    """Raise loudly if a consumer asks for the sealed element."""

    def __init__(self, count: int):
        self.count = count
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.count:
            raise AssertionError("sealed sentinel was requested")
        record = v6_record(self.index)
        self.index += 1
        return json.dumps(record, ensure_ascii=False) + "\n"


class AuditSftV6VerticalGapTests(unittest.TestCase):
    def test_reader_never_requests_the_9401st_sentinel_item(self):
        lines = SentinelLines(AUDITED_RECORD_LIMIT)
        records, prefix_hash = read_prefix_from_lines(lines)
        self.assertEqual(len(records), AUDITED_RECORD_LIMIT)
        self.assertEqual(lines.index, AUDITED_RECORD_LIMIT)
        self.assertEqual(len(prefix_hash), 64)

    def test_file_reader_does_not_parse_invalid_sealed_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for index in range(4):
                    handle.write(json.dumps(v6_record(index), ensure_ascii=False) + "\n")
                handle.write("THIS IS SEALED AND NOT JSON\n")
            records, prefix_hash = read_nonsealed_prefix(path, limit=4)
            self.assertEqual(len(records), 4)
            self.assertEqual(len(prefix_hash), 64)

    def test_gap_report_is_needs_revision_and_contains_only_aggregate_templates(self):
        first = v6_record(1, family="project_concept_explanation")
        first["messages"] = [
            first["messages"][0],
            {
                "role": "assistant",
                "content": "可以先学习 Token，但秘密正文绝不能出现在报告里。",
            },
            {"role": "user", "content": "那萧炎是谁？"},
            {"role": "assistant", "content": first["answer"]},
        ]
        second = v6_record(2, family="natural_single_turn_support")
        report = audit_v6_vertical_gap([first, second], expected_records=2)

        self.assertEqual(report["status"], "needs_revision")
        self.assertEqual(report["domain_alignment"]["project_concept_records"], 1)
        self.assertEqual(report["domain_alignment"]["generic_support_records"], 1)
        self.assertEqual(
            report["template_statistics"]["forbidden_marker_counts"]["可以先"],
            1,
        )
        self.assertEqual(
            report["template_statistics"]["forbidden_marker_count_unit"],
            "assistant_messages_containing_marker",
        )
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("秘密正文", serialized)
        self.assertFalse(report["privacy"]["sealed_body_accessed"])


if __name__ == "__main__":
    unittest.main()
