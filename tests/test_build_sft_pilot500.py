import json
import unittest
from pathlib import Path

from build_sft_pilot500 import (
    EXPECTED_FINAL_SPLITS,
    choose_split_point,
    is_clean_evidence,
    validate_final_records,
)


class BuildSftPilot500Tests(unittest.TestCase):
    def make_records(self):
        records = []
        corpus_lines = []
        for split, count in EXPECTED_FINAL_SPLITS.items():
            for index in range(count):
                record_number = len(records)
                evidence = f"证据{record_number}"
                corpus_lines.append(evidence)
                records.append(
                    {
                        "id": f"{split}_{index}",
                        "question": f"问题{record_number}？",
                        "answer": f"答案{record_number}。",
                        "evidence": evidence,
                        "source_line": record_number + 1,
                        "topic": f"topic_{record_number}",
                        "split": split,
                    }
                )
        characters = set(
            "".join(
                record["question"] + record["answer"] for record in records
            )
        )
        stoi = {char: index for index, char in enumerate(characters)}
        return records, corpus_lines, stoi

    def test_cloze_split_uses_a_natural_comma(self):
        evidence = "萧炎缓缓抬起了手掌，青色火焰随之升腾而起"
        split_point = choose_split_point(evidence)
        self.assertIsNotNone(split_point)
        self.assertTrue(evidence[:split_point].endswith("，"))

    def test_rejects_web_noise_and_quote_fragments(self):
        self.assertFalse(is_clean_evidence("本书最新章节点击这里继续阅读斗气大陆内容"))
        self.assertFalse(is_clean_evidence("萧炎说道：“这不是一条完整的无引号训练句子”"))

    def test_generated_dataset_keeps_exact_counts_and_old_tests(self):
        path = Path("data/sft/sft_expansion400_v1.jsonl")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 400)
        self.assertEqual(len({record["question"] for record in records}), 400)
        self.assertEqual(
            {record["generation_method"] for record in records},
            {
                "grounded_cloze_1_v1",
                "grounded_cloze_2_v1",
                "grounded_cloze_3_v1",
                "grounded_cloze_4_v1",
            },
        )

    def test_validation_rejects_new_cross_split_source_leakage(self):
        records, corpus_lines, stoi = self.make_records()
        records[-1]["source_line"] = records[100]["source_line"]
        records[-1]["evidence"] = records[100]["evidence"]
        with self.assertRaisesRegex(ValueError, "new source lines leak|introduced"):
            validate_final_records(records, corpus_lines, stoi)


if __name__ == "__main__":
    unittest.main()
