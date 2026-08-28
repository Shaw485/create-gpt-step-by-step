import unittest
from collections import Counter

from build_sft_v5_1_no_math import (
    HELD_OUT_CANONICAL_QUESTIONS,
    EXPECTED_REPAIR_FAMILIES,
    allocate_grouped_splits,
    assert_no_math_or_pollution,
    canonicalize_question,
    clean_base_records,
    evidence_entity_candidates,
    filter_canonical_duplicates,
    held_out_prompt_matches,
    is_arithmetic_text,
    is_math_topic_text,
    natural_conversation_candidates,
    repair_candidates,
    validate_repair_quality,
)


class BuildSftV51NoMathTests(unittest.TestCase):
    def test_clean_base_removes_math_meta_tasks_and_polluted_prefix(self):
        records = [
            {
                "id": "math",
                "task_family": "basic_reasoning",
                "question": "2加3等于几？",
                "answer": "5。",
            },
            {
                "id": "domain",
                "task_family": "domain_switching",
                "question": "新闻题要回答章节吗？",
                "answer": "不应该。",
            },
            {
                "id": "copy",
                "task_family": "continuation_rewrite_instruction",
                "question": "请原样重复以下问题，不要回答。",
                "answer": "问题。",
            },
            {
                "id": "prefix",
                "task_family": "general_chat",
                "question": "别提小说章节，回答：你好",
                "answer": "你好。",
            },
            {
                "id": "fact",
                "task_family": "direct_fact",
                "question": "异火榜排名是多少？",
                "answer": "排名第十三。",
            },
        ]

        kept, removed, reasons = clean_base_records(records)

        self.assertEqual([record["id"] for record in kept], ["fact"])
        self.assertEqual(len(removed), 4)
        self.assertEqual(reasons["arithmetic"], 1)
        self.assertEqual(reasons["domain_switching"], 1)
        self.assertEqual(reasons["continuation_rewrite_instruction"], 1)
        self.assertEqual(reasons["polluted_prompt_prefix"], 1)

    def test_arithmetic_detector_is_narrow(self):
        self.assertTrue(is_arithmetic_text("请回答2加3等于几？"))
        self.assertTrue(is_arithmetic_text("零加一等于几？"))
        self.assertTrue(is_arithmetic_text("两减一是多少？"))
        self.assertTrue(is_arithmetic_text("二乘三等于几？"))
        self.assertTrue(is_arithmetic_text("六除二的结果是多少？"))
        self.assertTrue(is_arithmetic_text("请计算2×3。"))
        self.assertTrue(is_arithmetic_text("6/2等于几？"))
        self.assertTrue(is_arithmetic_text("如果有6个苹果，一共有几个？"))
        self.assertTrue(is_arithmetic_text("原有六本书，拿走两本，还剩几本？"))
        self.assertFalse(is_arithmetic_text("异火榜排名是多少？"))
        self.assertFalse(is_arithmetic_text("第300章的标题是什么？"))
        self.assertFalse(is_arithmetic_text("日期是2026-08-28。"))
        self.assertFalse(is_arithmetic_text("网址是https://example.com/docs/2。"))
        self.assertTrue(is_math_topic_text("我应该如何学习数学？"))
        self.assertFalse(is_math_topic_text("第300章的标题是什么？"))

    def test_neutral_prefix_cannot_bypass_eval_holdout(self):
        canonical = canonicalize_question("自然一点回答：今天天气怎么样？")

        self.assertIn(canonical, HELD_OUT_CANONICAL_QUESTIONS)

    def test_embedded_eval_prompt_cannot_bypass_holdout(self):
        question = "如果用户问‘今天天气怎么样？’，应该说明不能看实时天气。"

        self.assertEqual(
            held_out_prompt_matches(question),
            (canonicalize_question("今天天气怎么样？"),),
        )

        records = [
            {
                "id": "wrapped-weather",
                "task_family": "capability_boundary",
                "question": question,
                "answer": "说明能力边界。",
            }
        ]
        kept, removed, reasons = clean_base_records(records)

        self.assertEqual(kept, [])
        self.assertEqual([record["id"] for record in removed], ["wrapped-weather"])
        self.assertEqual(reasons["held_out_evaluation_prompt_overlap"], 1)

    def test_canonical_duplicate_filter_rejects_full_width_punctuation_variant(self):
        candidates = [
            {"question": "TopK是什么？"},
            {"question": "自然一点回答：今天天气怎么样？"},
        ]

        accepted, rejected = filter_canonical_duplicates(
            candidates,
            {canonicalize_question("TopK是什么?")}
            | HELD_OUT_CANONICAL_QUESTIONS,
        )

        self.assertEqual(accepted, [])
        self.assertEqual(rejected, 2)

    def test_repair_candidates_are_clean_and_cover_expected_families(self):
        candidates = repair_candidates()

        assert_no_math_or_pollution(candidates)
        self.assertEqual(
            {record["task_family"] for record in candidates},
            EXPECTED_REPAIR_FAMILIES,
        )

    def test_grouped_splits_do_not_leak_semantic_groups(self):
        allocated = allocate_grouped_splits(repair_candidates())
        group_splits = {}
        for record in allocated:
            group_splits.setdefault(record["group_id"], set()).add(record["split"])

        self.assertTrue(all(len(splits) == 1 for splits in group_splits.values()))
        self.assertEqual({"train", "val", "test"}, {r["split"] for r in allocated})

    def test_repair_quality_gate_passes_and_answers_are_diverse(self):
        allocated = allocate_grouped_splits(repair_candidates())

        quality = validate_repair_quality(allocated)

        self.assertEqual(quality["semantic_group_leaks"], 0)
        for family, metrics in quality["families"].items():
            threshold = 0.30 if family == "instruction_following_repair" else 0.75
            self.assertGreaterEqual(metrics["unique_answer_ratio"], threshold)

    def test_natural_candidates_have_unique_answers(self):
        candidates = natural_conversation_candidates()

        counts = Counter(record["answer"] for record in candidates)
        self.assertEqual(len(counts), len(candidates))
        self.assertEqual(max(counts.values()), 1)

    def test_evidence_answer_names_second_entity_without_copying_first(self):
        candidates = evidence_entity_candidates()
        matching = [
            record
            for record in candidates
            if "韩枫和谁一起出现" in record["question"]
            and "紫研" in record["question"]
        ]

        self.assertEqual(len(matching), 1)
        self.assertIn("紫研", matching[0]["answer"])
        self.assertNotEqual(matching[0]["answer"], "韩枫。")


if __name__ == "__main__":
    unittest.main()
