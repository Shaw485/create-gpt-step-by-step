from hashlib import sha256
import unittest

from repair_teacher_sft_v4 import (
    CorpusEvidenceLocator,
    assign_grouped_splits,
    ai_pre_review,
    build_ai_pre_review_summary,
    build_review_priority_summary,
    first_sentence,
    mapped_family,
    normalize_answer,
    normalize_question,
    remove_control_characters,
    rebase_chapter_references,
    title_from_chapter_heading,
    rebalance_task_families,
)
from build_sft_v4 import TASK_FAMILY_QUOTAS
from collections import Counter


class TeacherSftV4RepairTests(unittest.TestCase):
    def test_locator_returns_exact_span_after_whitespace_normalization(self):
        lines = ["第一章 测试", "  萧炎 轻声道。"]
        digest = sha256("\n".join(lines).encode()).hexdigest()
        result = CorpusEvidenceLocator(lines, digest).locate("萧炎轻声道。", 1)
        self.assertIsNotNone(result)
        assert result is not None
        span = result.evidence["span"]
        self.assertEqual(
            lines[span["start_line"] - 1][span["start_character"] : span["end_character"]],
            result.evidence["text"],
        )
        self.assertTrue(result.chapter_matches_claim)

    def test_locator_can_rebind_a_cleaned_quote_inside_the_claimed_chapter(self):
        lines = [
            "第一章 测试",
            "萧炎抬起头，望着远处的山峰，然后缓缓向前走去。",
        ]
        digest = sha256("\n".join(lines).encode()).hexdigest()
        result = CorpusEvidenceLocator(lines, digest).locate(
            "广告文字萧炎抬起头，望着远处的山峰，然后缓缓向前走去。尾部文字",
            1,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.evidence["match_method"], "fuzzy_chapter_rebind")
        self.assertTrue(result.chapter_matches_claim)

    def test_locator_can_bind_directly_to_a_chapter_heading(self):
        lines = ["第一章 测试", "正文。", "第二章 后续", "更多正文。"]
        digest = sha256("\n".join(lines).encode()).hexdigest()
        result = CorpusEvidenceLocator(lines, digest).locate_chapter_heading(2)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.evidence["text"], "第二章 后续")
        self.assertEqual(result.evidence["span"]["start_line"], 3)
        self.assertEqual(result.evidence["match_method"], "chapter_heading")

    def test_heading_title_removes_control_characters_before_tokenization(self):
        self.assertEqual(title_from_chapter_heading("第二章 后\x7f续"), "后续")
        self.assertEqual(remove_control_characters("大\x7f战"), "大战")

    def test_rebase_replaces_only_the_stale_chapter_reference(self):
        text = "第九百六十一章之后对照第九百六十二章。"
        self.assertEqual(
            rebase_chapter_references(text, 961, 943),
            "第943章之后对照第九百六十二章。",
        )

    def test_category_mapping_splits_correction_and_unknown(self):
        self.assertEqual(
            mapped_family({"category": "core_fact"}), "direct_fact"
        )
        self.assertEqual(
            mapped_family(
                {"category": "correction_unanswerable", "subcategory": "unanswerable"}
            ),
            "ambiguity_unknown_clarification",
        )

    def test_chapter_focus_repair_makes_high_frequency_answer_specific(self):
        answer, repairs = normalize_answer(
            {
                "answer": "主要是萧炎。",
                "subcategory": "chapter_focus",
                "source": {"chapter_number": 12},
            }
        )
        self.assertEqual(answer, "第12章的情节主要围绕萧炎展开。")
        self.assertIn("contextualized_repeated_chapter_focus_answer", repairs)

    def test_chapter_metadata_answers_drop_unasked_padding(self):
        for subcategory, expected in (
            ("chapter_title", "第12章的标题是《测试标题》。"),
            ("chapter_locate", "《测试标题》是小说第12章的标题。"),
            ("chapter_order", "下一章是第12章《测试标题》。"),
        ):
            answer, repairs = normalize_answer(
                {
                    "answer": "错误的冗长答案。",
                    "subcategory": subcategory,
                    "source": {"chapter_number": 12, "chapter_title": "测试标题"},
                }
            )
            self.assertEqual(answer, expected)
            self.assertTrue(repairs)

    def test_future_realm_is_not_misreported_as_current_realm(self):
        answer, repairs = normalize_answer(
            {
                "answer": "是二星斗尊。",
                "subcategory": "realm_state",
                "entities": ["萧炎", "二星斗尊"],
                "source": {"evidence_quote": "萧炎想要达到二星斗尊，至少还需半年。"},
            }
        )
        self.assertIn("尚未达到二星斗尊", answer)
        self.assertIn("没有明确给出当前星级", answer)
        self.assertIn("corrected_future_realm_as_unknown_current_state", repairs)

    def test_cause_question_grammar_is_repaired_without_changing_meaning(self):
        question, repairs = normalize_question(
            {
                "question": "小说第五百十九章中，是什么导致萧炎并未回去？"
            }
        )
        self.assertEqual(
            question, "小说第五百十九章中，萧炎并未回去的原因是什么？"
        )
        self.assertIn("repaired_cause_question_grammar", repairs)

    def test_first_sentence_keeps_one_complete_chinese_sentence(self):
        self.assertEqual(first_sentence("第一句。第二句。"), "第一句。")
        self.assertEqual(first_sentence("没有句号"), "没有句号")

    def test_incomplete_kinship_answer_is_repaired_from_verbatim_evidence(self):
        answer, repairs = normalize_answer(
            {
                "question": "萧炎与萧玉之间是什么关系？",
                "answer": "是萧玉。",
                "subcategory": "kinship",
                "source": {"evidence_quote": "她是萧炎的表姐，萧玉。"},
            }
        )
        self.assertEqual(answer, "萧玉是萧炎的表姐。")
        self.assertIn("repaired_incomplete_kinship_answer", repairs)

    def test_grouped_split_keeps_topic_and_chapter_together(self):
        records = []
        for index in range(3000):
            records.append(
                {
                    "knowledge_unit_id": f"topic-{index // 2}",
                    "source": {"chapter_number": index // 4},
                }
            )
        splits, groups = assign_grouped_splits(records)
        self.assertEqual(splits.count("train"), 2400)
        self.assertEqual(splits.count("val"), 300)
        self.assertEqual(splits.count("test"), 300)
        by_chapter = {}
        for record, split, group in zip(records, splits, groups):
            chapter = record["source"]["chapter_number"]
            by_chapter.setdefault(chapter, set()).add((split, group))
        self.assertTrue(all(len(values) == 1 for values in by_chapter.values()))

    def test_teacher_distribution_rebalances_to_frozen_family_quotas(self):
        categories = {
            "core_fact": 650,
            "worldbuilding_concept": 300,
            "long_tail_detail": 250,
            "character_relation": 400,
            "timeline_event": 400,
            "cause_motivation_result": 400,
            "comparison_synthesis": 200,
            "plot_summary_extraction": 250,
            "false_premise": 57,
            "unanswerable": 93,
        }
        records = []
        index = 0
        for category, count in categories.items():
            for _ in range(count):
                records.append(
                    {
                        "id": f"id-{index}",
                        "category": (
                            "correction_unanswerable"
                            if category in {"false_premise", "unanswerable"}
                            else category
                        ),
                        "subcategory": category,
                        "knowledge_unit_id": f"topic-{index // 2}",
                    }
                )
                index += 1
        assignments = rebalance_task_families(records)
        self.assertEqual(
            Counter(family for family, _ in assignments.values()),
            Counter(TASK_FAMILY_QUOTAS),
        )

    def test_review_priority_is_disjoint_and_never_infers_human_approval(self):
        def record(split, flags):
            return {"split": split, "origin": {"repair_flags": flags}}

        summary = build_review_priority_summary(
            [
                record("test", ["evidence_absent_from_frozen_v4"]),
                record("train", ["fuzzy_chapter_rebind_requires_review"]),
                record("train", ["transformed_task_requires_review"]),
                record("train", []),
            ]
        )
        self.assertEqual(sum(summary["counts"].values()), 4)
        self.assertEqual(summary["counts"]["P0_evaluation_human_review"], 1)
        self.assertEqual(summary["counts"]["P1_training_provenance_review"], 1)
        self.assertEqual(summary["counts"]["P2_training_semantic_review"], 1)
        self.assertEqual(summary["clean_training_candidate_count"], 1)
        self.assertFalse(summary["human_approval_was_inferred"])

    def test_ai_pre_review_prioritizes_blockers_without_approving(self):
        def record(flags):
            return {
                "split": "test",
                "origin": {"repair_flags": flags},
            }

        rows = [
            record(["claimed_chapter_mismatch"]),
            record(["global_claim_requires_index_review"]),
            record(["transformed_task_requires_review"]),
            record([]),
        ]
        self.assertEqual(ai_pre_review(rows[0])[1], "fix_before_human_review")
        summary = build_ai_pre_review_summary(rows)
        self.assertEqual(summary["evaluation_count"], 4)
        self.assertFalse(summary["human_approval_was_inferred"])


if __name__ == "__main__":
    unittest.main()
