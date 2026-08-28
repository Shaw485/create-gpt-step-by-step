import unittest
from collections import Counter

from build_sft_v5_2_entity_routing import (
    NEW_FAMILIES,
    clean_v5_1_records,
    hidden_prompt_matches,
    identity_candidates,
    is_meta_introduction_clarification,
    repair_candidates,
    validate_entity_routing,
)
from sft_v5_entity_spec import CORE_ENTITY_NAMES, KNOWN_ENTITY_PROFILES


class BuildSftV52EntityRoutingTests(unittest.TestCase):
    def test_cleaner_removes_sparse_known_bare_unknown_and_meta_intro(self):
        records = [
            {"id": "known", "task_family": "novel_known_entity", "question": "甲", "answer": "乙"},
            {"id": "unknown", "task_family": "honest_unknown_general", "question": "甲是谁？", "answer": "资料不足。"},
            {
                "id": "meta",
                "task_family": "ambiguity_unknown_clarification",
                "question": "如果用户现在只说‘请介绍萧炎’，应该怎么回应？",
                "answer": "请先说明作品。",
            },
            {"id": "keep", "task_family": "direct_fact", "question": "第1章？", "answer": "答案。"},
        ]

        kept, removed, reasons = clean_v5_1_records(records)

        self.assertEqual([record["id"] for record in kept], ["keep"])
        self.assertEqual({record["id"] for record in removed}, {"known", "unknown", "meta"})
        self.assertEqual(sum(reasons.values()), 3)

    def test_meta_intro_detector_does_not_remove_local_evidence_unknown(self):
        record = {
            "task_family": "ambiguity_unknown_clarification",
            "question": "只看片段，能知道吴昊的身高吗？",
            "answer": "不能；片段没有给出身高。",
        }

        self.assertFalse(is_meta_introduction_clarification(record))

    def test_core_entities_get_more_train_paraphrases(self):
        counts = Counter(
            name
            for record in identity_candidates()
            if record["split"] == "train"
            for name in KNOWN_ENTITY_PROFILES
            if name in record["question"]
        )

        self.assertEqual(set(counts), set(KNOWN_ENTITY_PROFILES))
        for name in KNOWN_ENTITY_PROFILES:
            expected = 20 if name in CORE_ENTITY_NAMES else 8
            self.assertEqual(counts[name], expected)

    def test_repair_pack_has_all_families_and_no_hidden_prompt_overlap(self):
        candidates = repair_candidates()

        self.assertEqual({record["task_family"] for record in candidates}, NEW_FAMILIES)
        self.assertTrue(all(not hidden_prompt_matches(record["question"]) for record in candidates))

    def test_routing_gate_rejects_known_entity_refusal(self):
        records = repair_candidates()
        bad = dict(records[0])
        bad["answer"] = "现有资料不足，无法确定。"
        records[0] = bad

        with self.assertRaisesRegex(ValueError, "known entity routed to refusal"):
            validate_entity_routing(records)


if __name__ == "__main__":
    unittest.main()
