import json
from pathlib import Path
import tempfile
import unittest

from build_sft_v7_vertical import EvidenceChunk, _record
from validate_sft_v7_vertical import (
    CAPABILITY_BOUNDARY,
    GROUNDED_SINGLE,
    PARAMETRIC_CORE,
    PUBLIC_SPLIT,
    SCHEMA_VERSION,
    SEALED_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    VERTICAL_CHAT,
    stable_hash,
    text_sha256,
    parse_args,
    validate_dataset_files,
    validate_records_by_split,
)


class FakeTokenizer:
    special_to_id = {
        "<BOS>": 0,
        "<USER>": 1,
        "<ASSISTANT>": 2,
        "<EOS>": 3,
        "<PAD>": 4,
    }

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


def evidence_chunk(path: Path, text: str, *, line: int = 1) -> dict:
    sha256 = text_sha256(text)
    heading_line = 1
    return {
        "source_path": str(path),
        "source_split": "formal_pretrain_train",
        "chapter_number": 1,
        "chapter_title": "测试章节",
        "chapter_heading_line": heading_line,
        "chapter_sha256": text_sha256("测试章节"),
        "line_start": line,
        "line_end": line,
        "text": text,
        "text_sha256": sha256,
        "chunk_sha256": stable_hash("sft-v7-chunk", heading_line, line, sha256),
    }


def record(
    record_id: str,
    split: str,
    corpus_path: Path,
    *,
    question: str | None = None,
    answer: str = "萧炎是小说中的核心人物。",
    dimension: str = PARAMETRIC_CORE,
    metric: str = "entity_fact_accuracy",
    template: str | None = None,
    style: str = "direct_fact",
    track: str = "train_only",
    triplet: str = "",
    evidence_values: list[dict] | None = None,
    known_fact: bool | None = None,
    needs_evidence: bool | None = None,
    evidence_sufficient: bool | None = None,
) -> dict:
    question = question or f"萧炎是谁{record_id}？"
    if known_fact is None:
        known_fact = dimension == PARAMETRIC_CORE
    if needs_evidence is None:
        needs_evidence = dimension == CAPABILITY_BOUNDARY
    if evidence_sufficient is None:
        evidence_sufficient = dimension in {PARAMETRIC_CORE, GROUNDED_SINGLE}
    if evidence_values is None:
        evidence_values = (
            [evidence_chunk(corpus_path, "萧炎是小说中的核心人物。")]
            if evidence_sufficient
            else []
        )
    bundle_sha = text_sha256(
        "|".join(str(item["chunk_sha256"]) for item in evidence_values)
    )
    required_terms = ["萧炎"] if evidence_sufficient else []
    return {
        "schema_version": SCHEMA_VERSION,
        "id": record_id,
        "split": split,
        "primary_dimension": dimension,
        "task_family": (
            "core_identity" if dimension == PARAMETRIC_CORE else "test_family"
        ),
        "semantic_group": f"semantic_{record_id}",
        "fact_id": f"fact_{record_id}",
        "generalization_policy": track,
        "question": question,
        "answer": answer,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "evidence": {
            "status": (
                "verified_train_corpus"
                if evidence_values
                else "insufficient"
                if needs_evidence
                else "not_applicable"
            ),
            "sufficient_for_answer": evidence_sufficient,
            "negative_type": "missing_evidence" if needs_evidence else None,
            "bundle_sha256": bundle_sha,
            "chunks": evidence_values,
        },
        "evaluation": {
            "metric": metric,
            "required_terms": required_terms,
            "forbidden_terms": ["资料不足"] if known_fact else [],
            "known_fact": known_fact,
            "needs_evidence": needs_evidence,
            "evidence_sufficient": evidence_sufficient,
            "acceptance_case_id": "",
            "calibration_triplet_id": triplet,
        },
        "generation": {
            "prompt_template_id": template or f"template_{record_id}",
            "prompt_template_sha256": text_sha256(template or f"template_{record_id}"),
            "answer_style_id": style,
        },
        "encoding_audit": {
            "sequence_tokens": 10,
            "supervised_tokens": 4,
            "last_answer_tokens": 3,
            "assistant_turns": 1,
            "eos_targets": 1,
            "masked_user_and_role_tokens": 6,
        },
        "coverage": {"entities": ["萧炎"], "concepts": []},
        "provenance": {"generation_method": "unit_test"},
        "review": {"status": "approved", "evidence_recomputed": True},
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )


class ValidateSftV7VerticalTests(unittest.TestCase):
    def test_default_validation_manifest_cannot_overwrite_builder_manifest(self):
        args = parse_args([])
        self.assertEqual(args.manifest.name, "validation_manifest.json")
        self.assertNotEqual(args.manifest, Path("data/sft/v7/manifest.json"))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.corpus = self.root / "train.txt"
        self.corpus.write_text(
            "萧炎是小说中的核心人物。\n药尘是萧炎的老师。\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def validate_small(self, values_by_split):
        return validate_records_by_split(
            values_by_split,
            corpus_path=self.corpus,
            tokenizer=None,
            enforce_release_gates=False,
        )

    def test_valid_builder_schema_core_record_passes_contract(self):
        report, risks = self.validate_small(
            {TRAIN_SPLIT: [record("case_a", TRAIN_SPLIT, self.corpus)]}
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(risks, [])

    def test_core_fact_accepts_two_reviewed_evidence_items(self):
        evidence_values = [
            evidence_chunk(self.corpus, "萧炎是小说中的核心人物。", line=1),
            evidence_chunk(self.corpus, "药尘是萧炎的老师。", line=2),
        ]
        item = record(
            "case_two_core_evidence",
            TRAIN_SPLIT,
            self.corpus,
            evidence_values=evidence_values,
        )
        report, risks = self.validate_small({TRAIN_SPLIT: [item]})
        self.assertEqual(report["status"], "passed")
        self.assertFalse(
            any(
                risk["code"]
                in {
                    "core_fact_requires_one_or_two_evidence_items",
                    "single_evidence_dimension_requires_one_item",
                }
                for risk in risks
            )
        )

    def test_exact_answer_repeat_covers_all_non_reviewed_core_families(self):
        repeated_answer = "我会围绕小说文本回答这个问题。"
        records = []
        for index in range(6):
            item = record(
                f"chat_repeat_{index}",
                TRAIN_SPLIT,
                self.corpus,
                answer=repeated_answer,
                dimension=VERTICAL_CHAT,
                metric="conversation_quality",
                evidence_values=[],
                known_fact=False,
                needs_evidence=False,
                evidence_sufficient=False,
            )
            records.append(item)
        for index in range(8):
            item = record(
                f"reviewed_core_repeat_{index}",
                TRAIN_SPLIT,
                self.corpus,
            )
            item["task_family"] = "known_core_direct"
            records.append(item)

        report, _ = self.validate_small({TRAIN_SPLIT: records})
        self.assertEqual(report["quality"]["maximum_general_exact_answer_repeat"], 6)
        self.assertFalse(report["record_body_emitted"])

    def test_actual_builder_record_is_accepted_by_validator_contract(self):
        source = "萧炎是小说中的核心人物。"
        chunk = EvidenceChunk(
            split=TRAIN_SPLIT,
            chapter_number=1,
            chapter_title="测试章节",
            chapter_heading_line=1,
            chapter_sha256=text_sha256("测试章节"),
            line_number=1,
            text=source,
            clean_text=source,
            terms=("萧炎",),
        )
        item = _record(
            split=TRAIN_SPLIT,
            dimension=PARAMETRIC_CORE,
            family="known_core_direct",
            record_index=0,
            messages=[
                {"role": "user", "content": "萧炎是谁？"},
                {"role": "assistant", "content": source},
            ],
            chunks=[chunk],
            corpus_path=str(self.corpus),
            corpus_sha256=text_sha256(self.corpus.read_text(encoding="utf-8")),
            tokenizer=FakeTokenizer(),
            prompt_id="train:known_core_direct:p0",
            prompt_text="{question}",
            style_id="train:known_core_direct:a0",
            entity="萧炎",
            semantic_group="known-core:test",
            fact_id="known:test",
            generalization="seen_fact_unseen_wording",
            evidence_status="reviewed_exact_train_lines",
            negative_type=None,
            metric="keypoints",
            required_terms=("萧炎",),
            forbidden_terms=("资料不足",),
            known_fact=True,
            needs_evidence=False,
            evidence_sufficient=True,
        )
        report, risks = self.validate_small({TRAIN_SPLIT: [item]})
        self.assertEqual(report["status"], "passed")
        self.assertEqual(risks, [])

    def test_known_core_refusal_and_evaluation_terms_are_blocked(self):
        item = record(
            "case_refusal",
            TRAIN_SPLIT,
            self.corpus,
            answer="资料不足，无法确认。",
        )
        report, _ = self.validate_small({TRAIN_SPLIT: [item]})
        codes = report["risk_code_counts"]
        self.assertEqual(codes["known_core_false_refusal"], 1)
        self.assertEqual(codes["required_term_missing_from_answer"], 1)
        self.assertEqual(codes["forbidden_term_present_in_answer"], 1)

    def test_evidence_line_chunk_and_bundle_hashes_are_recomputed(self):
        item = record("case_evidence", TRAIN_SPLIT, self.corpus)
        item["evidence"]["chunks"][0]["text_sha256"] = "0" * 64
        report, _ = self.validate_small({TRAIN_SPLIT: [item]})
        codes = report["risk_code_counts"]
        self.assertEqual(codes["evidence_sha256_mismatch"], 1)
        self.assertEqual(codes["evidence_chunk_sha256_mismatch"], 1)
        self.assertEqual(codes["evidence_bundle_sha256_mismatch"], 1)

    def test_prompt_template_id_cannot_cross_splits(self):
        train = record(
            "case_train",
            TRAIN_SPLIT,
            self.corpus,
            template="shared_template",
            track="seen_fact_unseen_wording",
        )
        val = record(
            "case_val",
            VAL_SPLIT,
            self.corpus,
            question="请直接说明萧炎的身份。",
            template="shared_template",
            track="seen_fact_unseen_wording",
        )
        train["semantic_group"] = "shared_core_fact"
        val["semantic_group"] = "shared_core_fact"
        report, _ = self.validate_small({TRAIN_SPLIT: [train], VAL_SPLIT: [val]})
        self.assertEqual(report["prompt_template_split_leaks"], 1)
        self.assertEqual(report["semantic_group_split_leaks"], 0)
        self.assertEqual(report["evidence_sha_split_leaks"], 0)

    def test_complete_known_needs_evidence_grounded_triplet_is_recognized(self):
        triplet = "triplet_alpha"
        known = record("triplet_known", TRAIN_SPLIT, self.corpus, triplet=triplet)
        needs = record(
            "triplet_needs",
            TRAIN_SPLIT,
            self.corpus,
            answer="这个长尾断言需要检索证据后才能确认。",
            dimension=CAPABILITY_BOUNDARY,
            metric="boundary_accuracy",
            triplet=triplet,
            evidence_values=[],
            known_fact=False,
            needs_evidence=True,
            evidence_sufficient=False,
        )
        grounded = record(
            "triplet_grounded",
            TRAIN_SPLIT,
            self.corpus,
            answer="根据材料，可以确认萧炎是小说中的核心人物。",
            dimension=GROUNDED_SINGLE,
            metric="evidence_support",
            triplet=triplet,
            known_fact=False,
            needs_evidence=False,
            evidence_sufficient=True,
        )
        report, risks = self.validate_small(
            {TRAIN_SPLIT: [known, needs, grounded]}
        )
        self.assertEqual(
            report["calibration"]["complete_triplets_by_split"][TRAIN_SPLIT], 1
        )
        self.assertEqual(report["calibration"]["incomplete_triplets"], 0)
        self.assertFalse(
            any(risk["code"] == "known_core_false_refusal" for risk in risks)
        )

    def test_acceptance_case_is_not_counted_as_calibration_triplet(self):
        item = record("acceptance_only", TRAIN_SPLIT, self.corpus)
        item["evaluation"]["acceptance_case_id"] = "known_core_acceptance_only"
        item["evaluation"]["calibration_triplet_id"] = ""
        report, _ = self.validate_small({TRAIN_SPLIT: [item]})
        self.assertEqual(report["calibration"]["incomplete_triplets"], 0)

    def test_reviewed_alias_is_grounded_by_its_corpus_name(self):
        self.corpus.write_text(
            self.corpus.read_text(encoding="utf-8") + "药老正在指导萧炎。\n",
            encoding="utf-8",
        )
        evidence_values = [
            evidence_chunk(self.corpus, "药老正在指导萧炎。", line=3)
        ]
        item = record(
            "alias_grounding",
            TRAIN_SPLIT,
            self.corpus,
            answer="这段证据写到了药尘与萧炎。",
            dimension=GROUNDED_SINGLE,
            metric="normalized_f1",
            evidence_values=evidence_values,
            known_fact=False,
            needs_evidence=False,
            evidence_sufficient=True,
        )
        item["evaluation"]["required_terms"] = ["药尘"]
        report, risks = self.validate_small({TRAIN_SPLIT: [item]})
        self.assertEqual(report["status"], "passed")
        self.assertFalse(
            any(
                risk["code"] == "required_terms_not_grounded_in_evidence"
                for risk in risks
            )
        )

    def test_repeated_chapter_number_with_distinct_headings_is_not_a_leak(self):
        train_item = record(
            "chapter_train",
            TRAIN_SPLIT,
            self.corpus,
            dimension=GROUNDED_SINGLE,
            metric="normalized_f1",
            style="train_style",
            known_fact=False,
            needs_evidence=False,
            evidence_sufficient=True,
        )
        val_evidence = evidence_chunk(
            self.corpus, "药尘是萧炎的老师。", line=2
        )
        val_evidence["chapter_heading_line"] = 2
        val_evidence["chunk_sha256"] = stable_hash(
            "sft-v7-chunk",
            2,
            2,
            val_evidence["text_sha256"],
        )
        val_item = record(
            "chapter_val",
            VAL_SPLIT,
            self.corpus,
            answer="药尘是萧炎的老师。",
            dimension=GROUNDED_SINGLE,
            metric="normalized_f1",
            style="val_style",
            evidence_values=[val_evidence],
            known_fact=False,
            needs_evidence=False,
            evidence_sufficient=True,
        )
        report, _ = self.validate_small(
            {TRAIN_SPLIT: [train_item], VAL_SPLIT: [val_item]}
        )
        self.assertEqual(report["chapter_split_leaks"], 0)

    def test_project_concept_literal_eos_and_bad_encoding_are_p0_risks(self):
        item = record(
            "case_project",
            TRAIN_SPLIT,
            self.corpus,
            question="Token是什么？",
            answer="Token是模型处理文本的编号。<EOS>",
            dimension=VERTICAL_CHAT,
            metric="routing_accuracy",
            evidence_values=[],
            known_fact=False,
            needs_evidence=False,
            evidence_sufficient=False,
        )
        item["encoding_audit"]["eos_targets"] = 0
        report, _ = self.validate_small({TRAIN_SPLIT: [item]})
        codes = report["risk_code_counts"]
        self.assertEqual(codes["positive_project_concept_training"], 1)
        self.assertEqual(codes["literal_special_token_in_content"], 1)
        self.assertEqual(codes["encoding_eos_target_mismatch"], 1)

    def test_sealed_path_is_rejected_before_any_file_access_without_flag(self):
        with self.assertRaises(PermissionError):
            validate_dataset_files(
                train_path=self.root / "missing-train.jsonl",
                val_path=self.root / "missing-val.jsonl",
                public_path=self.root / "missing-public.jsonl",
                sealed_path=self.root / "SEALED_MUST_NOT_BE_TOUCHED.jsonl",
                allow_sealed_build_validation=False,
                corpus_path=self.corpus,
                tokenizer=None,
                enforce_release_gates=False,
            )

    def test_explicit_build_validation_manifest_contains_only_aggregates_and_hashes(self):
        paths = {}
        for index, split in enumerate(
            (TRAIN_SPLIT, VAL_SPLIT, PUBLIC_SPLIT, SEALED_SPLIT)
        ):
            path = self.root / f"{split}.jsonl"
            item = record(
                f"case_{split}",
                split,
                self.corpus,
                question=f"请说明萧炎的身份，编号{index}。",
                template=f"template_{split}",
                track="seen_fact_unseen_wording",
            )
            item["semantic_group"] = "shared_core_fact"
            if split == SEALED_SPLIT:
                item["answer"] += " SEALED_SECRET_BODY"
                item["messages"][-1]["content"] = item["answer"]
            write_jsonl(path, [item])
            paths[split] = path

        report, manifest, _ = validate_dataset_files(
            train_path=paths[TRAIN_SPLIT],
            val_path=paths[VAL_SPLIT],
            public_path=paths[PUBLIC_SPLIT],
            sealed_path=paths[SEALED_SPLIT],
            allow_sealed_build_validation=True,
            corpus_path=self.corpus,
            tokenizer=None,
            enforce_release_gates=False,
        )
        serialized_manifest = json.dumps(manifest, ensure_ascii=False)
        serialized_report = json.dumps(report, ensure_ascii=False)
        self.assertTrue(report["sealed_body_accessed"])
        self.assertEqual(manifest["files"][SEALED_SPLIT]["records"], 1)
        self.assertEqual(len(manifest["files"][SEALED_SPLIT]["sha256"]), 64)
        self.assertNotIn("SEALED_SECRET_BODY", serialized_manifest)
        self.assertNotIn("SEALED_SECRET_BODY", serialized_report)
        self.assertFalse(manifest["record_body_emitted"])


if __name__ == "__main__":
    unittest.main()
