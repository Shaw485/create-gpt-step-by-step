from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from bpe_tokenizer import BPETokenizer
import evaluate_sft_v7_public as evaluate


def tiny_tokenizer() -> BPETokenizer:
    base = ["你", "问", "答", "甲", "乙", "。"]
    specials = ["<UNK>", "<BOS>", "<USER>", "<ASSISTANT>", "<EOS>", "<PAD>"]
    return BPETokenizer(base + specials, [], specials)


def evaluation_metadata(**updates) -> dict:
    value = {
        "metric": "required_and_keypoints",
        "required_terms": ["答"],
        "forbidden_terms": ["错"],
        "keypoints": [["答", "回答"]],
        "known_fact": True,
        "needs_evidence": False,
        "evidence_sufficient": True,
    }
    value.update(updates)
    return value


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logits: list[float]):
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor(logits, dtype=torch.float32))
        self.config = SimpleNamespace(block_size=32)

    def forward(self, inputs: torch.Tensor):
        logits = self.fixed_logits.view(1, 1, -1).expand(
            inputs.shape[0], inputs.shape[1], -1
        )
        return logits, None


class AnswerThenEosModel(torch.nn.Module):
    def __init__(self, vocab_size: int, answer_id: int, eos_id: int, unk_id: int):
        super().__init__()
        self.config = SimpleNamespace(block_size=64)
        self.vocab_size = vocab_size
        self.answer_id = answer_id
        self.eos_id = eos_id
        self.unk_id = unk_id

    def forward(self, inputs: torch.Tensor):
        logits = torch.full((*inputs.shape, self.vocab_size), -10.0, device=inputs.device)
        for row in range(inputs.shape[0]):
            if int(inputs[row, -1]) == self.answer_id:
                logits[row, -1, self.eos_id] = 10.0
            else:
                logits[row, -1, self.unk_id] = 12.0
                logits[row, -1, self.answer_id] = 11.0
        return logits, None


class EvaluateSftV7PublicTests(unittest.TestCase):
    def test_teacher_forced_loss_is_token_weighted_and_nontrivial(self):
        model = FixedLogitModel([0.2, 1.0, -0.4])
        records = [
            {
                "input_ids": torch.tensor([0, 1], dtype=torch.long),
                "labels": torch.tensor([1, 2], dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([2], dtype=torch.long),
                "labels": torch.tensor([1], dtype=torch.long),
            },
        ]
        result = evaluate.evaluate_teacher_forced_loss(
            model, records, pad_token_id=0, batch_size=1, device=torch.device("cpu")
        )
        logits = torch.tensor([[0.2, 1.0, -0.4]]).expand(3, -1)
        expected = float(F.cross_entropy(logits, torch.tensor([1, 2, 1])))

        self.assertEqual(result["records"], 2)
        self.assertEqual(result["supervised_tokens"], 3)
        self.assertAlmostEqual(result["loss"], expected, places=6)
        self.assertGreater(result["loss"], 0.1)
        self.assertAlmostEqual(result["perplexity"], math.exp(expected), places=6)

    def test_scoring_covers_terms_keypoints_known_refusal_and_generation_quality(self):
        passed = evaluate.score_generated_answer(
            "答。",
            evaluation_metadata(),
            stopped_on_eos=True,
            truncated=False,
            dimension="core_facts_and_corrections",
        )
        self.assertTrue(passed["required_case_pass"])
        self.assertTrue(passed["forbidden_case_pass"])
        self.assertTrue(passed["keypoint_case_pass"])
        self.assertFalse(passed["known_fact_misrefusal"])
        self.assertTrue(passed["stopped_on_eos"])

        refused = evaluate.score_generated_answer(
            "现有资料不足，无法确认。可以先记录问题。",
            evaluation_metadata(),
            stopped_on_eos=False,
            truncated=True,
            dimension="core_facts_and_corrections",
        )
        self.assertTrue(refused["known_fact_misrefusal"])
        self.assertTrue(refused["meta_phrase"])
        self.assertTrue(refused["truncated"])

        repeated = evaluate.score_generated_answer(
            "甲乙甲乙甲乙甲乙甲乙甲乙",
            evaluation_metadata(required_terms=[], forbidden_terms=[], keypoints=[]),
            stopped_on_eos=True,
            truncated=False,
            dimension="novel_expression",
        )
        self.assertTrue(repeated["mechanical_repetition"])
        self.assertEqual(
            repeated["open_expression_review"],
            "ai_assisted_and_independent_human_review_pending",
        )

    def test_insufficient_evidence_and_boundary_recovery_are_separate(self):
        insufficient = evaluate.score_generated_answer(
            "证据不足，无法确认。",
            evaluation_metadata(
                required_terms=[],
                forbidden_terms=[],
                keypoints=[],
                known_fact=False,
                needs_evidence=True,
                evidence_sufficient=False,
            ),
            stopped_on_eos=True,
            truncated=False,
            dimension="capability_boundary",
        )
        recovered = evaluate.score_generated_answer(
            "根据片段可以回答。",
            evaluation_metadata(
                required_terms=["回答"],
                forbidden_terms=[],
                keypoints=["回答"],
                known_fact=False,
                needs_evidence=True,
                evidence_sufficient=True,
            ),
            stopped_on_eos=True,
            truncated=False,
            dimension="capability_boundary",
        )
        summary = evaluate.summarize_scores([insufficient, recovered])

        self.assertTrue(insufficient["insufficient_evidence_stopped"])
        self.assertTrue(recovered["boundary_recovered"])
        self.assertEqual(summary["insufficient_evidence"]["correct_stop_rate"], 1.0)
        self.assertEqual(summary["boundary_recovery"]["recovery_rate"], 1.0)

    def test_empty_answer_never_counts_as_sufficient_or_recovered(self):
        empty = evaluate.score_generated_answer(
            " 。。。 ",
            evaluation_metadata(
                required_terms=[],
                forbidden_terms=[],
                keypoints=[],
                known_fact=False,
                needs_evidence=True,
                evidence_sufficient=True,
                capability_mode="grounded_answer",
                calibration_triplet_id="triplet-1",
            ),
            stopped_on_eos=True,
            truncated=False,
            dimension="single_evidence_qa",
        )

        self.assertTrue(empty["empty_answer"])
        self.assertTrue(empty["sufficient_evidence_case"])
        self.assertFalse(empty["sufficient_evidence_answered"])
        self.assertTrue(empty["boundary_recovery_case"])
        self.assertFalse(empty["boundary_recovered"])

    def test_normalized_exact_and_character_multiset_f1_are_explicit_proxies(self):
        exact = evaluate.score_generated_answer(
            "甲，乙！",
            evaluation_metadata(metric="exact"),
            reference_answer="甲乙",
            stopped_on_eos=True,
            truncated=False,
            dimension="novel_expression",
        )
        f1 = evaluate.score_generated_answer(
            "甲甲乙",
            evaluation_metadata(metric="normalized_f1"),
            reference_answer="甲乙乙",
            stopped_on_eos=True,
            truncated=False,
            dimension="single_evidence_qa",
        )

        self.assertEqual(exact["normalized_exact_match"], 1.0)
        self.assertIsNone(exact["normalized_char_multiset_f1"])
        self.assertAlmostEqual(f1["normalized_char_multiset_f1"], 2 / 3)
        self.assertIn("lexical proxy", evaluate.NORMALIZED_CHAR_F1_DEFINITION)

    def test_automatic_gates_can_pass_but_external_pending_blocks_candidate(self):
        def scored(
            dimension: str,
            family: str,
            answer: str = "答",
            **metadata_updates,
        ) -> dict:
            metadata = evaluation_metadata(**metadata_updates)
            score = evaluate.score_generated_answer(
                answer,
                metadata,
                reference_answer="答",
                stopped_on_eos=True,
                truncated=False,
                dimension=dimension,
            )
            return {"dimension": dimension, "task_family": family, **score}

        rows = [
            scored("core_facts_and_corrections", "known_core_direct"),
            scored(
                "single_evidence_qa",
                "passage_answer",
                metric="normalized_f1",
                known_fact=False,
                capability_mode="grounded_answer",
            ),
            scored(
                "single_evidence_qa",
                "passage_insufficient",
                answer="证据不足，无法确认。",
                required_terms=[],
                keypoints=[],
                known_fact=False,
                needs_evidence=True,
                evidence_sufficient=False,
            ),
            scored(
                "rag_evidence_composition",
                "rag_compose",
                known_fact=False,
                capability_mode="grounded_answer",
            ),
            scored(
                "vertical_chat_multiturn_eos",
                "chat_multiturn",
                known_fact=False,
                capability_mode="interaction",
            ),
            scored(
                "novel_expression",
                "summary",
                known_fact=False,
                required_terms=[],
                keypoints=[],
                capability_mode="expression",
            ),
            scored(
                "capability_boundary",
                "boundary_need_evidence",
                answer="证据不足，无法确认。",
                required_terms=[],
                keypoints=[],
                known_fact=False,
                needs_evidence=True,
                evidence_sufficient=False,
            ),
            scored(
                "single_evidence_qa",
                "passage_answer",
                known_fact=False,
                needs_evidence=False,
                evidence_sufficient=True,
                capability_mode="grounded_answer",
                calibration_triplet_id="triplet-1",
            ),
        ]
        gates = evaluate.build_gate_results(rows)

        self.assertTrue(gates["automatic_gates_passed"])
        self.assertTrue(all(gate["status"] == "pending" for gate in gates["external_gates"]))
        self.assertFalse(gates["external_gates_passed"])
        self.assertFalse(gates["candidate_eligible"])

    def test_one_failed_automatic_gate_makes_candidate_ineligible(self):
        row = {
            "dimension": "vertical_chat_multiturn_eos",
            "task_family": "chat_multiturn",
            **evaluate.score_generated_answer(
                "",
                evaluation_metadata(known_fact=False, capability_mode="interaction"),
                stopped_on_eos=False,
                truncated=True,
                dimension="vertical_chat_multiturn_eos",
            ),
        }
        gates = evaluate.build_gate_results([row])
        empty_gate = next(
            gate for gate in gates["automatic_gates"] if gate["id"] == "chat_empty"
        )

        self.assertEqual(empty_gate["value"], 1.0)
        self.assertFalse(empty_gate["passed"])
        self.assertFalse(gates["automatic_gates_passed"])
        self.assertFalse(gates["candidate_eligible"])

    def test_public_pair_binds_id_dimension_family_and_evaluation(self):
        metadata = evaluation_metadata()
        source = {
            "id": "p1",
            "split": "public_diagnostic",
            "primary_dimension": "parameter_core_fact_and_correction",
            "task_family": "core",
            "answer": "答",
            "messages": [
                {"role": "user", "content": "你问"},
                {"role": "assistant", "content": "答"},
            ],
            "evaluation": metadata,
        }
        tensor = {
            "id": "p1",
            "split": "public_diagnostic",
            "primary_dimension": "parameter_core_fact_and_correction",
            "task_family": "core",
            "input_ids": torch.tensor([0]),
            "labels": torch.tensor([2]),
            "evaluation": metadata,
        }
        paired = evaluate.validate_public_pair(
            [source], {"public_records": [tensor]}, expected_count=1
        )
        self.assertEqual(paired[0]["dimension"], "core_facts_and_corrections")

        changed = dict(tensor)
        changed["evaluation"] = evaluation_metadata(required_terms=["乙"])
        with self.assertRaisesRegex(
            evaluate.SFTV7PublicEvaluationError, "scoring metadata"
        ):
            evaluate.validate_public_pair(
                [source], {"public_records": [changed]}, expected_count=1
            )

    def test_jsonl_rejects_forbidden_field_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "public_diagnostic.jsonl"
            record = {
                "id": "p1",
                "split": "public_diagnostic",
                "sealed_test": {"answer": "never"},
            }
            path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "forbidden field"):
                evaluate.read_public_jsonl(path)

    def test_stable_case_seed_is_order_independent(self):
        self.assertEqual(
            evaluate.stable_case_seed(42, "public-7"),
            evaluate.stable_case_seed(42, "public-7"),
        )
        self.assertNotEqual(
            evaluate.stable_case_seed(42, "public-7"),
            evaluate.stable_case_seed(42, "public-8"),
        )

    def test_public_report_uses_no_sequence_matcher_hard_gate(self):
        source = Path(evaluate.__file__).read_text(encoding="utf-8")
        self.assertNotIn("SequenceMatcher", source)
        self.assertNotIn("mean_similarity", source)

    def test_cli_success_limits_samples_and_logs_no_bodies(self):
        tokenizer = tiny_tokenizer()
        special = tokenizer.special_to_id
        answer_id = tokenizer.char_to_id["答"]
        model = AnswerThenEosModel(
            tokenizer.vocab_size, answer_id, special["<EOS>"], special["<UNK>"]
        )
        paired = []
        for index, dimension in enumerate(evaluate.CANONICAL_DIMENSIONS):
            metadata = evaluation_metadata()
            source = {
                "id": f"p{index}",
                "messages": [
                    {"role": "user", "content": "你问"},
                    {"role": "assistant", "content": "答"},
                ],
            }
            tensor = {
                "id": f"p{index}",
                "input_ids": torch.tensor([tokenizer.char_to_id["你"]]),
                "labels": torch.tensor([answer_id]),
            }
            paired.append(
                {
                    "source": source,
                    "tensor": tensor,
                    "dimension": dimension,
                    "task_family": f"family-{index}",
                    "evaluation": metadata,
                }
            )
        payload = {
            "special_token_ids": special,
            "tokenizer_sha256": "a" * 64,
            "bpe_token_manifest_sha256": "b" * 64,
            "sft_dataset_manifest_sha256": "e" * 64,
        }
        config = {
            "logging": {"max_bytes": 8192, "backup_count": 1, "console": False}
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "report.json"
            markdown_path = root / "report.md"
            logs = root / "logs"
            argv = [
                "--public-jsonl",
                str(root / "public_diagnostic.jsonl"),
                "--public-tensors",
                str(root / "public_diagnostic_tensors.pt"),
                "--checkpoint",
                str(root / "checkpoint.pt"),
                "--report",
                str(report_path),
                "--markdown",
                str(markdown_path),
                "--log-dir",
                str(logs),
                "--device",
                "cpu",
                "--top-k",
                "1",
                "--samples-per-dimension",
                "1",
            ]
            with (
                mock.patch.object(evaluate, "load_config", return_value=config),
                mock.patch.object(
                    evaluate,
                    "load_and_validate_public_inputs",
                    return_value=(paired, payload),
                ),
                mock.patch.object(evaluate, "load_bound_tokenizer", return_value=tokenizer),
                mock.patch.object(evaluate, "select_device", return_value=torch.device("cpu")),
                mock.patch.object(
                    evaluate,
                    "load_model_bundle",
                    return_value=(
                        model,
                        {"step": 20},
                        {"step": 20, "base_checkpoint_sha256": "c" * 64},
                    ),
                ),
                mock.patch.object(evaluate, "file_sha256", return_value="d" * 64),
            ):
                self.assertEqual(evaluate.main(argv), 0)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                report["status"],
                "diagnostic_complete_candidate_pending_or_ineligible",
            )
            self.assertEqual(report["teacher_forced"]["records"], 6)
            self.assertEqual(len(report["samples"]), 6)
            self.assertEqual(set(report["dimensions"]), set(evaluate.CANONICAL_DIMENSIONS))
            self.assertEqual(report["open_expression_review"]["ai_assisted_quality_review"], "pending")
            self.assertIn("automatic_gates", report)
            self.assertIn("external_gates", report)
            self.assertFalse(report["candidate_eligible"])
            self.assertEqual(report["generation_configuration"]["generation_batch_size"], 8)
            self.assertEqual(
                report["generation_configuration"]["batching_policy"],
                "nearby_length_sorted_left_padded_bounded_chunks",
            )
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("自动候选硬门", markdown)
            self.assertIn("外部待审硬门", markdown)
            log_text = "".join(path.read_text(encoding="utf-8") for path in logs.glob("*.jsonl"))
            self.assertNotIn("你问", log_text)
            self.assertNotIn("答", log_text)
            self.assertIn('"module": "cloud.generation"', log_text)
            self.assertIn('"module": "cloud.evaluation"', log_text)
            self.assertIn("generation_batch_size=8", log_text)

    def test_cli_failure_log_contains_code_but_no_source_body(self):
        config = {
            "logging": {"max_bytes": 4096, "backup_count": 1, "console": False}
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root / "logs"
            argv = [
                "--public-jsonl",
                str(root / "public_diagnostic.jsonl"),
                "--public-tensors",
                str(root / "public_diagnostic_tensors.pt"),
                "--checkpoint",
                str(root / "checkpoint.pt"),
                "--log-dir",
                str(logs),
            ]
            with (
                mock.patch.object(evaluate, "load_config", return_value=config),
                mock.patch.object(
                    evaluate,
                    "load_and_validate_public_inputs",
                    side_effect=evaluate.SFTV7PublicEvaluationError(
                        "public_sha", "safe failure"
                    ),
                ),
            ):
                with self.assertRaises(evaluate.SFTV7PublicEvaluationError):
                    evaluate.main(argv)
            log_text = "".join(path.read_text(encoding="utf-8") for path in logs.glob("*.jsonl"))
            self.assertIn("error_code=public_sha", log_text)
            self.assertNotIn("小说正文秘密", log_text)


if __name__ == "__main__":
    unittest.main()
