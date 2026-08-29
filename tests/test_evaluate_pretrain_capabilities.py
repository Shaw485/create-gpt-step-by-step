import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
import torch.nn.functional as functional

from bpe_tokenizer import BPETokenizer
from evaluate_pretrain_capabilities import (
    build_markdown_report,
    close_audit_loggers,
    configure_audit_loggers,
    deterministic_window_starts,
    evaluate_cloze,
    evaluate_held_out,
    generate_continuation,
    generation_diagnostics,
    main,
    parse_args,
    resolve_log_levels,
    summarize_generations,
    validate_args,
    validate_probe_artifact_for_evaluation,
    validate_pretraining_provenance,
    write_audit_outputs,
)
from training_runtime import canonical_json_sha256


class TransitionLanguageModel(torch.nn.Module):
    """Tiny deterministic next-token model used without a real checkpoint."""

    def __init__(self, transition_logits: torch.Tensor, block_size: int = 4):
        super().__init__()
        self.register_buffer("transition_logits", transition_logits.float())
        self.config = SimpleNamespace(
            block_size=block_size,
            vocab_size=transition_logits.shape[-1],
        )

    def forward(self, token_ids, target_ids=None):
        logits = self.transition_logits[token_ids]
        loss = None
        if target_ids is not None:
            loss = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1)
            )
        return logits, loss


def tiny_tokenizer() -> BPETokenizer:
    return BPETokenizer(
        tokens=["a", "b", "c", "\n", "<EOS>", "<PAD>"],
        merges=[],
        special_tokens=["<EOS>", "<PAD>"],
    )


def merged_candidate_tokenizer() -> BPETokenizer:
    return BPETokenizer(
        tokens=["a", "b", "c", "\n", "bb", "<EOS>", "<PAD>"],
        merges=[(1, 1, 4)],
        special_tokens=["<EOS>", "<PAD>"],
    )


class PretrainCapabilityAuditTests(unittest.TestCase):
    def test_deterministic_windows_cover_split_endpoints(self):
        self.assertEqual(deterministic_window_starts(10, 4, 3), [0, 2, 5])
        self.assertEqual(deterministic_window_starts(6, 4, 20), [0, 1])
        with self.assertRaises(ValueError):
            deterministic_window_starts(4, 4, 1)

    def test_held_out_loss_and_perplexity_are_finite_and_reproducible(self):
        transitions = torch.full((6, 6), -8.0)
        transitions[0, 1] = 8.0
        transitions[1, 0] = 8.0
        model = TransitionLanguageModel(transitions, block_size=4)
        model.train()
        data = torch.tensor([0, 1] * 8, dtype=torch.long)

        first = evaluate_held_out(
            model,
            data,
            window_count=4,
            batch_size=2,
            device=torch.device("cpu"),
        )
        second = evaluate_held_out(
            model,
            data,
            window_count=4,
            batch_size=2,
            device=torch.device("cpu"),
        )

        self.assertAlmostEqual(first["loss"], second["loss"])
        self.assertLess(first["perplexity"], 1.01)
        self.assertEqual(first["top1_accuracy"], 1.0)
        self.assertTrue(model.training)

    def test_validation_cross_entropy_is_nontrivial_and_exact_for_uniform_logits(self):
        model = TransitionLanguageModel(torch.zeros((6, 6)), block_size=3)
        data = torch.tensor([0, 1, 2, 0, 2, 1, 0, 1], dtype=torch.long)

        result = evaluate_held_out(
            model,
            data,
            window_count=3,
            batch_size=2,
            device=torch.device("cpu"),
        )

        self.assertAlmostEqual(result["loss"], torch.log(torch.tensor(6.0)).item())
        self.assertAlmostEqual(result["perplexity"], 6.0, places=5)
        self.assertGreater(result["loss"], 1.0)

    def test_generation_reports_eos_and_mechanical_empty_output(self):
        transitions = torch.zeros((6, 6))
        transitions[0, 4] = 10.0
        model = TransitionLanguageModel(transitions)

        result = generate_continuation(
            model,
            tiny_tokenizer(),
            "a",
            max_new_tokens=5,
            max_characters=10,
            temperature=1.0,
            top_k=1,
            generator=torch.Generator().manual_seed(1),
            device=torch.device("cpu"),
        )
        metrics = generation_diagnostics(result.continuation)

        self.assertEqual(result.stop_reason, "eos")
        self.assertTrue(result.eos_emitted)
        self.assertTrue(metrics["degeneration_flags"]["empty"])

    def test_generation_masks_all_special_tokens_except_eos(self):
        transitions = torch.zeros((6, 6))
        transitions[0, 5] = 100.0  # PAD would win without masking.
        transitions[0, 1] = 10.0
        model = TransitionLanguageModel(transitions)

        result = generate_continuation(
            model,
            tiny_tokenizer(),
            "a",
            max_new_tokens=1,
            max_characters=10,
            temperature=1.0,
            top_k=1,
            generator=torch.Generator().manual_seed(1),
            device=torch.device("cpu"),
        )

        self.assertEqual(result.continuation, "b")
        self.assertEqual(result.generated_token_ids, [1])

    def test_repetition_summary_does_not_treat_eos_as_quality_score(self):
        first = {
            "stop_reason": "eos",
            "eos_emitted": True,
            **generation_diagnostics("abca"),
        }
        second = {
            "stop_reason": "max_new_tokens",
            "eos_emitted": False,
            **generation_diagnostics("aaaaaaa"),
        }
        summary = summarize_generations([first, second])

        self.assertEqual(summary["eos_stop_rate"], 0.5)
        self.assertEqual(summary["stop_reason_counts"]["eos"], 1)
        self.assertGreater(summary["mechanical_degeneration_rate"], 0.0)

    def test_declarative_cloze_ranks_correct_candidate(self):
        transitions = torch.zeros((6, 6))
        transitions[0, 1] = 9.0
        transitions[0, 2] = -9.0
        transitions[3, 2] = 5.0
        model = TransitionLanguageModel(transitions)

        report = evaluate_cloze(
            model,
            tiny_tokenizer(),
            [
                {
                    "id": "relation",
                    "context": "a",
                    "candidates": ["b", "c"],
                    "correct": "b",
                    "frequency_tier": "high",
                }
            ],
            torch.device("cpu"),
        )

        self.assertEqual(report["top1_accuracy"], 1.0)
        self.assertEqual(report["cases"][0]["predicted"], "b")
        self.assertTrue(report["diagnostic_not_qa_hard_gate"])
        score = report["cases"][0]["scores"][0]
        self.assertIn("total_log_probability", score)
        self.assertIn("mean_token_log_probability", score)
        self.assertIn("per_character_log_probability", score)
        self.assertIn("context_lift", score)
        self.assertGreater(
            report["metrics"]["context_lift"]["top1_accuracy"], 0.0
        )
        self.assertEqual(report["by_frequency_tier"]["high"]["case_count"], 1)
        self.assertTrue(
            report["validation_prefix_ranking_not_entity_knowledge_hard_gate"]
        )

    def test_multitoken_candidate_formulas_are_reproducible_and_can_disagree(self):
        # Context row: c is likelier than the one-token/two-character BPE token
        # "bb".  Once a first c is emitted, another c is extremely likely.
        # Neutral newline strongly disfavors bb.  This intentionally makes the
        # four rankings disagree, so no implementation can silently collapse
        # length correction or prior correction into one opaque score.
        logits = torch.full((7, 7), -4.0)
        logits[0, 4] = 0.0  # P(bb | a)
        logits[0, 2] = 0.4  # P(c | a)
        logits[2, 2] = 3.0  # P(c | c)
        logits[3, :] = -4.0
        logits[3, 4] = -2.0  # P(bb | newline)
        logits[3, 2] = 0.0   # P(c | newline)
        model = TransitionLanguageModel(logits)
        tokenizer = merged_candidate_tokenizer()

        report = evaluate_cloze(
            model,
            tokenizer,
            [
                {
                    "id": "length-counterexample",
                    "context": "a",
                    "candidates": ["bb", "c", "cc"],
                    "correct": "cc",
                    "frequency_tier": "low",
                }
            ],
            torch.device("cpu"),
        )

        contextual = torch.log_softmax(logits, dim=-1)
        scores = {
            score["candidate"]: score for score in report["cases"][0]["scores"]
        }
        expected_cc_total = float(contextual[0, 2] + contextual[2, 2])
        expected_cc_neutral = float(contextual[3, 2] + contextual[2, 2])
        self.assertAlmostEqual(scores["cc"]["total_log_probability"], expected_cc_total)
        self.assertAlmostEqual(
            scores["cc"]["mean_token_log_probability"], expected_cc_total / 2
        )
        self.assertAlmostEqual(
            scores["cc"]["per_character_log_probability"], expected_cc_total / 2
        )
        self.assertAlmostEqual(
            scores["cc"]["context_lift"],
            expected_cc_total - expected_cc_neutral,
        )
        expected_bb_total = float(contextual[0, 4])
        self.assertEqual(scores["bb"]["token_count"], 1)
        self.assertEqual(scores["bb"]["character_count"], 2)
        self.assertAlmostEqual(
            scores["bb"]["per_character_log_probability"], expected_bb_total / 2
        )

        rankings = report["cases"][0]["rankings"]
        self.assertIn("candidate_token_count", report["formulae"]["mean_token_log_probability"])
        self.assertIn("neutral_newline_prefix", report["formulae"]["context_lift"])
        self.assertEqual(rankings["total_log_probability"]["predicted"], "c")
        self.assertEqual(
            rankings["mean_token_log_probability"]["predicted"], "cc"
        )
        self.assertEqual(
            rankings["per_character_log_probability"]["ordered_candidates"],
            ["cc", "bb", "c"],
        )
        self.assertEqual(rankings["context_lift"]["predicted"], "bb")

    def test_pretraining_provenance_rejects_post_training_checkpoint(self):
        model_config = {"vocab_size": 4, "block_size": 4}
        manifest_sha256 = "c" * 64
        valid = {
            "config_sha256": "a" * 64,
            "extra": {
                "model_config": model_config,
                "initial_checkpoint": None,
                "token_manifest_sha256": manifest_sha256,
            },
        }
        result = validate_pretraining_provenance(
            valid,
            expected_config_sha256="a" * 64,
            expected_model_config=model_config,
            expected_token_manifest_sha256=manifest_sha256,
        )
        self.assertTrue(result["post_training_markers_absent"])

        contaminated = {
            **valid,
            "extra": {
                "model_config": model_config,
                "payload_summary": {},
                "token_manifest_sha256": manifest_sha256,
            },
        }
        with self.assertRaisesRegex(ValueError, "post-training"):
            validate_pretraining_provenance(
                contaminated,
                expected_config_sha256="a" * 64,
                expected_model_config=model_config,
                expected_token_manifest_sha256=manifest_sha256,
            )
        with self.assertRaisesRegex(ValueError, "signature mismatch"):
            validate_pretraining_provenance(
                valid,
                expected_config_sha256="b" * 64,
                expected_model_config=model_config,
                expected_token_manifest_sha256=manifest_sha256,
            )

        wrong_manifest = {
            **valid,
            "extra": {**valid["extra"], "token_manifest_sha256": "d" * 64},
        }
        with self.assertRaisesRegex(ValueError, "token manifest"):
            validate_pretraining_provenance(
                wrong_manifest,
                expected_config_sha256="a" * 64,
                expected_model_config=model_config,
                expected_token_manifest_sha256=manifest_sha256,
            )
        continued = {
            **valid,
            "extra": {**valid["extra"], "initial_checkpoint": {"step": 1}},
        }
        with self.assertRaisesRegex(ValueError, "from-scratch"):
            validate_pretraining_provenance(
                continued,
                expected_config_sha256="a" * 64,
                expected_model_config=model_config,
                expected_token_manifest_sha256=manifest_sha256,
                require_initial_checkpoint_none=True,
            )

    def test_test_split_requires_an_explicit_allow_flag(self):
        with self.assertRaisesRegex(ValueError, "test split is sealed"):
            validate_args(parse_args(["--held-out-split", "test"]))

        allowed = parse_args(["--held-out-split", "test", "--allow-test"])
        validate_args(allowed)

    def test_formal_mode_requires_probe_artifact(self):
        with self.assertRaisesRegex(ValueError, "--formal requires --cloze"):
            validate_args(parse_args(["--formal"]))

    def test_probe_artifact_binds_usage_splits_cases_prompts_and_metadata(self):
        train_sha = "1" * 64
        val_sha = "2" * 64
        cloze_probe = {
            "id": "val-cloze-1",
            "probe_type": "cloze_candidate_ranking",
            "capability": "corpus_entity_prediction",
            "prompt": "a",
            "candidates": [
                {"text": "b", "frequency_tier": "high"},
                {"text": "c", "frequency_tier": "high"},
            ],
            "expected": {"text": "b", "candidate_index": 0},
            "entity": {"text": "b", "frequency_tier": "high"},
            "source": {"split": "val", "role": "held_out"},
            "evidence": {"sha256": "3" * 64},
        }
        continuation_probe = {
            "id": "val-continuation-1",
            "probe_type": "held_out_continuation",
            "capability": "novel_next_text_prediction",
            "prompt": "abc",
            "expected": {"continuation": "b"},
            "source": {"split": "val", "role": "held_out"},
            "evidence": {"sha256": "4" * 64},
        }
        compact_case = {
            "id": "val-cloze-1",
            "context": "a",
            "candidates": ["b", "c"],
            "correct": "b",
        }
        artifact = {
            "schema_version": "pretrain-capability-probes/v1",
            "usage": {
                "training_allowed": False,
                "sft_information_used": False,
                "test_split_read": False,
            },
            "build": {
                "inputs": {
                    "train": {"sha256": train_sha},
                    "val": {"sha256": val_sha},
                }
            },
            "validation": {"passed": True, "failures": []},
            "evaluator_compatibility": {
                "prompts_content_sha256": hashlib.sha256(b"abc\n").hexdigest(),
                "cases_canonical_sha256": canonical_json_sha256(
                    {"cases": [compact_case]}
                ),
            },
            "probes": [cloze_probe, continuation_probe],
            "cases": [compact_case],
            "continuation_prompts": ["abc"],
        }
        manifest = {
            "splits": {
                "train": {"text_sha256": train_sha},
                "val": {"text_sha256": val_sha},
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "probes.json"
            prompts_path = root / "prompts.txt"
            artifact_path.write_text(
                json.dumps(artifact, ensure_ascii=False), encoding="utf-8"
            )
            prompts_path.write_text("abc\n", encoding="utf-8")

            result = validate_probe_artifact_for_evaluation(
                artifact_path,
                prompts_path,
                manifest,
                require_formal_declarations=True,
            )
            self.assertTrue(result["formal_status_eligible"])
            self.assertEqual(result["cases"][0]["frequency_tier"], "high")
            self.assertEqual(result["cases"][0]["source"]["split"], "val")
            self.assertEqual(
                result["cases"][0]["candidate_metadata"][0]["text"], "b"
            )
            self.assertEqual(
                result["cases"][0]["probe_metadata"]["id"], "val-cloze-1"
            )
            self.assertEqual(result["validation_cloze_count"], 1)
            self.assertEqual(len(result["prompts_sha256"]), 64)
            self.assertTrue(result["formal_declarations_required"])

            prompts_path.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prompts file content"):
                validate_probe_artifact_for_evaluation(
                    artifact_path,
                    prompts_path,
                    manifest,
                    require_formal_declarations=True,
                )

    def test_probe_artifact_rejects_train_cases_and_unsafe_usage(self):
        base_artifact = {
            "schema_version": "pretrain-capability-probes/v1",
            "usage": {
                "training_allowed": False,
                "sft_information_used": False,
                "test_split_read": False,
            },
            "build": {
                "inputs": {
                    "train": {"sha256": "1" * 64},
                    "val": {"sha256": "2" * 64},
                }
            },
            "validation": {"passed": True, "failures": []},
            "probes": [],
            "cases": [
                {"id": "train-case", "context": "a", "candidates": ["b", "c"], "correct": "b"}
            ],
            "continuation_prompts": ["abc"],
        }
        manifest = {
            "splits": {
                "train": {"text_sha256": "1" * 64},
                "val": {"text_sha256": "2" * 64},
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "probes.json"
            prompts_path = root / "prompts.txt"
            prompts_path.write_text("abc\n", encoding="utf-8")
            artifact_path.write_text(json.dumps(base_artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation cloze probes"):
                validate_probe_artifact_for_evaluation(
                    artifact_path, prompts_path, manifest
                )

            unsafe = {
                **base_artifact,
                "cases": [],
                "usage": {**base_artifact["usage"], "training_allowed": True},
            }
            artifact_path.write_text(json.dumps(unsafe), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training_allowed=false"):
                validate_probe_artifact_for_evaluation(
                    artifact_path, prompts_path, manifest
                )

    def test_formal_probe_requires_hash_declarations_and_rejects_any_test_probe(self):
        manifest = {
            "splits": {
                "train": {"text_sha256": "1" * 64},
                "val": {"text_sha256": "2" * 64},
            }
        }
        val_continuation = {
            "id": "val-cont",
            "probe_type": "held_out_continuation",
            "prompt": "abc",
            "source": {"split": "val"},
        }
        base = {
            "schema_version": "pretrain-capability-probes/v1",
            "usage": {
                "training_allowed": False,
                "sft_information_used": False,
                "test_split_read": False,
            },
            "build": {
                "inputs": {
                    "train": {"sha256": "1" * 64},
                    "val": {"sha256": "2" * 64},
                }
            },
            "validation": {"passed": True, "failures": []},
            "probes": [val_continuation],
            "cases": [],
            "continuation_prompts": ["abc"],
            "evaluator_compatibility": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "probes.json"
            prompts_path = root / "prompts.txt"
            prompts_path.write_text("abc\n", encoding="utf-8")

            artifact_path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cases_canonical_sha256"):
                validate_probe_artifact_for_evaluation(
                    artifact_path,
                    prompts_path,
                    manifest,
                    require_formal_declarations=True,
                )

            missing_prompts_hash = {
                **base,
                "evaluator_compatibility": {
                    "cases_canonical_sha256": canonical_json_sha256({"cases": []})
                },
            }
            artifact_path.write_text(
                json.dumps(missing_prompts_hash), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "prompts_content_sha256"):
                validate_probe_artifact_for_evaluation(
                    artifact_path,
                    prompts_path,
                    manifest,
                    require_formal_declarations=True,
                )

            test_probe_artifact = {
                **missing_prompts_hash,
                "probes": [
                    val_continuation,
                    {
                        "id": "forbidden-test-probe",
                        "probe_type": "held_out_continuation",
                        "prompt": "secret",
                        "source": {"split": "test"},
                    },
                ],
                "evaluator_compatibility": {
                    **missing_prompts_hash["evaluator_compatibility"],
                    "prompts_content_sha256": hashlib.sha256(b"abc\n").hexdigest(),
                },
            }
            artifact_path.write_text(
                json.dumps(test_probe_artifact), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "test-split probe"):
                validate_probe_artifact_for_evaluation(
                    artifact_path,
                    prompts_path,
                    manifest,
                    require_formal_declarations=True,
                )

    def test_logging_is_per_module_rotating_and_redacts_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            levels = resolve_log_levels(["data=OFF", "orchestrator=INFO"])
            loggers = configure_audit_loggers(
                root,
                "test-run",
                levels,
                max_bytes=1024,
                backup_count=1,
                console=False,
            )
            loggers["data"].info("this must stay disabled")
            loggers["orchestrator"].info(
                "audit succeeded",
                extra={"context": {"api_key": "super-secret", "count": 1}},
            )
            close_audit_loggers(loggers)

            orchestrator_text = (root / "test-run.orchestrator.jsonl").read_text(
                encoding="utf-8"
            )
            data_text = (root / "test-run.data.jsonl").read_text(encoding="utf-8")
            payload = json.loads(orchestrator_text)
            self.assertEqual(payload["run_id"], "test-run")
            self.assertEqual(payload["context"]["api_key"], "[REDACTED]")
            self.assertNotIn("super-secret", orchestrator_text)
            self.assertEqual(data_text, "")

    def test_failure_path_is_actionable_and_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exit_code = main(
                [
                    "--config",
                    str(root / "missing-config.json"),
                    "--log-dir",
                    str(root / "logs"),
                    "--output-json",
                    str(root / "audit.json"),
                    "--output-markdown",
                    str(root / "audit.md"),
                    "--no-console-log",
                ]
            )
            self.assertEqual(exit_code, 1)
            log_path = next((root / "logs").glob("*.orchestrator.jsonl"))
            entries = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            payload = entries[-1]
            self.assertEqual(payload["level"], "ERROR")
            self.assertEqual(payload["context"]["operation"], "run_audit")
            self.assertIn("audit failed", payload["message"])
            self.assertIn("exception", payload)

    def test_json_and_markdown_outputs_state_base_lm_scope(self):
        report = {
            "schema_version": "pretrain-capability-audit/v1",
            "run_id": "test-run",
            "status": "AUDIT_COMPLETE_MANUAL_REVIEW_REQUIRED",
            "checkpoint": {"path": "base.pt", "step": 10},
            "device": "cpu",
            "model": {"parameter_count": 123},
            "validation_diagnostic": {
                "split": "val",
                "loss": 2.0,
                "perplexity": 7.389,
                "top1_accuracy": 0.2,
            },
            "probe_provenance": {
                "mode": "validated_artifact",
                "artifact_sha256": "a" * 64,
            },
            "generation_summary": {
                "eos_stop_rate": 0.0,
                "mechanical_degeneration_rate": 0.0,
            },
            "generations": [
                {
                    "prompt": "a",
                    "continuation": "b",
                    "stop_reason": "max_new_tokens",
                    "four_gram_repetition": 0.0,
                    "longest_character_run": 1,
                }
            ],
            "cloze": {
                "top1_accuracy": 1.0,
                "mean_reciprocal_rank": 1.0,
                "cases": [
                    {
                        "context": "a",
                        "correct": "b",
                        "predicted": "b",
                        "correct_rank": 1,
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "audit.json"
            markdown_path = root / "audit.md"
            write_audit_outputs(report, json_path, markdown_path)
            markdown = markdown_path.read_text(encoding="utf-8")
            payload = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], report["status"])
        self.assertIn("基础语言模型", markdown)
        self.assertIn("问答格式不属于本阶段硬门", markdown)
        self.assertIn("Validation diagnostic", markdown)
        self.assertNotIn("Held-out Loss", markdown)
        self.assertIn("validation-prefix ranking", markdown)
        self.assertIn("人工", markdown)
        self.assertEqual(markdown, build_markdown_report(report))


if __name__ == "__main__":
    unittest.main()
