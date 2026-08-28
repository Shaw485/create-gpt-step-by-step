import unittest
import json
from pathlib import Path
import tempfile

from evaluate_story_harness_v4 import (
    _load_records,
    apply_automatic_gates,
    longest_character_run,
    longest_corpus_overlap,
    ngram_repetition,
    sample_metrics,
    summarize_samples,
)


class StoryHarnessMetricTests(unittest.TestCase):
    def test_detects_repetition_and_character_runs(self):
        self.assertGreater(ngram_repetition("试试试试试试", 2), 0.0)
        self.assertEqual(longest_character_run("好试试试吧"), 3)

    def test_finds_longest_training_overlap(self):
        corpus = "夜色笼罩着山谷，少年缓缓前行。"
        self.assertEqual(longest_corpus_overlap("山谷，少年缓缓", corpus, minimum=2), 7)

    def test_sample_metrics_keep_semantic_judgment_separate(self):
        metrics = sample_metrics("萧炎一笑。\n\n众人退后。", "萧炎一笑。众人退后。")
        self.assertIn("four_gram_repetition", metrics)
        self.assertNotIn("coherence", metrics)
        self.assertEqual(metrics["paragraphs"], 2)

    def test_automatic_gates_veto_repetition_without_fake_composite_score(self):
        summary = {
            "sample_count": 5,
            "mean_characters": 100,
            "mean_han_ratio": 0.75,
            "mean_four_gram_repetition": 0.03,
            "maximum_character_run": 9,
            "maximum_train_overlap": 12,
        }
        decision = apply_automatic_gates(summary, prompt_count=5)
        self.assertEqual(decision["status"], "REVIEW")
        self.assertFalse(decision["checks"]["character_run"])

    def test_selected_checkpoint_sample_overrides_periodic_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            history = root / "history.json"
            selected = root / "selected.json"
            baseline.write_text(
                json.dumps({"checkpoint_step": 10, "samples": []}), encoding="utf-8"
            )
            history.write_text(
                json.dumps(
                    {
                        "history": [
                            {
                                "step": 20,
                                "samples": [{"prompt": "开头", "continuation": "旧样本"}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            selected.write_text(
                json.dumps(
                    {
                        "checkpoint_step": 20,
                        "samples": [{"prompt": "开头", "continuation": "发布候选样本"}],
                    }
                ),
                encoding="utf-8",
            )

            records = _load_records(baseline, history, selected, {"开头"})

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["continuation"], "发布候选样本")

    def test_summarizes_one_checkpoint_for_training_time_diagnostics(self):
        result = summarize_samples(
            [
                {"prompt": "开头一", "continuation": "少年缓缓走入山谷。" * 10},
                {"prompt": "开头二", "continuation": "夜色笼罩着安静城池。" * 10},
            ],
            "训练语料中没有这些完整续写。",
            prompt_count=2,
        )

        self.assertEqual(result["summary"]["sample_count"], 2)
        self.assertEqual(len(result["samples"]), 2)
        self.assertIn("automatic_gates", result["summary"])


if __name__ == "__main__":
    unittest.main()
