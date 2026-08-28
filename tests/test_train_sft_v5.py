from __future__ import annotations

import unittest

import torch

from train_sft_v5 import (
    WeightedEpochSampler,
    build_sampler,
    pool_for_record,
    selection_score,
)


def record(record_id: str, family: str) -> dict:
    return {
        "id": record_id,
        "task_family": family,
        "input_ids": torch.tensor([1, 2]),
        "labels": torch.tensor([-100, 2]),
    }


class TrainSftV5Tests(unittest.TestCase):
    def test_epoch_sampler_visits_every_record_before_repeat(self):
        records = [record(str(index), "direct_fact") for index in range(12)]
        sampler = WeightedEpochSampler(
            records,
            torch.Generator().manual_seed(7),
            {"all": 1.0},
            lambda _record: "all",
        )

        first_epoch = sampler.sample_indices(len(records))

        self.assertEqual(len(set(first_epoch)), len(records))
        self.assertEqual(sampler.coverage_summary()["coverage"], 1.0)

    def test_mixture_sampler_uses_exact_long_run_proportions(self):
        records = (
            [record(f"r{index}", "direct_fact") for index in range(20)]
            + [record(f"p{index}", "novel_core_entity_v5_2") for index in range(20)]
            + [record(f"k{index}", "novel_known_entity_v5_2") for index in range(20)]
            + [record(f"c{index}", "novel_unknown_grounded_v5_2") for index in range(20)]
        )
        sampler = build_sampler(
            records,
            torch.Generator().manual_seed(8),
            "mixture",
            0.45,
            0.35,
            0.15,
            0.05,
        )

        sampler.sample_indices(1000)

        self.assertEqual(
            sampler.pool_draw_counts,
            {"replay": 450, "core": 350, "known": 150, "contrast": 50},
        )

    def test_sampler_resume_reproduces_future_indices(self):
        records = [record(str(index), "direct_fact") for index in range(15)]
        first = WeightedEpochSampler(
            records,
            torch.Generator().manual_seed(9),
            {"all": 1.0},
            lambda _record: "all",
        )
        first.sample_indices(11)
        state = first.state_dict()
        generator_state = first.generator.get_state()
        expected = first.sample_indices(20)

        resumed_generator = torch.Generator()
        resumed_generator.set_state(generator_state)
        resumed = WeightedEpochSampler(
            records,
            resumed_generator,
            {"all": 1.0},
            lambda _record: "all",
        )
        resumed.load_state_dict(state)

        self.assertEqual(resumed.sample_indices(20), expected)

    def test_pool_routing(self):
        self.assertEqual(pool_for_record(record("1", "direct_fact")), "replay")
        self.assertEqual(
            pool_for_record(record("core", "novel_core_entity_v5_2")),
            "core",
        )
        self.assertEqual(
            pool_for_record(record("2", "novel_known_entity_v5_2")),
            "known",
        )
        self.assertEqual(
            pool_for_record(record("3", "novel_unknown_grounded_v5_2")),
            "contrast",
        )

    def test_selection_score_rewards_global_and_entity_behavior(self):
        base = {
            "passed": 10,
            "eos_count": 25,
            "by_category": {"小说人物": {"passed": 1}},
        }
        better_entity = {
            "passed": 11,
            "eos_count": 25,
            "by_category": {"小说人物": {"passed": 2}},
        }

        self.assertGreater(
            selection_score(better_entity, 2.0),
            selection_score(base, 2.0),
        )


if __name__ == "__main__":
    unittest.main()
