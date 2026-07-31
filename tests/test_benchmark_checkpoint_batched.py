import random
import unittest

import numpy as np
import torch
import torch.nn as nn

from ai_observation import ACTION_COUNT, SUIT_ACTION_START
from batched_pesten import (
    ActorPolicy,
    BatchedPesten,
    PHASE_SUIT,
    POLICY_RANDOM,
)
from benchmark_checkpoint_batched import (
    FIXED_POOL,
    UNSEALED_SEED_LIMIT,
    _exact_non_neural_action,
    assemble_benchmark_payload,
    build_game_specs,
    run_batched_specs,
    run_canonical_specs,
)
from hybrid_agent import make_ppo_agent_factory


class LowestLegalActor(nn.Module):
    def forward(self, observations):
        logits = -torch.arange(
            ACTION_COUNT,
            dtype=observations.dtype,
            device=observations.device,
        )
        return logits.expand(len(observations), -1)


class HighestCardActor(nn.Module):
    def forward(self, observations):
        logits = torch.arange(
            ACTION_COUNT,
            dtype=observations.dtype,
            device=observations.device,
        ).expand(len(observations), -1).clone()
        logits[:, -1] = -100.0  # Prefer a card over voluntary pass.
        return logits


def policies_and_factories():
    device = torch.device("cpu")
    candidate_actor = LowestLegalActor()
    parent_actor = HighestCardActor()
    policies = {
        "candidate": ActorPolicy(
            candidate_actor,
            device,
            temperature=1.0,
            collect_data=False,
            greedy=True,
        ),
        "parent": ActorPolicy(
            parent_actor,
            device,
            temperature=1.0,
            collect_data=False,
            greedy=True,
        ),
    }
    factories = {
        "candidate": make_ppo_agent_factory(
            candidate_actor,
            device,
            shield=False,
        ),
        "parent": make_ppo_agent_factory(
            parent_actor,
            device,
            shield=False,
        ),
    }
    return policies, factories


class BatchedCheckpointBenchmarkTests(unittest.TestCase):
    def test_layout_uses_common_deals_and_rotates_both_seats(self):
        specs = build_game_specs(
            pairs=3,
            base_seed=71_331,
            pool_names=("random", "smart", "smarter"),
        )

        direct = [spec for spec in specs if spec.section == "direct"]
        self.assertEqual(len(direct), 6)
        for pair_index in range(3):
            pair = [
                spec for spec in direct if spec.pair_index == pair_index
            ]
            self.assertEqual(
                [spec.evaluated_seat for spec in pair],
                [0, 1],
            )
            self.assertEqual(len({spec.deal_seed for spec in pair}), 1)

        random_pair = [
            spec
            for spec in specs
            if spec.section == "pool"
            and spec.opponent == "random"
            and spec.pair_index == 0
        ]
        self.assertEqual(len(random_pair), 4)
        self.assertEqual(
            {
                (spec.evaluated_key, spec.evaluated_seat)
                for spec in random_pair
            },
            {
                ("candidate", 0),
                ("candidate", 1),
                ("parent", 0),
                ("parent", 1),
            },
        )
        self.assertEqual(len({spec.deal_seed for spec in random_pair}), 1)
        self.assertTrue(
            all(0 <= spec.deal_seed < UNSEALED_SEED_LIMIT for spec in specs)
        )

    def test_random_suit_choice_uses_canonical_python_rng_stream(self):
        seed = 88_123
        engine = BatchedPesten(
            np.full((1, 2), POLICY_RANDOM, dtype=np.int8),
            np.full((1, 2), None, dtype=object),
            np.asarray([seed], dtype=np.uint32),
        )
        engine.phase[0] = PHASE_SUIT
        expected_rng = random.Random(seed)
        actual_rng = random.Random(seed)

        for _ in range(12):
            expected = SUIT_ACTION_START + expected_rng.randint(0, 3)
            actual = _exact_non_neural_action(engine, 0, actual_rng)
            self.assertEqual(actual, expected)

    def test_batched_outcomes_match_canonical_game_exactly(self):
        policies, factories = policies_and_factories()
        specs = build_game_specs(
            pairs=10,
            base_seed=2_607_311,
            pool_names=("random", "smart", "smarter"),
        )

        batched, statistics = run_batched_specs(
            specs,
            policies,
            max_turns=500,
        )
        canonical, _ = run_canonical_specs(
            specs,
            candidate_factory=factories["candidate"],
            parent_factory=factories["parent"],
            max_turns=500,
        )

        self.assertEqual(batched, canonical)
        self.assertEqual(statistics["games"], len(specs))
        self.assertGreater(statistics["policy_decisions"], 0)
        self.assertLess(
            statistics["inference_calls"],
            statistics["policy_decisions"],
        )
        self.assertGreater(statistics["mean_actor_batch_size"], 1.0)

    def test_payload_uses_checkpoint_benchmark_score_and_delta_semantics(self):
        policies, _ = policies_and_factories()
        pairs = 5
        base_seed = 4_210_777
        specs = build_game_specs(
            pairs=pairs,
            base_seed=base_seed,
            pool_names=("random", "smart", "smarter"),
        )
        outcomes, _ = run_batched_specs(specs, policies, max_turns=500)

        payload = assemble_benchmark_payload(
            specs,
            outcomes,
            pairs=pairs,
            base_seed=base_seed,
            pool_names=("random", "smart", "smarter"),
            bootstrap_samples=100,
        )

        direct_scores = [
            record["score"] for record in payload["direct"]["per_pair"]
        ]
        self.assertAlmostEqual(
            payload["direct"]["candidate_score"]["mean"],
            float(np.mean(direct_scores)),
        )
        self.assertAlmostEqual(
            payload["direct"]["candidate_advantage"]["mean"],
            float(np.mean([2.0 * score - 1.0 for score in direct_scores])),
        )
        all_pool_deltas = []
        for matchup in payload["pool"]:
            deltas = [record["delta"] for record in matchup["per_pair"]]
            all_pool_deltas.extend(deltas)
            self.assertAlmostEqual(
                matchup["paired_delta"]["mean"],
                float(np.mean(deltas)),
            )
            for record in matchup["per_pair"]:
                self.assertEqual(
                    record["delta"],
                    record["candidate"]["score"]
                    - record["parent"]["score"],
                )
        self.assertAlmostEqual(
            payload["combined_pool_delta"]["mean"],
            float(np.mean(all_pool_deltas)),
        )
        self.assertEqual(
            set(FIXED_POOL),
            {"random", "smart", "smarter"},
        )


if __name__ == "__main__":
    unittest.main()
