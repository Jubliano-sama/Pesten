import unittest

import torch

from batched_pesten import ActorPolicy
from benchmark_multiplayer_symmetry import assemble, build_specs, run_specs


class MultiplayerSymmetryBenchmarkTests(unittest.TestCase):
    def test_specs_rotate_each_variant_through_every_seat(self):
        specs = build_specs(deals=2, players=4, base_seed=89_000_001)
        self.assertEqual(len(specs), 16)
        for variant in ("suit_only", "player_orbit"):
            selected = [spec for spec in specs if spec.variant == variant]
            self.assertEqual(
                sorted(spec.evaluated_seat for spec in selected[:4]),
                [0, 1, 2, 3],
            )
            for spec in selected:
                self.assertEqual(spec.actor_keys.count(variant), 1)
                self.assertEqual(spec.actor_keys.count("parent"), 3)

    def test_rejects_non_development_seed_or_player_count(self):
        with self.assertRaises(ValueError):
            build_specs(deals=1, players=1, base_seed=1)
        with self.assertRaises(ValueError):
            build_specs(deals=1, players=8, base_seed=1)
        with self.assertRaises(ValueError):
            build_specs(deals=2, players=4, base_seed=89_999_999)

    def test_draw_share_matches_multiplayer_null_share(self):
        class PassActor(torch.nn.Module):
            def forward(self, observations):
                logits = torch.full(
                    (len(observations), 55),
                    -1_000.0,
                    dtype=observations.dtype,
                    device=observations.device,
                )
                logits[:, 54] = 1_000.0
                return logits

        specs = build_specs(deals=1, players=3, base_seed=89_100_001)
        policy = ActorPolicy(
            PassActor(),
            torch.device("cpu"),
            temperature=1.0,
            collect_data=False,
            greedy=True,
        )
        scores, _ = run_specs(
            specs,
            {
                "parent": policy,
                "suit_only": policy,
                "player_orbit": policy,
            },
            max_turns=1,
        )
        self.assertTrue(all(score == 1.0 / 3.0 for score in scores))

    def test_assemble_uses_per_deal_seat_rotations(self):
        specs = build_specs(deals=2, players=2, base_seed=89_000_001)
        scores = []
        for spec in specs:
            if spec.variant == "suit_only":
                scores.append(float(spec.evaluated_seat == 0))
            else:
                scores.append(1.0)
        result = assemble(
            specs,
            scores,
            deals=2,
            players=2,
            base_seed=89_000_001,
            bootstrap_samples=100,
        )
        self.assertEqual(result["suit_only_score"]["mean"], 0.5)
        self.assertEqual(result["player_orbit_score"]["mean"], 1.0)
        self.assertEqual(
            result["paired_player_orbit_minus_suit_only"]["mean"], 0.5
        )


if __name__ == "__main__":
    unittest.main()
