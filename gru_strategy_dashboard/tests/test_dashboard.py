from __future__ import annotations

import unittest

from gru_strategy_dashboard.analyze import (
    RANK_NAMES,
    StrategyAccumulator,
    rank_for_card,
    wilson_interval,
)
from gru_strategy_dashboard.artifact import build_artifact


class RankTests(unittest.TestCase):
    def test_all_card_ids_collapse_to_fourteen_numbers(self) -> None:
        grouped = {rank_for_card(card_id) for card_id in range(50)}
        self.assertEqual(grouped, set(RANK_NAMES))
        for suit in range(4):
            self.assertEqual(rank_for_card(suit * 12), 1)
            self.assertEqual(rank_for_card(suit * 12 + 11), 12)
        self.assertEqual(rank_for_card(48), 13)
        self.assertEqual(rank_for_card(49), 0)

    def test_wilson_interval_contains_observed_rate(self) -> None:
        low, high = wilson_interval(70, 100)
        self.assertLess(low, 0.7)
        self.assertGreater(high, 0.7)


class AccumulatorTests(unittest.TestCase):
    def test_playing_one_number_keeps_other_legal_numbers(self) -> None:
        accumulator = StrategyAccumulator()
        observation = [0.0] * 94
        observation[0] = 1.0
        observation[1] = 1.0
        observation[69] = 2.0
        observation[70] = 4.0
        observation[91] = 2.0
        observation[92] = 2.0
        legal = [False] * 55
        legal[0] = True
        legal[1] = True
        legal[54] = True
        accumulator.add_decision(
            mode="natural",
            player_count=2,
            game_index=0,
            seat=0,
            decision_number=1,
            observation=observation,
            legal_actions=legal,
            action=0,
            top_probability=0.7,
            chosen_probability=0.7,
            uncertainty=0.5,
            value=0.1,
            memory_changed=False,
        )
        rows = accumulator.final_rows()["rank_choices"]
        all_players = {
            row["rank"]: row
            for row in rows
            if row["action_style"] == "Varied choices"
            and row["view_players"] == "All table sizes"
        }
        self.assertEqual(all_players["Ace"]["used"], 1)
        self.assertEqual(all_players["2"]["held"], 1)


class ArtifactTests(unittest.TestCase):
    def test_artifact_has_filters_charts_and_model_shape(self) -> None:
        accumulator = StrategyAccumulator()
        rows = accumulator.final_rows()
        summary = {
            "generated_at": "2026-08-12T12:00:00Z",
            "checkpoint": {
                "path": "live_training/teacher/latest.pt",
                "sha256": "0" * 64,
                "epoch": 4097,
                "model": {"hidden_size": 192, "num_layers": 2},
            },
            "protocol": {
                "player_counts": list(range(2, 9)),
                "games_per_player_count_per_style": 1,
                "action_styles": ["Varied choices", "Always picks its favorite"],
                "hold_definition": "test hold definition",
                "memory_test": "test memory definition",
                "memory_layer_test": "test layer definition",
                "input_pull_test": "test input definition",
                "memory_map": "test map definition",
                "memory_readout": "test readout definition",
                "interpretability_samples": 0,
                "base_seed": 1,
                "device": "cpu",
                "max_turns": 100,
                "torch_threads": 1,
            },
            "findings": ["Test finding."],
            "datasets": {
                **rows,
                "input_groups": [{"part": "Input", "values": 94, "meaning": "test"}],
                "output_groups": [{"part": "Output", "values": 55, "meaning": "test"}],
                "model_shape": [{"stage": "Input", "exact_shape": "batch x 94", "plain_meaning": "test"}],
            },
        }
        artifact = build_artifact(summary)
        self.assertEqual(artifact["surface"], "dashboard")
        self.assertEqual(len(artifact["manifest"]["filters"]), 2)
        self.assertEqual(len(artifact["manifest"]["charts"]), 5)
        self.assertEqual(len(artifact["manifest"]["tables"]), 1)
        self.assertIn("model_shape", artifact["snapshot"]["datasets"])
        self.assertIn("memory_map", artifact["snapshot"]["datasets"])


if __name__ == "__main__":
    unittest.main()
