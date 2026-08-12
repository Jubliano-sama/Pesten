import numpy as np
import pytest

from benchmark_action_head_slope_batched import (
    _endpoint_advantage_summary,
    _make_layout,
    _slope_advantage_summary,
    _slope_values,
)
from benchmark_rlvr_pool import OPPONENTS, PLAYER_WEIGHTS


def test_layout_uses_common_deals_for_every_checkpoint_and_seat():
    labels = ("full_0", "full_60", "factorized_0", "factorized_60")
    records, roles, seeds = _make_layout(
        {
            "players": 3,
            "blocks": 2,
            "opponent": "self_anchor",
            "labels": labels,
            "seed": 800,
        }
    )

    assert len(records) == 2 * len(labels) * 3
    for block in range(2):
        start = block * len(labels) * 3
        stop = (block + 1) * len(labels) * 3
        assert set(seeds[start:stop]) == {800 + block}
    for row, (_, label, candidate_seat) in enumerate(records):
        assert roles[row][candidate_seat] == ("temporal", label)
        assert sum(role == ("temporal", "self_anchor") for role in roles[row]) == 2


def test_slope_values_fit_each_matched_block_independently():
    row = {
        "scores": {
            "phase_0": [0.2, 0.4],
            "phase_2": [0.4, 0.3],
            "phase_5": [0.7, 0.15],
        }
    }

    slopes = _slope_values(
        row,
        ["phase_0", "phase_2", "phase_5"],
        [0, 2, 5],
    )

    assert slopes[0] == pytest.approx(0.1)
    assert slopes[1] == pytest.approx(-0.05)


def test_paired_slope_and_endpoint_advantage_keep_player_weights():
    phases = [0, 1, 2]
    full_labels = [f"full_{phase}" for phase in phases]
    factorized_labels = [f"factorized_{phase}" for phase in phases]
    raw = []
    for opponent in OPPONENTS:
        for players in PLAYER_WEIGHTS:
            raw.append(
                {
                    "opponent": opponent,
                    "players": players,
                    "scores": {
                        **{
                            label: [0.2 + 0.01 * phase] * 3
                            for label, phase in zip(full_labels, phases)
                        },
                        **{
                            label: [0.2 + 0.02 * phase] * 3
                            for label, phase in zip(
                                factorized_labels, phases
                            )
                        },
                    },
                }
            )

    slope = _slope_advantage_summary(
        raw, factorized_labels, full_labels, phases
    )
    endpoint = _endpoint_advantage_summary(
        raw, factorized_labels, full_labels
    )
    weighted_players = sum(
        players * weight for players, weight in PLAYER_WEIGHTS.items()
    )

    assert slope["score"]["estimate"] == pytest.approx(0.01)
    assert slope["normalized_win_share"]["estimate"] == pytest.approx(
        0.01 * weighted_players
    )
    assert endpoint["score"]["estimate"] == pytest.approx(0.02)
    assert endpoint["normalized_win_share"]["estimate"] == pytest.approx(
        0.02 * weighted_players
    )
    assert np.isclose(slope["score"]["standard_error"], 0.0)
