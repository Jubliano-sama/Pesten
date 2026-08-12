import pytest

from benchmark_rlvr_pool import OPPONENTS, PLAYER_WEIGHTS, _weighted_summary
from benchmark_rlvr_pool_batched import _make_layout


def test_weighted_summary_reports_raw_and_player_normalized_change():
    strata = []
    for opponent in OPPONENTS:
        for players in range(2, 9):
            baseline = 1.0 / players
            strata.append(
                {
                    "opponent": opponent,
                    "players": players,
                    "before_scores": [baseline, baseline],
                    "after_scores": [baseline + 0.01, baseline + 0.01],
                }
            )

    result = _weighted_summary(strata)
    expected_normalized = 0.01 * sum(
        PLAYER_WEIGHTS[players] * players for players in range(2, 9)
    )
    assert result["delta"] == pytest.approx(0.01)
    assert result["delta_normalized_win_share"] == pytest.approx(
        expected_normalized
    )
    assert result["delta_standard_error"] == 0.0
    assert result["delta_normalized_standard_error"] == 0.0
    assert result["delta_normalized_hoeffding_margin99_9"] > (
        result["delta_hoeffding_margin99_9"]
    )


def test_batched_pool_layout_matches_every_variant_seat_on_each_deal():
    records, roles, seeds = _make_layout(
        {
            "opponent": "self_anchor",
            "players": 4,
            "blocks": 3,
            "seed": 1000,
        }
    )
    assert len(records) == 3 * 2 * 4
    for block in range(3):
        selected = records[block * 8 : (block + 1) * 8]
        assert {record[1] for record in selected} == {"before", "after"}
        assert sorted(record[2] for record in selected[:4]) == [0, 1, 2, 3]
        assert sorted(record[2] for record in selected[4:]) == [0, 1, 2, 3]
        assert set(seeds[block * 8 : (block + 1) * 8]) == {1000 + block}
    for row, (_, variant, candidate_seat, _) in enumerate(records):
        assert roles[row][candidate_seat] == ("temporal", variant)
