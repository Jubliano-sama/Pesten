import pytest

from benchmark_rlvr_pool import OPPONENTS, PLAYER_WEIGHTS, _weighted_summary


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
