import numpy as np
import pytest

from benchmark_random_1v1_batched import (
    _hoeffding_interval,
    _model_summary,
    _paired_summary,
)
from benchmark_rlvr_pool_batched import _make_layout


def test_model_summary_reports_clustered_outight_win_bounds() -> None:
    win_shares = np.asarray([1.0, 0.5, 1.0, 0.0])
    scores = np.asarray([1.0, 0.5, 1.0, 0.25])
    result = _model_summary(
        win_shares,
        scores,
        {"wins": 5, "losses": 2, "draws": 1},
    )

    assert result["blocks"] == 4
    assert result["games"] == 8
    assert result["outright_win_rate"] == pytest.approx(0.625)
    assert result["score_rate_draws_half"] == pytest.approx(0.6875)
    assert result["outright_win_rate_hoeffding_ci99_9"] == pytest.approx(
        _hoeffding_interval(0.625, 4)
    )


def test_paired_summary_uses_common_deal_differences() -> None:
    before = np.asarray([0.0, 0.5, 1.0, 0.5])
    after = np.asarray([0.5, 0.5, 1.0, 1.0])

    result = _paired_summary(before, after)

    assert result["after_minus_before"] == pytest.approx(0.25)
    assert result["standard_error"] == pytest.approx(
        np.asarray([0.5, 0.0, 0.0, 0.5]).std(ddof=1) / 2
    )


def test_layout_retains_ablation_flags_for_worker_evaluation() -> None:
    task = {
        "opponent": "canonical_random",
        "players": 2,
        "blocks": 1,
        "seed": 99,
        "disable_pass_before": False,
        "disable_pass_after": True,
        "pass_only_one_card_before": False,
        "pass_only_one_card_after": False,
    }
    records, _, _ = _make_layout(task)

    assert {record[1] for record in records} == {"before", "after"}
    assert task["disable_pass_before"] is False
    assert task["disable_pass_after"] is True
