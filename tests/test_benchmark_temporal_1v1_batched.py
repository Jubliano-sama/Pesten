import numpy as np
import pytest

from benchmark_temporal_1v1_batched import _summary


def test_temporal_1v1_summary_uses_independent_seat_pair_blocks():
    result = _summary(
        np.asarray([1.0, 0.5, 0.0, 1.0]),
        {"wins": 5, "losses": 2, "draws": 1},
    )

    assert result["blocks"] == 4
    assert result["games"] == 8
    assert result["score"] == pytest.approx(0.625)
    assert result["paired_standard_error"] == pytest.approx(
        np.asarray([1.0, 0.5, 0.0, 1.0]).std(ddof=1) / 2
    )
