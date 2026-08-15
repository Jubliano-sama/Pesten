import argparse
from pathlib import Path

import pytest

from benchmark_rlvr_pool_batched import _make_layout
from benchmark_teacher_gamma_precision import (
    CONFIDENCE_SPECS,
    load_allocation,
    parse_model,
)


def test_multiarm_layout_uses_every_variant_on_the_same_seat_rotations():
    records, roles, seeds = _make_layout(
        {
            "players": 3,
            "blocks": 2,
            "opponent": "self_anchor",
            "seed": 100,
            "variants": ("gamma_0p95", "gamma_0p97", "gamma_0p99"),
            "policy_names": (
                "gamma_0p95",
                "gamma_0p97",
                "gamma_0p99",
                "self_anchor",
            ),
            "self_anchor_name": "self_anchor",
        }
    )

    assert len(records) == len(roles) == len(seeds) == 2 * 3 * 3
    assert set(seeds.tolist()) == {100, 101}
    for (_, variant, candidate_seat, _), row_roles in zip(records, roles):
        assert row_roles[candidate_seat] == ("temporal", variant)
        assert all(
            role == ("temporal", "self_anchor")
            for seat, role in enumerate(row_roles)
            if seat != candidate_seat
        )


def test_checked_in_precision_allocation_is_complete_and_fixed():
    path = Path("benchmarks/teacher_gru_gamma_precision_plan_20260814.json")
    allocation, payload = load_allocation(path)

    assert len(allocation) == 21
    assert sum(allocation.values()) == 462_463
    assert payload["target_normalized_ci99_9_half_width"] == 0.005
    assert payload["corrected_evaluation_seed"] == 4_002_608_140
    assert "Do not extend" in payload["stopping_rule"]


def test_gamma_0p995_plan_uses_requested_99_percent_precision():
    path = Path(
        "benchmarks/teacher_gru_gamma_0p995_precision_plan_20260815.json"
    )
    allocation, payload = load_allocation(path)

    assert len(allocation) == 21
    assert sum(allocation.values()) == 277_479
    assert payload["confidence_level"] == 0.99
    assert payload["target_normalized_ci_half_width"] == 0.005
    assert payload["corrected_evaluation_seed"] == 3_602_608_150
    assert payload["allocation_method"][
        "expected_half_width_under_source_variance"
    ] < 0.005
    assert CONFIDENCE_SPECS["0.99"][0] == pytest.approx(
        payload["normal_quantile"]
    )


def test_gamma_0p9975_plan_uses_requested_95_percent_precision():
    path = Path(
        "benchmarks/teacher_gru_gamma_0p9975_precision_plan_20260815.json"
    )
    allocation, payload = load_allocation(path)

    assert len(allocation) == 21
    assert sum(allocation.values()) == 166_488
    assert payload["confidence_level"] == 0.95
    assert payload["target_normalized_ci_half_width"] == 0.005
    assert payload["corrected_evaluation_seed"] == 3_202_608_150
    assert payload["allocation_method"][
        "expected_half_width_under_source_variance"
    ] < 0.005
    assert CONFIDENCE_SPECS["0.95"][0] == pytest.approx(
        payload["normal_quantile"]
    )


def test_model_argument_requires_a_safe_unique_style_label():
    assert parse_model("gamma_0p95=model.pt") == (
        "gamma_0p95",
        Path("model.pt"),
    )
    with pytest.raises(argparse.ArgumentTypeError):
        parse_model("gamma-0.95=model.pt")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_model("model.pt")
