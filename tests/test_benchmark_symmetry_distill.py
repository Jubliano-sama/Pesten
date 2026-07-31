import pytest

from benchmark_symmetry_distill import (
    DEFAULT_POOLS,
    build_linear_specs,
)


def test_specs_use_unhashed_consecutive_common_deal_ranges():
    specs = build_linear_specs(
        pairs=3,
        base_seed=88_100_001,
        pool_names=("random", "smart"),
    )
    direct_seeds = sorted(
        {
            spec.deal_seed
            for spec in specs
            if spec.section == "direct"
        }
    )
    random_seeds = sorted(
        {
            spec.deal_seed
            for spec in specs
            if spec.opponent == "random"
        }
    )
    smart_seeds = sorted(
        {
            spec.deal_seed
            for spec in specs
            if spec.opponent == "smart"
        }
    )
    assert direct_seeds == [88_100_001, 88_100_002, 88_100_003]
    assert random_seeds == [88_200_001, 88_200_002, 88_200_003]
    assert smart_seeds == [88_300_001, 88_300_002, 88_300_003]
    assert len(specs) == 3 * (2 + 2 * 4)


def test_specs_fail_before_crossing_90m():
    with pytest.raises(ValueError, match="below"):
        build_linear_specs(
            pairs=2,
            base_seed=89_900_000,
            pool_names=DEFAULT_POOLS,
        )
