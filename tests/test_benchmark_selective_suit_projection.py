import pytest

from benchmark_selective_suit_projection import (
    ALWAYS_SUIT_KEY,
    CHAMPION_KEY,
    build_specs,
    validate_seed_ranges,
)


def test_grid_and_confirmation_ranges_are_disjoint_and_below_90m():
    validate_seed_ranges(
        grid_seed=86_000_001,
        grid_pairs=20,
        confirmation_seed=88_000_001,
        confirmation_pairs=60,
        pool_count=3,
    )
    with pytest.raises(ValueError, match="disjoint"):
        validate_seed_ranges(
            grid_seed=88_000_001,
            grid_pairs=20,
            confirmation_seed=88_100_001,
            confirmation_pairs=60,
            pool_count=3,
        )


def test_specs_share_deals_without_policy_specific_seed_derivation():
    specs = build_specs(
        base_seed=86_000_001,
        pairs=2,
        direct_matchups=(
            ("rule_a", CHAMPION_KEY),
            (ALWAYS_SUIT_KEY, CHAMPION_KEY),
        ),
        pool_policy_keys=("rule_a", ALWAYS_SUIT_KEY, CHAMPION_KEY),
        pool_names=("random",),
    )
    direct_seeds = {
        spec.deal_seed for spec in specs if spec.section == "direct"
    }
    pool_seeds = {
        spec.deal_seed for spec in specs if spec.section == "pool"
    }
    assert direct_seeds == {86_000_001, 86_000_002}
    assert pool_seeds == {86_100_001, 86_100_002}
    assert len(specs) == 2 * 2 * 2 + 3 * 2 * 2
