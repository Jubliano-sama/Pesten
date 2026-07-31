from benchmark_robust_suit_aggregation import (
    CHAMPION_KEY,
    MEAN_LOGITS_KEY,
    methods_passing_screen,
)


def summary(
    *,
    mean_pool=0.60,
    majority_pool=0.61,
    borda_pool=0.62,
    mean_champion=0.52,
    majority_champion=0.52,
    borda_champion=0.51,
    majority_mean=0.50,
    borda_mean=0.55,
):
    return {
        "pool_comparisons": {
            MEAN_LOGITS_KEY: {"combined_score": mean_pool},
            "majority_vote": {"combined_score": majority_pool},
            "mean_borda": {"combined_score": borda_pool},
        },
        "direct": {
            f"{MEAN_LOGITS_KEY}_vs_{CHAMPION_KEY}": {
                "evaluated_score": mean_champion
            },
            f"majority_vote_vs_{CHAMPION_KEY}": {
                "evaluated_score": majority_champion
            },
            f"mean_borda_vs_{CHAMPION_KEY}": {
                "evaluated_score": borda_champion
            },
            f"majority_vote_vs_{MEAN_LOGITS_KEY}": {
                "evaluated_score": majority_mean
            },
            f"mean_borda_vs_{MEAN_LOGITS_KEY}": {
                "evaluated_score": borda_mean
            },
        },
    }


def test_screen_gate_requires_all_three_predeclared_conditions():
    assert methods_passing_screen(summary()) == ["majority_vote"]
    assert methods_passing_screen(
        summary(majority_pool=0.60)
    ) == []
    assert methods_passing_screen(
        summary(majority_champion=0.51)
    ) == []
    assert methods_passing_screen(
        summary(majority_mean=0.49)
    ) == []
