from pathlib import Path

import pytest

from projected_ppo_iterative import (
    register_unique_deal_seeds,
    run_iterative_lane,
)


CHAMPION = Path(__file__).resolve().parents[1] / "ppo_actor_100run3.pth"


def rollout(*seeds):
    return {
        "deal_seed_rng_replay_verified": True,
        "deal_seed_count": len(seeds),
        "deal_seeds": [{"deal_seed": seed} for seed in seeds],
    }


def test_deal_registry_rejects_internal_and_cross_rollout_reuse():
    registry = set()
    assert register_unique_deal_seeds(
        registry,
        rollout(11, 12),
        label="first",
    ) == [11, 12]
    with pytest.raises(RuntimeError, match="reused"):
        register_unique_deal_seeds(
            registry,
            rollout(12, 13),
            label="overlap",
        )
    with pytest.raises(RuntimeError, match="within rollout"):
        register_unique_deal_seeds(
            registry,
            rollout(14, 14),
            label="duplicate",
        )


def test_tiny_iterative_lane_uses_fresh_rollouts_and_saves_each_iterate(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "iterative"
    output_dir.mkdir()
    registry = set()
    lane = run_iterative_lane(
        CHAMPION,
        output_dir,
        {
            "name": "tiny",
            "learning_rate": 3e-6,
            "kl_coefficient": 0.5,
            "entropy_coefficient": 1e-3,
            "target_kl": 1e-3,
        },
        lane_seed=84_711_001,
        iterations=2,
        critic_games=2,
        critic_epochs=1,
        games=2,
        update_epochs=1,
        minibatch_size=16,
        diagnostic_chunk_size=8,
        base_chunk_size=8,
        rollout_batch_size=2,
        temperature=0.8,
        torch_threads=2,
        deal_registry=registry,
    )

    assert len(lane["iterations"]) == 2
    assert len(registry) == 6
    batches = {
        item["fresh_rollout"]["sha256"]
        for item in lane["iterations"]
    }
    assert len(batches) == 2
    for item in lane["iterations"]:
        assert item["update"]["target_kl_satisfied"]
        assert item["update"]["policy_updated"]
        assert (output_dir / item["checkpoint"]["path"]).is_file()
