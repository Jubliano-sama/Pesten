from pathlib import Path

import pytest
import torch

pufferlib = pytest.importorskip("pufferlib")
if not str(getattr(pufferlib, "__version__", "")).startswith("4"):
    pytest.skip("requires pinned PufferLib 4 runtime", allow_module_level=True)

from temporal_reward_league import (
    pfsp_weights,
    save_reward_checkpoint,
    train_reward_population,
)
from temporal_puffer_v4 import NativePufferTemporalPolicy


def test_accepted_module_has_no_ppo_actor_or_imitation_import_path():
    source = Path("temporal_reward_league.py").read_text(encoding="utf-8")
    prohibited_imports = (
        "from benchmark_agents",
        "from ppo2",
        "from temporal_league",
        "from temporal_checkpoint_diagnostics",
        "from temporal_rejected_imitation_sweep",
    )
    assert not any(item in source for item in prohibited_imports)
    assert "ppo_actor_" not in source
    assert "torch.load(" not in source


def test_pfsp_prioritizes_difficult_historical_opponents():
    weights = pfsp_weights((0.2, 0.5, 0.8))
    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] > weights[1] > weights[2]


def test_reward_checkpoint_proves_random_parentless_lineage(tmp_path):
    torch.manual_seed(61)
    policy = NativePufferTemporalPolicy(hidden_size=16)
    path = tmp_path / "reward.pt"
    metadata = save_reward_checkpoint(
        path,
        policy,
        initialization_seed=61,
        initial_state_sha256="a" * 64,
        recipe={"schedule": ["current_selfplay"]},
        epoch=1,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    lineage = payload["lineage"]
    assert metadata["sha256"]
    assert lineage["initialization"] == "torch_random_initialization"
    assert lineage["parent_checkpoint"] is None
    assert lineage["imitation_checkpoint"] is None
    assert lineage["teacher_checkpoint"] is None
    assert not lineage["ppo_training_opponent"]
    assert not lineage["prohibited_training_sources_used"]


def test_tiny_accepted_training_never_loads_actor_and_has_no_ppo_contribution(
    tmp_path, monkeypatch
):
    def forbidden_load(*args, **kwargs):
        raise AssertionError("accepted training attempted to load a checkpoint")

    monkeypatch.setattr(torch, "load", forbidden_load)
    result = train_reward_population(
        seed=67,
        device="cpu",
        epochs=1,
        games_per_epoch=10,
        development_pairs=1,
        pfsp_pairs=1,
        hidden_size=16,
        num_layers=1,
        mode_conditioned_heads=False,
        learning_rate=3e-4,
        update_epochs=1,
        minibatch_trajectories=8,
        gamma=0.97,
        gae_lambda=0.95,
        clip_ratio=0.2,
        target_kl=0.02,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        output_dir=tmp_path,
        schedule=(
            "current_selfplay",
            "current_selfplay",
            "historical_temporal",
            "smart_rule",
        ),
    )
    epoch = result["epochs"][0]
    assert "ppo_champion" not in epoch["contribution"]["games"]
    assert "ppo_champion" not in epoch["contribution"]["decisions"]
    assert result["lineage"]["parent_checkpoint"] is None
    assert result["best_checkpoint"]["development_evaluation"][
        "contains_ppo_benchmark"
    ] is False
    assert Path(result["accepted_manifest"]["path"]).is_file()
