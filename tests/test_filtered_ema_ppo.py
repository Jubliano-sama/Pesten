from pathlib import Path

import pytest
import torch
from torch.distributions import Categorical

from ai_observation import ACTION_COUNT, OBSERVATION_SIZE
from anchored_ppo import CHAMPION_SHA256, file_sha256
from filtered_ema_ppo import (
    assert_no_recorded_seed_collisions,
    clipped_policy_surrogate,
    filtered_actor_update,
    initialize_ema,
    should_save_checkpoint,
    top_advantage_indices,
    update_ema,
)
from ppo2 import StudentNet
from seed_integrity import RecordedSeedUse


ROOT = Path(__file__).resolve().parents[1]
CHAMPION = ROOT / "ppo_actor_100run3.pth"


def test_top_advantage_indices_exact_count_order_and_tie_break() -> None:
    advantages = torch.tensor([1.0, 3.0, 3.0, -2.0, 2.0])
    assert top_advantage_indices(advantages, 0.4).tolist() == [1, 2]
    assert top_advantage_indices(advantages, 0.5).tolist() == [1, 2, 4]
    assert top_advantage_indices(advantages, 1.0).tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.01, float("nan")])
def test_top_advantage_indices_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError):
        top_advantage_indices(torch.ones(3), fraction)


def test_full_fraction_surrogate_is_standard_clipped_ppo() -> None:
    new = torch.tensor([-0.1, -0.9, -0.2], requires_grad=True)
    old = torch.tensor([-0.2, -0.7, -0.2])
    advantages = torch.tensor([1.2, -0.5, 0.3])
    actual, ratios = clipped_policy_surrogate(
        new,
        old,
        advantages,
        clip=0.2,
    )
    expected_ratios = torch.exp(new - old)
    expected = torch.minimum(
        expected_ratios * advantages,
        expected_ratios.clamp(0.8, 1.2) * advantages,
    ).mean()
    assert torch.equal(ratios, expected_ratios)
    assert torch.equal(actual, expected)
    actual.backward()
    assert bool(torch.isfinite(new.grad).all())


def test_ema_initialization_and_update_formula() -> None:
    online = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        online.weight.copy_(torch.tensor([[2.0, -4.0]]))
    ema = initialize_ema(online)
    assert not any(parameter.requires_grad for parameter in ema.parameters())
    with torch.no_grad():
        online.weight.copy_(torch.tensor([[6.0, 4.0]]))
    update_ema(ema, online, decay=0.75)
    assert torch.allclose(ema.weight, torch.tensor([[3.0, -2.0]]))


@pytest.mark.parametrize("decay", [-0.1, 1.0, float("nan")])
def test_ema_rejects_invalid_decay(decay: float) -> None:
    online = torch.nn.Linear(1, 1)
    ema = initialize_ema(online)
    with pytest.raises(ValueError):
        update_ema(ema, online, decay=decay)


def test_verified_champion_round_trips_as_raw_student_checkpoint() -> None:
    assert file_sha256(CHAMPION) == CHAMPION_SHA256
    actor = StudentNet()
    state = torch.load(CHAMPION, map_location="cpu", weights_only=True)
    actor.load_state_dict(state)
    ema = initialize_ema(actor)
    assert set(ema.state_dict()) == set(state)
    for name, value in ema.state_dict().items():
        assert torch.equal(value, state[name])


def test_recorded_seed_collision_audit_fails_closed() -> None:
    uses = [
        RecordedSeedUse(
            path="benchmarks/prior.json",
            key_path="deal_seed",
            seed=123,
        )
    ]
    with pytest.raises(RuntimeError, match="123@benchmarks/prior.json"):
        assert_no_recorded_seed_collisions([122, 123], uses)
    audit = assert_no_recorded_seed_collisions([124, 125], uses)
    assert audit["recorded_conflict_count"] == 0
    assert audit["planned_seed_count"] == 2


def test_actor_update_keeps_rollout_batchnorm_mode() -> None:
    actor = torch.nn.Sequential(
        torch.nn.BatchNorm1d(OBSERVATION_SIZE),
        torch.nn.Linear(OBSERVATION_SIZE, ACTION_COUNT),
    ).eval()
    ema = initialize_ema(actor)
    observations = torch.randn(8, OBSERVATION_SIZE)
    legal = torch.ones(8, ACTION_COUNT, dtype=torch.bool)
    actions = torch.arange(8) % ACTION_COUNT
    with torch.inference_mode():
        behavior = Categorical(logits=actor(observations)).log_prob(actions)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    log = filtered_actor_update(
        actor,
        ema,
        optimizer,
        observations,
        actions,
        behavior,
        torch.linspace(-1.0, 1.0, 8),
        legal,
        keep_fraction=0.5,
        temperature=1.0,
        clip=0.2,
        entropy_coefficient=0.0,
        batch_size=8,
        epochs=1,
        gradient_clip=1.0,
        ema_decay=0.999,
        generator=torch.Generator().manual_seed(7),
    )
    assert log["maximum_initial_ratio_error"] < 1e-6
    assert log["epochs"][0]["maximum_batch_ratio_error"] < 1e-6
    assert not actor.training


def test_checkpoint_cadence_always_saves_final() -> None:
    saved = [
        iteration
        for iteration in range(1, 11)
        if should_save_checkpoint(
            iteration,
            total_iterations=10,
            checkpoint_every=4,
        )
    ]
    assert saved == [4, 8, 10]


@pytest.mark.parametrize(
    ("iteration", "total", "every"),
    [(0, 8, 2), (1, 0, 2), (1, 8, 0), (9, 8, 2)],
)
def test_checkpoint_cadence_rejects_invalid_values(
    iteration: int,
    total: int,
    every: int,
) -> None:
    with pytest.raises(ValueError):
        should_save_checkpoint(
            iteration,
            total_iterations=total,
            checkpoint_every=every,
        )
