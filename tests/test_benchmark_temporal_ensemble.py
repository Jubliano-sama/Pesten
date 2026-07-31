import pytest
import torch
import random

import game
import numpy as np

from benchmark_residual_checkpoint import (
    deterministic_evaluation_agent_factory,
)
from benchmark_temporal_ensemble import (
    DEFAULT_DEVELOPMENT_SEED,
    SEALED_SEED_START,
    parse_member_specification,
    run_ensemble_screen,
)
from randomAgent import AgentRandom
from residual_temporal_policy import ResidualRecurrentTemporalPolicy
from benchmark_model_zoo import DEFAULT_TEACHER
from temporal_ensemble import TemporalRecurrentEnsemble
from temporal_puffer_agent import temporal_agent_factory


def _checkpoint(path, seed):
    import torch

    torch.manual_seed(seed)
    policy = ResidualRecurrentTemporalPolicy(
        champion_path=DEFAULT_TEACHER,
        hidden_size=8,
        num_layers=1,
        cell="gru",
    )
    policy.save_checkpoint(path)


def test_small_screen_reuses_exact_seeds_across_combinations(tmp_path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    _checkpoint(first, 41)
    _checkpoint(second, 42)
    calls = []

    def fake_comparison(
        candidate,
        base,
        opponent,
        *,
        pairs,
        seed,
        candidate_name,
        opponent_name,
    ):
        calls.append(
            (
                candidate,
                base,
                candidate_name,
                opponent_name,
                pairs,
                seed,
                torch.get_num_threads(),
            )
        )
        candidate_score = (
            0.75 if opponent_name == "ppo_champion" else 0.25
        )
        base_score = 0.5
        candidate_scores = [candidate_score] * pairs
        base_scores = [base_score] * pairs
        deltas = [
            candidate_score - base_score
        ] * pairs
        return {
            "candidate_result": {
                "score": candidate_score,
                "seed": seed,
            },
            "base_result": {"score": base_score, "seed": seed},
            "comparison": {
                "candidate_minus_base_score": deltas[0],
                "seed": seed,
            },
            "_candidate_pair_scores": candidate_scores,
            "_base_pair_scores": base_scores,
            "_pair_deltas": deltas,
        }

    previous_threads = torch.get_num_threads()
    temporary_previous = 2 if previous_threads != 2 else 3
    torch.set_num_threads(temporary_previous)
    try:
        payload = run_ensemble_screen(
            members={"first": first, "second": second},
            combinations=("mean_logits", "majority_vote"),
            pairs=3,
            seed=DEFAULT_DEVELOPMENT_SEED,
            cpu_threads=1,
            evaluate_comparison=fake_comparison,
            opponents={"ppo_champion": object(), "random": object()},
        )
        assert torch.get_num_threads() == temporary_previous
    finally:
        torch.set_num_threads(previous_threads)
    assert [call[5] for call in calls[:2]] == [
        DEFAULT_DEVELOPMENT_SEED,
        DEFAULT_DEVELOPMENT_SEED + 100_000,
    ]
    assert [call[5] for call in calls[2:]] == [
        DEFAULT_DEVELOPMENT_SEED,
        DEFAULT_DEVELOPMENT_SEED + 100_000,
    ]
    assert [call[4] for call in calls[:2]] == [3, 20]
    assert all(call[6] == 1 for call in calls)
    assert set(payload["combinations"]) == {
        "mean_logits",
        "majority_vote",
    }
    assert "independent recurrent state" in payload[
        "information_boundary"
    ]
    assert set(payload["member_metadata"]) == {"first", "second"}
    assert all(
        len(metadata["sha256"]) == 64
        for metadata in payload["member_metadata"].values()
    )
    for screen in payload["combinations"].values():
        assert set(screen["results"]) == {
            "ppo_champion",
            "random",
        }
        assert set(screen["base_results"]) == {
            "ppo_champion",
            "random",
        }
        assert screen["paired_candidate_minus_base"][
            "ppo_champion"
        ]["candidate_minus_base_score"] == 0.25
        assert screen["paired_candidate_minus_base"][
            "random"
        ]["candidate_minus_base_score"] == -0.25
        assert screen["combined_pool_delta"][
            "candidate_minus_base_score"
        ] == pytest.approx((0.25 * 3 - 0.25 * 20) / 23)
        assert screen["combined_pool_delta"]["bootstrap"] == (
            "stratified within opponent"
        )
    assert payload["evaluation_runtime"] == {
        "deterministic_fast_path": True,
        "records_training_trajectories": False,
        "cpu_threads": 1,
    }


def test_sealed_seed_and_bad_member_specification_are_rejected(tmp_path):
    checkpoint = tmp_path / "member.pt"
    _checkpoint(checkpoint, 43)
    with pytest.raises(ValueError, match="sealed"):
        run_ensemble_screen(
            members={"member": checkpoint},
            combinations=("mean_logits",),
            pairs=1,
            seed=SEALED_SEED_START,
            evaluate_comparison=lambda *args, **kwargs: {},
            opponents={"random": object()},
            base_factory=object(),
        )
    with pytest.raises(ValueError, match="NAME=CHECKPOINT"):
        parse_member_specification("missing-separator")


def test_opponent_filter_runs_only_requested_common_stratum(tmp_path):
    checkpoint = tmp_path / "member.pt"
    _checkpoint(checkpoint, 44)
    calls = []

    def fake_comparison(
        candidate,
        base,
        opponent,
        *,
        pairs,
        seed,
        candidate_name,
        opponent_name,
    ):
        calls.append(opponent_name)
        scores = [0.5] * pairs
        return {
            "candidate_result": {"score": 0.5},
            "base_result": {"score": 0.5},
            "comparison": {"candidate_minus_base_score": 0.0},
            "_candidate_pair_scores": scores,
            "_base_pair_scores": scores,
            "_pair_deltas": [0.0] * pairs,
        }

    payload = run_ensemble_screen(
        members={"member": checkpoint},
        combinations=("mean_logits",),
        pairs=4,
        seed=DEFAULT_DEVELOPMENT_SEED,
        opponent_names=("ppo_champion",),
        evaluate_comparison=fake_comparison,
        opponents={
            "ppo_champion": object(),
            "random": object(),
        },
    )
    assert calls == ["ppo_champion"]
    assert payload["opponents"] == ["ppo_champion"]
    screen = payload["combinations"]["mean_logits"]
    assert screen["combined_pool_delta"]["opponents"] == [
        "ppo_champion"
    ]


def _game_fingerprint(candidate_factory, seed, candidate_seat):
    np.random.seed(seed)
    random.seed(seed)
    factories = (
        [candidate_factory, AgentRandom]
        if candidate_seat == 0
        else [AgentRandom, candidate_factory]
    )
    match = game.Game(factories)
    turns, winner = match.auto_simulate()
    return (
        turns,
        winner,
        match.current_player_index,
        match.direction,
        match.penalty_amount,
        match.current_sort,
        match.current_true_number,
        tuple(match.sorts_played),
        tuple(
            tuple(sorted(event.items()))
            for event in match.public_history
        ),
    )


def test_fast_evaluation_agent_has_exact_standard_agent_game_parity(tmp_path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    _checkpoint(first, 45)
    _checkpoint(second, 46)
    ensemble = TemporalRecurrentEnsemble.from_checkpoints(
        (first, second),
        champion_path=DEFAULT_TEACHER,
        combination="mean_logits",
    ).eval()
    standard = temporal_agent_factory(
        ensemble, deterministic=True, seed=551
    )
    fast = deterministic_evaluation_agent_factory(
        ensemble, seed=551
    )
    for candidate_seat in (0, 1):
        assert _game_fingerprint(
            standard, 1_551, candidate_seat
        ) == _game_fingerprint(fast, 1_551, candidate_seat)
