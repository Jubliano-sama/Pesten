import argparse
import random

import game
import numpy as np
import pytest
import torch

from benchmark_model_zoo import DEFAULT_TEACHER
from benchmark_residual_checkpoint import (
    DEFAULT_OPPONENT_NAMES,
    _default_evaluation_dependencies,
    build_argument_parser,
    detect_residual_checkpoint_format,
    deterministic_evaluation_agent_factory,
    deterministic_evaluation_runtime,
    evaluate_candidate_and_base_paired,
    evaluate_paired,
    evaluate_residual_checkpoint,
    filter_opponents,
    load_residual_checkpoint,
    parse_opponent_names,
    stratified_paired_bootstrap_interval,
)
from randomAgent import AgentRandom
from residual_temporal_policy import (
    CHECKPOINT_FORMAT,
    RECURRENT_CHECKPOINT_FORMAT,
    ResidualCausalTemporalPolicy,
    ResidualRecurrentTemporalPolicy,
)
from temporal_puffer_agent import temporal_agent_factory


def _save_causal(path):
    policy = ResidualCausalTemporalPolicy(
        champion_path=DEFAULT_TEACHER,
        d_model=16,
        num_layers=1,
        num_heads=4,
        feedforward_size=32,
        max_context=16,
    )
    policy.save_checkpoint(path)


def _save_recurrent(path, cell="gru"):
    policy = ResidualRecurrentTemporalPolicy(
        champion_path=DEFAULT_TEACHER,
        hidden_size=16,
        num_layers=1,
        cell=cell,
    )
    policy.save_checkpoint(path)


@pytest.mark.parametrize(
    ("family", "expected_format", "expected_type"),
    (
        ("causal", CHECKPOINT_FORMAT, ResidualCausalTemporalPolicy),
        ("gru", RECURRENT_CHECKPOINT_FORMAT, ResidualRecurrentTemporalPolicy),
        ("lstm", RECURRENT_CHECKPOINT_FORMAT, ResidualRecurrentTemporalPolicy),
    ),
)
def test_format_dispatch_loads_each_residual_family(
    tmp_path, family, expected_format, expected_type
):
    checkpoint = tmp_path / f"{family}.pt"
    if family == "causal":
        _save_causal(checkpoint)
    else:
        _save_recurrent(checkpoint, family)
    assert detect_residual_checkpoint_format(checkpoint) == expected_format
    policy = load_residual_checkpoint(
        checkpoint, champion_path=DEFAULT_TEACHER
    )
    assert isinstance(policy, expected_type)
    assert policy.loaded_checkpoint_format == expected_format
    if family != "causal":
        assert policy.cell == family
    assert not policy.teacher.training


@pytest.mark.parametrize("family", ("causal", "gru", "lstm"))
def test_evaluation_uses_common_pool_for_each_checkpoint_family(
    tmp_path, family
):
    checkpoint = tmp_path / f"{family}.pt"
    if family == "causal":
        _save_causal(checkpoint)
    else:
        _save_recurrent(checkpoint, family)
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
                opponent,
                pairs,
                seed,
                candidate_name,
                opponent_name,
            )
        )
        candidate_score = (
            1.0 if opponent_name == "ppo_champion" else 0.25
        )
        base_score = 0.5
        candidate_scores = [candidate_score] * pairs
        base_scores = [base_score] * pairs
        deltas = [
            candidate_score - base_score
        ] * pairs
        return {
            "candidate_result": {"pairs": pairs, "seed": seed},
            "base_result": {"pairs": pairs, "seed": seed},
            "comparison": {
                "pairs": pairs,
                "candidate_minus_base_score": deltas[0],
            },
            "_candidate_pair_scores": candidate_scores,
            "_base_pair_scores": base_scores,
            "_pair_deltas": deltas,
        }

    payload = evaluate_residual_checkpoint(
        checkpoint=checkpoint,
        champion_path=DEFAULT_TEACHER,
        pairs=12,
        seed=417,
        evaluate_pair=lambda *args, **kwargs: None,
        evaluate_comparison=fake_comparison,
        opponents={"ppo_champion": object(), "random": object()},
    )
    assert len(calls) == 2
    assert calls[0][3:] == (
        12,
        417,
        payload["policy"]["architecture"],
        "ppo_champion",
    )
    assert calls[1][3:] == (
        20,
        100_417,
        payload["policy"]["architecture"],
        "random",
    )
    expected_class = (
        "ResidualCausalTemporalPolicy"
        if family == "causal"
        else "ResidualRecurrentTemporalPolicy"
    )
    assert payload["policy"]["class"] == expected_class
    assert payload["policy"]["recurrent_cell"] == (
        None if family == "causal" else family
    )
    assert payload["base_policy"]["frozen"]
    assert payload["results"]["ppo_champion"] == {
        "pairs": 12,
        "seed": 417,
    }
    assert payload["base_results"]["random"] == {
        "pairs": 20,
        "seed": 100_417,
    }
    assert payload["paired_candidate_minus_base"][
        "ppo_champion"
    ]["candidate_minus_base_score"] == 0.5
    assert payload["paired_candidate_minus_base"][
        "random"
    ]["candidate_minus_base_score"] == -0.25
    assert payload["combined_pool_delta"][
        "candidate_minus_base_score"
    ] == pytest.approx(0.03125)
    assert payload["combined_pool_delta"][
        "paired_bootstrap_ci95"
    ] == pytest.approx([0.03125, 0.03125])


def test_unknown_checkpoint_format_is_rejected(tmp_path):
    checkpoint = tmp_path / "unknown.pt"
    torch.save({"format": "not-a-residual-policy"}, checkpoint)
    with pytest.raises(ValueError, match="unsupported residual"):
        load_residual_checkpoint(
            checkpoint, champion_path=DEFAULT_TEACHER
        )


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
        tuple(
            tuple(held.number for held in player.mydeck.cards)
            for player in match.players
        ),
        tuple(held.number for held in match.grab_deck.cards),
        tuple(held.number for held in match.game_deck.cards),
    )


@pytest.mark.parametrize("cell", ("gru", "lstm"))
def test_fast_deterministic_factory_has_exact_recurrent_game_parity(cell):
    torch.manual_seed(81_331)
    policy = ResidualRecurrentTemporalPolicy(
        champion_path=DEFAULT_TEACHER,
        hidden_size=16,
        num_layers=1,
        cell=cell,
    ).eval()
    with torch.no_grad():
        policy.residual_actor.weight.normal_(std=0.02)
        policy.residual_actor.bias.normal_(std=0.02)
    reference = temporal_agent_factory(
        policy, deterministic=True, seed=19_001
    )
    optimized = deterministic_evaluation_agent_factory(
        policy, seed=19_001
    )

    reference_games = {
        (seed, candidate_seat): _game_fingerprint(
            reference, seed, candidate_seat
        )
        for seed in range(81_400, 81_408)
        for candidate_seat in (0, 1)
    }
    with deterministic_evaluation_runtime("cpu", cpu_threads=1):
        for seed in range(81_400, 81_408):
            for candidate_seat in (0, 1):
                assert _game_fingerprint(
                    optimized, seed, candidate_seat
                ) == reference_games[(seed, candidate_seat)]


def test_cpu_evaluation_thread_limit_is_temporary_and_reported(tmp_path):
    checkpoint = tmp_path / "gru.pt"
    _save_recurrent(checkpoint)
    previous_threads = torch.get_num_threads()
    observed_threads = []

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
        del candidate, base, opponent, candidate_name, opponent_name
        observed_threads.append(torch.get_num_threads())
        scores = [0.5] * pairs
        return {
            "candidate_result": {"ok": True},
            "base_result": {"ok": True},
            "comparison": {"ok": True},
            "_candidate_pair_scores": scores,
            "_base_pair_scores": scores,
            "_pair_deltas": [0.0] * pairs,
        }

    payload = evaluate_residual_checkpoint(
        checkpoint=checkpoint,
        champion_path=DEFAULT_TEACHER,
        pairs=2,
        seed=91_001,
        cpu_threads=1,
        evaluate_pair=lambda *args, **kwargs: None,
        evaluate_comparison=fake_comparison,
        opponents={"ppo_champion": object()},
    )
    assert observed_threads == [1]
    assert torch.get_num_threads() == previous_threads
    assert payload["evaluation_runtime"] == {
        "deterministic_fast_path": True,
        "records_training_trajectories": False,
        "cpu_threads": 1,
    }


def test_evaluation_runtime_rejects_invalid_thread_count():
    with pytest.raises(ValueError, match="at least one"):
        with deterministic_evaluation_runtime(
            "cpu", cpu_threads=0
        ):
            pass


def test_default_screening_dependencies_do_not_require_pufferlib():
    paired, opponents = _default_evaluation_dependencies(
        DEFAULT_TEACHER, "cpu"
    )
    assert paired is evaluate_paired
    assert tuple(opponents) == (
        "ppo_champion",
        "bayesian_tree",
        "smart",
        "smarter",
        "random",
    )


def test_opponents_cli_defaults_to_full_pool_and_parses_selected_order():
    parser = build_argument_parser()
    default_args = parser.parse_args(
        ["--checkpoint", "candidate.pt", "--seed", "91"]
    )
    assert default_args.opponents == DEFAULT_OPPONENT_NAMES

    selected = parser.parse_args(
        [
            "--checkpoint",
            "candidate.pt",
            "--seed",
            "91",
            "--opponents",
            " random,ppo_champion ",
        ]
    )
    assert selected.opponents == ("random", "ppo_champion")

    direct_only = parser.parse_args(
        [
            "--checkpoint",
            "candidate.pt",
            "--seed",
            "91",
            "--opponents",
            "ppo_champion",
        ]
    )
    assert direct_only.opponents == ("ppo_champion",)


@pytest.mark.parametrize(
    "value",
    (
        "",
        ",",
        "random,",
        "random,random",
        "random,unknown",
    ),
)
def test_opponents_cli_rejects_empty_duplicate_or_unknown_names(value):
    with pytest.raises(argparse.ArgumentTypeError) as error:
        parse_opponent_names(value)
    assert "opponent" in str(error.value)


def test_filter_opponents_preserves_requested_order_and_validates_mapping():
    pool = {
        "ppo_champion": object(),
        "smart": object(),
        "random": object(),
    }
    selected = filter_opponents(pool, ("random", "smart"))
    assert tuple(selected) == ("random", "smart")
    assert selected["random"] is pool["random"]
    assert tuple(pool) == ("ppo_champion", "smart", "random")

    with pytest.raises(ValueError, match="unknown opponent"):
        filter_opponents(pool, ("smarter",))


def test_selected_pool_keeps_champion_base_and_full_pool_seed_offsets(
    tmp_path,
):
    checkpoint = tmp_path / "gru.pt"
    _save_recurrent(checkpoint)
    pool = {
        "ppo_champion": object(),
        "smart": object(),
        "random": object(),
    }
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
        del candidate, candidate_name
        calls.append(
            (base, opponent, pairs, seed, opponent_name)
        )
        scores = [0.5] * pairs
        return {
            "candidate_result": {"opponent": opponent_name},
            "base_result": {"opponent": opponent_name},
            "comparison": {"opponent": opponent_name},
            "_candidate_pair_scores": scores,
            "_base_pair_scores": scores,
            "_pair_deltas": [0.0] * pairs,
        }

    payload = evaluate_residual_checkpoint(
        checkpoint=checkpoint,
        champion_path=DEFAULT_TEACHER,
        pairs=12,
        seed=91_400,
        evaluate_pair=lambda *args, **kwargs: None,
        evaluate_comparison=fake_comparison,
        opponents=pool,
        opponent_names=("random", "smart"),
    )

    assert calls == [
        (pool["ppo_champion"], pool["random"], 20, 291_400, "random"),
        (pool["ppo_champion"], pool["smart"], 20, 191_400, "smart"),
    ]
    assert payload["opponents"] == ["random", "smart"]
    assert list(payload["results"]) == ["random", "smart"]
    assert payload["combined_pool_delta"]["opponents"] == [
        "random",
        "smart",
    ]


def test_local_paired_evaluator_is_seed_deterministic():
    arguments = {
        "pairs": 4,
        "seed": 91_100,
        "candidate_name": "candidate",
        "opponent_name": "random",
    }
    first = evaluate_paired(AgentRandom, AgentRandom, **arguments)
    second = evaluate_paired(AgentRandom, AgentRandom, **arguments)
    first.pop("elapsed_seconds")
    second.pop("elapsed_seconds")
    assert first == second
    assert first["games"] == 8
    assert first["wins"] + first["losses"] + first["draws"] == 8


def test_candidate_base_comparison_uses_identical_deals_and_seats():
    comparison = evaluate_candidate_and_base_paired(
        AgentRandom,
        AgentRandom,
        AgentRandom,
        pairs=8,
        seed=91_200,
        candidate_name="candidate",
        opponent_name="random",
    )
    assert comparison["candidate_result"]["score"] == comparison[
        "base_result"
    ]["score"]
    assert comparison["comparison"][
        "candidate_minus_base_score"
    ] == 0.0
    assert comparison["comparison"]["paired_bootstrap_ci95"] == [
        0.0,
        0.0,
    ]
    assert comparison["_pair_deltas"] == [0.0] * 8


def test_stratified_pool_bootstrap_preserves_opponent_weights():
    interval = stratified_paired_bootstrap_interval(
        {
            "large": [0.5] * 12,
            "small": [-0.25] * 20,
        },
        seed=91_300,
    )
    assert interval == pytest.approx((0.03125, 0.03125))
