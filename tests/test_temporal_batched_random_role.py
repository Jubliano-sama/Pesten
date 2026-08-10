import pytest

from random_specialist_workflow import PhaseZeroTracePolicy
from temporal_batched_speed import (
    collect_league_batch_batched,
    ordered_legal_actions,
    sample_audited_uniform_action,
)


def _collect_trace():
    trace = []
    episodes, metadata = collect_league_batch_batched(
        PhaseZeroTracePolicy().eval(),
        None,
        (),
        games=4,
        seed=81_020_001,
        schedule=("random",),
        random_opponent_seeds=(81_030_001, 81_030_002, 81_030_003, 81_030_004),
        max_turns=180,
        trace_sink=trace,
    )
    return episodes, metadata, trace


def test_random_schedule_uses_audited_uniform_local_rng_and_preserves_schema():
    episodes, metadata, trace = _collect_trace()

    assert metadata["games"] == {"random": 4}
    assert set(metadata) == {"games", "decisions", "total_turns", "trajectories"}
    assert len(episodes) == len(trace) == 4
    for game_trace in trace:
        assert game_trace["opponent_rng_seed"] != game_trace["deal_seed"]
        assert game_trace["recurrent"][0]["initial_state_sha256"]
        for action in game_trace["actions"]:
            assert action["action"] in ordered_legal_actions(action["legal_actions"])
            if action["role"] == "audited_uniform_random":
                assert action["role"] != "randomAgent.AgentRandom"

    for trajectory, _ in episodes:
        assert trajectory
        assert all(step.decision.legal_actions[step.action] for step in trajectory)


def test_random_schedule_is_reproducible_with_same_separate_opponent_seed_bank():
    first_episodes, first_metadata, first_trace = _collect_trace()
    second_episodes, second_metadata, second_trace = _collect_trace()

    assert first_metadata == second_metadata
    assert first_trace == second_trace
    assert [
        [step.action for step in trajectory] for trajectory, _ in first_episodes
    ] == [
        [step.action for step in trajectory] for trajectory, _ in second_episodes
    ]


def test_random_schedule_requires_explicit_independent_opponent_rng_bank():
    with pytest.raises(ValueError, match="random_opponent_seed"):
        collect_league_batch_batched(
            PhaseZeroTracePolicy().eval(),
            None,
            (),
            games=1,
            seed=81_040_001,
            schedule=("random",),
            max_turns=20,
        )


def test_uniform_sampler_uses_ascending_legal_order():
    legal = tuple(index in (2, 7, 54) for index in range(55))
    assert ordered_legal_actions(legal) == (2, 7, 54)
    # Seed 2 selects index 0 on NumPy's documented RandomState stream, which
    # proves the returned action is the first *ordered* legal action.
    import numpy as np

    assert sample_audited_uniform_action(legal, np.random.RandomState(2)) == 2


def test_canonical_random_is_an_explicitly_separate_coupled_rng_role():
    trace = []
    _, metadata = collect_league_batch_batched(
        PhaseZeroTracePolicy().eval(),
        None,
        (),
        games=2,
        seed=4_028_451_531,
        schedule=("canonical_random",),
        max_turns=180,
        trace_sink=trace,
    )

    assert metadata["games"] == {"canonical_random": 2}
    assert all("opponent_rng_seed" not in item for item in trace)
    assert all(
        item["canonical_random_rng"] == {
            "deal_seed_shared_with_engine": item["deal_seed"],
            "suit_rng": "random.Random(deal_seed).randint(0, 3)",
            "independent_opponent_rng": False,
        }
        for item in trace
    )
    with pytest.raises(ValueError, match="must not receive"):
        collect_league_batch_batched(
            PhaseZeroTracePolicy().eval(),
            None,
            (),
            games=1,
            seed=4_028_451_531,
            schedule=("canonical_random",),
            random_opponent_seeds=(1,),
            max_turns=20,
        )
