from pathlib import Path

import numpy as np
import pytest
import torch

pufferlib = pytest.importorskip("pufferlib")
if not str(getattr(pufferlib, "__version__", "")).startswith("4"):
    pytest.skip("requires pinned PufferLib 4 runtime", allow_module_level=True)

from batched_pesten import BatchedPesten, PHASE_DRAW_RETRY, POLICY_RL
from benchmark_agents import load_ppo_agent
from temporal_league import _snapshot, collect_league_batch
from temporal_observation import (
    DecisionInput,
    EventKind,
    PublicEventToken,
    build_draw_retry_legal_actions,
    build_legal_actions,
    build_private_observation,
    serialize_safe_history,
)
from temporal_puffer_v4 import NativePufferTemporalPolicy
from temporal_batched_speed import (
    BatchRolloutStats,
    LEAGUE_SCHEDULE,
    SafeEventCursor,
    TemporalSeatRuntime,
    _batch_temporal_actions,
    build_safe_decision,
    collect_league_batch_batched,
)


def _decision(events=(), *, seat=0):
    own_cards = (4, 16, 49) if seat == 0 else (8, 20, 48)
    observation = build_private_observation(
        own_cards=own_cards,
        current_suit="harten",
        current_rank=5,
        public_hand_sizes=(3, 3),
        observer_index=seat,
        current_player_index=seat,
        direction=1,
        penalty_amount=0,
        public_sorts_played=(2, 1, 0, 0, 0),
    )
    legal = build_legal_actions(
        own_cards=own_cards,
        current_suit="harten",
        current_rank=5,
        penalty_amount=0,
    )
    return DecisionInput(observation, legal, tuple(events))


def _runtime(policy, *, seat, seed):
    return TemporalSeatRuntime(
        game_index=seat,
        absolute_game_index=seat,
        seat=seat,
        policy=policy,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        state=policy.initial_state(),
        cursor=SafeEventCursor(),
    )


def test_variable_event_batch_matches_individual_native_steps():
    torch.manual_seed(41)
    policy = NativePufferTemporalPolicy(
        hidden_size=16, num_layers=2
    ).eval()
    decisions = (
        _decision(
            (
                PublicEventToken(EventKind.EPISODE_START),
                PublicEventToken(
                    EventKind.DRAW_COUNT,
                    actor_relative=1,
                    count=2,
                ),
            ),
            seat=0,
        ),
        _decision(
            (
                PublicEventToken(EventKind.EPISODE_START),
                PublicEventToken(
                    EventKind.PLAY,
                    actor_relative=1,
                    card_id=7,
                    suit_id=0,
                    count=1,
                ),
                PublicEventToken(
                    EventKind.CHOOSE_SUIT,
                    actor_relative=1,
                    suit_id=2,
                ),
            ),
            seat=1,
        ),
    )
    scalar_states = []
    scalar_values = []
    scalar_logits = []
    with torch.inference_mode():
        for decision in decisions:
            result = policy.step(decision, policy.initial_state())
            scalar_states.append(result.state)
            scalar_values.append(float(result.value.item()))
            scalar_logits.append(result.logits)

    runtimes = (
        _runtime(policy, seat=0, seed=101),
        _runtime(policy, seat=1, seed=102),
    )
    stats = BatchRolloutStats()
    _batch_temporal_actions(
        policy,
        list(zip(runtimes, decisions)),
        stats,
    )

    assert stats.temporal_decisions == 2
    assert stats.temporal_forward_calls == 4
    for index, runtime in enumerate(runtimes):
        assert runtime.safe_history == [decisions[index]]
        assert len(runtime.trajectory) == 1
        assert runtime.trajectory[0].value == pytest.approx(
            scalar_values[index], abs=2e-6
        )
        for batched_part, scalar_part in zip(
            runtime.state, scalar_states[index]
        ):
            assert torch.allclose(
                batched_part, scalar_part, atol=2e-6, rtol=2e-6
            )
        action = runtime.trajectory[0].action
        expected_log_probability = torch.log_softmax(
            scalar_logits[index], dim=-1
        )[action]
        assert runtime.trajectory[0].old_log_prob == pytest.approx(
            float(expected_log_probability.item()), abs=2e-6
        )


def test_deterministic_temporal_batch_uses_masked_argmax():
    torch.manual_seed(43)
    policy = NativePufferTemporalPolicy(
        hidden_size=16, num_layers=2
    ).eval()
    decisions = (_decision(seat=0), _decision(seat=1))
    runtimes = (
        _runtime(policy, seat=0, seed=201),
        _runtime(policy, seat=1, seed=202),
    )
    expected = [
        int(policy.step(item, policy.initial_state()).logits.argmax())
        for item in decisions
    ]
    actual = _batch_temporal_actions(
        policy,
        list(zip(runtimes, decisions)),
        None,
        deterministic=True,
    )
    assert actual == expected


def test_drawn_identity_and_opponent_cards_never_cross_safe_boundary():
    kinds = np.full((1, 2), POLICY_RL, dtype=np.int8)
    keys = np.full((1, 2), None, dtype=object)
    engine_a = BatchedPesten(kinds, keys, np.asarray([83], np.uint32))
    engine_b = BatchedPesten(kinds, keys, np.asarray([83], np.uint32))
    initial_sizes = tuple(int(value) for value in engine_a.hand_sizes[0])
    cursor_a = SafeEventCursor(
        last_hand_sizes=initial_sizes, decision_count=1
    )
    cursor_b = SafeEventCursor(
        last_hand_sizes=initial_sizes, decision_count=1
    )

    top_index = int(engine_a.draw_counts[0]) - 1
    engine_a.draw_piles[0, top_index] = 5
    engine_b.draw_piles[0, top_index] = 42
    assert engine_a._draw_card(0, 1) == 5
    assert engine_b._draw_card(0, 1) == 42
    assert not np.array_equal(engine_a.hands[0, 1], engine_b.hands[0, 1])

    decision_a = build_safe_decision(engine_a, 0, 0, cursor_a)
    decision_b = build_safe_decision(engine_b, 0, 0, cursor_b)
    assert decision_a == decision_b
    assert decision_a.public_events == (
        PublicEventToken(
            EventKind.DRAW_COUNT, actor_relative=1, count=1
        ),
    )
    serialized = serialize_safe_history((decision_a,))
    assert serialized == serialize_safe_history((decision_b,))
    assert b"drawn_card" not in serialized
    assert b"opponent_hand" not in serialized


def test_draw_retry_safe_mask_matches_engine_legality():
    kinds = np.full((1, 2), POLICY_RL, dtype=np.int8)
    keys = np.full((1, 2), None, dtype=object)
    engine = BatchedPesten(kinds, keys, np.asarray([84], np.uint32))
    drawn_card = int(engine.hands[0, 0, 0])
    engine.current_player[0] = 0
    engine.phase[0] = PHASE_DRAW_RETRY
    engine.drawn_card[0] = drawn_card
    cursor = SafeEventCursor(
        last_hand_sizes=tuple(int(value) for value in engine.hand_sizes[0]),
        decision_count=1,
    )

    safe_decision = build_safe_decision(engine, 0, 0, cursor)

    assert safe_decision.legal_actions[drawn_card]
    assert safe_decision.legal_actions[54]
    assert safe_decision.legal_actions == build_draw_retry_legal_actions(
        drawn_card
    )


def test_full_fixed_mixture_matches_scalar_league_exactly_on_cpu():
    torch.manual_seed(731)
    policy = NativePufferTemporalPolicy(
        hidden_size=16, num_layers=1
    ).eval()
    historical = [_snapshot(policy)]
    champion = load_ppo_agent(
        Path("ppo_actor_100run3.pth"), "cpu"
    )

    scalar_episodes, scalar_metadata = collect_league_batch(
        policy,
        champion,
        historical,
        games=10,
        seed=731,
    )
    stats = BatchRolloutStats()
    batched_episodes, batched_metadata = collect_league_batch_batched(
        policy,
        champion,
        historical,
        games=10,
        seed=731,
        schedule=LEAGUE_SCHEDULE,
        stats=stats,
    )

    assert batched_metadata == scalar_metadata
    assert batched_metadata["games"] == {
        "current_selfplay": 4,
        "historical_temporal": 2,
        "ppo_champion": 2,
        "bayesian_tree": 1,
        "smart_rule": 1,
    }
    assert len(batched_episodes) == len(scalar_episodes) == 14
    for (scalar_trajectory, scalar_won), (
        batched_trajectory,
        batched_won,
    ) in zip(scalar_episodes, batched_episodes):
        assert batched_won == scalar_won
        assert len(batched_trajectory) == len(scalar_trajectory)
        for scalar_step, batched_step in zip(
            scalar_trajectory, batched_trajectory
        ):
            assert batched_step.decision == scalar_step.decision
            assert batched_step.action == scalar_step.action
            assert batched_step.old_log_prob == pytest.approx(
                scalar_step.old_log_prob, abs=2e-6
            )
            assert batched_step.value == pytest.approx(
                scalar_step.value, abs=2e-6
            )

    assert stats.vector_games == 9
    assert stats.tree_games == 1
    assert stats.temporal_decisions >= sum(
        batched_metadata["decisions"].values()
    )
    assert stats.champion_decisions > 0
    assert stats.engine_bytes > 0


def test_explicit_no_ppo_schedule_never_requires_champion():
    torch.manual_seed(97)
    policy = NativePufferTemporalPolicy(
        hidden_size=8, num_layers=1
    ).eval()
    historical = [_snapshot(policy)]
    schedule = (
        "current_selfplay",
        "historical_temporal",
        "smart_rule",
    )
    stats = BatchRolloutStats()

    episodes, contribution = collect_league_batch_batched(
        policy,
        None,
        historical,
        games=len(schedule),
        seed=97,
        schedule=schedule,
        max_turns=60,
        stats=stats,
    )

    assert episodes
    assert contribution["games"] == {
        "current_selfplay": 1,
        "historical_temporal": 1,
        "smart_rule": 1,
    }
    assert "ppo_champion" not in contribution["games"]
    assert "ppo_champion" not in contribution["decisions"]
    assert stats.champion_decisions == 0
    assert stats.champion_forward_calls == 0
