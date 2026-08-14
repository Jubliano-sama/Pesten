"""Batched temporal-policy league rollouts with canonical Pesten semantics.

The expensive recurrent policy and frozen PPO champion are evaluated in
batches.  Bayesian-tree games deliberately remain on the canonical scalar
``Game`` implementation because their search depends on a live game object.
Episodes are merged back into scalar league order before training.

Only information already admitted by :mod:`temporal_observation` crosses the
policy boundary.  In particular, draws become public ``DRAW_COUNT`` events;
drawn card identities and draw-pile state are never included.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import random
import time
from typing import Hashable, Mapping, Sequence

import numpy as np
import torch

import game
from batched_pesten import (
    BatchedPesten,
    PHASE_DRAW_RETRY,
    PHASE_SUIT,
    POLICY_RANDOM,
    POLICY_RL,
    POLICY_SMART,
    POLICY_SMARTER,
    SUIT_NAMES,
)
from hardcodedAgent import HardcodedStrategyAgent
from ppo2 import build_action_masks
from temporal_observation import (
    DecisionInput,
    EventKind,
    PASS_ACTION,
    PublicEventToken,
    SUIT_ACTION_START,
    build_draw_retry_legal_actions,
    build_legal_actions,
    build_private_observation,
    relative_index,
)
from temporal_puffer_agent import TrajectoryStep, temporal_agent_factory


LEAGUE_SCHEDULE = (
    "current_selfplay",
    "current_selfplay",
    "current_selfplay",
    "current_selfplay",
    "historical_temporal",
    "historical_temporal",
    "ppo_champion",
    "ppo_champion",
    "bayesian_tree",
    "smart_rule",
)
# The fixed default mixture remains unchanged.  ``random`` is the existing
# audited-uniform role with an independent opponent-RNG bank.  The explicit
# ``canonical_random`` lane mirrors ``randomAgent.AgentRandom`` instead: its
# game RNG is deliberately coupled to the deal and recycle stream.
VALID_OPPONENT_TYPES = frozenset(
    (
        *LEAGUE_SCHEDULE,
        "smarter_rule",
        "random_rule",
        "random",
        "canonical_random",
    )
)

ROLE_CURRENT = "current"
ROLE_HISTORICAL = "historical"
ROLE_PPO = "ppo_champion"
ROLE_SMART = "smart_rule"
ROLE_SMARTER = "smarter_rule"
ROLE_AUDITED_UNIFORM_RANDOM = "audited_uniform_random"
ROLE_CANONICAL_RANDOM = "canonical_random"

# ``random`` is intentionally *not* the legacy ``randomAgent.AgentRandom``
# role.  The campaign specialist needs an auditable fixed opponent whose RNG
# is isolated from deck/deal randomness.  The scalar mirror lives in
# ``random_specialist_workflow``; both implementations use the helpers below.
AUDITED_UNIFORM_RANDOM_CONTRACT = {
    "id": "pesten-audited-uniform-random-v1",
    "rng_algorithm": "numpy.random.RandomState (MT19937)",
    "rng_draw": "randint(0, len(ordered_legal_actions))",
    "legal_action_order": "ascending integer action code",
    "draw_retry": "only the drawn card, then pass action 54",
    "suit_actions": "ascending action codes 50, 51, 52, 53",
    "policy_visibility": "opponent RNG seed and action sampler never cross the DecisionInput boundary",
}

CANONICAL_RANDOM_AGENT_CONTRACT = {
    "id": "randomAgent.AgentRandom-canonical-v1",
    "scalar_source": "randomAgent.AgentRandom",
    "game_rng": "per-game numpy.random.RandomState(deal_seed), shared by deal, hand shuffle, and recycle",
    "suit_rng": "per-game random.Random(deal_seed).randint(0, 3)",
    "opponent_rng_seed": None,
    "policy_visibility": "canonical opponent RNG state never crosses the DecisionInput boundary",
}


@dataclass(slots=True)
class SafeEventCursor:
    """Per-seat public-event cursor used by the recurrent safe adapter."""

    last_hand_sizes: tuple[int, ...] = ()
    public_cursor: int = 0
    decision_count: int = 0


@dataclass(slots=True)
class TemporalSeatRuntime:
    """Policy-local state for one temporal seat in one vector game."""

    game_index: int
    absolute_game_index: int
    seat: int
    policy: torch.nn.Module
    generator: torch.Generator
    state: tuple[torch.Tensor, ...]
    cursor: SafeEventCursor
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    safe_history: list[DecisionInput] = field(default_factory=list)


@dataclass(slots=True)
class BatchRolloutStats:
    """Optional profiler counters; not part of the scalar league result."""

    vector_games: int = 0
    tree_games: int = 0
    temporal_decisions: int = 0
    temporal_forward_calls: int = 0
    champion_decisions: int = 0
    champion_forward_calls: int = 0
    engine_bytes: int = 0
    vector_seconds: float = 0.0
    tree_seconds: float = 0.0
    safe_observation_event_seconds: float = 0.0
    temporal_feature_encoding_seconds: float = 0.0
    temporal_recurrent_inference_transfer_seconds: float = 0.0
    temporal_sampling_state_scatter_seconds: float = 0.0
    champion_inference_transfer_seconds: float = 0.0
    rules_stepping_seconds: float = 0.0
    max_turns: int = 0
    turn_limit_draws: int = 0


def ordered_legal_actions(
    legal_actions: Sequence[bool],
) -> tuple[int, ...]:
    """Return the contract's canonical ascending legal action order."""

    actions = tuple(
        index for index, is_legal in enumerate(legal_actions) if is_legal
    )
    if not actions:
        raise ValueError("audited uniform opponent requires a legal action")
    return actions


def sample_audited_uniform_action(
    legal_actions: Sequence[bool],
    rng: np.random.RandomState,
) -> int:
    """Sample exactly uniformly from the contract's ordered legal actions."""

    actions = ordered_legal_actions(legal_actions)
    return actions[int(rng.randint(0, len(actions)))]


def audited_uniform_legal_actions(
    engine: BatchedPesten,
    game_index: int,
) -> tuple[bool, ...]:
    """Build the contract mask at a batched-engine decision boundary.

    This is deliberately environment-side code.  It reads the current
    opponent's own cards only to choose its action and never hands that data
    to the recurrent policy.
    """

    seat = int(engine.current_player[game_index])
    phase = int(engine.phase[game_index])
    if phase == PHASE_SUIT:
        # ``build_legal_actions`` owns the public 55-action encoding.
        return build_legal_actions(
            own_cards=(),
            current_suit=SUIT_NAMES[int(engine.current_sort[game_index])],
            current_rank=int(engine.current_rank[game_index]),
            penalty_amount=int(engine.penalty_amount[game_index]),
            change_suit_mode=True,
        )
    if phase == PHASE_DRAW_RETRY:
        return build_draw_retry_legal_actions(
            int(engine.drawn_card[game_index])
        )
    return build_legal_actions(
        own_cards=_own_cards(engine, game_index, seat),
        current_suit=SUIT_NAMES[int(engine.current_sort[game_index])],
        current_rank=int(engine.current_rank[game_index]),
        penalty_amount=int(engine.penalty_amount[game_index]),
    )


def _state_sha256(state: Sequence[torch.Tensor]) -> str:
    """Hash recurrent state for a diagnostic trace outside policy inputs."""

    digest = hashlib.sha256()
    for part in state:
        value = part.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def league_schedule() -> tuple[str, ...]:
    """Return the fixed 40/20/20/10/10 opponent mixture."""

    return LEAGUE_SCHEDULE


def _hand_sizes(engine: BatchedPesten, game_index: int) -> tuple[int, ...]:
    return tuple(int(value) for value in engine.hand_sizes[game_index])


def _own_cards(
    engine: BatchedPesten,
    game_index: int,
    seat: int,
) -> tuple[int, ...]:
    size = int(engine.hand_sizes[game_index, seat])
    return tuple(
        int(number) for number in engine.hands[game_index, seat, :size]
    )


def build_safe_decision(
    engine: BatchedPesten,
    game_index: int,
    seat: int,
    cursor: SafeEventCursor,
) -> DecisionInput:
    """Build the same safe decision/event stream as ``TemporalPestenAgent``."""

    player_count = engine.num_players
    direction = int(engine.direction[game_index])
    current_sizes = _hand_sizes(engine, game_index)
    # The scalar adapter first establishes its episode marker at the player's
    # first decision, after any earlier seats may already have acted.
    if cursor.decision_count == 0:
        cursor.last_hand_sizes = current_sizes
    events: list[PublicEventToken] = []
    if cursor.public_cursor == 0 and cursor.decision_count == 0:
        events.append(PublicEventToken(EventKind.EPISODE_START))

    predicted_sizes = list(cursor.last_hand_sizes)
    history = engine.public_events[game_index]
    for item in history[cursor.public_cursor :]:
        action = item[0]
        actor = int(item[1])
        actor_relative = relative_index(
            actor, seat, player_count, direction
        )
        if action == "play":
            number = int(item[2])
            events.append(
                PublicEventToken(
                    EventKind.PLAY,
                    actor_relative=actor_relative,
                    card_id=number,
                    suit_id=int(number // 12 if number < 48 else 4),
                    count=1,
                )
            )
            if 0 <= actor < len(predicted_sizes):
                predicted_sizes[actor] -= 1
        elif action == "choose_sort":
            events.append(
                PublicEventToken(
                    EventKind.CHOOSE_SUIT,
                    actor_relative=actor_relative,
                    suit_id=int(item[2]),
                )
            )

    for actor, (predicted, current) in enumerate(
        zip(predicted_sizes, current_sizes)
    ):
        draw_count = current - predicted
        if draw_count > 0:
            events.append(
                PublicEventToken(
                    EventKind.DRAW_COUNT,
                    actor_relative=relative_index(
                        actor, seat, player_count, direction
                    ),
                    count=draw_count,
                )
            )

    cursor.public_cursor = len(history)
    cursor.last_hand_sizes = current_sizes
    cursor.decision_count += 1

    own_cards = _own_cards(engine, game_index, seat)
    suit_mode = int(engine.phase[game_index]) == PHASE_SUIT
    current_suit = SUIT_NAMES[int(engine.current_sort[game_index])]
    current_rank = int(engine.current_rank[game_index])
    penalty_amount = int(engine.penalty_amount[game_index])
    observation = build_private_observation(
        own_cards=own_cards,
        current_suit=current_suit,
        current_rank=current_rank,
        public_hand_sizes=current_sizes,
        observer_index=seat,
        current_player_index=int(engine.current_player[game_index]),
        direction=direction,
        penalty_amount=penalty_amount,
        public_sorts_played=tuple(
            int(value) for value in engine.sorts_played[game_index]
        ),
        change_suit_mode=suit_mode,
    )
    legal = build_legal_actions(
        own_cards=own_cards,
        current_suit=current_suit,
        current_rank=current_rank,
        penalty_amount=penalty_amount,
        change_suit_mode=suit_mode,
    )
    if int(engine.phase[game_index]) == PHASE_DRAW_RETRY:
        # The legacy engine permits only the just-drawn card or pass during
        # this immediate retry. The drawn card is the acting player's own
        # card; keep it in the action mask without exposing it as a public
        # event or observation feature.
        drawn_card = int(engine.drawn_card[game_index])
        legal = build_draw_retry_legal_actions(drawn_card)
    return DecisionInput(observation, legal, tuple(events))


def _stack_states(
    runtimes: Sequence[TemporalSeatRuntime],
) -> tuple[torch.Tensor, ...]:
    if not runtimes:
        raise ValueError("cannot stack an empty recurrent-state batch")
    part_count = len(runtimes[0].state)
    if any(len(runtime.state) != part_count for runtime in runtimes):
        raise ValueError("recurrent state structures differ within a policy")
    return tuple(
        torch.cat([runtime.state[part] for runtime in runtimes], dim=1)
        for part in range(part_count)
    )


def _scatter_states(
    runtimes: Sequence[TemporalSeatRuntime],
    states: tuple[torch.Tensor, ...],
) -> None:
    for batch_index, runtime in enumerate(runtimes):
        runtime.state = tuple(
            part[:, batch_index : batch_index + 1].detach()
            for part in states
        )


def _batch_temporal_actions(
    policy: torch.nn.Module,
    entries: Sequence[tuple[TemporalSeatRuntime, DecisionInput]],
    stats: BatchRolloutStats | None,
    *,
    deterministic: bool = False,
    disable_strategic_pass: bool = False,
    strategic_pass_max_hand_size: int | None = None,
    engine: BatchedPesten | None = None,
    roles: Mapping[
        tuple[int, int], tuple[str, torch.nn.Module | None, int]
    ] | None = None,
    all_runtimes: Mapping[tuple[int, int], TemporalSeatRuntime] | None = None,
    privileged_critic_runtime: object | None = None,
    capture_privileged_snapshots: bool = False,
) -> list[int]:
    """Advance variable-length event streams, then sample one action per seat."""

    if not entries:
        return []
    capture_privileged = (
        capture_privileged_snapshots or privileged_critic_runtime is not None
    )
    if capture_privileged and (
        engine is None or roles is None or all_runtimes is None
    ):
        raise ValueError(
            "privileged temporal capture requires engine, roles, and all runtimes"
        )
    parameter = next(policy.parameters())
    device, dtype = parameter.device, parameter.dtype
    runtimes = [entry[0] for entry in entries]
    decisions = [entry[1] for entry in entries]
    feature_started = time.perf_counter()
    event_rows = [policy._event_features(decision) for decision in decisions]
    if stats is not None:
        stats.temporal_feature_encoding_seconds += (
            time.perf_counter() - feature_started
        )

    inference_started = time.perf_counter()
    with torch.inference_mode():
        max_events = max((len(rows) for rows in event_rows), default=0)
        for event_index in range(max_events):
            selected = [
                index
                for index, rows in enumerate(event_rows)
                if event_index < len(rows)
            ]
            selected_runtimes = [runtimes[index] for index in selected]
            tokens = torch.tensor(
                [event_rows[index][event_index] for index in selected],
                device=device,
                dtype=dtype,
            )
            _, _, next_states = policy.puffer_policy.forward_eval(
                tokens, _stack_states(selected_runtimes)
            )
            _scatter_states(selected_runtimes, next_states)
            if stats is not None:
                stats.temporal_forward_calls += 1

        tokens = torch.tensor(
            [policy._decision_features(decision) for decision in decisions],
            device=device,
            dtype=dtype,
        )
        logits, values, next_states = policy.puffer_policy.forward_eval(
            tokens, _stack_states(runtimes)
        )
        legal = torch.tensor(
            [decision.legal_actions for decision in decisions],
            device=device,
            dtype=torch.bool,
        )
        if disable_strategic_pass and strategic_pass_max_hand_size is not None:
            raise ValueError(
                "disable_strategic_pass and strategic_pass_max_hand_size "
                "are mutually exclusive"
            )
        if disable_strategic_pass or strategic_pass_max_hand_size is not None:
            # Keep PASS legal when it is the only card-phase action (forced
            # draw or penalty acceptance).  For an ablation, remove strategic
            # pass either universally or above a safe observed hand-size cap.
            # Suit decisions are unaffected.
            has_playable_card = torch.any(legal[..., :50], dim=-1)
            strategic_pass_allowed = torch.zeros_like(has_playable_card)
            if strategic_pass_max_hand_size is not None:
                strategic_pass_allowed = torch.tensor(
                    [
                        decision.observation.values[91]
                        <= strategic_pass_max_hand_size
                        for decision in decisions
                    ],
                    device=device,
                    dtype=torch.bool,
                )
            legal[..., PASS_ACTION] &= (
                ~has_playable_card | strategic_pass_allowed
            )
        masked_logits = logits.masked_fill(
            ~legal, torch.finfo(logits.dtype).min
        )
        probabilities = (
            torch.softmax(masked_logits.float(), dim=-1).cpu()
            if not deterministic or capture_privileged
            else None
        )
        if stats is not None:
            stats.temporal_recurrent_inference_transfer_seconds += (
                time.perf_counter() - inference_started
            )
        log_probabilities = torch.log_softmax(masked_logits, dim=-1)
        scatter_started = time.perf_counter()
        _scatter_states(runtimes, next_states)
        scatter_seconds = time.perf_counter() - scatter_started

    privileged_by_index: dict[int, object] = {}
    privileged_values: dict[int, float] = {}
    if capture_privileged:
        # Local import preserves the actor module's one-way information-safe
        # dependency boundary; the privileged module itself uses this rollout
        # helper for auditable trace replay.
        from privileged_critic import capture_privileged_snapshot

        snapshots = []
        snapshot_indices = []
        stream_ids = []
        assert probabilities is not None
        assert engine is not None and roles is not None
        assert all_runtimes is not None
        for index, (runtime, decision) in enumerate(entries):
            role_name = roles[(runtime.game_index, runtime.seat)][0]
            if role_name != ROLE_CURRENT:
                continue
            absolute_roles = tuple(
                roles[(runtime.game_index, seat)][0]
                for seat in range(engine.num_players)
            )
            memories = tuple(
                (
                    all_runtimes[(runtime.game_index, seat)].state
                    if (runtime.game_index, seat) in all_runtimes
                    else None
                )
                for seat in range(engine.num_players)
            )
            snapshot = capture_privileged_snapshot(
                engine,
                runtime.game_index,
                runtime.seat,
                safe_observation=decision.observation.values,
                absolute_roles=absolute_roles,
                actor_probabilities=probabilities[index].numpy(),
                actor_memory=runtime.state,
                public_value=float(values[index].reshape(-1)[0].item()),
                all_actor_memories=memories,
            )
            snapshots.append(snapshot)
            snapshot_indices.append(index)
            stream_ids.append((runtime.absolute_game_index, runtime.seat))
            privileged_by_index[index] = snapshot
        if privileged_critic_runtime is not None and snapshots:
            outputs = privileged_critic_runtime.evaluate(snapshots, stream_ids)
            for index, value in zip(
                snapshot_indices, outputs["value"].tolist()
            ):
                privileged_values[index] = float(value)

    # Sampling happens only after the immutable baseline inputs and values have
    # been captured.  This makes the chronology match the no-action schema,
    # rather than merely relying on the snapshot builder not to accept actions.
    sampling_started = time.perf_counter()
    if deterministic:
        actions = masked_logits.argmax(dim=-1).cpu().tolist()
    else:
        assert probabilities is not None
        actions = [
            int(
                torch.multinomial(
                    probabilities[index],
                    1,
                    generator=runtime.generator,
                ).item()
            )
            for index, runtime in enumerate(runtimes)
        ]

    for index, (runtime, decision) in enumerate(entries):
        action = actions[index]
        runtime.safe_history.append(decision)
        runtime.trajectory.append(
            TrajectoryStep(
                decision=decision,
                action=action,
                old_log_prob=float(
                    log_probabilities[index, action].item()
                ),
                value=privileged_values.get(
                    index, float(values[index].reshape(-1)[0].item())
                ),
                privileged_snapshot=privileged_by_index.get(index),
            )
        )
    if stats is not None:
        stats.temporal_decisions += len(entries)
        stats.temporal_forward_calls += 1
        stats.temporal_sampling_state_scatter_seconds += (
            scatter_seconds + time.perf_counter() - sampling_started
        )
    return actions


def _champion_actor_and_device(champion_factory):
    actor = getattr(champion_factory, "shared_actor", None)
    device = getattr(champion_factory, "shared_device", None)
    if actor is None and isinstance(champion_factory, torch.nn.Module):
        actor = champion_factory
    if actor is None:
        raise ValueError(
            "champion_factory must expose shared_actor or be a torch module"
        )
    if device is None:
        device = next(actor.parameters()).device
    return actor, torch.device(device)


def _batch_champion_actions(
    engine: BatchedPesten,
    game_indices: Sequence[int],
    champion_factory,
    stats: BatchRolloutStats | None,
) -> list[int]:
    """Run the frozen legacy champion as one masked greedy batch."""

    if not game_indices:
        return []
    started = time.perf_counter()
    actor, device = _champion_actor_and_device(champion_factory)
    observations = torch.from_numpy(
        np.stack(
            [
                engine.observation(
                    game_index,
                    suit_mode=(
                        int(engine.phase[game_index]) == PHASE_SUIT
                    ),
                )
                for game_index in game_indices
            ]
        )
    ).to(device=device)
    with torch.inference_mode():
        logits = actor(observations)
        legal = build_action_masks(observations)
        actions = (
            logits.masked_fill(
                ~legal, torch.finfo(logits.dtype).min
            )
            .argmax(dim=-1)
            .cpu()
            .tolist()
        )
    if stats is not None:
        stats.champion_decisions += len(game_indices)
        stats.champion_forward_calls += 1
        stats.champion_inference_transfer_seconds += (
            time.perf_counter() - started
        )
    return [int(action) for action in actions]


def _champion_has_card_decision(
    engine: BatchedPesten,
    game_index: int,
) -> bool:
    """Match legacy PPO's compatible-card check, including duplicate cards."""

    observation = engine.observation(game_index)
    return bool(np.any(observation[50:100] >= 1))


def _tree_factory(game_instance):
    return HardcodedStrategyAgent(
        game_instance,
        belief_samples=8,
        search_depth=3,
        beam_width=3,
    )


def _collect_scalar_tree_game(
    policy: torch.nn.Module,
    *,
    absolute_game_index: int,
    deal_seed: int,
    player_count: int,
    max_turns: int,
) -> tuple[list[tuple[Sequence[TrajectoryStep], bool]], int]:
    np.random.seed(deal_seed)
    random.seed(deal_seed)
    current_factory = temporal_agent_factory(
        policy, deterministic=False, seed=deal_seed * 2
    )
    current_seat = absolute_game_index % player_count
    factories = [
        current_factory if seat == current_seat else _tree_factory
        for seat in range(player_count)
    ]
    match = game.Game(factories)
    turns, winner = match.auto_simulate(max_turns=max_turns)
    trajectory = tuple(match.players[current_seat].trajectory)
    return [(trajectory, winner == current_seat)], turns, winner


def _make_vector_layout(
    *,
    game_indices: Sequence[int],
    games: int,
    seed: int,
    schedule_offset: int,
    schedule: Sequence[str],
    historical: Sequence[torch.nn.Module],
    random_opponent_seeds: Sequence[int] | None,
    player_count: int,
) -> tuple[
    list[int],
    np.ndarray,
    np.ndarray,
    dict[tuple[int, int], tuple[str, torch.nn.Module | None, int]],
]:
    vector_absolute_indices = [
        index
        for index in game_indices
        if not (
            schedule[(schedule_offset + index) % len(schedule)]
            == "bayesian_tree"
            and player_count < 7
        )
    ]
    policy_kinds = np.full(
        (len(vector_absolute_indices), player_count), POLICY_RL, np.int8
    )
    actor_keys = np.empty(
        (len(vector_absolute_indices), player_count), dtype=object
    )
    roles: dict[
        tuple[int, int], tuple[str, torch.nn.Module | None, int]
    ] = {}

    if "random" in schedule:
        if random_opponent_seeds is None:
            raise ValueError(
                "random schedule entries require one independent "
                "random_opponent_seed per requested game"
            )
        if len(random_opponent_seeds) != games:
            raise ValueError(
                "random_opponent_seeds must contain exactly `games` entries"
            )
        if any(
            not 0 <= int(value) < 2**32
            for value in random_opponent_seeds
        ):
            raise ValueError(
                "random_opponent_seeds must be uint32-compatible"
            )

    # ``canonical_random`` deliberately has no independent opponent stream:
    # its only NumPy generator is the per-game deck generator and its suit
    # stream is derived directly from the deal seed.  Rejecting a supplied
    # bank here avoids a superficially harmless parameter silently becoming
    # part of the canonical contract later.
    if (
        {"canonical_random", "random_rule"}.intersection(schedule)
        and random_opponent_seeds is not None
    ):
        raise ValueError(
            "canonical_random schedule entries must not receive an "
            "independent random_opponent_seed bank"
        )

    for vector_index, absolute_index in enumerate(vector_absolute_indices):
        opponent_type = schedule[
            (schedule_offset + absolute_index) % len(schedule)
        ]
        if opponent_type == "bayesian_tree" and player_count >= 7:
            opponent_type = "smarter_rule"
        deal_seed = seed + absolute_index
        if opponent_type == "current_selfplay":
            seat_roles = tuple(
                (ROLE_CURRENT, None, deal_seed * 2 + seat)
                for seat in range(player_count)
            )
        else:
            current_seat = absolute_index % player_count
            if opponent_type == "historical_temporal":
                if not historical:
                    raise ValueError(
                        "historical snapshots are required by the league mix"
                    )
                # Mix the deal namespace into historical selection so a short
                # opponent schedule cannot lock onto only a residue subset of
                # a bounded snapshot pool. This remains deterministic and
                # covers every retained generation over long campaigns.
                historical_index = (
                    absolute_index * 0x9E3779B1 + seed
                ) % len(historical)
                opponent_policy = historical[historical_index]
                opponent_role = (
                    ROLE_HISTORICAL,
                    opponent_policy,
                    deal_seed + 99,
                )
            elif opponent_type == "ppo_champion":
                opponent_role = (ROLE_PPO, None, 0)
            elif opponent_type == "smart_rule":
                opponent_role = (ROLE_SMART, None, 0)
            elif opponent_type == "smarter_rule":
                opponent_role = (ROLE_SMARTER, None, 0)
            elif opponent_type == "random":
                assert random_opponent_seeds is not None
                opponent_role = (
                    ROLE_AUDITED_UNIFORM_RANDOM,
                    None,
                    int(random_opponent_seeds[absolute_index]),
                )
            elif opponent_type in ("canonical_random", "random_rule"):
                # The integer is the deal seed, not an independent opponent
                # namespace.  BatchedPesten owns the matching per-game NumPy
                # stream and shares it with deal/recycle/random hand shuffles.
                opponent_role = (ROLE_CANONICAL_RANDOM, None, deal_seed)
            else:
                raise AssertionError(f"unexpected vector role {opponent_type}")
            current_role = (ROLE_CURRENT, None, deal_seed * 2)
            seat_roles = tuple(
                current_role if seat == current_seat else opponent_role
                for seat in range(player_count)
            )

        for seat, role in enumerate(seat_roles):
            role_name, _, _ = role
            roles[(vector_index, seat)] = role
            actor_keys[vector_index, seat] = (
                role_name,
                id(role[1]) if role[1] is not None else 0,
            )
            if role_name == ROLE_SMART:
                policy_kinds[vector_index, seat] = POLICY_SMART
            elif role_name == ROLE_SMARTER:
                policy_kinds[vector_index, seat] = POLICY_SMARTER
            elif role_name == ROLE_CANONICAL_RANDOM:
                policy_kinds[vector_index, seat] = POLICY_RANDOM

    seeds = np.asarray(
        [seed + index for index in vector_absolute_indices],
        dtype=np.uint32,
    )
    return vector_absolute_indices, policy_kinds, actor_keys, roles


def collect_league_batch_batched(
    policy: torch.nn.Module,
    champion_factory,
    historical: Sequence[torch.nn.Module],
    *,
    games: int,
    seed: int,
    schedule_offset: int = 0,
    schedule: Sequence[str] = LEAGUE_SCHEDULE,
    random_opponent_seeds: Sequence[int] | None = None,
    max_turns: int = 1000,
    stats: BatchRolloutStats | None = None,
    trace_sink: list[dict] | None = None,
    player_count: int = 2,
    game_indices: Sequence[int] | None = None,
    privileged_critic: object | None = None,
    capture_privileged_snapshots: bool = False,
) -> tuple[list[tuple[Sequence[TrajectoryStep], bool]], dict]:
    """Collect an explicit league schedule with batched neural decisions.

    The default retains the frozen 40/20/20/10/10 equivalence benchmark.
    Training recipes should pass ``schedule`` explicitly; omitted opponent
    types are never constructed or evaluated.  Opt-in ``random`` entries use
    the separately seeded audited-uniform contract; opt-in
    ``canonical_random`` is the separately named legacy ``AgentRandom``
    contract with its coupled deal/opponent RNG.  ``trace_sink`` is an opt-in
    phase-zero diagnostic: it remains environment-side and never changes the
    return schema or policy inputs.  ``privileged_critic`` is an opt-in,
    environment-side baseline: its pre-action values replace only
    ``TrajectoryStep.value`` for GAE, while the actor continues to receive the
    exact same safe ``DecisionInput``.  Captured snapshots can be retained for
    an external critic update without entering actor tokens.
    """

    if games < 0:
        raise ValueError("games must be non-negative")
    if player_count < 2:
        raise ValueError("player_count must be at least two")
    selected_game_indices = tuple(
        range(games) if game_indices is None else game_indices
    )
    if (
        tuple(sorted(set(selected_game_indices))) != selected_game_indices
        or any(index < 0 or index >= games for index in selected_game_indices)
    ):
        raise ValueError(
            "game_indices must be unique, sorted, and inside the game range"
        )
    schedule = tuple(schedule)
    if not schedule:
        raise ValueError("schedule must contain at least one opponent type")
    unknown_opponents = set(schedule) - VALID_OPPONENT_TYPES
    if unknown_opponents:
        raise ValueError(
            "unknown opponent types: "
            + ", ".join(sorted(unknown_opponents))
        )
    if (
        privileged_critic is not None or capture_privileged_snapshots
    ) and "bayesian_tree" in schedule and player_count < 7:
        raise ValueError(
            "privileged critic capture is unavailable for scalar bayesian-tree "
            "games; use the vector opponent lanes"
        )
    if stats is None:
        stats = BatchRolloutStats()
    privileged_runtime = None
    if privileged_critic is not None:
        from privileged_critic import (
            AveragedPrivilegedCritic,
            AveragedPrivilegedCriticRuntime,
            PrivilegedCriticRuntime,
        )

        if hasattr(privileged_critic, "evaluate"):
            privileged_runtime = privileged_critic
        elif isinstance(privileged_critic, AveragedPrivilegedCritic):
            privileged_runtime = AveragedPrivilegedCriticRuntime(
                privileged_critic
            )
        else:
            privileged_runtime = PrivilegedCriticRuntime(privileged_critic)
    (
        vector_absolute_indices,
        policy_kinds,
        actor_keys,
        roles,
    ) = _make_vector_layout(
        game_indices=selected_game_indices,
        games=games,
        seed=seed,
        schedule_offset=schedule_offset,
        schedule=schedule,
        historical=historical,
        random_opponent_seeds=random_opponent_seeds,
        player_count=player_count,
    )
    absolute_to_vector = {
        absolute: vector
        for vector, absolute in enumerate(vector_absolute_indices)
    }
    episodes_by_game: dict[
        int, list[tuple[Sequence[TrajectoryStep], bool]]
    ] = {}
    contributions = Counter()
    game_counts = Counter()
    total_turns = 0
    turn_limit_draws = 0

    stats.vector_games = len(vector_absolute_indices)
    stats.tree_games = len(selected_game_indices) - len(vector_absolute_indices)
    stats.max_turns = max_turns
    if vector_absolute_indices:
        engine = BatchedPesten(
            policy_kinds,
            actor_keys,
            np.asarray(
                [seed + index for index in vector_absolute_indices],
                dtype=np.uint32,
            ),
            max_turns=max_turns,
            smart_agents_have_game_reference=True,
        )
        stats.engine_bytes = engine.allocated_bytes
        runtimes: dict[tuple[int, int], TemporalSeatRuntime] = {}
        random_generators: dict[int, np.random.RandomState] = {}
        canonical_suit_generators: dict[int, random.Random] = {}
        canonical_trace_vectors: set[int] = set()
        trace_by_vector: dict[int, dict] = {}
        if trace_sink is not None:
            for vector_index, absolute_index in enumerate(
                vector_absolute_indices
            ):
                opening_count = int(engine.discard_counts[vector_index])
                trace_by_vector[vector_index] = {
                    "absolute_game_index": absolute_index,
                    "deal_seed": int(seed + absolute_index),
                    "initial_deal": {
                        "hands": [
                            list(_own_cards(engine, vector_index, seat))
                            for seat in range(player_count)
                        ],
                        "opening_card": int(
                            engine.discards[vector_index, opening_count - 1]
                        ),
                    },
                    "actions": [],
                    "public_events": (),
                    "recurrent": [],
                }
        for (vector_index, seat), role in roles.items():
            role_name, opponent_policy, agent_seed = role
            if role_name == ROLE_AUDITED_UNIFORM_RANDOM:
                random_generators[vector_index] = np.random.RandomState(
                    agent_seed
                )
                if trace_sink is not None:
                    trace_by_vector[vector_index][
                        "opponent_rng_seed"
                    ] = int(agent_seed)
                continue
            if role_name == ROLE_CANONICAL_RANDOM:
                # ``engine.rngs[vector_index]`` is already seeded with this
                # deal seed and remains the sole NumPy stream for initial deal,
                # AgentRandom hand shuffles, and draw-pile recycle.  Scalar
                # AgentRandom uses Python's module RNG only for changeSort.
                canonical_suit_generators[vector_index] = random.Random(
                    agent_seed
                )
                canonical_trace_vectors.add(vector_index)
                if trace_sink is not None:
                    trace_by_vector[vector_index]["canonical_random_rng"] = {
                        "deal_seed_shared_with_engine": int(agent_seed),
                        "suit_rng": "random.Random(deal_seed).randint(0, 3)",
                        "independent_opponent_rng": False,
                    }
                continue
            if role_name not in (ROLE_CURRENT, ROLE_HISTORICAL):
                continue
            seat_policy = (
                policy if role_name == ROLE_CURRENT else opponent_policy
            )
            assert seat_policy is not None
            runtimes[(vector_index, seat)] = TemporalSeatRuntime(
                game_index=vector_index,
                absolute_game_index=vector_absolute_indices[vector_index],
                seat=seat,
                policy=seat_policy,
                generator=torch.Generator(device="cpu").manual_seed(
                    agent_seed
                ),
                state=seat_policy.initial_state(),
                cursor=SafeEventCursor(),
            )
            if trace_sink is not None:
                trace_by_vector[vector_index]["recurrent"].append(
                    {
                        "seat": seat,
                        "role": role_name,
                        "initial_state_sha256": _state_sha256(
                            runtimes[(vector_index, seat)].state
                        ),
                    }
                )

        vector_started = time.perf_counter()
        while True:
            active = engine.active_games()
            if len(active) == 0:
                break
            actions: dict[int, int] = {}
            temporal_groups: dict[
                int,
                tuple[
                    torch.nn.Module,
                    list[tuple[TemporalSeatRuntime, DecisionInput]],
                ],
            ] = {}
            champion_games: list[int] = []
            pending_decisions: dict[int, DecisionInput] = {}
            random_masks: dict[int, tuple[bool, ...]] = {}
            canonical_hand_orders: dict[int, list[int]] = {}

            for vector_value in active:
                vector_index = int(vector_value)
                seat = int(engine.current_player[vector_index])
                role_name, _, _ = roles[(vector_index, seat)]
                if role_name in (ROLE_CURRENT, ROLE_HISTORICAL):
                    runtime = runtimes[(vector_index, seat)]
                    safe_started = time.perf_counter()
                    decision = build_safe_decision(
                        engine, vector_index, seat, runtime.cursor
                    )
                    stats.safe_observation_event_seconds += (
                        time.perf_counter() - safe_started
                    )
                    group = temporal_groups.setdefault(
                        id(runtime.policy), (runtime.policy, [])
                    )
                    group[1].append((runtime, decision))
                    pending_decisions[vector_index] = decision
                elif role_name == ROLE_PPO:
                    if (
                        int(engine.phase[vector_index]) != PHASE_SUIT
                        and not _champion_has_card_decision(
                            engine, vector_index
                        )
                    ):
                        actions[vector_index] = 54
                    else:
                        champion_games.append(vector_index)
                else:
                    rules_started = time.perf_counter()
                    if role_name == ROLE_AUDITED_UNIFORM_RANDOM:
                        legal = audited_uniform_legal_actions(
                            engine, vector_index
                        )
                        random_masks[vector_index] = legal
                        actions[vector_index] = sample_audited_uniform_action(
                            legal, random_generators[vector_index]
                        )
                    elif role_name == ROLE_CANONICAL_RANDOM:
                        if int(engine.phase[vector_index]) == PHASE_SUIT:
                            legal = audited_uniform_legal_actions(
                                engine, vector_index
                            )
                            random_masks[vector_index] = legal
                            actions[vector_index] = (
                                SUIT_ACTION_START
                                + canonical_suit_generators[vector_index].randint(
                                    0, 3
                                )
                            )
                        else:
                            # Reuse BatchedPesten.POLICY_RANDOM's exact
                            # in-place hand shuffle.  The resulting hand order
                            # is diagnostic evidence and catches accidental
                            # use of an independent opponent RNG stream.
                            actions[vector_index] = engine.random_action(
                                vector_index
                            )
                            canonical_hand_orders[vector_index] = list(
                                _own_cards(
                                    engine,
                                    vector_index,
                                    int(engine.current_player[vector_index]),
                                )
                            )
                    else:
                        actions[vector_index] = engine.non_neural_action(
                            vector_index
                        )
                    stats.rules_stepping_seconds += (
                        time.perf_counter() - rules_started
                    )

            for _, (seat_policy, entries) in temporal_groups.items():
                chosen = _batch_temporal_actions(
                    seat_policy,
                    entries,
                    stats,
                    engine=engine,
                    roles=roles,
                    all_runtimes=runtimes,
                    privileged_critic_runtime=privileged_runtime,
                    capture_privileged_snapshots=(
                        capture_privileged_snapshots
                    ),
                )
                for (runtime, _), action in zip(entries, chosen):
                    actions[runtime.game_index] = action

            if champion_games:
                champion_actions = _batch_champion_actions(
                    engine, champion_games, champion_factory, stats
                )
                actions.update(zip(champion_games, champion_actions))
            if len(actions) != len(active):
                raise RuntimeError(
                    "each active game must produce exactly one action"
                )
            if trace_sink is not None:
                for vector_value in active:
                    vector_index = int(vector_value)
                    seat = int(engine.current_player[vector_index])
                    role_name, _, _ = roles[(vector_index, seat)]
                    entry = {
                        "seat": seat,
                        "role": role_name,
                        "phase": int(engine.phase[vector_index]),
                        "action": int(actions[vector_index]),
                    }
                    if vector_index in canonical_trace_vectors:
                        entry.update(
                            {
                                "penalty_amount": int(
                                    engine.penalty_amount[vector_index]
                                ),
                                "current_sort": int(
                                    engine.current_sort[vector_index]
                                ),
                                "current_rank": int(
                                    engine.current_rank[vector_index]
                                ),
                            }
                        )
                    if vector_index in pending_decisions:
                        entry["legal_actions"] = list(
                            pending_decisions[
                                vector_index
                            ].legal_actions
                        )
                    if vector_index in random_masks:
                        entry["legal_actions"] = list(
                            random_masks[vector_index]
                        )
                    if vector_index in canonical_hand_orders:
                        entry["opponent_hand_order"] = canonical_hand_orders[
                            vector_index
                        ]
                    trace_by_vector[vector_index]["actions"].append(entry)
            rules_started = time.perf_counter()
            for vector_value in active:
                vector_index = int(vector_value)
                engine.apply_action(vector_index, actions[vector_index])
            stats.rules_stepping_seconds += (
                time.perf_counter() - rules_started
            )

        stats.vector_seconds = time.perf_counter() - vector_started
        total_turns += int(engine.turn_counts.sum())
        for vector_index, absolute_index in enumerate(
            vector_absolute_indices
        ):
            opponent_type = schedule[
                (schedule_offset + absolute_index) % len(schedule)
            ]
            if opponent_type == "bayesian_tree" and player_count >= 7:
                opponent_type = "smarter_rule"
            game_counts[opponent_type] += 1
            winner = int(engine.winners[vector_index])
            if (
                winner < 0
                and int(engine.turn_counts[vector_index]) >= max_turns
            ):
                turn_limit_draws += 1
            current_seats = [
                seat
                for seat in range(player_count)
                if roles[(vector_index, seat)][0] == ROLE_CURRENT
            ]
            game_episodes = []
            for seat in current_seats:
                trajectory = tuple(runtimes[(vector_index, seat)].trajectory)
                game_episodes.append((trajectory, winner == seat))
                contributions[opponent_type] += len(trajectory)
            episodes_by_game[absolute_index] = game_episodes
            if trace_sink is not None:
                trace = trace_by_vector[vector_index]
                trace["winner"] = winner
                trace["turn_count"] = int(engine.turn_counts[vector_index])
                trace["public_events"] = tuple(
                    tuple(item) for item in engine.public_events[vector_index]
                )
                for record in trace["recurrent"]:
                    runtime = runtimes[(vector_index, record["seat"])]
                    record["final_state_sha256"] = _state_sha256(
                        runtime.state
                    )
                    record["decision_count"] = len(runtime.trajectory)
                trace_sink.append(trace)

    tree_started = time.perf_counter()
    for absolute_index in selected_game_indices:
        opponent_type = schedule[
            (schedule_offset + absolute_index) % len(schedule)
        ]
        if opponent_type != "bayesian_tree" or player_count >= 7:
            continue
        tree_episodes, turns, winner = _collect_scalar_tree_game(
            policy,
            absolute_game_index=absolute_index,
            deal_seed=seed + absolute_index,
            player_count=player_count,
            max_turns=max_turns,
        )
        episodes_by_game[absolute_index] = tree_episodes
        total_turns += turns
        if winner is None and turns >= max_turns:
            turn_limit_draws += 1
        game_counts[opponent_type] += 1
        contributions[opponent_type] += sum(
            len(trajectory) for trajectory, _ in tree_episodes
        )
    stats.tree_seconds = time.perf_counter() - tree_started
    stats.turn_limit_draws = turn_limit_draws

    episodes = [
        episode
        for absolute_index in selected_game_indices
        for episode in episodes_by_game.get(absolute_index, ())
    ]
    return episodes, {
        "games": dict(game_counts),
        "decisions": dict(contributions),
        "total_turns": total_turns,
        "trajectories": len(episodes),
    }
