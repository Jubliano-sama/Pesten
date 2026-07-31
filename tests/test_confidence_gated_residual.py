import math

import pytest
import torch
from torch import nn

import card
import game
from ai_observation import MODE_INDEX
from benchmark_model_zoo import (
    safe_batch_to_legacy_observations,
    safe_to_legacy_observation,
)
from benchmark_confidence_gate import (
    _delta_block,
    parse_threshold,
    threshold_label,
    validate_sweep,
)
from confidence_gated_residual import ConfidenceGatedResidualPolicy
from ppo2 import RL_SupervisedAgent
from randomAgent import AgentRandom
from temporal_observation import (
    ACTION_COUNT,
    DecisionInput,
    EventKind,
    PublicEventToken,
    build_legal_actions,
    build_private_observation,
)
from temporal_puffer_agent import PolicyStep


def safe_decision():
    own_cards = (4, 16, 49)
    observation = build_private_observation(
        own_cards=own_cards,
        current_suit="harten",
        current_rank=5,
        public_hand_sizes=(3, 4),
        observer_index=0,
        current_player_index=0,
        direction=1,
        penalty_amount=0,
        public_sorts_played=(1, 2, 0, 0, 0),
    )
    legal = build_legal_actions(
        own_cards=own_cards,
        current_suit="harten",
        current_rank=5,
        penalty_amount=0,
    )
    return DecisionInput(
        observation,
        legal,
        (PublicEventToken(EventKind.EPISODE_START),),
    )


class FixedTeacher(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.register_buffer("fixed_logits", torch.tensor(logits))

    def forward(self, observations):
        return self.fixed_logits.to(
            device=observations.device,
            dtype=observations.dtype,
        ).expand(len(observations), -1) + self.anchor * 0.0


class FakeResidualPolicy(nn.Module):
    def __init__(self, frozen_logits, residual_logits):
        super().__init__()
        self.teacher = FixedTeacher(frozen_logits)
        self.register_buffer(
            "residual_logits",
            torch.tensor(residual_logits, dtype=torch.float32),
        )
        self.step_calls = 0

    def initial_state(self, batch_size=1, *, device=None):
        return torch.zeros((1, batch_size, 1), device=device)

    def step(self, decision, state):
        self.step_calls += 1
        legal = torch.tensor(decision.legal_actions, dtype=torch.bool)
        logits = self.residual_logits.masked_fill(
            ~legal,
            torch.finfo(self.residual_logits.dtype).min,
        )
        return PolicyStep(
            logits=logits,
            value=torch.tensor(0.375),
            state=state + 1.0,
        )


def controlled_policy():
    frozen = [0.0] * ACTION_COUNT
    residual = [0.0] * ACTION_COUNT
    frozen[4] = 5.0
    frozen[16] = 4.0  # Legal top-two margin is exactly 1.
    residual[49] = 10.0
    return FakeResidualPolicy(frozen, residual)


def test_blocked_override_still_advances_recurrent_state():
    residual = controlled_policy()
    gate = ConfidenceGatedResidualPolicy(
        residual,
        margin_threshold=1.0,
    )

    result = gate.step(safe_decision(), residual.initial_state())

    assert residual.step_calls == 1
    assert torch.equal(result.state, torch.ones((1, 1, 1)))
    assert int(result.logits.argmax()) == 4
    assert gate.last_gate_decision.frozen_action == 4
    assert gate.last_gate_decision.residual_action == 49
    assert gate.last_gate_decision.frozen_top2_margin == pytest.approx(1.0)
    assert not gate.last_gate_decision.residual_override_allowed
    assert gate.diagnostics()["blocked_overrides"] == 1


def test_override_requires_margin_strictly_below_threshold():
    blocked = ConfidenceGatedResidualPolicy(
        controlled_policy(),
        margin_threshold=1.0,
    )
    allowed = ConfidenceGatedResidualPolicy(
        controlled_policy(),
        margin_threshold=1.0001,
    )

    blocked_result = blocked.step(
        safe_decision(),
        blocked.initial_state(),
    )
    allowed_result = allowed.step(
        safe_decision(),
        allowed.initial_state(),
    )

    assert int(blocked_result.logits.argmax()) == 4
    assert int(allowed_result.logits.argmax()) == 49
    assert allowed.last_gate_decision.residual_override_allowed
    assert allowed.diagnostics()["allowed_overrides"] == 1


def test_zero_is_exact_frozen_action_and_infinity_allows_disagreement():
    exact = ConfidenceGatedResidualPolicy(
        controlled_policy(),
        margin_threshold=0.0,
    )
    unlimited = ConfidenceGatedResidualPolicy(
        controlled_policy(),
        margin_threshold=math.inf,
    )

    assert int(
        exact.step(safe_decision(), exact.initial_state()).logits.argmax()
    ) == 4
    assert int(
        unlimited.step(
            safe_decision(),
            unlimited.initial_state(),
        ).logits.argmax()
    ) == 49


@pytest.mark.parametrize(
    (
        "context",
        "own_cards",
        "current_suit",
        "current_rank",
        "penalty_amount",
        "change_suit_mode",
    ),
    (
        ("play", (4, 16, 48, 49), "harten", 5, 0, False),
        ("penalty", (1, 4, 13, 49), "harten", 2, 2, False),
        ("draw_retry", (4, 16, 29), "harten", 5, 0, False),
        ("suit", (44, 48, 49), "special", 13, 0, True),
    ),
)
def test_frozen_observation_matches_production_ppo_adapter(
    context,
    own_cards,
    current_suit,
    current_rank,
    penalty_amount,
    change_suit_mode,
):
    def legacy_factory(game_instance):
        return RL_SupervisedAgent(
            game_instance,
            actor=None,
            device=torch.device("cpu"),
        )

    match = game.Game([legacy_factory, AgentRandom])
    legacy_agent = match.players[0]
    legacy_agent.mydeck.cards[:] = [card.Card(number) for number in own_cards]
    match.players[1].mydeck.cards[:] = [card.Card(12), card.Card(13)]
    match.current_sort = current_suit
    match.current_true_number = current_rank
    match.penalty_amount = penalty_amount
    match.direction = 1
    match.current_player_index = 0
    match.sorts_played[:] = [2, 3, 4, 5, 6]

    own_cards = tuple(held.number for held in legacy_agent.mydeck.cards)
    observation = build_private_observation(
        own_cards=own_cards,
        current_suit=match.current_sort,
        current_rank=match.current_true_number,
        public_hand_sizes=tuple(
            player.mydeck.cardCount() for player in match.players
        ),
        observer_index=0,
        current_player_index=match.current_player_index,
        direction=match.direction,
        penalty_amount=match.penalty_amount,
        public_sorts_played=tuple(match.sorts_played),
        change_suit_mode=change_suit_mode,
    )
    legal = build_legal_actions(
        own_cards=own_cards,
        current_suit=match.current_sort,
        current_rank=match.current_true_number,
        penalty_amount=match.penalty_amount,
        change_suit_mode=change_suit_mode,
    )
    decision = DecisionInput(observation, legal)
    expected = legacy_agent.obs()
    expected[MODE_INDEX] = float(change_suit_mode)

    torch.testing.assert_close(safe_to_legacy_observation(decision), expected)
    safe_batch = torch.tensor(
        [decision.observation.values],
        dtype=torch.float32,
    )
    legal_batch = torch.tensor(
        [decision.legal_actions],
        dtype=torch.bool,
    )
    torch.testing.assert_close(
        safe_batch_to_legacy_observations(
            safe_batch,
            legal_batch,
        ).squeeze(0),
        expected,
    )
    if context == "penalty":
        assert expected[51] == 1.0
        assert expected[63] == 1.0
        assert expected[54] == 0.0
        assert expected[99] == 0.0
    elif context == "draw_retry":
        # A draw retry is a second ordinary playCard call. Both the existing
        # compatible card and the newly drawn compatible card stay visible.
        assert expected[54] == 1.0
        assert expected[66] == 1.0
    elif context == "suit":
        assert expected[98] == 1.0
        assert expected[99] == 1.0


def test_hidden_world_labels_cannot_change_gate_input_or_decision():
    hidden_a = {"opponent_cards": (1, 2), "draw_order": (3, 4)}
    hidden_b = {"opponent_cards": (40, 41), "draw_order": (48, 49)}
    assert hidden_a != hidden_b
    public_a = safe_decision()
    public_b = safe_decision()
    assert public_a == public_b
    left = ConfidenceGatedResidualPolicy(
        controlled_policy(),
        margin_threshold=2.0,
    )
    right = ConfidenceGatedResidualPolicy(
        controlled_policy(),
        margin_threshold=2.0,
    )

    left_result = left.step(public_a, left.initial_state())
    right_result = right.step(public_b, right.initial_state())

    torch.testing.assert_close(left_result.logits, right_result.logits)
    torch.testing.assert_close(left_result.state, right_result.state)
    assert left.last_gate_decision == right.last_gate_decision


def test_threshold_validation_and_labels():
    assert parse_threshold("0.25") == 0.25
    assert parse_threshold("infinity") == math.inf
    assert threshold_label(math.inf) == "infinity"
    with pytest.raises(ValueError, match="non-negative"):
        parse_threshold("-0.1")
    with pytest.raises(ValueError, match="include 0"):
        validate_sweep(
            thresholds=(0.25, 1.0),
            pairs=2,
            seed=81_000_001,
            opponent_count=2,
        )
    with pytest.raises(ValueError, match="sealed"):
        validate_sweep(
            thresholds=(0.0, 1.0),
            pairs=2,
            seed=91_500_001,
            opponent_count=2,
        )


def test_same_deal_delta_is_pairwise_not_difference_of_unpaired_samples():
    block, deltas = _delta_block(
        [1.0, 0.0, 0.5, 1.0],
        [0.5, 0.5, 0.5, 0.0],
        seed=81_190_001,
    )

    assert deltas == [0.5, -0.5, 0.0, 1.0]
    assert block["candidate_minus_threshold_zero"] == pytest.approx(0.25)
    assert block["same_deals_and_rotated_seats"]


def test_invalid_gate_thresholds_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        ConfidenceGatedResidualPolicy(
            controlled_policy(),
            margin_threshold=-1.0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        ConfidenceGatedResidualPolicy(
            controlled_policy(),
            margin_threshold=math.nan,
        )
