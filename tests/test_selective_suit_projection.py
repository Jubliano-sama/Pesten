import inspect
import math

import torch

from ai_observation import (
    ACTION_COUNT,
    COMPATIBLE_START,
    MODE_INDEX,
    OBSERVATION_SIZE,
    PASS_ACTION,
)
from batched_suit_symmetry import BatchedSuitAveragedActor
from ppo2 import build_action_masks
from selective_suit_projection import (
    PREDECLARED_RULE_GRID,
    SelectiveSuitProjectionActor,
    SelectiveSuitRule,
)


class SuitBiasedActor(torch.nn.Module):
    def forward(self, observations):
        logits = observations.new_zeros((len(observations), ACTION_COUNT))
        for suit in range(4):
            logits[:, suit * 12 + 2] = float(suit + 1)
        logits[:, PASS_ACTION] = -5.0
        return logits


def legal_observation() -> torch.Tensor:
    observation = torch.zeros(OBSERVATION_SIZE)
    observation[2] = 1
    observation[14] = 1
    observation[26] = 1
    observation[38] = 1
    observation[COMPATIBLE_START + 2] = 1
    observation[COMPATIBLE_START + 14] = 1
    observation[COMPATIBLE_START + 26] = 1
    observation[COMPATIBLE_START + 38] = 1
    observation[MODE_INDEX] = 0
    return observation


def test_selector_api_has_no_seed_game_or_hidden_world_input():
    parameters = set(
        inspect.signature(SelectiveSuitProjectionActor.forward).parameters
    )
    assert parameters == {"self", "observations"}
    assert not parameters & {
        "seed",
        "game",
        "opponent_cards",
        "draw_pile",
    }


def test_predeclared_grid_is_small_unique_and_fixed():
    assert len(PREDECLARED_RULE_GRID) == 6
    assert len({rule.name for rule in PREDECLARED_RULE_GRID}) == 6
    assert all(
        isinstance(rule, SelectiveSuitRule)
        for rule in PREDECLARED_RULE_GRID
    )


def test_diagnostics_and_gate_use_legal_current_observation_only():
    actor = SelectiveSuitProjectionActor(
        SuitBiasedActor(),
        SelectiveSuitRule("open", 0.0, math.inf),
    )
    observation = legal_observation().unsqueeze(0)
    diagnostics = actor.diagnostics(observation)
    assert diagnostics.legacy_action.item() == 38
    assert diagnostics.orbit_action.item() == 2
    assert diagnostics.action_changed.item()
    assert diagnostics.orbit_vote_agreement.item() == 0.25
    assert actor.selection_mask(diagnostics).item()
    output = actor(observation)
    mask = build_action_masks(observation)
    assert output.masked_fill(~mask, -torch.inf).argmax().item() == 2


def test_closed_gate_is_bit_exact_legacy():
    base = SuitBiasedActor()
    actor = SelectiveSuitProjectionActor(
        base,
        SelectiveSuitRule("closed", 1.0, 0.0),
    )
    observation = legal_observation().unsqueeze(0)
    assert torch.equal(actor(observation), base(observation))
    assert actor.statistics()["projected_decisions"] == 0


def test_open_gate_matches_always_orbit_greedy_actions():
    base = SuitBiasedActor()
    actor = SelectiveSuitProjectionActor(
        base,
        SelectiveSuitRule("open", 0.0, math.inf),
    )
    always = BatchedSuitAveragedActor(
        base, alpha=1.0, aggregation="logits"
    )
    observations = torch.stack(
        (legal_observation(), legal_observation())
    )
    legal = build_action_masks(observations)
    selective_actions = actor(observations).masked_fill(
        ~legal, -torch.inf
    ).argmax(dim=-1)
    always_actions = always(observations).masked_fill(
        ~legal, -torch.inf
    ).argmax(dim=-1)
    assert torch.equal(selective_actions, always_actions)
