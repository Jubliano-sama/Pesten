import json

import pytest
import torch
from torch import nn

import card
import deck
from ai_observation import MODE_INDEX, PASS_ACTION
from benchmark_population_distill import (
    benchmark_paired,
    inference_throughput,
    paired_bootstrap_interval,
)
from population_distill import (
    DistillConfig,
    PopulationStudent,
    PopulationStudentAgent,
    SEALED_FINAL_SEED_START,
    build_public_observation,
    distill_student,
    legal_action_mask,
    population_targets,
    public_heuristic_logits,
    save_checkpoint,
    validate_development_seed,
)
from randomAgent import AgentRandom


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(121, 55, bias=False)
        with torch.no_grad():
            values = torch.arange(55 * 121, dtype=torch.float32)
            self.linear.weight.copy_(values.reshape(55, 121) / values.numel())

    def forward(self, observations):
        return self.linear(observations)


def observation(
    own_cards=(4, 16, 49),
    *,
    hand_sizes=(3, 5),
    player_index=0,
    direction=1,
    penalty=0,
    suit_mode=False,
):
    return build_public_observation(
        own_cards=own_cards,
        current_sort="harten",
        current_rank=5,
        penalty_amount=penalty,
        public_sorts_played=(2, 1, 0, 3, 1),
        public_hand_sizes=hand_sizes,
        player_index=player_index,
        direction=direction,
        change_suit_mode=suit_mode,
    )


def test_hidden_worlds_cannot_change_public_observation_or_population_target():
    # These environment-side values intentionally never cross the API.
    hidden_a = {"other_cards": (0, 1, 2), "draw_order": (8, 9, 10)}
    hidden_b = {"other_cards": (30, 31, 32), "draw_order": (47, 48, 49)}
    assert hidden_a != hidden_b

    view_a = observation()
    view_b = observation()
    assert view_a.numpy().tobytes() == view_b.numpy().tobytes()
    teacher = TinyTeacher().eval()
    target_a = population_targets(view_a.unsqueeze(0), teacher)
    target_b = population_targets(view_b.unsqueeze(0), teacher)
    assert torch.equal(target_a, target_b)


def test_population_targets_and_student_actions_are_strictly_legal():
    views = torch.stack(
        (
            observation((4, 16, 49)),
            observation((4,), penalty=2),
            observation((48, 12), suit_mode=True),
        )
    )
    teacher = TinyTeacher().eval()
    targets = population_targets(views, teacher)
    masks = legal_action_mask(views)
    assert torch.all(targets[~masks] == 0)
    torch.testing.assert_close(
        targets.sum(-1), torch.ones(len(views)), atol=1e-6, rtol=0
    )

    student = PopulationStudent("compact")
    logits = student(views).masked_fill(~masks, torch.finfo(torch.float32).min)
    chosen = logits.argmax(-1)
    assert torch.all(masks.gather(1, chosen[:, None]))
    assert views[2, MODE_INDEX] == 1
    assert 50 <= int(chosen[2]) <= 53


def test_heuristics_respect_special_finish_and_suit_choice_rules():
    final_joker = observation((49,))
    experts = public_heuristic_logits(final_joker.unsqueeze(0))
    assert torch.all(experts[:, 0, PASS_ACTION] > experts[:, 0, 49])

    suit_view = observation((4, 16, 28), suit_mode=True)
    suit_experts = public_heuristic_logits(suit_view.unsqueeze(0))
    mask = legal_action_mask(suit_view)
    assert torch.nonzero(mask).flatten().tolist() == [50, 51, 52, 53]
    assert torch.all(suit_experts[:, 0, ~mask] < -1e8)


def test_adapter_reads_opponent_count_but_not_opponent_card_identities():
    class CountOnlyDeck:
        def cardCount(self):
            return 7

        @property
        def cards(self):
            raise AssertionError("opponent cards must not be inspected")

    class Other:
        mydeck = CountOnlyDeck()

    class PublicGame:
        current_sort = "harten"
        current_true_number = 5
        penalty_amount = 0
        sorts_played = (0, 0, 0, 0, 0)
        direction = 1

    game_instance = PublicGame()
    actor = PopulationStudent("compact").eval()
    agent = PopulationStudentAgent(game_instance, actor, torch.device("cpu"))
    agent.mydeck = deck.Deck([card.Card(4), card.Card(16)])
    game_instance.players = [agent, Other()]
    view = agent._observation()
    assert view[106] == 2
    assert view[107] == 7


def test_tiny_distillation_is_reproducible_and_checkpoint_stores_no_rollouts(
    tmp_path,
):
    views = torch.stack(
        [
            observation(
                own_cards=(index % 12, 12 + index % 12, 49),
                hand_sizes=(3, 2 + index % 6),
                direction=-1 if index % 2 else 1,
            )
            for index in range(32)
        ]
    )
    teacher = TinyTeacher().eval()
    config = DistillConfig(seed=19, epochs=2, batch_size=8)
    first, first_metrics = distill_student(
        views,
        teacher,
        architecture="compact",
        config=config,
        device=torch.device("cpu"),
    )
    second, second_metrics = distill_student(
        views,
        teacher,
        architecture="compact",
        config=config,
        device=torch.device("cpu"),
    )
    for left, right in zip(first.state_dict().values(), second.state_dict().values()):
        assert torch.equal(left, right)
    assert first_metrics["illegal_actions"] == 0
    assert first_metrics["cross_entropy"] == second_metrics["cross_entropy"]

    checkpoint = tmp_path / "student.pt"
    save_checkpoint(checkpoint, first, {"seed": 19, "samples": len(views)})
    payload = torch.load(checkpoint, weights_only=False)
    assert set(payload) == {
        "format",
        "observation_schema",
        "architecture",
        "state_dict",
        "metadata",
    }
    assert "observations" not in payload
    assert "trajectories" not in payload
    assert json.dumps(payload["metadata"], sort_keys=True) == (
        '{"samples": 32, "seed": 19}'
    )


def test_development_seed_guard_and_paired_uncertainty_are_reproducible():
    validate_development_seed(82_000_000, 100)
    with pytest.raises(ValueError, match="development seeds"):
        validate_development_seed(SEALED_FINAL_SEED_START, 1)
    scores = [0.0, 0.5, 1.0, 1.0]
    assert paired_bootstrap_interval(scores, seed=7, samples=500) == (
        paired_bootstrap_interval(scores, seed=7, samples=500)
    )

    result = benchmark_paired(
        AgentRandom,
        AgentRandom,
        candidate_name="random_a",
        opponent_name="random_b",
        pairs=2,
        seed=82_100_000,
    )
    assert result["games"] == 4
    assert result["wins"] + result["losses"] + result["draws"] == 4
    assert len(result["paired_score_ci95"]) == 2
    speed = inference_throughput(
        PopulationStudent("compact"),
        torch.device("cpu"),
        batch_size=4,
        iterations=2,
    )
    assert speed["samples"] == 8
    assert speed["samples_per_second"] > 0
