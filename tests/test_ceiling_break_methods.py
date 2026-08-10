from pathlib import Path

import torch

from ceiling_break_checkpoint_sweep import actor_iteration
from evolutionary_rank_policy import BIAS_DIMENSIONS, RankBiasPolicy
from fitted_q_policy import ActionValueNet, FittedQResidualPolicy
from random_fitted_q import return_targets


class _ZeroActor(torch.nn.Module):
    def forward(self, observations):
        return observations.new_zeros((len(observations), 55))


def test_checkpoint_iteration_accepts_ppo_and_reinforce_names():
    assert actor_iteration(Path("ppo_actor_17.pth")) == 17
    assert actor_iteration(Path("reinforce_actor_003.pth")) == 3


def test_action_value_policy_zero_scale_is_exact_base():
    observations = torch.randn(4, 121)
    base = _ZeroActor()
    action_value = ActionValueNet(width=16, blocks=1)
    policy = FittedQResidualPolicy(
        base,
        action_value,
        scale=0.0,
    )
    assert torch.equal(policy(observations), base(observations))


def test_return_targets_are_discounted_and_episode_indexed():
    targets, episode_ids, outcomes = return_targets(
        torch.tensor([0.0, 1.0, 0.0, -1.0]),
        [2, 2],
        gamma=0.5,
    )
    assert torch.equal(targets, torch.tensor([0.5, 1.0, -0.5, -1.0]))
    assert torch.equal(episode_ids, torch.tensor([0, 0, 1, 1]))
    assert outcomes == {
        "games": 2,
        "wins": 1,
        "draws": 0,
        "losses": 1,
        "sampled_score": 0.5,
    }


def test_rank_bias_policy_maps_card_ranks_suits_and_pass():
    biases = torch.arange(BIAS_DIMENSIONS, dtype=torch.float32)
    policy = RankBiasPolicy(_ZeroActor(), biases)
    logits = policy(torch.zeros(1, 121))[0]
    assert logits[0] == 1  # standard card 0 has rank 1
    assert logits[48] == 13
    assert logits[49] == 0
    assert torch.equal(logits[50:54], biases[14:18])
    assert logits[54] == biases[18]

