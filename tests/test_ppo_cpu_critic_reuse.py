import copy

import torch

from ai_observation import OBSERVATION_SIZE
from ppo2 import PPO_Supervised


def test_reused_cpu_critic_forward_preserves_gradients_and_bn_semantics() -> None:
    torch.manual_seed(85_152_301)
    trainer = PPO_Supervised(
        use_pretrained=False,
        old_save=(),
        rollout_device="cpu",
        training_actor_device="cpu",
        compile_updates=False,
        past_policy_probability=0.0,
        torch_threads=2,
    )
    observations = torch.randn(64, OBSERVATION_SIZE)
    targets = torch.randn(64)
    initial_state = copy.deepcopy(trainer.critic.state_dict())

    trainer.critic.load_state_dict(initial_state)
    with torch.no_grad():
        trainer.critic_update_forward(observations)
    expected = trainer.critic_update_forward(observations).squeeze(-1)
    torch.nn.functional.mse_loss(expected, targets).backward()
    expected_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in trainer.critic.named_parameters()
    }
    expected_state = copy.deepcopy(trainer.critic.state_dict())

    trainer.critic.load_state_dict(initial_state)
    trainer.critic.zero_grad(set_to_none=True)
    actual = trainer._critic_forward_with_two_bn_updates(
        observations
    )
    torch.nn.functional.mse_loss(actual, targets).backward()

    assert torch.equal(actual, expected)
    for name, parameter in trainer.critic.named_parameters():
        assert torch.equal(parameter.grad, expected_gradients[name]), name
    for name, expected_value in expected_state.items():
        actual_value = trainer.critic.state_dict()[name]
        if name.endswith(("running_mean", "running_var")):
            assert torch.allclose(
                actual_value,
                expected_value,
                rtol=0.0,
                atol=5e-7,
            ), name
        else:
            assert torch.equal(actual_value, expected_value), name
