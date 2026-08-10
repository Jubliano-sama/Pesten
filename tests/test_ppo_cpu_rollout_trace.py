import random
import copy

import numpy as np
import torch

from ai_observation import OBSERVATION_SIZE
from ppo2 import PPO_Supervised, _HybridRolloutInference


def _trainer(*, traced: bool) -> PPO_Supervised:
    return PPO_Supervised(
        use_pretrained=False,
        old_save=(),
        rollout_device="cpu",
        rollout_batch_size=32,
        training_actor_device="cpu",
        compile_updates=False,
        trace_cpu_rollout_actor=traced,
        past_policy_probability=0.0,
        torch_threads=2,
    )


def test_traced_cpu_rollout_actor_is_dynamic_and_syncs_exactly() -> None:
    torch.manual_seed(84_940_001)
    trainer = _trainer(traced=True)
    assert trainer.rollout_actor is not trainer.actor
    assert trainer.trace_cpu_rollout_actor
    assert not trainer.hybrid_rollout_inference
    assert trainer.hybrid_rollout_threshold == 64

    for batch_size in (1, 7, 32, 65):
        observations = torch.randn(batch_size, OBSERVATION_SIZE)
        with torch.inference_mode():
            assert torch.equal(
                trainer.actor(observations),
                trainer.rollout_actor(observations),
            )

    with torch.no_grad():
        next(trainer.actor.parameters()).add_(0.125)
    trainer._sync_rollout_actor()
    observations = torch.randn(31, OBSERVATION_SIZE)
    with torch.inference_mode():
        assert torch.equal(
            trainer.actor(observations),
            trainer.rollout_actor(observations),
        )


def test_trace_preserves_complete_stochastic_rollout() -> None:
    torch.manual_seed(84_940_002)
    eager = _trainer(traced=False)
    actor_state = eager.actor.state_dict()

    torch.manual_seed(84_940_003)
    traced = _trainer(traced=True)
    traced.actor.load_state_dict(actor_state)
    traced._sync_rollout_actor()

    def rollout(trainer):
        random.seed(84_940_004)
        np.random.seed(84_940_004)
        torch.manual_seed(84_940_004)
        return trainer.generate_data(
            num_episodes=64,
            temperature=0.8,
            agent_sets=[["smart", "rl"]],
        )

    expected = rollout(eager)
    actual = rollout(traced)
    for expected_tensor, actual_tensor in zip(
        expected[:4],
        actual[:4],
    ):
        assert torch.equal(expected_tensor, actual_tensor)
    assert expected[4] == actual[4]
    assert eager.last_rollout_metrics == traced.last_rollout_metrics


def test_hybrid_sync_skips_shared_storage_and_copies_fallback() -> None:
    source = torch.nn.Linear(3, 2)
    cpu_actor = copy.deepcopy(source)
    shared = _HybridRolloutInference(
        cpu_actor,
        source,
        source,
        threshold=64,
    )
    assert shared.gpu_actor_shares_source_storage

    with torch.no_grad():
        source.weight.add_(0.25)
    shared.sync_from(source)
    assert torch.equal(cpu_actor.weight, source.weight)

    fallback_gpu = copy.deepcopy(source)
    fallback = _HybridRolloutInference(
        copy.deepcopy(source),
        fallback_gpu,
        source,
        threshold=64,
    )
    assert not fallback.gpu_actor_shares_source_storage
    with torch.no_grad():
        source.bias.sub_(0.5)
    fallback.sync_from(source)
    assert torch.equal(fallback_gpu.bias, source.bias)
