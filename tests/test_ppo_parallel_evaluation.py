import random

import numpy as np
import pytest
import torch

import card
import game
import ppo2
from ai_observation import (
    COMPATIBLE_START,
    MODE_INDEX,
    OBSERVATION_SIZE,
)
from benchmark_ppo_parallel_evaluation import select_parallel_evaluation
from ppo2 import (
    PPO_Supervised,
    StudentNet,
    _init_evaluation_worker,
    _run_evaluation_game,
    RL_SupervisedAgent,
)


def _trainer(*, evaluation_workers=2):
    return PPO_Supervised(
        use_pretrained=False,
        old_save=(),
        rollout_device="cpu",
        rollout_batch_size=32,
        training_actor_device="cpu",
        compile_updates=False,
        past_policy_probability=0.0,
        torch_threads=2,
        evaluation_workers=evaluation_workers,
    )


def test_seeded_canonical_evaluation_worker_is_exactly_repeatable():
    torch.manual_seed(84_950_001)
    actor = StudentNet().eval()
    _init_evaluation_worker(actor.state_dict())
    task = (0, 84_950_002, ("smart", "rl"))

    first = _run_evaluation_game(task)
    second = _run_evaluation_game(task)

    assert first == second


def test_no_collection_observation_matches_training_features_exactly():
    actor = StudentNet().eval()
    match = game.Game([])
    evaluated = RL_SupervisedAgent(match, actor, torch.device("cpu"))
    opponent = RL_SupervisedAgent(match, actor, torch.device("cpu"))
    match.players = [evaluated, opponent]
    match.num_players = 2
    match.current_sort = "harten"
    match.current_true_number = 3
    match.direction = -1
    match.sorts_played[:] = [2, 3, 5, 7, 11]
    evaluated.mydeck.cards[:] = [
        card.Card(0),
        card.Card(0),
        card.Card(14),
        card.Card(48),
    ]
    opponent.mydeck.cards[:] = [card.Card(22), card.Card(30)]

    evaluated.collect_data = True
    training_observation = evaluated.obs()
    evaluated.collect_data = False
    evaluation_observation = evaluated.obs()
    next_evaluation_observation = evaluated.obs()

    assert torch.equal(training_observation, evaluation_observation)
    assert evaluation_observation.dtype == torch.float32
    assert (
        evaluation_observation.data_ptr()
        != next_evaluation_observation.data_ptr()
    )


@pytest.mark.parametrize("change_sort", [False, True])
def test_direct_evaluation_mask_samples_same_action(change_sort):
    torch.manual_seed(84_950_010)
    actor = StudentNet().eval()
    match = game.Game([])
    agent = RL_SupervisedAgent(match, actor, torch.device("cpu"))
    agent.eval_mode = True
    observation = torch.zeros(OBSERVATION_SIZE)
    observation[COMPATIBLE_START + 3] = 1
    observation[COMPATIBLE_START + 17] = 1
    observation[MODE_INDEX] = int(change_sort)

    for seed in range(84_950_011, 84_950_031):
        agent.collect_data = True
        torch.manual_seed(seed)
        expected_action, _ = agent.act_rl(observation)
        agent.collect_data = False
        torch.manual_seed(seed)
        actual_action, log_probability = agent.act_rl(observation)

        assert actual_action == expected_action
        assert log_probability == 0.0


def test_large_evaluation_uses_configured_worker_pool(monkeypatch):
    class FakePool:
        def __init__(
            self,
            workers,
            initializer,
            initargs,
        ):
            self.workers = workers
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def map(self, function, tasks, chunksize):
            assert self.workers == 2
            assert chunksize >= 1
            return [function(task) for task in tasks]

    class FakeContext:
        def Pool(self, workers, initializer, initargs):
            return FakePool(workers, initializer, initargs)

    monkeypatch.setattr(
        ppo2,
        "PARALLEL_EVALUATION_MIN_EPISODES",
        4,
    )
    monkeypatch.setattr(
        ppo2.multiprocessing,
        "get_context",
        lambda method: FakeContext(),
    )
    random.seed(84_950_003)
    np.random.seed(84_950_003)
    torch.manual_seed(84_950_003)
    trainer = _trainer(evaluation_workers=2)

    metrics = trainer.evaluate_policy(
        num_episodes=4,
        agent_sets=[["smart", "rl"]],
    )

    assert metrics["episodes"] == 4
    assert metrics["workers"] == 2
    assert metrics["parallel"] is True
    assert 0 <= metrics["wins"] <= 4
    assert metrics["average_turns"] > 0
    assert trainer.actor.training is False
    assert trainer.critic.training is True


@pytest.mark.parametrize("workers", [True, 0, -1, 1.5])
def test_evaluation_workers_must_be_a_positive_integer(workers):
    with pytest.raises(ValueError, match="evaluation_workers"):
        _trainer(evaluation_workers=workers)


def test_parallel_evaluation_selection_requires_exact_material_gain():
    assert select_parallel_evaluation(
        exact=True,
        paired_speedup=3.05,
    )["selected_parallel_evaluation"]
    assert not select_parallel_evaluation(
        exact=False,
        paired_speedup=3.05,
    )["selected_parallel_evaluation"]
    assert not select_parallel_evaluation(
        exact=True,
        paired_speedup=1.24,
    )["selected_parallel_evaluation"]
