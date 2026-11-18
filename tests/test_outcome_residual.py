"""Check the information boundary and exact counterfactual training branches."""

from copy import deepcopy

import numpy as np
import pytest
import torch

pytest.importorskip("pufferlib.models")

from batched_pesten import BatchedPesten, POLICY_RANDOM
from branch_outcome_policy import (
    BranchActionHead,
    BranchOutcomeCore,
    _decision_features,
    legal_mask,
)
from rollout_policy_improvement import fork_rows
from temporal_observation import (
    DecisionInput,
    PlayerPrivateObservation,
    build_draw_retry_legal_actions,
)
from train_outcome_residual import PublicHistoryCore, teacher


def test_forks_copy_draw_and_random_policy_streams_without_mutating_source():
    kinds = np.full((2, 3), POLICY_RANDOM, np.int8)
    original = BatchedPesten(kinds, np.full_like(kinds, 0), np.array([291, 902], np.uint32))
    unchanged = deepcopy(original)
    branch = fork_rows(original, np.array([0, 0, 1]))
    # RandomAgent shuffles the actual hand in place, consuming the same RNG that
    # shuffles a recycled deck. Fork two identical copies through complete games.
    while not branch.game_over[:2].all():
        for row in range(2):
            action = 50 if branch.phase[row] == 2 else branch.random_action(row)
            branch.apply_action(row, action)
        assert branch.current_player[0] == branch.current_player[1]
        assert np.array_equal(branch.hands[0], branch.hands[1])
        if branch.turn_counts[0] > 1000:
            pytest.fail("test game exceeded its turn limit")
    assert np.array_equal(original.hands, unchanged.hands)
    assert original.public_events == unchanged.public_events
    assert original.rngs[0].randint(2**30) == unchanged.rngs[0].randint(2**30)


def test_hidden_draw_identity_is_absent_from_played_card_memory():
    state = torch.zeros(1, 2, 50)
    events = torch.zeros(2, 100)
    events[:, 94] = torch.tensor([1 / 3, 2 / 3])
    events[0, 96] = (18 + 1) / 50
    events[1, 98] = 5 / 54
    next_state = PublicHistoryCore.played_cards(events, state)
    assert next_state[0, 0, 18] == 1
    assert next_state[0, 1].count_nonzero() == 0
    assert state.count_nonzero() == 0


def test_retry_is_carried_to_student_and_stripped_before_anchor():
    obs = torch.zeros(94)
    obs[[2, 3, 8]] = 1
    obs[50] = obs[57] = 1
    obs[[69, 70, 89, 91, 92]] = torch.tensor([3.0, 5.0, 1.0, 3.0, 2.0])
    decision = DecisionInput(
        PlayerPrivateObservation(tuple(obs.tolist())), build_draw_retry_legal_actions(8)
    )
    token = torch.tensor([_decision_features(decision)])
    assert token[0, 98] == 9
    assert legal_mask(token, token[:, 98].long()).nonzero()[:, 1].tolist() == [8, 54]
    actor = teacher(1)
    anchor = deepcopy(actor.puffer_policy)
    wrapper = BranchOutcomeCore(anchor, BranchActionHead(), 10.0, 8.0)
    with torch.inference_mode():
        expected = actor.step(decision, actor.initial_state())
        actual, _, state = wrapper.forward_eval(token, wrapper.initial_state(1, "cpu"))
        torch.testing.assert_close(actual[0, [8, 54]], expected.logits[[8, 54]])
        for left, right in zip(state[:-1], expected.state, strict=True):
            torch.testing.assert_close(left, right)


def test_action_head_equivariant_under_global_ordinary_rank_and_suit_relabeling():
    torch.manual_seed(13)
    head = BranchActionHead(graph=True, style_heads=True)
    nn = torch.nn
    nn.init.normal_(head.output[-1].weight)
    obs = torch.zeros(1, 94)
    obs[0, [2, 5, 14, 29, 42, 48, 50, 58, 89]] = 1
    obs[0, 69:73] = torch.tensor([6.0, 3.0, 5.0, 4.0])
    obs[0, 82:86] = torch.tensor([3.0, 5.0, 9.0, 6.0])
    obs[0, 91:93] = torch.tensor([6.0, 4.0])
    seen = torch.rand(1, 50)
    value = torch.tensor([0.21])
    suits = [2, 0, 3, 1]
    ranks = [0, 1, 5, 3, 4, 2, 6, 7, 10, 9, 8, 11]
    cmap = torch.tensor([12 * suits[a // 12] + ranks[a % 12] for a in range(48)] + [48, 49])
    amap = torch.tensor([*cmap.tolist(), *(50 + s for s in suits), 54])
    transformed = obs.clone()
    transformed[:, cmap] = obs[:, :50]
    transformed[:, [50 + s for s in suits]] = obs[:, 50:54]
    transformed[:, [56 + r for r in ranks]] = obs[:, 56:68]
    transformed[:, [82 + s for s in suits]] = obs[:, 82:86]
    seen2 = torch.zeros_like(seen)
    seen2[:, cmap] = seen
    mask = torch.ones((1, 55), dtype=torch.bool)
    mix = torch.tensor([0.7])
    torch.testing.assert_close(
        head(obs, seen, value, mask, mix),
        head(transformed, seen2, value, mask, mix)[:, amap],
        rtol=1e-5,
        atol=2e-6,
    )


def test_public_beliefs_follow_card_relabeling_with_nonempty_history():
    from branch_public_belief import belief_nodes, initial_history
    from public_lookahead import BELIEF_WIDTH, DECLARATION, EVIDENCE, LAST_CARD, PILE_SIZE

    generator = torch.Generator().manual_seed(14)
    obs = torch.zeros(1, 94)
    obs[0, [2, 5, 14, 29, 42, 48]] = 1
    obs[0, 69:73] = torch.tensor([6.0, 3.0, 5.0, 4.0])
    obs[0, 89] = -1
    obs[0, 92] = 4
    history = initial_history(1, "cpu")[0]
    history[:, PILE_SIZE - 3 : PILE_SIZE + 1] = torch.tensor([13.0, 36.0, 42.0, 3.0])
    history[:, LAST_CARD + 1 : LAST_CARD + 4] = torch.tensor([13.0, 36.0, 42.0])
    history[:, DECLARATION + 1 : DECLARATION + 4] = torch.tensor([1.0, 3.0, 0.0])
    history[:, EVIDENCE : EVIDENCE + 4 * BELIEF_WIDTH] = torch.randn(
        4 * BELIEF_WIDTH, generator=generator
    )
    frequencies = torch.randn(2, 3, 9, 10, generator=generator)
    suits = [2, 0, 3, 1]
    ranks = [0, 1, 5, 3, 4, 2, 6, 7, 10, 9, 8, 11]
    cards = torch.tensor([12 * suits[a // 12] + ranks[a % 12] for a in range(48)] + [48, 49])
    new_obs = obs.clone()
    new_obs[:, cards] = obs[:, :50]
    new_history = history.clone()
    for seat in range(4):
        start = EVIDENCE + seat * BELIEF_WIDTH
        new_history[:, [start + s for s in suits]] = history[:, start : start + 4]
        new_history[:, [start + 5 + r for r in ranks]] = history[:, start + 5 : start + 17]
    for column in (*range(PILE_SIZE - 3, PILE_SIZE), *range(LAST_CARD + 1, LAST_CARD + 4)):
        new_history[:, column] = cards[history[:, column].long() - 1] + 1
    for column in (DECLARATION + 1, DECLARATION + 2):
        new_history[:, column] = suits[int(history[0, column]) - 1] + 1
    first = belief_nodes(obs, history, frequencies).flatten(1, 2)
    second = belief_nodes(new_obs, new_history, frequencies).flatten(1, 2)
    torch.testing.assert_close(first, second[:, cards[:48]], rtol=1e-5, atol=2e-6)


def test_scoring_counterfactuals_leaves_real_episode_and_root_history_identical(monkeypatch):
    from types import SimpleNamespace
    import rollout_policy_improvement as trainer
    from train_outcome_residual import collection_policy

    policy = collection_policy(1)
    args = SimpleNamespace(
        games=18, seed=614250003, horizons=[1, 4], margin=8.0, explore=0.2, keep=0.7, beliefs=False
    )

    def score(hands, sizes, seat, direction, suit, rank, penalty, random_others, weights):
        return (1 / sizes[:, 0].clip(min=1)).astype(np.float32)

    actual = trainer.collect_chunk(policy, args, 1, score, np.zeros(1))
    with monkeypatch.context() as patch:
        patch.setattr(
            trainer,
            "score_branches",
            lambda p, e, s, r, rows, forced, focal, rand, h, scorer, w: np.zeros(
                (len(forced), len(h)), np.float32
            ),
        )
        reference = trainer.collect_chunk(policy, args, 1, score, np.zeros(1))
    for name in actual.keys() - {"q"}:
        np.testing.assert_array_equal(actual[name], reference[name], err_msg=name)
    assert (actual["q"] >= 0).all() and (actual["q"] <= 1).all()
    assert (actual["q"] != 0).any()
