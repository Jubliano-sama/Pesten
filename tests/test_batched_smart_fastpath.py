import numpy as np
import torch
from torch.nn import functional as F
import pytest

from batched_pesten import (
    CARD_COMPATIBILITY_MASKS,
    CARD_IS_PENALTY,
    CARD_RANK,
    CARD_RANK_MASKS,
    CARD_SORT,
    PENALTY_RANKS,
    POLICY_SMART,
    BatchedPesten,
    _compatible,
)
from benchmark_rollout_smart_fastpath import (
    DEVELOPMENT_SEED_LIMIT,
    LegacySmartBatchedPesten,
    assert_rollout_parity,
    run_once,
)
from benchmark_rollout_lookup_tables import (
    LegacyLookupBatchedPesten,
    recomputed_tables,
)
from benchmark_training_speed import _module_sha256, _rollout_sha256
from ppo2 import DEFAULT_TORCH_THREADS, PPO_Supervised, StudentNet


def _legacy_smart_suit(hand: np.ndarray) -> int:
    ranks = CARD_RANK[hand]
    suits = CARD_SORT[hand]
    best_suit = 0
    best_score = -float("inf")
    for suit in range(4):
        same_suit = suits == suit
        total = int(np.count_nonzero(same_suit))
        penalty = int(
            np.count_nonzero(
                same_suit & np.isin(ranks, tuple(PENALTY_RANKS))
            )
        )
        score = (
            0.43244878288347055 * total
            - (-1.8463831782446556) * penalty
        )
        if score > best_score:
            best_score = score
            best_suit = suit
    return best_suit


def _assert_same_sgd_update(reference, optimized) -> None:
    left = StudentNet().train()
    right = StudentNet().train()
    right.load_state_dict(left.state_dict())
    left_optimizer = torch.optim.SGD(left.parameters(), lr=1e-5)
    right_optimizer = torch.optim.SGD(right.parameters(), lr=1e-5)
    for model, optimizer, rollout in (
        (left, left_optimizer, reference),
        (right, right_optimizer, optimized),
    ):
        logits = model(rollout.observations)
        loss = F.cross_entropy(logits, rollout.actions)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert all(
        torch.equal(left_tensor, right.state_dict()[name])
        for name, left_tensor in left.state_dict().items()
    )


def test_compatibility_table_is_exhaustively_equal_to_canonical_predicate():
    for suit in range(5):
        for rank in range(14):
            for number in range(50):
                assert bool(CARD_COMPATIBILITY_MASKS[suit, rank, number]) == (
                    _compatible(number, suit, rank)
                )


def test_card_lookup_tables_are_read_only():
    for table in (
        CARD_RANK_MASKS,
        CARD_COMPATIBILITY_MASKS,
        CARD_IS_PENALTY,
    ):
        assert not table.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            table.flat[0] = not bool(table.flat[0])


def test_smart_suit_histogram_is_exact_for_random_duplicate_hands():
    kinds = np.full((1, 2), POLICY_SMART, dtype=np.int8)
    keys = np.full((1, 2), None, dtype=object)
    engine = BatchedPesten(kinds, keys, np.asarray([84_730_001]))
    rng = np.random.RandomState(12345)
    player = int(engine.current_player[0])
    for _ in range(10_000):
        size = int(rng.randint(0, 55))
        # Sampling with replacement deliberately covers duplicate jokers,
        # penalty ties, ordinary duplicates, and empty hands.
        hand = rng.randint(0, 50, size=size).astype(np.int8)
        engine.hands[0, player].fill(-1)
        engine.hands[0, player, :size] = hand
        engine.hand_sizes[0, player] = size
        assert engine.choose_suit(0, POLICY_SMART) == _legacy_smart_suit(
            hand
        )


def test_throughput_hashes_detect_model_and_rollout_changes():
    model = torch.nn.Linear(3, 2)
    model_copy = torch.nn.Linear(3, 2)
    model_copy.load_state_dict(model.state_dict())
    assert _module_sha256(model) == _module_sha256(model_copy)
    with torch.no_grad():
        model_copy.bias[0].add_(1)
    assert _module_sha256(model) != _module_sha256(model_copy)

    batch = (
        torch.zeros(2, 3),
        torch.tensor([0, 1]),
        torch.tensor([-0.1, -0.2]),
        torch.tensor([0.0, 1.0]),
        [2],
    )
    changed = (*batch[:3], torch.tensor([0.0, -1.0]), batch[4])
    assert _rollout_sha256(batch) != _rollout_sha256(changed)


def test_fixed_seed_rollout_is_bit_exact_after_smart_fastpath():
    torch.manual_seed(123)
    actor = StudentNet().eval()
    reference, _ = run_once(
        LegacySmartBatchedPesten,
        actor,
        games=24,
        seed=84_700_001,
        device=torch.device("cpu"),
        max_turns=200,
    )
    optimized, _ = run_once(
        BatchedPesten,
        actor,
        games=24,
        seed=84_700_001,
        device=torch.device("cpu"),
        max_turns=200,
    )
    parity = assert_rollout_parity(reference, optimized)
    assert parity["samples"] > 0
    assert parity["tensor_fields_bit_exact"]


def test_development_seed_guard_uses_shared_limit():
    actor = torch.nn.Linear(121, 55).eval()
    try:
        run_once(
            BatchedPesten,
            actor,
            games=2,
            seed=DEVELOPMENT_SEED_LIMIT - 1,
            device=torch.device("cpu"),
            max_turns=10,
        )
    except ValueError as error:
        assert "shared development limit" in str(error)
    else:
        raise AssertionError("shared development limit was accepted")


def test_identical_rollouts_produce_bit_exact_update():
    torch.manual_seed(321)
    rollout_actor = StudentNet().eval()
    reference, _ = run_once(
        LegacySmartBatchedPesten,
        rollout_actor,
        games=16,
        seed=84_710_001,
        device=torch.device("cpu"),
        max_turns=120,
    )
    optimized, _ = run_once(
        BatchedPesten,
        rollout_actor,
        games=16,
        seed=84_710_001,
        device=torch.device("cpu"),
        max_turns=120,
    )
    assert_rollout_parity(reference, optimized)
    _assert_same_sgd_update(reference, optimized)


def test_lookup_table_rollout_and_update_are_bit_exact():
    torch.manual_seed(777)
    actor = StudentNet().eval()
    with recomputed_tables():
        reference, _ = run_once(
            LegacyLookupBatchedPesten,
            actor,
            games=24,
            seed=84_740_001,
            device=torch.device("cpu"),
            max_turns=180,
        )
    optimized, _ = run_once(
        BatchedPesten,
        actor,
        games=24,
        seed=84_740_001,
        device=torch.device("cpu"),
        max_turns=180,
    )
    assert_rollout_parity(reference, optimized)
    _assert_same_sgd_update(reference, optimized)


def test_live_game_reference_smart_rollout_is_also_bit_exact():
    torch.manual_seed(654)
    actor = StudentNet().eval()
    reference, _ = run_once(
        LegacySmartBatchedPesten,
        actor,
        games=20,
        seed=84_720_001,
        device=torch.device("cpu"),
        max_turns=160,
        smart_agents_have_game_reference=True,
    )
    optimized, _ = run_once(
        BatchedPesten,
        actor,
        games=20,
        seed=84_720_001,
        device=torch.device("cpu"),
        max_turns=160,
        smart_agents_have_game_reference=True,
    )
    assert_rollout_parity(reference, optimized)


def test_ppo_thread_cap_defaults_to_eight_and_is_configurable():
    previous_threads = torch.get_num_threads()
    try:
        trainer = PPO_Supervised(
            use_pretrained=False,
            old_save=(),
            rollout_device="cpu",
            training_actor_device="cpu",
            compile_updates=False,
            torch_threads=2,
        )
        assert DEFAULT_TORCH_THREADS == 8
        assert trainer.torch_threads == 2
        assert torch.get_num_threads() == 2
        with pytest.raises(ValueError, match="torch_threads"):
            PPO_Supervised(
                use_pretrained=False,
                old_save=(),
                torch_threads=0,
            )
        with pytest.raises(ValueError, match="torch_threads"):
            PPO_Supervised(
                use_pretrained=False,
                old_save=(),
                torch_threads=2.5,
            )
    finally:
        torch.set_num_threads(previous_threads)
