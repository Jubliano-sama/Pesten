import numpy as np
import pytest
import torch

from ai_observation import (
    ACTION_COUNT,
    COMPATIBLE_START,
    DIRECTION_INDEX,
    MODE_INDEX,
    OBSERVATION_SIZE,
    PASS_ACTION,
    PLAYER_HAND_SIZES_START,
    PLAYER_INDEX,
    populate_game_context,
)
from batched_pesten import (
    ActorPolicy,
    BatchedPesten,
    PHASE_DRAW_RETRY,
    POLICY_RL,
)
from batched_suit_symmetry import (
    SUIT_PERMUTATIONS,
    BatchedSuitAveragedActor,
    _permutation_indices,
)
from benchmark_checkpoint_batched import (
    build_game_specs,
    run_batched_specs,
    run_canonical_specs,
)
from benchmark_suit_symmetry_batched import (
    _canonical_candidate_factory,
    run_batched_symmetry_benchmark,
)
from corrected_ppo_agent import make_corrected_ppo_agent_factory
from ppo2 import build_action_masks
from suit_symmetry_ppo_agent import (
    ALL_SUIT_PERMUTATIONS,
    make_suit_symmetry_ppo_factory,
    map_permuted_logits_back,
    permute_suit_observation,
    suit_averaged_logits,
)


class StableSuitBiasedActor(torch.nn.Module):
    def forward(self, observations):
        base = torch.full(
            (len(observations), ACTION_COUNT),
            -20.0,
            dtype=observations.dtype,
            device=observations.device,
        )
        for suit in range(4):
            for rank_index in range(12):
                base[:, suit * 12 + rank_index] = (
                    rank_index + 0.1 * suit
                )
            base[:, 50 + suit] = 0.1 * suit
        base[:, 48] = 12.0
        base[:, 49] = 11.0
        base[:, PASS_ACTION] = -10.0
        return base


class CountingReverseActor(StableSuitBiasedActor):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, observations):
        self.calls += 1
        return -super().forward(observations)


class AdversarialTieActor(torch.nn.Module):
    """Pointwise actor with deliberate ties and extreme finite logits."""

    def forward(self, observations):
        logits = observations[:, :ACTION_COUNT].clone()
        shared = observations[:, 0] - observations[:, 49]
        logits[:, 0] = shared
        logits[:, 12] = shared
        logits[:, 48] = 1.0e20
        logits[:, 49] = 1.0e20
        logits[:, 50:54] = -1.0e20
        logits[:, PASS_ACTION] = 0.0
        return logits


def frozen_commit_suit_only_logits(actor, observations):
    """The suit-only tensor program frozen in commit 7de18e0."""
    observation_indices, action_indices = _permutation_indices()
    observation_indices = observation_indices.to(observations.device)
    action_indices = action_indices.to(observations.device)
    transformed = torch.gather(
        observations.unsqueeze(1).expand(
            -1, len(SUIT_PERMUTATIONS), -1
        ),
        dim=2,
        index=observation_indices.unsqueeze(0).expand(
            len(observations), -1, -1
        ),
    )
    transformed_logits = actor(
        transformed.reshape(-1, OBSERVATION_SIZE)
    ).reshape(
        len(observations),
        len(SUIT_PERMUTATIONS),
        ACTION_COUNT,
    )
    mapped_back = torch.gather(
        transformed_logits,
        dim=-1,
        index=action_indices.unsqueeze(0).expand(
            len(observations), -1, -1
        ),
    )
    return mapped_back.mean(dim=1)


def fixture_observations(count=7):
    generator = torch.Generator().manual_seed(19)
    observations = torch.randn(
        count,
        OBSERVATION_SIZE,
        generator=generator,
    )
    observations[:, :100] = (
        observations[:, :100] > 0
    ).float()
    observations[:, 105] = 0
    return observations


def test_cpu_chunking_is_bit_exact_against_frozen_commit_on_ties():
    actor = AdversarialTieActor()
    values = torch.linspace(
        -1.0e10,
        1.0e10,
        steps=129 * OBSERVATION_SIZE,
    )
    observations = values.reshape(129, OBSERVATION_SIZE)
    observations[:, COMPATIBLE_START:COMPATIBLE_START + 50] = (
        observations[:, COMPATIBLE_START:COMPATIBLE_START + 50] > 0
    )
    observations[::2, MODE_INDEX] = 0
    observations[1::2, MODE_INDEX] = 1
    wrapper = BatchedSuitAveragedActor(
        actor,
        alpha=1.0,
        aggregation="logits",
        base_chunk_size=64,
    )

    with torch.inference_mode():
        expected = frozen_commit_suit_only_logits(actor, observations)
        actual = wrapper(observations)

    assert torch.equal(actual, expected)
    masks = build_action_masks(observations)
    expected_actions = expected.masked_fill(~masks, -torch.inf).argmax(-1)
    actual_actions = actual.masked_fill(~masks, -torch.inf).argmax(-1)
    assert torch.equal(actual_actions, expected_actions)


def test_alpha_zero_is_bit_exact_legacy_and_skips_augmentation():
    actor = StableSuitBiasedActor()
    wrapper = BatchedSuitAveragedActor(actor, alpha=0.0)
    observations = fixture_observations()

    expected = actor(observations)
    actual = wrapper(observations)

    assert torch.equal(actual, expected)
    assert wrapper.statistics()["augmented_observations"] == 0
    assert wrapper.statistics()["underlying_actor_calls"] == 1


def test_alpha_one_logits_match_canonical_per_observation_logits_and_actions():
    torch.manual_seed(23)
    actor = torch.nn.Linear(OBSERVATION_SIZE, ACTION_COUNT)
    observations = fixture_observations()
    wrapper = BatchedSuitAveragedActor(
        actor,
        alpha=1.0,
        aggregation="logits",
        base_chunk_size=3,
    )

    actual = wrapper(observations)
    expected = torch.stack(
        [
            suit_averaged_logits(actor, observation)[0]
            for observation in observations
        ]
    )

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    masks = build_action_masks(observations)
    assert torch.equal(
        actual.masked_fill(~masks, -torch.inf).argmax(dim=-1),
        expected.masked_fill(~masks, -torch.inf).argmax(dim=-1),
    )


def test_probability_projection_and_alpha_blend_match_manual_policy_average():
    torch.manual_seed(29)
    actor = torch.nn.Linear(OBSERVATION_SIZE, ACTION_COUNT)
    observations = fixture_observations(3)
    alpha = 0.35
    wrapper = BatchedSuitAveragedActor(
        actor,
        alpha=alpha,
        aggregation="probabilities",
    )

    actual = wrapper(observations)
    manual = []
    for observation in observations:
        mapped = []
        for permutation in ALL_SUIT_PERMUTATIONS:
            transformed = permute_suit_observation(
                observation, permutation
            )
            logits = actor(transformed.unsqueeze(0)).squeeze(0)
            mask = build_action_masks(transformed)
            mapped.append(
                map_permuted_logits_back(
                    logits.masked_fill(
                        ~mask,
                        torch.finfo(logits.dtype).min,
                    ),
                    permutation,
                ).softmax(-1)
            )
        averaged_probability = torch.stack(mapped).mean(0)
        legacy_logits = actor(observation.unsqueeze(0)).squeeze(0)
        legacy_mask = build_action_masks(observation)
        blended = torch.lerp(
            legacy_logits.masked_fill(
                ~legacy_mask,
                torch.finfo(legacy_logits.dtype).min,
            ).softmax(-1),
            averaged_probability,
            alpha,
        )
        manual.append(
            blended.clamp_min(torch.finfo(blended.dtype).tiny).log()
        )

    torch.testing.assert_close(
        actual,
        torch.stack(manual),
        atol=2e-6,
        rtol=2e-6,
    )


def test_player_dihedral_orbit_matches_manual_average_and_is_invariant():
    torch.manual_seed(31)
    actor = torch.nn.Linear(OBSERVATION_SIZE, ACTION_COUNT)
    observation = fixture_observations(1).squeeze(0)
    observation[
        PLAYER_HAND_SIZES_START : PLAYER_HAND_SIZES_START + 13
    ] = 0
    observation[
        PLAYER_HAND_SIZES_START : PLAYER_HAND_SIZES_START + 3
    ] = torch.tensor([5.0, 7.0, 4.0])
    observation[PLAYER_INDEX] = 2
    observation[DIRECTION_INDEX] = 1
    wrapper = BatchedSuitAveragedActor(
        actor,
        alpha=1.0,
        player_orbit_average=True,
    )

    actual = wrapper(observation.unsqueeze(0)).squeeze(0)
    manual = []
    for direction in (1.0, -1.0):
        for player_index in range(3):
            transformed = observation.clone()
            transformed[PLAYER_INDEX] = player_index
            transformed[DIRECTION_INDEX] = direction
            manual.append(suit_averaged_logits(actor, transformed)[0])
    expected = torch.stack(manual).mean(0)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    relabeled = observation.clone()
    relabeled[PLAYER_INDEX] = 0
    relabeled[DIRECTION_INDEX] = -1
    torch.testing.assert_close(
        wrapper(relabeled.unsqueeze(0)).squeeze(0),
        actual,
        atol=2e-6,
        rtol=2e-6,
    )


@pytest.mark.parametrize("player_count", range(2, 14))
def test_dihedral_player_relabeling_preserves_relative_hand_size_semantics(
    player_count,
):
    hand_sizes = list(range(1, player_count + 1))
    observer = player_count - 1
    direction = -1 if player_count % 2 else 1
    original = torch.zeros(OBSERVATION_SIZE)
    populate_game_context(
        original,
        sorts_played=[3, 5, 7, 11, 13],
        hand_sizes=hand_sizes,
        player_index=observer,
        direction=direction,
    )

    for reflected in (False, True):
        offset = player_count // 2

        def relabel(index):
            return (offset + (-index if reflected else index)) % player_count

        relabeled_sizes = [0] * player_count
        for old_index, size in enumerate(hand_sizes):
            relabeled_sizes[relabel(old_index)] = size
        relabeled = torch.zeros(OBSERVATION_SIZE)
        populate_game_context(
            relabeled,
            sorts_played=[3, 5, 7, 11, 13],
            hand_sizes=relabeled_sizes,
            player_index=relabel(observer),
            direction=-direction if reflected else direction,
        )

        torch.testing.assert_close(
            relabeled[
                PLAYER_HAND_SIZES_START:
                PLAYER_HAND_SIZES_START + player_count
            ],
            original[
                PLAYER_HAND_SIZES_START:
                PLAYER_HAND_SIZES_START + player_count
            ],
        )


def _three_player_observation():
    observation = fixture_observations(1).squeeze(0)
    observation[
        PLAYER_HAND_SIZES_START : PLAYER_HAND_SIZES_START + 13
    ] = 0
    observation[
        PLAYER_HAND_SIZES_START : PLAYER_HAND_SIZES_START + 3
    ] = torch.tensor([5.0, 7.0, 4.0])
    observation[PLAYER_INDEX] = 2
    observation[DIRECTION_INDEX] = 1
    return observation


def test_rotation_subgroup_matches_manual_average_and_is_rotation_invariant():
    torch.manual_seed(37)
    actor = torch.nn.Linear(OBSERVATION_SIZE, ACTION_COUNT)
    observation = _three_player_observation()
    wrapper = BatchedSuitAveragedActor(
        actor,
        alpha=1.0,
        player_orbit_mode="rotations",
    )

    actual = wrapper(observation.unsqueeze(0)).squeeze(0)
    manual = []
    for player_index in range(3):
        transformed = observation.clone()
        transformed[PLAYER_INDEX] = player_index
        manual.append(suit_averaged_logits(actor, transformed)[0])
    expected = torch.stack(manual).mean(0)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    rotated = observation.clone()
    rotated[PLAYER_INDEX] = 0
    torch.testing.assert_close(
        wrapper(rotated.unsqueeze(0)).squeeze(0),
        actual,
        atol=2e-6,
        rtol=2e-6,
    )
    statistics = wrapper.statistics()
    assert statistics["player_orbit_mode"] == "rotations"
    assert statistics["augmented_observations"] == 2 * 24 * 3


def test_reflection_subgroup_matches_manual_average_and_is_invariant():
    torch.manual_seed(41)
    actor = torch.nn.Linear(OBSERVATION_SIZE, ACTION_COUNT)
    observation = _three_player_observation()
    wrapper = BatchedSuitAveragedActor(
        actor,
        alpha=1.0,
        player_orbit_mode="reflection",
    )

    actual = wrapper(observation.unsqueeze(0)).squeeze(0)
    reflected = observation.clone()
    reflected[PLAYER_INDEX] = 1
    reflected[DIRECTION_INDEX] = -1
    expected = torch.stack(
        (
            suit_averaged_logits(actor, observation)[0],
            suit_averaged_logits(actor, reflected)[0],
        )
    ).mean(0)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        wrapper(reflected.unsqueeze(0)).squeeze(0),
        actual,
        atol=2e-6,
        rtol=2e-6,
    )
    statistics = wrapper.statistics()
    assert statistics["player_orbit_mode"] == "reflection"
    assert statistics["augmented_observations"] == 2 * 24 * 2


def test_full_mode_is_backward_compatible_with_boolean_alias():
    torch.manual_seed(43)
    actor = torch.nn.Linear(OBSERVATION_SIZE, ACTION_COUNT)
    observations = torch.stack(
        (_three_player_observation(), _three_player_observation())
    )
    observations[1, PLAYER_INDEX] = 0
    observations[1, DIRECTION_INDEX] = -1

    legacy = BatchedSuitAveragedActor(
        actor,
        player_orbit_average=True,
    )
    explicit = BatchedSuitAveragedActor(
        actor,
        player_orbit_mode="full",
    )

    torch.testing.assert_close(
        legacy(observations),
        explicit(observations),
        atol=0,
        rtol=0,
    )
    assert legacy.statistics()["player_orbit_mode"] == "full"
    assert explicit.statistics()["player_orbit_average"]


def test_player_orbit_mode_validation_is_fail_closed():
    actor = StableSuitBiasedActor()
    with pytest.raises(ValueError, match="player_orbit_mode"):
        BatchedSuitAveragedActor(actor, player_orbit_mode="sampled")
    with pytest.raises(ValueError, match="compatible only"):
        BatchedSuitAveragedActor(
            actor,
            player_orbit_average=True,
            player_orbit_mode="rotations",
        )


@pytest.mark.parametrize("mode", ["rotations", "reflection"])
def test_subgroup_batched_games_match_canonical_exactly(mode):
    actor = StableSuitBiasedActor()
    result = run_batched_symmetry_benchmark(
        actor,
        torch.device("cpu"),
        pairs=1,
        base_seed=7_811_331,
        pool_names=("random",),
        max_turns=100,
        bootstrap_samples=20,
        player_orbit_mode=mode,
        canonical_throughput_pairs=1,
    )

    assert result["projection"]["player_orbit_mode"] == mode
    assert result["throughput"]["speedup"] > 0


def test_wrapper_is_actor_policy_compatible_and_retry_mask_stays_exact():
    actor = BatchedSuitAveragedActor(
        StableSuitBiasedActor(), alpha=1.0
    )
    policy = ActorPolicy(
        actor,
        torch.device("cpu"),
        temperature=1.0,
        collect_data=False,
        greedy=True,
    )
    assert policy.actor is actor

    engine = BatchedPesten(
        np.full((1, 2), POLICY_RL, dtype=np.int8),
        np.full((1, 2), "candidate", dtype=object),
        np.asarray([711_331], dtype=np.uint32),
    )
    player = int(engine.current_player[0])
    engine.phase[0] = PHASE_DRAW_RETRY
    engine.drawn_card[0] = 16
    engine.hand_counts[0, player] = 0
    engine.hand_counts[0, player, 3] = 1
    engine.hand_counts[0, player, 16] = 1
    observation = torch.from_numpy(engine.observation(0))
    mask = build_action_masks(observation)

    assert torch.nonzero(mask).flatten().tolist() == [16, PASS_ACTION]


def test_batched_full_games_match_canonical_suit_symmetry_exactly():
    actor = StableSuitBiasedActor()
    device = torch.device("cpu")
    wrapper = BatchedSuitAveragedActor(actor, alpha=1.0)
    policies = {
        "candidate": ActorPolicy(
            wrapper, device, 1.0, False, greedy=True
        ),
        "parent": ActorPolicy(
            actor, device, 1.0, False, greedy=True
        ),
    }
    specs = build_game_specs(
        pairs=5,
        base_seed=6_711_331,
        pool_names=("random", "smart", "smarter"),
    )

    batched, _ = run_batched_specs(specs, policies, max_turns=500)
    canonical, _ = run_canonical_specs(
        specs,
        candidate_factory=make_suit_symmetry_ppo_factory(actor, device),
        parent_factory=make_corrected_ppo_agent_factory(actor, device),
        max_turns=500,
    )

    assert batched == canonical


def test_benchmark_reports_canonical_parity_and_speedup():
    actor = StableSuitBiasedActor()
    result = run_batched_symmetry_benchmark(
        actor,
        torch.device("cpu"),
        pairs=1,
        base_seed=7_711_331,
        pool_names=("random",),
        max_turns=100,
        bootstrap_samples=20,
        player_orbit_average=True,
        canonical_throughput_pairs=1,
    )

    assert result["projection"]["alpha"] == 1.0
    assert result["projection"]["aggregation"] == "logits"
    assert result["projection"]["player_orbit_average"]
    assert result["projection"]["parent_actor_is_candidate"]
    assert result["throughput"]["speedup"] > 0
    assert result["throughput"]["batched"]["mean_actor_batch_size"] > 1


def test_distinct_parent_actor_is_used_in_batched_and_canonical_paths():
    candidate = StableSuitBiasedActor()
    parent = CountingReverseActor()
    result = run_batched_symmetry_benchmark(
        candidate,
        torch.device("cpu"),
        parent_actor=parent,
        pairs=2,
        base_seed=8_711_331,
        pool_names=("random",),
        max_turns=100,
        bootstrap_samples=20,
        canonical_throughput_pairs=2,
    )

    assert not result["projection"]["parent_actor_is_candidate"]
    assert parent.calls > 0
    assert result["pairs_per_matchup"] == 2
    assert len(result["direct"]["per_pair"]) == 2
    # Reaching a speedup means the distinct-parent canonical outcomes exactly
    # matched the batched outcomes; the benchmark fails closed otherwise.
    assert result["throughput"]["speedup"] > 0


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        torch.device("meta"),
        *(
            [torch.device("cuda")]
            if torch.cuda.is_available()
            else []
        ),
    ],
)
def test_canonical_projection_wrapper_moves_gather_buffers_to_device(device):
    actor = StableSuitBiasedActor().to(device)
    factory = _canonical_candidate_factory(
        actor,
        device,
        alpha=0.5,
        aggregation="probabilities",
        player_orbit_average=True,
        base_chunk_size=8,
    )
    agent = factory(object())

    assert agent.actor.observation_indices.device == device
    assert agent.actor.action_indices.device == device
