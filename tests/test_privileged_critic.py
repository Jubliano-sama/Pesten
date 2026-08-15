from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from batched_pesten import POLICY_RL, BatchedPesten
from investigate_privileged_critic import (
    batches_for_epoch,
    candidate_representation,
    candidate_spec,
    configure_critic_trainable_scope,
    fit_value_calibration,
    hot_start_critic_weights,
)
from privileged_critic import (
    DATA_FIELDS,
    MAX_PILE_CARDS,
    AveragedPrivilegedCritic,
    AveragedPrivilegedCriticRuntime,
    CausalTemporalValueModel,
    CriticFeatureConfig,
    DEFAULT_GAMMA,
    FactorizedCardEmbedding,
    PrivilegedCriticRuntime,
    PrivilegedSnapshot,
    capture_privileged_snapshot,
    discounted_return_targets,
    discounted_returns,
    episode_from_snapshots,
    load_privileged_critic_checkpoint,
    load_privileged_critic_ensemble_checkpoint,
    pack_episodes,
    replay_privileged_trace,
    snapshots_to_model_batch,
    update_privileged_critic_from_ppo,
    value_metrics,
)
from temporal_batched_speed import SafeEventCursor, build_safe_decision


class TinyCausalPolicy(torch.nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.hidden_size = hidden_size

    def initial_state(self):
        return (torch.zeros(1, 1, self.hidden_size),)

    def step(self, decision, state):
        logits = torch.full((55,), torch.finfo(torch.float32).min)
        legal = torch.tensor(decision.legal_actions, dtype=torch.bool)
        logits[legal] = torch.linspace(-0.5, 0.5, int(legal.sum()))
        next_state = (state[0] + 0.125,)
        return SimpleNamespace(
            logits=logits,
            value=next_state[0].mean(),
            state=next_state,
        )


def make_engine(seed=881, players=3):
    kinds = np.full((1, players), POLICY_RL, dtype=np.int8)
    keys = np.full((1, players), None, dtype=object)
    return BatchedPesten(kinds, keys, np.asarray([seed], dtype=np.uint32))


def make_snapshot(engine, policy, cursors, states):
    seat = int(engine.current_player[0])
    decision = build_safe_decision(engine, 0, seat, cursors[seat])
    result = policy.step(decision, states[seat])
    states[seat] = result.state
    probabilities = torch.softmax(result.logits, dim=-1).numpy()
    return capture_privileged_snapshot(
        engine,
        0,
        seat,
        safe_observation=decision.observation.values,
        absolute_roles=("current",) * engine.num_players,
        actor_probabilities=probabilities,
        actor_memory=result.state,
        public_value=float(result.value),
    )


def test_snapshot_is_pre_action_complete_and_immutable():
    engine = make_engine()
    policy = TinyCausalPolicy()
    cursors = {seat: SafeEventCursor() for seat in range(engine.num_players)}
    states = {seat: policy.initial_state() for seat in range(engine.num_players)}
    snapshot = make_snapshot(engine, policy, cursors, states)

    draw_count = int(engine.draw_counts[0])
    assert snapshot.draw_length == draw_count
    assert snapshot.draw_order[0] == engine.draw_piles[0, draw_count - 1]
    assert snapshot.hands.shape == (13, 50)
    assert snapshot.hands[0].sum() == engine.hand_sizes[0, 0]
    assert snapshot.hand_order.shape == (13, MAX_PILE_CARDS)
    assert snapshot.hand_lengths[0] == engine.hand_sizes[0, 0]
    assert np.array_equal(
        snapshot.hand_order[0, : snapshot.hand_lengths[0]],
        engine.hands[0, 0, : engine.hand_sizes[0, 0]],
    )
    assert snapshot.actor_memory.shape == (policy.hidden_size,)
    assert snapshot.player_actor_memory.shape == (
        13,
        policy.hidden_size,
    )
    assert snapshot.player_actor_memory_mask[0]
    reconstructed_memory = (
        snapshot.player_actor_memory[0].astype(np.float32)
        * float(snapshot.player_actor_memory_scale[0])
    )
    assert np.allclose(
        reconstructed_memory,
        snapshot.actor_memory.astype(np.float32),
        atol=float(snapshot.player_actor_memory_scale[0]) * 0.58 + 2e-3,
    )
    assert not snapshot.draw_order.flags.writeable
    with pytest.raises(ValueError):
        snapshot.draw_order[0] = 49

    frozen_draw = snapshot.draw_order.copy()
    engine.apply_action(0, 54)
    assert np.array_equal(snapshot.draw_order, frozen_draw)
    assert not hasattr(snapshot, "action")
    assert not hasattr(snapshot, "winner")
    assert not hasattr(snapshot, "seed")
    tensorized = snapshots_to_model_batch([snapshot])
    assert tensorized["hands"].shape == (1, 13, 50)
    assert tensorized["public_value"].shape == (1,)


def test_snapshot_schema_has_no_outcome_or_rng_leakage_fields():
    names = {field.name.lower() for field in fields(PrivilegedSnapshot)}
    denied_fragments = (
        "action",
        "reward",
        "return",
        "winner",
        "won",
        "future",
        "seed",
        "random",
        "rng",
        "generator",
    )
    assert not {
        name
        for name in names
        if any(fragment in name for fragment in denied_fragments)
    }


def test_snapshot_and_gru_support_thirteen_players():
    engine = make_engine(seed=883, players=13)
    policy = TinyCausalPolicy()
    cursors = {seat: SafeEventCursor() for seat in range(13)}
    states = {seat: policy.initial_state() for seat in range(13)}
    snapshot = make_snapshot(engine, policy, cursors, states)
    assert snapshot.player_mask.sum() == 13
    assert (
        snapshot.hands.sum()
        + snapshot.draw_counts.sum()
        + snapshot.discard_counts.sum()
        == 108
    )
    model = CausalTemporalValueModel(
        actor_memory_size=8,
        backbone="gru",
        hidden_size=32,
        num_layers=1,
        group_width=16,
        transformer_heads=4,
    ).eval()
    runtime = PrivilegedCriticRuntime(model)
    output = runtime.evaluate([snapshot], [(883, 0)])
    assert output["value"].shape == (1,)


def build_direct_trace(seed=991, players=3, max_turns=120):
    engine = make_engine(seed=seed, players=players)
    cursors = {seat: SafeEventCursor() for seat in range(players)}
    opening_count = int(engine.discard_counts[0])
    trace = {
        "absolute_game_index": 0,
        "deal_seed": seed,
        "initial_deal": {
            "hands": [
                engine.hands[0, seat, : int(engine.hand_sizes[0, seat])]
                .astype(int)
                .tolist()
                for seat in range(players)
            ],
            "opening_card": int(engine.discards[0, opening_count - 1]),
        },
        "actions": [],
    }
    while len(engine.active_games()) and int(engine.turn_counts[0]) < max_turns:
        seat = int(engine.current_player[0])
        decision = build_safe_decision(engine, 0, seat, cursors[seat])
        legal = np.flatnonzero(decision.legal_actions)
        card_actions = legal[legal < 50]
        action = int(card_actions[0] if len(card_actions) else legal[0])
        trace["actions"].append(
            {
                "seat": seat,
                "role": "current",
                "phase": int(engine.phase[0]),
                "action": action,
                "legal_actions": list(decision.legal_actions),
            }
        )
        engine.apply_action(0, action)
    trace["winner"] = int(engine.winners[0])
    trace["turn_count"] = int(engine.turn_counts[0])
    trace["public_events"] = tuple(
        tuple(item) for item in engine.public_events[0]
    )
    return trace


def test_trace_replay_reconstructs_every_current_seat_episode():
    policy = TinyCausalPolicy()
    trace = build_direct_trace()
    episodes = replay_privileged_trace(
        policy,
        trace,
        opponent_type="current_selfplay",
        player_count=3,
    )

    assert episodes
    assert {int(episode["seat"]) for episode in episodes}.issubset({0, 1, 2})
    assert sum(len(episode["returns"]) for episode in episodes) == len(
        trace["actions"]
    )
    for episode in episodes:
        returns = discounted_returns(
            len(episode["returns"]), won=bool(episode["won"])
        )
        assert np.allclose(episode["returns"], returns)
        assert np.all(episode["player_actor_memory_mask"][:, :3])


def test_privileged_returns_prefer_default_draw_to_loss():
    loss = discounted_returns(2, won=False, gamma=1.0)
    draw = discounted_returns(2, won=None, gamma=1.0)

    assert np.allclose(loss, np.asarray([-1.02, -1.01], dtype=np.float32))
    assert np.allclose(draw, np.asarray([-0.77, -0.76], dtype=np.float32))


def synthetic_snapshots():
    engine = make_engine(seed=992, players=3)
    policy = TinyCausalPolicy()
    cursors = {seat: SafeEventCursor() for seat in range(engine.num_players)}
    states = {seat: policy.initial_state() for seat in range(engine.num_players)}
    snapshots = []
    while len(snapshots) < 3:
        snapshots.append(make_snapshot(engine, policy, cursors, states))
        engine.apply_action(0, 54)
    return snapshots


def synthetic_model_batch():
    snapshots = synthetic_snapshots()
    episode = episode_from_snapshots(
        snapshots,
        won=False,
        deal_seed=992,
        seat=0,
        player_count=3,
    )
    packed = pack_episodes([episode])
    batch = {name: packed[name].unsqueeze(0) for name in DATA_FIELDS}
    mask = torch.ones(1, 3, dtype=torch.bool)
    return batch, mask


@pytest.mark.parametrize("backbone", ("gru", "lstm", "transformer"))
def test_temporal_backbones_are_causal(backbone):
    torch.manual_seed(17)
    batch, mask = synthetic_model_batch()
    model = CausalTemporalValueModel(
        actor_memory_size=8,
        features=CriticFeatureConfig(
            draw_top_n=4,
            discard_top_n=4,
            played_top_n=4,
        ),
        backbone=backbone,
        hidden_size=32,
        num_layers=1,
        group_width=16,
        transformer_heads=4,
    ).eval()
    changed = {name: value.clone() for name, value in batch.items()}
    changed["safe_observation"][:, 2] += 7
    changed["hands"][:, 2].zero_()
    changed["actor_probabilities"][:, 2] = changed[
        "actor_probabilities"
    ][:, 2].roll(1, dims=-1)

    with torch.inference_mode():
        original = model(batch, mask=mask)["value"]
        perturbed = model(changed, mask=mask)["value"]

    assert torch.allclose(original[:, :2], perturbed[:, :2], atol=1e-6)


@pytest.mark.parametrize("backbone", ("mlp", "gru", "lstm"))
def test_streaming_step_matches_full_causal_forward(backbone):
    torch.manual_seed(18)
    batch, mask = synthetic_model_batch()
    model = CausalTemporalValueModel(
        actor_memory_size=8,
        backbone=backbone,
        hidden_size=32,
        num_layers=1,
        group_width=16,
        transformer_heads=4,
        dropout=0.0,
    ).eval()
    state = model.initial_state(batch_size=1)
    streamed = []
    with torch.inference_mode():
        full = model(batch, mask=mask)["value"]
        for time_index in range(mask.shape[1]):
            turn = {
                name: value[:, time_index] for name, value in batch.items()
            }
            output, state = model.step(turn, state)
            streamed.append(output["value"])

    assert torch.allclose(full, torch.stack(streamed, dim=1), atol=1e-6)


def test_pack_and_metrics_contracts():
    batch, _ = synthetic_model_batch()
    assert batch["draw_order"].shape[-1] == MAX_PILE_CARDS
    target = batch["returns"].float().numpy().reshape(-1)
    exact = value_metrics(target, target)
    assert exact["mse"] == 0
    assert exact["explained_variance"] == 1


def test_default_discount_is_gamma_0p995():
    assert DEFAULT_GAMMA == pytest.approx(0.995)
    assert DEFAULT_GAMMA**20 == pytest.approx(0.9046104802746175)
    for won in (False, True):
        expected = discounted_returns(20, won=won)
        actual = discounted_return_targets(
            torch.arange(20, 0, -1),
            torch.full((20,), won),
        )
        assert np.allclose(actual.numpy(), expected, atol=2e-7)


def test_global_length_buckets_cap_pathological_padding():
    class LengthCorpus:
        def __init__(self):
            self.lengths = [1, 2, 3, 4, 5, 17, 31, 128, 300]

        def __len__(self):
            return len(self.lengths)

        def length(self, index):
            return self.lengths[index]

    corpus = LengthCorpus()
    batches = batches_for_epoch(
        corpus,
        batch_size=4,
        seed=19,
        shuffle=True,
        max_batch_decisions=128,
    )
    assert sorted(index for batch in batches for index in batch) == list(
        range(len(corpus))
    )
    for batch in batches:
        padded = max(corpus.length(index) for index in batch) * len(batch)
        assert padded <= 128 or len(batch) == 1


def test_policy_aware_critic_starts_as_exact_public_value_residual():
    batch, mask = synthetic_model_batch()
    model = CausalTemporalValueModel(
        actor_memory_size=8,
        backbone="gru",
        hidden_size=32,
        num_layers=1,
        group_width=16,
        transformer_heads=4,
    ).eval()

    with torch.inference_mode():
        values = model(batch, mask=mask)["value"]

    assert torch.equal(values, batch["public_value"].float())

    no_public = CausalTemporalValueModel(
        actor_memory_size=8,
        features=CriticFeatureConfig(public_value=False),
        backbone="gru",
        hidden_size=32,
        num_layers=1,
        group_width=16,
        transformer_heads=4,
    ).eval()
    with torch.inference_mode():
        no_public_values = no_public(batch, mask=mask)["value"]
    assert torch.count_nonzero(no_public_values) == 0


def test_player_count_value_calibration_is_explicit_and_reversible():
    batch, mask = synthetic_model_batch()
    model = CausalTemporalValueModel(
        actor_memory_size=8,
        backbone="mlp",
        hidden_size=32,
        num_layers=1,
        group_width=16,
        transformer_heads=4,
        dropout=0.0,
    ).eval()
    prediction = np.asarray([-1.0, 0.0, 1.0, -0.5, 0.5])
    target = 0.5 * prediction + 0.2
    counts = np.asarray([3, 3, 3, 4, 4])
    fit = fit_value_calibration(
        model, prediction, target, counts, mode="player_count"
    )
    assert fit["mse_after_fit"] < 1e-12

    with torch.inference_mode():
        calibrated = model(batch, mask=mask)["value"]
    assert torch.allclose(
        calibrated,
        batch["public_value"].float() * 0.5 + 0.2,
        atol=1e-6,
    )
    model.reset_value_calibration()
    with torch.inference_mode():
        reset = model(batch, mask=mask)["value"]
    assert torch.equal(reset, batch["public_value"].float())


def test_hot_start_loads_weights_resets_calibration_and_records_source(tmp_path):
    source = CausalTemporalValueModel(
        actor_memory_size=8,
        backbone="mlp",
        hidden_size=32,
        num_layers=1,
        group_width=16,
        dropout=0.0,
    )
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.125)
    source.set_value_calibration([1.5] * 14, [0.25] * 14)
    checkpoint = tmp_path / "source.pt"
    torch.save(
        {
            "format": "pesten-causal-privileged-critic-v1",
            "candidate": "selected_moments_exact_mlp",
            "epoch": 7,
            "model": source.model_config(),
            "state_dict": source.state_dict(),
            "target_gamma": 0.9330329915368074,
        },
        checkpoint,
    )
    target = CausalTemporalValueModel(
        actor_memory_size=8,
        backbone="mlp",
        hidden_size=32,
        num_layers=1,
        group_width=16,
        dropout=0.0,
    )

    provenance = hot_start_critic_weights(target, checkpoint)

    assert provenance["source_epoch"] == 7
    assert provenance["source_target_gamma"] == pytest.approx(
        0.9330329915368074
    )
    assert provenance["optimizer_initialization"] == "fresh AdamW"
    assert provenance["source_validation_calibration_reset"] is True
    for name, parameter in target.named_parameters():
        assert torch.equal(parameter, dict(source.named_parameters())[name])
    assert torch.equal(
        target.value_calibration_scale,
        torch.ones_like(target.value_calibration_scale),
    )
    assert torch.equal(
        target.value_calibration_bias,
        torch.zeros_like(target.value_calibration_bias),
    )

    incompatible = CausalTemporalValueModel(
        actor_memory_size=8,
        backbone="mlp",
        hidden_size=64,
        num_layers=1,
        group_width=16,
        dropout=0.0,
    )
    with pytest.raises(ValueError, match="architecture does not match"):
        hot_start_critic_weights(incompatible, checkpoint)


def test_value_head_scope_freezes_hot_start_representation():
    model = CausalTemporalValueModel(
        actor_memory_size=8,
        backbone="mlp",
        hidden_size=32,
        num_layers=1,
        group_width=16,
        dropout=0.0,
    )
    evidence = configure_critic_trainable_scope(model, "value_head")

    assert evidence["scope"] == "value_head"
    assert evidence["trainable_parameters"] > 0
    assert evidence["trainable_parameters"] < evidence["total_parameters"]
    assert all(
        parameter.requires_grad == name.startswith("value_head.")
        for name, parameter in model.named_parameters()
    )


def test_hand_order_feature_is_order_sensitive_and_maskable():
    batch, _ = synthetic_model_batch()
    changed = {name: value.clone() for name, value in batch.items()}
    row = changed["hand_order"][0, 0, 0]
    valid = row[row >= 0]
    right = next(
        index for index in range(1, len(valid)) if valid[index] != valid[0]
    )
    row[0], row[right] = row[right].clone(), row[0].clone()
    model = CausalTemporalValueModel(
        actor_memory_size=8,
        hidden_size=32,
        num_layers=1,
        group_width=16,
        transformer_heads=4,
    ).eval()
    enabled = CriticFeatureConfig()
    disabled = CriticFeatureConfig(hand_order=False)

    with torch.inference_mode():
        original_enabled = model.snapshot_encoder(batch, enabled)
        changed_enabled = model.snapshot_encoder(changed, enabled)
        original_disabled = model.snapshot_encoder(batch, disabled)
        changed_disabled = model.snapshot_encoder(changed, disabled)

    assert not torch.allclose(original_enabled, changed_enabled)
    assert torch.equal(original_disabled, changed_disabled)


def test_rank_suit_card_tokens_are_lossless_and_pad_separately():
    embedding = FactorizedCardEmbedding(16)
    pairs = list(
        zip(
            embedding.card_suits[:50].tolist(),
            embedding.card_ranks[:50].tolist(),
        )
    )
    assert len(set(pairs)) == 50
    assert pairs[0] == (0, 1)
    assert pairs[47] == (3, 12)
    assert pairs[48] == (4, 13)
    assert pairs[49] == (4, 0)
    assert (
        int(embedding.card_suits[50]),
        int(embedding.card_ranks[50]),
    ) == (5, 14)


@pytest.mark.parametrize(
    ("hand_encoder", "card_tokenization"),
    (
        ("token_gru", "exact"),
        ("token_gru", "rank_suit"),
        ("token_bigru", "rank_suit"),
    ),
)
def test_token_hand_encoders_replace_counts_and_retain_order(
    hand_encoder, card_tokenization
):
    batch, _ = synthetic_model_batch()
    features = CriticFeatureConfig(safe_observation=False)
    model = CausalTemporalValueModel(
        actor_memory_size=8,
        features=features,
        backbone="mlp",
        hidden_size=32,
        num_layers=1,
        group_width=16,
        transformer_heads=4,
        dropout=0.0,
        hand_encoder=hand_encoder,
        card_tokenization=card_tokenization,
    ).eval()
    changed_counts = {name: value.clone() for name, value in batch.items()}
    changed_counts["hands"][:, :, 0].zero_()
    changed_order = {name: value.clone() for name, value in batch.items()}
    row = changed_order["hand_order"][0, 0, 0]
    valid = row[row >= 0]
    right = next(
        index for index in range(1, len(valid)) if valid[index] != valid[0]
    )
    row[0], row[right] = row[right].clone(), row[0].clone()

    with torch.inference_mode():
        original = model.snapshot_encoder(batch, features)
        counts_only = model.snapshot_encoder(changed_counts, features)
        reordered = model.snapshot_encoder(changed_order, features)

    assert torch.equal(original, counts_only)
    assert not torch.allclose(original, reordered)
    assert model.model_config()["hand_encoder"] == hand_encoder
    assert model.model_config()["card_tokenization"] == card_tokenization


def test_opponent_memory_feature_is_sensitive_and_maskable():
    batch, _ = synthetic_model_batch()
    changed = {name: value.clone() for name, value in batch.items()}
    changed["player_actor_memory"][:, :, 1, 0] += 10
    changed["player_actor_memory_mask"][:, :, 1] = True
    changed["player_actor_memory_scale"][:, :, 1] = 0.05
    model = CausalTemporalValueModel(
        actor_memory_size=8,
        hidden_size=32,
        num_layers=1,
        group_width=16,
        transformer_heads=4,
    ).eval()
    enabled = CriticFeatureConfig()
    disabled = CriticFeatureConfig(opponent_memories=False)

    with torch.inference_mode():
        original_enabled = model.snapshot_encoder(batch, enabled)
        changed_enabled = model.snapshot_encoder(changed, enabled)
        original_disabled = model.snapshot_encoder(batch, disabled)
        changed_disabled = model.snapshot_encoder(changed, disabled)

    assert not torch.allclose(original_enabled, changed_enabled)
    assert torch.equal(original_disabled, changed_disabled)


def test_checkpoint_round_trip_is_actor_independent(tmp_path):
    batch, mask = synthetic_model_batch()
    model = CausalTemporalValueModel(
        actor_memory_size=8,
        backbone="gru",
        hidden_size=32,
        num_layers=1,
        group_width=16,
        transformer_heads=4,
        dropout=0.0,
    ).eval()
    checkpoint = tmp_path / "critic.pt"
    torch.save(
        {
            "format": "pesten-causal-privileged-critic-v1",
            "model": model.model_config(),
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )

    loaded, payload = load_privileged_critic_checkpoint(checkpoint)
    with torch.inference_mode():
        expected = model(batch, mask=mask)["value"]
        actual = loaded(batch, mask=mask)["value"]

    assert payload["format"] == "pesten-causal-privileged-critic-v1"
    assert torch.equal(expected, actual)

    legacy_state = dict(model.state_dict())
    legacy_state.pop("value_calibration_scale")
    legacy_state.pop("value_calibration_bias")
    torch.save(
        {
            "format": "pesten-causal-privileged-critic-v1",
            "model": model.model_config(),
            "state_dict": legacy_state,
        },
        checkpoint,
    )
    legacy, _ = load_privileged_critic_checkpoint(checkpoint)
    assert torch.all(legacy.value_calibration_scale == 1)
    assert torch.all(legacy.value_calibration_bias == 0)

    legacy_config = model.model_config()
    legacy_config.pop("hand_encoder")
    legacy_config.pop("card_tokenization")
    torch.save(
        {
            "format": "pesten-causal-privileged-critic-v1",
            "model": legacy_config,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    compatible, _ = load_privileged_critic_checkpoint(checkpoint)
    assert compatible.hand_encoder == "counts_moments"
    assert compatible.card_tokenization == "exact"


def test_feature_ablation_names_are_backbone_independent():
    for backbone in ("mlp", "gru", "lstm", "transformer"):
        resolved, features = candidate_spec(f"no_draw_{backbone}")
        assert resolved == backbone
        assert not features.draw_composition
        assert not features.draw_order
        resolved, features = candidate_spec(f"draw_top_8_{backbone}")
        assert resolved == backbone
        assert features.draw_top_n == 8
        resolved, features = candidate_spec(
            f"selected_token_bigru_rank_suit_{backbone}"
        )
        assert resolved == backbone
        assert not features.safe_observation
        assert candidate_representation(
            f"selected_token_bigru_rank_suit_{backbone}"
        ) == ("token_bigru", "rank_suit")


def test_online_runtime_batches_independent_stream_state():
    snapshots = synthetic_snapshots()
    model = CausalTemporalValueModel(
        actor_memory_size=8,
        backbone="gru",
        hidden_size=32,
        num_layers=1,
        group_width=16,
        transformer_heads=4,
        dropout=0.0,
    ).eval()
    torch.nn.init.normal_(model.value_head[-1].weight, std=0.1)
    runtime = PrivilegedCriticRuntime(model)

    first = runtime.evaluate([snapshots[0]], [(992, 0)])["value"]
    second = runtime.evaluate([snapshots[1]], [(992, 0)])["value"]
    parallel_fresh = runtime.evaluate(
        [snapshots[0], snapshots[1]], [(993, 0), (994, 0)]
    )["value"]

    assert first.shape == (1,)
    assert second.shape == (1,)
    assert torch.allclose(first, parallel_fresh[:1], atol=1e-6, rtol=0)
    assert len(runtime.states) == 3
    runtime.reset([(992, 0)])
    assert (992, 0) not in runtime.states


def test_ensemble_offline_and_streaming_values_match():
    batch, mask = synthetic_model_batch()
    snapshots = synthetic_snapshots()
    models = []
    for residual in (0.2, 0.6):
        model = CausalTemporalValueModel(
            actor_memory_size=8,
            backbone="mlp",
            hidden_size=32,
            num_layers=1,
            group_width=16,
            transformer_heads=4,
            dropout=0.0,
        ).eval()
        with torch.no_grad():
            model.value_head[-1].bias.fill_(residual)
        models.append(model)
    ensemble = AveragedPrivilegedCritic(models).eval()
    scale = np.ones(14, dtype=np.float32)
    bias = np.zeros(14, dtype=np.float32)
    scale[3], bias[3] = 0.5, 0.1
    ensemble.set_value_calibration(scale, bias)

    with torch.inference_mode():
        offline = ensemble(batch, mask=mask)["value"]
    expected = (batch["public_value"].float() + 0.4) * 0.5 + 0.1
    assert torch.allclose(offline, expected, atol=1e-6)

    runtime = AveragedPrivilegedCriticRuntime(ensemble)
    streamed = []
    for index, snapshot in enumerate(snapshots):
        streamed.append(
            runtime.evaluate([snapshot], [(992, 0)])["value"]
        )
    assert torch.allclose(
        torch.stack(streamed, dim=1), expected, atol=1e-6
    )


def test_ensemble_checkpoint_is_self_contained(tmp_path):
    models = [
        CausalTemporalValueModel(
            actor_memory_size=8,
            backbone="mlp",
            hidden_size=32,
            num_layers=1,
            group_width=16,
            transformer_heads=4,
            dropout=0.0,
        ).eval()
        for _ in range(2)
    ]
    path = tmp_path / "ensemble.pt"
    torch.save(
        {
            "format": "pesten-causal-privileged-critic-ensemble-v1",
            "members": [
                {
                    "model": model.model_config(),
                    "state_dict": model.state_dict(),
                }
                for model in models
            ],
            "value_calibration_scale": torch.ones(14),
            "value_calibration_bias": torch.zeros(14),
        },
        path,
    )

    loaded, payload = load_privileged_critic_ensemble_checkpoint(path)

    assert len(loaded.models) == 2
    assert payload["format"].endswith("ensemble-v1")


def test_on_policy_external_critic_update_uses_retained_snapshots_only():
    snapshots = synthetic_snapshots()
    trajectory = tuple(
        SimpleNamespace(privileged_snapshot=snapshot)
        for snapshot in snapshots
    )
    model = CausalTemporalValueModel(
        actor_memory_size=8,
        backbone="mlp",
        hidden_size=32,
        num_layers=1,
        group_width=16,
        transformer_heads=4,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.value_head[-1].weight.detach().clone()
    calibration_before = model.value_calibration_scale.clone()

    result = update_privileged_critic_from_ppo(
        model,
        [(trajectory, True)],
        optimizer,
        update_epochs=1,
        minibatch_trajectories=1,
        win_coefficient=0.0,
        remaining_coefficient=0.0,
    )

    assert result["trajectories"] == 1
    assert result["decisions"] == len(snapshots)
    assert not torch.equal(before, model.value_head[-1].weight)
    assert torch.equal(calibration_before, model.value_calibration_scale)
