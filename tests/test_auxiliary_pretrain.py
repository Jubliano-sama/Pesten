from dataclasses import dataclass

import torch

from auxiliary_pretrain import (
    NO_PUBLIC_EVENT,
    AuxiliaryBatch,
    PublicHistoryAuxiliaryModel,
    auxiliary_metrics,
    initialize_residual_encoder,
    pack_auxiliary_episodes,
    pretrain_auxiliary_encoder,
    validate_development_seed,
)
from benchmark_auxiliary_pretrain import canonical_behavior_digest
from benchmark_model_zoo import DEFAULT_TEACHER
from residual_temporal_policy import ResidualRecurrentTemporalPolicy
from temporal_model_zoo import SafeSequenceBatch
from temporal_observation import (
    DecisionInput,
    EventKind,
    PublicEventToken,
    build_legal_actions,
    build_private_observation,
)
from temporal_puffer_agent import TrajectoryStep


@dataclass(frozen=True)
class Sample:
    decision: DecisionInput
    action: int = 54


def make_decision(
    *,
    hand_sizes=(3, 4, 5),
    own_cards=(4,),
    events=(),
):
    observation = build_private_observation(
        own_cards=own_cards,
        current_suit="harten",
        current_rank=5,
        public_hand_sizes=hand_sizes,
        observer_index=0,
        current_player_index=0,
        direction=1,
        penalty_amount=0,
        public_sorts_played=(2, 1, 0, 1, 0),
    )
    legal = build_legal_actions(
        own_cards=own_cards,
        current_suit="harten",
        current_rank=5,
        penalty_amount=0,
    )
    return DecisionInput(observation, legal, tuple(events))


def synthetic_episode(offset=0):
    return (
        Sample(
            make_decision(
                hand_sizes=(3, 4 + offset, 5),
                events=(PublicEventToken(EventKind.EPISODE_START),),
            )
        ),
        Sample(
            make_decision(
                hand_sizes=(3, 6 + offset, 5),
                events=(
                    PublicEventToken(
                        EventKind.DRAW_COUNT,
                        actor_relative=1,
                        count=2,
                    ),
                ),
            )
        ),
        Sample(
            make_decision(
                hand_sizes=(3, 5 + offset, 5),
                events=(
                    PublicEventToken(
                        EventKind.PLAY,
                        actor_relative=1,
                        card_id=23,
                        suit_id=1,
                        count=1,
                    ),
                ),
            )
        ),
    )


def test_targets_use_only_next_public_event_and_opponent_size_delta():
    batch = pack_auxiliary_episodes([synthetic_episode()])
    assert batch.sample_count == 2
    assert batch.next_event_targets.tolist() == [
        int(EventKind.DRAW_COUNT),
        int(EventKind.PLAY),
    ]
    assert batch.opponent_hand_delta_targets.tolist() == [2.0, -1.0]
    no_event = (
        Sample(make_decision()),
        Sample(make_decision(events=())),
    )
    empty_target = pack_auxiliary_episodes([no_event])
    assert empty_target.next_event_targets.item() == NO_PUBLIC_EVENT


def test_gru_prediction_at_current_decision_cannot_see_future_tokens():
    torch.manual_seed(7)
    model = PublicHistoryAuxiliaryModel(
        hidden_size=16, num_layers=1, cell="gru"
    ).eval()
    batch = pack_auxiliary_episodes([synthetic_episode()])
    with torch.no_grad():
        original_event, original_delta = model.forward_batch(batch)
    changed_tokens = batch.safe.tokens.clone()
    first_prediction_time = int(batch.prediction_rows[0, 1])
    changed_tokens[:, first_prediction_time + 1 :] = 123.0
    changed_safe = SafeSequenceBatch(
        tokens=changed_tokens,
        valid_tokens=batch.safe.valid_tokens,
        decision_rows=batch.safe.decision_rows,
        legal_actions=batch.safe.legal_actions,
        targets=batch.safe.targets,
    )
    changed = AuxiliaryBatch(
        safe=changed_safe,
        prediction_rows=batch.prediction_rows,
        next_event_targets=batch.next_event_targets,
        opponent_hand_delta_targets=batch.opponent_hand_delta_targets,
    )
    with torch.no_grad():
        changed_event, changed_delta = model.forward_batch(changed)
    assert torch.equal(original_event[0], changed_event[0])
    assert torch.equal(original_delta[0], changed_delta[0])


def test_auxiliary_training_reports_heldout_metrics_and_updates_encoder():
    torch.manual_seed(11)
    model = PublicHistoryAuxiliaryModel(
        hidden_size=16, num_layers=1, cell="gru"
    )
    training = [synthetic_episode(index) for index in range(6)]
    validation = [synthetic_episode(7), synthetic_episode(8)]
    before = model.token_projection[0].weight.detach().clone()
    result = pretrain_auxiliary_encoder(
        model,
        training,
        validation,
        device="cpu",
        epochs=2,
        batch_episodes=2,
        learning_rate=1e-3,
        seed=501,
    )
    assert not torch.equal(before, model.token_projection[0].weight)
    assert result["before"]["samples"] == 4
    assert result["after"]["samples"] == 4
    assert 0.0 <= result["after"]["next_event_accuracy"] <= 1.0
    assert result["after"]["opponent_delta_mae"] >= 0.0
    assert auxiliary_metrics(
        model, validation, device="cpu"
    ) == result["after"]


def test_encoder_transfer_leaves_actor_critic_and_teacher_untouched():
    torch.manual_seed(13)
    auxiliary = PublicHistoryAuxiliaryModel(
        hidden_size=16, num_layers=1, cell="gru"
    )
    torch.manual_seed(14)
    policy = ResidualRecurrentTemporalPolicy(
        champion_path=DEFAULT_TEACHER,
        hidden_size=16,
        num_layers=1,
        cell="gru",
    )
    protected = {
        name: tensor.detach().clone()
        for name, tensor in policy.state_dict().items()
        if (
            "actor" in name
            or name.startswith("critic.")
            or name.startswith("teacher.")
        )
    }
    transfer = initialize_residual_encoder(policy, auxiliary)
    assert transfer["encoder_sha256_before"] != transfer[
        "encoder_sha256_after"
    ]
    assert policy.token_projection.state_dict().keys() == (
        auxiliary.token_projection.state_dict().keys()
    )
    assert all(
        torch.equal(
            tensor,
            auxiliary.token_projection.state_dict()[name],
        )
        for name, tensor in policy.token_projection.state_dict().items()
    )
    assert all(
        torch.equal(before, policy.state_dict()[name])
        for name, before in protected.items()
    )


def test_auxiliary_checkpoint_and_behavior_digest_contract(tmp_path):
    model = PublicHistoryAuxiliaryModel(
        hidden_size=8, num_layers=1, cell="lstm"
    )
    checkpoint = tmp_path / "aux.pt"
    model.save_checkpoint(checkpoint)
    restored = PublicHistoryAuxiliaryModel.load_checkpoint(checkpoint)
    assert restored.cell == "lstm"
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            model.state_dict().values(), restored.state_dict().values()
        )
    )

    decision = make_decision()
    left = [(
        (TrajectoryStep(decision, 54, -0.2, 1.0),),
        False,
    )]
    different_critic = [(
        (TrajectoryStep(decision, 54, -0.2, -9.0),),
        False,
    )]
    different_action = [(
        (TrajectoryStep(decision, 4, -0.2, 1.0),),
        False,
    )]
    assert canonical_behavior_digest(left) == canonical_behavior_digest(
        different_critic
    )
    assert canonical_behavior_digest(left) != canonical_behavior_digest(
        different_action
    )


def test_sealed_auxiliary_seed_is_rejected():
    try:
        validate_development_seed(92_000_000)
    except ValueError as error:
        assert "sealed" in str(error)
    else:
        raise AssertionError("sealed seed unexpectedly accepted")
