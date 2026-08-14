"""Information-safe recurrent distillation from a Puffer teacher to a student."""

from __future__ import annotations

import random
from typing import Sequence

import torch
import torch.nn.functional as F

from temporal_observation import PASS_ACTION
from temporal_training import _gather_native_decisions, _pack_native_token_batch


def _forward_decisions(policy, episodes):
    parameter = next(policy.parameters())
    packed_rows = []
    decision_indices = []
    legal_masks = []
    for trajectory, _ in episodes:
        if not trajectory:
            continue
        rows = []
        indices = []
        for step in trajectory:
            rows.extend(policy._event_features(step.decision))
            rows.append(policy._decision_features(step.decision))
            indices.append(len(rows) - 1)
            legal_masks.append(step.decision.legal_actions)
        packed_rows.append(rows)
        decision_indices.append(indices)
    if not packed_rows:
        raise ValueError("distillation requires at least one decision")
    max_length = max(len(rows) for rows in packed_rows)
    tokens = _pack_native_token_batch(
        packed_rows,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    all_logits, all_values = policy.forward_native_sequence(tokens)
    logits, values = _gather_native_decisions(
        all_logits.reshape(len(packed_rows), max_length, -1),
        all_values.reshape(len(packed_rows), max_length),
        decision_indices,
        max_length=max_length,
        device=parameter.device,
    )
    masks = torch.tensor(
        legal_masks, device=parameter.device, dtype=torch.bool
    )
    logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
    return logits, values, masks


def initialize_widened_min_gru(student, teacher) -> dict:
    """Embed a narrower MinGRU teacher as an exact student subnetwork.

    The layer count must remain unchanged. Teacher channels occupy the first
    part of every wider tensor and cannot read from new channels. New channels
    retain their random encoder and recurrent features, while the decoder
    initially ignores them. Batched outputs are therefore mathematically
    identical (up to GEMM roundoff), but decoder gradients can activate the
    new features on the first RL update.
    """
    teacher_hidden = int(teacher.hidden_size)
    student_hidden = int(student.hidden_size)
    teacher_layers = int(teacher.num_layers)
    student_layers = int(student.num_layers)
    if student_hidden <= teacher_hidden:
        raise ValueError("widened student must be wider than its teacher")
    if student_layers != teacher_layers:
        raise ValueError(
            "function-preserving MinGRU widening requires equal layer counts"
        )
    if bool(student.mode_conditioned_heads) != bool(
        teacher.mode_conditioned_heads
    ):
        raise ValueError("teacher and student decoder modes must match")
    if getattr(student, "action_head", None) != getattr(
        teacher, "action_head", None
    ):
        raise ValueError("teacher and student action heads must match")

    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    with torch.no_grad():
        encoder = "puffer_policy.encoder.encoder"
        student_state[f"{encoder}.weight"][:teacher_hidden].copy_(
            teacher_state[f"{encoder}.weight"]
        )
        student_state[f"{encoder}.bias"][:teacher_hidden].copy_(
            teacher_state[f"{encoder}.bias"]
        )
        if getattr(student, "action_head", None) == "factorized_23":
            decoder_names = ("scores", "value_function")
        elif student.mode_conditioned_heads:
            decoder_names = ("card_and_pass", "suit", "value_function")
        else:
            decoder_names = ("decoder", "value_function")
        for name in decoder_names:
            prefix = f"puffer_policy.decoder.{name}"
            student_state[f"{prefix}.weight"][:, teacher_hidden:].zero_()
            student_state[f"{prefix}.weight"][:, :teacher_hidden].copy_(
                teacher_state[f"{prefix}.weight"]
            )
            student_state[f"{prefix}.bias"].copy_(
                teacher_state[f"{prefix}.bias"]
            )

        for layer in range(teacher_layers):
            key = f"puffer_policy.network.layers.{layer}.weight"
            teacher_weight = teacher_state[key]
            student_weight = student_state[key]
            for chunk in range(3):
                teacher_rows = student_weight[
                    chunk * student_hidden : (
                        chunk * student_hidden + teacher_hidden
                    )
                ]
                teacher_rows.zero_()
                teacher_rows[:, :teacher_hidden].copy_(
                    teacher_weight[
                        chunk * teacher_hidden : (chunk + 1) * teacher_hidden
                    ]
                )
        student.load_state_dict(student_state, strict=True)
    return {
        "kind": "function_preserving_min_gru_widening",
        "teacher_hidden_size": teacher_hidden,
        "student_hidden_size": student_hidden,
        "num_layers": teacher_layers,
        "new_channels": student_hidden - teacher_hidden,
        "new_channels_randomly_featured": True,
        "new_channels_decoder_zero_initialized": True,
    }


def initialize_factorized_min_gru(student, teacher) -> dict:
    """Copy a MinGRU teacher and project its 48 card rows into suit + rank.

    The recurrent body and value function remain exact. The 4-by-12 ordinary
    card tensor is projected onto the additive two-way model
    ``card_logit[suit, rank] = suit_score + rank_score``. Jack, joker, pass,
    and announced-suit rows copy exactly into their dedicated learned scores.
    """
    from temporal_puffer_v4 import (
        FACTORIZED_ACTION_HEAD,
        MODE_CONDITIONED_ACTION_HEAD,
    )

    if getattr(student, "action_head", None) != FACTORIZED_ACTION_HEAD:
        raise ValueError("student must use the factorized_23 action head")
    if getattr(teacher, "action_head", None) not in (
        None,
        MODE_CONDITIONED_ACTION_HEAD,
    ):
        raise ValueError("teacher must use the 55-way mode-conditioned head")
    if not bool(getattr(teacher, "mode_conditioned_heads", False)):
        raise ValueError("teacher must use mode-conditioned heads")
    dimensions = (
        int(student.hidden_size),
        int(student.num_layers),
        int(teacher.hidden_size),
        int(teacher.num_layers),
    )
    if dimensions[:2] != dimensions[2:]:
        raise ValueError(
            "factorized distillation initialization requires the same "
            "MinGRU width and layer count"
        )

    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    copied_tensors = 0
    with torch.no_grad():
        for name in student_state:
            if name.startswith("puffer_policy.encoder.") or name.startswith(
                "puffer_policy.network."
            ):
                student_state[name].copy_(teacher_state[name])
                copied_tensors += 1

        teacher_value = "puffer_policy.decoder.value_function"
        for suffix in ("weight", "bias"):
            student_state[f"{teacher_value}.{suffix}"].copy_(
                teacher_state[f"{teacher_value}.{suffix}"]
            )
            copied_tensors += 1

        score_prefix = "puffer_policy.decoder.scores"
        card_prefix = "puffer_policy.decoder.card_and_pass"
        suit_prefix = "puffer_policy.decoder.suit"
        projection_errors = {}
        for suffix in ("weight", "bias"):
            teacher_cards = teacher_state[f"{card_prefix}.{suffix}"]
            teacher_suits = teacher_state[f"{suit_prefix}.{suffix}"]
            student_scores = student_state[f"{score_prefix}.{suffix}"]
            ordinary = teacher_cards[:48].reshape(
                4, 12, *teacher_cards.shape[1:]
            )
            suit_means = ordinary.mean(dim=1)
            rank_means = ordinary.mean(dim=0)
            grand_mean = ordinary.mean(dim=(0, 1))

            # This gauge is the minimum-L2-norm decomposition among all
            # equivalent suit/rank offsets.
            student_scores[0:4].copy_(suit_means - 0.25 * grand_mean)
            student_scores[4:16].copy_(rank_means - 0.75 * grand_mean)
            student_scores[16:19].copy_(teacher_cards[48:51])
            student_scores[19:23].copy_(teacher_suits)

            reconstructed = (
                student_scores[0:4].unsqueeze(1)
                + student_scores[4:16].unsqueeze(0)
            )
            error = reconstructed - ordinary
            projection_errors[suffix] = {
                "mean_squared_error": float(error.square().mean().item()),
                "max_absolute_error": float(error.abs().max().item()),
            }
        student.load_state_dict(student_state, strict=True)

    return {
        "kind": "same_body_additive_suit_rank_projection",
        "teacher_action_head": MODE_CONDITIONED_ACTION_HEAD,
        "student_action_head": FACTORIZED_ACTION_HEAD,
        "learned_scores": 23,
        "concrete_actions": 55,
        "hidden_size": int(student.hidden_size),
        "num_layers": int(student.num_layers),
        "copied_body_and_value_tensors": copied_tensors,
        "ordinary_card_projection": projection_errors,
        "special_rows_copied_exactly": {
            "jack": [48, 16],
            "joker": [49, 17],
            "pass": [54, 18],
            "announced_suits": [[50, 53], [19, 22]],
        },
    }


def initialize_hawk_residual(student, teacher) -> dict:
    """Copy a MinGRU teacher into the base and zero the Hawk output heads."""
    from temporal_puffer_v4 import HawkResidualTemporalPolicy

    if not isinstance(student, HawkResidualTemporalPolicy):
        raise TypeError("student must be a HawkResidualTemporalPolicy")
    if int(teacher.hidden_size) != student.base_hidden_size:
        raise ValueError("residual base hidden size must match the teacher")
    if int(teacher.num_layers) != student.base_num_layers:
        raise ValueError("residual base layer count must match the teacher")
    if bool(teacher.mode_conditioned_heads) != bool(
        student.mode_conditioned_heads
    ):
        raise ValueError("teacher and residual decoder modes must match")

    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    base_prefix = "puffer_policy.base."
    teacher_prefix = "puffer_policy."
    with torch.no_grad():
        for name, value in teacher_state.items():
            target = base_prefix + name[len(teacher_prefix) :]
            if target not in student_state:
                raise KeyError(f"residual base is missing teacher tensor {target}")
            student_state[target].copy_(value)

        adapter_decoder = "puffer_policy.adapter.decoder."
        zeroed = 0
        for name, value in student_state.items():
            if name.startswith(adapter_decoder):
                value.zero_()
                zeroed += value.numel()
        if not zeroed:
            raise RuntimeError("residual adapter has no decoder parameters")
        student.load_state_dict(student_state, strict=True)
    return {
        "kind": "function_preserving_hawk_residual",
        "teacher_hidden_size": int(teacher.hidden_size),
        "teacher_num_layers": int(teacher.num_layers),
        "adapter_hidden_size": student.adapter_hidden_size,
        "adapter_num_layers": student.adapter_num_layers,
        "adapter_recurrent_size": student.adapter_recurrent_size,
        "adapter_mlp_expansion": student.adapter_mlp_expansion,
        "zeroed_adapter_decoder_parameters": zeroed,
    }


def initialize_min_gru_residual(student, teacher) -> dict:
    """Embed a teacher exactly and zero a wider/deeper MinGRU branch."""
    from temporal_puffer_v4 import MinGRUResidualTemporalPolicy

    if not isinstance(student, MinGRUResidualTemporalPolicy):
        raise TypeError("student must be a MinGRUResidualTemporalPolicy")
    if int(teacher.hidden_size) != student.base_hidden_size:
        raise ValueError("residual base hidden size must match the teacher")
    if int(teacher.num_layers) != student.base_num_layers:
        raise ValueError("residual base layer count must match the teacher")
    if bool(teacher.mode_conditioned_heads) != bool(
        student.mode_conditioned_heads
    ):
        raise ValueError("teacher and residual decoder modes must match")
    if getattr(teacher, "action_head", None) != student.action_head:
        raise ValueError("teacher and residual action heads must match")
    if student.adapter_hidden_size <= student.base_hidden_size:
        raise ValueError("residual MinGRU branch must be wider than the base")
    if student.adapter_num_layers <= student.base_num_layers:
        raise ValueError("residual MinGRU branch must be deeper than the base")

    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    base_prefix = "puffer_policy.base."
    teacher_prefix = "puffer_policy."
    copied = 0
    zeroed = 0
    with torch.no_grad():
        for name, value in teacher_state.items():
            target = base_prefix + name[len(teacher_prefix) :]
            if target not in student_state:
                raise KeyError(f"residual base is missing teacher tensor {target}")
            student_state[target].copy_(value)
            copied += value.numel()

        adapter_decoder = "puffer_policy.adapter.decoder."
        for name, value in student_state.items():
            if name.startswith(adapter_decoder):
                value.zero_()
                zeroed += value.numel()
        if not zeroed:
            raise RuntimeError("residual MinGRU branch has no decoder parameters")
        student.load_state_dict(student_state, strict=True)
    return {
        "kind": "function_preserving_wider_deeper_min_gru_residual",
        "teacher_hidden_size": int(teacher.hidden_size),
        "teacher_num_layers": int(teacher.num_layers),
        "adapter_hidden_size": student.adapter_hidden_size,
        "adapter_num_layers": student.adapter_num_layers,
        "action_head": student.action_head,
        "copied_base_parameters": copied,
        "zeroed_adapter_decoder_parameters": zeroed,
    }


@torch.no_grad()
def evaluate_puffer_distillation(
    student,
    teacher,
    episodes,
    *,
    minibatch_trajectories: int = 128,
) -> dict:
    """Measure distribution, action, value, and strategic-pass fidelity."""
    if minibatch_trajectories < 1:
        raise ValueError("evaluation minibatch size must be positive")
    student.eval()
    teacher.eval()
    totals = {
        "samples": 0,
        "agreement": 0,
        "forward_kl": 0.0,
        "value_squared_error": 0.0,
        "max_legal_logit_error": 0.0,
        "strategic_opportunities": 0,
        "teacher_strategic_passes": 0,
        "student_strategic_passes": 0,
    }
    usable = [episode for episode in episodes if episode[0]]
    for start in range(0, len(usable), minibatch_trajectories):
        batch = usable[start : start + minibatch_trajectories]
        teacher_logits, teacher_values, masks = _forward_decisions(
            teacher, batch
        )
        student_logits, student_values, _ = _forward_decisions(student, batch)
        teacher_log = torch.log_softmax(teacher_logits, dim=-1)
        student_log = torch.log_softmax(student_logits, dim=-1)
        teacher_probabilities = teacher_log.exp()
        forward_kl = (
            teacher_probabilities * (teacher_log - student_log)
        ).sum(dim=-1)
        teacher_actions = teacher_logits.argmax(dim=-1)
        student_actions = student_logits.argmax(dim=-1)
        strategic = masks[:, PASS_ACTION] & masks[:, :PASS_ACTION].any(dim=-1)
        sample_count = int(teacher_actions.numel())
        opportunities = int(strategic.sum().item())
        finite_error = (student_logits - teacher_logits).abs().masked_select(
            masks
        )
        totals["samples"] += sample_count
        totals["agreement"] += int(
            (teacher_actions == student_actions).sum().item()
        )
        totals["forward_kl"] += float(forward_kl.sum().item())
        totals["value_squared_error"] += float(
            (student_values - teacher_values).square().sum().item()
        )
        totals["max_legal_logit_error"] = max(
            totals["max_legal_logit_error"],
            float(finite_error.max().item()),
        )
        totals["strategic_opportunities"] += opportunities
        totals["teacher_strategic_passes"] += int(
            ((teacher_actions == PASS_ACTION) & strategic).sum().item()
        )
        totals["student_strategic_passes"] += int(
            ((student_actions == PASS_ACTION) & strategic).sum().item()
        )
    samples = totals["samples"]
    if not samples:
        raise ValueError("distillation evaluation requires decisions")
    opportunities = totals["strategic_opportunities"]
    return {
        "samples": samples,
        "action_agreement": totals["agreement"] / samples,
        "forward_kl": totals["forward_kl"] / samples,
        "value_mse": totals["value_squared_error"] / samples,
        "max_legal_logit_error": totals["max_legal_logit_error"],
        "strategic_pass_opportunities": opportunities,
        "teacher_strategic_pass_rate": (
            totals["teacher_strategic_passes"] / opportunities
            if opportunities
            else None
        ),
        "student_strategic_pass_rate": (
            totals["student_strategic_passes"] / opportunities
            if opportunities
            else None
        ),
    }


def distill_puffer_policy(
    student,
    teacher,
    episodes: Sequence,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    minibatch_trajectories: int,
    seed: int,
    temperature: float = 1.5,
    value_coefficient: float = 0.25,
    strategic_pass_weight: float = 2.0,
    reverse_kl_coefficient: float = 0.25,
    hard_target_coefficient: float = 0.1,
    disagreement_weight: float = 1.0,
    length_bucketed: bool = True,
) -> list[dict]:
    """Match the teacher's masked distribution, value, and strategic passes."""
    if epochs < 1 or minibatch_trajectories < 1:
        raise ValueError("epochs and minibatch size must be positive")
    if (
        temperature <= 0
        or strategic_pass_weight < 1
        or reverse_kl_coefficient < 0
        or hard_target_coefficient < 0
        or disagreement_weight < 1
    ):
        raise ValueError("invalid distillation temperature/pass weight")
    device = next(student.parameters()).device
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    usable = [episode for episode in episodes if episode[0]]
    rng = random.Random(seed)
    logs = []
    for epoch in range(epochs):
        if length_bucketed:
            order = sorted(
                range(len(usable)),
                key=lambda index: sum(
                    len(step.decision.public_events) + 1
                    for step in usable[index][0]
                ),
            )
            minibatches = [
                order[start : start + minibatch_trajectories]
                for start in range(0, len(order), minibatch_trajectories)
            ]
            for indices in minibatches:
                rng.shuffle(indices)
            rng.shuffle(minibatches)
        else:
            order = list(range(len(usable)))
            rng.shuffle(order)
            minibatches = [
                order[start : start + minibatch_trajectories]
                for start in range(0, len(order), minibatch_trajectories)
            ]
        totals = {
            "loss": 0.0,
            "kl": 0.0,
            "value_mse": 0.0,
            "agreement": 0,
            "samples": 0,
            "strategic_samples": 0,
            "teacher_strategic_passes": 0,
            "student_strategic_passes": 0,
        }
        student.train()
        for indices in minibatches:
            batch = [usable[index] for index in indices]
            with torch.no_grad():
                teacher_logits, teacher_values, masks = _forward_decisions(
                    teacher, batch
                )
            student_logits, student_values, _ = _forward_decisions(
                student, batch
            )
            teacher_probabilities = torch.softmax(
                teacher_logits / temperature, dim=-1
            )
            teacher_log_probabilities = torch.log_softmax(
                teacher_logits / temperature, dim=-1
            )
            student_log_probabilities = torch.log_softmax(
                student_logits / temperature, dim=-1
            )
            student_probabilities = student_log_probabilities.exp()
            per_sample_kl = (
                teacher_probabilities
                * (teacher_log_probabilities - student_log_probabilities)
            ).sum(dim=-1) * (temperature * temperature)
            reverse_kl = (
                student_probabilities
                * (student_log_probabilities - teacher_log_probabilities)
            ).sum(dim=-1) * (temperature * temperature)
            teacher_actions = teacher_logits.argmax(dim=-1)
            hard_target = F.cross_entropy(
                student_logits,
                teacher_actions,
                reduction="none",
            )
            strategic = masks[:, PASS_ACTION] & masks[:, :PASS_ACTION].any(dim=-1)
            weights = torch.ones_like(per_sample_kl)
            weights[strategic] = strategic_pass_weight
            disagreement = (
                student_logits.detach().argmax(dim=-1) != teacher_actions
            )
            weights[disagreement] *= disagreement_weight
            kl = (
                (
                    per_sample_kl
                    + reverse_kl_coefficient * reverse_kl
                    + hard_target_coefficient * hard_target
                )
                * weights
            ).sum() / weights.sum()
            value_mse = (student_values - teacher_values).square().mean()
            loss = kl + value_coefficient * value_mse
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                student_actions = student_logits.argmax(dim=-1)
                samples = int(teacher_actions.numel())
                strategic_samples = int(strategic.sum().item())
                totals["loss"] += float(loss.detach()) * samples
                totals["kl"] += float(kl.detach()) * samples
                totals["value_mse"] += float(value_mse.detach()) * samples
                totals["agreement"] += int(
                    (teacher_actions == student_actions).sum().item()
                )
                totals["samples"] += samples
                totals["strategic_samples"] += strategic_samples
                totals["teacher_strategic_passes"] += int(
                    ((teacher_actions == PASS_ACTION) & strategic).sum().item()
                )
                totals["student_strategic_passes"] += int(
                    ((student_actions == PASS_ACTION) & strategic).sum().item()
                )
        samples = totals["samples"]
        strategic_samples = totals["strategic_samples"]
        logs.append(
            {
                "epoch": epoch + 1,
                "loss": totals["loss"] / samples,
                "kl": totals["kl"] / samples,
                "value_mse": totals["value_mse"] / samples,
                "action_agreement": totals["agreement"] / samples,
                "samples": samples,
                "strategic_pass_opportunities": strategic_samples,
                "teacher_strategic_pass_rate": (
                    totals["teacher_strategic_passes"] / strategic_samples
                    if strategic_samples else None
                ),
                "student_strategic_pass_rate": (
                    totals["student_strategic_passes"] / strategic_samples
                    if strategic_samples else None
                ),
            }
        )
    return logs
