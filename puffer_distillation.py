"""Information-safe recurrent distillation from a Puffer teacher to a student."""

from __future__ import annotations

import random
from typing import Sequence

import torch

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
) -> list[dict]:
    """Match the teacher's masked distribution, value, and strategic passes."""
    if epochs < 1 or minibatch_trajectories < 1:
        raise ValueError("epochs and minibatch size must be positive")
    if temperature <= 0 or strategic_pass_weight < 1:
        raise ValueError("invalid distillation temperature/pass weight")
    device = next(student.parameters()).device
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    usable = [episode for episode in episodes if episode[0]]
    rng = random.Random(seed)
    logs = []
    for epoch in range(epochs):
        order = list(range(len(usable)))
        rng.shuffle(order)
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
        for start in range(0, len(order), minibatch_trajectories):
            batch = [
                usable[index]
                for index in order[start : start + minibatch_trajectories]
            ]
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
            per_sample_kl = (
                teacher_probabilities
                * (teacher_log_probabilities - student_log_probabilities)
            ).sum(dim=-1) * (temperature * temperature)
            strategic = masks[:, PASS_ACTION] & masks[:, :PASS_ACTION].any(dim=-1)
            weights = torch.ones_like(per_sample_kl)
            weights[strategic] = strategic_pass_weight
            kl = (per_sample_kl * weights).sum() / weights.sum()
            value_mse = (student_values - teacher_values).square().mean()
            loss = kl + value_coefficient * value_mse
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                teacher_actions = teacher_logits.argmax(dim=-1)
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
