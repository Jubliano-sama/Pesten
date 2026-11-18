"""Weighted native policy inference with independent recurrent histories."""

from __future__ import annotations

import math
from typing import Sequence

from torch import nn


class NativeWeightedEnsembleCore(nn.Module):
    """Average unmasked logits while preserving every member's own state.

    Members use the native ``initial_state(batch_size, device)`` and
    ``forward_eval(tokens, state_tuple)`` interfaces. Suit-orbit members can
    be supplied directly; their packed states remain separate. The evaluator
    applies the legal-action mask after combining the member logits.
    """

    def __init__(self, members: Sequence[nn.Module], weights: Sequence[float]):
        super().__init__()
        members = tuple(members)
        weights = tuple(float(weight) for weight in weights)
        if not members or len(members) != len(weights):
            raise ValueError("provide one weight for every member of a nonempty ensemble")
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("ensemble weights must be finite and nonnegative")
        scale = max(weights)
        if scale == 0:
            raise ValueError("ensemble weights must have a positive sum")
        total = sum(weight / scale for weight in weights)
        # Python scalars work with CPU and CUDA members without extra device
        # buffers. Scaling first also avoids overflowing large finite weights.
        self.weights = tuple((weight / scale) / total for weight in weights)
        self.members = nn.ModuleList(members)
        self._state_sizes: tuple[int, ...] | None = None

    def initial_state(self, batch_size: int, device):
        states = tuple(member.initial_state(batch_size, device) for member in self.members)
        self._state_sizes = tuple(len(state) for state in states)
        return tuple(part for state in states for part in state)

    def _split_states(self, states):
        if self._state_sizes is None:
            raise RuntimeError("call initial_state before advancing ensemble histories")
        if len(states) != sum(self._state_sizes):
            raise ValueError("ensemble state tuple does not match its members")
        offset = 0
        for size in self._state_sizes:
            yield states[offset : offset + size]
            offset += size

    def forward_eval(self, tokens, states):
        combined_logits = combined_values = None
        next_states = []
        for member, weight, state in zip(self.members, self.weights, self._split_states(states)):
            logits, values, next_state = member.forward_eval(tokens, state)
            if combined_logits is None:
                combined_logits, combined_values = logits * weight, values * weight
            else:
                combined_logits = combined_logits + logits * weight
                combined_values = combined_values + values * weight
            next_states.extend(next_state)
        return combined_logits, combined_values, tuple(next_states)

    def forward_events_eval(self, tokens, states):
        next_states = []
        for member, state in zip(self.members, self._split_states(states)):
            if hasattr(member, "forward_events_eval"):
                next_state = member.forward_events_eval(tokens, state)
            else:
                _, _, next_state = member.forward_eval(tokens, state)
            next_states.extend(next_state)
        return tuple(next_states)
