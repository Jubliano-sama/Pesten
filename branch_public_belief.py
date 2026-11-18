"""Public card-availability and opponent-hand features for the rollout student."""

import torch
from torch import nn

from public_lookahead import (
    BELIEF_WIDTH,
    DECLARATION,
    EVIDENCE,
    HISTORY_SIZE,
    LAST_CARD,
    LONG_RECENT,
    PILE_SIZE,
    _advance_tracking,
)


def initial_history(batch, device):
    return torch.zeros(1, batch, HISTORY_SIZE, device=device)


def advance_history(tokens, state):
    updated = _advance_tracking(
        tokens.detach().cpu().numpy(), state[0].detach().cpu().numpy(), True, True
    )
    return torch.from_numpy(updated).to(tokens)[None]


class TrackingCollector(nn.Module):
    tracks_public_beliefs = True

    def __init__(self, core):
        super().__init__()
        self.core = core

    def initial_state(self, batch, device):
        return (*self.core.initial_state(batch, device), initial_history(batch, device))

    def forward_events_eval(self, tokens, state):
        return (
            *self.core.forward_events_eval(tokens, state[:-1]),
            advance_history(tokens, state[-1]),
        )

    def forward_eval(self, tokens, state):
        logits, value, base = self.core.forward_eval(tokens, state[:-1])
        history = advance_history(tokens, state[-1])
        self.last_history = history[0]
        self.last_logits = self.core.last_logits
        self.last_counts = self.core.last_counts
        self.last_value = self.core.last_value
        return logits, value, (*base, history)


def belief_nodes(obs, history, frequencies):
    """Return 21 equivariant card features from visible discards and public actions."""
    batch = len(obs)
    n = obs[:, 92].long().clamp(2, 4)
    sizes = obs[:, 69:73].long()
    card = torch.arange(50, device=obs.device)
    suit = torch.where(card < 48, card // 12, 4)
    rank = torch.where(card < 48, card % 12 + 1, torch.where(card == 48, 13, 0))
    categories = torch.zeros_like(card)
    for index, special_rank in enumerate((1, 2, 7, 8, 13, 0), start=1):
        categories = torch.where(rank == special_rank, index, categories)
    ordinary = categories == 0
    multiplicities = torch.where(card < 48, 1, torch.where(card == 48, 4, 2))
    # Count only the publicly observed suffix that fits the inferred current pile.
    # Opening burns and individual hidden draws are unavailable.
    pile = history[:, LONG_RECENT:PILE_SIZE].long()
    recent = torch.arange(54, device=obs.device)[None] >= (54 - history[:, PILE_SIZE, None])
    known = ((pile > 0) & recent).to(obs.dtype)
    discarded = torch.zeros(batch, 50, device=obs.device)
    discarded.scatter_add_(1, (pile - 1).clamp(0, 49), known)
    available = (multiplicities - obs[:, :50] - discarded).clamp(min=0)
    availability = available[:, :48].reshape(-1, 4, 12)
    outputs = [
        availability,
        availability.sum(2, keepdim=True).expand_as(availability) / 12,
        availability.sum(1, keepdim=True).expand_as(availability) / 4,
    ]

    idx = torch.arange(batch, device=obs.device)
    for relative in (torch.ones_like(n), n - 1, 2 % n):
        clockwise = (relative * obs[:, 89].long()) % n
        opponent_size = sizes[idx, relative].clamp_min(1)
        evidence = EVIDENCE + clockwise * BELIEF_WIDTH
        last = history[idx, LAST_CARD + clockwise].long() - 1
        declared = history[idx, DECLARATION + clockwise].long() - 1
        last_suit = suit[last.clamp_min(0)]
        last_rank = rank[last.clamp_min(0)]
        log_count = history[idx[:, None], evidence[:, None] + suit[None]]
        rank_index = (4 + rank).clamp(max=16)
        log_count += history[idx[:, None], evidence[:, None] + rank_index[None]] * (card < 48)
        for role in range(2):
            factors = frequencies[role, n - 2, (opponent_size - 1).clamp(0, 8)]
            preference = factors[:, categories].clone()
            preference += factors[:, 7:8] * (suit == last_suit[:, None]) * (last >= 0)[:, None]
            preference += factors[:, 8:9] * (rank == last_rank[:, None]) * (last >= 0)[:, None]
            preference += factors[:, 9:10] * (suit == declared[:, None])
            weights = available * (preference + log_count).clamp(-6, 6).exp()
            probability = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
            regular = probability[:, :48].reshape(-1, 4, 12)
            ordinary_probability = (probability * ordinary)[:, :48].reshape(-1, 4, 12)
            by_suit = regular.sum(2, keepdim=True).expand_as(regular)
            by_rank = regular.sum(1, keepdim=True).expand_as(regular)
            exposure = (
                ordinary_probability.sum(2, keepdim=True)
                + ordinary_probability.sum(1, keepdim=True)
                - ordinary_probability
            )
            present = (relative != 0)[:, None, None]
            outputs.extend((by_suit * present, by_rank * present, exposure * present))
    return torch.stack(outputs, -1)
