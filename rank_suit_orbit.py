"""Average a recurrent Pesten actor over suit and ordinary-rank relabelings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import torch

from strongest_policy_search import NativeSuitOrbitCore
from temporal_puffer_v4 import load_temporal_policy_checkpoint


# All other ranks have an effect or an exceptional finishing rule.
ORDINARY = (3, 4, 5, 6, 9, 10, 11, 12)


def rank_maps():
    """Three Latin squares give every ordinary rank each label three times."""
    rng = random.Random(4387)
    maps = []
    for _ in range(3):
        inputs = rng.sample(ORDINARY, len(ORDINARY))
        outputs = rng.sample(ORDINARY, len(ORDINARY))
        for shift in range(len(ORDINARY)):
            permutation = list(range(14))
            for position, rank in enumerate(inputs):
                permutation[rank] = outputs[(position + shift) % len(ORDINARY)]
            maps.append(tuple(permutation))
    rng.shuffle(maps)
    return maps


class RankSuitOrbitCore(NativeSuitOrbitCore):
    """Pair the 24 suit permutations with 24 balanced ordinary-rank permutations."""

    def __init__(self, core):
        super().__init__(core, alpha=1, orbit_size=24)
        for view, (suits, ranks) in enumerate(zip(self.suit_permutations, rank_maps(), strict=True)):
            cards = [suits[card // 12] * 12 + ranks[1 + card % 12] - 1
                     for card in range(48)] + [48, 49]
            # Logits and event IDs go from original to transformed coordinates.
            # Reading hand counts and rank one-hots uses the inverse permutation.
            self.card_maps[view] = torch.tensor(cards)
            self.action_indices[view, :50] = self.card_maps[view]
            self.observation_indices[view, :50] = torch.argsort(self.card_maps[view])
            self.observation_indices[view, 55:69] = 55 + torch.argsort(torch.tensor(ranks))


def load_candidate(path, device="cpu"):
    """Load a JSON profile with a repository-relative native actor and its hash."""
    profile = json.loads(Path(path).read_text())
    if profile.get("kind") != "pesten_rank_suit24":
        raise ValueError("expected a pesten_rank_suit24 profile")
    checkpoint = Path(__file__).resolve().parent / profile["checkpoint"]
    with checkpoint.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != profile["sha256"]:
            raise ValueError("rank/suit actor checkpoint hash mismatch")
    policy, _ = load_temporal_policy_checkpoint(checkpoint, device=device)
    policy.puffer_policy = RankSuitOrbitCore(policy.puffer_policy)
    return policy.to(device).eval().requires_grad_(False)
