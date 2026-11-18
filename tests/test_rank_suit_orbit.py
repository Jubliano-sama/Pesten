from collections import Counter

import pytest
import torch

pytest.importorskip("pufferlib")

import card
from rank_suit_orbit import ORDINARY, RankSuitOrbitCore, rank_maps
from temporal_observation import EventKind, PublicEventToken
from temporal_puffer_v4 import NativePufferTemporalPolicy


def test_relabelings_preserve_compatibility_effects_and_balanced_rank_coverage():
    core = RankSuitOrbitCore(NativePufferTemporalPolicy(hidden_size=8).puffer_policy)
    for rank in ORDINARY:
        assert Counter(mapping[rank] for mapping in rank_maps()) == dict.fromkeys(ORDINARY, 3)
    for view, (suits, ranks) in enumerate(zip(core.suit_permutations, rank_maps(), strict=True)):
        remap = core.card_maps[view].tolist()
        for original_id, mapped_id in enumerate(remap):
            original = card.Card(original_id)
            mapped = card.Card(mapped_id)
            assert mapped.truenumber == ranks[original.truenumber]
            if original.truenumber not in ORDINARY:
                assert mapped.truenumber == original.truenumber
            for s, new_s in enumerate((*suits, 4)):
                for top_rank in range(14):
                    assert original.compatible(card.sorts[s], top_rank) == mapped.compatible(
                        card.sorts[new_s], ranks[top_rank])


def test_rank_and_suit_maps_move_hand_events_top_rank_and_logits_consistently():
    policy = NativePufferTemporalPolicy(hidden_size=8)
    core = RankSuitOrbitCore(policy.puffer_policy)
    decision = torch.zeros(100)
    decision[99] = 1
    decision[12 + 8] = 2  # two nines of the second suit
    decision[55 + 9] = 1
    decision[50 + 1] = 1
    decision[82 + 1] = 7
    event = PublicEventToken(EventKind.PLAY, 1, 20, 1, 1)
    tokens = torch.vstack((decision, torch.tensor(policy._event_features_from_events([event])[0])))
    expanded = core.relabel_tokens(tokens).reshape(2, 24, 100)
    for view, (suits, ranks) in enumerate(zip(core.suit_permutations, rank_maps(), strict=True)):
        mapped = suits[1] * 12 + ranks[9] - 1
        assert expanded[0, view, mapped] == 2
        assert expanded[0, view, 55 + ranks[9]] == 1
        assert expanded[0, view, 50 + suits[1]] == 1
        assert expanded[0, view, 82 + suits[1]] == 7
        assert round(float(expanded[1, view, 96]) * 50) == 1 + mapped
        assert round(float(expanded[1, view, 97]) * 5) == 1 + suits[1]
        assert core.action_indices[view, 20] == mapped
        assert core.action_indices[view, 51] == 50 + suits[1]
        assert core.action_indices[view, 54] == 54
