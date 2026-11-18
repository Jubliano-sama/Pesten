"""Serve the frozen public-information search on two- through four-player tables."""
import json
from pathlib import Path

import numpy as np
import torch

from production_temporal_policy import encode_event_token, file_sha256
from public_lookahead import (
    EMPTY_WEIGHTS, HISTORY_SIZE, _advance_tracking, _batch_values, _decision_features,
)


ROOT = Path(__file__).resolve().parent
PROFILE = ROOT / "live_training/improvement_20261118/card_belief/profile.json"
PROFILE_SHA256 = "03c8c111927fdfe6d3355fe47b6bc1d12e260ea31514622392e5de8aced08426"


def verified_profile():
    if file_sha256(PROFILE) != PROFILE_SHA256:
        raise ValueError("web search profile differs from the evaluated finalist")
    config = json.loads(PROFILE.read_bytes())
    for key in ("posterior", "rollout", "card_belief"):
        artifact = config[key]
        if file_sha256(ROOT / artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"web search {key} hash mismatch")
    return config


class WebSearch:
    """Immutable shared models. Each seat supplies its own public history."""

    def __init__(self, profile):
        self.profile = profile
        table = json.loads((ROOT / profile["posterior"]["path"]).read_bytes())["weights"]
        self.posterior = np.ascontiguousarray(table, dtype=np.float64).ravel()
        self.rollout = np.ascontiguousarray(
            np.load(ROOT / profile["rollout"]["path"], allow_pickle=False), dtype=np.float64)
        self.card_belief = np.ascontiguousarray(
            np.load(ROOT / profile["card_belief"]["path"], allow_pickle=False), dtype=np.float64)

    def score(self, decision, history, logits):
        if history is None:
            history = np.zeros((1, HISTORY_SIZE), dtype=np.float32)
        profile = self.profile
        for event in decision.public_events:
            token = np.asarray([encode_event_token(event)], dtype=np.float32)
            history = _advance_tracking(token, history, profile["infer_voids"], profile["track_pile"])
        token = np.asarray([_decision_features(decision)], dtype=np.float32)
        history = _advance_tracking(token, history, profile["infer_voids"], profile["track_pile"])
        delta = _batch_values(token, history, profile["samples"], profile["horizon"], profile["infer_voids"],
                              EMPTY_WEIGHTS, self.posterior, EMPTY_WEIGHTS, self.rollout, self.card_belief)
        # The serving image's older CPU PyTorch uses a different NumPy ABI.
        return logits + profile["trust"] * logits.new_tensor(delta.tolist()), history


if __name__ == "__main__":
    # Compile the exact serving kernels into the image under its runtime user.
    import numba
    from temporal_observation import DecisionInput, EventKind, PublicEventToken
    from temporal_observation import build_legal_actions, build_private_observation

    own = (0, 2, 15, 29)
    cards = dict(own_cards=own, current_suit="harten", current_rank=5,
                 penalty_amount=0, change_suit_mode=False)
    observation = build_private_observation(**cards, public_hand_sizes=(4, 3, 5),
                                            observer_index=0, current_player_index=0,
                                            direction=1, public_sorts_played=(0,) * 5)
    decision = DecisionInput(observation, build_legal_actions(**cards),
                             (PublicEventToken(EventKind.EPISODE_START),))
    logits, _ = WebSearch(verified_profile()).score(decision, None, torch.zeros(1, 55))
    if not torch.isfinite(logits).all() or logits.abs().max() == 0:
        raise RuntimeError("evaluated search did not produce a move preference")
    print(f"Search kernels ready with Numba {numba.__version__}")
