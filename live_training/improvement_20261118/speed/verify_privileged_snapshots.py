"""Print reproducible per-field digests of real 2-, 3-, and 4-player rollouts."""
import argparse
from dataclasses import fields
import hashlib
import json

import numpy as np
import torch

from benchmark import ACTOR
from privileged_critic import PrivilegedSnapshot
from temporal_batched_speed import collect_league_batch_batched
from temporal_puffer_v4 import load_temporal_policy_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traced", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy, _ = load_temporal_policy_checkpoint(ACTOR)
    if args.traced:
        from vector_puffer_rollout import _trace_worker_min_gru
        _trace_worker_min_gru(policy, shared_float_weights=True)
    for players in (2, 3, 4):
        hashes = {field.name: hashlib.sha256() for field in fields(PrivilegedSnapshot)}
        actor = hashlib.sha256()
        with torch.inference_mode():
            episodes, _ = collect_league_batch_batched(
                policy, None, [], games=48, seed=552000200 + players * 1000,
                schedule_offset=0, schedule=("current_selfplay", "canonical_random"),
                player_count=players, max_turns=1000, capture_privileged_snapshots=True,
            )
        for trajectory, outcome in episodes:
            for step in trajectory:
                actor.update(str((step.action, step.old_log_prob, step.value)).encode())
                for name, hasher in hashes.items():
                    hasher.update(np.asarray(getattr(step.privileged_snapshot, name)).tobytes())
            actor.update(str(outcome).encode())
        print(json.dumps({
            "players": players, "games": 48, "steps": sum(len(trajectory) for trajectory, _ in episodes),
            "actor": actor.hexdigest(), "fields": {key: value.hexdigest() for key, value in hashes.items()},
        }))


if __name__ == "__main__":
    main()
