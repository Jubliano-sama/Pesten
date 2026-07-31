"""Native PufferLib 4 Pesten rollout/training smoke benchmark."""

from __future__ import annotations

import json
import platform
from pathlib import Path
import time

import pufferlib
import torch

from temporal_puffer_v4 import (
    NativePufferTemporalPolicy,
    PUFFERLIB_REVISION,
)
from temporal_training import collect_smoke_episodes, ppo_update


def run_native_smoke(
    *,
    seed: int = 431,
    episode_count: int = 2,
    max_turns: int = 160,
) -> dict:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    policy = NativePufferTemporalPolicy(hidden_size=48, num_layers=1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    before = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
    }
    started = time.perf_counter()
    episodes, rollout = collect_smoke_episodes(
        policy,
        seed=seed,
        episode_count=episode_count,
        max_turns=max_turns,
    )
    metrics = ppo_update(policy, episodes, optimizer)
    elapsed = time.perf_counter() - started
    parameters_changed = any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in policy.named_parameters()
    )
    return {
        "schema_version": 1,
        "seed": seed,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "integration": {
            "native_pufferlib": True,
            "pufferlib_version": str(pufferlib.__version__),
            "official_git_revision": PUFFERLIB_REVISION,
            "policy_class": (
                f"{type(policy.puffer_policy).__module__}."
                f"{type(policy.puffer_policy).__name__}"
            ),
            "network_class": (
                f"{type(policy.puffer_policy.network).__module__}."
                f"{type(policy.puffer_policy.network).__name__}"
            ),
        },
        "model": {
            "hidden_size": policy.hidden_size,
            "num_layers": policy.num_layers,
            "actor_actions": 55,
            "critic_privileged": False,
        },
        "rollout": rollout,
        "training": {
            **metrics,
            "parameters_changed": parameters_changed,
        },
        "elapsed_seconds": round(elapsed, 6),
    }


if __name__ == "__main__":
    destination = Path(
        "benchmarks/puffer4_native_temporal_smoke_20260731.json"
    )
    result = run_native_smoke()
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
