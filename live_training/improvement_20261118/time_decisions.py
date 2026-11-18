"""Time both policies on the same public decision histories from scalar Game matches."""

import argparse
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

import benchmark_multiplayer_winrate as bench
import benchmark_rlvr_pool_batched as evaluator
from game import Game
from randomAgent import AgentRandom
from temporal_puffer_agent import temporal_agent_factory


def collect(factory, blocks, seed):
    episodes = []
    for index, (opponent, players) in enumerate(bench.CELLS):
        opponents = factory if opponent == "self_anchor" else AgentRandom
        for offset in range(blocks):
            deal = seed + index * bench.STRIDE + offset
            for seat in range(players):
                np.random.seed(deal)
                random.seed(deal)
                factories = [opponents] * players
                factories[seat] = factory
                game = Game(factories)
                game.auto_simulate()
                episodes.append((opponent, players, game.players[seat].safe_history))
    return episodes


@torch.inference_mode()
def replay(policy, episodes):
    elapsed, actions = [], []
    for _, _, decisions in episodes:
        state = policy.initial_state()
        for decision in decisions:
            start = time.perf_counter_ns()
            result = policy.step(decision, state)
            action = result.logits.argmax().item()
            elapsed.append((time.perf_counter_ns() - start) / 1e6)
            actions.append(int(action))
            state = result.state
    return elapsed, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=("native", *bench.CANDIDATE_MODULES))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1663209400)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    if args.repeats < 1 or args.blocks < 1 or args.output.exists():
        parser.error("use positive repeats and blocks, and a fresh output")
    bench.initialize(args.candidate, args.kind)
    policies = evaluator._POLICIES
    baseline = temporal_agent_factory(policies["baseline"], deterministic=True)
    episodes = collect(baseline, args.blocks, args.seed)
    for policy in policies.values():
        replay(policy, episodes[:2])
    timings = {variant: [] for variant in bench.VARIANTS}
    reference_actions = {}
    for i in range(args.repeats):
        for key in (bench.VARIANTS if i % 2 == 0 else reversed(bench.VARIANTS)):
            elapsed, actions = replay(policies[key], episodes)
            if key in reference_actions and actions != reference_actions[key]:
                raise AssertionError("actions changed between repeated timings")
            reference_actions[key] = actions
            timings[key].append(elapsed)
    report = {
        "candidate_path": str(args.candidate), "candidate_sha256": bench.file_sha256(args.candidate),
        "seed": args.seed, "stride": bench.STRIDE, "blocks_per_cell": args.blocks,
        "games": len(episodes), "decisions": len(reference_actions["baseline"]), "repeats": args.repeats,
        "action_changes": sum(a != b for a, b in zip(*reference_actions.values(), strict=True)),
        "protocol": "Single CPU thread, batch size one. Same public teacher-played scalar Game histories "
                    "across six cells, all seat rotations. Time policy.step including public-event processing, "
                    "legal mask and argmax. Exclude dealing, state initialization, model load and compilation.",
    }
    for key, values in timings.items():
        v = np.asarray(values)
        report[key] = {"mean_ms": v.mean().item(), "median_ms": np.median(v).item(),
                       "p95_ms": np.quantile(v, .95).item(), "repeat_mean_ms": v.mean(1).tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as writer:
        json.dump(report, writer, indent=2)
        writer.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
