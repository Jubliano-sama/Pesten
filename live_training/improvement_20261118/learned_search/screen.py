"""Evaluate several leaf-search configurations on one matched development bank."""

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch

import benchmark_rlvr_pool_batched as evaluator
from benchmark_multiplayer_winrate import make_tasks, orbit24, summarize
from public_lookahead import load_candidate
from strongest_policy_search import TEACHER


def write_comparisons(path, labels, seed, blocks):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    comparisons = {}
    for baseline, candidate in combinations(labels, 2):
        paired = []
        for row in rows:
            item = {field: row[field] for field in ("opponent", "players", "seed")}
            item["totals"] = {"baseline": row["totals"][baseline], "candidate": row["totals"][candidate]}
            for source, dest in ((baseline, "baseline"), (candidate, "candidate")):
                for outcome in ("win", "loss", "draw"):
                    item[f"{dest}_{outcome}_shares"] = row[f"{source}_{outcome}_shares"]
            paired.append(item)
        result = summarize(paired, seed=seed, blocks=blocks)
        comparisons[f"{candidate}_vs_{baseline}"] = {
            name: result[name] for name in ("macro", "by_opponent", "by_players", "cells")}
    dest = path.with_name(path.stem + "_comparisons.json")
    dest.write_text(json.dumps(comparisons, indent=2) + "\n")
    for pair, result in comparisons.items():
        macro = result["macro"]
        print(pair, round(macro["delta"] * 100, 3),
              [round(bound * 100, 3) for bound in macro["paired_ci95"]], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--blocks", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("profiles", nargs="+", type=Path)
    args = ap.parse_args()
    labels = ["base", *(profile.stem for profile in args.profiles)]
    if args.summarize_only:
        write_comparisons(args.output, labels, args.seed, args.blocks)
        return
    torch.set_num_threads(1)
    evaluator._initialize_multi_worker({"base": str(TEACHER)}, "ppo_actor_100run3.pth")
    policies = evaluator._POLICIES
    policies["base"] = orbit24(policies["base"])
    for profile in args.profiles:
        if profile.stem in policies:
            raise ValueError("duplicate candidate label")
        policies[profile.stem] = load_candidate(profile)
    tasks = make_tasks(args.seed, args.blocks, args.chunk)
    # Interleave cells within each pass through the bank.
    tasks.sort(key=lambda task: ((task["seed"] - args.seed) % 1_000_000, task["players"], task["opponent"]))
    with args.output.open("x") as f:
        for task in tasks:
            task.update(variants=tuple(policies), policy_names=tuple(policies), self_anchor_name="base")
            started = time.monotonic()
            result = evaluator._evaluate_task(task)
            result.update(seed=task["seed"], elapsed_seconds=time.monotonic()-started)
            f.write(json.dumps(result)+"\n")
            f.flush()
            print(result["opponent"], result["players"], result["seed"], round(result["elapsed_seconds"],1),
                  {name:row.get("wins",0) for name,row in result["totals"].items()}, flush=True)
    write_comparisons(args.output, labels, args.seed, args.blocks)


if __name__ == "__main__":
    main()
