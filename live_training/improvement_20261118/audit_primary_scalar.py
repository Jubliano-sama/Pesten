"""Replay the first 48 primary deals per cell in scalar game.Game and compare the fixed chunks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


FROZEN = "7b38179a6d89467a2e46616eadacc49b6d26778a3676d842b95cf69b11d05efb"
DEALS = 48
OUTCOMES = ("wins", "losses", "draws", "turns")
FINAL = Path(__file__).resolve().parent / "final"


def scalar_deals(bench, candidate, manifest):
    from benchmark_rlvr_pool import _candidate_score
    from multiplayer_eval_checkpoints import chunk_log
    from randomAgent import AgentRandom
    from temporal_puffer_agent import temporal_agent_factory

    bench.initialize(str(candidate), manifest["candidate_kind"])
    factories = {name: temporal_agent_factory(policy, deterministic=True)
                 for name, policy in bench.engine_eval._POLICIES.items()}
    tasks = bench.make_tasks(manifest["holdout"]["base_seed"], DEALS, 1)
    header = {"kind": "pesten_scalar_primary_audit_v1", "freeze_manifest_sha256": FROZEN,
              "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = FINAL / "primary_scalar_audit.blocks.jsonl"
    with chunk_log(path, header=header, tasks=tasks, resume=path.exists()) as (saved, append):
        for task in tasks:
            seed = task["seed"]
            if seed in saved:
                continue
            players = task["players"]
            opponent = factories["baseline"] if task["opponent"] == "self_anchor" else AgentRandom
            row = {"seed": seed, "players": players, "opponent": task["opponent"], "totals": {}}
            for name in bench.VARIANTS:
                score, wins, losses, draws, turns = _candidate_score(
                    factories[name], opponent, players=players, seed=seed)
                row[f"{name}_scores"] = [score]
                for label, value in (("win", wins), ("loss", losses), ("draw", draws)):
                    row[f"{name}_{label}_shares"] = [value / players]
                row["totals"][name] = dict(wins=wins, losses=losses, draws=draws, turns=turns)
            append(row)
            saved[seed] = row
            if len(saved) % DEALS == 0:
                print(f"Scalar audit: {len(saved)}/{len(tasks)} deals", flush=True)
    return saved


def compare(bench, manifest, scalar):
    from multiplayer_eval_checkpoints import encode
    base = manifest["holdout"]["base_seed"]
    wanted = {base + i * bench.STRIDE: cell for i, cell in enumerate(bench.CELLS)}
    recorded = {}
    with (FINAL / "primary.chunks.jsonl").open("rb") as handle:
        header = json.loads(handle.readline())
        if header["freeze_manifest_sha256"] != FROZEN or header["chunk_blocks"] != DEALS or header["seed"] != base:
            raise ValueError("unexpected primary evaluator checkpoint")
        for line in handle:
            if not line.endswith(b"\n"):
                continue
            record = json.loads(line)
            row = record["row"]
            seed = row["seed"]
            if seed not in wanted:
                continue
            if ((row["opponent"], row["players"]) != wanted[seed] or seed in recorded
                    or hashlib.sha256(encode(row)).hexdigest() != record["sha256"]):
                raise ValueError("inconsistent primary chunk")
            recorded[seed] = row
    if set(recorded) != set(wanted):
        print(f"Scalar audit complete, waiting for {len(wanted) - len(recorded)} primary chunks", flush=True)
        return
    cells, differences = [], []
    for start, (opponent, players) in wanted.items():
        batch = recorded[start]
        cell = {"opponent": opponent, "players": players, "seed_start": start}
        for variant in bench.VARIANTS:
            actual = [scalar[seed]["totals"][variant] for seed in range(start, start + DEALS)]
            totals = {k: sum(row[k] for row in actual) for k in OUTCOMES}
            official = {k: batch["totals"][variant].get(k, 0) for k in OUTCOMES}
            cell[variant] = {"scalar": totals, "primary": official}
            for i, row in enumerate(actual):
                reference = {k: round(batch[f"{variant}_{outcome}_shares"][i] * players)
                             for k, outcome in (("wins", "win"), ("draws", "draw"), ("losses", "loss"))}
                compared = {k: row[k] for k in reference}
                if compared != reference:
                    differences.append({"opponent": opponent, "players": players, "seed": start + i,
                                        "variant": variant, "scalar": compared, "primary": reference})
        cells.append(cell)
    result = {"manifest_sha256": FROZEN, "deals_per_cell": DEALS, "games_per_engine": 2 * DEALS * 18,
              "scope": "first full fixed chunk of each of six cells from the original primary bank",
              "per_deal_outcomes_match": not differences,
              "per_cell_summed_turns_match": all(c[v]["scalar"]["turns"] == c[v]["primary"]["turns"]
                                                for c in cells for v in bench.VARIANTS),
              "cells": cells, "per_deal_outcome_differences": differences,
              "scalar_deals_file": "primary_scalar_audit.blocks.jsonl"}
    with (FINAL / "primary_scalar_audit.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"Compared all {result['games_per_engine']} matches; {len(differences)} per-deal outcome differences.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    sys.path.insert(0, str(snapshot))
    os.chdir(snapshot)
    import benchmark_multiplayer_winrate as bench
    freeze_path = snapshot / "frozen_candidate.json"
    manifest = json.loads(freeze_path.read_text())
    candidate = snapshot / manifest["candidate"]
    if bench.verify_freeze(freeze_path, candidate, manifest["candidate_kind"]) != FROZEN:
        raise ValueError("audit requires the locked primary snapshot")
    rows = scalar_deals(bench, candidate, manifest)
    if bench.verify_freeze(freeze_path, candidate, manifest["candidate_kind"]) != FROZEN:
        raise ValueError("primary inputs changed during audit")
    compare(bench, manifest, rows)


if __name__ == "__main__":
    main()
