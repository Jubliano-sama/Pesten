"""Rank completed candidate screens on identical deals, seats and incumbent outcomes."""

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np


CELLS = [(opponent, n) for opponent in ("self_anchor", "canonical_random") for n in (2, 3, 4)]


def paired_difference(left, right):
    deltas = left - right
    mean = float(deltas.mean())
    error = float(np.sqrt(deltas.var(axis=1, ddof=1).sum() / deltas.shape[1]) / len(CELLS))
    half = NormalDist().inv_cdf(.975) * error
    return {"delta": mean, "ci95": [mean - half, mean + half], "cells": deltas.mean(axis=1).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screens", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = None
    bank = None
    records = {}
    shares = {}
    for directory in args.screens:
        summary = json.loads((directory / "summary.json").read_text())
        rows = [json.loads(line) for line in (directory / "rows.jsonl").read_text().splitlines()]
        cells = [sorted((r for r in rows if (r["opponent"], r["players"]) == key), key=lambda r: r["seed"])
                 for key in CELLS]
        deals = np.array([[seed for r in group for seed in range(r["seed"], r["seed"] + len(r["baseline_win_shares"]))]
                          for group in cells], dtype=np.int64)
        if bank is None:
            bank = deals
        np.testing.assert_array_equal(bank, deals)
        if deals.shape != (len(CELLS), summary["blocks_per_cell"]):
            raise ValueError(f"incomplete screen in {directory}")
        for name in ("baseline", *summary["profiles"]):
            sample = np.array([[s for r in group for s in r[name + "_win_shares"]] for group in cells])
            if name == "baseline":
                if baseline is None:
                    baseline = sample
                np.testing.assert_array_equal(sample, baseline)
                continue
            if name in shares:
                raise ValueError(f"duplicate candidate {name}")
            shares[name] = sample
            comparison = paired_difference(sample, baseline)
            np.testing.assert_allclose(comparison["delta"], summary["results"][name]["macro"]["delta"], atol=1e-12)
            records[name] = {**summary["profiles"][name], "mean_win_rate": sample.mean().item(),
                             "versus_incumbent": comparison}
    ranking = sorted(shares, key=lambda name: -records[name]["mean_win_rate"])
    for name in ranking:
        records[name]["versus_leader"] = paired_difference(shares[name], shares[ranking[0]])
    summary = {"base_seed": int(bank[0, 0]), "blocks_per_cell": bank.shape[1], "cell_order": CELLS,
               "incumbent_mean_win_rate": baseline.mean().item(), "ranking": ranking, "candidates": records}
    with args.output.open("x") as writer:
        json.dump(summary, writer, indent=2)
        writer.write("\n")
    for name in ranking:
        comparison = records[name]["versus_incumbent"]
        print(f"{name:<25} {comparison['delta']*100:+.2f} pp")


if __name__ == "__main__":
    main()
