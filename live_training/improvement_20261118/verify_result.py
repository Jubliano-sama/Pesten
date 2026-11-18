"""Verify a multiplayer win-rate report from its raw seat-rotated deal counts."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np


OPPONENTS = ("self_anchor", "canonical_random")
PLAYERS = (2, 3, 4)
CELLS = [(opponent, n) for opponent in OPPONENTS for n in PLAYERS]
ARMS = ("baseline", "candidate")
Z95 = NormalDist().inv_cdf(.975)


def check(label, actual, expected):
    """Reject discrepancies in both scalars and vectors."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=1e-12):
        raise ValueError(f"{label} disagrees with raw deal counts")


def index_cells(rows):
    indexed = {(row["opponent"], row["players"]): row for row in rows}
    if len(rows) != len(CELLS) or set(indexed) != set(CELLS):
        raise ValueError("expected six distinct opponent/player cells")
    return indexed


def percentile(values):
    return np.quantile(values, [.025, .975]).tolist()


def verify(report, resamples, seed):
    protocol = report["protocol"]
    blocks = protocol["blocks_per_cell"]
    if blocks < 2 or protocol["seed_stride"] < blocks:
        raise ValueError("deal banks must be disjoint and support a variance estimate")
    raw = index_cells(report["raw_blocks"])
    recorded = index_cells(report["cells"])
    rng = np.random.default_rng(seed)
    rates, variances, bootstrap, cells = [], [], [], []
    games_per_policy = 0

    def summarize(indices, claimed):
        mean = np.mean([rates[i] for i in indices], axis=0)
        delta = float(mean[1] - mean[0])
        se = math.sqrt(sum(variances[i] for i in indices)) / len(indices)
        normal = [delta - Z95 * se, delta + Z95 * se]
        for name, actual in (("delta", delta), ("standard_error", se), ("paired_ci95", normal)):
            check(name, actual, claimed[name])
        sampled = np.mean([bootstrap[i] for i in indices], axis=0)
        summary = {"baseline": float(mean[0]), "candidate": float(mean[1]), "delta": delta,
                   "standard_error": se, "paired_ci95": normal,
                   "bootstrap_delta_ci95": percentile(sampled[:, 1] - sampled[:, 0])}
        if len(indices) > 1:
            for arm in ARMS:
                check(arm, summary[arm], claimed[arm])
        return summary

    for i, (opponent, n) in enumerate(CELLS):
        row = raw[opponent, n]
        cell = recorded[opponent, n]
        check("deal seed", row["seed_start"], protocol["base_seed"] + i * protocol["seed_stride"])
        check("deals", cell["blocks"], blocks)
        games = blocks * n
        games_per_policy += games
        check("games", cell["games_per_policy"], games)
        pair = []
        counts = {}
        for arm in ARMS:
            wins, draws = (np.asarray(row[arm][field]) for field in ("wins", "draws"))
            for vector in (wins, draws):
                if (vector.shape != (blocks,) or vector.dtype.kind not in "iu"
                        or (vector < 0).any() or (vector > n).any()):
                    raise ValueError(f"invalid {arm} counts in {opponent}/{n}p")
            losses = n - wins - draws
            if (losses < 0).any():
                raise ValueError(f"seat count exceeded in {opponent}/{n}p")
            counts[arm] = {key: int(vector.sum()) for key, vector in
                           (("wins", wins), ("losses", losses), ("draws", draws))}
            for key, number in counts[arm].items():
                check(arm + " " + key, number, cell[arm][key])
            check(arm + " win rate", counts[arm]["wins"] / games, cell[arm]["win_rate"])
            pair.append(wins)
        paired_counts = np.column_stack(pair)
        differences = (pair[1] - pair[0]) / n
        check("per-deal win delta", differences, row["paired_win_rate_deltas"])
        rates.append(paired_counts.sum(axis=0) / games)
        variances.append(float(differences.var(ddof=1) / blocks))

        # Resample whole deal pairs. Grouping identical (old wins, new wins)
        # tuples is exactly equivalent to drawing individual blocks with replacement.
        outcomes, frequency = np.unique(paired_counts, axis=0, return_counts=True)
        draws = rng.multinomial(blocks, frequency / blocks, size=resamples)
        bootstrap.append(draws @ outcomes / games)
        cells.append({"opponent": opponent, "players": n, "games_per_policy": games,
                      "counts": counts, **summarize([i], cell)})

    check("total games", games_per_policy, protocol["games_per_policy"])
    opponents = {key: summarize([i for i, cell in enumerate(CELLS) if cell[0] == key],
                                report["by_opponent"][key]) for key in OPPONENTS}
    players = {key: summarize([i for i, cell in enumerate(CELLS) if cell[1] == key],
                              report["by_players"][str(key)]) for key in PLAYERS}
    macro = summarize(range(len(CELLS)), report["macro"])
    pooled = np.mean(bootstrap, axis=0)
    macro.update({arm + "_bootstrap_ci95": percentile(pooled[:, i]) for i, arm in enumerate(ARMS)})
    return {"games_per_policy": games_per_policy, "deal_blocks_per_cell": blocks,
            "resamples": resamples, "bootstrap_seed": seed, "macro": macro,
            "by_opponent": opponents, "by_players": players, "cells": cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=243_111_826)
    args = parser.parse_args()
    if args.resamples < 10_000 or args.seed < 0:
        parser.error("use at least 10,000 resamples and a nonnegative seed")
    data = args.result.read_bytes()
    verified = verify(json.loads(data), args.resamples, args.seed)
    print(json.dumps({"result_sha256": hashlib.sha256(data).hexdigest(), **verified},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
