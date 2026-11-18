"""Summarize matched, all-seat raw-win differences in the local search screens."""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
BANKS = {
    "728M": ("opponent_prior*.jsonl", 128, ("base", "128_72_72_0")),
    "733M": ("ablate*.jsonl", 120, ("base", "ab_base")),
    "735M": ("public_pile*.jsonl", 192, ("base", "pile_128", "pile_128_rollout")),
    "735M-first64": ("public_pile*.jsonl", 64, ("base", "pile_128_rollout")),
}


def load_deals(pattern):
    deals = defaultdict(dict)
    for path in sorted(DIRECTORY.glob(pattern)):
        for row in map(json.loads, path.read_text().splitlines()):
            profiles = {f"t{j}": Path(p).stem for j, p in enumerate(row["trusts"])}
            profiles["base"] = "base"
            for column in row["totals"]:
                name = profiles[column]
                shares = row[column + "_win_shares"]
                if len(shares) != row["blocks"]:
                    raise ValueError(f"incomplete chunk in {path}")
                for i, fraction in enumerate(shares):
                    record = deals[row["opponent"], row["players"], row["seed"] + i]
                    if name in record:
                        raise ValueError(f"duplicate deal block for {name} in {path}")
                    record[name] = fraction
    return deals


def paired(deals, name, reference, requested):
    samples = defaultdict(list)
    for (opp, num, _), record in deals.items():
        if name in record and reference in record:
            samples[opp, num].append((record[name], record[reference]))
    cells = {}
    variances = []
    for (opp, num), pair in sorted(samples.items()):
        difference = [100 * (left - right) for left, right in pair]
        cells[f"{opp}_{num}p"] = {
            "deal_blocks": len(pair),
            "candidate_win_pct": 100 * statistics.mean(a for a, _ in pair),
            "reference_win_pct": 100 * statistics.mean(b for _, b in pair),
            "difference_pp": statistics.mean(difference),
        }
        variances.append(statistics.variance(difference) / len(pair))
    full = len(cells) == 6 and all(
        x["deal_blocks"] == requested for x in cells.values()
    )
    result = {"complete": full, "cells": cells}
    if full:
        mean = statistics.mean(x["difference_pp"] for x in cells.values())
        radius = 1.96 * math.sqrt(sum(variances)) / 6
        result.update(
            macro_difference_pp=mean, paired_95_ci_pp=[mean - radius, mean + radius]
        )
    return result


def summarize():
    report = {}
    for label, (pattern, requested, comparators) in BANKS.items():
        data = load_deals(pattern)
        if label == "735M-first64":
            # Every cell starts on a separate multiple of 100,000 on this bank.
            data = {
                key: record
                for key, record in data.items()
                if key[2] % 100_000 < requested
            }
        profiles = sorted(
            {name for record in data.values() for name in record} - {"base"}
        )
        report[label] = {
            reference: {
                p: paired(data, p, reference, requested)
                for p in profiles
                if p != reference
            }
            for reference in comparators
        }
    return report


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
