"""Compute the paired effect of the conditional hand model over the same search control."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmark_multiplayer_winrate import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tournament", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.tournament / "summary.json").read_text())
    rows = []
    for line in (args.tournament / "rows.jsonl").read_text().splitlines():
        source = json.loads(line)
        row = {key: source[key] for key in ("opponent", "players", "seed")}
        row["totals"] = {}
        for dest, player in [("baseline", "fixed"), ("candidate", "conditional")]:
            row["totals"][dest] = source["totals"][player]
            for key in ("win_shares", "loss_shares", "draw_shares"):
                row[dest + "_" + key] = source[player + "_" + key]
        rows.append(row)
    result = summarize(rows, seed=metadata["seed"], blocks=metadata["blocks_per_cell"])
    result.pop(
        "raw_blocks"
    )  # Per-deal vectors for both variants are already in the source JSONL.
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["macro"])


if __name__ == "__main__":
    main()
