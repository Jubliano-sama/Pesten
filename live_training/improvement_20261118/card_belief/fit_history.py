"""Refit the compact conditional card model on exact public-history sampling odds."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from branch_public_belief import TrackingCollector
from public_card_belief import FEATURES
from train_card_belief import fit_log_odds, stats
from train_outcome_residual import collection_policy
from validate_histories import score_games


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunks", type=int, default=54)
    p.add_argument("--games", type=int, default=192)
    p.add_argument("--seed", type=int, default=1656200000)
    p.add_argument("--orbit", type=int, default=4)
    p.add_argument("--iterations", type=int, default=16)
    args = p.parse_args()
    torch.set_num_threads(1)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    old_model = np.zeros((2, 3, 5, FEATURES))
    prior = np.asarray(json.loads(args.prior.read_text())["weights"])
    model = collection_policy(args.orbit)
    model.puffer_policy = TrackingCollector(model.puffer_policy)
    groups = [[], []]
    shards = []
    for chunk in range(args.chunks):
        n = 2 + chunk % 3
        seed = args.seed + chunk * args.games
        valid = chunk % 12 < 3
        examples = []
        score_games(n, args.games, seed, model, prior, old_model, examples)
        batch = tuple(np.concatenate(x) for x in zip(*examples))
        groups[int(valid)].append(batch)
        shards.append(
            dict(
                seed=seed,
                games=args.games,
                players=n,
                validation=valid,
                hands=len(batch[-1]),
            )
        )
        if chunk % 6 == 5:
            print("collected", chunk + 1, "/", args.chunks, flush=True)
    train, test = [tuple(np.concatenate(x) for x in zip(*pieces)) for pieces in groups]
    coeff = fit_log_odds(train, args.iterations)
    _, _, losses, counts = stats(*test, coeff, False)
    rows = []
    for role in (0, 1):
        for n in (2, 3, 4):
            group = slice((role * 3 + n - 2) * 5, (role * 3 + n - 1) * 5)
            a, b, c = losses[group].sum(0)
            rows.append(
                dict(
                    role=("teacher", "random")[role],
                    players=n,
                    opponent_hands=int(counts[group].sum()),
                    cards=int(c),
                    baseline=float(a / c),
                    fitted=float(b / c),
                )
            )
    a, b, c = losses.sum(0)
    report = dict(
        seed=args.seed,
        orbit=args.orbit,
        iterations=args.iterations,
        shards=shards,
        train_opponent_hands=len(train[-1]),
        validation_opponent_hands=len(test[-1]),
        validation_cards=int(c),
        baseline=float(a / c),
        fitted=float(b / c),
        cells=rows,
        baseline_prior_sha256=hashlib.sha256(args.prior.read_bytes()).hexdigest(),
    )
    np.save(out / "card_odds.npy", coeff.reshape(2, 3, 5, -1))
    (out / "fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "shards"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
