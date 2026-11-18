"""Learn search trust from paired interventions and terminal win differences."""

import argparse
import json
from pathlib import Path

import numpy as np

from distill_public_rollouts import discard_history
from public_lookahead import EMPTY_WEIGHTS, _batch_values


GRID = np.array([0., 8., 16., 32., 50., 72., 100., 144., 220., 360.])


def label(path, posterior, samples, horizon):
    with np.load(path) as d:
        # These pairs prove that both endpoints were legal card plays or both
        # were suit declarations. A drawn-card/pay-or-pass retry is excluded.
        keep = (d["a"] != 54) & (d["b"] != 54)
        obs = d["obs"][keep].astype(np.float32)
        recent = discard_history(d["seen"][keep].astype(np.float32))
        tokens = np.zeros((len(obs), 100), dtype=np.float32)
        tokens[:, :94] = obs
        tokens[:, 99] = 1
        # The role label is a training target. The playing actor interpolates
        # learned trust weights using only its public estimate of this role.
        role = d["random"][keep].astype(np.float32)
        search = _batch_values(tokens, recent, samples, horizon, False,
                               EMPTY_WEIGHTS, posterior, .2 + .6 * role)
        idx = np.arange(len(obs))
        difference = search[idx, d["b"][keep]] - search[idx, d["a"][keep]]
        return np.column_stack((role, obs[:,92]-2, obs[:,93], d["gap"][keep],
                                difference, d["delta"][keep]))


def fit(data):
    role, players, phase, gap, difference, delta = data.T
    # The control half uses the original choice A. The measured delta is
    # win(B) - win(A), measured under the same deal and unchanged continuation.
    utility = (gap[:, None] + GRID * difference[:, None] > 0) * delta[:, None]
    rates = np.zeros((2, 3, 2))
    for mode in (0, 1):
        mode_rows = phase == mode
        pooled = np.mean(utility[mode_rows], axis=0)
        for n in range(3):
            size_rows = mode_rows & (players == n)
            group = (utility[size_rows].sum(0) + 1200 * pooled) / (size_rows.sum() + 1200)
            for r in (0, 1):
                cell = size_rows & (role == r)
                returns = (utility[cell].sum(0) + 1000 * group) / (cell.sum() + 1000)
                # Cross-configuration differences in observed game returns are
                # noisy. Smooth the empirical maximizer on a 0.3-point scale.
                weights = np.exp((returns - returns.max()) / .003)
                rates[r, n, mode] = np.expm1(np.average(np.log1p(GRID), weights=weights))
    return rates


def mean_gain(data, trust):
    return float(((data[:, 3] + trust * data[:, 4] > 0) * data[:, 5]).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--posterior", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=144)
    ap.add_argument("--horizon", type=int, default=72)
    args = ap.parse_args()
    paths = sorted(args.pairs.glob("pair_*.npz"))
    if len(paths) < 18:
        ap.error("at least 18 deal shards required")
    posterior = np.array(json.loads(args.posterior.read_text())["weights"], dtype=float).ravel()
    rows = [label(p, posterior, args.samples, args.horizon) for p in paths]
    train, validation = [np.vstack([r for i, r in enumerate(rows) if (i % 18 < 3) == v]) for v in (False, True)]
    trust = fit(train)
    weights = trust[tuple(validation[:, :3].astype(int).T)]
    report = dict(kind="pesten_empirical_search_trust", trust=trust.tolist(), grid=GRID.tolist(),
                  samples=args.samples, horizon=args.horizon, training_pairs=len(train),
                  validation_pairs=len(validation), training_shards=[p.name for p in paths],
                  payoff="single paired intervention; gain is per eligible two-move choice",
                  validation_gain_calibrated=mean_gain(validation, weights),
                  validation_gain_fixed72=mean_gain(validation, 72.))
    with args.output.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({k: v for k,v in report.items() if k != "training_shards"}))


if __name__ == "__main__":
    main()
