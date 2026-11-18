"""Fit per-card log odds from recorded games and evaluate on disjoint game shards."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from public_card_belief import (
    CATEGORY,
    FEATURES,
    FAMILIES,
    card_features,
    njit,
    size_bin,
)


@njit(cache=True)
def encode(hands, sizes, directions, suits, ranks, penalties, roles, prior):
    samples, n = sizes.shape
    m = samples * (n - 1)
    src = np.zeros((m, 50), np.float32)
    target = np.zeros((m, 50), np.uint8)
    features = np.zeros((m, 50, FAMILIES), np.uint8)
    groups = np.zeros(m, np.uint8)
    for i in range(samples):
        own = np.zeros(50, np.uint8)
        for j in range(sizes[i, 0]):
            own[hands[i, 0, j]] += 1
        remain = np.ones(50, np.int16)
        remain[48], remain[49] = 4, 2
        remain -= own
        if 1 <= ranks[i] <= 12 and 0 <= suits[i] < 4:
            remain[12 * suits[i] + ranks[i] - 1] -= 1
        elif ranks[i] == 13:
            remain[48] -= 1
        elif ranks[i] == 0:
            remain[49] -= 1
        ordered = sizes[i, (np.arange(n) * directions[i]) % n]
        for rel in range(1, n):
            a = i * (n - 1) + rel - 1
            absolute = (rel * directions[i]) % n
            s = int(ordered[rel])
            role = int(roles[i])
            groups[a] = (role * 3 + n - 2) * 5 + size_bin(s)
            for j in range(s):
                target[a, hands[i, absolute, j]] += 1
            for c in range(50):
                src[a, c] = max(0, remain[c]) * np.exp(
                    prior[role, n - 2, min(s, 9) - 1, CATEGORY[c]]
                )
                if target[a, c] and not src[a, c]:
                    raise ValueError(
                        "hand label conflicts with own hand and visible top card"
                    )
            features[a] = card_features(
                own, ordered, rel, suits[i], ranks[i], penalties[i]
            )
    return src, target, features, groups


@njit(cache=True)
def stats(prior, target, features, groups, params, fit):
    actual = np.zeros_like(params)
    expected = np.zeros_like(params)
    loss = np.zeros((30, 3))
    counts = np.zeros(30, np.int64)
    pred = np.empty(50)
    for i in range(len(groups)):
        g = groups[i]
        normal = 0.0
        normal0 = 0.0
        size = 0
        for c in range(50):
            size += target[i, c]
            total = 0.0
            for k in range(FAMILIES):
                total += params[g, features[i, c, k]]
            pred[c] = prior[i, c] * np.exp(min(2.5, max(-2.5, total)))
            normal += pred[c]
            normal0 += prior[i, c]
        counts[g] += 1
        for c in range(50):
            if pred[c] <= 0:
                continue
            p = pred[c] / normal
            if target[i, c]:
                loss[g, 0] -= target[i, c] * np.log(prior[i, c] / normal0)
                loss[g, 1] -= target[i, c] * np.log(p)
                loss[g, 2] += target[i, c]
            if fit:
                for k in range(FAMILIES):
                    f = features[i, c, k]
                    expected[g, f] += size * p
                    actual[g, f] += target[i, c]
    return actual, expected, loss, counts


def prepare(paths, prior):
    packed = [[], []]
    provenance = []
    for path in paths:
        shard = int(path.stem.split("_")[-1])
        validation = shard % 18 < 3
        with np.load(path) as data:
            rows = np.flatnonzero(data["seat"] == 0)
            episode = data["episode"][rows]
            role = data["random_others"][episode]
            batch = encode(
                data["hands"][rows],
                data["sizes"][rows],
                data["direction"][rows],
                data["suit"][rows],
                data["rank"][rows],
                data["penalty"][rows],
                role,
                prior,
            )
            packed[int(validation)].append(batch)
            deals = data["deals"]
            provenance.append(
                dict(
                    name=path.name,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    validation=bool(validation),
                    games=len(deals),
                    seed=int(deals[0]),
                    positions=len(rows),
                    orbit=int(data["orbit"]),
                )
            )
    return [
        tuple(np.concatenate(x) for x in zip(*pieces)) for pieces in packed
    ], provenance


def fit_log_odds(train, iterations):
    coeff = np.zeros((30, FEATURES))
    prior_strength = 25.0
    for _ in range(iterations):
        actual, expected, _, _ = stats(*train, coeff, True)
        # Damped iterative scaling. Pseudocounts shrink sparse bins towards the frequency prior.
        correction = np.log((actual + prior_strength) / (expected + prior_strength))
        coeff[:, 1:] += 0.38 * (correction[:, 1:] - 0.012 * coeff[:, 1:])
        coeff[:, 0] = 0.0
    return coeff


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shards", type=int, default=102)
    p.add_argument("--iterations", type=int, default=16)
    args = p.parse_args()
    out = args.output
    out.mkdir(exist_ok=False, parents=True)
    prior = np.asarray(json.loads(args.prior.read_text())["weights"])
    if args.shards < 6 or args.iterations < 1:
        p.error("at least six shards and one iteration are required")
    paths = [args.data / f"shard_{index:04}.npz" for index in range(args.shards)]
    (train, test), records = prepare(paths, prior)
    coeff = fit_log_odds(train, args.iterations)
    result = stats(*test, coeff, False)
    loss = result[2].reshape(2, 3, 5, 3)
    counts = result[3].reshape(2, 3, 5)
    rows = []
    for role in range(2):
        for n in range(2, 5):
            a, b, c = loss[role, n - 2].sum(0)
            rows.append(
                dict(
                    role=("teacher", "random")[role],
                    players=n,
                    opponent_hands=int(counts[role, n - 2].sum()),
                    cards=int(c),
                    baseline=float(a / c),
                    fitted=float(b / c),
                )
            )
    a, b, c = loss.sum((0, 1, 2))
    report = dict(
        shards=records,
        iterations=args.iterations,
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
