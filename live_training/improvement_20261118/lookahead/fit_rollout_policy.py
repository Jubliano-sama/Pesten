"""Fit a small linear continuation policy to the frozen teacher's public action logits."""

from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from public_lookahead import _choose
from public_rollout_policy import F, C, choose_proxy, features, summarize, njit


@njit(cache=True)
def rows(obs, legal, logits):
    count = 0
    for i in range(len(obs)):
        if not obs[i, 93] and legal[i].sum() > 1:
            count += legal[i].sum() - 1
    arr = np.zeros((count, F * C), np.float32)
    target = np.zeros(count, np.float32)
    hand = np.empty(54, np.int8)
    x = np.empty(F, np.float64)
    offset = 0
    for i in range(len(obs)):
        if obs[i, 93] or legal[i].sum() < 2:
            continue
        h = 0
        for a in range(50):
            for j in range(int(obs[i, a])):
                hand[h] = a
                h += 1
        n = int(obs[i, 92])
        opp = int(obs[i, 70])
        suit = obs[i, 50:55].argmax()
        rank = obs[i, 55:69].argmax()
        pen = int(obs[i, 90])
        sizesuit, ordinarysuit, tempo, ranks, normal, wild = summarize(hand, h, n)
        best = 54
        for a in range(50):
            if legal[i, a] and logits[i, a] > logits[i, best]:
                best = a
        topcat = features(
            x,
            best,
            sizesuit,
            ordinarysuit,
            tempo,
            ranks,
            h,
            rank,
            suit,
            pen,
            opp,
            n,
            normal,
            wild,
        )
        best_features = x.copy()
        for a in range(55):
            if not legal[i, a] or a == best:
                continue
            cat = features(
                x,
                a,
                sizesuit,
                ordinarysuit,
                tempo,
                ranks,
                h,
                rank,
                suit,
                pen,
                opp,
                n,
                normal,
                wild,
            )
            arr[offset, topcat * F : (topcat + 1) * F] = best_features
            arr[offset, cat * F : (cat + 1) * F] -= x
            target[offset] = min(14.0, logits[i, best] - logits[i, a])
            offset += 1
    return arr, target


@njit(cache=True)
def validation_counts(obs, legal, logits, weights):
    totals = np.zeros((2, 3))
    hand = np.empty(54, np.int8)
    for i in range(len(obs)):
        if obs[i, 93] or legal[i, :50].sum() < 2:
            continue
        size = 0
        for card in range(50):
            for _ in range(int(obs[i, card])):
                hand[size] = card
                size += 1
        players = int(obs[i, 92])
        next_size = int(obs[i, 70])
        suit = obs[i, 50:55].argmax()
        rank = obs[i, 55:69].argmax()
        penalty = int(obs[i, 90])
        best = logits[i][legal[i]].max()
        for model in range(2):
            if model:
                index = choose_proxy(
                    hand, size, suit, rank, penalty, next_size, players, 1234, weights
                )
            else:
                index = _choose(
                    hand, size, suit, rank, penalty, next_size, players, False, 1234
                )
            action = hand[index] if index >= 0 else 54
            if not legal[i, action]:
                raise ValueError("continuation chose an illegal action")
            regret = best - logits[i, action]
            totals[model, 0] += 1
            totals[model, 1] += regret < 0.001
            totals[model, 2] += regret
    return totals


def validation(directory, weights):
    counts = np.zeros((2, 3))
    for path in sorted(directory.glob("chunk_*[0-9].npz")):
        with np.load(path) as data:
            counts += validation_counts(
                data["obs"].astype(np.float32),
                np.unpackbits(data["legal"], axis=1, bitorder="little")[:, :55].astype(
                    np.bool_
                ),
                data["logits"].astype(np.float32),
                weights,
            )
    if not counts[:, 0].all():
        raise ValueError("no held-out multi-card decisions")
    return {
        name: {
            "decisions": int(n),
            "teacher_top_one": matches / n,
            "mean_teacher_logit_regret": regret / n,
        }
        for name, (n, matches, regret) in zip(
            ("hand_rule", "fitted_continuation"), counts
        )
    }


if __name__ == "__main__":
    import pathlib
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="score the existing output weights on the supplied held-out shards",
    )
    args = parser.parse_args()
    if args.validate:
        print(
            json.dumps(
                validation(args.directory, np.load(args.output, allow_pickle=False)),
                indent=2,
            )
        )
        sys.exit(0)
    xtx = np.zeros((3, F * C, F * C), np.float64)
    xty = np.zeros((3, F * C), np.float64)
    count = 0
    for path in sorted(args.directory.glob("chunk_*[0-9].npz")):
        with np.load(path) as d:
            obs = d["obs"].astype(np.float32)
            legal = np.unpackbits(d["legal"], axis=1, bitorder="little")[:, :55].astype(
                np.bool_
            )
            logits = d["logits"].astype(np.float32)
        n = int(obs[0, 92])
        arr, tgt = rows(obs, legal, logits)
        # Favor close decisions. Large margins need less relative accuracy.
        a = arr / (1 + tgt[:, None] * 0.24) ** 0.5
        y = tgt / (1 + tgt * 0.24) ** 0.5
        xtx[n - 2] += a.T @ a
        xty[n - 2] += a.T @ y
        count += len(tgt)
    if not count:
        raise ValueError("no training comparisons")
    weights = np.stack(
        [np.linalg.solve(x + np.eye(F * C) * 15, y) for x, y in zip(xtx, xty)]
    ).reshape(3, C, F)
    print("pairs", count)
    np.save(args.output, weights)
