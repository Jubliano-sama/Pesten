"""Fit hidden-hand sampling frequencies from training hands and canonical Random games."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import numpy as np

from batched_pesten import BatchedPesten, CARD_RANK, POLICY_RANDOM, PHASE_SUIT

CATEGORIES = np.asarray(
    [
        1
        if rank == 1
        else 2
        if rank == 2
        else 3
        if rank == 7
        else 4
        if rank == 8
        else 5
        if rank == 13
        else 6
        if rank == 0
        else 0
        for rank in CARD_RANK
    ],
    dtype=np.int8,
)


def _add(counts, n, size, hand, last, declared):
    row = counts[n - 2, min(size, 9) - 1]
    grouped = np.bincount(CATEGORIES, weights=hand, minlength=7)
    row[:7] += grouped
    suited = hand[:48].reshape(4, 12)
    if 0 <= last < 48:
        row[7] += suited[last // 12].sum()
        row[8] += suited.sum()
        row[9] += suited[:, last % 12].sum()
        row[10] += suited.sum()
    if declared >= 0:
        row[11] += suited[declared].sum()
        row[12] += suited.sum()


def collect_strong(directory):
    counts = np.zeros((3, 9, 13))
    rows = 0
    sources = []
    for path in sorted(directory.glob("chunk_*[0-9].npz")):
        with np.load(path) as data:
            obs, actions = data["obs"], data["act"]
            episode, seed = data["episode"], int(data["seed"])
        n = int(obs[0, 92])
        last_episode = -1
        last = declared = -1
        for i in range(len(episode)):
            if episode[i] != last_episode:
                last = declared = -1
                last_episode = episode[i]
            elif actions[i - 1] < 50 and obs[i, 91] == obs[i - 1, 91] - 1:
                last = int(actions[i - 1])
                declared = -1
            elif 50 <= actions[i - 1] < 54:
                declared = int(actions[i - 1]) - 50
            # Sample hands once at an ordinary card decision. A suit
            # declaration is already reflected in the subsequent decision.
            if obs[i, 93] == 0:
                _add(counts, n, int(obs[i, 91]), obs[i, :50], last, declared)
                rows += 1
        sources.append(
            {
                "filename": path.name,
                "seed": seed,
                "games": int(episode.max() + 1),
                "decisions": len(episode),
            }
        )
    return counts, rows, sources


def collect_random(games_per_size, seed):
    counts = np.zeros((3, 9, 13))
    rows = 0
    for n in (2, 3, 4):
        for start in range(0, games_per_size, 128):
            m = min(128, games_per_size - start)
            seeds = np.arange(
                seed + n * 100000 + start,
                seed + n * 100000 + start + m,
                dtype=np.uint32,
            )
            suit_rng = [random.Random(int(s)) for s in seeds]
            engine = BatchedPesten(
                np.full((m, n), POLICY_RANDOM),
                np.full((m, n), None),
                seeds,
                max_turns=1000,
                record_public_events=True,
            )
            last = np.full((m, n), -1)
            declared = np.full((m, n), -1)
            while len(active := engine.active_games()):
                for g in active:
                    seat = engine.current_player[g]
                    if engine.phase[g] == PHASE_SUIT:
                        action = 50 + suit_rng[g].randint(0, 3)
                    else:
                        _add(
                            counts,
                            n,
                            int(engine.hand_sizes[g, seat]),
                            engine.hand_counts[g, seat],
                            last[g, seat],
                            declared[g, seat],
                        )
                        rows += 1
                        action = engine.random_action(g)
                    cursor = len(engine.public_events[g])
                    engine.apply_action(g, action)
                    for kind, actor, value in engine.public_events[g][cursor:]:
                        if kind == "play":
                            last[g, actor], declared[g, actor] = value, -1
                        elif kind == "choose_sort":
                            declared[g, actor] = value
    return counts, rows


def fit(strong, random_counts):
    counts = np.stack((strong, random_counts))
    physical = np.array([32, 4, 4, 4, 4, 4, 2]) / 54
    probabilities = (counts[..., :7] + 70 * physical) / (
        counts[..., :7].sum(-1, keepdims=True) + 70
    )
    weights = np.zeros((*counts.shape[:-1], 10))
    weights[..., :7] = np.log(probabilities / physical)
    for index, (pos, total, base) in enumerate(
        ((7, 8, 1 / 4), (9, 10, 1 / 12), (11, 12, 1 / 4)), start=7
    ):
        p = (counts[..., pos] + 70 * base) / (counts[..., total] + 70)
        weights[..., index] = np.log(p / (1 - p) * ((1 - base) / base)).clip(-2.5, 2.5)
    return weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("shards", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--random-games", type=int, default=1024, help="games at each table size"
    )
    p.add_argument("--seed", type=int, default=726000000)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError(f"output exists: {args.output}")
    strong, n, sources = collect_strong(args.shards)
    print(f"Collected {n} recurrent-player hands", flush=True)
    random_counts, m = collect_random(args.random_games, args.seed)
    print(f"Collected {m} random-player hands", flush=True)
    weights = fit(strong, random_counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "kind": "pesten_opponent_hand_frequencies",
                "weights": weights.tolist(),
                "strong_observations": n,
                "random_observations": m,
                "random_seed": args.seed,
                "random_games_per_table": args.random_games,
                "strong_shards": sources,
                "category_counts": np.stack((strong, random_counts))
                .astype(np.int64)
                .tolist(),
            },
            separators=(",", ":"),
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
