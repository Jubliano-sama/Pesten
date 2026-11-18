import sys, json, random
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import numpy as np
from batched_pesten import (
    BatchedPesten,
    PHASE_SUIT,
    POLICY_RANDOM,
    POLICY_SMARTER,
)


def estimate(recent, capacity, hand, k):
    maxes = np.ones(50, dtype="int8")
    maxes[48:] = [4, 2]
    maxes -= hand
    known = np.zeros(50, dtype="int8")
    # More recent public plays are more likely to belong to the current pile.
    for i, c in enumerate(recent[::-1]):
        if i >= k or sum(known) >= capacity:
            break
        if known[c] < maxes[c]:
            known[c] += 1
    return known


def run(n, m):
    seeds = np.arange(729000000 + n * 10000, 729000000 + n * 10000 + m, dtype=np.uint32)
    kinds = np.where(
        np.arange(m)[:, None] % 2 == 0,
        np.full((m, n), POLICY_RANDOM),
        np.full((m, n), POLICY_SMARTER),
    )
    e = BatchedPesten(
        kinds, np.full((m, n), None), seeds, max_turns=1000, record_public_events=True
    )
    suits = [random.Random(int(s)) for s in seeds]
    histories = [[] for _ in seeds]
    histpos = [0] * m
    last_fields = [None for _ in seeds]
    total = np.zeros((4, 3, 5))
    while len(active := e.active_games()):
        for g in active:
            seat = int(e.current_player[g])
            if seat == g % n:
                cap = 54 - int(e.hand_sizes[g].sum())
                prev = last_fields[g]
                recent = histories[g]
                for event in e.public_events[g][histpos[g] :]:
                    if event[0] == "play":
                        recent.append(event[2])
                histpos[g] = len(e.public_events[g])
                hp = e.hand_counts[g, seat].astype(np.int8)
                if prev is None:
                    kvals = [min(cap, k + len(recent)) for k in (1, 2, 3)]
                    # Accept a publicly observed ordinary opening card when no plays have occurred.
                    if not recent and e.current_rank[g] in (3, 4, 5, 6, 9, 10, 11, 12):
                        recent.append(
                            int(e.current_sort[g]) * 12 + int(e.current_rank[g]) - 1
                        )
                else:
                    last_cap, last_cards, old_kvals = prev
                    delta = len(recent) - last_cards
                    draws = last_cap - cap + delta
                    kvals = []
                    for old_k in old_kvals:
                        if old_k + delta <= cap:
                            kvals.append(old_k + delta)
                        else:
                            after = draws - (last_cap - old_k)
                            v = 1 + round((after - 1) * delta / max(1, draws))
                            kvals.append(max(1, min(cap, 1 + delta, v)))
                last_fields[g] = cap, len(recent), kvals
                actual = np.bincount(
                    e.discards[g, : int(e.discard_counts[g])], minlength=50
                )
                hidden = e.hand_counts[g].sum(0) - e.hand_counts[g, seat]
                candidates = [estimate(recent, cap, hp, k) for k in (24, *kvals)]
                for j, b in enumerate(candidates):
                    k = (
                        0
                        if sum(e.sorts_played[g]) < 20
                        else 1
                        if sum(e.sorts_played[g]) < 40
                        else 2
                    )
                    total[j, k] += np.array(
                        [
                            1,
                            np.minimum(b, actual).sum(),
                            (b - actual).clip(0).sum(),
                            (actual - b).clip(0).sum(),
                            np.minimum(hidden, (b - actual).clip(0)).sum(),
                        ]
                    )
            act = (
                50 + suits[g].randrange(4)
                if e.phase[g] == PHASE_SUIT and kinds[g, seat] == POLICY_RANDOM
                else e.non_neural_action(int(g))
            )
            e.apply_action(int(g), act)
    return total.astype(np.int64).tolist()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare public discard estimates with true piles on separate diagnostic games."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "games_per_table": 480,
        "base_seed": 729000000,
        "trackers": [
            "fixed24",
            "estimated_opening1",
            "estimated_opening2",
            "estimated_opening3",
        ],
        "phases": ["public plays <20", "20-39", ">=40"],
        "columns": [
            "observations",
            "true positives",
            "false positives",
            "false negatives",
            "false positives actually held by opponents",
        ],
        "by_players": {str(n): run(n, 480) for n in (2, 3, 4)},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
