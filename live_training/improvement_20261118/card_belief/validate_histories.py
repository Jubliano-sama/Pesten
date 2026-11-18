"""Compare belief log scores with the deployed public-history tracker on new complete games."""

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch

from batched_pesten import (
    BatchedPesten,
    CARD_RANK,
    CARD_SORT,
    PHASE_CARD,
    PHASE_SUIT,
    POLICY_RANDOM,
    POLICY_RL,
)
from branch_public_belief import TrackingCollector
from public_card_belief import (
    CATEGORY,
    FAMILIES,
    apply_card_belief,
    card_features,
    njit,
    size_bin,
)
from public_lookahead import (
    BELIEF_WIDTH,
    EVIDENCE,
    LAST_CARD,
    DECLARATION,
    PILE_SIZE,
    RECENT_PLAYS,
)
from temporal_batched_speed import (
    SafeEventCursor,
    TemporalSeatRuntime,
    _batch_temporal_actions,
    build_safe_decision,
)
from train_outcome_residual import collection_policy


@njit(cache=True)
def tracked_odds(obs, history, n, role, prior):
    """Reconstruct the search sampler’s public pool and unnormalized hand odds."""
    sizes = obs[69 : 69 + n].astype(np.int64)
    own = obs[:50]
    suit = np.argmax(obs[50:55])
    rank = np.argmax(obs[55:69])
    direction = int(obs[89])
    remaining = np.ones(50, np.int16)
    remaining[48], remaining[49] = 4, 2
    remaining -= own.astype(np.int16)
    capacity = 54 - sizes.sum()
    discard_count = 0
    observed = int(history[PILE_SIZE])
    start = PILE_SIZE - 1 if observed else RECENT_PLAYS - 1
    end = start - observed if observed else -1
    for j in range(start, end, -1):
        c = int(history[j]) - 1
        if c >= 0 and remaining[c] > 0 and discard_count < capacity:
            remaining[c] -= 1
            discard_count += 1
    if discard_count == 0 and capacity > 0 and suit < 4 and 1 <= rank <= 12:
        top = 12 * suit + rank - 1
        if remaining[top] > 0:
            remaining[top] -= 1
    w0 = np.ones((2, n, 50))
    for rel in range(1, n):
        absolute = rel * direction % n
        offset = EVIDENCE + absolute * BELIEF_WIDTH
        last = int(history[LAST_CARD + absolute]) - 1
        declared = int(history[DECLARATION + absolute]) - 1
        for c in range(50):
            s, r = CARD_SORT[c], CARD_RANK[c]
            base = history[offset + s]
            if c < 48:
                base += history[offset + 4 + r]
            f = prior[role, n - 2, min(9, sizes[rel]) - 1]
            logw = base + f[CATEGORY[c]]
            if 0 <= last < 48 and c < 48:
                logw += f[7] * (CARD_SORT[last] == s) + f[8] * (CARD_RANK[last] == r)
            if c < 48 and declared == s:
                logw += f[9] - max(0.0, base)
            w0[role, rel, c] = np.exp(logw)
    return w0, remaining


@njit(cache=True)
def encode_history(obs, history, targets, role, prior):
    n = len(targets)
    sizes = obs[69 : 69 + n].astype(np.int64)
    odds, remaining = tracked_odds(obs, history, n, role, prior)
    probs = np.asarray(odds[role, 1:] * remaining, dtype=np.float32)
    labels = np.asarray(targets[1:] * (remaining > 0), dtype=np.uint8)
    features = np.empty((n - 1, 50, FAMILIES), dtype=np.uint8)
    groups = np.empty(n - 1, dtype=np.uint8)
    for rel in range(1, n):
        groups[rel - 1] = (role * 3 + n - 2) * 5 + size_bin(sizes[rel])
        features[rel - 1] = card_features(
            obs[:50],
            sizes,
            rel,
            np.argmax(obs[50:55]),
            np.argmax(obs[55:69]),
            int(obs[90]),
        )
    return probs, labels, features, groups


@njit(cache=True)
def scores(obs, history, targets, role, prior, coeff):
    n = len(targets)
    sizes = obs[69 : 69 + n].astype(np.int64)
    own = obs[:50]
    suit = np.argmax(obs[50:55])
    rank = np.argmax(obs[55:69])
    penalty = int(obs[90])
    w0, remaining = tracked_odds(obs, history, n, role, prior)
    w1 = w0.copy()
    result = np.zeros(5)
    for rel in range(1, n):
        apply_card_belief(w1, own, sizes, rel, suit, rank, penalty, coeff)
        a = w0[role, rel] * remaining
        b = w1[role, rel] * remaining
        a /= a.sum()
        b /= b.sum()
        result[0] += 1
        for c in range(50):
            count = targets[rel, c]
            if count and remaining[c] > 0:
                result[1] += count
                result[3] -= count * np.log(a[c])
                result[4] -= count * np.log(b[c])
            elif count:
                result[2] += count
    return result


@torch.inference_mode()
def score_games(n, games, seed, model, prior, odds, examples=None):
    starts = np.arange(seed, seed + games, dtype=np.uint32)
    focal = np.arange(games) % n
    random_others = (np.arange(games) // n) % 2 == 0
    kinds = np.where(
        random_others[:, None] & (np.arange(n)[None, :] != focal[:, None]),
        POLICY_RANDOM,
        POLICY_RL,
    )
    board = BatchedPesten(
        kinds,
        np.full(kinds.shape, "teacher"),
        starts,
        max_turns=1000,
        record_public_events=True,
    )
    suit_rng = [random.Random(int(s)) for s in starts]
    rng = np.random.default_rng(seed)
    seats = {
        (g, s): TemporalSeatRuntime(
            g, g, s, model, torch.Generator(), model.initial_state(), SafeEventCursor()
        )
        for g in range(games)
        for s in range(n)
        if kinds[g, s] == POLICY_RL
    }
    totals = np.zeros((2, 5))
    while len(active := board.active_games()):
        actions = np.empty(len(active), np.int64)
        entries = []
        indices = []
        for i, g in enumerate(active):
            s = int(board.current_player[g])
            g = int(g)
            if kinds[g, s] == POLICY_RANDOM:
                actions[i] = (
                    50 + suit_rng[g].randint(0, 3)
                    if board.phase[g] == PHASE_SUIT
                    else board.random_action(g)
                )
            else:
                runtime = seats[g, s]
                entries.append(
                    (runtime, build_safe_decision(board, g, s, runtime.cursor))
                )
                indices.append(i)
        if entries:
            actions[indices] = _batch_temporal_actions(
                model, entries, None, deterministic=True
            )
            for runtime, decision in entries:
                g, s = runtime.game_index, runtime.seat
                if (
                    s == focal[g]
                    and board.phase[g] == PHASE_CARD
                    and rng.random() < 0.55
                ):
                    role = int(random_others[g])
                    order = (s + np.arange(n) * board.direction[g]) % n
                    inputs = (
                        np.asarray(decision.observation.values, np.float32),
                        runtime.state[-1][0, 0].numpy(),
                        board.hand_counts[g, order],
                        role,
                        prior,
                    )
                    if examples is None:
                        totals[role] += scores(*inputs, odds)
                    else:
                        examples.append(encode_history(*inputs))
                runtime.trajectory.clear()
                runtime.safe_history.clear()
        for g, a in zip(active, actions):
            board.apply_action(int(g), int(a))
    return totals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--games-per-table", type=int, default=768)
    p.add_argument("--orbit", type=int, default=4)
    p.add_argument("--seed", type=int, default=1650100000)
    args = p.parse_args()
    torch.set_num_threads(1)
    cfg = json.loads(args.profile.read_text())
    prior = np.asarray(
        json.loads(Path(cfg["posterior"]["path"]).read_text())["weights"]
    )
    odds = np.load(cfg["card_belief"]["path"])
    model = collection_policy(args.orbit)
    model.puffer_policy = TrackingCollector(model.puffer_policy)
    rows = []
    for n in (2, 3, 4):
        total = np.zeros((2, 5))
        for start in range(0, args.games_per_table, 192):
            total += score_games(
                n,
                min(192, args.games_per_table - start),
                args.seed + n * 100000 + start,
                model,
                prior,
                odds,
            )
        for role in (0, 1):
            samples, cards, blocked, a, b = total[role]
            row = dict(
                players=n,
                opponent=("teacher", "random")[role],
                hands=int(samples),
                possible_cards=int(cards),
                impossible_cards=int(blocked),
                baseline_nats=a / cards,
                conditional_nats=b / cards,
            )
            rows.append(row)
            print(row, flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                seed=args.seed,
                games_per_table=args.games_per_table,
                orbit=args.orbit,
                cells=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
