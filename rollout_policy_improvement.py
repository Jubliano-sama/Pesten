"""Distill exact-engine counterfactual rollouts into an information-safe action head.

Training branches can use a learned fully observed value as their terminal cost.
The deployed policy does not load this critic or receive any of its inputs.
"""

from __future__ import annotations

import argparse
from copy import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from batched_pesten import BatchedPesten, PHASE_CARD, PHASE_SUIT, POLICY_RANDOM, POLICY_RL
from temporal_batched_speed import (
    SafeEventCursor,
    TemporalSeatRuntime,
    _batch_temporal_actions,
    build_safe_decision,
)
from train_outcome_residual import collection_policy, file_hash


# Static tables are shared; these arrays all have a leading independent-game axis.
GAME_ARRAYS = (
    "policy_kinds",
    "actor_keys",
    "hands",
    "hand_counts",
    "hand_sizes",
    "draw_piles",
    "draw_counts",
    "discards",
    "discard_counts",
    "current_sort",
    "current_rank",
    "current_player",
    "turn_player",
    "direction",
    "penalty_amount",
    "sorts_played",
    "turn_counts",
    "cards_played",
    "game_over",
    "winners",
    "phase",
    "turn_started_with_one_card",
    "drawn_card",
)


class _ForkedShuffle:
    """Copy the deal RNG on its first use in a branch while the source is paused."""

    def __init__(self, source):
        self.source = source
        self.owned = None

    def shuffle(self, values):
        if self.owned is None:
            self.owned = np.random.RandomState(0)
            self.owned.set_state(self.source.get_state())
        self.owned.shuffle(values)


def fork_rows(engine, rows):
    branches = copy(engine)
    branches.num_games = len(rows)
    for name in GAME_ARRAYS:
        setattr(branches, name, getattr(engine, name)[rows].copy())
    branches.public_events = [engine.public_events[i].copy() for i in rows]
    branches.event_counts = dict(engine.event_counts)
    branches._native_event_deltas = engine._native_event_deltas.copy()
    branches.rngs = [_ForkedShuffle(engine.rngs[i]) for i in rows]
    return branches


def _leaf_scorer():
    from public_lookahead_leaf import _leaf_value, njit

    @njit(cache=True)
    def score(hands, sizes, seat, direction, suit, rank, penalty, random_others, weights):
        out = np.empty(len(sizes), np.float32)
        for i in range(len(out)):
            out[i] = _leaf_value(
                hands[i],
                sizes[i],
                int(seat[i]),
                int(direction[i]),
                int(suit[i]),
                int(rank[i]),
                int(penalty[i]),
                random_others[i],
                weights,
            )
        return out

    return score


def leaf_estimates(engine, rows, focal, random_others, score, weights):
    finished = engine.game_over[rows] | (engine.turn_counts[rows] >= engine.max_turns)
    result = (engine.winners[rows] == focal[rows]).astype(np.float32)
    live = rows[~finished]
    if len(live):
        order = (focal[live, None] + np.arange(engine.num_players)[None, :]) % engine.num_players
        result[~finished] = score(
            engine.hands[live[:, None], order],
            engine.hand_sizes[live[:, None], order],
            ((engine.current_player[live] - focal[live]) % engine.num_players).astype(np.int64),
            engine.direction[live],
            engine.current_sort[live],
            engine.current_rank[live],
            engine.penalty_amount[live],
            random_others[live],
            weights,
        )
    return result


def choose_actions(policy, engine, rows, seats, suit_rng):
    actions = np.empty(len(rows), np.int64)
    entries, indices = [], []
    for i, game in enumerate(rows):
        seat = int(engine.current_player[game])
        if engine.policy_kinds[game, seat] == POLICY_RANDOM:
            actions[i] = (
                50 + suit_rng[int(game)].randrange(4)
                if engine.phase[game] == PHASE_SUIT
                else engine.random_action(game)
            )
        else:
            runtime = seats[int(game), seat]
            entries.append((runtime, build_safe_decision(engine, game, seat, runtime.cursor)))
            indices.append(i)
    if entries:
        actions[indices] = _batch_temporal_actions(policy, entries, None, deterministic=True)
        for runtime, _ in entries:
            runtime.trajectory.clear()
            runtime.safe_history.clear()
    return actions, entries, indices


class _ForkedSeats(dict):
    def __init__(self, source, original):
        super().__init__()
        self.source = source
        self.original = original

    def __missing__(self, key):
        row, seat = key
        source = self.source[int(self.original[row]), seat]
        # State parts are read into a fresh batched bank by _batch_temporal_actions.
        rt = replace(
            source, game_index=row, cursor=replace(source.cursor), trajectory=[], safe_history=[]
        )
        self[key] = rt
        return rt


class _ForkedSuits(dict):
    def __init__(self, source, original):
        super().__init__()
        self.source = source
        self.original = original

    def __missing__(self, row):
        rng = random.Random(0)
        rng.setstate(self.source[int(self.original[row])].getstate())
        self[row] = rng
        return rng


def score_branches(
    policy,
    engine,
    seats,
    suit_rng,
    source_rows,
    forced,
    focal,
    random_others,
    horizons,
    scorer,
    weights,
):
    branches = fork_rows(engine, source_rows)
    branch_seats = _ForkedSeats(seats, source_rows)
    branch_suits = _ForkedSuits(suit_rng, source_rows)
    initial_turn = branches.turn_counts.copy()
    focal = focal[source_rows]
    random_others = random_others[source_rows]
    for row, act in enumerate(forced):
        branches.apply_action(row, int(act))
    scores = np.empty((len(forced), len(horizons)), np.float32)
    for i, horizon in enumerate(horizons):
        while True:
            active = branches.active_games()
            need = (branches.turn_counts[active] - initial_turn[active] < horizon) | (
                branches.phase[active] != PHASE_CARD
            )
            advancing = active[need]
            if not len(advancing):
                break
            choices, _, _ = choose_actions(policy, branches, advancing, branch_seats, branch_suits)
            for row, act in zip(advancing, choices):
                branches.apply_action(int(row), int(act))
        scores[:, i] = leaf_estimates(
            branches, np.arange(len(forced)), focal, random_others, scorer, weights
        )
    return scores


@torch.inference_mode()
def collect_chunk(policy, args, index, scorer, weights):
    games = args.games
    n = 2 + index % 3
    seed = args.seed + index * games
    rng = np.random.default_rng(seed + 80_000_000)
    deals = np.arange(seed, seed + games, dtype=np.uint32)
    focal = np.arange(games) % n
    random_others = np.arange(games) // n % 2 == 0
    kinds = np.where(
        random_others[:, None] & (np.arange(n)[None] != focal[:, None]), POLICY_RANDOM, POLICY_RL
    ).astype(np.int8)
    engine = BatchedPesten(
        kinds, np.full(kinds.shape, "policy", dtype=object), deals, max_turns=1000
    )
    seats = {
        (g, s): TemporalSeatRuntime(
            g, g, s, policy, torch.Generator(), policy.initial_state(), SafeEventCursor()
        )
        for g in range(games)
        for s in range(n)
        if kinds[g, s] == POLICY_RL
    }
    suit_rng = [random.Random(int(s)) for s in deals]
    data = {k: [] for k in ("obs", "seen", "value", "logits", "width", "act", "q", "episode")}
    if args.beliefs:
        data["tracking"] = []
    while len(active := engine.active_games()):
        moves, entries, indices = choose_actions(policy, engine, active, seats, suit_rng)
        if entries:
            core = policy.puffer_policy
            logits = core.last_logits.numpy()
            seen = core.last_counts.numpy()
            value = core.last_value.numpy().reshape(-1)
            source_rows, forced, observations, positions, widths = [], [], [], [], []
            for j, (runtime, dec) in enumerate(entries):
                g = runtime.game_index
                if runtime.seat != focal[g]:
                    continue
                best = int(moves[indices[j]])
                legal = np.flatnonzero(
                    np.asarray(dec.legal_actions) & (logits[j] >= logits[j, best] - args.margin)
                )
                if len(legal) < 2:
                    continue
                if rng.random() < args.explore:
                    p = np.exp((logits[j, legal] - logits[j, best]) / 1.5)
                    moves[indices[j]] = rng.choice(legal, p=p / p.sum())
                if rng.random() >= args.keep:
                    continue
                source_rows.extend([g] * len(legal))
                forced.extend(legal)
                widths.append(len(legal))
                positions.append(j)
                observations.append(dec.observation.values)
            if source_rows:
                source_rows = np.asarray(source_rows)
                forced = np.asarray(forced)
                # Copy inputs before a continuation invokes the same shared teacher.
                data["obs"].append(np.asarray(observations, np.float16))
                data["value"].append(value[positions].copy())
                if args.beliefs:
                    data["tracking"].append(core.last_history.numpy()[positions].astype(np.float16))
                data["seen"].append(seen[positions].astype(np.float16))
                data["logits"].append(logits[positions].astype(np.float16))
                data["episode"].append(
                    np.asarray([entries[j][0].game_index for j in positions], np.int32)
                )
                data["width"].append(np.asarray(widths, np.int8))
                data["act"].append(forced.astype(np.uint8))
                data["q"].append(
                    score_branches(
                        policy,
                        engine,
                        seats,
                        suit_rng,
                        source_rows,
                        forced,
                        focal,
                        random_others,
                        args.horizons,
                        scorer,
                        weights,
                    )
                )
        for row, move in zip(active, moves):
            engine.apply_action(int(row), int(move))
    if data["obs"]:
        out = {k: np.concatenate(v) for k, v in data.items()}
    else:
        # A small shard may finish without a sampled choice. It still counts
        # toward the completed games and can be combined with other shards.
        out = dict(
            obs=np.empty((0, 94), np.float16),
            seen=np.empty((0, 50), np.float16),
            value=np.empty(0, np.float32),
            logits=np.empty((0, 55), np.float16),
            width=np.empty(0, np.int8),
            act=np.empty(0, np.uint8),
            q=np.empty((0, len(args.horizons)), np.float32),
            episode=np.empty(0, np.int32),
        )
        if args.beliefs:
            from branch_public_belief import HISTORY_SIZE

            out["tracking"] = np.empty((0, HISTORY_SIZE), np.float16)
    return dict(
        **out,
        seed=seed,
        games=games,
        random_others=random_others,
        winners=engine.winners,
        focal=focal,
        horizons=args.horizons,
    )


def collect(args):
    policy = collection_policy(args.orbit)
    if args.beliefs:
        from branch_public_belief import TrackingCollector

        policy.puffer_policy = TrackingCollector(policy.puffer_policy)
    scorer = _leaf_scorer()
    weights = np.load(args.leaf)
    leaf_hash = file_hash(args.leaf)
    args.directory.mkdir(parents=True, exist_ok=True)
    for i in range(args.start, args.start + args.chunks):
        target = args.directory / f"fork_{i:04d}.npz"
        if target.exists():
            raise ValueError(f"shard exists: {target}")
        start = time.monotonic()
        data = collect_chunk(policy, args, i, scorer, weights)
        tmp = target.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **data, leaf_hash=leaf_hash)
        tmp.replace(target)
        print(
            json.dumps(
                dict(
                    shard=i,
                    games=args.games,
                    roots=len(data["obs"]),
                    branches=len(data["q"]),
                    leaf_hash=leaf_hash,
                    wall_seconds=time.monotonic() - start,
                )
            ),
            flush=True,
        )


def load_training_data(paths, horizon_weights, device, beliefs=False):
    collected = {k: [] for k in ("obs", "seen", "value", "logits", "q", "mask", "cell", "valid")}
    if beliefs:
        collected["tracking"] = []
    games = branches = 0
    leaf_hashes = set()
    for path in paths:
        shard = int(path.stem.split("_")[-1])
        with np.load(path) as d:
            width = d["width"].astype(np.int64)
            n = len(width)
            rows = np.repeat(np.arange(n), width)
            actions = d["act"]
            q = np.zeros((n, 55), dtype=np.float32)
            mask = np.zeros((n, 55), dtype=bool)
            q[rows, actions] = d["q"] @ horizon_weights
            mask[rows, actions] = True
            if beliefs:
                collected["tracking"].append(d["tracking"].astype(np.float32))
            for k in ("obs", "seen", "value", "logits"):
                collected[k].append(d[k].astype(np.float32))
            collected["q"].append(q)
            collected["mask"].append(mask)
            collected["valid"].append(np.full(n, shard % 15 < 3))
            collected["cell"].append(
                shard % 3 + 3 * d["random_others"][d["episode"]].astype(np.int64)
            )
            games += int(d["games"])
            branches += len(rows)
            leaf_hashes.add(str(d["leaf_hash"]))
    return (
        {k: torch.from_numpy(np.concatenate(v)).to(device) for k, v in collected.items()},
        dict(games=games, simulated_actions=branches, leaf_hashes=sorted(leaf_hashes)),
    )


def fit(args):
    from branch_outcome_policy import BranchActionHead
    from train_outcome_residual import TEACHER

    torch.manual_seed(args.fit_seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.fit_seed)
    paths = [args.directory / f"fork_{i:04d}.npz" for i in range(args.chunks)]
    if not all(path.is_file() for path in paths):
        raise ValueError("requested counterfactual shards are incomplete")
    weights = np.asarray(args.horizon_weights, dtype=np.float32)
    if np.any(weights < 0) or weights.sum() == 0:
        raise ValueError("horizon weights must be nonnegative and nonzero")
    data, info = load_training_data(
        paths, weights / weights.sum(), args.device, args.beliefs is not None
    )
    train = (~data["valid"]).nonzero().flatten()
    valid = data["valid"].nonzero().flatten()
    if not len(train) or not len(valid):
        raise ValueError("need training and validation deals")
    frequencies = None
    if args.beliefs:
        if not args.graph:
            raise ValueError("belief inputs require the per-card graph")
        frequencies = torch.tensor(
            json.loads(args.beliefs.read_text())["weights"], dtype=torch.float32
        )
    model = BranchActionHead(
        graph=args.graph, style_heads=args.style is not None, frequencies=frequencies
    ).to(args.device)
    style_bundle = (
        None
        if args.style is None
        else torch.load(args.style, map_location="cpu", weights_only=True)
    )
    if style_bundle and style_bundle["kind"] != "pesten_public_opponent_style":
        raise ValueError("unexpected style classifier checkpoint")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate * 0.15
    )
    logits, mask, q = data["logits"], data["mask"], data["q"]
    uniform = mask / mask.sum(1, keepdim=True)
    near_teacher = (logits / args.weight_temperature).masked_fill(~mask, -torch.inf).softmax(1)
    action_weight = 0.5 * uniform + 0.5 * near_teacher

    def centered(x):
        return x - (action_weight * x).sum(1, keepdim=True)

    q_centered = centered(q)
    logit_centered = centered(logits)
    slope = float(
        (
            (q_centered[train] * logit_centered[train] * action_weight[train]).sum()
            / (logit_centered[train].square() * action_weight[train]).sum()
        ).item()
    )
    prior = args.prior if args.prior is not None else slope
    labels = q_centered - prior * logit_centered
    # Each table size/opponent cell contributes equally even when its games
    # naturally produce more eligible decision points.
    freq = torch.bincount(data["cell"][train], minlength=6)
    if not torch.all(freq):
        raise ValueError("training examples missing an opponent/player-count cell")
    cell_weight = len(train) / (6 * freq)

    def predictions(idx):
        return model(
            data["obs"][idx],
            data["seen"][idx],
            data["value"][idx],
            mask[idx],
            (data["cell"][idx] >= 3).float() if args.style is not None else None,
            data["tracking"][idx] if args.beliefs else None,
        )

    def loss_for(idx, scores):
        aw = action_weight[idx]
        err = scores - (aw * scores).sum(1, keepdim=True) - labels[idx]
        row_mse = (aw * err.square()).sum(1)
        return (cell_weight[data["cell"][idx]] * row_mse).mean()

    best_loss = float("inf")
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train[torch.randperm(len(train), device=args.device)].split(args.batch):
            loss = loss_for(batch, predictions(batch))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        schedule.step()
        model.eval()
        with torch.inference_mode():
            total = 0.0
            for batch in valid.split(args.batch):
                total += float(loss_for(batch, predictions(batch))) * len(batch)
            loss = total / len(valid)
        if loss < best_loss:
            best_loss = loss
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(loss)
        print(f"epoch {epoch}, held-out action-advantage MSE {loss:.7f}", flush=True)
    model.load_state_dict(best)
    q_gain = {}
    with torch.inference_mode():
        scores = torch.cat([predictions(batch) for batch in valid.split(args.batch)])
        valid_logits = logits[valid].masked_fill(~mask[valid], -torch.inf)
        valid_q = q[valid]
        champion = valid_logits.argmax(1, keepdim=True)
        for scale in args.scales:
            best_action = (valid_logits + scale * scores).argmax(1, keepdim=True)
            gain = (valid_q.gather(1, best_action) - valid_q.gather(1, champion)).flatten()
            cells = [float(gain[data["cell"][valid] == i].mean()) for i in range(6)]
            q_gain[str(scale)] = {
                "cells": cells,
                "macro": sum(cells) / 6,
                "changed_fraction": float((best_action != champion).float().mean()),
            }
    args.output.mkdir(parents=True, exist_ok=True)
    for scale in args.scales:
        bundle = dict(
            kind="pesten_public_branch_outcome_v1",
            base=str(TEACHER),
            base_sha256=file_hash(TEACHER),
            graph=args.graph,
            scale=scale,
            margin=args.margin,
            head=best,
        )
        if frequencies is not None:
            bundle["beliefs"] = True
        if style_bundle:
            bundle["style"] = style_bundle
        torch.save(bundle, args.output / f"branch_{scale:g}.pt")
    info.update(
        roots=len(train) + len(valid),
        train_roots=len(train),
        validation_roots=len(valid),
        validation_mse=history,
        fitted_prior_slope=slope,
        prior=prior,
        validation_value_gain=q_gain,
        mean_legal_options=float(mask.float().sum(1).mean()),
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"
        },
    )
    (args.output / "fit.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--leaf", type=Path, required=True)
    p.add_argument("--orbit", type=int, choices=(1, 4, 24), default=1)
    p.add_argument("--games", type=int, default=192)
    p.add_argument("--chunks", type=int, default=24)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--seed", type=int, default=512_000_000)
    p.add_argument("--explore", type=float, default=0.2)
    p.add_argument("--keep", type=float, default=0.65)
    p.add_argument("--margin", type=float, default=8.0)
    p.add_argument("--beliefs", action="store_true")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 4])
    p.set_defaults(func=collect)
    f = sub.add_parser("fit")
    f.add_argument("--directory", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--chunks", type=int, required=True)
    f.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    f.add_argument(
        "--beliefs", type=Path, help="fit using tracker shards and the public hand-frequency model"
    )
    f.add_argument(
        "--style", type=Path, help="public posterior used to mix learned opponent-specific values"
    )
    f.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    f.add_argument("--epochs", type=int, default=14)
    f.add_argument("--batch", type=int, default=512)
    f.add_argument("--learning-rate", type=float, default=0.0015)
    f.add_argument("--weight-temperature", type=float, default=1.5)
    f.add_argument("--prior", type=float)
    f.add_argument("--scales", type=float, nargs="+", default=[15.0, 35.0, 65.0])
    f.add_argument("--horizon-weights", type=float, nargs="+", default=[0.35, 0.65])
    f.add_argument("--margin", type=float, default=8.0)
    f.add_argument("--fit-seed", type=int, default=512_750_000)
    f.set_defaults(func=fit)
    args = parser.parse_args()
    if args.operation == "collect" and (
        not 0 < args.keep <= 1
        or not 0 <= args.explore <= 1
        or min(args.games, args.chunks, *args.horizons) < 1
        or args.horizons != sorted(set(args.horizons))
    ):
        parser.error(
            "expected positive sample counts, sorted unique horizons, and valid probabilities"
        )
    torch.set_num_threads(1)
    args.func(args)


if __name__ == "__main__":
    main()
