"""Learn a suit-symmetric action-value correction from exploratory Pesten games."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from strongest_policy_search import NativeSuitOrbitCore, TEACHER
from temporal_puffer_v4 import load_temporal_policy_checkpoint
from temporal_batched_speed import (
    SafeEventCursor, TemporalSeatRuntime, _advance_event_states,
    _batch_temporal_actions, build_safe_decision,
)
from batched_pesten import BatchedPesten, PHASE_SUIT, POLICY_RANDOM, POLICY_RL


class PublicHistoryCore(nn.Module):
    """Add a decayed count of publicly played cards to the actor's memory."""

    def __init__(self, core):
        super().__init__()
        self.core = core

    def initial_state(self, batch_size, device):
        return (*self.core.initial_state(batch_size, device),
                torch.zeros(1, batch_size, 50, device=device))

    @staticmethod
    def played_cards(tokens, previous):
        play = (tokens[:, 99] == 0) & ((tokens[:, 94] * 3).round() == 1)
        card = (tokens[:, 96] * 50).round().long().sub(1).clamp(0, 49)
        counts = previous[0] * (1.0 - .025 * play.to(tokens.dtype)[:, None])
        counts.scatter_add_(1, card[:, None], play.to(tokens.dtype)[:, None])
        return counts[None]

    def forward_events_eval(self, tokens, state):
        return (*_advance_event_states(self.core, tokens, state[:-1]),
                self.played_cards(tokens, state[-1]))

    def forward_eval(self, tokens, state):
        logits, value, next_state = self.core.forward_eval(tokens, state[:-1])
        counts = self.played_cards(tokens, state[-1])
        logits = self.decision_logits(tokens, counts[0], logits, value)
        return logits, value, (*next_state, counts)


class _CollectionCore(PublicHistoryCore):
    def decision_logits(self, tokens, counts, logits, value):
        self.last_logits = logits
        self.last_counts = counts
        self.last_value = value
        return logits


def teacher(orbit_size=1, device="cpu", path=TEACHER, *, trace_cpu=False):
    policy, _ = load_temporal_policy_checkpoint(path, device=device)
    if trace_cpu and torch.device(device).type == "cpu":
        from vector_puffer_rollout import _trace_worker_min_gru
        _trace_worker_min_gru(policy, shared_float_weights=True)
    if orbit_size != 1:
        policy.puffer_policy = NativeSuitOrbitCore(
            policy.puffer_policy, alpha=1, orbit_size=orbit_size)
    return policy.to(device).eval().requires_grad_(False)


def collection_policy(orbit_size=1):
    """Prepare one traced CPU teacher for every simulated seat in a shard."""
    policy = teacher(orbit_size, trace_cpu=True)
    policy.puffer_policy = _CollectionCore(policy.puffer_policy)
    return policy.eval()


@torch.inference_mode()
def collect_chunk(policy, players, games, seed, temperature, epsilon):
    """Explore with the focal seat; other seats play greedy or canonical random."""
    rng = np.random.default_rng(seed + 100_000_000)
    designated = np.arange(games) % players
    is_random = (np.arange(games) // players) % 2 == 0
    kinds = np.where(is_random[:, None] & (np.arange(players)[None] != designated[:, None]),
                     POLICY_RANDOM, POLICY_RL).astype(np.int8)
    seeds = np.arange(seed, seed + games, dtype=np.uint32)
    engine = BatchedPesten(kinds, np.full(kinds.shape, "policy", dtype=object), seeds,
                           max_turns=1000, record_public_events=True)
    suit_rng = [random.Random(int(s)) for s in seeds]
    seats = {
        (g, s): TemporalSeatRuntime(g, g, s, policy, torch.Generator(),
                                     policy.initial_state(), SafeEventCursor())
        for g in range(games) for s in range(players) if kinds[g, s] == POLICY_RL
    }
    data = {k: [] for k in ("obs", "seen", "legal", "logits", "value", "act", "episode")}
    while len(active := engine.active_games()):
        entries = []
        actions = np.empty(len(active), dtype=np.int64)
        neural_idx = []
        for i, row in enumerate(active):
            seat = int(engine.current_player[row])
            if kinds[row, seat] == POLICY_RANDOM:
                actions[i] = (50 + suit_rng[row].randint(0, 3)
                              if engine.phase[row] == PHASE_SUIT else engine.random_action(row))
            else:
                runtime = seats[int(row), seat]
                entries.append((runtime, build_safe_decision(engine, row, seat, runtime.cursor)))
                neural_idx.append(i)
        if entries:
            greedy = _batch_temporal_actions(policy, entries, None, deterministic=True)
            actions[neural_idx] = greedy
            focal = [i for i, (r, _) in enumerate(entries) if r.seat == designated[r.game_index]]
            if focal:
                indices = np.asarray(focal)
                logits = policy.puffer_policy.last_logits.numpy()[indices]
                decisions = [entries[i][1] for i in focal]
                masks = np.asarray([d.legal_actions for d in decisions])
                masked = np.where(masks, logits, -np.inf)
                probs = np.exp((masked - masked.max(1, keepdims=True)) / temperature)
                probs /= probs.sum(1, keepdims=True)
                probs = (1 - epsilon) * probs + epsilon * masks / masks.sum(1, keepdims=True)
                u = rng.random(len(focal))
                sampled = (probs.cumsum(1) < u[:, None]).sum(1).clip(max=54)
                actions[np.asarray(neural_idx)[indices]] = sampled
                data["obs"].append(np.asarray([d.observation.values for d in decisions], dtype=np.float16))
                data["seen"].append(policy.puffer_policy.last_counts.numpy()[indices].astype(np.float16))
                data["legal"].append(np.packbits(masks, axis=1, bitorder="little"))
                data["logits"].append(logits.astype(np.float16))
                data["value"].append(policy.puffer_policy.last_value.numpy()[indices].reshape(-1).astype(np.float32))
                data["act"].append(sampled.astype(np.uint8))
                data["episode"].append(np.asarray([entries[i][0].game_index for i in focal], dtype=np.int32))
            for runtime, _ in entries:
                runtime.trajectory.clear()
                runtime.safe_history.clear()
        for row, action in zip(active, actions):
            engine.apply_action(int(row), int(action))
    output = {k: np.concatenate(v) for k, v in data.items()}
    order = np.argsort(output["episode"], kind="stable")
    output = {k: v[order] for k, v in output.items()}
    output["win"] = engine.winners == designated
    output["random"] = is_random
    output["temperature"] = temperature
    output["epsilon"] = epsilon
    output["seed"] = seed
    return output


def collect(args):
    policy = collection_policy(args.orbit)
    args.directory.mkdir(parents=True, exist_ok=True)
    for index in range(args.start, args.start + args.chunks):
        path = args.directory / f"chunk_{index:04d}.npz"
        if path.exists():
            raise ValueError(f"shard exists: {path}")
        started = time.monotonic()
        players = 2 + index % 3
        seed = args.seed + index * args.games
        data = collect_chunk(policy, players, args.games, seed, args.temperature, args.epsilon)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **data)
        temporary.replace(path)
        print(json.dumps(dict(shard=index, players=players, seed=seed, games=args.games,
                              decisions=len(data["act"]), elapsed=round(time.monotonic()-started,2))),
              flush=True)


ORDINARY = (2, 3, 4, 5, 8, 9, 10, 11)
SPECIAL = (0, 1, 6, 7)
ROOT = Path(__file__).resolve().parent


def public_features(obs, seen, value, pairs):
    """Invariant summaries of the actor's hand and public board for each action.

    The first two axes of 'cards' are suit and printed rank. Ordinary ranks
    and suits have no absolute identifiers in the learned summaries.
    """
    cards = obs[:, :48].reshape(-1, 4, 12)
    played = seen[:, :48].reshape(-1, 4, 12)
    specials = cards[:, :, SPECIAL]
    ordinary = cards[:, :, ORDINARY]
    suit_n = cards.sum(2)
    rank_n = cards.sum(1)
    ordinary_n = ordinary.sum(2)
    top_suit = obs[:, 50:55].argmax(1)
    top_rank = obs[:, 55:69].argmax(1)
    current = top_suit.clamp(max=3)
    row = torch.arange(len(obs), device=obs.device)
    own_n = obs[:, 91]
    n = obs[:, 92]
    next_n = obs[:, 70]
    prev_n = obs[row, 69 + n.long().clamp(2, 13) - 1]
    opponents = obs[:, 70:82].masked_fill(obs[:, 70:82] == 0, 99)
    min_n = opponents.amin(1)
    wild = obs[:, 48:50]
    sorted_suits = suit_n.sort(1).values
    ordinary_mult = rank_n[:, ORDINARY]
    same_top_suit = (top_suit < 4).float()
    regular_top = (top_rank >= 1) & (top_rank <= 12)
    top_rank_hand = rank_n[row, (top_rank - 1).clamp(0, 11)] * regular_top
    state = torch.cat([
        torch.stack([own_n.clamp(max=22)/8, next_n.clamp(max=22)/8, prev_n.clamp(max=22)/8,
                     min_n.clamp(max=22)/8, 1/own_n.clamp_min(1), 1/next_n.clamp_min(1),
                     1/min_n.clamp_min(1), n/4, obs[:, 90].clamp(max=12)/5, obs[:, 93],
                     value.flatten().clamp(-3, 3), same_top_suit * suit_n[row, current]/5,
                     top_rank_hand/3, (ordinary_mult > 0).sum(1)/5,
                     (ordinary_mult >= 2).sum(1)/3, ordinary_n.sum(1)/8,
                     (ordinary_n > 0).sum(1)/4, (suit_n > 0).sum(1)/4,
                     seen.sum(1)/35], 1),
        sorted_suits/5, specials.sum(1)/3, wild/3, (n[:, None] == obs.new_tensor([2,3,4])),
        (top_rank[:, None] == obs.new_tensor([0,1,2,7,8,13])),
    ], 1)

    rows, act = pairs
    card = act < 48
    suited = (act >= 50) & (act < 54)
    has_suit = card | suited
    suit = torch.where(suited, act-50, act//12).clamp(0, 3)
    rank = act.remainder(12)
    category = torch.zeros_like(act)
    for i, r in enumerate(SPECIAL, start=1):
        category = torch.where(card & (rank == r), i, category)
    category = torch.where(act == 48, 5, category)  # suit-changing jack
    category = torch.where(act == 49, 6, category)  # joker
    category = torch.where(suited, 7, category)
    category = torch.where(act == 54, 8, category)
    this_suit = cards[rows, suit]
    other_rank = rank_n[rows,rank] - cards[rows,suit,rank]
    suit_specials = this_suit[:, SPECIAL] * has_suit[:,None]
    same_suit_total = (suit_n[rows,suit] - card.float()) * has_suit
    ordinary_card = category == 0
    same_suit_ordinary = (ordinary_n[rows,suit] - ordinary_card.float()) * has_suit
    remaining_ordinary = ordinary_n[rows].sum(1) - ordinary_card.float()
    seen_rank = played[rows, :, rank].sum(1)
    seen_suit = played[rows,suit].sum(1)
    tempo = (category == 3) | ((category == 4) & (n[rows] == 2))
    tempo_follow = (suit_specials[:,2] + suit_specials[:,3] * (n[rows] == 2) - tempo.to(obs.dtype))
    action_features = torch.cat([
        nn.functional.one_hot(category, 9).to(obs.dtype), suit_specials,
        torch.stack([
            same_suit_total/5, same_suit_ordinary/4, (same_suit_total == 0) & has_suit,
            (same_suit_ordinary == 0) & has_suit, (remaining_ordinary == 0),
            (remaining_ordinary == 1), card * other_rank/3, card * (other_rank > 0),
            same_top_suit[rows] * (suit == top_suit[rows]) * has_suit,
            card * (rank+1 == top_rank[rows]), suit_n[rows].amax(1)/5 - same_suit_total/5,
            has_suit * seen_suit/9, card * seen_rank/3, tempo, has_suit * tempo_follow/2,
            (category == 1) * (prev_n[rows] - next_n[rows]).clamp(-7,7)/7,
            has_suit * obs[rows,82+suit].clamp(max=40)/20,
        ], 1)
    ], 1)
    return state, action_features


class OutcomeHead(nn.Module):
    def __init__(self, state_width, action_width):
        super().__init__()
        self.state = nn.Linear(state_width, 48)
        self.action = nn.Linear(action_width, 48, bias=False)
        self.output = nn.Sequential(nn.Tanh(), nn.Linear(48, 32), nn.Tanh(), nn.Linear(32, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def scores(self, state, actions, rows):
        return self.output(self.state(state)[rows] + self.action(actions)).flatten()


class OutcomeResidualCore(PublicHistoryCore):
    """Teacher with a fitted action advantage; legal masking stays with the caller."""

    def __init__(self, core, head, scale):
        super().__init__(core)
        self.head = head
        self.scale = scale

    def decision_logits(self, tokens, counts, logits, value):
        decision = tokens[:, 99:100] != 0
        held = tokens[:, :50] > 0
        special = torch.ones_like(tokens[:, 50:55], dtype=torch.bool)
        rows, act = torch.nonzero(decision & torch.cat((held,special), dim=1), as_tuple=True)
        state, features = public_features(tokens, counts, value, (rows,act))
        correction = torch.zeros_like(logits)
        correction[rows, act] = self.head.scores(state, features, rows)
        return logits + self.scale * correction


def file_hash(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_candidate(path, device="cpu"):
    """Load a deterministic recurrent candidate accepted by temporal_agent_factory."""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle["kind"] == "pesten_public_branch_outcome_v1":
        from branch_outcome_policy import from_bundle
        return from_bundle(bundle, device=device)
    if bundle["kind"] != "pesten_public_outcome_residual_v1":
        raise ValueError("unrecognized outcome residual checkpoint")
    base = ROOT / bundle["base"]
    if file_hash(base) != bundle["base_sha256"]:
        raise ValueError(f"base actor hash mismatch: {base}")
    policy = teacher(1, device=device, path=base)
    if bundle["orbit"] == "rank-suit":
        from rank_suit_orbit import RankSuitOrbitCore
        parent_core = RankSuitOrbitCore(policy.puffer_policy)
    elif bundle["orbit"] == "suit24":
        parent_core = NativeSuitOrbitCore(policy.puffer_policy)
    else:
        raise ValueError("unknown orbit configuration")
    head = OutcomeHead(*bundle["dimensions"])
    head.load_state_dict(bundle["head"])
    policy.puffer_policy = OutcomeResidualCore(parent_core, head, bundle["scale"])
    return policy.to(device).eval().requires_grad_(False)


def public_training_shard(path):
    with np.load(path) as d:
        episode = torch.from_numpy(d["episode"].copy()).long()
        obs = torch.from_numpy(d["obs"].astype(np.float32))
        seen = torch.from_numpy(d["seen"].astype(np.float32))
        value = torch.from_numpy(d["value"].copy())
        actions = torch.from_numpy(d["act"].astype(np.int64))
        legal = torch.from_numpy(np.unpackbits(d["legal"],axis=1,bitorder="little")[:, :55].astype(bool))
        logits = torch.from_numpy(d["logits"].astype(np.float32)).masked_fill(~legal, -torch.inf)
        temperature, epsilon = float(d["temperature"]), float(d["epsilon"])
        probs = (1-epsilon) * (logits/temperature).softmax(1) + epsilon*legal/legal.sum(1,keepdim=True)
        win = torch.from_numpy(d["win"].copy()).float()[episode]
        opponent = torch.from_numpy(d["random"].copy()).float()[episode]
        seed = int(d["seed"])
        n_games = len(d["win"])
    rows, act = legal.nonzero(as_tuple=True)
    state, features = public_features(obs, seen, value, (rows,act))
    terminal = torch.ones(len(state), dtype=torch.bool)
    terminal[:-1] = episode[1:] != episode[:-1]
    # Only legal action features are retained. Compact contiguous slices
    # make large, randomized training minibatches inexpensive.
    width = legal.sum(1)
    return dict(state=state, features=features, width=width,
                chosen=(actions[rows]==act).float(), prob=probs[rows, act],
                win=win, opponent=opponent, terminal=terminal, seed=seed, games=n_games)


def chunk_minibatches(n, size, rng):
    starts = np.arange(0, n, size)
    rng.shuffle(starts)
    return [(i,min(i+size,n)) for i in starts]


def expected_scores(head, d, start, stop):
    w = d["width"][start:stop]
    first, last = d["offset"][start], d["offset"][stop]
    rows = torch.repeat_interleave(torch.arange(stop-start), w, output_size=last-first)
    out = head.scores(d["state"][start:stop], d["features"][first:last], rows)
    treatment = d["chosen"][first:last]-d["prob"][first:last]
    effect = torch.zeros(stop-start).scatter_add_(0, rows, out*treatment)
    return effect


def fit(args):
    torch.manual_seed(args.fit_seed)
    rng = np.random.default_rng(args.fit_seed)
    shards = sorted(args.directory.glob("chunk_*.npz"))[:args.chunks]
    if len(shards) != args.chunks:
        raise ValueError("Not all requested training shards exist")
    data = [public_training_shard(p) for p in shards]
    games = sum(d["games"] for d in data)
    for d in data:
        d["offset"] = torch.cat([torch.zeros(1,dtype=torch.long),d["width"].cumsum(0)]).numpy()
    # Entire deal shards stay together; each split contains all three table sizes.
    train = [d for i,d in enumerate(data) if i%18 >= 3]
    valid = [d for i,d in enumerate(data) if i%18 < 3]
    if not train or not valid:
        raise ValueError("need at least four deal shards")
    dims = (data[0]["state"].shape[1], data[0]["features"].shape[1])
    value_net = nn.Sequential(nn.Linear(dims[0]+1,64), nn.Tanh(), nn.Linear(64,48), nn.Tanh(),
                              nn.Linear(48,1),nn.Sigmoid())
    def value(d):
        return value_net(torch.cat((d["state"],d["opponent"][:,None]),1)).flatten()
    optimizer = torch.optim.AdamW(value_net.parameters(),lr=.002)
    hist = dict(value_mse=[], advantage_mse=[])
    for epoch in range(args.value_epochs):
        rng.shuffle(train)
        for d in train:
            v_input = torch.cat((d["state"],d["opponent"][:,None]),1)
            for a,b in chunk_minibatches(len(v_input), 1024, rng):
                # Shuffle contiguous deal blocks across shards, preserving the order only
                # for the later TD return calculation.
                pred = value_net(v_input[a:b]).flatten()
                loss = (pred-d["win"][a:b]).square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            hist["value_mse"].append(float(torch.cat([(value(d)-d["win"]).square() for d in valid]).mean()))
    with torch.no_grad():
        for d in data:
            v = value(d)
            # Finite episodes, gamma=1. Build TD(lambda) in reverse from the binary win.
            values = v.numpy()
            targets = np.empty_like(values)
            y, end = d["win"].numpy(), d["terminal"].numpy()
            for i in range(len(v)-1, -1, -1):
                targets[i] = y[i] if end[i] else (1-args.td_lambda)*values[i+1]+args.td_lambda*targets[i+1]
            d["adv"] = torch.from_numpy(targets) - v
    head = OutcomeHead(*dims)
    optimizer = torch.optim.AdamW(head.parameters(),lr=args.learning_rate)
    best = None
    best_mse = float("inf")
    for epoch in range(args.head_epochs):
        rng.shuffle(train)
        for d in train:
            for a,b in chunk_minibatches(len(d["state"]),1024,rng):
                effect = expected_scores(head,d,a,b)
                informed = d["width"][a:b] > 1
                if not informed.any():
                    continue
                loss = (effect[informed]-d["adv"][a:b][informed]).square().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(),1.)
                optimizer.step()
        with torch.no_grad():
            err = []
            for d in valid:
                for a,b in chunk_minibatches(len(d["state"]),1024,rng):
                    effect = expected_scores(head,d,a,b)
                    err.append((effect-d["adv"][a:b]).square()[d["width"][a:b]>1])
            mse = float(torch.cat(err).mean())
        hist["advantage_mse"].append(mse)
        if mse < best_mse:
            best_mse, best = mse, {name:v.detach().clone() for name,v in head.state_dict().items()}
        print(f"head epoch {epoch+1} validation MSE {mse:.7f}",flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    base = Path(args.base).resolve()
    config = dict(kind="pesten_public_outcome_residual_v1",base=str(base.relative_to(ROOT)),
                  base_sha256=file_hash(base),dimensions=dims,head=best)
    for scale in args.scales:
        for orbit in args.orbits:
            path = args.output / f"residual_{orbit}_{scale:g}.pt"
            torch.save(dict(config,scale=scale,orbit=orbit),path)
    report = dict(games=games, decisions=sum(len(d["state"]) for d in data),
                  train_games=sum(d["games"] for d in train), validation_games=sum(d["games"] for d in valid),
                  first_seed=min(d["seed"] for d in data), last_seed=max(d["seed"]+d["games"]-1 for d in data),
                  baseline_mse=float(torch.cat([d["adv"][d["width"]>1].square() for d in valid]).mean()),
                  dimensions=dims, arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
                                              if k!="func"}, history=hist)
    (args.output/"fit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    roll = sub.add_parser("collect")
    roll.add_argument("--directory", type=Path, required=True)
    roll.add_argument("--seed", type=int, default=510_000_000)
    roll.add_argument("--games", type=int, default=384)
    roll.add_argument("--start", type=int, default=0)
    roll.add_argument("--chunks", type=int, default=24)
    roll.add_argument("--temperature", type=float, default=2)
    roll.add_argument("--epsilon", type=float, default=.04)
    roll.add_argument("--orbit", type=int, choices=(1,4,24), default=1)
    roll.set_defaults(func=collect)
    trainer = sub.add_parser("fit")
    trainer.add_argument("--directory", type=Path, required=True)
    trainer.add_argument("--output", type=Path, required=True)
    trainer.add_argument("--base", type=Path, default=TEACHER)
    trainer.add_argument("--chunks", type=int, default=72)
    trainer.add_argument("--fit-seed", type=int, default=513_000_001)
    trainer.add_argument("--value-epochs", type=int, default=7)
    trainer.add_argument("--head-epochs", type=int, default=8)
    trainer.add_argument("--td-lambda", type=float, default=.65)
    trainer.add_argument("--learning-rate", type=float, default=.001)
    trainer.add_argument("--scales", type=float, nargs="+", default=[8.,24.,48.])
    trainer.add_argument("--orbits", choices=("suit24","rank-suit"), nargs="+", default=["suit24"])
    trainer.set_defaults(func=fit)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.func(args)


if __name__ == "__main__":
    main()
