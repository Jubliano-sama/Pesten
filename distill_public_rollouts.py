"""Fit a public action-value head to Monte Carlo rollouts on exploratory trajectories."""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from public_lookahead import HISTORY_SIZE, RECENT_PLAYS, _batch_values
from train_outcome_residual import ROOT, TEACHER, OutcomeHead, file_hash, public_features


def discard_history(seen):
    """Approximate recent public discards from the collection's decayed counts."""
    recent = np.zeros((len(seen), HISTORY_SIZE), dtype=np.float32)
    order = np.argsort(seen, axis=1, kind="stable")[:, -RECENT_PLAYS:]
    recent[:, :RECENT_PLAYS] = np.where(np.take_along_axis(seen, order, axis=1) > .15, order + 1, 0)
    return recent


def label_shard(task):
    path, states, simulations, horizon, seed = task
    with np.load(path) as shard:
        rng = np.random.default_rng(seed + int(shard["seed"]))
        legal = np.unpackbits(shard["legal"], axis=1, bitorder="little")[:, :55].astype(bool)
        possible = np.flatnonzero(legal.sum(1) >= 2)
        chosen = rng.choice(possible, size=min(states, len(possible)), replace=False)
        obs = shard["obs"][chosen].astype(np.float32)
        seen = shard["seen"][chosen].astype(np.float32)
        value = shard["value"][chosen].astype(np.float32)
        legal = legal[chosen]
    tokens = np.zeros((len(obs), 100), dtype=np.float32)
    tokens[:, :94] = obs
    tokens[:, 99] = 1
    tokens[:, 98] = (obs[:, 93] == 0) & (legal[:, :50].sum(1) < 2)
    scores = _batch_values(tokens, discard_history(seen), simulations, horizon, False)
    rows, actions = np.nonzero(legal)
    state, features = public_features(torch.from_numpy(obs), torch.from_numpy(seen),
                                      torch.from_numpy(value), (torch.from_numpy(rows), torch.from_numpy(actions)))
    return state.numpy()[rows], features.numpy(), scores[rows, actions], len(obs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=72)
    parser.add_argument("--states-per-chunk", type=int, default=768)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=622106333)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if min(args.chunks, args.states_per_chunk, args.simulations, args.horizon,
           args.steps, args.workers) < 1 or args.chunks < 6:
        parser.error("use positive training parameters and at least six deal shards")
    paths = sorted(args.rollouts.glob("chunk_*.npz"))[:args.chunks]
    if len(paths) != args.chunks:
        parser.error("missing collection shards")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    tasks = [(p, args.states_per_chunk, args.simulations, args.horizon, args.seed) for p in paths]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        data = list(pool.map(label_shard, tasks))
    # Hold out entire contiguous deals in every format, never individual moves.
    def split(validation):
        group = [d for i, d in enumerate(data) if (i % 18 < 3) == validation]
        return [torch.from_numpy(np.concatenate([row[k] for row in group])) for k in range(3)]
    state, actions, target = split(False)
    v_state, v_actions, v_target = split(True)
    dims = (state.shape[1], actions.shape[1])
    head = OutcomeHead(*dims)
    optim = torch.optim.AdamW(head.parameters(), lr=.002)
    best_loss = float("inf")
    rows = torch.arange(1024)
    valid_history = []
    for step in range(1, args.steps + 1):
        indices = torch.randint(len(state), (len(rows),))
        estimate = head.scores(state[indices], actions[indices], rows)
        loss = (estimate - target[indices]).square().mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 100 == 0 or step == args.steps:
            with torch.no_grad():
                estimate = head.scores(v_state, v_actions, torch.arange(len(v_state)))
                validation = float((estimate - v_target).square().mean())
            valid_history.append([step, validation])
            if validation < best_loss:
                best_loss = validation
                weights = {k:v.clone() for k,v in head.state_dict().items()}
    for scale in (15, 35, 65):
        torch.save({"kind": "pesten_public_outcome_residual_v1", "head": weights, "dimensions": dims,
                    "scale": scale, "orbit": "suit24", "base": str(TEACHER.resolve().relative_to(ROOT)),
                    "base_sha256": file_hash(TEACHER)}, args.output / f"head_{scale}.pt")
    report = {"settings": vars(args), "states": sum(d[-1] for d in data), "train_actions": len(target),
              "validation_actions": len(v_target), "zero_residual_validation_mse": float(v_target.square().mean()),
              "best_validation_mse": best_loss, "validation_mse": valid_history,
              "targets": "public sampled model rollout action value minus mean of modeled card/suit alternatives; pass neutral",
              "data": {p.name: file_hash(p) for p in paths}}
    (args.output / "fit.json").write_text(json.dumps(report, default=str, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("states", "train_actions", "best_validation_mse")}), flush=True)


if __name__ == "__main__":
    main()
