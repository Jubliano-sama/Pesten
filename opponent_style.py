"""Estimate opponent style from a focal player's sequence of public decision states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def style_features(obs):
    """Features do not include seat IDs, seeds, opponent identities, or hidden cards."""
    obs = np.asarray(obs, dtype=np.float32)
    ordinary = obs[:, :48].reshape(-1, 4, 12)
    sizes = obs[:, 69:73]
    top_rank = obs[:, 55:69].argmax(1)
    top_suit = obs[:, 50:55].argmax(1).clip(max=3)
    suit_counts = ordinary.sum(2)
    rank_counts = ordinary.sum(1)
    counts = obs[:, 82:87]
    row = np.arange(len(obs))
    return np.column_stack((
        sizes.clip(max=26) / 10, 1 / sizes.clip(min=1), sizes == 1,
        obs[:, 92, None] == (2, 3, 4), top_rank[:, None] == (0, 1, 2, 7, 8, 13),
        obs[:, 90].clip(max=15) / 5, obs[:, 93], obs[:, 48:50] / 3,
        np.sort(suit_counts, axis=1) / 6, rank_counts[:, (0, 1, 6, 7)] / 3,
        suit_counts[row, top_suit] / 6, rank_counts[row, (top_rank - 1).clip(0, 11)] / 3,
        counts.sum(1).clip(max=160) / 60, counts[:, 4].clip(max=40) / 15,
        np.sort(counts[:, :4], axis=1).clip(max=50) / 20,
        counts[row, top_suit].clip(max=50) / 20,
    )).astype(np.float32)


class OpponentStyle(nn.Module):
    def __init__(self, width, hidden_size=40):
        super().__init__()
        self.gru = nn.GRU(width, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, inputs, state=None):
        out, state = self.gru(inputs, state)
        return self.readout(out).squeeze(-1), state

    def initial_state(self, batch, device):
        return torch.zeros((1, batch, self.gru.hidden_size), device=device)

    def observe(self, tokens, state):
        inputs = torch.from_numpy(style_features(tokens[:, :94].detach().cpu().numpy())).to(state)
        logits, next_state = self(inputs.unsqueeze(1), state)
        decisions = tokens[:, 99] == 1
        next_state = torch.where(decisions[None, :, None], next_state, state)
        return logits[:, 0].sigmoid().detach().cpu().numpy(), next_state


def load_style(path, device="cpu"):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["kind"] != "pesten_public_opponent_style":
        raise ValueError("unexpected opponent style checkpoint")
    model = OpponentStyle(data["input_size"], data["hidden_size"])
    model.load_state_dict(data["state_dict"])
    return model.to(device).eval().requires_grad_(False)


def read_games(paths):
    features, lengths, labels = [], [], []
    for path in paths:
        with np.load(path) as data:
            ids = data["episode"]
            if np.any(ids[1:] < ids[:-1]):
                raise ValueError("expected decisions in game order")
            counts = np.bincount(ids, minlength=len(data["random"]))
            features.append(style_features(data["obs"]))
            lengths.extend(counts[counts > 0])
            labels.extend(data["random"][counts > 0])
    lens = np.asarray(lengths)
    return np.vstack(features), np.cumsum(lens) - lens, lens, np.asarray(labels, dtype=np.float32)


def batches(data, rng, device):
    features, starts, lengths, label = data
    # Shuffle within coarse length buckets to reduce padding, then shuffle batches.
    order = rng.permutation(len(lengths))
    order = order[np.argsort(lengths[order] // 8, kind="stable")]
    groups = list(np.array_split(order, max(1, (len(order) + 255) // 256)))
    rng.shuffle(groups)
    for idx in groups:
        steps = np.arange(lengths[idx].max())
        mask = steps < lengths[idx, None]
        address = starts[idx, None] + np.minimum(steps, lengths[idx, None] - 1)
        yield (torch.from_numpy(features[address]).to(device), torch.from_numpy(label[idx]).to(device),
               torch.from_numpy(mask).to(device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=631229811)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shards = sorted(args.directory.glob("chunk_*.npz"))
    if len(shards) < 6:
        raise ValueError("need at least six collection shards")
    train = read_games([p for i, p in enumerate(shards) if i % 18 >= 3])
    valid = read_games([p for i, p in enumerate(shards) if i % 18 < 3])
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = OpponentStyle(train[0].shape[1]).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=.002)
    best = float("inf")
    metrics = []
    for epoch in range(args.epochs):
        model.train()
        for inputs, label, mask in batches(train, rng, args.device):
            logits, _ = model(inputs)
            loss = nn.functional.binary_cross_entropy_with_logits(logits[mask], label[:, None].expand_as(logits)[mask])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            opt.step()
        model.eval()
        sums = np.zeros(3)
        with torch.inference_mode():
            for inputs, label, mask in batches(valid, rng, args.device):
                logits, _ = model(inputs)
                y = label[:, None].expand_as(logits)[mask]
                p = logits[mask].sigmoid()
                sums += [len(y), float(((p >= .5) == y.bool()).sum()), float((p - y).square().sum())]
        row = {"epoch": epoch + 1, "validation_accuracy": sums[1] / sums[0], "brier": sums[2] / sums[0]}
        metrics.append(row)
        if row["brier"] < best:
            best = row["brier"]
            torch.save({"kind": "pesten_public_opponent_style", "state_dict": model.state_dict(),
                        "input_size": train[0].shape[1], "hidden_size": model.gru.hidden_size},
                       args.output / "style.pt")
        print(json.dumps(row), flush=True)
    report = {"metrics": metrics, "seed": args.seed, "shards": [p.name for p in shards],
              "train_games": len(train[1]), "valid_games": len(valid[1]),
              "train_decisions": len(train[0]), "valid_decisions": len(valid[0]),
              "chance_brier": .25, "target": "all opposing seats are canonical random vs frozen teacher",
              "validation": "whole game shards i modulo 18 in 0, 1, 2"}
    (args.output / "fit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
