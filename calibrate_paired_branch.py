"""Calibrate a rollout student's last layer against matched real game outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from branch_outcome_policy import BranchActionHead
from train_outcome_residual import file_hash


@torch.inference_mode()
def design(paths, bundle):
    head = BranchActionHead(
        graph=bundle["graph"],
        style_heads="style" in bundle,
        frequencies=bundle["head"].get("frequencies"),
    )
    head.load_state_dict(bundle["head"])
    x, delta, gap, groups, validation = [], [], [], [], []
    for path in paths:
        with np.load(path) as d:
            obs = torch.from_numpy(d["obs"].astype(np.float32))
            if not len(obs):
                continue
            seen = torch.from_numpy(d["seen"].astype(np.float32))
            value = torch.from_numpy(d["value"])
            index = torch.arange(len(obs))
            tracking = (
                torch.from_numpy(d["tracking"].astype(np.float32))
                if head.frequencies is not None
                else None
            )
            a = head.embeddings(obs, seen, value, index, torch.from_numpy(d["a"]), tracking)
            b = head.embeddings(obs, seen, value, index, torch.from_numpy(d["b"]), tracking)
            x.append((b - a).numpy())
            delta.append(d["delta"])
            gap.append(d["gap"])
            groups.append(int(d["players"]) - 2 + 3 * d["random"].astype(np.int64))
            shard = int(path.stem.split("_")[-1])
            validation.append(np.full(len(obs), shard % 18 < 3))
    return tuple(np.concatenate(v) for v in (x, delta, gap, groups, validation))


def ridge(x, residual, penalty):
    gram = x.T @ x
    diag = gram.diagonal().copy()
    gram.flat[:: len(gram) + 1] += penalty * np.maximum(diag, max(0.02 * diag.mean(), 1e-6))
    return np.linalg.solve(gram, x.T @ residual)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--chunks", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prior", type=float, default=0.004)
    p.add_argument("--scales", type=float, nargs="+", default=[40, 90])
    args = p.parse_args()
    torch.set_num_threads(1)
    bundle = torch.load(args.base, map_location="cpu", weights_only=True)
    files = sorted(args.directory.glob("pair_*.npz"))[: args.chunks]
    if len(files) != args.chunks:
        raise ValueError("missing matched-pair inputs")
    x, delta, gap, cell, valid = design(files, bundle)
    x = x.astype(np.float64)
    w = bundle["head"]["output.3.weight"].numpy()
    role = cell // 3 if len(w) == 2 else np.zeros(len(x), np.int64)
    initial = (x * w[role]).sum(1)
    target = delta - args.prior * gap
    residual = target - initial
    train = ~valid
    if not train.any() or not valid.any():
        raise ValueError("both fit and validation need full deal shards")

    def metric(error):
        return np.mean([np.mean(error[(cell == i) & valid] ** 2) for i in range(6)])

    baseline = float(metric(residual))
    options = []
    best_mse = baseline
    offsets = np.zeros((6, x.shape[1]))
    for penalty in (0.1, 0.4, 1.5, 6.0):
        global_offset = ridge(x[train], residual[train], penalty)
        candidate = []
        for i in range(6):
            rows = train & (cell == i)
            local = ridge(x[rows], residual[rows] - x[rows] @ global_offset, penalty * 3)
            candidate.append(global_offset + local)
        candidate = np.asarray(candidate)
        pred = (x * candidate[cell]).sum(1)
        mse = float(metric(residual - pred))
        options.append(dict(penalty=penalty, mse=mse))
        if mse < best_mse:
            best_mse = mse
            offsets = candidate
    table = offsets.reshape(2, 3, -1).transpose(1, 0, 2)
    if len(w) == 1:
        table = table.mean(1, keepdims=True)
    bundle["head"]["table_offsets"] = torch.from_numpy(table.copy()).float()
    args.output.mkdir(parents=True, exist_ok=True)
    for scale in args.scales:
        torch.save({**bundle, "scale": scale}, args.output / f"calibrated_{scale:g}.pt")
    info = dict(
        roots=len(x),
        training=int(train.sum()),
        validation=int(valid.sum()),
        uncalibrated_mse=baseline,
        calibrated_mse=best_mse,
        choices=options,
        fit_source=file_hash(args.base),
        table_offset_norm=float(np.linalg.norm(table)),
        prior=args.prior,
    )
    (args.output / "fit.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
