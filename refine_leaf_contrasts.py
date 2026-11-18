"""Refine an outcome-trained search value with exact, paired action interventions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from batched_pesten import PHASE_CARD
from paired_outcome_residual import paired_games
from public_lookahead_leaf import FEATURES, HIDDEN, batch_features
from train_outcome_residual import collection_policy, file_hash
from train_public_leaf import LeafNet, load_training_data


class PairedAfterstates:
    """Record just one fully resolved afterstate per arm of each paired intervention."""

    def __init__(self, pairs):
        self.recorded = np.zeros(2*pairs, bool)
        self.terminal = np.zeros(2*pairs, bool)
        self.features = np.zeros((2*pairs, 4, FEATURES), np.float16)
        self.focal_wins = np.zeros(2*pairs, bool)

    def __call__(self, engine, changed, focal, random_others):
        finished = engine.game_over | (engine.turn_counts >= engine.max_turns)
        self.focal_wins[:] = engine.winners == np.repeat(focal, 2)
        ready = np.repeat(changed, 2) & ~self.recorded & (finished | (engine.phase == PHASE_CARD))
        rows = np.flatnonzero(ready)
        self.recorded[rows] = True
        self.terminal[rows] = finished[rows]
        alive = rows[~finished[rows]]
        if not len(alive):
            return
        strong = focal[alive // 2]
        order = (strong[:, None] + np.arange(engine.num_players)) % engine.num_players
        self.features[alive] = batch_features(
            engine.hands[alive[:,None],order], engine.hand_sizes[alive[:,None],order],
            ((engine.current_player[alive]-strong)%engine.num_players).astype(np.int64),
            engine.direction[alive].astype(np.int64), engine.current_sort[alive].astype(np.int64),
            engine.current_rank[alive].astype(np.int64), engine.penalty_amount[alive].astype(np.int64),
            random_others[alive//2])


def collect(args):
    policy = collection_policy(args.orbit)
    args.data.mkdir(parents=True, exist_ok=True)
    for i in range(args.start, args.start+args.chunks):
        path = args.data / f"contrast_{i:04d}.npz"
        if path.exists():
            raise FileExistsError(path)
        n = 2+i%3
        seed = args.seed + i*args.pairs
        receiver = PairedAfterstates(args.pairs)
        result = paired_games(policy, n, args.pairs, seed, after_actions=receiver)
        ids = result["root_seed"] - seed
        if not receiver.recorded.reshape(-1,2)[ids].all():
            raise ValueError("missing one arm of an intervention")
        win = receiver.focal_wins.reshape(-1,2)[ids]
        if not np.array_equal(win[:,1].astype(float)-win[:,0],result["delta"]):
            raise ValueError("focal outcomes differ from the independent paired evaluator")
        with path.open("xb") as handle:
            np.savez_compressed(handle, state=receiver.features.reshape(-1,2,4,FEATURES)[ids],
                                terminal=receiver.terminal.reshape(-1,2)[ids], win=win,
                                deals=result["root_seed"], players=n, seed=seed, orbit=args.orbit,
                                pairs=args.pairs)
        print(i, n, len(ids), np.count_nonzero(result["delta"]), flush=True)


def restore_net(path):
    data = np.load(path,allow_pickle=False).astype(np.float32)
    model = LeafNet(FEATURES,HIDDEN)
    offset = FEATURES + FEATURES*HIDDEN
    with torch.no_grad():
        model.linear.weight.copy_(torch.from_numpy(data[:FEATURES][None,:]))
        model.hidden.weight.copy_(torch.from_numpy(data[FEATURES:offset].reshape(FEATURES,HIDDEN).T))
        model.hidden.bias.copy_(torch.from_numpy(data[offset:offset+HIDDEN]))
        model.out.weight.copy_(torch.from_numpy(data[offset+HIDDEN:-2][None,:]))
        model.out.bias.fill_(data[-2])
        model.draw.fill_(data[-1])
    return model


def fit(args):
    args.out.mkdir(exist_ok=True,parents=True)
    if (args.out/"leaf_weights.npy").exists() or (args.out/"fit.json").exists():
        raise FileExistsError(f"fit already exists in {args.out}")
    torch.manual_seed(args.seed)
    general, _ = load_training_data([args.states/f"shard_{i:04d}.npz" for i in range(args.state_chunks)])
    general = [x.to(args.device) for x in general]
    gf, gm, gy, gv = general
    train_general = torch.where(~gv)[0]
    grouped = []
    pairs_generated = 0
    sources = {}
    for i in range(args.chunks):
        path = args.data/f"contrast_{i:04d}.npz"
        with np.load(path) as d:
            pairs_generated += int(d["pairs"])
            sources[path.name] = file_hash(path)
            grouped.append((d["state"].astype(np.float32),d["terminal"],d["win"],
                            np.broadcast_to(np.arange(4)<int(d["players"]),(len(d["state"]),4)),
                            np.full(len(d["state"]),i%18<3)))
    feats, terminal, win, mask, valid = [torch.from_numpy(np.concatenate(parts)).to(args.device) for parts in zip(*grouped)]
    train, val = torch.where(~valid)[0], torch.where(valid)[0]
    def differences(model,ids):
        views = model(feats[ids].flatten(0,1),mask[ids].repeat_interleave(2,dim=0)).softmax(1)[:,0].reshape(-1,2)
        prediction = torch.where(terminal[ids],win[ids].float(),views)
        return prediction[:,1]-prediction[:,0]
    model = restore_net(args.base).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    delta = win[:,1].float() - win[:,0].float()
    best_loss = float("inf")
    best = None
    history = []
    for epoch in range(args.epochs+1):
        if epoch:
            for ids in train[torch.randperm(len(train),device=args.device)].split(args.batch_size):
                idx = train_general[torch.randint(len(train_general),(len(ids),),device=args.device)]
                aux = nn.functional.cross_entropy(model(gf[idx],gm[idx]),gy[idx].long())
                loss = (differences(model,ids) - delta[ids]).square().mean() + args.auxiliary*aux
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.inference_mode():
            err = float((differences(model,val)-delta[val]).square().mean())
            general_nll = float(nn.functional.cross_entropy(model(gf[gv],gm[gv]),gy[gv].long()))
        history.append({"epoch":epoch,"paired_validation_mse":err,"winner_validation_nll":general_nll})
        if err < best_loss:
            best_loss,best = err,model.pack().copy()
    np.save(args.out/"leaf_weights.npy",best,allow_pickle=False)
    result={"pairs_generated":pairs_generated,"scored_pairs":len(feats),"training_pairs":len(train),"validation_pairs":len(val),
            "base_sha256":file_hash(args.base),"weights_sha256":file_hash(args.out/"leaf_weights.npy"),
            "history":history,"chosen_epoch":min(history,key=lambda r:r["paired_validation_mse"]),
            "seed":args.seed,"auxiliary":args.auxiliary,"lr":args.lr,"sources":sources}
    (args.out/"fit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("pairs_generated","scored_pairs","chosen_epoch","weights_sha256")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="operation",required=True)
    col=subs.add_parser("collect")
    col.add_argument("--data",type=Path,required=True)
    col.add_argument("--start",type=int,default=0)
    col.add_argument("--chunks",type=int,default=60)
    col.add_argument("--pairs",type=int,default=384)
    col.add_argument("--seed",type=int,default=1_112_000_000)
    col.add_argument("--orbit",type=int,choices=(1,4,24),default=1)
    col.set_defaults(func=collect)
    trainer=subs.add_parser("fit")
    trainer.add_argument("--data",type=Path,required=True)
    trainer.add_argument("--states",type=Path,required=True)
    trainer.add_argument("--base",type=Path,required=True)
    trainer.add_argument("--out",type=Path,required=True)
    trainer.add_argument("--state-chunks",type=int,default=84)
    trainer.add_argument("--chunks",type=int,required=True)
    trainer.add_argument("--seed",type=int,default=1_112_100_002)
    trainer.add_argument("--epochs",type=int,default=30)
    trainer.add_argument("--lr",type=float,default=.0007)
    trainer.add_argument("--auxiliary",type=float,default=.14)
    trainer.add_argument("--batch-size",type=int,default=512)
    trainer.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    trainer.set_defaults(func=fit)
    args=parser.parse_args()
    torch.set_num_threads(1)
    args.func(args)
