"""Check leaf probabilities and time prospective searches on real public decision inputs."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch

from batched_pesten import CARD_RANK, CARD_SORT
from public_lookahead import HISTORY_SIZE, _batch_values
from refine_leaf_contrasts import restore_net
from train_public_leaf import load_training_data


def public_positions(directory, count, seed):
    rng = np.random.default_rng(seed)
    by_size = {n: [] for n in (2, 3, 4)}
    for file in sorted(directory.glob("shard_*.npz")):
        with np.load(file) as d:
            n = d["sizes"].shape[1]
            if len(by_size[n]) >= count:
                continue
            ids = np.flatnonzero(d["seat"] == 0)
            rng.shuffle(ids)
            for k in ids:
                if len(by_size[n]) == count:
                    break
                hand = d["hands"][k, 0, :d["sizes"][k, 0]]
                suit, rank, penalty = int(d["suit"][k]), int(d["rank"][k]), int(d["penalty"][k])
                playable = CARD_RANK[hand] == rank
                if penalty == 0:
                    playable |= (CARD_SORT[hand] == suit) | (hand >= 48)
                if len(np.unique(hand[playable])) < 2:
                    continue
                token = np.zeros(100, np.float32)
                token[:50] = np.bincount(hand, minlength=50)
                token[50+suit] = token[55+rank] = 1
                token[69:69+n] = d["sizes"][k, (np.arange(n)*int(d["direction"][k]))%n]
                token[89], token[90], token[91], token[92], token[99] = d["direction"][k], penalty, len(hand), n, 1
                by_size[n].append(token)
    return np.vstack([np.stack(by_size[n]) for n in (2,3,4)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--shards",type=int,default=60)
    p.add_argument("--positions",type=int,default=128)
    args=p.parse_args()
    torch.set_num_threads(1)
    state, _ = load_training_data([args.data / f"shard_{i:04d}.npz" for i in range(args.shards)])
    f,mask,y,valid=state
    f,mask,y=f[valid],mask[valid],y[valid].long()
    with torch.inference_mode():
        logits=restore_net(args.weights)(f,mask)
        sizes = torch.expm1(f[:,:,1] * 2).round()
        heuristic=(-.4*(sizes + 2.5*f[:,:,7] - .6*f[:,:,31])).masked_fill(~mask,-1e9)
        decisive = y!=4
        target=y.eq(0).float()
        value={"states":len(y), "decisive":int(decisive.sum()),
               "heuristic_decisive_nll":float(torch.nn.functional.cross_entropy(heuristic[decisive],y[decisive])),
               "leaf_decisive_nll":float(torch.nn.functional.cross_entropy(logits[decisive,:4],y[decisive])),
               "heuristic_focal_brier_all":float((heuristic.softmax(1)[:,0]-target).square().mean()),
               "leaf_focal_brier_all":float((logits.softmax(1)[:,0]-target).square().mean())}
    tokens=public_positions(args.data,args.positions,1140000107)
    hist=np.zeros((len(tokens),HISTORY_SIZE),np.float32)
    w=np.load(args.weights)
    empty=np.empty(0,np.float64)
    arms=[("h4_leaf",64,4,w),("h14_leaf",64,14,w),("h32_leaf",80,32,w),
          ("h32_heuristic",80,32,empty),("h80_heuristic",128,80,empty)]
    result={name:[] for name,*_ in arms}
    for _,s,h,v in arms:
        _batch_values(tokens[:2],hist[:2],s,h,False,v)
    for repeat in range(3):
        for name,s,h,v in (arms if repeat%2==0 else arms[::-1]):
            t=time.perf_counter()
            vals=_batch_values(tokens,hist,s,h,False,v)
            elapsed=time.perf_counter()-t
            assert np.isfinite(vals).all()
            result[name].append(elapsed*1000/len(tokens))
    data={"validation":value,"positions_per_format":args.positions,
          "scope":"Monte Carlo planner only, multi-card public decisions, uniform determinizations, one CPU thread, compiled Numba; exclude teacher inference",
          "milliseconds_per_decision": result}
    args.out.write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps(data),flush=True)


if __name__=="__main__":
    main()
