"""Fit a teacher-anchored action correction on matched single-move interventions."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from batched_pesten import BatchedPesten, PHASE_SUIT, POLICY_RANDOM, POLICY_RL
from temporal_batched_speed import SafeEventCursor, TemporalSeatRuntime, _batch_temporal_actions, build_safe_decision
from train_outcome_residual import OutcomeHead, TEACHER, collection_policy, file_hash, public_features


@torch.inference_mode()
def paired_games(policy, n, pairs, seed, after_actions=None):
    """Both copies share the deal and RNG streams until one public decision differs."""
    rng = np.random.default_rng(seed + 70_000_000)
    target = rng.integers(0, 8, size=pairs)
    focal = np.arange(pairs) % n
    vs_random = (np.arange(pairs) // n) % 2 == 0
    kinds = np.where(vs_random[:, None] & (np.arange(n)[None] != focal[:, None]),
                     POLICY_RANDOM, POLICY_RL).repeat(2, axis=0).astype(np.int8)
    seeds = np.arange(seed,seed+pairs,dtype=np.uint32).repeat(2)
    engine = BatchedPesten(kinds,np.full(kinds.shape,"policy",dtype=object),seeds,max_turns=1000)
    suits = [random.Random(int(s)) for s in seeds]
    runtimes = {(g,s): TemporalSeatRuntime(g,g,s,policy,torch.Generator(),policy.initial_state(),SafeEventCursor())
                for g in range(2*pairs) for s in range(n) if kinds[g,s] == POLICY_RL}
    changed = np.zeros(pairs, dtype=bool)
    records = []
    tracking = []
    track = hasattr(policy.puffer_policy, "tracks_public_beliefs")
    while len(active := engine.active_games()):
        entries, positions = [], []
        moves = np.empty(len(active),np.int64)
        for k,g in enumerate(active):
            seat = int(engine.current_player[g])
            if kinds[g,seat] == POLICY_RANDOM:
                moves[k] = 50+suits[g].randrange(4) if engine.phase[g] == PHASE_SUIT else engine.random_action(g)
            else:
                rt = runtimes[int(g),seat]
                entries.append((rt,build_safe_decision(engine,g,seat,rt.cursor)))
                positions.append(k)
        if entries:
            core = policy.puffer_policy
            greedy = _batch_temporal_actions(policy,entries,None,deterministic=True)
            moves[positions] = greedy
            logits = core.last_logits.numpy()
            for j,(rt,dec) in enumerate(entries):
                g = rt.game_index
                p = g // 2
                rt.trajectory.clear()
                rt.safe_history.clear()
                if g % 2 == 0 or changed[p] or rt.seat != focal[p]:
                    continue
                a = greedy[j]
                alternatives = np.flatnonzero(dec.legal_actions & (logits[j] >= logits[j,a]-5))
                alternatives = alternatives[alternatives != a]
                if len(alternatives) == 0:
                    continue
                if target[p] > 0:
                    target[p] -= 1
                    continue
                b = int(rng.choice(alternatives))
                # This is an intervention on the same action context, including its legal
                # mask on a draw retry. Validate equality against the untouched control.
                if engine.turn_counts[g] != engine.turn_counts[g-1] or not np.array_equal(engine.hands[g],engine.hands[g-1]):
                    raise RuntimeError("paired games diverged before the intervention")
                moves[positions[j]] = b
                changed[p] = True
                if track:
                    tracking.append(core.last_history[j].numpy().copy())
                records.append((p,dec.observation.values,core.last_counts[j].numpy().copy(),
                                float(core.last_value[j].item()),a,b,float(logits[j,b]-logits[j,a])))
        for g,m in zip(active,moves):
            engine.apply_action(int(g),int(m))
        if after_actions is not None:
            after_actions(engine, changed, focal, vs_random)
    # Full equality, including shared deck/random-agent RNG, for unperturbed pairs.
    result = engine.winners.reshape(-1,2)
    if (result[~changed,0] != result[~changed,1]).any():
        raise RuntimeError("identical policies produced unequal paired outcomes")
    if records:
        idx, obs, seen, value, a, b, gap = zip(*records)
    else:
        idx, obs, seen, value, a, b, gap = (), np.empty((0, 94)), np.empty((0, 50)), (), (), (), ()
    outcomes = result == focal[:,None]
    idx=np.asarray(idx,dtype=np.int64)
    data = dict(obs=np.asarray(obs,np.float16), seen=np.asarray(seen,np.float16),
                value=np.asarray(value,np.float32), a=np.asarray(a,np.int64), b=np.asarray(b,np.int64),
                gap=np.asarray(gap,np.float32), delta=outcomes[idx,1].astype(np.float32)-outcomes[idx,0],
                wins=outcomes.sum(0), pairs=pairs, root_seed=seed+idx, random=vs_random[idx],
                players=n, seed=seed)
    if track:
        from public_lookahead import HISTORY_SIZE
        data["tracking"] = np.asarray(tracking,np.float16).reshape(-1,HISTORY_SIZE)
    return data


def collect(args):
    p=collection_policy(args.orbit)
    if getattr(args, "beliefs", False):
        from branch_public_belief import TrackingCollector
        p.puffer_policy = TrackingCollector(p.puffer_policy)
    args.directory.mkdir(parents=True,exist_ok=True)
    for i in range(args.start,args.start+args.chunks):
        output=args.directory / f"pair_{i:04d}.npz"
        if output.exists():
            raise RuntimeError(f"shard exists: {output}")
        d=paired_games(p,2+i%3,args.pairs,args.seed+i*args.pairs)
        temp=output.with_suffix(".tmp.npz")
        np.savez_compressed(temp,**d)
        temp.replace(output)
        print(json.dumps(dict(shard=i,players=d["players"],pairs=args.pairs,roots=len(d["a"]),
                              changed_winners=int(np.count_nonzero(d["delta"])))),flush=True)


def fit(args):
    torch.manual_seed(args.fit_seed)
    data=[]
    games=0
    shards=sorted(args.directory.glob("pair_*.npz"))[:args.chunks]
    if len(shards)!=args.chunks:
        raise ValueError("missing paired-game shards")
    for i,path in enumerate(shards):
        with np.load(path) as d:
            games+=int(d["pairs"])*2
            if not len(d["a"]):
                continue
            obs=torch.from_numpy(d["obs"].astype(np.float32))
            seen=torch.from_numpy(d["seen"].astype(np.float32))
            value=torch.from_numpy(d["value"])
            actions=torch.from_numpy(np.stack([d["a"],d["b"]],1).reshape(-1))
            idx=torch.arange(len(obs)).repeat_interleave(2)
            state,features=public_features(obs,seen,value,(idx,actions))
            data.append((state,features.reshape(len(obs),2,-1),torch.from_numpy(d["delta"]),
                         torch.from_numpy(d["gap"]),torch.full((len(obs),),i%18<3)))
    if not data:
        raise ValueError("training and validation both need eligible interventions")
    state,features,delta,gap,valid=(torch.cat(t) for t in zip(*data))
    if not valid.any() or not (~valid).any():
        raise ValueError("training and validation both need eligible interventions")
    dims=(state.shape[1],features.shape[-1])
    base = torch.load(args.initial,map_location="cpu",weights_only=True)
    model=OutcomeHead(*dims)
    model.load_state_dict(base["head"])
    opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    def difference(idx):
        rows=torch.arange(len(idx)).repeat_interleave(2)
        return model.scores(state[idx],features[idx].flatten(0,1),rows).reshape(-1,2).diff(dim=1).flatten()
    y=delta-args.prior*gap
    train=torch.nonzero(~valid).flatten()
    val=torch.nonzero(valid).flatten()
    best_loss=float("inf")
    losses=[]
    best=None
    for ep in range(args.epochs+1):
        if ep:
            for idx in train[torch.randperm(len(train))].split(384):
                loss=(difference(idx)-y[idx]).square().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.inference_mode():
            loss=float(torch.cat([(difference(part)-y[part]).square() for part in val.split(512)]).mean())
        losses.append(loss)
        if loss<best_loss:
            best_loss=loss
            best={name:t.detach().clone() for name,t in model.state_dict().items()}
    args.output.mkdir(parents=True,exist_ok=True)
    for scale in args.scales:
        torch.save({**base,"head":best,"scale":scale},args.output/f"paired_{scale:g}.pt")
    info=dict(games=games,roots=len(state),training_roots=len(train),validation_roots=len(val),
              outcome_changed=int(delta.count_nonzero()),validation_mse=losses,
              prior_win_probability_per_logit=args.prior,selected_epoch=int(np.argmin(losses)),
              scales=args.scales,base_training_head_sha256=file_hash(args.initial),
              arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!="func"})
    (args.output/"fit.json").write_text(json.dumps(info,indent=2)+"\n")
    print(json.dumps(info),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    s=parser.add_subparsers(dest="operation",required=True)
    col=s.add_parser("collect")
    col.add_argument("--directory",type=Path,required=True)
    col.add_argument("--start",type=int,default=0)
    col.add_argument("--chunks",type=int,default=90)
    col.add_argument("--pairs",type=int,default=240)
    col.add_argument("--seed",type=int,default=511_000_000)
    col.add_argument("--beliefs",action="store_true")
    col.add_argument("--orbit",type=int,choices=(1,4,24),default=1)
    col.set_defaults(func=collect)
    f=s.add_parser("fit")
    f.add_argument("--directory",type=Path,required=True)
    f.add_argument("--initial",type=Path,required=True)
    f.add_argument("--output",type=Path,required=True)
    f.add_argument("--chunks",type=int,required=True)
    f.add_argument("--epochs",type=int,default=18)
    f.add_argument("--fit-seed",type=int,default=511_500_000)
    f.add_argument("--prior",type=float,default=.02)
    f.add_argument("--scales",type=float,nargs="+",default=[12.,32.])
    f.add_argument("--learning-rate",type=float,default=.001)
    f.set_defaults(func=fit)
    args=parser.parse_args()
    if args.operation == "collect" and (args.pairs < 1 or args.chunks < 1 or args.start < 0):
        parser.error("--pairs and --chunks must be positive, --start must be nonnegative")
    torch.set_num_threads(1)
    args.func(args)
