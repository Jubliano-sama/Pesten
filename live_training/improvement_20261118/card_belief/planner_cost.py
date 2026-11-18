"""Time the incremental planner cost on real states with multiple executable choices."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
from public_lookahead import (
    EMPTY_WEIGHTS,
    EMPTY_CARD_BELIEF,
    EMPTY_ROLLOUT,
    _batch_values,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--states-per-size", type=int, default=32)
    p.add_argument("--repeats", type=int, default=4)
    args = p.parse_args()
    cfg = json.loads(args.profile.read_text())
    posterior = np.asarray(
        json.loads(Path(cfg["posterior"]["path"]).read_text())["weights"]
    ).ravel()
    rollout = np.load(cfg["rollout"]["path"]) if "rollout" in cfg else EMPTY_ROLLOUT
    odds = np.load(cfg["card_belief"]["path"])
    obs = []
    hist = []
    rng = np.random.default_rng(911)
    for name in ("fork_0000.npz", "fork_0001.npz", "fork_0002.npz"):
        with np.load(args.data / name) as data:
            states = data["obs"].astype(np.float32)
            actions = data["act"]
            width = data["width"]
            offset = np.concatenate(([0], np.cumsum(width)))
            have_choices = np.array(
                [
                    states[i, 93] == 1
                    or (actions[offset[i] : offset[i + 1]] < 50).sum() >= 2
                    for i in range(len(states))
                ]
            )
            items = rng.choice(
                np.flatnonzero(have_choices), args.states_per_size, replace=False
            )
            tokens = np.zeros((len(items), 100), np.float32)
            tokens[:, :94] = states[items]
            tokens[:, 99] = 1
            obs.append(tokens)
            hist.append(data["tracking"][items].astype(np.float32))
    obs = np.concatenate(obs)
    hist = np.concatenate(hist)
    result = {}

    def run(weights):
        return _batch_values(
            obs,
            hist,
            cfg["samples"],
            cfg["horizon"],
            cfg["infer_voids"],
            EMPTY_WEIGHTS,
            posterior,
            EMPTY_WEIGHTS,
            rollout,
            weights,
        )

    for k, w in [("control", EMPTY_CARD_BELIEF), ("conditional", odds)]:
        v = run(w)
        result[k] = dict(
            value_sha256=hashlib.sha256(v.tobytes()).hexdigest(), wall=[], cpu=[]
        )
    for i in range(args.repeats):
        for k, w in (
            [("control", EMPTY_CARD_BELIEF), ("conditional", odds)]
            if i % 2 == 0
            else [("conditional", odds), ("control", EMPTY_CARD_BELIEF)]
        ):
            t = time.perf_counter()
            c = time.process_time()
            v = run(w)
            cpu = time.process_time() - c
            wall = time.perf_counter() - t
            assert hashlib.sha256(v.tobytes()).hexdigest() == result[k]["value_sha256"]
            result[k]["wall"].append(wall)
            result[k]["cpu"].append(cpu)
    for row in result.values():
        row["median_ms_per_choice"] = float(np.median(row["wall"]) * 1000 / len(obs))
    report = dict(
        states=len(obs), profile=str(args.profile), repeats=args.repeats, results=result
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
