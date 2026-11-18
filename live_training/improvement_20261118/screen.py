"""Compare several development candidates on one equal-weight, seat-rotated bank."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import benchmark_multiplayer_winrate as bench
import benchmark_rlvr_pool_batched as evaluator


def initialize(profiles):
    evaluator._initialize_multi_worker({"baseline": str(bench.TEACHER)}, str(bench.ROOT / "ppo_actor_100run3.pth"))
    policies = evaluator._POLICIES
    policies["baseline"] = bench.orbit24(policies["baseline"])
    for name, kind, path in profiles:
        if kind == "native":
            policies[name] = bench.orbit24(evaluator._load_policy(Path(path)))
        else:
            policies[name] = import_module(bench.CANDIDATE_MODULES[kind]).load_candidate(path, device="cpu")


def play(task):
    started = time.monotonic()
    return {"seed": task["seed"], **evaluator._evaluate_task(task), "wall_seconds": time.monotonic() - started}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", nargs=3, action="append", required=True, metavar=("NAME", "KIND", "PATH"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--blocks", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--resume", action="store_true", help="finish a partial screen from the pinned development roster")
    args = ap.parse_args()
    names = ["baseline", *(p[0] for p in args.profile)]
    if len(set(names)) != len(names):
        ap.error("duplicate name")
    tasks = bench.make_tasks(args.seed, args.blocks, args.chunk)
    for task in tasks:
        task.update(variants=tuple(names), policy_names=tuple(names), self_anchor_name="baseline")
    tasks.sort(key=lambda t: ((t["seed"] - args.seed) % bench.STRIDE, -t["players"], t["opponent"]))
    profiles = {name: {"kind":kind, "path":path, "sha256":bench.file_sha256(Path(path))} for name, kind, path in args.profile}
    rows = []
    if args.resume:
        roster = json.loads((args.output.parent / "roster.json").read_text())
        expected = roster["groups"][args.output.name]
        if ((args.output / "summary.json").exists() or args.blocks != roster["blocks_per_cell"]
                or args.seed != roster["seed"] or set(profiles) != set(expected)
                or any(p["sha256"] != expected[k]["sha256"] or p["path"] != expected[k]["file"]
                       for k, p in profiles.items())):
            ap.error("resume requires the original incomplete development screen and pinned profiles")
        rows = [json.loads(line) for line in (args.output / "rows.jsonl").read_text().splitlines()]
        planned = {(t["opponent"], t["players"], t["seed"]): t["blocks"] for t in tasks}
        done = set()
        for row in rows:
            key = (row["opponent"], row["players"], row["seed"])
            if (key in done or key not in planned or set(row["totals"]) != set(names)
                    or len(row["baseline_win_shares"]) != planned[key]):
                ap.error("saved chunks disagree with the requested deal and seat layout")
            done.add(key)
        tasks = [t for t in tasks if (t["opponent"], t["players"], t["seed"]) not in done]
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "rows.jsonl").open("a" if args.resume else "x") as handle, ProcessPoolExecutor(
            args.workers, initializer=initialize, initargs=(args.profile,)) as pool:
        for row in pool.map(play, tasks):
            rows.append(row)
            handle.write(json.dumps(row)+"\n")
            handle.flush()
            print(row["opponent"], row["players"], row["seed"], {k:v.get("wins",0) for k,v in row["totals"].items()}, flush=True)
    summary = {}
    for name in names[1:]:
        paired = []
        for row in rows:
            copy = {**row, "totals": {"baseline":row["totals"]["baseline"], "candidate":row["totals"][name]}}
            for prop in ("win_shares", "loss_shares", "draw_shares"):
                copy["candidate_" + prop] = row[name + "_" + prop]
            paired.append(copy)
        report = bench.summarize(paired, seed=args.seed, blocks=args.blocks)
        summary[name] = {k:v for k,v in report.items() if k!="raw_blocks"}
    result = {"profiles":profiles, "seed":args.seed, "blocks_per_cell":args.blocks, "results":summary}
    (args.output / "summary.json").write_text(json.dumps(result,indent=2)+"\n")
    for k,v in summary.items():
        print(k, v["macro"], flush=True)


if __name__ == "__main__":
    main()
