"""Time and verify all three fits used by the selected online-search agent."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from benchmark import ACTOR, ARMS, HERE, digest
from benchmark_outcome import data_hash
from benchmark_search_training import command, run as shared_stages, stage as stage_shared


CARD_BATCHES = ((1, 0, 60), (4, 60, 24), (24, 84, 18))
CARD_GAMES_PER_SHARD = 16
CARD_ITERATIONS = 14


def stage(args):
    shared_hashes, recipes = stage_shared(args)
    shared = args.scratch / "shared"
    hashes = json.loads((HERE / "selected_training_sources.json").read_text())
    with ZipFile(HERE / "selected_training_sources.zip") as archive:
        for name, expected in hashes.items():
            path = shared / name
            path.write_bytes(archive.read(name))
            assert digest(path) == expected
    warm = args.scratch / "warm-card"
    warm.mkdir()
    for index in range(6):
        n = 2 + index % 3
        hands = np.full((2, n, 54), -1, np.int8)
        hands[:, :, 0] = np.arange(n)
        np.savez(warm / f"shard_{index:04}.npz", hands=hands, sizes=np.ones((2, n), np.uint8),
                 seat=np.zeros(2, np.int8), direction=np.ones(2, np.int8), suit=np.zeros(2, np.int8),
                 rank=np.full(2, 11, np.int8), penalty=np.zeros(2, np.int16), episode=np.arange(2, dtype=np.uint8),
                 random_others=np.array([True, False]), deals=np.arange(2, dtype=np.uint32), orbit=1)
    prior = warm / "frequency.json"
    prior.write_text(json.dumps({"weights": np.zeros((2, 3, 9, 10)).tolist()}))
    command(args, "after", shared / "train_card_belief.py",
            ["--data", warm, "--prior", prior, "--output", warm / "fit", "--shards", 6, "--iterations", 1],
            warm / "jit.log")
    return shared_hashes, {**recipes, **hashes}


def run(args, arm, seed):
    row = shared_stages(args, arm, seed)
    shared = args.scratch / "shared"
    output = args.scratch / f"{arm}-{seed}"
    data = output / "conditional-hands"
    fit = output / "card-fit"
    extra = ["--eager-actor"] if arm == "before" else []
    for orbit, start, shards in CARD_BATCHES:
        # The full recipe uses a separate bank for exact 24-view collection.
        seed_offset = 4_000_000 if orbit == 24 else 1_000_000
        elapsed = command(args, arm, shared / "train_public_leaf.py",
                          ["collect", "--data", data, "--orbit", orbit, "--seed", seed + seed_offset,
                           "--start", start, "--chunks", shards, "--games", CARD_GAMES_PER_SHARD, "--keep", .18, *extra],
                          output / f"card-orbit-{orbit}.log")
        row[f"card_orbit_{orbit}_seconds"] = elapsed
    prior = output / "hand_frequencies.json"
    row["card_fit_seconds"] = command(args, arm, shared / "train_card_belief.py",
                                      ["--data", data, "--prior", prior, "--output", fit,
                                       "--shards", 102, "--iterations", CARD_ITERATIONS], output / "card-fit.log")
    report = json.loads((fit / "fit.json").read_text())
    training = [s for s in report["shards"] if not s["validation"]]
    validation = [s for s in report["shards"] if s["validation"]]
    assert len(report["shards"]) == 102 and len(validation) == 18 and report["iterations"] == CARD_ITERATIONS
    assert {s["orbit"] for s in validation} == {1, 4, 24}
    assert {(s["role"], s["players"]) for s in report["cells"]} == {
        (role, n) for role in ("teacher", "random") for n in (2, 3, 4)
    }
    assert all(s["cards"] > 0 and np.isfinite(s["fitted"]) for s in report["cells"])
    odds = np.load(fit / "card_odds.npy", allow_pickle=False)
    assert odds.shape[:3] == (2, 3, 5) and np.isfinite(odds).all()
    assert report["baseline_prior_sha256"] == row["frequency_sha256"]
    row.update({
        "card_shards_data_sha256": data_hash(data), "card_odds_sha256": digest(fit / "card_odds.npy"),
        "card_fit_report_sha256": digest(fit / "fit.json"), "card_fit_shape": list(odds.shape),
        "card_train_games": sum(s["games"] for s in training), "card_validation_games": sum(s["games"] for s in validation),
        "card_train_opponent_hands": report["train_opponent_hands"], "card_validation_opponent_hands": report["validation_opponent_hands"],
        "card_validation_cards": report["validation_cards"], "card_validation": report["cells"],
    })
    row["complete_games"] = row["focal_games"] + row["random_games"] + row["card_train_games"] + row["card_validation_games"]
    row["total_seconds"] = sum(v for k, v in row.items() if k.endswith("_seconds") and k != "total_seconds")
    return row


def background_cpu_ticks(cpu):
    siblings = (Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
                .read_text().strip().split(","))
    reserved = set()
    for entry in siblings:
        bounds = list(map(int, entry.split("-")))
        reserved.update(range(bounds[0], bounds[-1] + 1))
    total = busy = 0
    for line in Path("/proc/stat").read_text().splitlines():
        label, *values = line.split()
        if not label.startswith("cpu") or not label[3:].isdigit() or int(label[3:]) in reserved:
            continue
        fields = list(map(int, values[:8]))
        total += sum(fields)
        busy += sum(fields) - fields[3] - fields[4]
    return total, busy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", default="0")
    parser.add_argument("--seeds", default="630000100,640000100,650000100")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output file")
    # Exactly 1/24 of each full campaign game's category count per trial.
    args.games = 192
    args.random_games = 64
    args.scratch = args.scratch.resolve()
    args.scratch.mkdir(parents=True, exist_ok=False)
    shared_sources, recipes = stage(args)
    result = {"cpu": args.cpu, "actor_sha256": digest(ACTOR), "source_hashes": shared_sources, "recipes": recipes,
              "scale": "1/24 of the full candidate training game counts in all three components",
              "scaled_games": {"exploration": 1728, "all_random": 192, "conditional_labels": 1632},
              "actual_games": {"exploration": 41472, "all_random": 4608, "conditional_labels": 39168},
              "conditional_batches": CARD_BATCHES, "conditional_games_per_shard": CARD_GAMES_PER_SHARD,
              "conditional_iterations": CARD_ITERATIONS, "exploration_games_per_shard": args.games,
              "jit": "identical source and one prewarmed cache, one Numba and BLAS thread", "pairs": []}
    for index, seed in enumerate(map(int, args.seeds.split(","))):
        order = list(ARMS if index % 2 == 0 else reversed(ARMS))
        pair = {"seed": seed, "order": order, "other_cpu_busy_percent": {}}
        for arm in order:
            t0, b0 = background_cpu_ticks(args.cpu)
            pair[arm] = run(args, arm, seed)
            t1, b1 = background_cpu_ticks(args.cpu)
            pair["other_cpu_busy_percent"][arm] = 100 * (b1 - b0) / (t1 - t0)
            print(arm, seed, round(pair[arm]["total_seconds"], 3), flush=True)
        for key, value in pair["before"].items():
            if not key.endswith("_seconds"):
                assert value == pair["after"][key], f"{seed}: {key}"
        pair["every_artifact_and_work_count_exact"] = True
        result["pairs"].append(pair)
        totals = {arm: {key: sum(p[arm][key] for p in result["pairs"])
                        for key in pair[arm] if key.endswith("_seconds") or key in {
                            "complete_games", "focal_games", "random_games", "decisions", "continuation_training_pairs",
                            "strong_hand_observations", "random_hand_observations", "card_train_games", "card_validation_games",
                            "card_train_opponent_hands", "card_validation_opponent_hands", "card_validation_cards"}} for arm in ARMS}
        for row in totals.values():
            row["games_per_second"] = row["complete_games"] / row["total_seconds"]
        result["totals"] = totals
        result["speedup"] = totals["before"]["total_seconds"] / totals["after"]["total_seconds"]
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("verified", seed, "speedup", result["speedup"], flush=True)


if __name__ == "__main__":
    main()
