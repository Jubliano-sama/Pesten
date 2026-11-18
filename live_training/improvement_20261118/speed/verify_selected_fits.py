"""Replay all three production fits on each measured collection and verify file bytes."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from benchmark import HERE, ROOT, digest
from benchmark_search_training import FIT_SCRIPT, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True, help="completed benchmark scratch")
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh report path")
    result = json.loads(args.timing.read_text())
    args.scratch = args.scratch.resolve()
    check = SimpleNamespace(cpu=result["cpu"], scratch=args.scratch)
    scripts = [ROOT / "train_public_lookahead.py", ROOT / FIT_SCRIPT, ROOT / "train_card_belief.py"]
    replay_root = Path(tempfile.mkdtemp(prefix="production-fit-replay-", dir=args.scratch))
    validation = {"production_fit_sources": {str(s.relative_to(ROOT)): digest(s) for s in scripts},
                  "planned_seeds": [pair["seed"] for pair in result["pairs"]], "pairs": []}
    profile = json.loads((HERE.parent / "card_belief/profile.json").read_text())
    for key in ("posterior", "rollout", "card_belief"):
        assert digest(ROOT / profile[key]["path"]) == profile[key]["sha256"]
    validation["original_candidate_assets"] = {key: profile[key]["sha256"] for key in ("posterior", "rollout", "card_belief")}
    for pair in result["pairs"]:
        seed = pair["seed"]
        measured = args.scratch / f"after-{seed}"
        out = replay_root / str(seed)
        out.mkdir()
        frequency = out / "hand_frequencies.json"
        continuation = out / "rollout_weights.npy"
        fit = out / "card-fit"
        command(check, "after", scripts[0], [measured / "deals", frequency, "--random-games", 64,
                                              "--seed", seed + 8000], out / "frequency.log")
        command(check, "after", scripts[1], [measured / "deals", continuation], out / "continuation.log")
        command(check, "after", scripts[2], ["--data", measured / "conditional-hands", "--prior", frequency,
                                              "--output", fit, "--shards", 102, "--iterations", 14], out / "card.log")
        replayed = {"frequency_sha256": digest(frequency), "continuation_sha256": digest(continuation),
                    "card_odds_sha256": digest(fit / "card_odds.npy"), "card_fit_report_sha256": digest(fit / "fit.json")}
        assert all(value == pair[arm][key] for arm in ("before", "after") for key, value in replayed.items())
        validation["pairs"].append({"seed": seed, "all_bytes_identical_to_both_measured_arms": True, **replayed})
        validation["complete"] = len(validation["pairs"]) == len(validation["planned_seeds"])
        args.output.write_text(json.dumps(validation, indent=2) + "\n")
        print("production fits verified", seed, flush=True)


if __name__ == "__main__":
    main()
