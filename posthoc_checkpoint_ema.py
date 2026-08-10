#!/usr/bin/env python3
"""Build evaluation-free post-hoc EMAs over saved actor iterates.

This is deliberately labelled *post-hoc checkpoint EMA*.  It is not the
paper-exact minibatch parameter EMA used by ``filtered_ema_ppo.py``.  For each
projected-PPO lane, the accumulator starts from the verified production
champion and consumes the eight saved raw-base actor iterates in order:

    average_t = decay * average_(t-1) + (1 - decay) * iterate_t

Floating-point state tensors are averaged.  Non-floating state tensors must be
identical in every source and are copied unchanged.  The output remains a raw
``StudentNet`` state dictionary and therefore still requires exact 24-suit
projection at inference.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import torch

from anchored_ppo import CHAMPION_SHA256, file_sha256
from trpo_selfplay import module_sha256
from ppo2 import StudentNet


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    actor = StudentNet()
    actor.load_state_dict(state)
    return module_sha256(actor)


def checkpoint_ema(
    champion_state: dict[str, torch.Tensor],
    iterate_states: list[dict[str, torch.Tensor]],
    *,
    decay: float,
) -> tuple[dict[str, torch.Tensor], dict]:
    if not 0 <= decay < 1 or not math.isfinite(decay):
        raise ValueError("decay must be finite and in [0, 1)")
    if not iterate_states:
        raise ValueError("at least one iterate is required")
    keys = tuple(champion_state)
    if any(tuple(state) != keys for state in iterate_states):
        raise ValueError("checkpoint state structures differ")
    average = {
        name: tensor.detach().cpu().clone()
        for name, tensor in champion_state.items()
    }
    floating = []
    nonfloating = []
    for name, tensor in average.items():
        if tensor.dtype.is_floating_point:
            floating.append(name)
        else:
            nonfloating.append(name)
            if any(
                not torch.equal(tensor, state[name].detach().cpu())
                for state in iterate_states
            ):
                raise ValueError(
                    f"non-floating state changed across iterates: {name}"
                )
    with torch.no_grad():
        for state in iterate_states:
            for name in floating:
                average[name].mul_(decay).add_(
                    state[name].detach().cpu(),
                    alpha=1.0 - decay,
                )
    return average, {
        "initialization": "verified production champion",
        "ordered_iterate_count": len(iterate_states),
        "floating_tensors_averaged": len(floating),
        "nonfloating_tensors_verified_constant": len(nonfloating),
        "formula": (
            "average_t = decay*average_(t-1) + "
            "(1-decay)*iterate_t"
        ),
    }


def run(args: argparse.Namespace) -> dict:
    source_manifest = args.source_manifest.resolve()
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    champion_path = Path(source["champion"]["path"]).resolve()
    if file_sha256(champion_path) != CHAMPION_SHA256:
        raise RuntimeError("source champion is not the verified production actor")
    if source["champion"]["file_sha256"] != CHAMPION_SHA256:
        raise RuntimeError("source manifest declares a different champion")
    if source.get("algorithm") != "exact_24_suit_projected_ppo_iterative":
        raise ValueError("unexpected source training manifest")
    if not args.decays or any(
        not 0 <= decay < 1 or not math.isfinite(decay)
        for decay in args.decays
    ):
        raise ValueError("all decays must be finite and in [0, 1)")
    if len(set(args.decays)) != len(args.decays):
        raise ValueError("decays must be unique")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    champion_state = torch.load(
        champion_path,
        map_location="cpu",
        weights_only=True,
    )
    outputs = []
    for lane in source["lanes"]:
        iterations = lane["iterations"]
        if len(iterations) != 8:
            raise RuntimeError("checkpoint EMA requires exactly eight iterates")
        source_records = []
        states = []
        for expected_iteration, record in enumerate(iterations, start=1):
            if record["iteration"] != expected_iteration:
                raise RuntimeError("source iterations are not consecutive")
            checkpoint_record = record["checkpoint"]
            checkpoint = (
                source_manifest.parent / checkpoint_record["path"]
            ).resolve()
            actual_sha = file_sha256(checkpoint)
            if actual_sha != checkpoint_record["sha256"]:
                raise RuntimeError(f"source hash mismatch: {checkpoint}")
            state = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            if state_sha256(state) != checkpoint_record["actor_state_sha256"]:
                raise RuntimeError(f"source actor-state hash mismatch: {checkpoint}")
            states.append(state)
            source_records.append(
                {
                    "iteration": expected_iteration,
                    "path": str(checkpoint),
                    "sha256": actual_sha,
                    "actor_state_sha256": checkpoint_record[
                        "actor_state_sha256"
                    ],
                }
            )
        for decay in args.decays:
            average, reconstruction = checkpoint_ema(
                champion_state,
                states,
                decay=decay,
            )
            decay_slug = format(decay, "g").replace(".", "p")
            name = f"{lane['name']}_checkpoint_ema_{decay_slug}"
            checkpoint_path = output_dir / f"{name}.pth"
            torch.save(average, checkpoint_path)
            reloaded = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            if any(
                not torch.equal(average[key], reloaded[key])
                for key in average
            ):
                raise RuntimeError("saved checkpoint did not round-trip exactly")
            outputs.append(
                {
                    "name": name,
                    "lane": lane["name"],
                    "decay": float(decay),
                    "method": "post_hoc_checkpoint_ema",
                    "paper_exact_minibatch_ema": False,
                    "raw_base_requires_24_suit_projection": True,
                    "sources": source_records,
                    "reconstruction": reconstruction,
                    "checkpoint": {
                        "path": checkpoint_path.name,
                        "sha256": file_sha256(checkpoint_path),
                        "actor_state_sha256": state_sha256(reloaded),
                    },
                }
            )

    payload = {
        "schema_version": 1,
        "method": "post_hoc_checkpoint_ema",
        "status": "complete",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_free_construction": True,
        "paper_exact_minibatch_ema": False,
        "description": (
            "Champion-initialized sequential EMA over eight saved "
            "projected-PPO raw-base actor iterates."
        ),
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": file_sha256(source_manifest),
        },
        "champion": {
            "path": str(champion_path),
            "sha256": CHAMPION_SHA256,
            "actor_state_sha256": state_sha256(champion_state),
        },
        "decays": [float(decay) for decay in args.decays],
        "outputs": outputs,
        "privacy": {
            "no_game_data_consumed": True,
            "no_sealed_inputs_or_outcomes": True,
            "raw_base_observation_features": 121,
        },
        "promotion": {
            "eligible": False,
            "reason": "requires fresh direct and fixed-pool gates",
        },
    }
    manifest = output_dir / "posthoc_checkpoint_ema_manifest.json"
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path(
            "benchmarks/projected_ppo_iterative_dev/"
            "projected_ppo_iterative_metrics.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--decays",
        type=float,
        nargs="+",
        default=[0.5, 0.9],
    )
    return parser


def main() -> None:
    payload = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": payload["status"],
                "outputs": [
                    {
                        "name": output["name"],
                        **output["checkpoint"],
                    }
                    for output in payload["outputs"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
