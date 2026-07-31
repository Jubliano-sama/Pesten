import hashlib
import json
from pathlib import Path

import pytest
import torch

from batched_pesten import ActorPolicy
from benchmark_checkpoint_batched import run_batched_specs
from benchmark_nashpg_crossplay import (
    DEVELOPMENT_SEED_LIMIT,
    assemble_crossplay_payload,
    build_crossplay_specs,
    checkpoint_specs_from_nashpg_manifest,
    parse_checkpoint_argument,
)


class _FlatActor(torch.nn.Module):
    def forward(self, observations):
        return torch.zeros(
            (len(observations), 55),
            dtype=observations.dtype,
            device=observations.device,
        )


def test_checkpoint_argument_requires_stable_name_path_pair():
    assert parse_checkpoint_argument("outer_01=/tmp/outer_01.pth") == (
        "outer_01",
        Path("/tmp/outer_01.pth"),
    )
    with pytest.raises(ValueError, match="NAME=PATH"):
        parse_checkpoint_argument("/tmp/outer_01.pth")
    with pytest.raises(ValueError, match="model names"):
        parse_checkpoint_argument("bad name=/tmp/model.pth")


def test_nashpg_manifest_discovers_outer_checkpoints(tmp_path):
    checkpoint = tmp_path / "nashpg_alpha_0p2_outer_03.pth"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(b"checkpoint").hexdigest()
    manifest = tmp_path / "nashpg_pilot_metrics.json"
    manifest.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "alpha": 0.2,
                        "outer_rounds": [
                            {
                                "outer_iteration": 3,
                                "checkpoint": {
                                    "path": checkpoint.name,
                                    "sha256": digest,
                                },
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    specs = checkpoint_specs_from_nashpg_manifest(manifest)
    assert len(specs) == 1
    assert specs[0].name == "alpha_0p2_outer_03"
    assert specs[0].path == checkpoint.resolve()
    assert specs[0].sha256 == digest


def test_specs_use_fresh_explicit_seeds_and_common_pool_deals():
    specs, allocation = build_crossplay_specs(
        model_names=("incumbent", "outer_01", "outer_02"),
        pairs=2,
        base_seed=91_999_998,
        pool_names=("random", "smart"),
        recorded_seeds=(91_999_998, 91_999_999, 0),
    )
    # Three head-to-head matchups plus two shared fixed-pool deal streams.
    assert len(allocation.allocated_seeds) == (3 + 2) * 2
    assert set(allocation.allocated_seeds).isdisjoint(
        {91_999_998, 91_999_999, 0}
    )
    assert max(allocation.allocated_seeds) < DEVELOPMENT_SEED_LIMIT
    assert allocation.skipped_recorded_seeds[:3] == (
        91_999_998,
        91_999_999,
        0,
    )

    random_pair_zero = [
        spec
        for spec in specs
        if (
            spec.section == "fixed_pool"
            and spec.opponent == "random"
            and spec.pair_index == 0
        )
    ]
    assert len(random_pair_zero) == 6
    assert len({spec.deal_seed for spec in random_pair_zero}) == 1
    for model_name in ("incumbent", "outer_01", "outer_02"):
        model_specs = [
            spec
            for spec in random_pair_zero
            if spec.evaluated_key == model_name
        ]
        assert [spec.evaluated_seat for spec in model_specs] == [0, 1]


def test_fixture_actors_crossplay_and_pool_are_exactly_paired():
    specs, _ = build_crossplay_specs(
        model_names=("incumbent", "fixture"),
        pairs=3,
        base_seed=89_700_001,
        pool_names=("random", "smart", "smarter"),
    )
    actor = _FlatActor()
    policy = ActorPolicy(
        actor=actor,
        device=torch.device("cpu"),
        temperature=1.0,
        collect_data=False,
        greedy=True,
    )
    outcomes, stats = run_batched_specs(
        specs,
        {"incumbent": policy, "fixture": policy},
        max_turns=1_000,
    )
    payload = assemble_crossplay_payload(
        specs,
        outcomes,
        model_names=("incumbent", "fixture"),
        incumbent_name="incumbent",
        pairs=3,
        base_seed=89_700_001,
        pool_names=("random", "smart", "smarter"),
        bootstrap_samples=100,
    )
    assert stats["games"] == 42
    assert payload["head_to_head"][0]["model_a_score"]["mean"] == 0.5
    summaries = {entry["model"]: entry for entry in payload["model_summary"]}
    assert (
        summaries["fixture"]["combined_fixed_pool_delta_vs_incumbent"]["mean"]
        == 0.0
    )
    for opponent in payload["fixed_pool"]:
        by_model = {
            entry["model"]: entry["score"]["mean"]
            for entry in opponent["models"]
        }
        assert by_model["incumbent"] == by_model["fixture"]
