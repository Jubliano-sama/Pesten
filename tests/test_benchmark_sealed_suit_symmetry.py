import pytest
import torch

import benchmark_sealed_suit_symmetry as sealed_symmetry
from benchmark_model_zoo import KNOWN_TEACHER_SHA256
from benchmark_sealed_checkpoint import (
    DEFAULT_POOL,
    SEALED_SEED_END,
    SEALED_SEED_START,
)
from benchmark_sealed_suit_symmetry import (
    build_argument_parser,
    build_sealed_symmetry_specs,
    evaluate_sealed_suit_symmetry,
    refuse_output_overwrite,
    require_sealed_confirmation,
    validate_champion_precommit,
    validate_projection,
)


def test_symmetry_specs_use_direct_and_fixed_pool_disjoint_sealed_blocks():
    pairs = 5
    plan, specs = build_sealed_symmetry_specs(
        pairs=pairs,
        base_seed=SEALED_SEED_START + 600_000,
    )

    assert tuple(plan) == (
        "direct",
        "pool:random",
        "pool:smart",
        "pool:smarter",
    )
    allocated = [seed for block in plan.values() for seed in block]
    assert allocated == list(
        range(
            SEALED_SEED_START + 600_000,
            SEALED_SEED_START + 600_000 + pairs * 4,
        )
    )
    assert len(allocated) == len(set(allocated))
    assert all(
        SEALED_SEED_START <= seed < SEALED_SEED_END
        for seed in allocated
    )
    assert len(specs) == pairs * (2 + 4 * len(DEFAULT_POOL))

    direct = [spec for spec in specs if spec.section == "direct"]
    assert {
        (spec.pair_index, spec.evaluated_seat)
        for spec in direct
    } == {
        (pair_index, seat)
        for pair_index in range(pairs)
        for seat in (0, 1)
    }
    assert all(spec.actor_keys in {
        ("candidate", "parent"),
        ("parent", "candidate"),
    } for spec in direct)


def test_precommit_is_exactly_the_reviewed_production_champion():
    assert (
        validate_champion_precommit(KNOWN_TEACHER_SHA256)
        == KNOWN_TEACHER_SHA256
    )
    with pytest.raises(ValueError, match="exact production champion"):
        validate_champion_precommit("0" * 64)
    with pytest.raises(ValueError, match="exactly 64"):
        validate_champion_precommit(KNOWN_TEACHER_SHA256[:20])


def test_seed_interval_and_output_guards_are_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="base seed"):
        build_sealed_symmetry_specs(
            pairs=1,
            base_seed=SEALED_SEED_START - 1,
        )
    with pytest.raises(ValueError, match="exceeds"):
        build_sealed_symmetry_specs(
            pairs=2,
            base_seed=SEALED_SEED_END - 5,
        )

    output = tmp_path / "sealed.json"
    refuse_output_overwrite(output)
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        refuse_output_overwrite(output)


@pytest.mark.parametrize(
    ("alpha", "aggregation", "chunk", "message"),
    (
        (-0.1, "logits", 512, "alpha"),
        (1.1, "logits", 512, "alpha"),
        (1.0, "votes", 512, "aggregation"),
        (1.0, "logits", 0, "base_chunk_size"),
    ),
)
def test_projection_guards_fail_closed(
    alpha,
    aggregation,
    chunk,
    message,
):
    with pytest.raises(ValueError, match=message):
        validate_projection(
            alpha=alpha,
            aggregation=aggregation,
            player_orbit_average=False,
            base_chunk_size=chunk,
        )
    validate_projection(
        alpha=0.75,
        aggregation="probabilities",
        player_orbit_average=True,
        base_chunk_size=64,
    )


def test_confirmation_is_explicit_and_cli_defaults_to_fixed_pool():
    with pytest.raises(ValueError, match="confirmation"):
        require_sealed_confirmation(False)
    require_sealed_confirmation(True)

    args = build_argument_parser().parse_args(
        [
            "--expected-sha256",
            KNOWN_TEACHER_SHA256,
            "--base-seed",
            str(SEALED_SEED_START),
            "--output",
            "sealed-symmetry.json",
        ]
    )
    assert args.pool == "random,smart,smarter"
    assert args.alpha == 1.0
    assert args.aggregation == "logits"
    assert not args.player_orbit_average
    assert not args.confirm_sealed_evaluation


def test_evaluation_forwards_player_orbit_flag_to_projected_actor(
    monkeypatch,
    tmp_path,
):
    captured = {}

    class DummyActor(torch.nn.Module):
        def forward(self, observations):
            return observations

    class CapturingProjection(DummyActor):
        def __init__(self, actor, **kwargs):
            super().__init__()
            captured.update(kwargs)
            self.actor = actor

        def statistics(self):
            return {"player_orbit_average": captured["player_orbit_average"]}

    monkeypatch.setattr(
        sealed_symmetry,
        "verify_candidate_identity",
        lambda path, expected: expected,
    )
    monkeypatch.setattr(
        sealed_symmetry,
        "load_frozen_ppo_actor",
        lambda path, device: (DummyActor(), torch.device("cpu")),
    )
    monkeypatch.setattr(
        sealed_symmetry,
        "BatchedSuitAveragedActor",
        CapturingProjection,
    )
    monkeypatch.setattr(
        sealed_symmetry,
        "run_batched_specs",
        lambda specs, policies, max_turns: ([], {"games": 0}),
    )
    monkeypatch.setattr(
        sealed_symmetry,
        "assemble_benchmark_payload",
        lambda *args, **kwargs: {},
    )

    payload = evaluate_sealed_suit_symmetry(
        champion_path=tmp_path / "champion.pth",
        expected_sha256=KNOWN_TEACHER_SHA256,
        pairs=1,
        base_seed=SEALED_SEED_START,
        pool_names=DEFAULT_POOL,
        device_name="cpu",
        max_turns=10,
        bootstrap_samples=10,
        alpha=1.0,
        aggregation="logits",
        player_orbit_average=True,
        base_chunk_size=64,
        confirmed=True,
        output_path=tmp_path / "sealed-result.json",
        claim_ledger_path=tmp_path / "sealed-claim.json",
    )

    assert captured["player_orbit_average"] is True
    assert payload["candidate"]["player_orbit_average"] is True
    assert payload["actor_statistics"]["player_orbit_average"] is True
    claim = payload["sealed_holdout"]["global_claim"]
    assert claim["evaluator"] == "suit_symmetry_projection"
    assert claim["configuration"]["player_orbit_average"] is True
    assert (tmp_path / "sealed-claim.json").is_file()
