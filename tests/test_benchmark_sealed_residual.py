import hashlib

import pytest

from benchmark_residual_checkpoint import DEFAULT_OPPONENT_NAMES
from benchmark_sealed_checkpoint import (
    SEALED_SEED_END,
    SEALED_SEED_START,
)
from benchmark_sealed_residual import (
    assert_residual_information_boundary,
    build_argument_parser,
    refuse_output_overwrite,
    require_sealed_confirmation,
    sealed_residual_seed_plan,
    verify_candidate_identity,
)


def test_default_plan_has_disjoint_contiguous_blocks_inside_sealed_range():
    pairs = 11
    plan = sealed_residual_seed_plan(
        pairs=pairs,
        base_seed=SEALED_SEED_START + 400_000,
    )

    assert tuple(plan) == tuple(
        f"opponent:{name}" for name in DEFAULT_OPPONENT_NAMES
    )
    allocated = [seed for block in plan.values() for seed in block]
    assert len(allocated) == pairs * len(DEFAULT_OPPONENT_NAMES)
    assert len(allocated) == len(set(allocated))
    assert allocated == list(
        range(
            SEALED_SEED_START + 400_000,
            SEALED_SEED_START + 400_000 + len(allocated),
        )
    )
    assert all(
        SEALED_SEED_START <= seed < SEALED_SEED_END
        for seed in allocated
    )


@pytest.mark.parametrize(
    "base_seed",
    [SEALED_SEED_START - 1, SEALED_SEED_END],
)
def test_plan_rejects_nonsealed_base_seed(base_seed):
    with pytest.raises(ValueError, match="base seed"):
        sealed_residual_seed_plan(
            pairs=1,
            base_seed=base_seed,
        )


def test_plan_rejects_overflow_duplicate_empty_and_unknown_opponents():
    with pytest.raises(ValueError, match="exceeds"):
        sealed_residual_seed_plan(
            pairs=2,
            base_seed=SEALED_SEED_END - 5,
        )
    with pytest.raises(ValueError, match="unique"):
        sealed_residual_seed_plan(
            pairs=1,
            base_seed=SEALED_SEED_START,
            opponent_names=("smart", "smart"),
        )
    with pytest.raises(ValueError, match="at least one"):
        sealed_residual_seed_plan(
            pairs=1,
            base_seed=SEALED_SEED_START,
            opponent_names=(),
        )
    with pytest.raises(ValueError, match="unknown"):
        sealed_residual_seed_plan(
            pairs=1,
            base_seed=SEALED_SEED_START,
            opponent_names=("oracle",),
        )


class SafePolicy:
    architecture = "test_safe_residual"
    critic_privileged = False


class PrivilegedPolicy:
    architecture = "test_privileged_residual"
    critic_privileged = True


def test_information_boundary_contract_accepts_only_nonprivileged_policy():
    assert_residual_information_boundary(SafePolicy())
    with pytest.raises(RuntimeError, match="critic_privileged=False"):
        assert_residual_information_boundary(PrivilegedPolicy())


def test_confirmation_identity_and_output_guards_need_no_game(tmp_path):
    with pytest.raises(ValueError, match="confirmation"):
        require_sealed_confirmation(False)
    require_sealed_confirmation(True)

    checkpoint = tmp_path / "residual.pt"
    checkpoint.write_bytes(b"precommitted residual checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert verify_candidate_identity(checkpoint, digest) == digest
    with pytest.raises(ValueError, match="mismatch"):
        verify_candidate_identity(checkpoint, "0" * 64)

    output = tmp_path / "sealed.json"
    refuse_output_overwrite(output)
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        refuse_output_overwrite(output)


def test_cli_requires_checkpoint_digest_seed_output_and_confirmation_is_explicit():
    parser = build_argument_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            "candidate.pt",
            "--expected-sha256",
            "a" * 64,
            "--base-seed",
            str(SEALED_SEED_START),
            "--output",
            "sealed.json",
        ]
    )

    assert args.checkpoint.name == "candidate.pt"
    assert args.expected_sha256 == "a" * 64
    assert args.base_seed == SEALED_SEED_START
    assert tuple(args.opponents) == DEFAULT_OPPONENT_NAMES
    assert not args.confirm_sealed_evaluation
