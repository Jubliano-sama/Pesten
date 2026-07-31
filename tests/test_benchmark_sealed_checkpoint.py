import hashlib

import pytest

from benchmark_sealed_checkpoint import (
    DEFAULT_POOL,
    SEALED_SEED_END,
    SEALED_SEED_START,
    build_sealed_game_specs,
    refuse_output_overwrite,
    sealed_seed_plan,
    validate_expected_sha256,
    verify_candidate_identity,
)


def test_seed_plan_is_disjoint_collision_free_and_inside_sealed_interval():
    pairs = 7
    plan = sealed_seed_plan(
        pairs=pairs,
        base_seed=SEALED_SEED_START + 100_000,
    )

    assert tuple(plan) == (
        "direct",
        "pool:random",
        "pool:smart",
        "pool:smarter",
    )
    allocated = [seed for seeds in plan.values() for seed in seeds]
    assert len(allocated) == pairs * (1 + len(DEFAULT_POOL))
    assert len(allocated) == len(set(allocated))
    assert all(
        SEALED_SEED_START <= seed < SEALED_SEED_END
        for seed in allocated
    )


def test_specs_reuse_only_the_intended_matched_deal_and_rotate_seats():
    pairs = 3
    specs = build_sealed_game_specs(
        pairs=pairs,
        base_seed=SEALED_SEED_START + 200_000,
    )
    assert len(specs) == pairs * (2 + 4 * len(DEFAULT_POOL))

    logical_deals = {}
    for spec in specs:
        namespace = (
            "direct"
            if spec.section == "direct"
            else f"pool:{spec.opponent}"
        )
        key = (namespace, spec.pair_index)
        logical_deals.setdefault(key, set()).add(spec.deal_seed)
    assert all(len(seeds) == 1 for seeds in logical_deals.values())
    distinct = {next(iter(seeds)) for seeds in logical_deals.values()}
    assert len(distinct) == pairs * (1 + len(DEFAULT_POOL))

    direct_zero = [
        spec
        for spec in specs
        if spec.section == "direct" and spec.pair_index == 0
    ]
    assert [spec.evaluated_seat for spec in direct_zero] == [0, 1]
    smarter_zero = [
        spec
        for spec in specs
        if spec.opponent == "smarter" and spec.pair_index == 0
    ]
    assert {
        (spec.evaluated_key, spec.evaluated_seat)
        for spec in smarter_zero
    } == {
        ("candidate", 0),
        ("candidate", 1),
        ("parent", 0),
        ("parent", 1),
    }


@pytest.mark.parametrize(
    "base_seed",
    [SEALED_SEED_START - 1, SEALED_SEED_END],
)
def test_seed_plan_rejects_nonsealed_base_seed(base_seed):
    with pytest.raises(ValueError, match="base seed"):
        sealed_seed_plan(pairs=1, base_seed=base_seed)


def test_seed_plan_rejects_interval_overflow_duplicates_and_unknown_pool():
    with pytest.raises(ValueError, match="exceeds"):
        sealed_seed_plan(
            pairs=2,
            base_seed=SEALED_SEED_END - 5,
        )
    with pytest.raises(ValueError, match="unique"):
        sealed_seed_plan(
            pairs=1,
            base_seed=SEALED_SEED_START,
            pool_names=("random", "random"),
        )
    with pytest.raises(ValueError, match="unknown"):
        sealed_seed_plan(
            pairs=1,
            base_seed=SEALED_SEED_START,
            pool_names=("oracle",),
        )


def test_candidate_identity_requires_exact_precommitted_sha256(tmp_path):
    candidate = tmp_path / "candidate.pth"
    candidate.write_bytes(b"candidate checkpoint bytes")
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()

    assert validate_expected_sha256(digest) == digest
    assert verify_candidate_identity(candidate, digest) == digest
    with pytest.raises(ValueError, match="exactly 64"):
        verify_candidate_identity(candidate, digest[:16])
    with pytest.raises(ValueError, match="mismatch"):
        verify_candidate_identity(candidate, "0" * 64)


def test_output_guard_refuses_existing_file_directory_and_symlink(tmp_path):
    missing = tmp_path / "new.json"
    refuse_output_overwrite(missing)

    existing_file = tmp_path / "existing.json"
    existing_file.write_text("sealed", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        refuse_output_overwrite(existing_file)

    existing_directory = tmp_path / "results"
    existing_directory.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        refuse_output_overwrite(existing_directory)

    dangling_target = tmp_path / "missing-target"
    dangling_link = tmp_path / "dangling-link"
    dangling_link.symlink_to(dangling_target)
    assert not dangling_link.exists()
    with pytest.raises(FileExistsError, match="overwrite"):
        refuse_output_overwrite(dangling_link)
