from collections import OrderedDict

import pytest
import torch

from checkpoint_interpolation import (
    alpha_label,
    create_interpolated_checkpoints,
    interpolate_state_dicts,
    parse_alpha,
    validate_screen,
)


def state(weight, *, count=3):
    return OrderedDict(
        weight=torch.full((count,), float(weight)),
        counter=torch.tensor(7, dtype=torch.int64),
    )


def test_interpolation_endpoints_and_midpoint_are_exact():
    champion = state(2.0)
    candidate = state(6.0)

    left = interpolate_state_dicts(champion, candidate, alpha=0.0)
    middle = interpolate_state_dicts(champion, candidate, alpha=0.25)
    right = interpolate_state_dicts(champion, candidate, alpha=1.0)

    assert torch.equal(left["weight"], champion["weight"])
    assert torch.equal(middle["weight"], torch.full((3,), 3.0))
    assert torch.equal(right["weight"], candidate["weight"])
    assert left["counter"].item() == 7
    assert right["counter"].item() == 7


def test_checkpoint_writer_preserves_raw_state_dict_contract(tmp_path):
    champion_path = tmp_path / "champion.pth"
    candidate_path = tmp_path / "candidate.pth"
    torch.save(state(1.0), champion_path)
    torch.save(state(5.0), candidate_path)

    paths = create_interpolated_checkpoints(
        champion_path=champion_path,
        candidate_path=candidate_path,
        alphas=(0.0, 0.5, 1.0),
        output_dir=tmp_path / "soups",
    )

    assert set(paths) == {0.0, 0.5, 1.0}
    written = torch.load(paths[0.5], weights_only=True)
    assert list(written) == ["weight", "counter"]
    assert torch.equal(written["weight"], torch.full((3,), 3.0))


def test_incompatible_checkpoints_fail_closed():
    with pytest.raises(ValueError, match="keys"):
        interpolate_state_dicts(
            state(1.0),
            {"different": torch.ones(3)},
            alpha=0.5,
        )
    with pytest.raises(ValueError, match="shape"):
        interpolate_state_dicts(
            state(1.0),
            state(2.0, count=4),
            alpha=0.5,
        )
    changed_counter = state(2.0)
    changed_counter["counter"] = torch.tensor(8, dtype=torch.int64)
    with pytest.raises(ValueError, match="non-floating"):
        interpolate_state_dicts(
            state(1.0),
            changed_counter,
            alpha=0.5,
        )


def test_alpha_and_development_seed_validation():
    assert parse_alpha("0.25") == 0.25
    assert alpha_label(0.25) == "0p25"
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        parse_alpha("1.1")
    validate_screen(pairs=50, seed=89_000_001, opponent_count=3)
    with pytest.raises(ValueError, match="sealed"):
        validate_screen(
            pairs=50,
            seed=91_500_001,
            opponent_count=3,
        )
