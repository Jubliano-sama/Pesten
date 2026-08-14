import sys
from pathlib import Path

import pytest

from privileged_critic import DEFAULT_GAMMA
from train_puffer_fast import (
    DEFAULT_PRIVILEGED_CRITIC,
    _validate_privileged_target_gamma,
    parse_args,
)


def _parse(monkeypatch, *extra: str):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_puffer_fast.py",
            "--resume",
            "actor.pt",
            "--output-dir",
            "output",
            *extra,
        ],
    )
    return parse_args()


def test_promoted_mlp_and_quarter_at_twenty_are_defaults(monkeypatch):
    args = _parse(monkeypatch)

    assert args.privileged_critic == DEFAULT_PRIVILEGED_CRITIC
    assert args.privileged_critic.is_file()
    assert args.gamma == DEFAULT_GAMMA
    assert args.gamma**20 == pytest.approx(0.25, abs=1e-12)


def test_default_privileged_critic_has_explicit_opt_out(monkeypatch):
    args = _parse(monkeypatch, "--no-privileged-critic")

    assert args.privileged_critic is None


def test_privileged_critic_can_be_overridden(monkeypatch):
    replacement = Path("custom.pt")
    args = _parse(
        monkeypatch,
        "--privileged-critic",
        str(replacement),
        "--gamma",
        "0.97",
    )

    assert args.privileged_critic == replacement
    assert args.gamma == 0.97


def test_mismatched_critic_return_target_is_rejected():
    assert _validate_privileged_target_gamma(
        {"target_gamma": DEFAULT_GAMMA}, DEFAULT_GAMMA
    ) == pytest.approx(DEFAULT_GAMMA)
    assert _validate_privileged_target_gamma({}, 0.97) == 0.97

    with pytest.raises(ValueError, match="does not match PPO gamma"):
        _validate_privileged_target_gamma({"target_gamma": 0.97}, DEFAULT_GAMMA)
