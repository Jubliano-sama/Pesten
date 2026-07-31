import pytest

import interactive_game


def test_parse_agent_names_normalizes_but_rejects_empty_entries():
    assert interactive_game.parse_agent_names(
        " Human, BEST ,smart "
    ) == ["human", "best", "smart"]
    with pytest.raises(ValueError, match="non-empty"):
        interactive_game.parse_agent_names("human,,best")


def test_resolve_agents_loads_best_once_and_preserves_order(monkeypatch):
    best_factory = object()
    calls = []

    def load_best():
        calls.append(True)
        return best_factory

    monkeypatch.setattr(
        interactive_game,
        "load_canonical_game_factory",
        load_best,
    )

    resolved = interactive_game.resolve_agent_classes(
        ["human", "best", "best", "random"]
    )

    assert resolved == [
        interactive_game.HumanAgent,
        best_factory,
        best_factory,
        interactive_game.randomAgent.AgentRandom,
    ]
    assert len(calls) == 1


def test_unknown_agent_fails_without_loading_best(monkeypatch):
    def unexpected_load():
        pytest.fail("best checkpoint must not load for an invalid agent list")

    monkeypatch.setattr(
        interactive_game,
        "load_canonical_game_factory",
        unexpected_load,
    )

    with pytest.raises(ValueError, match=r"unknown agent name\(s\): bst"):
        interactive_game.resolve_agent_classes(["human", "best", "bst"])


def test_best_checkpoint_failure_has_explicit_context(monkeypatch):
    def missing_checkpoint():
        raise FileNotFoundError("champion checkpoint not found: missing.pth")

    monkeypatch.setattr(
        interactive_game,
        "load_canonical_game_factory",
        missing_checkpoint,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "failed to load verified 'best' agent: FileNotFoundError: "
            "champion checkpoint not found"
        ),
    ):
        interactive_game.resolve_agent_classes(["human", "best"])
