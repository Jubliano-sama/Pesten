import pytest
import torch

from posthoc_checkpoint_ema import checkpoint_ema


def test_checkpoint_ema_is_champion_initialized_and_ordered() -> None:
    champion = {
        "weight": torch.tensor([0.0]),
        "counter": torch.tensor(3, dtype=torch.int64),
    }
    iterates = [
        {
            "weight": torch.tensor([2.0]),
            "counter": torch.tensor(3, dtype=torch.int64),
        },
        {
            "weight": torch.tensor([6.0]),
            "counter": torch.tensor(3, dtype=torch.int64),
        },
    ]
    state, audit = checkpoint_ema(champion, iterates, decay=0.5)
    assert torch.equal(state["weight"], torch.tensor([3.5]))
    assert torch.equal(state["counter"], torch.tensor(3))
    assert audit["ordered_iterate_count"] == 2


def test_checkpoint_ema_rejects_changed_integer_state() -> None:
    champion = {"counter": torch.tensor(0, dtype=torch.int64)}
    iterates = [{"counter": torch.tensor(1, dtype=torch.int64)}]
    with pytest.raises(ValueError, match="non-floating state changed"):
        checkpoint_ema(champion, iterates, decay=0.9)
