import json
import random
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from export_production_policy import DEFAULT_SOURCE, export_production_policy
from production_temporal_policy import (
    ProductionSuitOrbitPolicy, TEACHER_CHECKPOINT_SHA256,
    encode_decision_token, encode_event_token,
    file_sha256, load_production_policy,
)


@pytest.fixture(scope="module")
def portable_artifact(tmp_path_factory):
    output = tmp_path_factory.mktemp("portable_teacher") / "teacher.pt"
    export_production_policy(DEFAULT_SOURCE, output)
    return output


def test_export_is_pinned_tensor_only_and_loads_frozen_without_puffer(portable_artifact):
    weights = torch.load(portable_artifact, map_location="cpu", weights_only=True)
    assert all(isinstance(value, torch.Tensor) and value.dtype == torch.float32 for value in weights.values())
    core = load_production_policy(portable_artifact)
    assert sum(p.numel() for p in core.parameters()) == 251384
    assert not core.training and not any(p.requires_grad for p in core.parameters())
    assert core.metadata["source_checkpoint_sha256"] == TEACHER_CHECKPOINT_SHA256
    assert core.metadata["weights_sha256"] == file_sha256(portable_artifact)
    command = """
import sys
import torch
from production_temporal_policy import ProductionSuitOrbitPolicy, load_production_policy
core = load_production_policy(sys.argv[1])
orbit = ProductionSuitOrbitPolicy(core)
assert 'pufferlib' not in sys.modules
assert 'temporal_training' not in sys.modules
assert 'temporal_puffer_agent' not in sys.modules
assert 'strongest_policy_search' not in sys.modules
assert core.initial_state(3).shape == (2, 3, 192)
assert orbit.initial_state(3).shape == (2, 3, 24 * 192)
with torch.inference_mode():
    orbit(torch.zeros(3, 100), orbit.initial_state(3))
"""
    subprocess.run([sys.executable, "-c", command, str(portable_artifact)], check=True)


def test_export_rejects_an_unpromoted_source_before_deserializing(tmp_path):
    source = tmp_path / "other.pt"
    source.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="pinned promoted"):
        export_production_policy(source, tmp_path / "output.pt")


@pytest.mark.parametrize("fault", ["weights", "source", "model", "dtype", "nonfinite"])
def test_loader_rejects_corrupt_or_mismatched_artifacts(portable_artifact, tmp_path, fault):
    target = tmp_path / "teacher.pt"
    shutil.copyfile(portable_artifact, target)
    manifest = json.loads(portable_artifact.with_suffix(".json").read_text())
    if fault == "weights":
        target.write_bytes(target.read_bytes() + b"corrupted")
    elif fault == "source":
        manifest["source_checkpoint_sha256"] = "0" * 64
    elif fault == "model":
        manifest["model"]["hidden_size"] = 384
    else:
        weights = torch.load(target, map_location="cpu", weights_only=True)
        name = next(iter(weights))
        if fault == "dtype":
            weights[name] = weights[name].half()
        else:
            weights[name].reshape(-1)[0] = float("nan")
        torch.save(weights, target)
        manifest["weights_sha256"] = file_sha256(target)
    target.with_suffix(".json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_production_policy(target)


@pytest.fixture(scope="module")
def native_teacher_and_histories():
    pufferlib = pytest.importorskip("pufferlib")
    if not str(getattr(pufferlib, "__version__", "")).startswith("4"):
        pytest.skip("native parity requires the PufferLib 4 reference runtime")
    import game
    from randomAgent import AgentRandom
    from temporal_puffer_agent import temporal_agent_factory
    from temporal_puffer_v4 import load_temporal_policy_checkpoint

    native, _ = load_temporal_policy_checkpoint(DEFAULT_SOURCE)
    native.eval()
    histories = []
    with torch.inference_mode():
        for players in range(2, 9):
            seed = 914100 + players
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            factory = temporal_agent_factory(native, deterministic=True, seed=seed)
            match = game.Game([factory] + [AgentRandom] * (players - 1))
            match.auto_simulate(max_turns=160)
            histories.append(tuple(match.players[0].safe_history))
    return native, histories


def test_native_parity_through_real_game_events_decisions_and_batches(
    portable_artifact, native_teacher_and_histories,
):
    torch.set_num_threads(1)
    native, histories = native_teacher_and_histories
    portable = load_production_policy(portable_artifact)
    decisions = suit_choices = 0
    event_kinds = set()
    chains = []
    with torch.inference_mode():
        for history in histories:
            reference_state = native.initial_state()
            state = portable.initial_state()
            chain = []
            for decision in history:
                for event in decision.public_events:
                    event_kinds.add(int(event.kind))
                    token = encode_event_token(event)
                    assert token == native._event_features_from_events((event,))[0]
                    chain.append(token)
                    _, _, state = portable(torch.tensor([token]), state)
                token = encode_decision_token(decision)
                assert token == native._decision_features(decision)
                chain.append(token)
                logits, value, state = portable(torch.tensor([token]), state)
                reference = native.step(decision, reference_state)
                reference_state = reference.state
                legal = torch.tensor(decision.legal_actions)
                masked = logits[0].masked_fill(~legal, torch.finfo(torch.float32).min)
                assert torch.equal(masked, reference.logits)
                assert torch.equal(value[0, 0], reference.value)
                assert torch.equal(state, reference_state[0])
                assert masked.argmax() == reference.logits.argmax()
                decisions += 1
                suit_choices += bool(decision.observation.values[93])
            chains.append(chain)
        assert decisions >= 100 and suit_choices >= 1 and event_kinds == {0, 1, 2, 3}

        for batch_size in (1, 2, 7):
            state = portable.initial_state(batch_size)
            native_state = native.initial_state(batch_size)
            for index in range(min(map(len, chains[:batch_size]))):
                tokens = torch.tensor([chain[index] for chain in chains[:batch_size]])
                logits, value, state = portable(tokens, state)
                expected_logits, expected_value, native_state = native.puffer_policy.forward_eval(tokens, native_state)
                assert torch.equal(logits, expected_logits)
                assert torch.equal(value, expected_value)
                assert torch.equal(state, native_state[0])


def test_full24_native_parity_through_carried_events_decisions_and_batches(
    portable_artifact, native_teacher_and_histories,
):
    from strongest_policy_search import NativeSuitOrbitCore

    torch.set_num_threads(1)
    native, histories = native_teacher_and_histories
    reference = NativeSuitOrbitCore(native.puffer_policy, alpha=1.0, orbit_size=24).eval()
    portable = ProductionSuitOrbitPolicy(load_production_policy(portable_artifact)).eval()
    assert portable.suit_permutations == reference.suit_permutations
    for name in ("observation_indices", "action_indices", "card_maps", "suit_maps"):
        assert torch.equal(getattr(portable, name), getattr(reference, name))
    chains = []
    decisions = suit_choices = 0
    event_kinds = set()
    with torch.inference_mode():
        for history in histories:
            state = portable.initial_state()
            expected_state = reference.initial_state(1, "cpu")
            chain = []
            for decision in history:
                for event in decision.public_events:
                    event_kinds.add(int(event.kind))
                    token = encode_event_token(event)
                    chain.append(token)
                    tokens = torch.tensor([token])
                    assert torch.equal(portable.relabel_tokens(tokens), reference.relabel_tokens(tokens))
                    _, _, state = portable(tokens, state)
                    expected_state = reference.forward_events_eval(tokens, expected_state)
                    assert torch.equal(state, expected_state[0])
                token = encode_decision_token(decision)
                chain.append(token)
                tokens = torch.tensor([token])
                logits, value, state = portable(tokens, state)
                expected_logits, expected_value, expected_state = reference.forward_eval(tokens, expected_state)
                assert torch.equal(logits, expected_logits)
                assert torch.equal(value, expected_value)
                assert torch.equal(state, expected_state[0])
                legal = torch.tensor([decision.legal_actions])
                assert torch.equal(
                    logits.masked_fill(~legal, torch.finfo(torch.float32).min).argmax(-1),
                    expected_logits.masked_fill(~legal, torch.finfo(torch.float32).min).argmax(-1),
                )
                decisions += 1
                suit_choices += bool(decision.observation.values[93])
            chains.append(chain)
        assert decisions >= 100 and suit_choices >= 1 and event_kinds == {0, 1, 2, 3}

        for batch_size in (1, 2, 7):
            state = portable.initial_state(batch_size)
            expected_state = reference.initial_state(batch_size, "cpu")
            for index in range(min(map(len, chains[:batch_size]))):
                tokens = torch.tensor([chain[index] for chain in chains[:batch_size]])
                logits, value, state = portable(tokens, state)
                expected_logits, expected_value, expected_state = reference.forward_eval(tokens, expected_state)
                assert torch.equal(logits, expected_logits)
                assert torch.equal(value, expected_value)
                assert torch.equal(state, expected_state[0])


@pytest.mark.parametrize("kind,card,suit,message", [
    (1, -1, 0, "card identity"),
    (1, 50, 4, "card identity"),
    (1, 12, 0, "card/suit mismatch"),
    (1, 48, 0, "card/suit mismatch"),
    (3, -1, -1, "ordinary suit"),
    (3, -1, 4, "ordinary suit"),
])
def test_orbit_rejects_invalid_public_event_suits(portable_artifact, kind, card, suit, message):
    core = ProductionSuitOrbitPolicy(load_production_policy(portable_artifact))
    token = torch.zeros(1, 100)
    token[0, 94] = kind / 3
    token[0, 96] = (card + 1) / 50
    token[0, 97] = (suit + 1) / 5
    with pytest.raises(ValueError, match=message):
        core(token, core.initial_state())


def test_runtime_rejects_wrong_state_shape_and_precision(portable_artifact):
    core = load_production_policy(portable_artifact)
    with pytest.raises(ValueError, match="shape"):
        core(torch.zeros(2, 100), core.initial_state(1))
    with pytest.raises(ValueError, match="float32"):
        core(torch.zeros(1, 100).half(), core.initial_state(1))
    orbit = ProductionSuitOrbitPolicy(core)
    with pytest.raises(ValueError, match="shape"):
        orbit(torch.zeros(1, 100), core.initial_state(1))
    with pytest.raises(ValueError, match="float32"):
        orbit(torch.zeros(1, 100).half(), orbit.initial_state(1))
    with pytest.raises(ValueError, match="float32"):
        orbit(torch.zeros(1, 100), orbit.initial_state(1).half())
