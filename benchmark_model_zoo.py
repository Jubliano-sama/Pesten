"""Short, common-data temporal architecture and decoder-adapter pilot.

The benchmark distills a frozen production PPO for only a few epochs.  It is
not a strength claim or a replacement for league evaluation.  Its purpose is
to compare model size, memory footprint, forward/training throughput, and
held-out imitation accuracy across broad architecture families.

No model download occurs implicitly.  ``--include-local-smollm`` loads the
reviewed local directory in cache-only mode and also creates a randomly
initialized backbone from that exact local config for a controlled comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
from typing import Callable, Sequence

import numpy as np
import torch

import card
import game
from ai_observation import (
    DIRECTION_INDEX,
    MODE_INDEX,
    OBSERVATION_SIZE as LEGACY_OBSERVATION_SIZE,
    PLAYER_HAND_SIZES_END,
    PLAYER_HAND_SIZES_START,
    PLAYER_INDEX,
    SORTS_PLAYED_START,
)
from ppo2 import StudentNet
from randomAgent import AgentRandom
from temporal_model_zoo import (
    CausalTransformerTemporalPolicy,
    DecoderLMTemporalPolicy,
    LargeRecurrentTemporalPolicy,
    SafeSequenceBatch,
    TinyRandomCausalDecoder,
    pack_safe_episodes,
    parameter_report,
)
from temporal_observation import ACTION_COUNT, DecisionInput
from temporal_puffer_agent import TemporalPestenAgent


DEFAULT_TEACHER = Path("ppo_actor_100run3.pth")
DEFAULT_SMOLLM = Path("experiments/models/SmolLM2-135M")
KNOWN_TEACHER_SHA256 = (
    "aac589b3c7300895604e6c77755631ab4a31c8bf9810768448ab2f3ff5bbe186"
)
_LEGACY_CARD_SUITS = torch.tensor(
    [card.sorts.index(card.Card(card_id).sort) for card_id in range(50)],
    dtype=torch.long,
)
_LEGACY_CARD_RANKS = torch.tensor(
    [card.Card(card_id).truenumber for card_id in range(50)],
    dtype=torch.long,
)


@dataclass(frozen=True, slots=True)
class ImitationSample:
    decision: DecisionInput
    action: int


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    family: str
    factory: Callable[[], torch.nn.Module]
    pretrained: bool = False
    note: str = ""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_to_legacy_observation(decision: DecisionInput) -> torch.Tensor:
    """Reconstruct the frozen PPO's public/private view from safe fields only."""
    if not isinstance(decision, DecisionInput):
        raise TypeError("teacher conversion accepts only DecisionInput")
    safe = decision.observation.values
    legacy = torch.zeros(LEGACY_OBSERVATION_SIZE, dtype=torch.float32)
    legacy[:50] = torch.tensor(safe[:50])

    if safe[93]:
        current_suit = card.sorts[
            max(
                range(len(card.sorts)),
                key=lambda index: safe[50 + index],
            )
        ]
        current_rank = max(range(14), key=lambda rank: safe[55 + rank])
        penalty_amount = safe[90]
        for card_id, count in enumerate(safe[:50]):
            if count <= 0:
                continue
            candidate = card.Card(card_id)
            compatible = (
                candidate.truenumber == current_rank
                if penalty_amount > 0
                else candidate.compatible(current_suit, current_rank)
            )
            if compatible:
                legacy[50 + card_id] = count
    else:
        # In ordinary play, including penalty and draw-retry calls, legal card
        # actions exactly equal the deployed PPO's compatible-card features.
        for card_id in range(50):
            if decision.legal_actions[card_id]:
                legacy[50 + card_id] = safe[card_id]

    legacy[SORTS_PLAYED_START : SORTS_PLAYED_START + 5] = torch.tensor(
        safe[82:87]
    )
    legacy[MODE_INDEX] = safe[93]
    legacy[PLAYER_HAND_SIZES_START:PLAYER_HAND_SIZES_END] = torch.tensor(
        safe[69:82]
    )
    player_count = max(2, int(round(safe[92])))
    legacy[PLAYER_INDEX] = round(safe[87] * (player_count - 1))
    legacy[DIRECTION_INDEX] = safe[89]
    return legacy


def safe_batch_to_legacy_observations(
    safe_observations: torch.Tensor,
    legal_actions: torch.Tensor,
) -> torch.Tensor:
    """Vectorized exact reconstruction of deployed PPO observations.

    Compatible-card features are observation features, not action-mask
    features.  This distinction matters during suit choice: only suit actions
    are legal, but ``RL_SupervisedAgent.changeSort`` retains the compatible
    card counts from its ordinary observation builder.
    """
    if safe_observations.ndim != 2 or safe_observations.shape[1] != 94:
        raise ValueError("safe_observations must have shape [N, 94]")
    if legal_actions.shape != (safe_observations.shape[0], ACTION_COUNT):
        raise ValueError("legal_actions must have shape [N, 55]")

    legacy = safe_observations.new_zeros(
        (safe_observations.shape[0], LEGACY_OBSERVATION_SIZE)
    )
    own_cards = safe_observations[:, :50]
    legacy[:, :50] = own_cards

    card_suits = _LEGACY_CARD_SUITS.to(safe_observations.device)
    card_ranks = _LEGACY_CARD_RANKS.to(safe_observations.device)
    current_suits = safe_observations[:, 50:55].argmax(dim=1)
    current_ranks = safe_observations[:, 55:69].argmax(dim=1)
    rank_matches = card_ranks.unsqueeze(0) == current_ranks.unsqueeze(1)
    ordinary_compatible = (
        rank_matches
        | (card_ranks.unsqueeze(0) == 0)
        | (card_ranks.unsqueeze(0) == 13)
        | (
            card_suits.unsqueeze(0)
            == current_suits.unsqueeze(1)
        )
    )
    under_penalty = safe_observations[:, 90].unsqueeze(1) > 0
    compatible = torch.where(
        under_penalty,
        rank_matches,
        ordinary_compatible,
    )
    legacy[:, 50:100] = own_cards * compatible.to(own_cards.dtype)

    legacy[:, SORTS_PLAYED_START : SORTS_PLAYED_START + 5] = (
        safe_observations[:, 82:87]
    )
    legacy[:, MODE_INDEX] = safe_observations[:, 93]
    legacy[:, PLAYER_HAND_SIZES_START:PLAYER_HAND_SIZES_END] = (
        safe_observations[:, 69:82]
    )
    player_count = safe_observations[:, 92].round().clamp(min=2)
    legacy[:, PLAYER_INDEX] = (
        safe_observations[:, 87] * (player_count - 1)
    ).round()
    legacy[:, DIRECTION_INDEX] = safe_observations[:, 89]
    return legacy


def load_frozen_teacher(path: Path) -> tuple[StudentNet, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"frozen PPO checkpoint not found: {path}")
    digest = file_sha256(path)
    if path.name == DEFAULT_TEACHER.name and digest != KNOWN_TEACHER_SHA256:
        raise RuntimeError(
            "production PPO checkpoint hash changed: "
            f"expected {KNOWN_TEACHER_SHA256}, got {digest}"
        )
    teacher = StudentNet()
    state = torch.load(path, map_location="cpu", weights_only=True)
    teacher.load_state_dict(state)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher, {
        "path": str(path),
        "sha256": digest,
        "architecture": "ppo2.StudentNet(DenseSkipNet 8x256)",
        "parameters": sum(p.numel() for p in teacher.parameters()),
        "frozen": True,
        "observation_size": LEGACY_OBSERVATION_SIZE,
        "action_count": ACTION_COUNT,
    }


def _teacher_recorder_factory(policy, teacher: StudentNet):
    class FrozenPPORecorder(TemporalPestenAgent):
        def __init__(self, game_instance):
            super().__init__(
                game_instance,
                policy,
                deterministic=True,
                seed=0,
            )
            self.imitation: list[ImitationSample] = []

        def _act(self, decision: DecisionInput) -> int:
            legacy = safe_to_legacy_observation(decision)
            with torch.inference_mode():
                logits = teacher(legacy.unsqueeze(0)).squeeze(0)
                legal = torch.tensor(
                    decision.legal_actions, dtype=torch.bool
                )
                logits = logits.masked_fill(
                    ~legal, torch.finfo(logits.dtype).min
                )
                action = int(logits.argmax().item())
            self.safe_history.append(decision)
            self.imitation.append(ImitationSample(decision, action))
            return action

    return FrozenPPORecorder


def collect_common_teacher_data(
    teacher: StudentNet,
    *,
    games: int,
    seed: int,
    max_turns: int,
) -> tuple[tuple[ImitationSample, ...], ...]:
    """Collect one immutable dataset reused by every architecture."""
    if games < 2:
        raise ValueError("at least two games are required for a held-out split")
    collector_policy = LargeRecurrentTemporalPolicy(
        hidden_size=8, num_layers=1, cell="gru"
    )
    recorder = _teacher_recorder_factory(collector_policy, teacher)
    episodes = []
    for game_index in range(games):
        deal_seed = seed + game_index
        random.seed(deal_seed)
        np.random.seed(deal_seed)
        seat = game_index % 2
        factories = (
            [recorder, AgentRandom]
            if seat == 0
            else [AgentRandom, recorder]
        )
        match = game.Game(factories)
        match.auto_simulate(max_turns=max_turns)
        samples = tuple(match.players[seat].imitation)
        if samples:
            episodes.append(samples)
    if len(episodes) < 2:
        raise RuntimeError("teacher collection produced fewer than two episodes")
    return tuple(episodes)


def split_episodes(
    episodes: Sequence[Sequence[ImitationSample]],
    *,
    evaluation_fraction: float = 0.34,
) -> tuple[tuple, tuple]:
    evaluation_count = max(1, round(len(episodes) * evaluation_fraction))
    evaluation_count = min(evaluation_count, len(episodes) - 1)
    return tuple(episodes[:-evaluation_count]), tuple(
        episodes[-evaluation_count:]
    )


def default_model_specs(
    *,
    max_context: int,
    include_local_smollm: bool = False,
    smollm_path: Path = DEFAULT_SMOLLM,
) -> list[ModelSpec]:
    specs = [
        ModelSpec(
            "gru_large_2x192",
            "recurrent",
            lambda: LargeRecurrentTemporalPolicy(
                hidden_size=192, num_layers=2, cell="gru"
            ),
            note="larger multi-layer GRU",
        ),
        ModelSpec(
            "lstm_large_2x192",
            "recurrent",
            lambda: LargeRecurrentTemporalPolicy(
                hidden_size=192, num_layers=2, cell="lstm"
            ),
            note="larger multi-layer LSTM",
        ),
        ModelSpec(
            "causal_transformer_3x128",
            "transformer",
            lambda: CausalTransformerTemporalPolicy(
                d_model=128,
                num_layers=3,
                num_heads=4,
                feedforward_size=384,
                max_context=max_context,
            ),
            note="compact causal Transformer over typed safe history",
        ),
        ModelSpec(
            "decoder_adapter_random_tiny",
            "decoder_adapter",
            lambda: DecoderLMTemporalPolicy(
                TinyRandomCausalDecoder(
                    hidden_size=96,
                    num_layers=2,
                    num_heads=4,
                    feedforward_size=256,
                ),
                max_context=max_context,
                source="random-small-config",
            ),
            note=(
                "runnable no-download adapter control; not a pretrained LLM"
            ),
        ),
    ]
    if include_local_smollm:
        if not (smollm_path / "config.json").is_file():
            raise FileNotFoundError(
                f"local SmolLM2 config is missing: {smollm_path}"
            )
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as error:
            raise RuntimeError(
                "--include-local-smollm requires transformers; use "
                "puffer4-native/.venv/bin/python in this checkout"
            ) from error

        def random_smollm():
            adapter_rng = torch.random.get_rng_state()
            config = AutoConfig.from_pretrained(
                smollm_path,
                local_files_only=True,
                trust_remote_code=False,
            )
            backbone = AutoModel.from_config(
                config, trust_remote_code=False
            )
            # Keep projection/head initialization identical to the pretrained
            # arm; random backbone construction otherwise advances the RNG.
            torch.random.set_rng_state(adapter_rng)
            return DecoderLMTemporalPolicy(
                backbone,
                max_context=max_context,
                freeze_backbone=True,
                source=f"random-config:{smollm_path}",
            )

        def pretrained_smollm():
            adapter_rng = torch.random.get_rng_state()
            backbone = AutoModel.from_pretrained(
                smollm_path,
                local_files_only=True,
                trust_remote_code=False,
                dtype="auto",
            )
            torch.random.set_rng_state(adapter_rng)
            return DecoderLMTemporalPolicy(
                backbone,
                max_context=max_context,
                freeze_backbone=True,
                source=f"pretrained:{smollm_path}",
            )

        specs.extend(
            (
                ModelSpec(
                    "smollm2_135m_random_frozen",
                    "decoder_adapter",
                    random_smollm,
                    pretrained=False,
                    note=(
                        "same local SmolLM2-135M config, random frozen "
                        "backbone; adapter and heads train"
                    ),
                ),
                ModelSpec(
                    "smollm2_135m_pretrained_frozen",
                    "decoder_adapter",
                    pretrained_smollm,
                    pretrained=True,
                    note=(
                        "local-only SmolLM2-135M pretrained frozen backbone; "
                        "adapter and heads train"
                    ),
                ),
            )
        )
    return specs


def imitation_metrics(
    model,
    batch: SafeSequenceBatch,
    *,
    batch_episodes: int | None = None,
) -> dict[str, float]:
    model.eval()
    if batch_episodes is None:
        batch_episodes = max(1, batch.tokens.shape[0])
    total_loss = 0.0
    correct = 0
    samples = 0
    with torch.inference_mode():
        for start in range(0, batch.tokens.shape[0], batch_episodes):
            mini_batch = _episode_slice(
                batch,
                start,
                min(start + batch_episodes, batch.tokens.shape[0]),
            )
            if not mini_batch.decision_count:
                continue
            logits, _ = model.forward_packed(mini_batch)
            total_loss += float(
                torch.nn.functional.cross_entropy(
                    logits, mini_batch.targets, reduction="sum"
                )
            )
            correct += int(
                (logits.argmax(dim=-1) == mini_batch.targets).sum()
            )
            samples += mini_batch.decision_count
    return {
        "cross_entropy": total_loss / samples if samples else math.nan,
        "accuracy": correct / samples if samples else math.nan,
        "samples": samples,
    }


def forward_benchmark(
    model,
    batch: SafeSequenceBatch,
    *,
    iterations: int,
    batch_episodes: int | None = None,
) -> dict[str, float]:
    model.eval()
    if batch_episodes is None:
        batch_episodes = max(1, batch.tokens.shape[0])

    def run_batches():
        for start in range(0, batch.tokens.shape[0], batch_episodes):
            mini_batch = _episode_slice(
                batch,
                start,
                min(start + batch_episodes, batch.tokens.shape[0]),
            )
            if mini_batch.decision_count:
                model.forward_packed(mini_batch)

    with torch.inference_mode():
        run_batches()
        _synchronize(batch.tokens.device)
        started = time.perf_counter()
        for _ in range(iterations):
            run_batches()
        _synchronize(batch.tokens.device)
    seconds = time.perf_counter() - started
    return {
        "seconds": seconds,
        "iterations": iterations,
        "decisions_per_second": (
            batch.decision_count * iterations / seconds
        ),
        "safe_tokens_per_second": batch.token_count * iterations / seconds,
    }


def short_imitation_train(
    model,
    batch: SafeSequenceBatch,
    *,
    epochs: int,
    learning_rate: float,
    batch_episodes: int,
) -> dict[str, float | int]:
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    model.train()
    _synchronize(batch.tokens.device)
    started = time.perf_counter()
    last_loss = math.nan
    optimizer_steps = 0
    processed_decisions = 0
    for _ in range(epochs):
        for start in range(0, batch.tokens.shape[0], batch_episodes):
            mini_batch = _episode_slice(
                batch, start, min(start + batch_episodes, batch.tokens.shape[0])
            )
            if not mini_batch.decision_count:
                continue
            logits, _ = model.forward_packed(mini_batch)
            loss = torch.nn.functional.cross_entropy(
                logits, mini_batch.targets
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            last_loss = float(loss.detach())
            optimizer_steps += 1
            processed_decisions += mini_batch.decision_count
    _synchronize(batch.tokens.device)
    seconds = time.perf_counter() - started
    return {
        "epochs": epochs,
        "optimizer_steps": optimizer_steps,
        "batch_episodes": batch_episodes,
        "seconds": seconds,
        "decisions_per_second": processed_decisions / seconds,
        "last_training_loss": last_loss,
    }


def benchmark_one(
    spec: ModelSpec,
    train_batch: SafeSequenceBatch,
    evaluation_batch: SafeSequenceBatch,
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    forward_iterations: int,
    learning_rate: float,
    batch_episodes: int,
    checkpoint_dir: Path | None = None,
) -> dict:
    torch.manual_seed(seed)
    rss_before = _current_rss_bytes()
    model = spec.factory().to(device)
    train_batch = train_batch.to(device)
    evaluation_batch = evaluation_batch.to(device)
    rss_after_create = _current_rss_bytes()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before = imitation_metrics(
        model,
        evaluation_batch,
        batch_episodes=batch_episodes,
    )
    forward = forward_benchmark(
        model,
        evaluation_batch,
        iterations=forward_iterations,
        batch_episodes=batch_episodes,
    )
    training = short_imitation_train(
        model,
        train_batch,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_episodes=batch_episodes,
    )
    after = imitation_metrics(
        model,
        evaluation_batch,
        batch_episodes=batch_episodes,
    )
    memory = {
        **parameter_report(model),
        "observed_rss_before_bytes": rss_before,
        "observed_rss_after_create_bytes": rss_after_create,
        "observed_rss_create_delta_bytes": max(
            0, rss_after_create - rss_before
        ),
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }
    result = {
        "name": spec.name,
        "family": spec.family,
        "pretrained": spec.pretrained,
        "note": spec.note,
        "architecture": getattr(model, "architecture", type(model).__name__),
        "backbone_source": getattr(model, "backbone_source", None),
        "memory": memory,
        "forward": forward,
        "training": training,
        "imitation_before": before,
        "imitation_after": after,
    }
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{spec.name}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        result["checkpoint"] = str(checkpoint_path)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_model_zoo(
    *,
    teacher_path: Path = DEFAULT_TEACHER,
    games: int = 6,
    seed: int = 7301,
    max_turns: int = 100,
    epochs: int = 2,
    forward_iterations: int = 3,
    learning_rate: float = 1e-3,
    batch_episodes: int = 4,
    device_name: str = "cpu",
    include_local_smollm: bool = False,
    smollm_path: Path = DEFAULT_SMOLLM,
    model_names: set[str] | None = None,
    checkpoint_dir: Path | None = None,
    teacher_workers: int = 1,
) -> dict:
    if epochs < 1 or forward_iterations < 1 or batch_episodes < 1:
        raise ValueError(
            "epochs, forward_iterations, and batch_episodes must be positive"
        )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    teacher, teacher_metadata = load_frozen_teacher(teacher_path)
    if teacher_workers > 1:
        # Local import avoids a top-level cycle: the worker utility reuses
        # ImitationSample and the recorder defined in this module.
        from temporal_teacher_parallel import (
            collect_teacher_episodes_parallel,
        )

        collected = collect_teacher_episodes_parallel(
            teacher_path=teacher_path,
            games=games,
            seed=seed,
            max_turns=max_turns,
            workers=teacher_workers,
        )
        episodes = tuple(episode for episode in collected if episode)
    else:
        episodes = collect_common_teacher_data(
            teacher, games=games, seed=seed, max_turns=max_turns
        )
    if len(episodes) < 2:
        raise RuntimeError("teacher collection produced fewer than two episodes")
    training_episodes, evaluation_episodes = split_episodes(episodes)
    train_batch = pack_safe_episodes(training_episodes)
    evaluation_batch = pack_safe_episodes(evaluation_episodes)
    max_context = max(
        32, train_batch.tokens.shape[1], evaluation_batch.tokens.shape[1]
    )
    specs = default_model_specs(
        max_context=max_context,
        include_local_smollm=include_local_smollm,
        smollm_path=smollm_path,
    )
    if model_names is not None:
        specs = [spec for spec in specs if spec.name in model_names]
        missing = model_names - {spec.name for spec in specs}
        if missing:
            raise ValueError(f"unknown/unavailable model names: {sorted(missing)}")
    results = [
        benchmark_one(
            spec,
            train_batch,
            evaluation_batch,
            device=device,
            seed=seed,
            epochs=epochs,
            forward_iterations=forward_iterations,
            learning_rate=learning_rate,
            batch_episodes=batch_episodes,
            checkpoint_dir=checkpoint_dir,
        )
        for spec in specs
    ]
    return {
        "schema_version": 1,
        "scope": (
            "short frozen-PPO imitation pilot; not a gameplay strength claim"
        ),
        "seed": seed,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
        "privacy": {
            "policy_input": "DecisionInput",
            "event_input": "PublicEventToken",
            "opponent_card_identities": False,
            "draw_pile_contents": False,
            "public_draw_representation": "actor plus count only",
            "continuous_safe_token_size": 100,
        },
        "teacher": teacher_metadata,
        "common_data": {
            "games_requested": games,
            "episodes_collected": len(episodes),
            "training_episodes": len(training_episodes),
            "evaluation_episodes": len(evaluation_episodes),
            "training_decisions": train_batch.decision_count,
            "evaluation_decisions": evaluation_batch.decision_count,
            "training_safe_tokens": train_batch.token_count,
            "evaluation_safe_tokens": evaluation_batch.token_count,
            "max_context": max_context,
            "max_turns_per_game": max_turns,
        },
        "training_recipe": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "optimizer": "AdamW",
            "batch_episodes": batch_episodes,
            "common_initialization_seed": seed,
            "teacher_targets": "frozen legal-masked PPO greedy actions",
            "teacher_collection_workers": teacher_workers,
        },
        "pretrained_requirements": {
            "implicit_downloads": False,
            "adapter_contract": (
                "project 100-d continuous safe token to decoder hidden size, "
                "call causal backbone with inputs_embeds, then 55-action and "
                "scalar-value heads"
            ),
            "local_smollm_path": str(smollm_path),
            "transformers_required_for_local_smollm": True,
            "recommended_command_python": "puffer4-native/.venv/bin/python",
        },
        "models": results,
    }


def _current_rss_bytes() -> int | None:
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def _episode_slice(
    batch: SafeSequenceBatch, start: int, end: int
) -> SafeSequenceBatch:
    selected = (
        (batch.decision_rows[:, 0] >= start)
        & (batch.decision_rows[:, 0] < end)
    )
    decision_rows = batch.decision_rows[selected].clone()
    decision_rows[:, 0] -= start
    return SafeSequenceBatch(
        tokens=batch.tokens[start:end],
        valid_tokens=batch.valid_tokens[start:end],
        decision_rows=decision_rows,
        legal_actions=batch.legal_actions[selected],
        targets=batch.targets[selected],
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--forward-iterations", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-episodes", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-local-smollm", action="store_true")
    parser.add_argument("--smollm-path", type=Path, default=DEFAULT_SMOLLM)
    parser.add_argument(
        "--models",
        nargs="+",
        help="optional exact model-name subset",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="optional directory for trained model state dictionaries",
    )
    parser.add_argument(
        "--teacher-workers",
        type=int,
        default=1,
        help="deterministic CPU worker processes for teacher collection",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = run_model_zoo(
        teacher_path=args.teacher,
        games=args.games,
        seed=args.seed,
        max_turns=args.max_turns,
        epochs=args.epochs,
        forward_iterations=args.forward_iterations,
        learning_rate=args.learning_rate,
        batch_episodes=args.batch_episodes,
        device_name=args.device,
        include_local_smollm=args.include_local_smollm,
        smollm_path=args.smollm_path,
        model_names=set(args.models) if args.models else None,
        checkpoint_dir=args.checkpoint_dir,
        teacher_workers=args.teacher_workers,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
