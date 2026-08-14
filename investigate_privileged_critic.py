#!/usr/bin/env python3
"""Collect, ablate, and scale causal omniscient PPO value estimators.

Examples
--------
Collect disjoint deal-seed splits with the promoted temporal actor::

    python investigate_privileged_critic.py collect \
      --output-dir live_training/privileged_critic_20260813

Run the predeclared model/feature screen on CUDA::

    python investigate_privileged_critic.py train \
      --output-dir live_training/privileged_critic_20260813 \
      --candidates screen --device cuda

The raw dataset and checkpoints are intentionally ignored by Git.  Compact
machine-readable findings are written below ``benchmarks/privileged_critic``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from benchmark_agents import load_ppo_agent
from privileged_critic import (
    AveragedPrivilegedCritic,
    DEFAULT_GAMMA,
    DATA_FIELDS,
    MAX_PILE_CARDS,
    CausalTemporalValueModel,
    CriticFeatureConfig,
    ROLE_CURRENT,
    ROLE_HISTORICAL,
    ROLE_NAMES,
    critic_loss,
    discounted_return_targets,
    load_privileged_critic_checkpoint,
    load_privileged_value_checkpoint,
    pack_episodes,
    parameter_count,
    public_feature_config,
    replay_privileged_trace,
    value_metrics,
)
from temporal_batched_speed import collect_league_batch_batched
from temporal_puffer_v4 import load_temporal_policy_checkpoint


DEFAULT_ACTOR = Path(
    "models/best/puffer_min_gru_192x2_plateau_teacher_epoch4387.pt"
)
DEFAULT_OUTPUT = Path("live_training/privileged_critic_20260813")
DEFAULT_REPORT = Path("benchmarks/privileged_critic/ablation_results.json")
DEFAULT_SCHEDULE = (
    "current_selfplay",
    "ppo_champion",
    "smart_rule",
    "smarter_rule",
    "historical_temporal",
)
SPLIT_OFFSETS = {
    "train": 0,
    "validation": 100_000_000,
    "test": 200_000_000,
    "confirmation": 300_000_000,
}


def atomic_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("at least one integer is required")
    return result


def collect_split(
    *,
    split: str,
    policy,
    champion,
    output_dir: Path,
    games_per_player_count: int,
    player_counts: Sequence[int],
    schedule: Sequence[str],
    base_seed: int,
    chunk_games: int,
    max_turns: int,
    gamma: float,
    chunk_shard_index: int,
    chunk_shard_count: int,
) -> dict[str, object]:
    split_dir = output_dir / "dataset" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    totals = {
        "games": 0,
        "episodes": 0,
        "decisions": 0,
        "wins": 0,
        "turn_limit_draws": 0,
    }
    chunks = []
    for player_count in player_counts:
        player_seed = (
            base_seed
            + SPLIT_OFFSETS[split]
            + int(player_count) * 1_000_000
        )
        for start in range(0, games_per_player_count, chunk_games):
            chunk_index = start // chunk_games
            if chunk_index % chunk_shard_count != chunk_shard_index:
                continue
            game_count = min(chunk_games, games_per_player_count - start)
            chunk_path = split_dir / (
                f"players_{player_count:02d}_games_{start:06d}_"
                f"{start + game_count:06d}.pt"
            )
            if chunk_path.exists():
                packed = torch.load(
                    chunk_path, map_location="cpu", weights_only=True
                )
                row = {
                    "path": str(chunk_path),
                    "player_count": int(player_count),
                    "game_start": start,
                    "games": game_count,
                    "episodes": int(len(packed["offsets"]) - 1),
                    "decisions": int(packed["returns"].numel()),
                    "wins": int(packed["won"].sum().item()),
                    "turn_limit_draws": 0,
                    "bytes": chunk_path.stat().st_size,
                    "seconds": 0.0,
                    "resumed": True,
                }
                chunks.append(row)
                for name in totals:
                    if name in row:
                        totals[name] += int(row[name])
                print(
                    json.dumps({"split": split, **row}, sort_keys=True),
                    flush=True,
                )
                continue
            traces: list[dict] = []
            chunk_started = time.perf_counter()
            _, metadata = collect_league_batch_batched(
                policy,
                champion,
                [policy],
                games=game_count,
                seed=player_seed + start,
                schedule_offset=start,
                schedule=schedule,
                player_count=player_count,
                max_turns=max_turns,
                trace_sink=traces,
            )
            if len(traces) != game_count:
                raise RuntimeError("rollout trace omitted a requested game")
            episodes = []
            for trace in traces:
                schedule_index = (
                    start + int(trace["absolute_game_index"])
                ) % len(schedule)
                episodes.extend(
                    replay_privileged_trace(
                        policy,
                        trace,
                        opponent_type=schedule[schedule_index],
                        player_count=player_count,
                        gamma=gamma,
                    )
                )
            packed = pack_episodes(episodes)
            temporary_chunk = chunk_path.with_suffix(".pt.tmp")
            torch.save(packed, temporary_chunk)
            temporary_chunk.replace(chunk_path)
            decisions = int(packed["returns"].numel())
            wins = int(packed["won"].sum().item())
            # The rollout helper's public metadata predates trace capture and
            # does not expose its internal draw counter.  The completed trace
            # has exactly the information needed, without consulting any RNG
            # state or replaying a second time.
            turn_limit_draws = sum(
                int(trace["winner"]) < 0
                and int(trace["turn_count"]) >= max_turns
                for trace in traces
            )
            row = {
                "path": str(chunk_path),
                "player_count": int(player_count),
                "game_start": start,
                "games": game_count,
                "episodes": len(episodes),
                "decisions": decisions,
                "wins": wins,
                "turn_limit_draws": turn_limit_draws,
                "bytes": chunk_path.stat().st_size,
                "seconds": time.perf_counter() - chunk_started,
            }
            chunks.append(row)
            for name in totals:
                if name in row:
                    totals[name] += int(row[name])
            print(json.dumps({"split": split, **row}, sort_keys=True), flush=True)
    manifest = {
        "split": split,
        "seed_namespace": {
            "base": base_seed,
            "split_offset": SPLIT_OFFSETS[split],
            "player_stride": 1_000_000,
        },
        "games_per_player_count": games_per_player_count,
        "player_counts": list(player_counts),
        "schedule": list(schedule),
        "gamma": gamma,
        "max_turns": max_turns,
        "chunk_shard": {
            "index": chunk_shard_index,
            "count": chunk_shard_count,
        },
        "totals": totals,
        "chunks": chunks,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(manifest, split_dir / "manifest.json")
    return manifest


def run_collection(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(args.torch_threads)
    policy, payload = load_temporal_policy_checkpoint(
        args.actor, device="cpu"
    )
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    champion = (
        load_ppo_agent(Path("ppo_actor_100run3.pth"), "cpu")
        if "ppo_champion" in args.schedule
        else None
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    collection_manifest = output_dir / "collection_manifest.json"
    if collection_manifest.exists() and not args.resume:
        raise FileExistsError(
            f"refusing to overwrite existing campaign {collection_manifest}; "
            "pass --resume to extend its chunk grid"
        )
    manifests = {}
    split_games = {
        "train": args.train_games_per_player_count,
        "validation": args.validation_games_per_player_count,
        "test": args.test_games_per_player_count,
    }
    for split, games in split_games.items():
        manifests[split] = collect_split(
            split=split,
            policy=policy,
            champion=champion,
            output_dir=output_dir,
            games_per_player_count=games,
            player_counts=args.player_counts,
            schedule=args.schedule,
            base_seed=args.seed,
            chunk_games=args.chunk_games,
            max_turns=args.max_turns,
            gamma=args.gamma,
            chunk_shard_index=args.chunk_shard_index,
            chunk_shard_count=args.chunk_shard_count,
        )
    result = {
        "format": "pesten-privileged-critic-collection-v1",
        "actor": {
            "path": str(args.actor),
            "sha256": file_sha256(args.actor),
            "epoch": payload.get("epoch"),
            "model": payload.get("model"),
        },
        "causal_boundary": (
            "pre-action physical state, current actor distribution, and "
            "current recurrent neural-seat memories; no action, reward, "
            "winner, future event, RNG state, or seed"
        ),
        "splits_are_deal_seed_disjoint": True,
        "splits": manifests,
    }
    atomic_json(result, collection_manifest)
    return result


def run_confirmation_collection(args: argparse.Namespace) -> dict[str, object]:
    """Collect a post-selection split in a fourth deal-seed namespace."""

    torch.set_num_threads(args.torch_threads)
    policy, payload = load_temporal_policy_checkpoint(
        args.actor, device="cpu"
    )
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    actor_sha256 = file_sha256(args.actor)
    output_dir = args.output_dir.resolve()
    campaign_manifest = output_dir / "collection_manifest.json"
    if not campaign_manifest.exists():
        raise FileNotFoundError(
            f"base critic campaign is missing: {campaign_manifest}"
        )
    campaign = json.loads(campaign_manifest.read_text(encoding="utf-8"))
    if campaign["actor"]["sha256"] != actor_sha256:
        raise ValueError("confirmation actor differs from the base campaign")
    confirmation_manifest = output_dir / "confirmation_manifest.json"
    confirmation_dir = output_dir / "dataset" / "confirmation"
    if confirmation_manifest.exists() and not args.resume:
        raise FileExistsError(
            f"refusing to overwrite {confirmation_manifest}; pass --resume"
        )
    if (
        confirmation_dir.exists()
        and any(confirmation_dir.glob("players_*.pt"))
        and not args.resume
    ):
        raise FileExistsError(
            f"confirmation chunks already exist in {confirmation_dir}; "
            "pass --resume"
        )
    champion = (
        load_ppo_agent(Path("ppo_actor_100run3.pth"), "cpu")
        if "ppo_champion" in args.schedule
        else None
    )
    split = collect_split(
        split="confirmation",
        policy=policy,
        champion=champion,
        output_dir=output_dir,
        games_per_player_count=args.games_per_player_count,
        player_counts=args.player_counts,
        schedule=args.schedule,
        base_seed=args.seed,
        chunk_games=args.chunk_games,
        max_turns=args.max_turns,
        gamma=args.gamma,
        chunk_shard_index=0,
        chunk_shard_count=1,
    )
    result = {
        "format": "pesten-privileged-critic-confirmation-v1",
        "actor": {
            "path": str(args.actor),
            "sha256": actor_sha256,
            "epoch": payload.get("epoch"),
            "model": payload.get("model"),
        },
        "selection_state": (
            "architecture and optimization settings locked before this split "
            "was generated"
        ),
        "base_split_offsets": {
            name: SPLIT_OFFSETS[name]
            for name in ("train", "validation", "test")
        },
        "confirmation": split,
    }
    atomic_json(result, confirmation_manifest)
    return result


def merge_collections(args: argparse.Namespace) -> dict[str, object]:
    """Hard-link disjoint player-count shards into one audited campaign."""

    output_dir = args.output_dir.resolve()
    shard_dirs = tuple(
        dict.fromkeys(
            [output_dir, *(path.resolve() for path in args.shards)]
        )
    )
    manifests = []
    for directory in shard_dirs:
        path = directory / "collection_manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"missing shard manifest: {path}")
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    reference = manifests[0]
    for manifest in manifests[1:]:
        if manifest["format"] != reference["format"]:
            raise ValueError("collection shards use different formats")
        if manifest["actor"]["sha256"] != reference["actor"]["sha256"]:
            raise ValueError("collection shards use different actors")
        if manifest["causal_boundary"] != reference["causal_boundary"]:
            raise ValueError("collection shards use different causal contracts")

    merged_splits = {}
    for split in ("train", "validation", "test"):
        player_counts: list[int] = []
        chunks = []
        totals = {
            "games": 0,
            "episodes": 0,
            "decisions": 0,
            "wins": 0,
            "turn_limit_draws": 0,
        }
        split_reference = reference["splits"][split]
        for manifest in manifests:
            shard = manifest["splits"][split]
            comparable = (
                "seed_namespace",
                "games_per_player_count",
                "schedule",
                "gamma",
                "max_turns",
            )
            if any(shard[key] != split_reference[key] for key in comparable):
                raise ValueError(
                    f"collection shard settings differ for split {split}"
                )
            overlap = set(player_counts) & set(shard["player_counts"])
            if overlap:
                raise ValueError(
                    f"player-count shards overlap in {split}: {sorted(overlap)}"
                )
            player_counts.extend(int(value) for value in shard["player_counts"])
            for row in shard["chunks"]:
                source = Path(row["path"]).resolve()
                destination = output_dir / "dataset" / split / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if source != destination:
                    if destination.exists():
                        if (
                            destination.stat().st_size != source.stat().st_size
                            or file_sha256(destination) != file_sha256(source)
                        ):
                            raise FileExistsError(
                                f"non-identical merged chunk: {destination}"
                            )
                    else:
                        os.link(source, destination)
                merged_row = dict(row)
                merged_row["path"] = str(destination)
                chunks.append(merged_row)
            for key in totals:
                totals[key] += int(shard["totals"][key])
        merged_splits[split] = {
            **{
                key: split_reference[key]
                for key in (
                    "split",
                    "seed_namespace",
                    "games_per_player_count",
                    "schedule",
                    "gamma",
                    "max_turns",
                )
            },
            "player_counts": sorted(player_counts),
            "totals": totals,
            "chunks": sorted(
                chunks,
                key=lambda row: (row["player_count"], row["game_start"]),
            ),
            "elapsed_seconds": sum(
                float(manifest["splits"][split]["elapsed_seconds"])
                for manifest in manifests
            ),
        }
    merged = {
        **{
            key: reference[key]
            for key in (
                "format",
                "actor",
                "causal_boundary",
                "splits_are_deal_seed_disjoint",
            )
        },
        "merged_from": [str(directory) for directory in shard_dirs],
        "splits": merged_splits,
    }
    atomic_json(merged, output_dir / "collection_manifest.json")
    return merged


class PackedEpisodeCorpus:
    """A memory-resident index over compact dataset chunks."""

    def __init__(self, directory: Path) -> None:
        paths = sorted(directory.glob("players_*.pt"))
        if not paths:
            raise FileNotFoundError(f"no critic dataset chunks in {directory}")
        self.payloads = [
            torch.load(
                path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            for path in paths
        ]
        self.entries: list[tuple[int, int, int, int]] = []
        self.actor_memory_size = int(self.payloads[0]["actor_memory_size"])
        for payload_index, payload in enumerate(self.payloads):
            if int(payload["actor_memory_size"]) != self.actor_memory_size:
                raise ValueError("dataset chunks disagree on actor memory size")
            offsets = payload["offsets"].tolist()
            for episode_index, (start, end) in enumerate(
                zip(offsets, offsets[1:])
            ):
                self.entries.append(
                    (payload_index, episode_index, int(start), int(end))
                )

    def __len__(self) -> int:
        return len(self.entries)

    def length(self, index: int) -> int:
        _, _, start, end = self.entries[index]
        return end - start

    def episode(self, index: int) -> dict[str, torch.Tensor]:
        payload_index, episode_index, start, end = self.entries[index]
        payload = self.payloads[payload_index]
        result = {
            name: payload[name][start:end]
            for name in DATA_FIELDS
        }
        result["player_count"] = payload["player_count"][episode_index]
        result["deal_seed"] = payload["deal_seed"][episode_index]
        result["seat"] = payload["seat"][episode_index]
        return result


def audit_corpus(corpus: PackedEpisodeCorpus) -> dict[str, object]:
    """Check packed physical-state invariants and summarize its coverage."""

    lengths = np.asarray(
        [corpus.length(index) for index in range(len(corpus))],
        dtype=np.int64,
    )
    deal_seeds: set[int] = set()
    player_counts: dict[str, int] = {}
    seats: dict[str, int] = {}
    wins = 0
    decisions = 0
    conservation_errors = 0
    own_hand_errors = 0
    hand_order_errors = 0
    actor_memory_errors = 0
    pile_count_errors = 0
    probability_errors = 0
    nonfinite_errors = 0
    for payload in corpus.payloads:
        deal_seeds.update(int(value) for value in payload["deal_seed"].tolist())
        for value in payload["player_count"].tolist():
            key = str(int(value))
            player_counts[key] = player_counts.get(key, 0) + 1
        for value in payload["seat"].tolist():
            key = str(int(value))
            seats[key] = seats.get(key, 0) + 1
        wins += int(payload["won"].sum().item())
        decisions += int(payload["returns"].numel())

        hands = payload["hands"].to(torch.int32).sum(dim=(1, 2))
        draw_cards = payload["draw_counts"].to(torch.int32).sum(dim=1)
        discard_cards = payload["discard_counts"].to(torch.int32).sum(dim=1)
        deck_count = torch.round(
            payload["scalars"][:, 30].float() * 2.0
        ).to(torch.int32)
        conservation_errors += int(
            ((hands + draw_cards + discard_cards) != deck_count * 54)
            .sum()
            .item()
        )
        own_hand_errors += int(
            (
                payload["safe_observation"][:, :50].float()
                != payload["hands"][:, 0].float()
            )
            .any(dim=1)
            .sum()
            .item()
        )
        order = payload["hand_order"].to(torch.int64)
        order_lengths = payload["hand_lengths"].to(torch.int64)
        positions = torch.arange(order.shape[-1]).view(1, 1, -1)
        order_valid = positions < order_lengths.unsqueeze(-1)
        reconstructed = torch.zeros_like(
            payload["hands"], dtype=torch.int16
        )
        reconstructed.scatter_add_(
            2,
            order.clamp(min=0, max=49).long(),
            order_valid.to(torch.int16),
        )
        hand_order_errors += int(
            (
                (reconstructed != payload["hands"].to(torch.int16))
                .any(dim=2)
                | (order_lengths != payload["hands"].sum(dim=2))
                | (order_valid & ((order < 0) | (order >= 50))).any(dim=2)
                | ((~order_valid) & (order != -1)).any(dim=2)
            )
            .sum()
            .item()
        )
        memory_mask = payload["player_actor_memory_mask"].bool()
        quantized_memory = payload["player_actor_memory"].float()
        memory_scale = payload["player_actor_memory_scale"].float()
        reconstructed_own_memory = (
            quantized_memory[:, 0] * memory_scale[:, 0].unsqueeze(-1)
        )
        own_memory = payload["actor_memory"].float()
        # Half-step int8 rounding plus fp16 scale quantization (amplified by
        # values as large as 127) stays below 0.58 of one stored scale.
        own_tolerance = memory_scale[:, 0].unsqueeze(-1) * 0.58 + 2e-3
        actor_memory_errors += int(
            (
                ~memory_mask[:, 0]
                | (memory_mask & ~payload["player_mask"].bool()).any(dim=1)
                | (
                    memory_mask
                    != (
                        (payload["role_ids"] == ROLE_CURRENT)
                        | (payload["role_ids"] == ROLE_HISTORICAL)
                    )
                ).any(dim=1)
                | (
                    (~memory_mask).unsqueeze(-1)
                    & (payload["player_actor_memory"] != 0)
                ).any(dim=(1, 2))
                | (
                    (reconstructed_own_memory - own_memory).abs()
                    > own_tolerance
                ).any(dim=1)
                | (~torch.isfinite(memory_scale)).any(dim=1)
            )
            .sum()
            .item()
        )
        pile_count_errors += int(
            (
                (draw_cards != payload["draw_length"].to(torch.int32))
                | (
                    discard_cards
                    != payload["discard_length"].to(torch.int32)
                )
            )
            .sum()
            .item()
        )
        probability_errors += int(
            (
                (
                    payload["actor_probabilities"].float().sum(dim=1)
                    - 1.0
                ).abs()
                > 2e-3
            )
            .sum()
            .item()
        )
        nonfinite_errors += int(
            (~torch.isfinite(payload["returns"].float())).sum().item()
        )

    invariant_errors = {
        "physical_card_conservation": conservation_errors,
        "safe_observation_own_hand": own_hand_errors,
        "hand_order_matches_counts": hand_order_errors,
        "quantized_actor_memories": actor_memory_errors,
        "pile_lengths_match_composition": pile_count_errors,
        "actor_probability_sum": probability_errors,
        "nonfinite_returns": nonfinite_errors,
    }
    return {
        "episodes": int(len(corpus)),
        "unique_deal_seeds": len(deal_seeds),
        "decisions": decisions,
        "wins": wins,
        "episode_length": {
            "min": int(lengths.min()),
            "median": float(np.median(lengths)),
            "p95": float(np.quantile(lengths, 0.95)),
            "max": int(lengths.max()),
        },
        "episodes_by_player_count": player_counts,
        "episodes_by_absolute_seat": seats,
        "invariant_errors": invariant_errors,
        "all_invariants_hold": not any(invariant_errors.values()),
        "deal_seeds": deal_seeds,
    }


def collate_episodes(
    corpus: PackedEpisodeCorpus,
    indices: Sequence[int],
    *,
    target_gamma: float | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    episodes = [corpus.episode(index) for index in indices]
    max_length = max(len(episode["returns"]) for episode in episodes)
    mask = torch.zeros(len(episodes), max_length, dtype=torch.bool)
    for row, episode in enumerate(episodes):
        mask[row, : len(episode["returns"])] = True
    batch: dict[str, torch.Tensor] = {}
    for name in DATA_FIELDS:
        sample = episodes[0][name]
        fill = (
            -1
            if name
            in ("hand_order", "draw_order", "discard_order", "recent_plays")
            else 0
        )
        target = torch.full(
            (len(episodes), max_length, *sample.shape[1:]),
            fill_value=fill,
            dtype=sample.dtype,
        )
        for row, episode in enumerate(episodes):
            length = len(episode[name])
            target[row, :length] = episode[name]
        batch[name] = target
    batch["player_count"] = torch.stack(
        [episode["player_count"] for episode in episodes]
    )
    batch["deal_seed"] = torch.stack(
        [episode["deal_seed"] for episode in episodes]
    )
    batch["seat"] = torch.stack([episode["seat"] for episode in episodes])
    if target_gamma is not None:
        batch["returns"] = discounted_return_targets(
            batch["remaining_decisions"],
            batch["win_target"],
            gamma=target_gamma,
        )
    return batch, mask


def batches_for_epoch(
    corpus: PackedEpisodeCorpus,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
    max_batch_decisions: int | None = None,
) -> list[list[int]]:
    indices = list(range(len(corpus)))
    rng = random.Random(seed)
    if shuffle:
        # Randomize equal-length examples, then globally bucket by length.
        # Local random buckets occasionally paired a 300-decision trajectory
        # with 15 unrelated episodes and multiplied every 108-card tensor.
        rng.shuffle(indices)
    indices.sort(key=corpus.length)
    decision_budget = (
        max(1, int(max_batch_decisions))
        if max_batch_decisions is not None
        else batch_size * 128
    )
    batches: list[list[int]] = []
    current: list[int] = []
    for index in indices:
        length = corpus.length(index)
        padded_decisions = length * (len(current) + 1)
        if current and (
            len(current) >= batch_size
            or padded_decisions > decision_budget
        ):
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    if shuffle:
        rng.shuffle(batches)
    return batches


def move_batch(
    batch: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    return (
        {
            name: value.to(device=device, non_blocking=True)
            for name, value in batch.items()
            if name not in ("deal_seed", "seat")
        },
        mask.to(device=device, non_blocking=True),
    )


REPRESENTATION_SUFFIXES = {
    "_moments_exact": ("counts_moments", "exact"),
    "_moments_rank_suit": ("counts_moments", "rank_suit"),
    "_token_gru_exact": ("token_gru", "exact"),
    "_token_gru_rank_suit": ("token_gru", "rank_suit"),
    "_token_bigru_exact": ("token_bigru", "exact"),
    "_token_bigru_rank_suit": ("token_bigru", "rank_suit"),
}


def _candidate_parts(
    name: str,
) -> tuple[str, str, str, str]:
    """Return backbone, feature name, hand encoder, and card tokenization."""

    backbones = ("mlp", "gru", "lstm", "transformer")
    backbone = next(
        (item for item in backbones if name.endswith(f"_{item}")), None
    )
    if backbone is None:
        raise ValueError(f"unknown critic candidate {name!r}")
    feature_name = name.removesuffix(f"_{backbone}")
    for suffix, representation in sorted(
        REPRESENTATION_SUFFIXES.items(), key=lambda row: -len(row[0])
    ):
        if feature_name.endswith(suffix):
            return (
                backbone,
                feature_name.removesuffix(suffix),
                representation[0],
                representation[1],
            )
    return backbone, feature_name, "counts_moments", "exact"


def candidate_representation(name: str) -> tuple[str, str]:
    _, _, hand_encoder, card_tokenization = _candidate_parts(name)
    return hand_encoder, card_tokenization


def candidate_spec(name: str) -> tuple[str, CriticFeatureConfig]:
    """Resolve an architecture plus independently maskable information set.

    Feature names deliberately work with every screened backbone.  This avoids
    attributing a feature's usefulness to a GRU-specific optimization result.
    """

    backbone, feature_name, _, _ = _candidate_parts(name)
    full = CriticFeatureConfig()
    if feature_name == "public":
        return backbone, public_feature_config()
    if feature_name == "full":
        return backbone, full
    if feature_name == "selected":
        # Current physical state plus policy state is Markov.  The actor-safe
        # observation duplicates those fields, deep draw order adds weak signal,
        # and pre-recycle discard order has no transition meaning because the
        # engine shuffles it.  Retain draw/discard composition and the next four
        # draws, which the validation ablations found to be the useful subset.
        return backbone, replace(
            full,
            safe_observation=False,
            draw_top_n=4,
            discard_order=False,
            discard_top_n=0,
        )
    replacements: dict[str, dict[str, object]] = {
        "no_safe_observation": {"safe_observation": False},
        "no_policy": {"policy_context": False},
        "no_public_value": {"public_value": False},
        "no_hands": {"opponent_hands": False, "hand_order": False},
        "no_hand_order": {"hand_order": False},
        "no_opponent_memories": {"opponent_memories": False},
        "no_draw": {
            "draw_composition": False,
            "draw_order": False,
            "draw_top_n": 0,
        },
        "no_discard": {
            "discard_composition": False,
            "discard_order": False,
            "discard_top_n": 0,
        },
        "no_history": {"played_history": False, "played_top_n": 0},
        "no_roles": {"player_roles": False},
        "no_controls": {"engine_controls": False},
    }
    if feature_name in replacements:
        return backbone, replace(full, **replacements[feature_name])
    if feature_name.startswith("draw_top_"):
        top_n = int(feature_name.removeprefix("draw_top_"))
        return backbone, replace(
            full, draw_top_n=top_n, draw_order=top_n > 0
        )
    raise ValueError(f"unknown critic candidate {name!r}")


SCREEN_CANDIDATES = (
    "public_gru",
    "full_mlp",
    "full_gru",
    "full_lstm",
    "full_transformer",
    "no_safe_observation_gru",
    "no_policy_gru",
    "no_public_value_gru",
    "no_hands_gru",
    "no_hand_order_gru",
    "no_opponent_memories_gru",
    "no_draw_gru",
    "no_discard_gru",
    "no_history_gru",
    "no_roles_gru",
    "no_controls_gru",
    "draw_top_0_gru",
    "draw_top_1_gru",
    "draw_top_4_gru",
    "draw_top_8_gru",
    "draw_top_16_gru",
    "draw_top_32_gru",
    "draw_top_108_gru",
)


def evaluate_model(
    model: torch.nn.Module,
    corpus: PackedEpisodeCorpus,
    *,
    device: torch.device,
    batch_size: int,
    amp: bool,
    return_predictions: bool = False,
    target_gamma: float | None = None,
) -> dict[str, object]:
    model.eval()
    predictions = []
    targets = []
    public_values = []
    win_probabilities = []
    win_targets = []
    remaining_predictions = []
    remaining_targets = []
    player_counts = []
    opponent_types = []
    episode_model_mse = []
    episode_public_mse = []
    episode_model_squared_error_sum = []
    episode_public_squared_error_sum = []
    episode_decisions = []
    episode_deal_seed = []
    inference_seconds = 0.0
    with torch.inference_mode():
        for indices in batches_for_epoch(
            corpus, batch_size=batch_size, seed=0, shuffle=False
        ):
            host_batch, host_mask = collate_episodes(
                corpus, indices, target_gamma=target_gamma
            )
            batch, mask = move_batch(host_batch, host_mask, device)
            started = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                outputs = model(batch, mask=mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - started
            predicted = outputs["value"].float().cpu()
            predicted_win = outputs["win_logit"].float().sigmoid().cpu()
            predicted_remaining = (
                torch.expm1(
                    outputs["remaining_log"].float().cpu()
                    * math.log1p(512.0)
                )
                .clamp(min=0.0, max=1024.0)
            )
            for row, corpus_index in enumerate(indices):
                length = corpus.length(corpus_index)
                row_prediction = predicted[row, :length].numpy()
                row_target = host_batch["returns"][row, :length].float().numpy()
                row_public = (
                    host_batch["public_value"][row, :length].float().numpy()
                )
                predictions.append(row_prediction)
                targets.append(row_target)
                public_values.append(row_public)
                win_probabilities.append(
                    predicted_win[row, :length].numpy()
                )
                win_targets.append(
                    host_batch["win_target"][row, :length].float().numpy()
                )
                remaining_predictions.append(
                    predicted_remaining[row, :length].numpy()
                )
                remaining_targets.append(
                    host_batch["remaining_decisions"][row, :length]
                    .float()
                    .numpy()
                )
                player_counts.append(
                    int(host_batch["player_count"][row].item())
                )
                role_ids = host_batch["role_ids"][row, 0].tolist()
                opponent_roles = [
                    int(role)
                    for role in role_ids
                    if int(role) not in (0, ROLE_CURRENT)
                ]
                opponent_types.append(
                    ROLE_NAMES.get(opponent_roles[0], "unknown")
                    if opponent_roles
                    else "current_selfplay"
                )
                episode_model_mse.append(
                    float(np.mean((row_prediction - row_target) ** 2))
                )
                episode_public_mse.append(
                    float(np.mean((row_public - row_target) ** 2))
                )
                episode_model_squared_error_sum.append(
                    float(np.sum((row_prediction - row_target) ** 2))
                )
                episode_public_squared_error_sum.append(
                    float(np.sum((row_public - row_target) ** 2))
                )
                episode_decisions.append(length)
                episode_deal_seed.append(
                    int(host_batch["deal_seed"][row].item())
                )
    flat_prediction = np.concatenate(predictions)
    flat_target = np.concatenate(targets)
    flat_public = np.concatenate(public_values)
    flat_win_probability = np.concatenate(win_probabilities)
    flat_win_target = np.concatenate(win_targets)
    flat_remaining_prediction = np.concatenate(remaining_predictions)
    flat_remaining_target = np.concatenate(remaining_targets)
    by_player_count = {}
    for count in sorted(set(player_counts)):
        selected = [
            index for index, value in enumerate(player_counts) if value == count
        ]
        by_player_count[str(count)] = value_metrics(
            np.concatenate([predictions[index] for index in selected]),
            np.concatenate([targets[index] for index in selected]),
        )
    by_opponent_type = {}
    for opponent_type in sorted(set(opponent_types)):
        selected = [
            index
            for index, value in enumerate(opponent_types)
            if value == opponent_type
        ]
        by_opponent_type[opponent_type] = value_metrics(
            np.concatenate([predictions[index] for index in selected]),
            np.concatenate([targets[index] for index in selected]),
        )
    result = {
        "model": value_metrics(flat_prediction, flat_target),
        "inherited_public_head": value_metrics(flat_public, flat_target),
        "auxiliary": {
            "win_brier": float(
                np.mean((flat_win_probability - flat_win_target) ** 2)
            ),
            "win_accuracy_at_half": float(
                np.mean(
                    (flat_win_probability >= 0.5) == (flat_win_target >= 0.5)
                )
            ),
            "remaining_decisions_mae": float(
                np.mean(
                    np.abs(flat_remaining_prediction - flat_remaining_target)
                )
            ),
        },
        "by_player_count": by_player_count,
        "by_opponent_type": by_opponent_type,
        "episodes": len(predictions),
        "decisions": int(flat_target.size),
        "inference_seconds": inference_seconds,
        "decisions_per_second": flat_target.size / max(inference_seconds, 1e-9),
        "episode_model_mse": episode_model_mse,
        "episode_public_mse": episode_public_mse,
        "episode_model_squared_error_sum": (
            episode_model_squared_error_sum
        ),
        "episode_public_squared_error_sum": (
            episode_public_squared_error_sum
        ),
        "episode_decisions": episode_decisions,
        "episode_deal_seed": episode_deal_seed,
    }
    if return_predictions:
        result["_prediction"] = flat_prediction
        result["_target"] = flat_target
        result["_public_value"] = flat_public
        result["_player_count"] = np.concatenate(
            [
                np.full(len(prediction), count, dtype=np.int16)
                for prediction, count in zip(predictions, player_counts)
            ]
        )
    return result


class CalibratedPublicValue(torch.nn.Module):
    """A fair validation-calibrated version of the inherited actor value."""

    def __init__(
        self,
        scale: Sequence[float],
        bias: Sequence[float],
    ) -> None:
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(scale).float())
        self.register_buffer("bias", torch.as_tensor(bias).float())

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del mask
        player_count = batch["player_mask"].sum(dim=-1).long().clamp(0, 13)
        value = (
            batch["public_value"].float() * self.scale[player_count]
            + self.bias[player_count]
        )
        return {
            "value": value,
            "win_logit": torch.zeros_like(value),
            "remaining_log": torch.zeros_like(value),
        }


def affine_calibration_parameters(
    prediction: np.ndarray,
    target: np.ndarray,
    player_count: np.ndarray,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit bounded affine parameters globally or independently per count."""

    if mode not in ("none", "global", "player_count"):
        raise ValueError(f"unknown value calibration mode {mode!r}")
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    player_count = np.asarray(player_count, dtype=np.int16)
    if not (
        prediction.shape == target.shape == player_count.shape
        and prediction.size
    ):
        raise ValueError("calibration arrays must be non-empty and aligned")
    scale = np.ones(14, dtype=np.float32)
    bias = np.zeros(14, dtype=np.float32)

    def fit(selected: np.ndarray) -> tuple[float, float]:
        x = prediction[selected]
        y = target[selected]
        variance = float(np.var(x))
        slope = (
            float(np.mean((x - x.mean()) * (y - y.mean())) / variance)
            if variance > 1e-12
            else 1.0
        )
        # A non-negative bounded correction is safer under policy drift than
        # an unconstrained validation fit.
        slope = float(np.clip(slope, 0.0, 2.0))
        intercept = float(y.mean() - slope * x.mean())
        return slope, intercept

    rows = {}
    if mode == "global":
        slope, intercept = fit(np.ones(len(prediction), dtype=np.bool_))
        scale[2:] = slope
        bias[2:] = intercept
        rows["global"] = {
            "samples": int(len(prediction)),
            "scale": slope,
            "bias": intercept,
        }
    elif mode == "player_count":
        for count in sorted(int(value) for value in np.unique(player_count)):
            selected = player_count == count
            slope, intercept = fit(selected)
            scale[count] = slope
            bias[count] = intercept
            rows[str(count)] = {
                "samples": int(selected.sum()),
                "scale": slope,
                "bias": intercept,
            }
    calibrated = prediction * scale[player_count] + bias[player_count]
    row = {
        "mode": mode,
        "fit_split": "validation",
        "groups": rows,
        "mse_before": float(np.mean((prediction - target) ** 2)),
        "mse_after_fit": float(np.mean((calibrated - target) ** 2)),
    }
    return scale, bias, row


def fit_value_calibration(
    model,
    prediction: np.ndarray,
    target: np.ndarray,
    player_count: np.ndarray,
    *,
    mode: str,
) -> dict[str, object]:
    """Fit a validation-only affine correction without touching model weights."""

    scale, bias, row = affine_calibration_parameters(
        prediction, target, player_count, mode=mode
    )
    model.set_value_calibration(scale, bias)
    return row


def train_candidate(
    name: str,
    *,
    train: PackedEpisodeCorpus,
    validation: PackedEpisodeCorpus,
    test: PackedEpisodeCorpus | None,
    output_dir: Path,
    device: torch.device,
    hidden_size: int,
    num_layers: int,
    group_width: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    win_coefficient: float,
    remaining_coefficient: float,
    dropout: float,
    seed: int,
    patience: int,
    min_epochs: int,
    amp: bool,
    run_label: str,
    calibration: str,
    target_gamma: float,
) -> dict[str, object]:
    backbone, features = candidate_spec(name)
    hand_encoder, card_tokenization = candidate_representation(name)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = CausalTemporalValueModel(
        actor_memory_size=train.actor_memory_size,
        features=features,
        backbone=backbone,
        hidden_size=hidden_size,
        num_layers=num_layers,
        group_width=group_width,
        dropout=dropout,
        hand_encoder=hand_encoder,
        card_tokenization=card_tokenization,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    candidate_dir = output_dir / "models" / run_label / name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = candidate_dir / "best.pt"
    started = time.perf_counter()

    def save_checkpoint(
        epoch: int,
        validation_mse: float,
        *,
        calibration_row: Mapping[str, object] | None = None,
    ) -> None:
        torch.save(
            {
                "format": "pesten-causal-privileged-critic-v1",
                "candidate": name,
                "epoch": epoch,
                "model": model.model_config(),
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "validation_mse": validation_mse,
                "calibration": calibration_row,
                "target_gamma": target_gamma,
            },
            checkpoint_path,
        )

    initial_started = time.perf_counter()
    initial_validation = evaluate_model(
        model,
        validation,
        device=device,
        batch_size=batch_size,
        amp=amp,
        target_gamma=target_gamma,
    )
    best_validation = float(initial_validation["model"]["mse"])
    save_checkpoint(0, best_validation)
    curve = [
        {
            "epoch": 0,
            "train_loss": None,
            "train_value_mse": None,
            "validation_mse": best_validation,
            "validation_explained_variance": initial_validation["model"][
                "explained_variance"
            ],
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": time.perf_counter() - initial_started,
        }
    ]
    print(json.dumps({"candidate": name, **curve[0]}, sort_keys=True), flush=True)
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        sums = {"loss": 0.0, "value_mse": 0.0, "samples": 0}
        epoch_started = time.perf_counter()
        for indices in batches_for_epoch(
            train,
            batch_size=batch_size,
            seed=seed + epoch,
            shuffle=True,
        ):
            host_batch, host_mask = collate_episodes(
                train, indices, target_gamma=target_gamma
            )
            batch, mask = move_batch(host_batch, host_mask, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                outputs = model(batch, mask=mask)
                loss, loss_row = critic_loss(
                    outputs,
                    batch,
                    mask,
                    win_coefficient=win_coefficient,
                    remaining_coefficient=remaining_coefficient,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            samples = int(mask.sum().item())
            sums["loss"] += loss_row["loss"] * samples
            sums["value_mse"] += loss_row["value_mse"] * samples
            sums["samples"] += samples
        scheduler.step()
        validation_result = evaluate_model(
            model,
            validation,
            device=device,
            batch_size=batch_size,
            amp=amp,
            target_gamma=target_gamma,
        )
        validation_mse = float(validation_result["model"]["mse"])
        row = {
            "epoch": epoch,
            "train_loss": sums["loss"] / sums["samples"],
            "train_value_mse": sums["value_mse"] / sums["samples"],
            "validation_mse": validation_mse,
            "validation_explained_variance": validation_result["model"][
                "explained_variance"
            ],
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": time.perf_counter() - epoch_started,
        }
        curve.append(row)
        print(json.dumps({"candidate": name, **row}, sort_keys=True), flush=True)
        if validation_mse < best_validation - 1e-7:
            best_validation = validation_mse
            stale_epochs = 0
            save_checkpoint(epoch, validation_mse)
        else:
            stale_epochs += 1
            if epoch >= min_epochs and stale_epochs >= patience:
                break

    best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["state_dict"])
    model.to(device)
    pre_calibration_validation = evaluate_model(
        model,
        validation,
        device=device,
        batch_size=batch_size,
        amp=amp,
        return_predictions=True,
        target_gamma=target_gamma,
    )
    calibration_row = fit_value_calibration(
        model,
        pre_calibration_validation.pop("_prediction"),
        pre_calibration_validation.pop("_target"),
        pre_calibration_validation.pop("_player_count"),
        mode=calibration,
    )
    pre_calibration_validation.pop("_public_value")
    validation_result = evaluate_model(
        model,
        validation,
        device=device,
        batch_size=batch_size,
        amp=amp,
        target_gamma=target_gamma,
    )
    save_checkpoint(
        int(best["epoch"]),
        float(validation_result["model"]["mse"]),
        calibration_row=calibration_row,
    )
    test_result = (
        evaluate_model(
            model,
            test,
            device=device,
            batch_size=batch_size,
            amp=amp,
            target_gamma=target_gamma,
        )
        if test is not None
        else None
    )
    result = {
        "candidate": name,
        "model": model.model_config(),
        "parameters": parameter_count(model),
        "target_gamma": target_gamma,
        "best_epoch": int(best["epoch"]),
        "curve": curve,
        "calibration": calibration_row,
        "validation_before_calibration": pre_calibration_validation,
        "validation": validation_result,
        "test": test_result,
        "checkpoint": str(checkpoint_path),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }
    atomic_json(result, candidate_dir / "result.json")
    return result


def paired_cluster_bootstrap_delta(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    seed: int,
    samples: int = 10_000,
) -> dict[str, float | int]:
    reference_sse = np.asarray(
        reference["episode_model_squared_error_sum"], dtype=np.float64
    )
    candidate_sse = np.asarray(
        candidate["episode_model_squared_error_sum"], dtype=np.float64
    )
    decisions = np.asarray(reference["episode_decisions"], dtype=np.float64)
    candidate_decisions = np.asarray(
        candidate["episode_decisions"], dtype=np.float64
    )
    seeds = np.asarray(reference["episode_deal_seed"], dtype=np.int64)
    candidate_seeds = np.asarray(
        candidate["episode_deal_seed"], dtype=np.int64
    )
    if not (
        reference_sse.shape
        == candidate_sse.shape
        == decisions.shape
        == candidate_decisions.shape
        == seeds.shape
        == candidate_seeds.shape
    ):
        raise ValueError("paired episode metrics must have matching shapes")
    if not np.array_equal(decisions, candidate_decisions) or not np.array_equal(
        seeds, candidate_seeds
    ):
        raise ValueError("paired evaluations must use identical episodes")
    unique_seeds, inverse = np.unique(seeds, return_inverse=True)
    cluster_reference = np.bincount(
        inverse, weights=reference_sse, minlength=len(unique_seeds)
    )
    cluster_candidate = np.bincount(
        inverse, weights=candidate_sse, minlength=len(unique_seeds)
    )
    cluster_decisions = np.bincount(
        inverse, weights=decisions, minlength=len(unique_seeds)
    )
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 250):
        count = min(250, samples - start)
        indices = rng.integers(
            0, len(unique_seeds), size=(count, len(unique_seeds))
        )
        denominator = cluster_decisions[indices].sum(axis=1)
        means[start : start + count] = (
            cluster_candidate[indices].sum(axis=1) / denominator
            - cluster_reference[indices].sum(axis=1) / denominator
        )
    low, high = np.quantile(means, (0.025, 0.975))
    return {
        "candidate_minus_reference_mse": float(
            candidate_sse.sum() / decisions.sum()
            - reference_sse.sum() / decisions.sum()
        ),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "episodes": int(len(decisions)),
        "independent_deal_clusters": int(len(unique_seeds)),
    }


def comparison_table(
    results: Sequence[Mapping[str, object]],
    *,
    evaluation_key: str,
    seed: int,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Build paired, deal-clustered comparisons for one held-out split."""

    references: dict[str, Mapping[str, object]] = {}
    if results:
        first_evaluation = results[0][evaluation_key]
        if first_evaluation is None:
            return {}
        inherited = dict(first_evaluation)
        inherited["episode_model_squared_error_sum"] = inherited[
            "episode_public_squared_error_sum"
        ]
        inherited["episode_model_mse"] = inherited["episode_public_mse"]
        references["inherited_public_head"] = inherited
    for reference_name in (
        "public_gru",
        "full_mlp",
        "full_gru",
        "full_lstm",
        "full_transformer",
    ):
        reference_row = next(
            (
                row
                for row in results
                if row["candidate"] == reference_name
            ),
            None,
        )
        if reference_row is not None:
            evaluation = reference_row[evaluation_key]
            if evaluation is not None:
                references[reference_name] = evaluation
    comparisons = {}
    for reference_index, (reference_name, reference) in enumerate(
        references.items()
    ):
        comparisons[reference_name] = {}
        for candidate_index, row in enumerate(results):
            evaluation = row[evaluation_key]
            if evaluation is None:
                continue
            comparisons[reference_name][row["candidate"]] = (
                paired_cluster_bootstrap_delta(
                    reference,
                    evaluation,
                    seed=(
                        seed
                        + 10_000 * reference_index
                        + candidate_index
                    ),
                )
            )
    return comparisons


EPISODE_EVALUATION_FIELDS = frozenset(
    {
        "episode_model_mse",
        "episode_public_mse",
        "episode_model_squared_error_sum",
        "episode_public_squared_error_sum",
        "episode_decisions",
        "episode_deal_seed",
    }
)


def compact_candidate_result(
    result: Mapping[str, object],
) -> dict[str, object]:
    """Remove bootstrap work arrays from a durable, reviewable report.

    Full per-episode arrays remain beside each ignored raw checkpoint.  The
    checked-in report retains aggregate metrics, subgroup metrics, curves, and
    already-computed deal-cluster confidence intervals.
    """

    compact = dict(result)
    for split in ("validation_before_calibration", "validation", "test"):
        evaluation = compact.get(split)
        if evaluation is None:
            continue
        compact[split] = {
            name: value
            for name, value in dict(evaluation).items()
            if name not in EPISODE_EVALUATION_FIELDS
        }
    return compact


def compact_evaluation(
    evaluation: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if evaluation is None:
        return None
    return {
        name: value
        for name, value in evaluation.items()
        if name not in EPISODE_EVALUATION_FIELDS and not name.startswith("_")
    }


def inherited_public_reference(
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    """Present the raw actor head as a paired-bootstrap reference model."""

    reference = dict(evaluation)
    reference["model"] = evaluation["inherited_public_head"]
    reference["episode_model_mse"] = evaluation["episode_public_mse"]
    reference["episode_model_squared_error_sum"] = evaluation[
        "episode_public_squared_error_sum"
    ]
    return reference


def run_ensemble(args: argparse.Namespace) -> dict[str, object]:
    """Lock an independently seeded ensemble on validation, then optionally test."""

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA ensemble evaluation was requested but unavailable")
    torch.set_num_threads(args.torch_threads)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset_dir = args.output_dir.resolve() / "dataset"
    train = PackedEpisodeCorpus(dataset_dir / "train")
    validation = PackedEpisodeCorpus(dataset_dir / "validation")
    test = (
        PackedEpisodeCorpus(dataset_dir / "test")
        if args.evaluate_test
        else None
    )
    corpora = {"train": train, "validation": validation}
    if test is not None:
        corpora["test"] = test
    audits = {name: audit_corpus(corpus) for name, corpus in corpora.items()}
    seed_sets = {
        name: audit.pop("deal_seeds") for name, audit in audits.items()
    }
    if any(not audit["all_invariants_hold"] for audit in audits.values()):
        raise RuntimeError("ensemble evaluation dataset invariant failure")
    overlaps = {}
    names = tuple(seed_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = seed_sets[left] & seed_sets[right]
            overlaps[f"{left}_vs_{right}"] = len(overlap)
            if overlap:
                raise RuntimeError(
                    f"deal seed leakage between {left} and {right}"
                )

    models = []
    member_payloads = []
    member_rows = []
    for index, source in enumerate(args.checkpoints):
        source = source.resolve()
        model, payload = load_privileged_critic_checkpoint(source, device=device)
        checkpoint_gamma = float(payload.get("target_gamma", 0.97))
        if not math.isclose(
            checkpoint_gamma, args.target_gamma, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"ensemble member {source} uses gamma={checkpoint_gamma}, "
                f"requested {args.target_gamma}"
            )
        if (
            int(model.model_config()["actor_memory_size"])
            != validation.actor_memory_size
        ):
            raise ValueError(
                f"ensemble member {source} actor-memory width disagrees with data"
            )
        validation_result = evaluate_model(
            model,
            validation,
            device=device,
            batch_size=args.batch_size,
            amp=not args.no_amp,
            target_gamma=args.target_gamma,
        )
        member_name = f"member_{index + 1}"
        member_rows.append(
            {
                "name": member_name,
                "source_checkpoint": str(source),
                "source_sha256": file_sha256(source),
                "candidate": payload.get("candidate"),
                "epoch": payload.get("epoch"),
                "parameters": parameter_count(model),
                "validation": validation_result,
            }
        )
        # A single shared affine correction is fitted after averaging.  Leaving
        # member-level validation calibration active would calibrate twice.
        model.reset_value_calibration()
        models.append(model)
        member_payloads.append(payload)

    ensemble = AveragedPrivilegedCritic(models).to(device).eval()
    validation_probe = evaluate_model(
        ensemble,
        validation,
        device=device,
        batch_size=args.batch_size,
        amp=not args.no_amp,
        return_predictions=True,
        target_gamma=args.target_gamma,
    )
    target = validation_probe.pop("_target")
    player_count = validation_probe.pop("_player_count")
    public_prediction = validation_probe.pop("_public_value")
    ensemble_prediction = validation_probe.pop("_prediction")
    ensemble_calibration = fit_value_calibration(
        ensemble,
        ensemble_prediction,
        target,
        player_count,
        mode=args.calibration,
    )
    public_scale, public_bias, public_calibration = (
        affine_calibration_parameters(
            public_prediction,
            target,
            player_count,
            mode=args.calibration,
        )
    )
    public_model = CalibratedPublicValue(public_scale, public_bias).to(device)
    validation_result = evaluate_model(
        ensemble,
        validation,
        device=device,
        batch_size=args.batch_size,
        amp=not args.no_amp,
        target_gamma=args.target_gamma,
    )
    calibrated_public_validation = evaluate_model(
        public_model,
        validation,
        device=device,
        batch_size=args.batch_size,
        amp=not args.no_amp,
        target_gamma=args.target_gamma,
    )

    validation_candidates = [
        ("ensemble", validation_result),
        *[
            (row["name"], row["validation"])
            for row in member_rows
        ],
    ]
    selected_name, selected_validation = min(
        validation_candidates,
        key=lambda row: float(row[1]["model"]["mse"]),
    )
    if selected_name == "ensemble":
        selected_model = ensemble
    else:
        selected_index = int(selected_name.removeprefix("member_")) - 1
        selected_model, _ = load_privileged_critic_checkpoint(
            args.checkpoints[selected_index], device=device
        )

    raw_public_validation = inherited_public_reference(validation_result)
    validation_comparisons = {
        "raw_public_head": paired_cluster_bootstrap_delta(
            raw_public_validation,
            selected_validation,
            seed=args.seed + 1,
        ),
        "calibrated_public_head": paired_cluster_bootstrap_delta(
            calibrated_public_validation,
            selected_validation,
            seed=args.seed + 2,
        ),
    }
    for index, row in enumerate(member_rows):
        if row["name"] != selected_name:
            validation_comparisons[row["name"]] = (
                paired_cluster_bootstrap_delta(
                    row["validation"],
                    selected_validation,
                    seed=args.seed + 100 + index,
                )
            )

    test_result = None
    calibrated_public_test = None
    test_comparisons = None
    if test is not None:
        test_result = evaluate_model(
            selected_model,
            test,
            device=device,
            batch_size=args.batch_size,
            amp=not args.no_amp,
            target_gamma=args.target_gamma,
        )
        calibrated_public_test = evaluate_model(
            public_model,
            test,
            device=device,
            batch_size=args.batch_size,
            amp=not args.no_amp,
            target_gamma=args.target_gamma,
        )
        test_comparisons = {
            "raw_public_head": paired_cluster_bootstrap_delta(
                inherited_public_reference(test_result),
                test_result,
                seed=args.seed + 10_001,
            ),
            "calibrated_public_head": paired_cluster_bootstrap_delta(
                calibrated_public_test,
                test_result,
                seed=args.seed + 10_002,
            ),
        }

    candidate_dir = (
        args.output_dir.resolve() / "models" / args.run_label / "ensemble"
    )
    candidate_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = candidate_dir / "best.pt"
    ensemble_payload = {
        "format": "pesten-causal-privileged-critic-ensemble-v1",
        "aggregation": "arithmetic_mean_then_player_count_affine_calibration",
        "members": [
            {
                "source_checkpoint": row["source_checkpoint"],
                "source_sha256": row["source_sha256"],
                "candidate": payload.get("candidate"),
                "epoch": payload.get("epoch"),
                "model": model.model_config(),
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
            }
            for row, payload, model in zip(
                member_rows, member_payloads, models
            )
        ],
        "value_calibration_scale": (
            ensemble.value_calibration_scale.detach().cpu()
        ),
        "value_calibration_bias": (
            ensemble.value_calibration_bias.detach().cpu()
        ),
        "calibration": ensemble_calibration,
        "validation_mse": validation_result["model"]["mse"],
        "target_gamma": args.target_gamma,
    }
    # Bundle members with identity calibration: the shared post-average
    # calibration is the only correction used by the ensemble loader.
    for member in ensemble_payload["members"]:
        member["state_dict"]["value_calibration_scale"] = torch.ones(14)
        member["state_dict"]["value_calibration_bias"] = torch.zeros(14)
    temporary = checkpoint_path.with_suffix(f".tmp.{os.getpid()}.pt")
    torch.save(ensemble_payload, temporary)
    temporary.replace(checkpoint_path)

    raw_result = {
        "format": "pesten-privileged-critic-ensemble-evaluation-v1",
        "selection_metric": "validation Monte Carlo return MSE",
        "target_gamma": args.target_gamma,
        "selected": selected_name,
        "ensemble": {
            "parameters": parameter_count(ensemble),
            "calibration": ensemble_calibration,
            "validation_before_calibration": validation_probe,
            "validation": validation_result,
        },
        "members": member_rows,
        "public_baseline": {
            "calibration": public_calibration,
            "validation": calibrated_public_validation,
        },
        "validation_comparisons": validation_comparisons,
        "test": test_result,
        "public_baseline_test": calibrated_public_test,
        "test_comparisons": test_comparisons,
    }
    atomic_json(raw_result, candidate_dir / "result.json")
    report = {
        "format": "pesten-privileged-critic-ensemble-evaluation-v1",
        "selection_metric": "validation Monte Carlo return MSE",
        "test_policy": (
            "opened once after architecture, features, capacity, seeds, "
            "calibration, aggregation, and finalist were locked on validation"
            if test is not None
            else "untouched; finalist selection uses validation only"
        ),
        "selected": selected_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "dataset": {
            "train_episodes": len(train),
            "validation_episodes": len(validation),
            "test_episodes": len(test) if test is not None else None,
            "actor_memory_size": validation.actor_memory_size,
            "audit": audits,
            "deal_seed_overlap": overlaps,
        },
        "ensemble": {
            "parameters": parameter_count(ensemble),
            "calibration": ensemble_calibration,
            "validation_before_calibration": compact_evaluation(
                validation_probe
            ),
            "validation": compact_evaluation(validation_result),
        },
        "members": [
            {
                **{
                    name: value
                    for name, value in row.items()
                    if name != "validation"
                },
                "validation": compact_evaluation(row["validation"]),
            }
            for row in member_rows
        ],
        "public_baseline": {
            "calibration": public_calibration,
            "validation": compact_evaluation(calibrated_public_validation),
        },
        "validation_comparisons": validation_comparisons,
        "test": compact_evaluation(test_result),
        "public_baseline_test": compact_evaluation(calibrated_public_test),
        "test_comparisons": test_comparisons,
    }
    atomic_json(report, args.report.resolve())
    return report


def expand_candidates(value: str) -> tuple[str, ...]:
    if value.strip() == "screen":
        return SCREEN_CANDIDATES
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("at least one critic candidate is required")
    for name in result:
        candidate_spec(name)
    return result


def run_training(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA critic training was requested but unavailable")
    torch.set_num_threads(args.torch_threads)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    dataset_dir = args.output_dir.resolve() / "dataset"
    train = PackedEpisodeCorpus(dataset_dir / "train")
    validation = PackedEpisodeCorpus(dataset_dir / "validation")
    test = (
        PackedEpisodeCorpus(dataset_dir / "test")
        if args.evaluate_test
        else None
    )
    memory_sizes = {train.actor_memory_size, validation.actor_memory_size}
    if test is not None:
        memory_sizes.add(test.actor_memory_size)
    if len(memory_sizes) != 1:
        raise ValueError("dataset splits disagree on actor memory shape")
    corpora = {"train": train, "validation": validation}
    if test is not None:
        corpora["test"] = test
    audits = {name: audit_corpus(corpus) for name, corpus in corpora.items()}
    seed_sets = {
        name: audit.pop("deal_seeds") for name, audit in audits.items()
    }
    for name, audit in audits.items():
        if not audit["all_invariants_hold"]:
            raise RuntimeError(
                f"critic dataset invariant failure in {name}: "
                f"{audit['invariant_errors']}"
            )
    overlaps = {}
    names = tuple(seed_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = seed_sets[left] & seed_sets[right]
            overlaps[f"{left}_vs_{right}"] = len(overlap)
            if overlap:
                raise RuntimeError(
                    f"deal seed leakage between {left} and {right}"
                )
    candidates = expand_candidates(args.candidates)
    results = []
    for index, name in enumerate(candidates):
        results.append(
            train_candidate(
                name,
                train=train,
                validation=validation,
                test=test,
                output_dir=args.output_dir.resolve(),
                device=device,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                group_width=args.group_width,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                win_coefficient=args.win_coefficient,
                remaining_coefficient=args.remaining_coefficient,
                dropout=args.dropout,
                # Same initialization namespace makes same-backbone feature
                # ablations paired rather than adding avoidable seed noise.
                seed=args.seed,
                patience=args.patience,
                min_epochs=args.min_epochs,
                amp=not args.no_amp,
                run_label=args.run_label,
                calibration=args.calibration,
                target_gamma=args.target_gamma,
            )
        )
    validation_comparisons = comparison_table(
        results,
        evaluation_key="validation",
        seed=args.seed + 90_000_000,
    )
    test_comparisons = (
        comparison_table(
            results,
            evaluation_key="test",
            seed=args.seed + 95_000_000,
        )
        if test is not None
        else None
    )
    ranking = sorted(
        (
            {
                "candidate": row["candidate"],
                "validation_mse": row["validation"]["model"]["mse"],
                "test_mse": (
                    row["test"]["model"]["mse"]
                    if row["test"] is not None
                    else None
                ),
                "test_explained_variance": (
                    row["test"]["model"]["explained_variance"]
                    if row["test"] is not None
                    else None
                ),
                "parameters": row["parameters"],
            }
            for row in results
        ),
        key=lambda item: item["validation_mse"],
    )
    report = {
        "format": "pesten-privileged-critic-ablation-v1",
        "selection_metric": "fresh validation Monte Carlo return MSE",
        "test_policy": (
            "explicitly enabled once for final candidates on the untouched "
            "deal-seed-disjoint test split"
            if args.evaluate_test
            else "untouched; model and feature selection use validation only"
        ),
        "device": str(device),
        "torch": torch.__version__,
        "compute": {
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu"
            ),
            "bfloat16_autocast": bool(
                not args.no_amp and device.type == "cuda"
            ),
            "tf32_matmul": bool(
                torch.backends.cuda.matmul.allow_tf32
                if device.type == "cuda"
                else False
            ),
        },
        "dataset": {
            "train_episodes": len(train),
            "validation_episodes": len(validation),
            "test_episodes": len(test) if test is not None else None,
            "actor_memory_size": train.actor_memory_size,
            "audit": audits,
            "deal_seed_overlap": overlaps,
        },
        "screen_capacity": {
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "group_width": args.group_width,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "win_coefficient": args.win_coefficient,
            "remaining_coefficient": args.remaining_coefficient,
            "dropout": args.dropout,
            "minimum_epochs": args.min_epochs,
            "calibration": args.calibration,
            "target_gamma": args.target_gamma,
            "target_retention_after_20_decisions": (
                args.target_gamma**20
            ),
        },
        "target_gamma": args.target_gamma,
        "target_retention_after_20_decisions": args.target_gamma**20,
        "run_label": args.run_label,
        "ranking": ranking,
        "inherited_public_head": (
            {
                "validation": results[0]["validation"][
                    "inherited_public_head"
                ],
                "test": (
                    results[0]["test"]["inherited_public_head"]
                    if results[0]["test"] is not None
                    else None
                ),
            }
            if results
            else None
        ),
        "paired_validation_mse_comparisons": validation_comparisons,
        "paired_test_mse_comparisons": test_comparisons,
        "results": [compact_candidate_result(row) for row in results],
    }
    atomic_json(report, args.report.resolve())
    return report


def run_locked_evaluation(args: argparse.Namespace) -> dict[str, object]:
    """Evaluate one already-locked checkpoint on fresh confirmation deals."""

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA critic evaluation was requested but unavailable")
    torch.set_num_threads(args.torch_threads)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset_dir = args.output_dir.resolve() / "dataset"
    corpora = {
        name: PackedEpisodeCorpus(dataset_dir / name)
        for name in ("train", "validation", "test", "confirmation")
    }
    memory_sizes = {corpus.actor_memory_size for corpus in corpora.values()}
    if len(memory_sizes) != 1:
        raise ValueError("critic dataset splits disagree on actor memory shape")
    audits = {name: audit_corpus(corpus) for name, corpus in corpora.items()}
    seed_sets = {
        name: audit.pop("deal_seeds") for name, audit in audits.items()
    }
    if any(not audit["all_invariants_hold"] for audit in audits.values()):
        raise RuntimeError("locked evaluation dataset invariant failure")
    overlaps = {}
    names = tuple(seed_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = seed_sets[left] & seed_sets[right]
            overlaps[f"{left}_vs_{right}"] = len(overlap)
            if overlap:
                raise RuntimeError(
                    f"deal seed leakage between {left} and {right}"
                )

    checkpoint = args.checkpoint.resolve()
    model, payload = load_privileged_value_checkpoint(
        checkpoint, device=device
    )
    checkpoint_gamma = float(payload.get("target_gamma", 0.97))
    if not math.isclose(
        checkpoint_gamma, args.target_gamma, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"checkpoint uses gamma={checkpoint_gamma}, requested "
            f"{args.target_gamma}"
        )
    model.eval()
    validation_probe = evaluate_model(
        model,
        corpora["validation"],
        device=device,
        batch_size=args.batch_size,
        amp=not args.no_amp,
        return_predictions=True,
        target_gamma=args.target_gamma,
    )
    public_scale, public_bias, public_calibration = (
        affine_calibration_parameters(
            validation_probe["_public_value"],
            validation_probe["_target"],
            validation_probe["_player_count"],
            mode=args.public_calibration,
        )
    )
    public_model = CalibratedPublicValue(public_scale, public_bias).to(device)
    confirmation = corpora["confirmation"]
    confirmation_result = evaluate_model(
        model,
        confirmation,
        device=device,
        batch_size=args.batch_size,
        amp=not args.no_amp,
        target_gamma=args.target_gamma,
    )
    calibrated_public = evaluate_model(
        public_model,
        confirmation,
        device=device,
        batch_size=args.batch_size,
        amp=not args.no_amp,
        target_gamma=args.target_gamma,
    )
    raw_public = inherited_public_reference(confirmation_result)
    comparisons = {
        "raw_public_head": paired_cluster_bootstrap_delta(
            raw_public, confirmation_result, seed=args.seed + 1
        ),
        "validation_calibrated_public_head": paired_cluster_bootstrap_delta(
            calibrated_public, confirmation_result, seed=args.seed + 2
        ),
    }
    report = {
        "format": "pesten-privileged-critic-locked-confirmation-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_format": payload.get("format"),
        "candidate": payload.get("candidate"),
        "checkpoint_epoch": payload.get("epoch"),
        "parameters": parameter_count(model),
        "target_gamma": args.target_gamma,
        "target_retention_after_20_decisions": args.target_gamma**20,
        "selection_policy": (
            "checkpoint and all hyperparameters were locked on train and "
            "validation before confirmation data were collected"
        ),
        "old_test_policy": (
            "the previously opened test split received only structural and "
            "seed-overlap audits; the locked model was not evaluated on it"
        ),
        "dataset": {
            "audit": audits,
            "deal_seed_overlap": overlaps,
        },
        "validation_model": compact_evaluation(validation_probe),
        "public_calibration": public_calibration,
        "confirmation": compact_evaluation(confirmation_result),
        "calibrated_public_confirmation": compact_evaluation(
            calibrated_public
        ),
        "paired_confirmation_mse_comparisons": comparisons,
    }
    atomic_json(report, args.report.resolve())
    return report


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20_260_813)
    parser.add_argument("--torch-threads", type=int, default=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    add_shared_arguments(collect)
    collect.add_argument("--actor", type=Path, default=DEFAULT_ACTOR)
    collect.add_argument(
        "--schedule",
        type=lambda value: tuple(
            item.strip() for item in value.split(",") if item.strip()
        ),
        default=DEFAULT_SCHEDULE,
    )
    collect.add_argument(
        "--player-counts", type=parse_ints, default=tuple(range(2, 9))
    )
    collect.add_argument("--train-games-per-player-count", type=int, default=512)
    collect.add_argument(
        "--validation-games-per-player-count", type=int, default=128
    )
    collect.add_argument("--test-games-per-player-count", type=int, default=256)
    collect.add_argument("--chunk-games", type=int, default=64)
    collect.add_argument("--max-turns", type=int, default=1000)
    collect.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    collect.add_argument(
        "--resume",
        action="store_true",
        help="reuse validated existing chunks and extend the campaign",
    )
    collect.add_argument("--chunk-shard-index", type=int, default=0)
    collect.add_argument("--chunk-shard-count", type=int, default=1)

    confirm = subparsers.add_parser("collect-confirmation")
    add_shared_arguments(confirm)
    confirm.add_argument("--actor", type=Path, default=DEFAULT_ACTOR)
    confirm.add_argument(
        "--schedule",
        type=lambda value: tuple(
            item.strip() for item in value.split(",") if item.strip()
        ),
        default=DEFAULT_SCHEDULE,
    )
    confirm.add_argument(
        "--player-counts", type=parse_ints, default=tuple(range(2, 9))
    )
    confirm.add_argument("--games-per-player-count", type=int, default=256)
    confirm.add_argument("--chunk-games", type=int, default=64)
    confirm.add_argument("--max-turns", type=int, default=1000)
    confirm.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    confirm.add_argument("--resume", action="store_true")

    merge = subparsers.add_parser("merge")
    merge.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    merge.add_argument(
        "--shards",
        type=Path,
        nargs="+",
        required=True,
        help="additional completed collection directories",
    )

    train = subparsers.add_parser("train")
    add_shared_arguments(train)
    train.add_argument("--candidates", default="screen")
    train.add_argument(
        "--run-label",
        default="screen",
        help="checkpoint namespace below OUTPUT/models (default: screen)",
    )
    train.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train.add_argument("--hidden-size", type=int, default=192)
    train.add_argument("--num-layers", type=int, default=2)
    train.add_argument("--group-width", type=int, default=64)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--patience", type=int, default=3)
    train.add_argument(
        "--min-epochs",
        type=int,
        default=1,
        help="train at least this many epochs before early stopping",
    )
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--win-coefficient", type=float, default=0.15)
    train.add_argument("--remaining-coefficient", type=float, default=0.05)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument(
        "--target-gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help=(
            "virtual return-target discount; defaults to 25%% retention "
            "after 20 controlled-seat decisions"
        ),
    )
    train.add_argument(
        "--calibration",
        choices=("none", "global", "player_count"),
        default="player_count",
        help="validation-fitted affine value calibration (default: per count)",
    )
    train.add_argument("--no-amp", action="store_true")
    train.add_argument(
        "--evaluate-test",
        action="store_true",
        help="evaluate the untouched test split; reserve this for finalists",
    )

    ensemble = subparsers.add_parser("ensemble")
    add_shared_arguments(ensemble)
    ensemble.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="independently seeded selected-model checkpoints",
    )
    ensemble.add_argument("--run-label", default="selected_ensemble")
    ensemble.add_argument(
        "--report",
        type=Path,
        default=Path("benchmarks/privileged_critic/finalist_validation.json"),
    )
    ensemble.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ensemble.add_argument("--batch-size", type=int, default=16)
    ensemble.add_argument(
        "--target-gamma", type=float, default=DEFAULT_GAMMA
    )
    ensemble.add_argument(
        "--calibration",
        choices=("none", "global", "player_count"),
        default="player_count",
    )
    ensemble.add_argument("--no-amp", action="store_true")
    ensemble.add_argument(
        "--evaluate-test",
        action="store_true",
        help="open the untouched split only after validation locks the finalist",
    )

    evaluate = subparsers.add_parser("evaluate-confirmation")
    add_shared_arguments(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument(
        "--report",
        type=Path,
        default=Path(
            "benchmarks/privileged_critic/token_gru_confirmation.json"
        ),
    )
    evaluate.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda"
    )
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument(
        "--target-gamma", type=float, default=DEFAULT_GAMMA
    )
    evaluate.add_argument(
        "--public-calibration",
        choices=("none", "global", "player_count"),
        default="player_count",
    )
    evaluate.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "merge":
        result = merge_collections(args)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return
    positive = [args.torch_threads]
    if args.command == "collect":
        positive.extend(
            (
                args.train_games_per_player_count,
                args.validation_games_per_player_count,
                args.test_games_per_player_count,
                args.chunk_games,
                args.max_turns,
            )
        )
        if any(count < 2 or count > 13 for count in args.player_counts):
            raise ValueError("player counts must be in [2, 13]")
        if not 0 < args.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if (
            args.chunk_shard_count < 1
            or not 0 <= args.chunk_shard_index < args.chunk_shard_count
        ):
            raise ValueError("chunk shard index must be inside shard count")
        if any(value < 1 for value in positive):
            raise ValueError(
                "count, capacity, and thread arguments must be positive"
            )
        result = run_collection(args)
    elif args.command == "collect-confirmation":
        positive.extend(
            (
                args.games_per_player_count,
                args.chunk_games,
                args.max_turns,
            )
        )
        if any(count < 2 or count > 13 for count in args.player_counts):
            raise ValueError("player counts must be in [2, 13]")
        if not 0 < args.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if any(value < 1 for value in positive):
            raise ValueError("count and thread arguments must be positive")
        result = run_confirmation_collection(args)
    elif args.command == "train":
        if (
            not args.run_label
            or Path(args.run_label).name != args.run_label
            or args.run_label in (".", "..")
        ):
            raise ValueError("run label must be one path-safe name")
        positive.extend(
            (
                args.hidden_size,
                args.num_layers,
                args.group_width,
                args.epochs,
                args.patience,
                args.min_epochs,
                args.batch_size,
            )
        )
        if args.learning_rate <= 0 or args.weight_decay < 0:
            raise ValueError(
                "learning rate must be positive and weight decay nonnegative"
            )
        if args.win_coefficient < 0 or args.remaining_coefficient < 0:
            raise ValueError("auxiliary loss coefficients must be nonnegative")
        if not 0 <= args.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 < args.target_gamma <= 1:
            raise ValueError("target gamma must be in (0, 1]")
        if args.min_epochs > args.epochs:
            raise ValueError("minimum epochs cannot exceed total epochs")
        if any(value < 1 for value in positive):
            raise ValueError(
                "count, capacity, and thread arguments must be positive"
            )
        result = run_training(args)
    elif args.command == "evaluate-confirmation":
        if args.batch_size < 1 or args.torch_threads < 1:
            raise ValueError("batch size and thread count must be positive")
        if not 0 < args.target_gamma <= 1:
            raise ValueError("target gamma must be in (0, 1]")
        result = run_locked_evaluation(args)
    else:
        if (
            not args.run_label
            or Path(args.run_label).name != args.run_label
            or args.run_label in (".", "..")
        ):
            raise ValueError("run label must be one path-safe name")
        if len(args.checkpoints) < 2:
            raise ValueError("an ensemble requires at least two checkpoints")
        if args.batch_size < 1 or args.torch_threads < 1:
            raise ValueError("batch size and thread count must be positive")
        if not 0 < args.target_gamma <= 1:
            raise ValueError("target gamma must be in (0, 1]")
        result = run_ensemble(args)
    if args.command in ("train", "ensemble", "evaluate-confirmation"):
        if args.command == "evaluate-confirmation":
            result = {
                "report": str(args.report.resolve()),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "confirmation_mse": result["confirmation"]["model"]["mse"],
                "raw_public_mse": result["confirmation"][
                    "inherited_public_head"
                ]["mse"],
                "selection_policy": result["selection_policy"],
            }
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
            return
        result = {
            "report": str(args.report.resolve()),
            **(
                {"ranking": result["ranking"]}
                if args.command == "train"
                else {
                    "selected": result["selected"],
                    "validation_mse": (
                        result["ensemble"]["validation"]["model"]["mse"]
                        if result["selected"] == "ensemble"
                        else next(
                            row["validation"]["model"]["mse"]
                            for row in result["members"]
                            if row["name"] == result["selected"]
                        )
                    ),
                }
            ),
            "test_policy": result["test_policy"],
        }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
