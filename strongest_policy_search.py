#!/usr/bin/env python3
"""Public-input-only native MinGRU suit-orbit inference and matched screens.

This research runner preserves one independent recurrent state for every suit
relabeling. It neither changes a checkpoint nor receives a game engine inside
the inference wrapper. Reports are development evidence, not promotion gates.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import permutations
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from temporal_suit_orbit import relabel_action


TEACHER = Path("models/best/puffer_min_gru_192x2_plateau_teacher_epoch4387.pt")
DEFAULT_OUTPUT = Path("live_training/strongest_20260907/search")


class NativeSuitOrbitCore(nn.Module):
    """Exact suit relabeling around a frozen native Puffer policy core.

    State packing keeps the evaluator's ordinary (layers, batch, width) API;
    width contains separate orbit states, never an averaged recurrent state.
    """

    def __init__(self, core: nn.Module, alpha: float = 1.0, orbit_size: int = 24):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if orbit_size not in (1, 4, 24):
            raise ValueError("orbit_size must be 1, 4, or 24")
        if not all(hasattr(core, name) for name in ("encoder", "network", "forward_eval")):
            raise TypeError("native suit orbit requires a simple native encoder/network core")
        self.frozen_core = core
        self.alpha = alpha
        self.orbit_size = orbit_size
        if orbit_size == 24:
            suit_permutations = tuple(permutations(range(4)))
        else:
            suit_permutations = tuple(
                tuple((suit + offset) % 4 for suit in range(4))
                for offset in range(orbit_size)
            )
        self.suit_permutations = suit_permutations
        observation_indices = []
        action_indices = []
        card_maps = []
        suit_maps = []
        for permutation in suit_permutations:
            indices = list(range(100))
            for old_suit, new_suit in enumerate(permutation):
                for rank in range(12):
                    indices[new_suit * 12 + rank] = old_suit * 12 + rank
                indices[50 + new_suit] = 50 + old_suit
                indices[82 + new_suit] = 82 + old_suit
            observation_indices.append(indices)
            action_indices.append([relabel_action(a, permutation) for a in range(55)])
            card_maps.append([
                relabel_action(card, permutation) for card in range(50)
            ])
            suit_maps.append([*permutation, 4])
        self.register_buffer("observation_indices", torch.tensor(observation_indices))
        self.register_buffer("action_indices", torch.tensor(action_indices))
        self.register_buffer("card_maps", torch.tensor(card_maps))
        self.register_buffer("suit_maps", torch.tensor(suit_maps))

    def relabel_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[-1] != 100:
            raise ValueError("native orbit input must have shape (batch, 100)")
        # Advanced indexing makes an independent B,O,100 tensor.
        transformed = tokens[:, self.observation_indices]
        event_rows = tokens[:, 99] == 0
        kinds = torch.round(tokens[:, 94] * 3).long()
        play = event_rows & (kinds == 1)
        choose_suit = event_rows & (kinds == 3)
        if play.any():
            card_ids = torch.round(tokens[play, 96] * 50).long() - 1
            if ((card_ids < 0) | (card_ids >= 50)).any():
                raise ValueError("invalid public PLAY card identity")
            play_suit_ids = torch.round(tokens[play, 97] * 5).long() - 1
            expected_suit_ids = torch.where(card_ids < 48, card_ids // 12, 4)
            if (play_suit_ids != expected_suit_ids).any():
                raise ValueError("public PLAY card/suit mismatch")
            transformed[play, :, 96] = (self.card_maps[:, card_ids].T + 1) / 50
        if choose_suit.any():
            announced = torch.round(tokens[choose_suit, 97] * 5).long() - 1
            if ((announced < 0) | (announced >= 4)).any():
                raise ValueError("CHOOSE_SUIT must name an ordinary suit")
        suit_rows = play | choose_suit
        if suit_rows.any():
            suit_ids = torch.round(tokens[suit_rows, 97] * 5).long() - 1
            if ((suit_ids < 0) | (suit_ids >= 5)).any():
                raise ValueError("invalid public event suit")
            transformed[suit_rows, :, 97] = (self.suit_maps[:, suit_ids].T + 1) / 5
        return transformed.flatten(0, 1)

    def initial_state(self, batch_size: int, device):
        states = self.frozen_core.initial_state(batch_size * self.orbit_size, device)
        return self._pack(states, batch_size)

    def _unpack(self, states, batch_size):
        return tuple(
            state.reshape(state.shape[0], batch_size * self.orbit_size, -1)
            for state in states
        )

    def _pack(self, states, batch_size):
        return tuple(
            state.reshape(state.shape[0], batch_size, -1)
            for state in states
        )

    def forward_events_eval(self, tokens, states):
        batch_size = tokens.shape[0]
        expanded = self.relabel_tokens(tokens)
        hidden = self.frozen_core.encoder(expanded)
        _, next_states = self.frozen_core.network.forward_eval(
            hidden, self._unpack(states, batch_size)
        )
        return self._pack(next_states, batch_size)

    def forward_eval(self, tokens, states):
        batch_size = tokens.shape[0]
        logits, values, next_states = self.frozen_core.forward_eval(
            self.relabel_tokens(tokens), self._unpack(states, batch_size)
        )
        logits = logits.reshape(batch_size, self.orbit_size, 55)
        canonical = logits.gather(2, self.action_indices.expand(batch_size, -1, -1))
        combined = canonical[:, 0] * (1.0 - self.alpha) + canonical.mean(1) * self.alpha
        values = values.reshape(batch_size, self.orbit_size, -1).mean(1)
        return combined, values, self._pack(next_states, batch_size)

    def forward(self, tokens):
        """Evaluate full safe trajectories with an independent orbit per row.

        A suit permutation stays fixed across every event and decision of a
        trajectory. The native sequence core starts each expanded trajectory
        from zero state, just as ``initial_state`` plus repeated eval calls do.
        """
        if tokens.ndim != 3 or tokens.shape[-1] != 100:
            raise ValueError("native orbit sequence must have shape (batch, time, 100)")
        batch_size, length, _ = tokens.shape
        expanded = self.relabel_tokens(tokens.reshape(-1, 100))
        expanded = expanded.reshape(batch_size, length, self.orbit_size, 100)
        expanded = expanded.permute(0, 2, 1, 3).reshape(
            batch_size * self.orbit_size, length, 100
        )
        logits, values = self.frozen_core(expanded)
        logits = logits.reshape(batch_size, self.orbit_size, length, 55)
        canonical = logits.gather(
            3, self.action_indices[None, :, None, :].expand(batch_size, -1, length, -1)
        )
        combined = canonical[:, 0] * (1.0 - self.alpha) + canonical.mean(1) * self.alpha
        values = values.reshape(batch_size, self.orbit_size, length, -1).mean(1)
        return combined.reshape(batch_size * length, 55), values.reshape(batch_size * length, -1)


def _initialize_worker(checkpoint: str, mlp: str, variants: dict):
    import benchmark_rlvr_pool_batched as benchmark

    paths = {label: config.get("checkpoint", checkpoint) for label, config in variants.items()}
    paths["self_anchor"] = paths["base"]
    benchmark._initialize_multi_worker(paths, mlp)
    for label, config in variants.items():
        if config["alpha"] == 0 and config["orbit_size"] == 1:
            continue
        policy = deepcopy(benchmark._POLICIES[label])
        policy.puffer_policy = NativeSuitOrbitCore(
            policy.puffer_policy, alpha=config["alpha"], orbit_size=config["orbit_size"]
        ).eval()
        benchmark._POLICIES[label] = policy


def _evaluate_task(task):
    from benchmark_rlvr_pool_batched import _evaluate_task as evaluate

    result = evaluate(task)
    result["seed_start_inclusive"] = task["seed"]
    result["seed_end_exclusive"] = task["seed"] + task["blocks"]
    return result


def summarize(rows, variants):
    grouped = {}
    for row in rows:
        key = f"{row['opponent']}/{row['players']}p"
        group = grouped.setdefault(key, {label: [] for label in variants})
        for label in variants:
            group[label].extend(row[f"{label}_scores"])
    result = {}
    for key, group in grouped.items():
        base = np.asarray(group["base"])
        result[key] = {}
        for label, raw in group.items():
            values = np.asarray(raw)
            deltas = values - base
            sem = float(deltas.std(ddof=1) / math.sqrt(len(deltas))) if len(deltas) > 1 else 0.0
            score_sem = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
            hoeffding_radius = math.sqrt(math.log(2000.0) / (2 * len(values)))
            result[key][label] = {
                "blocks": len(values),
                "score": float(values.mean()),
                "delta_vs_base": float(deltas.mean()),
                "paired_normal_ci95": [float(deltas.mean() - 1.96 * sem), float(deltas.mean() + 1.96 * sem)],
                "paired_score_normal_ci99_9": [float(values.mean() - 3.2905267314919255 * score_sem), float(values.mean() + 3.2905267314919255 * score_sem)],
                "paired_score_hoeffding_ci99_9": [max(0.0, float(values.mean() - hoeffding_radius)), min(1.0, float(values.mean() + hoeffding_radius))],
            }
            if "reference_orbit24_a1" in group:
                difference = values - np.asarray(group["reference_orbit24_a1"])
                orbit_sem = float(difference.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
                radius = 3.2905267314919255 * orbit_sem
                result[key][label]["delta_vs_reference_orbit"] = float(difference.mean())
                result[key][label]["paired_vs_reference_orbit_normal_ci99_9"] = [float(difference.mean() - radius), float(difference.mean() + radius)]
    return result


def summarize_pool(rows, variants):
    """Use the established player-count weights and equal opponent weights."""
    from benchmark_rlvr_pool import PLAYER_WEIGHTS

    grouped = {}
    for row in rows:
        key = (row["opponent"], row["players"])
        group = grouped.setdefault(key, {label: [] for label in variants})
        for label in variants:
            group[label].extend(row[f"{label}_scores"])
    if not grouped:
        return {}
    raw_weights = {key: PLAYER_WEIGHTS[key[1]] for key in grouped}
    total_weight = sum(raw_weights.values())
    result = {}
    for label in variants:
        score = delta = variance = 0.0
        orbit_delta = orbit_variance = 0.0
        for key, group in grouped.items():
            weight = raw_weights[key] / total_weight
            values = np.asarray(group[label])
            differences = values - np.asarray(group["base"])
            score += weight * float(values.mean())
            delta += weight * float(differences.mean())
            if len(values) > 1:
                variance += weight ** 2 * float(differences.var(ddof=1)) / len(values)
            if "reference_orbit24_a1" in group:
                orbit_differences = values - np.asarray(group["reference_orbit24_a1"])
                orbit_delta += weight * float(orbit_differences.mean())
                if len(values) > 1:
                    orbit_variance += weight ** 2 * float(orbit_differences.var(ddof=1)) / len(values)
        radius = 3.2905267314919255 * math.sqrt(variance)
        result[label] = {
            "weighted_score": score, "weighted_delta_vs_base": delta,
            "paired_normal_ci99_9": [delta - radius, delta + radius],
        }
        if "reference_orbit24_a1" in variants:
            orbit_radius = 3.2905267314919255 * math.sqrt(orbit_variance)
            result[label]["weighted_delta_vs_reference_orbit"] = orbit_delta
            result[label]["paired_vs_reference_orbit_normal_ci99_9"] = [orbit_delta - orbit_radius, orbit_delta + orbit_radius]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=TEACHER)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--mlp", type=Path, default=Path("ppo_actor_100run3.pth"))
    parser.add_argument("--alphas", default="0.25,0.5,1")
    parser.add_argument("--orbit-size", type=int, default=24, choices=(1, 4, 24))
    parser.add_argument("--players", default="2")
    parser.add_argument("--opponents", default="self_anchor,mlp")
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--chunk-blocks", type=int, default=25)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=87090701)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "native_orbit_screen.json")
    args = parser.parse_args()
    if not 1 <= args.workers <= 3 or min(args.blocks, args.chunk_blocks) < 1:
        parser.error("use 1-3 workers and positive block counts")
    reference_checkpoint = args.reference_checkpoint or args.checkpoint
    variants = {"base": {"alpha": 0, "orbit_size": 1, "checkpoint": str(reference_checkpoint)}}
    if reference_checkpoint.resolve() != args.checkpoint.resolve():
        variants["candidate_plain"] = {"alpha": 0, "orbit_size": 1, "checkpoint": str(args.checkpoint)}
        variants["reference_orbit24_a1"] = {"alpha": 1, "orbit_size": 24, "checkpoint": str(reference_checkpoint)}
    for alpha in map(float, args.alphas.split(",")):
        variants[f"orbit{args.orbit_size}_a{alpha:g}"] = {"alpha": alpha, "orbit_size": args.orbit_size, "checkpoint": str(args.checkpoint)}
    tasks = []
    for opponent_index, opponent in enumerate(args.opponents.split(",")):
        for players in map(int, args.players.split(",")):
            if not 2 <= players <= 8:
                parser.error("player counts must be between 2 and 8")
            for start in range(0, args.blocks, args.chunk_blocks):
                tasks.append({
                    "opponent": opponent, "players": players,
                    "seed": args.seed + opponent_index * 100000 + players * 10000 + start,
                    "blocks": min(args.chunk_blocks, args.blocks - start),
                    "variants": list(variants), "policy_names": [*variants, "self_anchor"],
                })
    if any(task["seed"] + task["blocks"] >= 92000000 for task in tasks):
        parser.error("development seeds must stay below the existing 92m sealed namespace")
    started = time.perf_counter()
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    reference_sha256 = hashlib.sha256(reference_checkpoint.read_bytes()).hexdigest()
    rows = []
    context = multiprocessing.get_context("spawn")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with context.Pool(args.workers, initializer=_initialize_worker,
                      initargs=(str(args.checkpoint), str(args.mlp), variants)) as pool:
        for row in pool.imap_unordered(_evaluate_task, tasks, chunksize=1):
            rows.append(row)
            payload = {
                "kind": "native_min_gru_suit_orbit_development_screen",
                "development_only": True, "complete": len(rows) == len(tasks),
                "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_sha256,
                "reference_checkpoint": str(reference_checkpoint), "reference_checkpoint_sha256": reference_sha256,
                "variants": variants,
                "protocol": {"seed": args.seed, "blocks_per_stratum": args.blocks,
                             "players": args.players, "opponents": args.opponents,
                             "matched_seat_rotations": True, "deterministic_actions": True,
                             "information_boundary": "own hand and public 100-value native tokens only"},
                "elapsed_seconds": time.perf_counter() - started,
                "finished_tasks": len(rows), "total_tasks": len(tasks),
                "summary": summarize(rows, variants),
                "weighted_pool": summarize_pool(rows, variants), "rows": rows,
            }
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n")
            temporary.replace(args.output)
            print(json.dumps({"finished_tasks": len(rows), "total_tasks": len(tasks),
                              "elapsed_seconds": payload["elapsed_seconds"],
                              "summary": payload["summary"]}), flush=True)


if __name__ == "__main__":
    main()
