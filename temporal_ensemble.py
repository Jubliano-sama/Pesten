"""Privacy-safe deterministic ensembles of recurrent temporal policies."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from benchmark_model_zoo import DEFAULT_TEACHER
from residual_temporal_policy import ResidualRecurrentTemporalPolicy
from temporal_observation import ACTION_COUNT, DecisionInput
from temporal_puffer_agent import PolicyStep


COMBINATION_MODES = (
    "mean_logits",
    "mean_probabilities",
    "majority_vote",
)


class TemporalRecurrentEnsemble(nn.Module):
    """Combine recurrent policies while retaining one safe state per member."""

    architecture = "residual_recurrent_ensemble"
    critic_privileged = False

    def __init__(
        self,
        members: Sequence[ResidualRecurrentTemporalPolicy],
        *,
        combination: str = "mean_logits",
        member_names: Sequence[str] | None = None,
        checkpoint_metadata: Sequence[dict] | None = None,
    ) -> None:
        super().__init__()
        if not members:
            raise ValueError("an ensemble requires at least one member")
        if combination not in COMBINATION_MODES:
            raise ValueError(
                f"combination must be one of {COMBINATION_MODES}"
            )
        if any(
            not isinstance(member, ResidualRecurrentTemporalPolicy)
            for member in members
        ):
            raise TypeError(
                "ensemble members must be ResidualRecurrentTemporalPolicy"
            )
        champion_hashes = {member.champion_sha256 for member in members}
        if len(champion_hashes) != 1:
            raise ValueError("ensemble members must share the same PPO base")
        self.members = nn.ModuleList(members)
        self.combination = combination
        self.member_names = tuple(
            member_names
            if member_names is not None
            else (f"member_{index}" for index in range(len(members)))
        )
        if len(self.member_names) != len(members):
            raise ValueError("member_names must align with members")
        self.checkpoint_metadata = tuple(checkpoint_metadata or ())
        self.champion_sha256 = next(iter(champion_hashes))
        self._freeze_members()

    @classmethod
    def from_checkpoints(
        cls,
        checkpoints: Sequence[Path],
        *,
        champion_path: Path = DEFAULT_TEACHER,
        device: torch.device | str = "cpu",
        combination: str = "mean_logits",
        member_names: Sequence[str] | None = None,
    ) -> "TemporalRecurrentEnsemble":
        if not checkpoints:
            raise ValueError("at least one checkpoint path is required")
        members = []
        metadata = []
        for path in checkpoints:
            path = Path(path)
            member = ResidualRecurrentTemporalPolicy.load_checkpoint(
                path,
                champion_path=champion_path,
                map_location=device,
            ).eval()
            members.append(member)
            metadata.append(
                {
                    "path": str(path),
                    "sha256": _file_sha256(path),
                    "cell": member.cell,
                    "hidden_size": member.hidden_size,
                    "num_layers": member.num_layers,
                }
            )
        return cls(
            members,
            combination=combination,
            member_names=member_names,
            checkpoint_metadata=metadata,
        ).to(device)

    def _freeze_members(self) -> None:
        for member in self.members:
            member.eval()
            for parameter in member.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "TemporalRecurrentEnsemble":
        # This is an evaluation ensemble, not a jointly trainable policy.
        super().train(False)
        self._freeze_members()
        return self

    def initial_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if batch_size != 1:
            raise ValueError("online ensemble state supports batch_size=1")
        flattened = []
        for member in self.members:
            member_state = member.initial_state(
                batch_size=batch_size, device=device
            )
            if isinstance(member_state, torch.Tensor):
                flattened.append(member_state)
            else:
                flattened.extend(member_state)
        return tuple(flattened)

    def _split_state(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        if not isinstance(state, tuple):
            raise TypeError("ensemble state must be a flat tensor tuple")
        expected_parts = sum(
            2 if member.cell == "lstm" else 1 for member in self.members
        )
        if len(state) != expected_parts or any(
            not isinstance(part, torch.Tensor) for part in state
        ):
            raise TypeError(
                f"ensemble state must contain {expected_parts} tensors"
            )
        split = []
        cursor = 0
        for member in self.members:
            if member.cell == "lstm":
                split.append((state[cursor], state[cursor + 1]))
                cursor += 2
            else:
                split.append(state[cursor])
                cursor += 1
        return split

    def step(
        self,
        decision: DecisionInput,
        state: tuple[torch.Tensor, ...],
    ) -> PolicyStep:
        if not isinstance(decision, DecisionInput):
            raise TypeError("ensemble accepts only DecisionInput")
        member_states = self._split_state(state)
        member_results = []
        next_state_parts = []
        with torch.no_grad():
            for member, member_state in zip(
                self.members, member_states
            ):
                result = member.step(decision, member_state)
                member_results.append(result)
                if isinstance(result.state, torch.Tensor):
                    next_state_parts.append(result.state)
                else:
                    next_state_parts.extend(result.state)
        stacked_logits = torch.stack(
            [result.logits for result in member_results]
        )
        legal = torch.tensor(
            decision.legal_actions,
            device=stacked_logits.device,
            dtype=torch.bool,
        )
        combined = combine_legal_logits(
            stacked_logits,
            legal,
            mode=self.combination,
        )
        value = torch.stack(
            [result.value for result in member_results]
        ).mean()
        return PolicyStep(
            logits=combined,
            value=value,
            state=tuple(next_state_parts),
        )


def combine_legal_logits(
    member_logits: torch.Tensor,
    legal_actions: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Combine already-safe member outputs with deterministic tie-breaking."""
    if mode not in COMBINATION_MODES:
        raise ValueError(f"unknown combination mode: {mode}")
    if (
        member_logits.ndim != 2
        or member_logits.shape[1] != ACTION_COUNT
    ):
        raise ValueError("member_logits must have shape [members, 55]")
    if legal_actions.shape != (ACTION_COUNT,) or (
        legal_actions.dtype is not torch.bool
    ):
        raise ValueError("legal_actions must be a boolean [55] tensor")
    if not bool(legal_actions.any()):
        raise ValueError("at least one action must be legal")

    if mode == "mean_logits":
        combined = member_logits[:, legal_actions].mean(dim=0)
    else:
        probabilities = torch.softmax(member_logits, dim=-1)
        mean_probabilities = probabilities.mean(dim=0)
        if mode == "mean_probabilities":
            combined = torch.log(
                mean_probabilities[legal_actions].clamp_min(
                    torch.finfo(mean_probabilities.dtype).tiny
                )
            )
        else:
            actions = member_logits.argmax(dim=-1)
            votes = torch.zeros(
                ACTION_COUNT,
                device=member_logits.device,
                dtype=member_logits.dtype,
            )
            votes.scatter_add_(
                0,
                actions,
                torch.ones_like(actions, dtype=member_logits.dtype),
            )
            # Vote count decides first; mean probability is a deterministic
            # sub-unit tie breaker without overruling a one-vote advantage.
            combined = (
                votes[legal_actions]
                + mean_probabilities[legal_actions]
            )

    result = member_logits.new_full(
        (ACTION_COUNT,), torch.finfo(member_logits.dtype).min
    )
    result[legal_actions] = combined
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
