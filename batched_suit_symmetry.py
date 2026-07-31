"""Batched ActorPolicy-compatible suit-symmetry inference."""

from __future__ import annotations

from itertools import permutations

import torch

from ai_observation import (
    ACTION_COUNT,
    COMPATIBLE_END,
    COMPATIBLE_START,
    MODE_INDEX,
    OBSERVATION_SIZE,
    PASS_ACTION,
    PLAYER_HAND_SIZES_END,
    PLAYER_HAND_SIZES_START,
    PLAYER_INDEX,
    SORTS_PLAYED_START,
    SUIT_ACTION_END,
    SUIT_ACTION_START,
    DIRECTION_INDEX,
)


SUIT_PERMUTATIONS = tuple(permutations(range(4)))
CARDS_PER_SUIT = 12
PLAYER_ORBIT_MODES = ("none", "rotations", "reflection", "full")


def _action_masks(observations: torch.Tensor) -> torch.Tensor:
    masks = observations.new_zeros(
        (*observations.shape[:-1], ACTION_COUNT),
        dtype=torch.bool,
    )
    suit_mode = observations[..., MODE_INDEX] != 0
    masks[..., PASS_ACTION] = ~suit_mode
    masks[..., :50] = (
        observations[..., COMPATIBLE_START:COMPATIBLE_END] >= 1
    ) & ~suit_mode.unsqueeze(-1)
    masks[..., SUIT_ACTION_START:SUIT_ACTION_END] = suit_mode.unsqueeze(-1)
    return masks


def _permutation_indices() -> tuple[torch.Tensor, torch.Tensor]:
    observation_indices = torch.arange(OBSERVATION_SIZE).repeat(
        len(SUIT_PERMUTATIONS), 1
    )
    action_indices = torch.arange(ACTION_COUNT).repeat(
        len(SUIT_PERMUTATIONS), 1
    )
    for permutation_index, permutation in enumerate(SUIT_PERMUTATIONS):
        for original_suit, transformed_suit in enumerate(permutation):
            original_cards = torch.arange(
                original_suit * CARDS_PER_SUIT,
                (original_suit + 1) * CARDS_PER_SUIT,
            )
            transformed_cards = torch.arange(
                transformed_suit * CARDS_PER_SUIT,
                (transformed_suit + 1) * CARDS_PER_SUIT,
            )
            observation_indices[
                permutation_index, transformed_cards
            ] = original_cards
            observation_indices[
                permutation_index,
                COMPATIBLE_START + transformed_cards,
            ] = COMPATIBLE_START + original_cards
            observation_indices[
                permutation_index,
                SORTS_PLAYED_START + transformed_suit,
            ] = SORTS_PLAYED_START + original_suit

            # To map back, each original action gathers the corresponding
            # action from the transformed suit coordinates.
            action_indices[
                permutation_index, original_cards
            ] = transformed_cards
            action_indices[
                permutation_index,
                SUIT_ACTION_START + original_suit,
            ] = SUIT_ACTION_START + transformed_suit
    return observation_indices, action_indices


class BatchedSuitAveragedActor(torch.nn.Module):
    """Blend legacy logits with full 24-way suit-averaged logits."""

    def __init__(
        self,
        actor: torch.nn.Module,
        *,
        alpha: float = 1.0,
        aggregation: str = "logits",
        player_orbit_average: bool = False,
        player_orbit_mode: str | None = None,
        base_chunk_size: int = 512,
    ):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between zero and one")
        if base_chunk_size < 1:
            raise ValueError("base_chunk_size must be positive")
        if aggregation not in ("logits", "probabilities"):
            raise ValueError(
                "aggregation must be 'logits' or 'probabilities'"
            )
        if player_orbit_mode is None:
            player_orbit_mode = (
                "full" if player_orbit_average else "none"
            )
        if player_orbit_mode not in PLAYER_ORBIT_MODES:
            raise ValueError(
                "player_orbit_mode must be one of "
                + ", ".join(PLAYER_ORBIT_MODES)
            )
        if player_orbit_average and player_orbit_mode != "full":
            raise ValueError(
                "player_orbit_average=True is compatible only with "
                "player_orbit_mode='full'"
            )
        self.actor = actor
        self.alpha = float(alpha)
        self.aggregation = aggregation
        self.player_orbit_mode = player_orbit_mode
        self.player_orbit_average = player_orbit_mode != "none"
        self.base_chunk_size = int(base_chunk_size)
        observation_indices, action_indices = _permutation_indices()
        self.register_buffer(
            "observation_indices",
            observation_indices,
            persistent=False,
        )
        self.register_buffer(
            "action_indices",
            action_indices,
            persistent=False,
        )
        self.forward_calls = 0
        self.base_observations = 0
        self.augmented_observations = 0
        self.underlying_actor_calls = 0

    def reset_statistics(self) -> None:
        self.forward_calls = 0
        self.base_observations = 0
        self.augmented_observations = 0
        self.underlying_actor_calls = 0

    def statistics(self) -> dict:
        return {
            "alpha": self.alpha,
            "aggregation": self.aggregation,
            "player_orbit_average": self.player_orbit_average,
            "player_orbit_mode": self.player_orbit_mode,
            "forward_calls": self.forward_calls,
            "base_observations": self.base_observations,
            "augmented_observations": self.augmented_observations,
            "underlying_actor_calls": self.underlying_actor_calls,
            "mean_base_batch_size": (
                self.base_observations / self.forward_calls
                if self.forward_calls
                else 0.0
            ),
        }

    def _map_and_project(
        self,
        transformed_logits: torch.Tensor,
        transformed_observations: torch.Tensor,
    ) -> torch.Tensor:
        """Map suit actions back and project over every orbit dimension."""
        if self.aggregation == "probabilities":
            transformed_logits = transformed_logits.masked_fill(
                ~_action_masks(transformed_observations),
                torch.finfo(transformed_logits.dtype).min,
            )
        mapped_back = torch.gather(
            transformed_logits,
            dim=-1,
            index=self.action_indices.view(
                1,
                len(SUIT_PERMUTATIONS),
                *([1] * (transformed_logits.ndim - 3)),
                ACTION_COUNT,
            ).expand(*transformed_logits.shape[:-1], ACTION_COUNT),
        )
        if self.aggregation == "probabilities":
            mapped_back = mapped_back.softmax(dim=-1)
        orbit_dimensions = tuple(range(1, mapped_back.ndim - 1))
        return mapped_back.mean(dim=orbit_dimensions)

    def _suit_only_chunk(self, observations: torch.Tensor) -> torch.Tensor:
        transformed = torch.gather(
            observations.unsqueeze(1).expand(
                -1, len(SUIT_PERMUTATIONS), -1
            ),
            dim=2,
            index=self.observation_indices.unsqueeze(0).expand(
                len(observations), -1, -1
            ),
        )
        flattened = transformed.reshape(-1, OBSERVATION_SIZE)
        transformed_logits = self.actor(flattened).reshape(
            len(observations),
            len(SUIT_PERMUTATIONS),
            ACTION_COUNT,
        )
        self.underlying_actor_calls += 1
        self.augmented_observations += len(flattened)
        return self._map_and_project(transformed_logits, transformed)

    def _player_and_suit_chunk(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        player_counts = (
            observations[
                :, PLAYER_HAND_SIZES_START:PLAYER_HAND_SIZES_END
            ]
            > 0
        ).sum(dim=-1)
        if torch.any(player_counts < 2):
            raise ValueError(
                "player orbit requires at least two positive hand-size slots"
            )
        projected = observations.new_empty(
            (len(observations), ACTION_COUNT)
        )
        for player_count_tensor in torch.unique(player_counts):
            player_count = int(player_count_tensor.item())
            selected = torch.nonzero(
                player_counts == player_count_tensor
            ).flatten()
            group = observations[selected]
            suit_transformed = torch.gather(
                group.unsqueeze(1).expand(
                    -1, len(SUIT_PERMUTATIONS), -1
                ),
                dim=2,
                index=self.observation_indices.unsqueeze(0).expand(
                    len(group), -1, -1
                ),
            )
            original_player_index = group[:, PLAYER_INDEX].long()
            original_direction = group[:, DIRECTION_INDEX]
            if self.player_orbit_mode == "full":
                orbit_size = 2 * player_count
                orbit_player_indices = torch.arange(
                    player_count,
                    dtype=torch.long,
                    device=group.device,
                ).repeat(2).view(1, -1).expand(len(group), -1)
                orbit_directions = torch.cat(
                    (
                        original_direction[:, None].expand(
                            -1, player_count
                        ),
                        -original_direction[:, None].expand(
                            -1, player_count
                        ),
                    ),
                    dim=1,
                )
            elif self.player_orbit_mode == "rotations":
                orbit_size = player_count
                orbit_player_indices = torch.arange(
                    player_count,
                    dtype=torch.long,
                    device=group.device,
                ).view(1, -1).expand(len(group), -1)
                orbit_directions = original_direction[:, None].expand(
                    -1, player_count
                )
            elif self.player_orbit_mode == "reflection":
                orbit_size = 2
                reflected_index = torch.remainder(
                    -original_player_index, player_count
                )
                orbit_player_indices = torch.stack(
                    (original_player_index, reflected_index), dim=1
                )
                orbit_directions = torch.stack(
                    (original_direction, -original_direction), dim=1
                )
            else:
                raise RuntimeError("player orbit mode is unexpectedly disabled")
            transformed = suit_transformed.unsqueeze(2).expand(
                -1, -1, orbit_size, -1
            ).clone()
            transformed[:, :, :, PLAYER_INDEX] = (
                orbit_player_indices[:, None, :]
            )
            transformed[:, :, :, DIRECTION_INDEX] = (
                orbit_directions[:, None, :]
            )
            flattened = transformed.reshape(-1, OBSERVATION_SIZE)
            transformed_logits = self.actor(flattened).reshape(
                len(group),
                len(SUIT_PERMUTATIONS),
                orbit_size,
                ACTION_COUNT,
            )
            self.underlying_actor_calls += 1
            self.augmented_observations += len(flattened)
            projected[selected] = self._map_and_project(
                transformed_logits,
                transformed,
            )
        return projected

    def _averaged_chunk(self, observations: torch.Tensor) -> torch.Tensor:
        if self.player_orbit_mode != "none":
            return self._player_and_suit_chunk(observations)
        return self._suit_only_chunk(observations)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != OBSERVATION_SIZE:
            raise ValueError(
                f"expected observations shaped (batch, {OBSERVATION_SIZE})"
            )
        self.forward_calls += 1
        self.base_observations += len(observations)
        if self.alpha == 0.0:
            self.underlying_actor_calls += 1
            return self.actor(observations)

        averaged_parts = []
        for start in range(0, len(observations), self.base_chunk_size):
            averaged_parts.append(
                self._averaged_chunk(
                    observations[start : start + self.base_chunk_size]
                )
            )
        averaged = torch.cat(averaged_parts)
        if self.aggregation == "logits":
            if self.alpha == 1.0:
                return averaged
            legacy = self.actor(observations)
            self.underlying_actor_calls += 1
            return torch.lerp(legacy, averaged, self.alpha)

        if self.alpha == 1.0:
            blended_probabilities = averaged
        else:
            legacy = self.actor(observations)
            self.underlying_actor_calls += 1
            blended_probabilities = torch.lerp(
                legacy.masked_fill(
                    ~_action_masks(observations),
                    torch.finfo(legacy.dtype).min,
                ).softmax(dim=-1),
                averaged,
                self.alpha,
            )
        return blended_probabilities.clamp_min(
            torch.finfo(blended_probabilities.dtype).tiny
        ).log()
