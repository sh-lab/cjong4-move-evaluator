"""Personality-specific teacher rewards reconstructed from dataset facts."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import (
    FACT_CALL_AVAILABLE,
    FACT_CHOSE_CALL,
    FACT_CHOSE_RIICHI,
    FACT_DEAL_IN_ACTION,
    FACT_PLAYER_DEALT_IN,
    FACT_PLAYER_WON,
    FACT_PLAYER_WON_KOKUSHI,
    FACT_RIICHI_AVAILABLE,
    MAX_DISCARDS,
    ROUND_END_EXHAUSTIVE_DRAW,
    TENPAI_YES,
)

PERSONALITIES = (
    "standard",
    "safe",
    "menzen",
    "call",
    "speed",
    "kokushi",
    "riichi",
    "dama",
)
DEFAULT_PERSONALITY_WEIGHT = 0.05
DEFAULT_REWARD_SCALE = 8000.0


class RewardFacts(Protocol):
    """Dataset columns needed to reconstruct personality rewards."""

    targets: np.ndarray
    fact_flags: np.ndarray
    deal_in_points: np.ndarray
    round_discard_counts: np.ndarray
    round_end_types: np.ndarray
    tenpai_statuses: np.ndarray


def _has(flags: np.ndarray, fact: int) -> np.ndarray:
    return (flags & np.uint16(fact)) != 0


def personality_component(
    dataset: RewardFacts,
    personality: str,
    *,
    reward_scale: float = DEFAULT_REWARD_SCALE,
) -> np.ndarray:
    """Return the unitless preference component for one personality preset."""
    if personality not in PERSONALITIES:
        raise ValueError(f"unknown personality: {personality}")
    if not math.isfinite(reward_scale) or reward_scale <= 0.0:
        raise ValueError("reward_scale must be finite and positive")

    flags = np.asarray(dataset.fact_flags, dtype=np.uint16)
    component = np.zeros(flags.shape, dtype=np.float64)

    if personality == "standard":
        return component.astype(np.float32)

    if personality == "safe":
        dealt_in = _has(flags, FACT_PLAYER_DEALT_IN)
        caused_deal_in = _has(flags, FACT_DEAL_IN_ACTION)
        loss = np.asarray(dataset.deal_in_points, dtype=np.float64) / reward_scale
        component[dealt_in] -= loss[dealt_in]
        component[caused_deal_in] -= loss[caused_deal_in]
    elif personality == "menzen":
        component[_has(flags, FACT_CHOSE_CALL)] = -1.0
    elif personality == "call":
        available = _has(flags, FACT_CALL_AVAILABLE)
        chose = _has(flags, FACT_CHOSE_CALL)
        component[available & chose] = 1.0
        component[available & ~chose] = -1.0
    elif personality == "speed":
        round_progress = np.asarray(
            dataset.round_discard_counts, dtype=np.float64
        ) / MAX_DISCARDS
        urgency = np.clip(1.0 - round_progress, 0.0, 1.0)
        won = _has(flags, FACT_PLAYER_WON)
        exhaustive_tenpai = (
            np.asarray(dataset.round_end_types) == ROUND_END_EXHAUSTIVE_DRAW
        ) & (np.asarray(dataset.tenpai_statuses) == TENPAI_YES)
        component[won] = urgency[won]
        component[exhaustive_tenpai] = 0.25 * urgency[exhaustive_tenpai]
    elif personality == "kokushi":
        component[_has(flags, FACT_PLAYER_WON_KOKUSHI)] = 1.0
    elif personality in ("riichi", "dama"):
        available = _has(flags, FACT_RIICHI_AVAILABLE)
        chose = _has(flags, FACT_CHOSE_RIICHI)
        riichi_preference = np.zeros(flags.shape, dtype=np.float64)
        riichi_preference[available & chose] = 1.0
        riichi_preference[available & ~chose] = -1.0
        component = (
            riichi_preference if personality == "riichi" else -riichi_preference
        )

    if not np.isfinite(component).all():
        raise ValueError("personality reward component is not finite")
    return component.astype(np.float32)


def compose_targets(
    dataset: RewardFacts,
    personality: str,
    *,
    personality_weight: float = DEFAULT_PERSONALITY_WEIGHT,
    reward_scale: float = DEFAULT_REWARD_SCALE,
) -> np.ndarray:
    """Add a personality preference to the dataset's default score target."""
    if not math.isfinite(personality_weight) or personality_weight < 0.0:
        raise ValueError("personality_weight must be finite and nonnegative")
    targets = np.asarray(dataset.targets, dtype=np.float32)
    component = personality_component(
        dataset, personality, reward_scale=reward_scale
    )
    result = targets.astype(np.float64) + personality_weight * component
    if not np.isfinite(result).all():
        raise ValueError("composed teacher targets are not finite")
    return result.astype(np.float32)


class TeacherDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """View of a feature dataset with reconstructed teacher targets."""

    def __init__(self, source: Dataset, targets: np.ndarray):
        if len(source) != len(targets):
            raise ValueError("teacher target count does not match dataset")
        self.source = source
        self.targets = np.asarray(targets, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features, _ = self.source[index]
        return features, torch.tensor(self.targets[index], dtype=torch.float32)


def build_teacher_dataset(
    source: RewardFacts,
    personality: str,
    *,
    personality_weight: float = DEFAULT_PERSONALITY_WEIGHT,
    reward_scale: float = DEFAULT_REWARD_SCALE,
) -> TeacherDataset:
    """Build a training view for one of the eight personality presets."""
    return TeacherDataset(
        source,  # type: ignore[arg-type]
        compose_targets(
            source,
            personality,
            personality_weight=personality_weight,
            reward_scale=reward_scale,
        ),
    )
