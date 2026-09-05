"""Fixed PyTorch model used by cjong4-move-evaluator."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .dataset import FEATURE_COUNT, FEATURE_SCHEMA_VERSION

MODEL_ARCHITECTURE = f"mlp-{FEATURE_COUNT}-128-64-16-1"
MODEL_DIMENSIONS = (FEATURE_COUNT, 128, 64, 16, 1)
CHECKPOINT_FORMAT_VERSION = 1


def checkpoint_metadata() -> dict[str, Any]:
    """Return compatibility metadata stored in every training checkpoint."""
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": FEATURE_COUNT,
        "architecture": MODEL_ARCHITECTURE,
        "dimensions": list(MODEL_DIMENSIONS),
    }


class MoveEvaluator(nn.Module):
    """The fixed feature-count -> 128 -> 64 -> 16 -> 1 value network."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURE_COUNT, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    @staticmethod
    def checkpoint_metadata() -> dict[str, Any]:
        return checkpoint_metadata()
