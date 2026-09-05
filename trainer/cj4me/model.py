"""Physical-tile PyTorch model used by cjong4-move-evaluator."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .dataset import (
    FEATURE_COUNT,
    FEATURE_SCHEMA_VERSION,
    SCORE_INPUT_COUNT,
    TILE_COUNT,
    TILE_EMBEDDING_COUNT,
    TILE_FEATURE_COUNT,
    TILE_FEATURES_COUNT,
)

MODEL_TILE_HIDDEN_COUNT = 16
MODEL_HIDDEN1_COUNT = 128
MODEL_HIDDEN2_COUNT = 64
MODEL_HIDDEN3_COUNT = 16
MODEL_ARCHITECTURE = "tile-13-16-8x136-score-1123-128-64-16-1"
MODEL_DIMENSIONS = (
    TILE_FEATURE_COUNT,
    MODEL_TILE_HIDDEN_COUNT,
    TILE_EMBEDDING_COUNT,
    SCORE_INPUT_COUNT,
    MODEL_HIDDEN1_COUNT,
    MODEL_HIDDEN2_COUNT,
    MODEL_HIDDEN3_COUNT,
    1,
)
CHECKPOINT_FORMAT_VERSION = 2


def checkpoint_metadata() -> dict[str, Any]:
    """Return compatibility metadata stored in every training checkpoint."""
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": FEATURE_COUNT,
        "architecture": MODEL_ARCHITECTURE,
        "dimensions": list(MODEL_DIMENSIONS),
        "tile_count": TILE_COUNT,
    }


class MoveEvaluator(nn.Module):
    """The shared Tile Encoder followed by the global action-value network."""

    def __init__(self) -> None:
        super().__init__()
        self.tile_encoder = nn.Sequential(
            nn.Linear(TILE_FEATURE_COUNT, MODEL_TILE_HIDDEN_COUNT),
            nn.ReLU(),
            nn.Linear(MODEL_TILE_HIDDEN_COUNT, TILE_EMBEDDING_COUNT),
            nn.ReLU(),
        )
        self.score_network = nn.Sequential(
            nn.Linear(SCORE_INPUT_COUNT, MODEL_HIDDEN1_COUNT),
            nn.ReLU(),
            nn.Linear(MODEL_HIDDEN1_COUNT, MODEL_HIDDEN2_COUNT),
            nn.ReLU(),
            nn.Linear(MODEL_HIDDEN2_COUNT, MODEL_HIDDEN3_COUNT),
            nn.ReLU(),
            nn.Linear(MODEL_HIDDEN3_COUNT, 1),
        )

    def score_input(self, features: torch.Tensor) -> torch.Tensor:
        """Encode all physical tiles and append the 35 context features."""
        if features.shape[-1] != FEATURE_COUNT:
            raise ValueError(f"features must have shape (..., {FEATURE_COUNT})")
        tiles = features[..., :TILE_FEATURES_COUNT].reshape(
            *features.shape[:-1], TILE_COUNT, TILE_FEATURE_COUNT
        )
        embeddings = self.tile_encoder(tiles).flatten(start_dim=-2)
        context = features[..., TILE_FEATURES_COUNT:]
        return torch.cat((embeddings, context), dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.score_network(self.score_input(features))

    @staticmethod
    def checkpoint_metadata() -> dict[str, Any]:
        return checkpoint_metadata()
