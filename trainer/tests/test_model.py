import torch
from torch import nn

from cj4me.dataset import FEATURE_COUNT, FEATURE_SCHEMA_VERSION
from cj4me.model import MODEL_DIMENSIONS, MoveEvaluator, checkpoint_metadata


def test_model_has_fixed_architecture():
    model = MoveEvaluator()
    linear_layers = [layer for layer in model.network if isinstance(layer, nn.Linear)]
    relu_layers = [layer for layer in model.network if isinstance(layer, nn.ReLU)]

    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (FEATURE_COUNT, 128),
        (128, 64),
        (64, 16),
        (16, 1),
    ]
    assert len(relu_layers) == 3


def test_forward_shape_and_finite_result():
    torch.manual_seed(1)
    model = MoveEvaluator()
    result = model(torch.zeros(4, FEATURE_COUNT))
    assert result.shape == (4, 1)
    assert torch.isfinite(result).all()


def test_checkpoint_metadata_describes_compatibility():
    metadata = checkpoint_metadata()
    model_metadata = MoveEvaluator.checkpoint_metadata()
    assert MODEL_DIMENSIONS == (FEATURE_COUNT, 128, 64, 16, 1)
    assert metadata == model_metadata
    assert model_metadata["checkpoint_format_version"] == 1
    assert model_metadata["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert model_metadata["feature_count"] == FEATURE_COUNT
    assert model_metadata["dimensions"] == [FEATURE_COUNT, 128, 64, 16, 1]
