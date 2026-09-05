import argparse
import struct

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cj4me.dataset import (
    DATASET_FORMAT_VERSION,
    DATASET_MAGIC,
    DATASET_RECORD_SIZE,
    FEATURE_COUNT,
    FEATURE_SCHEMA_VERSION,
)
from cj4me.train import (
    RewardMetrics,
    ValidationTracker,
    evaluate_reward_metrics,
    format_reward_metrics,
    train,
)


def write_dataset(path):
    with path.open("wb") as stream:
        stream.write(
            struct.pack(
                "<8s6I",
                DATASET_MAGIC,
                DATASET_FORMAT_VERSION,
                FEATURE_SCHEMA_VERSION,
                FEATURE_COUNT,
                1,
                DATASET_RECORD_SIZE,
                0,
            )
        )
        stream.write(np.zeros(FEATURE_COUNT, dtype="<f4").tobytes())
        stream.write(struct.pack("<fBBH", 0.0, 0, 0, 0))


def test_validation_tracker_uses_min_delta_and_patience():
    tracker = ValidationTracker(patience=2, min_delta=0.01)

    assert tracker.update(1.0, 1) == (True, False)
    assert tracker.update(0.995, 2) == (True, False)
    assert tracker.update(0.994, 3) == (True, True)
    assert tracker.best_epoch == 3
    assert tracker.best_loss == pytest.approx(0.994)


def test_zero_patience_disables_early_stopping():
    tracker = ValidationTracker(patience=0, min_delta=0.0)

    assert tracker.update(1.0, 1) == (True, False)
    assert tracker.update(2.0, 2) == (False, False)


def test_reward_metrics_split_zero_and_nonzero_targets():
    predictions = torch.tensor([[0.5], [0.0], [0.5], [-1.0]])
    targets = torch.tensor([0.0, 0.0, 1.0, -2.0])
    loader = DataLoader(TensorDataset(predictions, targets), batch_size=2)

    metrics = evaluate_reward_metrics(nn.Identity(), loader, torch.device("cpu"))

    assert metrics.records == 4
    assert metrics.zero_records == 2
    assert metrics.nonzero_records == 2
    assert metrics.zero_ratio == pytest.approx(0.5)
    assert metrics.mse == pytest.approx(0.375)
    assert metrics.zero_mse == pytest.approx(0.125)
    assert metrics.nonzero_mse == pytest.approx(0.625)
    assert metrics.nonzero_mae == pytest.approx(0.75)
    assert metrics.nonzero_sign_accuracy == pytest.approx(1.0)
    assert metrics.zero_baseline_mse == pytest.approx(1.25)
    assert metrics.baseline_improvement == pytest.approx(0.7)
    assert metrics.nonzero_baseline_mse == pytest.approx(2.5)
    assert metrics.nonzero_baseline_improvement == pytest.approx(0.75)


def test_all_zero_metrics_report_nonzero_values_as_unavailable():
    predictions = torch.zeros(2, 1)
    targets = torch.zeros(2)
    loader = DataLoader(TensorDataset(predictions, targets), batch_size=2)

    metrics = evaluate_reward_metrics(nn.Identity(), loader, torch.device("cpu"))
    output = format_reward_metrics("validation", metrics)

    assert metrics.zero_mse == 0.0
    assert metrics.nonzero_mse is None
    assert metrics.nonzero_sign_accuracy is None
    assert metrics.baseline_improvement is None
    assert "nonzero_mse=N/A" in output
    assert "baseline_improvement=N/A" in output


def metrics_with_mse(mse):
    return RewardMetrics(
        records=1,
        zero_records=1,
        nonzero_records=0,
        mse=mse,
        zero_mse=mse,
        nonzero_mse=None,
        nonzero_mae=None,
        nonzero_sign_accuracy=None,
        zero_baseline_mse=0.0,
        baseline_improvement=None,
        nonzero_baseline_mse=None,
        nonzero_baseline_improvement=None,
    )


def test_training_saves_best_epoch_and_stops(tmp_path, monkeypatch):
    training_path = tmp_path / "training.cj4medata"
    validation_path = tmp_path / "validation.cj4medata"
    output_path = tmp_path / "best.pt"
    write_dataset(training_path)
    write_dataset(validation_path)
    metrics = iter(
        (
            metrics_with_mse(0.2),
            metrics_with_mse(0.25),
            metrics_with_mse(0.1),
            metrics_with_mse(0.5),
        )
    )
    monkeypatch.setattr(
        "cj4me.train.evaluate_reward_metrics",
        lambda model, loader, device: next(metrics),
    )
    args = argparse.Namespace(
        dataset=str(training_path),
        validation_dataset=str(validation_path),
        epochs=10,
        batch_size=1,
        lr=1e-3,
        patience=1,
        min_delta=0.0,
        seed=1,
        device="cpu",
        output=str(output_path),
    )

    checkpoint = train(args)
    saved = torch.load(output_path, map_location="cpu", weights_only=True)

    assert checkpoint["epoch"] == 1
    assert checkpoint["completed_epochs"] == 2
    assert checkpoint["early_stopped"] is True
    assert checkpoint["best_validation_loss"] == pytest.approx(0.25)
    assert checkpoint["training_metrics_at_best"]["mse"] == pytest.approx(0.2)
    assert checkpoint["validation_metrics_at_best"]["mse"] == pytest.approx(0.25)
    assert saved["epoch"] == 1
    assert saved["completed_epochs"] == 2
    assert saved["early_stopped"] is True
