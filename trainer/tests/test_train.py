import argparse
import struct

import numpy as np
import pytest
import torch

from cj4me.dataset import (
    DATASET_FORMAT_VERSION,
    DATASET_MAGIC,
    DATASET_RECORD_SIZE,
    FEATURE_COUNT,
    FEATURE_SCHEMA_VERSION,
)
from cj4me.train import ValidationTracker, train


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


def test_training_saves_best_epoch_and_stops(tmp_path, monkeypatch):
    training_path = tmp_path / "training.cj4medata"
    validation_path = tmp_path / "validation.cj4medata"
    output_path = tmp_path / "best.pt"
    write_dataset(training_path)
    write_dataset(validation_path)
    losses = iter((0.25, 0.5))
    monkeypatch.setattr(
        "cj4me.train.mean_loss", lambda model, loader, loss, device: next(losses)
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
    assert saved["epoch"] == 1
    assert saved["completed_epochs"] == 2
    assert saved["early_stopped"] is True
