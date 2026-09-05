"""Command-line training entry point for the CJ4ME value model."""

from __future__ import annotations

import argparse
import copy
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import DatasetV1
from .model import MoveEvaluator, checkpoint_metadata


@dataclass
class ValidationTracker:
    """Track the best validation loss and an optional early-stop patience."""

    patience: int
    min_delta: float
    best_loss: float = math.inf
    best_epoch: int = 0
    patience_reference_loss: float = math.inf
    epochs_without_improvement: int = 0

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ValueError("patience must be nonnegative")
        if self.min_delta < 0 or not math.isfinite(self.min_delta):
            raise ValueError("min_delta must be finite and nonnegative")

    def update(self, loss: float, epoch: int) -> tuple[bool, bool]:
        if not math.isfinite(loss):
            raise ValueError("validation loss is not finite")
        new_best = loss < self.best_loss
        if new_best:
            self.best_loss = loss
            self.best_epoch = epoch
        significant_improvement = (
            loss < self.patience_reference_loss - self.min_delta
        )
        if significant_improvement:
            self.patience_reference_loss = loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        should_stop = (
            self.patience > 0
            and self.epochs_without_improvement >= self.patience
        )
        return new_best, should_stop


@dataclass(frozen=True)
class RewardMetrics:
    """Regression metrics split by exactly-zero and nonzero rewards."""

    records: int
    zero_records: int
    nonzero_records: int
    mse: float
    zero_mse: float | None
    nonzero_mse: float | None
    nonzero_mae: float | None
    nonzero_sign_accuracy: float | None
    zero_baseline_mse: float
    baseline_improvement: float | None
    nonzero_baseline_mse: float | None
    nonzero_baseline_improvement: float | None

    @property
    def zero_ratio(self) -> float:
        return self.zero_records / self.records


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(requested)


def evaluate_reward_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> RewardMetrics:
    model.eval()
    records = 0
    zero_records = 0
    squared_error_sum = 0.0
    zero_squared_error_sum = 0.0
    nonzero_squared_error_sum = 0.0
    nonzero_absolute_error_sum = 0.0
    nonzero_baseline_squared_error_sum = 0.0
    nonzero_sign_correct = 0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device).unsqueeze(1)
            predictions = model(features)
            if predictions.shape != targets.shape:
                raise ValueError("model output shape does not match targets")
            errors = predictions - targets
            squared_errors = errors.square()
            zero_mask = targets == 0.0
            nonzero_mask = ~zero_mask

            records += targets.numel()
            zero_records += int(zero_mask.sum().item())
            squared_error_sum += float(squared_errors.sum().item())
            zero_squared_error_sum += float(squared_errors[zero_mask].sum().item())
            nonzero_squared_error_sum += float(
                squared_errors[nonzero_mask].sum().item()
            )
            nonzero_absolute_error_sum += float(
                errors[nonzero_mask].abs().sum().item()
            )
            nonzero_baseline_squared_error_sum += float(
                targets[nonzero_mask].square().sum().item()
            )
            nonzero_sign_correct += int(
                ((predictions[nonzero_mask] * targets[nonzero_mask]) > 0.0)
                .sum()
                .item()
            )

    if records == 0:
        raise ValueError("metrics dataset is empty")
    if not all(
        math.isfinite(value)
        for value in (
            squared_error_sum,
            zero_squared_error_sum,
            nonzero_squared_error_sum,
            nonzero_absolute_error_sum,
            nonzero_baseline_squared_error_sum,
        )
    ):
        raise ValueError("reward metrics are not finite")
    nonzero_records = records - zero_records
    mse = squared_error_sum / records
    zero_baseline_mse = nonzero_baseline_squared_error_sum / records
    baseline_improvement = (
        1.0 - mse / zero_baseline_mse if zero_baseline_mse > 0.0 else None
    )
    if nonzero_records:
        nonzero_mse = nonzero_squared_error_sum / nonzero_records
        nonzero_baseline_mse = (
            nonzero_baseline_squared_error_sum / nonzero_records
        )
        nonzero_improvement = (
            1.0 - nonzero_mse / nonzero_baseline_mse
            if nonzero_baseline_mse > 0.0
            else None
        )
        nonzero_mae = nonzero_absolute_error_sum / nonzero_records
        sign_accuracy = nonzero_sign_correct / nonzero_records
    else:
        nonzero_mse = None
        nonzero_baseline_mse = None
        nonzero_improvement = None
        nonzero_mae = None
        sign_accuracy = None
    return RewardMetrics(
        records=records,
        zero_records=zero_records,
        nonzero_records=nonzero_records,
        mse=mse,
        zero_mse=(zero_squared_error_sum / zero_records if zero_records else None),
        nonzero_mse=nonzero_mse,
        nonzero_mae=nonzero_mae,
        nonzero_sign_accuracy=sign_accuracy,
        zero_baseline_mse=zero_baseline_mse,
        baseline_improvement=baseline_improvement,
        nonzero_baseline_mse=nonzero_baseline_mse,
        nonzero_baseline_improvement=nonzero_improvement,
    )


def _format_optional(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2%}" if percent else f"{value:.8f}"


def format_reward_metrics(name: str, metrics: RewardMetrics) -> str:
    return (
        f"{name}_metrics records={metrics.records} zero={metrics.zero_ratio:.2%} "
        f"mse={metrics.mse:.8f} "
        f"zero_mse={_format_optional(metrics.zero_mse)} "
        f"nonzero_mse={_format_optional(metrics.nonzero_mse)} "
        f"nonzero_mae={_format_optional(metrics.nonzero_mae)} "
        "nonzero_sign_accuracy="
        f"{_format_optional(metrics.nonzero_sign_accuracy, percent=True)} "
        f"zero_baseline_mse={metrics.zero_baseline_mse:.8f} "
        "baseline_improvement="
        f"{_format_optional(metrics.baseline_improvement, percent=True)} "
        "nonzero_baseline_mse="
        f"{_format_optional(metrics.nonzero_baseline_mse)} "
        "nonzero_baseline_improvement="
        f"{_format_optional(metrics.nonzero_baseline_improvement, percent=True)}"
    )


def train(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = select_device(args.device)
    training_set = DatasetV1(args.dataset)
    validation_set = DatasetV1(args.validation_dataset)
    if Path(args.dataset).resolve() == Path(args.validation_dataset).resolve():
        raise ValueError("training and validation datasets must be different files")
    if len(training_set) == 0:
        raise ValueError("training dataset is empty")
    if len(validation_set) == 0:
        raise ValueError("validation dataset is empty")

    loader_generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(
        training_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
    )
    training_metrics_loader = DataLoader(
        training_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = MoveEvaluator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.MSELoss()
    tracker = ValidationTracker(args.patience, args.min_delta)
    best_checkpoint: dict | None = None
    completed_epochs = 0
    early_stopped = False

    print(
        f"device={device} train={len(training_set)} "
        f"validation={len(validation_set)}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for features, targets in training_loader:
            features = features.to(device)
            targets = targets.to(device).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.shape[0]
            total_count += features.shape[0]

        training_loss = total_loss / total_count
        if not math.isfinite(training_loss):
            raise ValueError("training loss is not finite")
        training_metrics = evaluate_reward_metrics(
            model, training_metrics_loader, device
        )
        validation_metrics = evaluate_reward_metrics(
            model, validation_loader, device
        )
        validation_loss = validation_metrics.mse
        improved, should_stop = tracker.update(validation_loss, epoch)
        print(
            f"epoch={epoch}/{args.epochs} "
            f"train_loss={training_loss:.8f} validation_loss={validation_loss:.8f}"
            f"{' best' if improved else ''}"
        )
        print(format_reward_metrics("train", training_metrics))
        print(format_reward_metrics("validation", validation_metrics))
        completed_epochs = epoch
        if improved:
            best_checkpoint = {
                "metadata": checkpoint_metadata(),
                "model_state_dict": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                },
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "epoch": epoch,
                "best_validation_loss": validation_loss,
                "training_loss_at_best": training_loss,
                "training_metrics_at_best": asdict(training_metrics),
                "validation_metrics_at_best": asdict(validation_metrics),
                "seed": args.seed,
                "training_records": len(training_set),
                "validation_records": len(validation_set),
                "training_dataset": str(Path(args.dataset)),
                "validation_dataset": str(Path(args.validation_dataset)),
                "epochs_requested": args.epochs,
                "patience": args.patience,
                "min_delta": args.min_delta,
            }
        if should_stop:
            early_stopped = True
            print(
                f"early_stop epoch={epoch} best_epoch={tracker.best_epoch} "
                f"best_validation_loss={tracker.best_loss:.8f}"
            )
            break

    if best_checkpoint is None:
        raise ValueError("training did not produce a valid checkpoint")
    best_checkpoint["completed_epochs"] = completed_epochs
    best_checkpoint["early_stopped"] = early_stopped
    best_checkpoint["epochs_without_improvement"] = (
        tracker.epochs_without_improvement
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_checkpoint, output)
    print(
        f"saved best checkpoint: {output} epoch={tracker.best_epoch} "
        f"validation_loss={tracker.best_loss:.8f}"
    )
    return best_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the CJ4ME move evaluator")
    parser.add_argument("--dataset", required=True, help="input .cj4medata file")
    parser.add_argument(
        "--validation-dataset",
        required=True,
        help="independently generated validation .cj4medata file",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="stop after this many epochs without validation improvement; 0 disables",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="minimum validation-loss decrease counted as an improvement",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default="model.pt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.patience < 0:
        parser.error("--patience must be nonnegative")
    if args.min_delta < 0 or not math.isfinite(args.min_delta):
        parser.error("--min-delta must be finite and nonnegative")
    try:
        train(args)
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
