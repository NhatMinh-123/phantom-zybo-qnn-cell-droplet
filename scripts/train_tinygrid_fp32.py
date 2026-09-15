#!/usr/bin/env python3
"""Train and evaluate the FP32 TinyGridNet occupancy model."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tinygrid_qnn.config import TinyGridConfig
from tinygrid_qnn.data import FEATURE_MODES, FeatureGridDataset, occupancy_counts
from tinygrid_qnn.metrics import choose_thresholds, multiclass_report
from tinygrid_qnn.model import TinyGridNetFP32, count_parameters


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "dataset" / "cell_droplet_tinygrid_feature_v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-mode", choices=tuple(FEATURE_MODES), default="all")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--dice-weight", type=float, default=0.25)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loss_value(
    logits: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
    dice_weight: float,
) -> torch.Tensor:
    bce = criterion(logits, target)
    probabilities = torch.sigmoid(logits)
    numerator = 2.0 * (probabilities * target).sum(dim=(0, 2, 3)) + 1.0
    denominator = probabilities.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3)) + 1.0
    dice_loss = 1.0 - (numerator / denominator).mean()
    return bce + dice_weight * dice_loss


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    dice_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    for features, target in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = loss_value(logits, target, criterion, dice_weight)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * features.shape[0]
        sample_count += features.shape[0]
        all_targets.append(target.detach().cpu().numpy())
        all_probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    return (
        total_loss / max(1, sample_count),
        np.concatenate(all_targets),
        np.concatenate(all_probabilities),
    )


def report_at_half(targets: np.ndarray, probabilities: np.ndarray, config: TinyGridConfig) -> dict[str, object]:
    return multiclass_report(
        targets,
        probabilities,
        np.full(config.output_channels, 0.5, dtype=np.float32),
        config.class_names,
    )


def plot_history(history: list[dict[str, float]], output: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["valid_loss"] for row in history], label="valid")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(epochs, [row["valid_macro_f1"] for row in history], color="#168f55")
    axes[1].set_title("Validation macro F1 at 0.50")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    config = TinyGridConfig()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    datasets = {
        split: FeatureGridDataset(
            args.data.resolve(),
            split,
            feature_mode=args.feature_mode,
            augment=args.augment and split == "train",
        )
        for split in ("train", "valid", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }
    positive, total = occupancy_counts(datasets["train"])
    negative = total - positive
    positive_weight = np.clip(negative / np.maximum(positive, 1), 1.0, 20.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device).view(2, 1, 1)
    )
    model = TinyGridNetFP32(
        config,
        input_channels=len(FEATURE_MODES[args.feature_mode]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6, min_lr=1e-6
    )
    total_parameters, trainable_parameters = count_parameters(model)
    print(
        f"device={device} mode={args.feature_mode} samples="
        f"{len(datasets['train'])}/{len(datasets['valid'])}/{len(datasets['test'])} "
        f"parameters={total_parameters} pos_weight={positive_weight.tolist()}",
        flush=True,
    )

    history: list[dict[str, float]] = []
    best_f1 = -1.0
    stale_epochs = 0
    started = time.perf_counter()
    best_path = output / "best_fp32.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss, train_targets, train_probabilities = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            args.dice_weight,
            optimizer,
        )
        with torch.no_grad():
            valid_loss, valid_targets, valid_probabilities = run_epoch(
                model,
                loaders["valid"],
                criterion,
                device,
                args.dice_weight,
            )
        train_report = report_at_half(train_targets, train_probabilities, config)
        valid_report = report_at_half(valid_targets, valid_probabilities, config)
        valid_f1 = float(valid_report["macro_f1"])
        scheduler.step(valid_f1)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "train_macro_f1": float(train_report["macro_f1"]),
            "valid_macro_f1": valid_f1,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if valid_f1 > best_f1 + 1e-5:
            best_f1 = valid_f1
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config.to_dict(),
                    "feature_mode": args.feature_mode,
                    "epoch": epoch,
                    "validation_macro_f1_at_0_5": valid_f1,
                },
                best_path,
            )
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch={epoch:03d} loss={train_loss:.4f}/{valid_loss:.4f} "
                f"macro_f1={float(train_report['macro_f1']):.4f}/{valid_f1:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )
        if stale_epochs >= args.patience:
            print(f"early_stop epoch={epoch} best_f1={best_f1:.4f}", flush=True)
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        valid_loss, valid_targets, valid_probabilities = run_epoch(
            model, loaders["valid"], criterion, device, args.dice_weight
        )
        test_loss, test_targets, test_probabilities = run_epoch(
            model, loaders["test"], criterion, device, args.dice_weight
        )
    thresholds = choose_thresholds(valid_targets, valid_probabilities)
    validation_report = multiclass_report(
        valid_targets, valid_probabilities, thresholds, config.class_names
    )
    test_report = multiclass_report(test_targets, test_probabilities, thresholds, config.class_names)
    checkpoint["thresholds"] = thresholds.tolist()
    torch.save(checkpoint, best_path)

    with (output / "training_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, output / "training_curves.png")
    result = {
        "model": "TinyGridNetFP32 two-channel occupancy",
        "feature_mode": args.feature_mode,
        "channel_indices": list(FEATURE_MODES[args.feature_mode]),
        "parameters": {"total": total_parameters, "trainable": trainable_parameters},
        "samples": {split: len(dataset) for split, dataset in datasets.items()},
        "train_positive_cells": {
            name: int(positive[index]) for index, name in enumerate(config.class_names)
        },
        "positive_weight": {
            name: float(positive_weight[index]) for index, name in enumerate(config.class_names)
        },
        "best_epoch": int(checkpoint["epoch"]),
        "training_seconds": time.perf_counter() - started,
        "validation_loss": valid_loss,
        "test_loss": test_loss,
        "validation": validation_report,
        "test": test_report,
        "config": config.to_dict(),
    }
    (output / "metrics_fp32.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
