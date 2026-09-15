"""Train the tiny W4A6 particle/background patch classifier."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.patch_classifier import (
    PatchClassifierConfig,
    TinyQuantPatchClassifier,
    count_patch_parameters,
    patch_config_from_dict,
)
from qnn.patch_preprocess import TRANSFORMS, preprocess_patch


@dataclass(frozen=True)
class BinaryMetrics:
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    specificity: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "threshold": self.threshold,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "specificity": self.specificity,
        }


class PatchFolderDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        *,
        input_size: int,
        augment: bool,
        input_transform: str = "raw",
    ) -> None:
        self.root = root
        self.split = split
        self.input_size = input_size
        self.augment = augment
        self.input_transform = input_transform
        self.samples: list[tuple[Path, float]] = []
        for class_name, label in (("background", 0.0), ("particle", 1.0)):
            directory = root / split / class_name
            if not directory.exists():
                raise FileNotFoundError(directory)
            for path in sorted(directory.glob("*.png")):
                self.samples.append((path, label))
        if not self.samples:
            raise RuntimeError(f"No patches found for split {split}")

    @property
    def positive_count(self) -> int:
        return sum(label == 1.0 for _, label in self.samples)

    @property
    def negative_count(self) -> int:
        return sum(label == 0.0 for _, label in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: np.ndarray) -> np.ndarray:
        if random.random() < 0.50:
            image = cv2.flip(image, 1)
        if random.random() < 0.50:
            image = cv2.flip(image, 0)
        rotations = random.randrange(4)
        if rotations:
            image = np.rot90(image, rotations).copy()

        pixels = image.astype(np.float32) / 255.0
        gain = random.uniform(0.78, 1.22)
        bias = random.uniform(-0.10, 0.10)
        pixels = pixels * gain + bias
        if random.random() < 0.30:
            gamma = random.uniform(0.75, 1.35)
            pixels = np.power(np.clip(pixels, 0, 1), gamma)

        if random.random() < 0.25:
            pixels = cv2.GaussianBlur(
                pixels,
                (3, 3),
                random.uniform(0.35, 1.0),
            )
        if random.random() < 0.18:
            kernel = np.zeros((3, 3), dtype=np.float32)
            if random.random() < 0.5:
                kernel[1, :] = 1.0 / 3.0
            else:
                kernel[:, 1] = 1.0 / 3.0
            pixels = cv2.filter2D(
                pixels,
                -1,
                kernel,
                borderType=cv2.BORDER_REFLECT_101,
            )
        if random.random() < 0.35:
            noise_sigma = random.uniform(0.005, 0.045)
            noise = np.random.normal(
                0.0,
                noise_sigma,
                pixels.shape,
            ).astype(np.float32)
            pixels += noise
        return np.clip(pixels, 0, 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not read patch: {path}")
        if image.shape != (self.input_size, self.input_size):
            image = cv2.resize(
                image,
                (self.input_size, self.input_size),
                interpolation=cv2.INTER_AREA,
            )
        image = preprocess_patch(image, self.input_transform)
        if self.augment:
            pixels = self._augment(image)
        else:
            pixels = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(
            np.ascontiguousarray(pixels[None], dtype=np.float32)
        )
        return tensor, torch.tensor([label], dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny quantized binary patch classifier."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "dataset" / "microplastic_patch32_grouped_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "qnn_microplastic_patch32_w4a6_v1",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument(
        "--positive-weight",
        type=float,
        default=0.0,
        help="Positive BCE weight; values <= 0 use the train class ratio.",
    )
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--activation-bits", type=int, default=6)
    parser.add_argument("--input-bits", type=int, default=8)
    parser.add_argument("--output-bits", type=int, default=8)
    parser.add_argument("--channels", default="8,12,16")
    parser.add_argument(
        "--spatial-head",
        action="store_true",
        help="Use a learned full-spatial quantized head instead of global average pooling.",
    )
    parser.add_argument("--input-transform", choices=TRANSFORMS, default="raw")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional classifier checkpoint to continue from.",
    )
    parser.add_argument(
        "--initialize",
        type=Path,
        default=None,
        help=(
            "Optional classifier checkpoint used for model weights only; "
            "epoch and optimizer state are reset."
        ),
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> BinaryMetrics:
    predicted = probabilities >= threshold
    actual = labels >= 0.5
    true_positive = int(np.logical_and(predicted, actual).sum())
    false_positive = int(np.logical_and(predicted, ~actual).sum())
    true_negative = int(np.logical_and(~predicted, ~actual).sum())
    false_negative = int(np.logical_and(~predicted, actual).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    f1 = (
        2.0 * precision * recall / max(precision + recall, 1e-12)
    )
    accuracy = (true_positive + true_negative) / max(len(labels), 1)
    return BinaryMetrics(
        threshold=float(threshold),
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        accuracy=float(accuracy),
        specificity=float(specificity),
    )


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    minimum_recall: float,
    step: float,
) -> tuple[BinaryMetrics, str]:
    thresholds = np.arange(step, 1.0, step)
    metrics = [
        binary_metrics(labels, probabilities, float(threshold))
        for threshold in thresholds
    ]
    feasible = [
        item for item in metrics if item.recall >= minimum_recall
    ]
    if feasible:
        selected = max(
            feasible,
            key=lambda item: (
                item.f1,
                item.precision,
                item.specificity,
                item.threshold,
            ),
        )
        policy = (
            f"max validation F1 subject to recall >= {minimum_recall:.3f}"
        )
    else:
        selected = max(
            metrics,
            key=lambda item: (
                item.f1,
                item.recall,
                item.precision,
            ),
        )
        policy = (
            "recall constraint infeasible; max validation F1, then recall"
        )
    return selected, policy


def run_epoch(
    model: TinyQuantPatchClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        labels.append(targets.detach().cpu().numpy().reshape(-1))
        probabilities.append(
            torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        )
    return (
        float(np.mean(losses)),
        np.concatenate(labels),
        np.concatenate(probabilities),
    )


def save_checkpoint(
    path: Path,
    *,
    model: TinyQuantPatchClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    threshold: float,
    validation_metrics: BinaryMetrics,
    data: Path,
    threshold_policy: str,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "config": model.config.to_dict(),
            "threshold": threshold,
            "validation_metrics": validation_metrics.to_dict(),
            "threshold_policy": threshold_policy,
            "data": str(data.resolve()),
        },
        path,
    )


def save_history(
    history: list[dict[str, float | int]],
    output: Path,
) -> None:
    with (output / "history.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(epochs, [row["train_loss"] for row in history])
    axes[0, 0].plot(epochs, [row["valid_loss"] for row in history])
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend(("train", "validation"))
    axes[0, 1].plot(
        epochs,
        [row["valid_precision"] for row in history],
        label="precision",
    )
    axes[0, 1].plot(
        epochs,
        [row["valid_recall"] for row in history],
        label="recall",
    )
    axes[0, 1].set_title("Validation precision / recall")
    axes[0, 1].legend()
    axes[1, 0].plot(
        epochs,
        [row["valid_f1"] for row in history],
    )
    axes[1, 0].set_title("Validation F1")
    axes[1, 1].plot(
        epochs,
        [row["learning_rate"] for row in history],
    )
    axes[1, 1].set_title("Learning rate")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.set_xlabel("epoch")
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=170)
    plt.close(figure)


def save_confusion_matrix(
    metrics: BinaryMetrics,
    path: Path,
) -> None:
    matrix = np.array(
        [
            [metrics.true_negative, metrics.false_positive],
            [metrics.false_negative, metrics.true_positive],
        ]
    )
    figure, axis = plt.subplots(figsize=(4.7, 4.1))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=13,
            )
    axis.set_xticks((0, 1), ("background", "particle"))
    axis.set_yticks((0, 1), ("background", "particle"))
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Ground truth")
    axis.set_title(f"Test confusion matrix @ {metrics.threshold:.2f}")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def benchmark_model(
    model: TinyQuantPatchClassifier,
    device: torch.device,
    *,
    batch_size: int,
    iterations: int = 200,
) -> dict[str, float]:
    inputs = torch.rand(
        batch_size,
        1,
        model.config.input_size,
        model.config.input_size,
        device=device,
    )
    model.eval()
    with torch.inference_mode():
        for _ in range(20):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "batch_size": batch_size,
        "iterations": iterations,
        "mean_batch_ms": elapsed * 1000.0 / iterations,
        "mean_patch_ms": elapsed * 1000.0 / iterations / batch_size,
        "patches_per_second": iterations * batch_size / elapsed,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    data = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    channels = tuple(
        int(value.strip()) for value in args.channels.split(",") if value.strip()
    )
    config = PatchClassifierConfig(
        input_size=32,
        channels=channels,
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        input_bits=args.input_bits,
        output_bits=args.output_bits,
        spatial_head=args.spatial_head,
        input_transform=args.input_transform,
    )
    model = TinyQuantPatchClassifier(config).to(device)
    start_epoch = 1
    if args.resume is not None and args.initialize is not None:
        raise ValueError("Use only one of --resume and --initialize")
    if args.initialize is not None:
        checkpoint = torch.load(
            args.initialize,
            map_location=device,
            weights_only=False,
        )
        saved_config = patch_config_from_dict(checkpoint["config"])
        if saved_config != config:
            raise ValueError("Initialize checkpoint configuration does not match")
        model.load_state_dict(checkpoint["model_state"])
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        saved_config = patch_config_from_dict(checkpoint["config"])
        if saved_config != config:
            raise ValueError("Resume checkpoint configuration does not match")
        model.load_state_dict(checkpoint["model_state"])
        start_epoch = int(checkpoint["epoch"]) + 1

    datasets = {
        split: PatchFolderDataset(
            data,
            split,
            input_size=config.input_size,
            augment=split == "train",
            input_transform=config.input_transform,
        )
        for split in ("train", "valid", "test")
    }
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_worker,
            generator=generator,
        )
        for split, dataset in datasets.items()
    }

    automatic_positive_weight = datasets["train"].negative_count / max(
        datasets["train"].positive_count,
        1,
    )
    positive_weight = (
        args.positive_weight
        if args.positive_weight > 0
        else automatic_positive_weight
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )
    if args.resume is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    total_parameters, trainable_parameters = count_patch_parameters(model)
    print(
        f"Device={device}; parameters={total_parameters:,}; "
        f"train={len(datasets['train'])}, valid={len(datasets['valid'])}, "
        f"test={len(datasets['test'])}; pos_weight={positive_weight:.3f}"
    )

    history: list[dict[str, float | int]] = []
    best_f1 = -math.inf
    best_epoch = -1
    stale_epochs = 0
    training_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_labels, train_probabilities = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
        )
        with torch.inference_mode():
            valid_loss, valid_labels, valid_probabilities = run_epoch(
                model,
                loaders["valid"],
                criterion,
                device,
                optimizer=None,
            )
        valid_metrics, threshold_policy = select_threshold(
            valid_labels,
            valid_probabilities,
            minimum_recall=args.min_recall,
            step=args.threshold_step,
        )
        train_metrics = binary_metrics(
            train_labels,
            train_probabilities,
            valid_metrics.threshold,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_precision": train_metrics.precision,
                "train_recall": train_metrics.recall,
                "train_f1": train_metrics.f1,
                "valid_precision": valid_metrics.precision,
                "valid_recall": valid_metrics.recall,
                "valid_f1": valid_metrics.f1,
                "valid_accuracy": valid_metrics.accuracy,
                "threshold": valid_metrics.threshold,
                "learning_rate": learning_rate,
            }
        )
        save_checkpoint(
            output / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            threshold=valid_metrics.threshold,
            validation_metrics=valid_metrics,
            data=data,
            threshold_policy=threshold_policy,
        )
        improved = valid_metrics.f1 > best_f1 + 1e-5
        if improved:
            best_f1 = valid_metrics.f1
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                output / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                threshold=valid_metrics.threshold,
                validation_metrics=valid_metrics,
                data=data,
                threshold_policy=threshold_policy,
            )
        else:
            stale_epochs += 1
        scheduler.step(valid_metrics.f1)
        print(
            f"{epoch:03d}/{args.epochs} "
            f"loss={train_loss:.4f}/{valid_loss:.4f} "
            f"P={valid_metrics.precision:.4f} "
            f"R={valid_metrics.recall:.4f} "
            f"F1={valid_metrics.f1:.4f} "
            f"thr={valid_metrics.threshold:.2f}"
        )
        if stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} stale epochs")
            break

    training_seconds = time.perf_counter() - training_start
    save_history(history, output)
    best_checkpoint = torch.load(
        output / "best.pt",
        map_location=device,
        weights_only=False,
    )
    best_model = TinyQuantPatchClassifier(
        patch_config_from_dict(best_checkpoint["config"])
    ).to(device)
    best_model.load_state_dict(best_checkpoint["model_state"])
    with torch.inference_mode():
        test_loss, test_labels, test_probabilities = run_epoch(
            best_model,
            loaders["test"],
            criterion,
            device,
            optimizer=None,
        )
    selected_threshold = float(best_checkpoint["threshold"])
    test_metrics = binary_metrics(
        test_labels,
        test_probabilities,
        selected_threshold,
    )
    save_confusion_matrix(
        test_metrics,
        output / "test_confusion_matrix.png",
    )
    speed = benchmark_model(
        best_model,
        device,
        batch_size=min(args.batch_size, 64),
    )
    summary = {
        "model": "TinyQuantPatchClassifier",
        "purpose": (
            "candidate-gated particle/background classification inside one "
            "tracked droplet"
        ),
        "data": str(data),
        "output": str(output),
        "device": str(device),
        "config": best_model.config.to_dict(),
        "parameters": {
            "total": total_parameters,
            "trainable": trainable_parameters,
        },
        "training": {
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "seconds": training_seconds,
            "positive_weight": positive_weight,
            "threshold_policy": best_checkpoint["threshold_policy"],
        },
        "validation": best_checkpoint["validation_metrics"],
        "test": {
            "loss": test_loss,
            **test_metrics.to_dict(),
        },
        "runtime_benchmark": speed,
        "warning": (
            "Patch-level metrics use bootstrap labels from the existing "
            "class-0 boxes. End-to-end video precision/recall still requires "
            "a manually reviewed video test set."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary["test"], indent=2))
    print(json.dumps(speed, indent=2))
    print(f"Best checkpoint: {output / 'best.pt'}")


if __name__ == "__main__":
    main()
