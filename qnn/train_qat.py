from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from qnn.dataset import YoloDetectionDataset, detection_collate
from qnn.detection import count_matches, decode_predictions, detector_loss
from qnn.model import DetectorConfig, TinyQuantDetector, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the W4A4 detector with Brevitas QAT")
    parser.add_argument("--data", type=Path, default=Path("dataset/cell_droplet_yolo_grouped"))
    parser.add_argument("--output", type=Path, default=Path("models/qnn_cell_droplet"))
    parser.add_argument("--report", type=Path, default=Path("reports/qnn_cell_droplet"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--box-weight", type=float, default=5.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--iou-loss-weight",
        type=float,
        default=0.0,
        help="Aligned IoU loss added to the offset/size box loss",
    )
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--activation-bits", type=int, default=4)
    parser.add_argument("--output-bits", type=int, default=8)
    parser.add_argument("--input-width", type=int, default=192)
    parser.add_argument("--input-height", type=int, default=144)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--downsample-width", type=int, default=0)
    parser.add_argument("--downsample-height", type=int, default=0)
    parser.add_argument("--channels", type=int, nargs=4, default=(8, 12, 16, 16))
    parser.add_argument(
        "--slots-per-class",
        type=int,
        nargs="+",
        default=(2, 1),
        help="Output slots for each class, e.g. cell=2 droplet=1",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--metrics-every", type=int, default=1)
    parser.add_argument(
        "--horizontal-shift",
        type=float,
        default=0.0,
        help="Maximum random horizontal translation as a fraction of image width",
    )
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-valid-batches", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def run_epoch(
    model: TinyQuantDetector,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int,
    compute_detection_metrics: bool = False,
    box_weight: float = 5.0,
    focal_gamma: float = 2.0,
    iou_loss_weight: float = 0.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "objectness": 0.0, "box": 0.0, "batches": 0.0}
    true_positive = [0] * model.config.num_classes
    false_positive = [0] * model.config.num_classes
    false_negative = [0] * model.config.num_classes

    for batch_index, (images, targets) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            predictions = model(images)
            losses = detector_loss(
                predictions,
                targets,
                num_classes=model.config.num_classes,
                anchors=model.config.anchors,
                slots_per_class=model.config.slots_per_class,
                box_weight=box_weight,
                focal_gamma=focal_gamma,
                iou_loss_weight=iou_loss_weight,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        totals["loss"] += float(losses.total.detach().item())
        totals["objectness"] += float(losses.objectness.detach().item())
        totals["box"] += float(losses.box.detach().item())
        totals["batches"] += 1

        if not training and compute_detection_metrics:
            decoded = decode_predictions(
                predictions.detach(),
                confidence_threshold=0.25,
                anchors=model.config.anchors,
                slots_per_class=model.config.slots_per_class,
            )
            tp, fp, fn = count_matches(
                decoded,
                targets,
                num_classes=model.config.num_classes,
            )
            true_positive = [left + right for left, right in zip(true_positive, tp)]
            false_positive = [left + right for left, right in zip(false_positive, fp)]
            false_negative = [left + right for left, right in zip(false_negative, fn)]

    batches = max(totals.pop("batches"), 1.0)
    metrics = {name: value / batches for name, value in totals.items()}
    if not training and compute_detection_metrics:
        tp_sum = sum(true_positive)
        fp_sum = sum(false_positive)
        fn_sum = sum(false_negative)
        precision = tp_sum / max(tp_sum + fp_sum, 1)
        recall = tp_sum / max(tp_sum + fn_sum, 1)
        metrics.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / max(precision + recall, 1e-9),
            }
        )
        for class_id, class_name in enumerate(model.config.class_names):
            class_precision = true_positive[class_id] / max(
                true_positive[class_id] + false_positive[class_id], 1
            )
            class_recall = true_positive[class_id] / max(
                true_positive[class_id] + false_negative[class_id], 1
            )
            metrics[f"{class_name}_precision"] = class_precision
            metrics[f"{class_name}_recall"] = class_recall
    elif not training:
        metrics.update({"precision": float("nan"), "recall": float("nan"), "f1": float("nan")})
        for class_name in model.config.class_names:
            metrics[f"{class_name}_precision"] = float("nan")
            metrics[f"{class_name}_recall"] = float("nan")
    return metrics


def save_checkpoint(
    path: Path,
    model: TinyQuantDetector,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    config = DetectorConfig(
        input_width=args.input_width,
        input_height=args.input_height,
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        output_bits=args.output_bits,
        downsample=args.downsample,
        downsample_width=args.downsample_width,
        downsample_height=args.downsample_height,
        channels=tuple(args.channels),
        slots_per_class=tuple(args.slots_per_class),
        anchors=((0.049, 0.052), (0.347, 0.366)),
    )
    model = TinyQuantDetector(config).to(device)
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(
                "Initialization checkpoint is not architecture-compatible: "
                f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
            )
        print(f"Initialized weights from {args.init_checkpoint}")
    total_parameters, trainable_parameters = count_parameters(model)

    train_dataset = YoloDetectionDataset(
        args.data,
        "train",
        input_size=(config.image_width, config.image_height),
        num_classes=config.num_classes,
        augment=True,
        horizontal_shift=args.horizontal_shift,
    )
    valid_dataset = YoloDetectionDataset(
        args.data,
        "valid",
        input_size=(config.image_width, config.image_height),
        num_classes=config.num_classes,
        augment=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=detection_collate,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=detection_collate,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    print(
        f"device={device} train={len(train_dataset)} valid={len(valid_dataset)} "
        f"parameters={total_parameters} trainable={trainable_parameters} "
        f"grid={config.grid_width}x{config.grid_height} "
        f"slots={config.slots_per_class} W{config.weight_bits}A{config.activation_bits}"
        f"O{config.output_bits}"
    )
    rows: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            box_weight=args.box_weight,
            focal_gamma=args.focal_gamma,
            iou_loss_weight=args.iou_loss_weight,
        )
        valid_metrics = run_epoch(
            model,
            valid_loader,
            device,
            optimizer=None,
            max_batches=args.max_valid_batches,
            compute_detection_metrics=(epoch % args.metrics_every == 0 or epoch == args.epochs),
            box_weight=args.box_weight,
            focal_gamma=args.focal_gamma,
            iou_loss_weight=args.iou_loss_weight,
        )
        scheduler.step(valid_metrics["loss"])
        row: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
        }
        rows.append(row)
        save_checkpoint(args.output / "last.pt", model, optimizer, epoch, valid_metrics)
        valid_loss = float(valid_metrics["loss"])
        valid_f1 = float(valid_metrics["f1"])
        if valid_loss < best_loss:
            best_loss = valid_loss
            epochs_without_improvement = 0
            save_checkpoint(args.output / "best_loss.pt", model, optimizer, epoch, valid_metrics)
            save_checkpoint(args.output / "best.pt", model, optimizer, epoch, valid_metrics)
        else:
            epochs_without_improvement += 1
        if np.isfinite(valid_f1) and valid_f1 > best_f1:
            best_f1 = valid_f1
            save_checkpoint(args.output / "best_fixed_f1.pt", model, optimizer, epoch, valid_metrics)

        metric_text = (
            f"P={valid_metrics['precision']:.3f} R={valid_metrics['recall']:.3f} "
            f"F1={valid_metrics['f1']:.3f}"
            if np.isfinite(valid_metrics["f1"])
            else "metrics=skipped"
        )
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} {metric_text}"
        )
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs")
            break

    best_checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    final_metrics = run_epoch(
        model,
        valid_loader,
        device,
        optimizer=None,
        max_batches=args.max_valid_batches,
        compute_detection_metrics=True,
        box_weight=args.box_weight,
        focal_gamma=args.focal_gamma,
        iou_loss_weight=args.iou_loss_weight,
    )

    fieldnames = list(rows[0])
    with (args.report / "training_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "best_valid_loss": best_loss,
        "best_valid_f1": best_f1,
        "checkpoint_selection": "validation_loss; thresholds calibrated after training",
        "best_checkpoint_epoch": int(best_checkpoint["epoch"]),
        "best_checkpoint_validation": final_metrics,
        "epochs_completed": len(rows),
        "parameters": total_parameters,
        "training_augmentation": {
            "horizontal_shift": args.horizontal_shift,
        },
        "loss": {
            "box_weight": args.box_weight,
            "focal_gamma": args.focal_gamma,
            "iou_loss_weight": args.iou_loss_weight,
        },
        "config": config.to_dict(),
    }
    (args.report / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Best checkpoint: {args.output / 'best.pt'}")


if __name__ == "__main__":
    main()
