from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont

from qnn.dataset import YoloDetectionDataset
from qnn.detection import Detection, count_matches, decode_predictions
from qnn.evaluate_qat import collect_predictions, evaluate_cached
from qnn.fpga_io import decode_accumulator, load_manifest
from qnn.model import TinyQuantDetector, config_from_dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_COLORS = ((232, 55, 76), (0, 128, 225))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path
    evaluation: Path
    postprocess: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit cell/droplet QNN failures image by image"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "cell_droplet_roi384_grouped",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "qnn_detection_quality_audit",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--worst-count", type=int, default=6)
    return parser.parse_args()


def default_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            "W4A4 (FPGA hien tai)",
            PROJECT_ROOT / "models" / "qnn_cell_droplet_v2_w4a4_grouped" / "best.pt",
            PROJECT_ROOT
            / "reports"
            / "qnn_cell_droplet_v2_w4a4_grouped"
            / "evaluation.json",
            PROJECT_ROOT
            / "reports"
            / "qnn_droplet_postprocess"
            / "w4a4"
            / "postprocess_config.json",
        ),
        ModelSpec(
            "W4A8 (ung vien)",
            PROJECT_ROOT / "models" / "qnn_cell_droplet_v2_w4a8_grouped" / "best.pt",
            PROJECT_ROOT
            / "reports"
            / "qnn_cell_droplet_v2_w4a8_grouped"
            / "evaluation.json",
            PROJECT_ROOT
            / "reports"
            / "qnn_droplet_postprocess"
            / "w4a8"
            / "postprocess_config.json",
        ),
        ModelSpec(
            "W4A6 (ung vien vua FPGA)",
            PROJECT_ROOT / "models" / "qnn_cell_droplet_v2_w4a6_grouped" / "best.pt",
            PROJECT_ROOT
            / "reports"
            / "qnn_cell_droplet_v2_w4a6_grouped"
            / "evaluation.json",
            PROJECT_ROOT
            / "reports"
            / "qnn_droplet_postprocess"
            / "w4a6"
            / "postprocess_config.json",
        ),
        ModelSpec(
            "W4A4 192x192 (cell tot)",
            PROJECT_ROOT
            / "models"
            / "qnn_cell_droplet_v2_w4a4_square192_grouped"
            / "best.pt",
            PROJECT_ROOT
            / "reports"
            / "qnn_cell_droplet_v2_w4a4_square192_grouped"
            / "evaluation.json",
            PROJECT_ROOT
            / "reports"
            / "qnn_droplet_postprocess"
            / "w4a4_square192"
            / "postprocess_config.json",
        ),
        ModelSpec(
            "W4A6 192x192 (giot uu tien)",
            PROJECT_ROOT
            / "models"
            / "qnn_cell_droplet_v2_w4a6_square192_grouped"
            / "best.pt",
            PROJECT_ROOT
            / "reports"
            / "qnn_cell_droplet_v2_w4a6_square192_grouped"
            / "evaluation.json",
            PROJECT_ROOT
            / "reports"
            / "qnn_droplet_postprocess"
            / "w4a6_square192"
            / "postprocess_config.json",
        ),
        ModelSpec(
            "W8A8 (moc chat luong)",
            PROJECT_ROOT / "models" / "qnn_cell_droplet_v2_w8a8_grouped" / "best.pt",
            PROJECT_ROOT
            / "reports"
            / "qnn_cell_droplet_v2_w8a8_grouped"
            / "evaluation.json",
            PROJECT_ROOT
            / "reports"
            / "qnn_droplet_postprocess"
            / "w8a8"
            / "postprocess_config.json",
        ),
    ]


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def read_postprocess(
    spec: ModelSpec, class_names: tuple[str, ...]
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[dict[str, float] | None, ...] | None,
    tuple[dict[str, float] | None, ...] | None,
]:
    if spec.postprocess.is_file():
        payload = json.loads(spec.postprocess.read_text(encoding="utf-8"))
        selected = payload["selected"]
        thresholds = tuple(
            float(selected["confidence_thresholds"][name]) for name in class_names
        )
        nms_iou = tuple(float(selected["nms_iou"][name]) for name in class_names)
        selected_constraints = selected.get("box_constraints")
        box_constraints = None
        if selected_constraints is not None:
            box_constraints = tuple(selected_constraints.get(name) for name in class_names)
        selected_calibration = selected.get("box_calibration")
        box_calibration = None
        if selected_calibration is not None:
            box_calibration = tuple(
                selected_calibration.get(name) for name in class_names
            )
        return thresholds, nms_iou, box_constraints, box_calibration
    payload = json.loads(spec.evaluation.read_text(encoding="utf-8"))
    selected = payload["validation"]["thresholds"]
    return (
        tuple(float(selected[name]) for name in class_names),
        (0.45,) * len(class_names),
        None,
        None,
    )


def target_counts(target: torch.Tensor, num_classes: int) -> list[int]:
    return [int((target[:, 0] == class_id).sum().item()) for class_id in range(num_classes)]


def detection_counts(detections: list[Detection], num_classes: int) -> list[int]:
    return [sum(item.class_id == class_id for item in detections) for class_id in range(num_classes)]


def audit_model(
    spec: ModelSpec,
    data_root: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[list[Detection]],
    YoloDetectionDataset,
]:
    checkpoint = torch.load(spec.checkpoint, map_location=device, weights_only=False)
    config = config_from_dict(checkpoint["config"])
    dataset = YoloDetectionDataset(
        data_root,
        "test",
        input_size=(config.image_width, config.image_height),
        num_classes=config.num_classes,
    )
    model = TinyQuantDetector(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    cached = collect_predictions(
        model, dataset, device=device, batch_size=batch_size
    )
    thresholds, nms_iou, box_constraints, box_calibration = read_postprocess(
        spec, config.class_names
    )
    aggregate = evaluate_cached(
        cached,
        config=config,
        thresholds=thresholds,
        nms_iou=nms_iou,
        box_constraints=box_constraints,
        box_calibration=box_calibration,
    )

    all_detections: list[list[Detection]] = []
    all_targets: list[torch.Tensor] = []
    for predictions, targets in cached:
        all_detections.extend(
            decode_predictions(
                predictions,
                confidence_threshold=thresholds,
                nms_iou=nms_iou,
                box_constraints=box_constraints,
                box_calibration=box_calibration,
                anchors=config.anchors,
                slots_per_class=config.slots_per_class,
            )
        )
        all_targets.extend(targets)

    rows: list[dict[str, object]] = []
    status_counts = {"correct": 0, "miss": 0, "false_positive": 0, "mixed": 0}
    for image_path, detections, target in zip(dataset.images, all_detections, all_targets):
        tp, fp, fn = count_matches(
            [detections], [target], num_classes=config.num_classes
        )
        gt = target_counts(target, config.num_classes)
        predicted = detection_counts(detections, config.num_classes)
        droplet_fp = fp[1]
        droplet_fn = fn[1]
        if droplet_fp and droplet_fn:
            status = "mixed"
        elif droplet_fn:
            status = "miss"
        elif droplet_fp:
            status = "false_positive"
        else:
            status = "correct"
        status_counts[status] += 1
        rows.append(
            {
                "model": spec.name,
                "image": image_path.name,
                "gt_cell": gt[0],
                "pred_cell": predicted[0],
                "tp_cell": tp[0],
                "fp_cell": fp[0],
                "fn_cell": fn[0],
                "gt_droplet": gt[1],
                "pred_droplet": predicted[1],
                "tp_droplet": tp[1],
                "fp_droplet": droplet_fp,
                "fn_droplet": droplet_fn,
                "droplet_status": status,
            }
        )

    aggregate["name"] = spec.name
    aggregate["checkpoint"] = str(spec.checkpoint)
    aggregate["nms_iou"] = dict(zip(config.class_names, nms_iou))
    aggregate["box_constraints"] = box_constraints
    aggregate["image_status"] = status_counts
    aggregate["input_size"] = [config.image_width, config.image_height]
    return aggregate, rows, all_detections, dataset


def yolo_target_to_detections(target: torch.Tensor) -> list[Detection]:
    result: list[Detection] = []
    for row in target:
        class_id = int(row[0].item())
        center_x, center_y, width, height = (float(value) for value in row[1:])
        result.append(
            Detection(
                class_id=class_id,
                confidence=1.0,
                box=(
                    center_x - width / 2,
                    center_y - height / 2,
                    center_x + width / 2,
                    center_y + height / 2,
                ),
            )
        )
    return result


def draw_detections(
    image_path: Path,
    detections: list[Detection],
    *,
    title: str,
    class_names: tuple[str, ...],
    show_confidence: bool,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        box = (
            round(x1 * width),
            round(y1 * height),
            round(x2 * width),
            round(y2 * height),
        )
        color = CLASS_COLORS[detection.class_id]
        draw.rectangle(box, outline=color, width=3)
        label = class_names[detection.class_id]
        if show_confidence:
            label += f" {detection.confidence:.2f}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0] + 6
        text_height = text_box[3] - text_box[1] + 4
        label_y = max(0, box[1] - text_height)
        draw.rectangle(
            (box[0], label_y, box[0] + text_width, label_y + text_height),
            fill=color,
        )
        draw.text((box[0] + 3, label_y + 2), label, fill="white", font=font)

    title_height = 28
    canvas = Image.new("RGB", (width, height + title_height), "white")
    canvas.paste(image, (0, title_height))
    title_draw = ImageDraw.Draw(canvas)
    title_draw.text((8, 8), title, fill="black", font=font)
    return canvas


def make_worst_case_sheet(
    output: Path,
    dataset: YoloDetectionDataset,
    model_specs: list[ModelSpec],
    per_model_rows: dict[str, list[dict[str, object]]],
    per_model_detections: dict[str, list[list[Detection]]],
    *,
    count: int,
) -> list[str]:
    current_rows = per_model_rows[model_specs[0].name]
    ranked = sorted(
        range(len(current_rows)),
        key=lambda index: (
            10 * int(current_rows[index]["fn_droplet"])
            + 7 * int(current_rows[index]["fp_droplet"])
            + 2 * int(current_rows[index]["fn_cell"])
            + int(current_rows[index]["fp_cell"]),
            int(current_rows[index]["gt_droplet"]),
        ),
        reverse=True,
    )[:count]

    rows: list[Image.Image] = []
    for index in ranked:
        _, target = dataset[index]
        panels = [
            draw_detections(
                dataset.images[index],
                yolo_target_to_detections(target),
                title="Nhan that",
                class_names=("cell", "droplet"),
                show_confidence=False,
            )
        ]
        for spec in model_specs:
            panels.append(
                draw_detections(
                    dataset.images[index],
                    per_model_detections[spec.name][index],
                    title=spec.name,
                    class_names=("cell", "droplet"),
                    show_confidence=True,
                )
            )
        row = Image.new(
            "RGB",
            (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
            "white",
        )
        offset = 0
        for panel in panels:
            row.paste(panel, (offset, 0))
            offset += panel.width
        rows.append(row)

    sheet = Image.new(
        "RGB",
        (max(row.width for row in rows), sum(row.height for row in rows)),
        "white",
    )
    offset = 0
    for row in rows:
        sheet.paste(row, (0, offset))
        offset += row.height
    sheet.save(output, quality=92)
    return [dataset.images[index].name for index in ranked]


def make_metric_plot(output: Path, summaries: list[dict[str, object]]) -> None:
    names = [str(item["name"]).split(" (")[0] for item in summaries]
    droplet = [item["classes"]["droplet"] for item in summaries]
    values = {
        "Precision": [100 * float(item["precision"]) for item in droplet],
        "Recall": [100 * float(item["recall"]) for item in droplet],
        "F1": [100 * float(item["f1"]) for item in droplet],
    }
    positions = list(range(len(names)))
    bar_width = 0.24
    fig, axis = plt.subplots(figsize=(9, 5.2))
    for metric_index, (metric, metric_values) in enumerate(values.items()):
        offsets = [value + (metric_index - 1) * bar_width for value in positions]
        bars = axis.bar(offsets, metric_values, bar_width, label=metric)
        axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    axis.set_xticks(positions, names)
    axis.set_ylabel("Percent (%)")
    axis.set_ylim(0, 105)
    axis.set_title("Chat luong phat hien droplet tren tap test grouped")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_hardware_sheet(output: Path, reports_dir: Path) -> list[dict[str, object]]:
    rows: list[Image.Image] = []
    summaries: list[dict[str, object]] = []
    manifest = load_manifest()
    for report_path in sorted(reports_dir.glob("frame_?????_report.json")):
        frame_id = int(report_path.stem.split("_")[1])
        if frame_id < 4:
            continue
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        image_path = Path(payload["image"])
        label_path = image_path.parents[1] / "labels" / f"{image_path.stem}.txt"
        target_rows = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields:
                target_rows.append([float(value) for value in fields])
        target = torch.tensor(target_rows, dtype=torch.float32)
        accumulator_path = report_path.with_name(
            report_path.name.replace("_report.json", "_accumulator.npy")
        )
        import numpy as np

        detections = decode_accumulator(np.load(accumulator_path), manifest)[0]
        tp, fp, fn = count_matches([detections], [target], num_classes=2)
        summaries.append(
            {
                "frame_id": frame_id,
                "image": image_path.name,
                "cell": {"tp": tp[0], "fp": fp[0], "fn": fn[0]},
                "droplet": {"tp": tp[1], "fp": fp[1], "fn": fn[1]},
                "hardware_exact": bool(payload["hardware_vs_checkpoint"]["exact"]),
            }
        )
        panels = [
            draw_detections(
                image_path,
                yolo_target_to_detections(target),
                title=f"Frame {frame_id}: nhan that",
                class_names=("cell", "droplet"),
                show_confidence=False,
            ),
            draw_detections(
                image_path,
                detections,
                title=f"Frame {frame_id}: FPGA W4A4 + NMS moi",
                class_names=("cell", "droplet"),
                show_confidence=True,
            ),
        ]
        row = Image.new("RGB", (panels[0].width + panels[1].width, panels[0].height), "white")
        row.paste(panels[0], (0, 0))
        row.paste(panels[1], (panels[0].width, 0))
        rows.append(row)
    if rows:
        sheet = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "white")
        offset = 0
        for row in rows:
            sheet.paste(row, (0, offset))
            offset += row.height
        sheet.save(output, quality=92)
    return summaries


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    specs = default_models()
    for spec in specs:
        if not spec.checkpoint.is_file() or not spec.evaluation.is_file():
            raise FileNotFoundError(f"Missing artifact for {spec.name}")

    args.output.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    rows_by_model: dict[str, list[dict[str, object]]] = {}
    detections_by_model: dict[str, list[list[Detection]]] = {}
    dataset: YoloDetectionDataset | None = None
    for spec in specs:
        summary, rows, detections, model_dataset = audit_model(
            spec, args.data, device=device, batch_size=args.batch_size
        )
        if dataset is None:
            dataset = model_dataset
        else:
            expected_images = [path.name for path in dataset.images]
            actual_images = [path.name for path in model_dataset.images]
            if actual_images != expected_images:
                raise RuntimeError(
                    f"Test image order differs for {spec.name}; comparison is unsafe"
                )
        summaries.append(summary)
        all_rows.extend(rows)
        rows_by_model[spec.name] = rows
        detections_by_model[spec.name] = detections

    if dataset is None:
        raise RuntimeError("No models were audited")

    with (args.output / "per_image_cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    worst_images = make_worst_case_sheet(
        args.output / "worst_cases_comparison.jpg",
        dataset,
        specs,
        rows_by_model,
        detections_by_model,
        count=args.worst_count,
    )
    make_metric_plot(args.output / "droplet_model_comparison.png", summaries)
    hardware = make_hardware_sheet(
        args.output / "current_fpga_cases.jpg",
        PROJECT_ROOT / "reports" / "fpga_uart_hardware",
    )
    payload = {
        "data": str(args.data),
        "split": "test",
        "images": len(dataset),
        "device": str(device),
        "models": summaries,
        "worst_images_for_current_fpga_model": worst_images,
        "current_hardware_cases": hardware,
        "important_note": (
            "frame_00001 to frame_00003 were produced before the threshold-ROM path fix; "
            "only frame_00004 and later are current validated hardware results"
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
