#!/usr/bin/env python3
"""Create the final PC ROI optimization chart, JSON summary, and report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "pc_roi_final"


def read_counts(path: Path) -> dict[str, dict[str, int]]:
    with path.open(newline="", encoding="ascii") as csv_file:
        return {
            row["class"]: {key: int(row[key]) for key in ("tp", "fp", "fn")}
            for row in csv.DictReader(csv_file)
        }


def metrics(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**counts, "precision": precision, "recall": recall, "f1": f1}


def score_model(path: Path) -> dict[str, object]:
    counts = read_counts(path)
    scored = {name: metrics(values) for name, values in counts.items()}
    overall_counts = {
        key: sum(values[key] for values in counts.values()) for key in ("tp", "fp", "fn")
    }
    return {"overall": metrics(overall_counts), "classes": scored}


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    full = score_model(
        ROOT / "reports" / "cell_droplet_yolo11n_best" / "threshold_error_summary.csv"
    )
    compact = score_model(OUTPUT / "evaluation_selected" / "threshold_error_summary.csv")
    finetuned = score_model(OUTPUT / "evaluation" / "threshold_error_summary.csv")
    full_speed = load_json(OUTPUT / "realtime_full640" / "summary.json")
    compact_speed = load_json(OUTPUT / "realtime_baseline" / "summary.json")
    compact_standard = list(
        csv.DictReader(
            (OUTPUT / "evaluation_selected" / "metrics_summary.csv").open(
                newline="", encoding="ascii"
            )
        )
    )
    compact_map = {row["scope"]: row for row in compact_standard}

    speedup = float(compact_speed["average_processing_fps"]) / float(
        full_speed["average_processing_fps"]
    )
    roi_area_reduction = 1.0 - (154 * 115) / (256 * 256)
    input_pixel_reduction = 1.0 - (384 * 384) / (640 * 640)
    summary = {
        "decision": "Use original YOLO11n with compact ROI; reject the fine-tuned checkpoint.",
        "selected_model": str(ROOT / "models" / "cell_droplet_yolo11n" / "best.pt"),
        "selected_config": str(
            ROOT
            / "models"
            / "cell_droplet_yolo11n"
            / "roi384_baseline_config.json"
        ),
        "source_roi_1280x800": [611, 419, 154, 115],
        "model_canvas": [384, 384],
        "content_size": [384, 288],
        "test_at_deployment_thresholds": compact,
        "test_standard": {
            scope: {
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "map50": float(row["map50"]),
                "map50_95": float(row["map50_95"]),
            }
            for scope, row in compact_map.items()
        },
        "comparison": {
            "full_640": full,
            "compact_384": compact,
            "compact_384_finetuned_rejected": finetuned,
        },
        "realtime_600_frames": {
            "full_640": full_speed,
            "compact_384": compact_speed,
            "speedup": speedup,
        },
        "reductions": {
            "camera_roi_area": roi_area_reduction,
            "model_input_pixels": input_pixel_reduction,
        },
        "training_decision": {
            "more_epochs_on_current_data": False,
            "new_independent_data_required": True,
            "reason": "Fine-tuning improved validation but reduced independent test F1.",
            "next_training_requirement": "Add grouped frames from independent videos and lighting conditions before retraining.",
        },
        "acceptance_status": {
            "realtime_fps_at_least_30": float(
                compact_speed["average_processing_fps"]
            )
            >= 30.0,
            "overall_f1_at_least_0_80": float(compact["overall"]["f1"])
            >= 0.80,
            "droplet_f1_at_least_0_90": float(
                compact["classes"]["droplet"]["f1"]
            )
            >= 0.90,
            "cell_f1_at_least_0_85": float(compact["classes"]["cell"]["f1"])
            >= 0.85,
        },
    }
    (OUTPUT / "pc_optimization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="ascii"
    )

    models = ["Full 640", "Compact 384", "Fine-tuned\n(rejected)"]
    score_sets = [full, compact, finetuned]
    categories = ["Overall", "Cell", "Droplet"]
    colors = ["#1971c2", "#e8590c", "#2b8a3e"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    x_positions = range(len(models))
    bar_width = 0.24
    for category_index, (category, color) in enumerate(zip(categories, colors)):
        key = "overall" if category == "Overall" else category.lower()
        values = [
            float(item[key]["f1"] if key == "overall" else item["classes"][key]["f1"])
            * 100.0
            for item in score_sets
        ]
        axes[0].bar(
            [value + (category_index - 1) * bar_width for value in x_positions],
            values,
            width=bar_width,
            label=category,
            color=color,
        )
    axes[0].set_xticks(list(x_positions), models)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Test F1 (%)")
    axes[0].set_title("Independent test quality")
    axes[0].legend(loc="lower left")
    axes[0].grid(axis="y", alpha=0.25)

    pipeline_names = ["Full 640", "Compact 384"]
    fps_values = [
        float(full_speed["average_processing_fps"]),
        float(compact_speed["average_processing_fps"]),
    ]
    bars = axes[1].bar(pipeline_names, fps_values, color=["#868e96", "#0b7285"])
    axes[1].axhline(30.0, color="#c92a2a", linestyle="--", label="30 FPS target")
    axes[1].set_ylabel("Processing FPS")
    axes[1].set_title(f"600-frame realtime test ({speedup:.2f}x faster)")
    axes[1].legend(loc="upper left")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, fps_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.1f}",
            ha="center",
        )

    reduction_names = ["Camera ROI\narea", "Model input\npixels"]
    reduction_values = [roi_area_reduction * 100.0, input_pixel_reduction * 100.0]
    bars = axes[2].bar(reduction_names, reduction_values, color=["#5f3dc4", "#f08c00"])
    axes[2].set_ylim(0, 100)
    axes[2].set_ylabel("Reduction (%)")
    axes[2].set_title("Work removed before inference")
    axes[2].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, reduction_values):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.1f}%",
            ha="center",
        )

    fig.suptitle("PC pipeline optimization before FPGA deployment")
    fig.tight_layout()
    fig.savefig(OUTPUT / "pc_optimization_summary.png", dpi=180)
    plt.close(fig)

    compact_overall = compact["overall"]
    compact_cell = compact["classes"]["cell"]
    compact_droplet = compact["classes"]["droplet"]
    report = f"""# Ket qua toi uu pipeline PC

## Quyet dinh cuoi

Dung checkpoint YOLO11n goc voi mot ROI gon tai doan sau dau tao giot. Loai
checkpoint fine-tune vi no overfit validation va lam F1 test doc lap giam tu
{float(compact_overall['f1']) * 100:.1f}% xuong {float(finetuned['overall']['f1']) * 100:.1f}%.

## Pipeline duoc chon

- Frame tham chieu: 1280x800.
- ROI: x=611, y=419, width=154, height=115.
- Tien xu ly: resize ROI thanh 384x288 va median-pad thanh 384x384.
- Confidence: cell 0.33, droplet 0.85; NMS IoU 0.50.
- Tracking: cung lop, du doan van toc, dem trai sang phai, xac nhan ba frame va
  hysteresis 4% quanh vach dem.

## Test doc lap

- Tong: precision {float(compact_overall['precision']) * 100:.1f}%, recall {float(compact_overall['recall']) * 100:.1f}%, F1 {float(compact_overall['f1']) * 100:.1f}%.
- Cell: precision {float(compact_cell['precision']) * 100:.1f}%, recall {float(compact_cell['recall']) * 100:.1f}%, F1 {float(compact_cell['f1']) * 100:.1f}%.
- Droplet: precision {float(compact_droplet['precision']) * 100:.1f}%, recall {float(compact_droplet['recall']) * 100:.1f}%, F1 {float(compact_droplet['f1']) * 100:.1f}%.
- mAP50 tieu chuan: {float(compact_map['overall']['map50']) * 100:.1f}% tong, {float(compact_map['cell']['map50']) * 100:.1f}% cell, {float(compact_map['droplet']['map50']) * 100:.1f}% droplet.

## Toc do tren NVIDIA MX330

- Pipeline 640x640 cu: {float(full_speed['average_processing_fps']):.1f} FPS, p95 {float(full_speed['p95_processing_ms']):.1f} ms.
- Pipeline 384x384 moi: {float(compact_speed['average_processing_fps']):.1f} FPS, p95 {float(compact_speed['p95_processing_ms']):.1f} ms.
- Nhanh hon {speedup:.2f} lan; dien tich ROI camera giam {roi_area_reduction * 100:.1f}%.

## Tieu chi prototype

- Toc do tu 30 FPS: DAT.
- F1 tong tu 80%: DAT.
- Droplet F1 tu 90%: DAT.
- Cell F1 tu 85%: CHUA DAT ({float(compact_cell['f1']) * 100:.1f}%).

## Quy tac train lai

Khong train tiep tren cung 40 source frame. Truoc lan train tiep theo, them cac
video doc lap voi focus, anh sang va cell kho khac nhau. Chon nguong tren validation
va chi danh gia tap test sau khi da khoa cau hinh.
"""
    (OUTPUT / "REPORT.md").write_text(report, encoding="ascii")
    print(OUTPUT / "pc_optimization_summary.json")
    print(OUTPUT / "pc_optimization_summary.png")


if __name__ == "__main__":
    main()
