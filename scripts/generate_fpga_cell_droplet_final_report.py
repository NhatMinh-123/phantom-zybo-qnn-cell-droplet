from __future__ import annotations

import csv
import json
import math
import shutil
import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"E:\fpga")
OUT = ROOT / "reports" / "final_report_2026_09_02"
ASSETS = OUT / "assets"

SPEC = Path(r"C:\Users\DELL\Downloads\FPGA_Cell_Droplet_Final_Report_Spec.md")
DATASET = ROOT / "dataset" / "15micro_yolov11_v1"
YOLO = ROOT / "models" / "15micro" / "yolo11n_colab_v1_clean"
YOLO_METRICS = YOLO / "metrics_summary.json"
YOLO_RUN = YOLO / "yolo11n_baseline"
YOLO_TEST = YOLO / "eval" / "test"
SOURCE_VIDEO = ROOT / "data" / "raw" / "09_07_2026" / "09_07_2026" / "3.4.mp4"

FPGA_ROOT = ROOT / "final_results" / "zybo_z7_10_qnn_60fps_104mhz"
FPGA_RESULTS = FPGA_ROOT / "results.json"
TRACK_DIR = FPGA_ROOT / "video_3_4_exact_tracking"
TRACK_REPORT = TRACK_DIR / "report.json"
TRACK_EVENTS = TRACK_DIR / "confirmed_events.csv"
TRACK_FRAMES = TRACK_DIR / "frame_summary.csv"
SYSTEM_EVIDENCE = ROOT / "final_results" / "system_evidence_2026_08_28"
SYSTEM_SUMMARY = SYSTEM_EVIDENCE / "system_test_summary.json"

ARTY_ACCURACY = ROOT / "final_results" / "arty_s7_25_qnn_detection" / "08_sparse_stream_realtime_12m" / "evaluation" / "fpga_labeled_test_30" / "metrics.json"
PC_QNN_REPORT = ROOT / "final_results" / "15micro_pipeline_v1" / "18_pc_qnn96_roi120_balanced_video_3_5" / "report.json"

VIDEO_MAIN_URL = "https://drive.google.com/file/d/1XtFskmwLJvyORudBzdn8jEPzcZHY0oeM/view"
VIDEO_SMOKE_URL = "https://drive.google.com/file/d/12jvM9jY3e4rHb4r5iJLhpetFt3X-y5FY/view"
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1Dk7WdtnHExftWusxpR6KKiUrFtg_WjHd"
GOOGLE_DOC_DETAILED_URL = "https://docs.google.com/document/d/1-pLv_1gIKKU228BowYMTY2zmNyQVE4vNEQK-pi02IjI/edit"
WORD_DETAILED_URL = "https://drive.google.com/file/d/1sqrPiO5uVQbbZTsSnH0rFm97_IsksOa2/view"
PDF_DETAILED_URL = "https://drive.google.com/file/d/1gl7FlgEIJiVOdq9zv_fRz7_IlfAGPxq9/view"


COLORS = {
    "blue": "1F4E78",
    "blue2": "2F75B5",
    "green": "2E8B57",
    "orange": "ED7D31",
    "red": "C00000",
    "gray": "666666",
    "light": "EAF2F8",
    "light_green": "E2F0D9",
    "light_orange": "FCE4D6",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFCFE",
        }
    )


def save_fig(fig: plt.Figure, name: str) -> Path:
    path = ASSETS / name
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def count_dataset() -> dict:
    split_counts: dict[str, int] = {}
    class_counts = {0: 0, 1: 0}
    for split in ("train", "valid", "test"):
        split_counts[split] = len(list((DATASET / split / "images").glob("*")))
        for label in (DATASET / split / "labels").glob("*.txt"):
            for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) >= 5 and int(parts[0]) in class_counts:
                    class_counts[int(parts[0])] += 1
    return {
        "splits": split_counts,
        "classes": {"cell": class_counts[0], "droplet": class_counts[1]},
        "images": sum(split_counts.values()),
        "boxes": sum(class_counts.values()),
    }


def read_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {index} from {path}")
    return frame


def put_label(frame: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = xy
    cv2.rectangle(frame, (x, y - 25), (x + max(100, len(text) * 10), y), color, -1)
    cv2.putText(frame, text, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


def make_source_and_roi_images() -> dict[str, Path]:
    frame = read_frame(SOURCE_VIDEO, 120)
    original = ASSETS / "source_frame_3_4_frame120.jpg"
    cv2.imwrite(str(original), frame)

    annotated = frame.copy()
    rois = {
        "ROI 1": (628, 410, 748, 530),
        "ROI 2": (748, 410, 868, 530),
    }
    for idx, (name, (x1, y1, x2, y2)) in enumerate(rois.items()):
        color = (40, 180, 40) if idx == 0 else (255, 150, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        put_label(annotated, name, (x1, y1), color)
    roi_overlay = ASSETS / "dual_roi_on_source_frame.jpg"
    cv2.imwrite(str(roi_overlay), annotated)

    crops = []
    for x1, y1, x2, y2 in rois.values():
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (360, 360), interpolation=cv2.INTER_CUBIC)
        crops.append(crop)
    panel = np.hstack(crops)
    cv2.putText(panel, "ROI 1 - candidate", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 120, 30), 2, cv2.LINE_AA)
    cv2.putText(panel, "ROI 2 - confirm", (372, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 0), 2, cv2.LINE_AA)
    crop_path = ASSETS / "dual_roi_crops.jpg"
    cv2.imwrite(str(crop_path), panel)
    return {"original": original, "overlay": roi_overlay, "crops": crop_path}


def draw_yolo_annotations(image_path: Path, label_path: Path, output: Path) -> Path:
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    names = {0: "cell", 1: "droplet"}
    colors = {0: (40, 40, 220), 1: (180, 40, 160)}
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        color = colors.get(cls, (0, 180, 180))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, names.get(cls, str(cls)), (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(output), image)
    return output


def make_dataset_images() -> dict[str, Path]:
    image_path = next((DATASET / "test" / "images").glob("*.jpg"))
    label_path = DATASET / "test" / "labels" / f"{image_path.stem}.txt"
    annotated = draw_yolo_annotations(image_path, label_path, ASSETS / "dataset_annotated_sample.jpg")

    keywords = ["native", "high_light", "low_light", "soft_blur", "resolution_loss", "sensor_jpeg"]
    picked: list[tuple[str, Path]] = []
    all_images = list(DATASET.glob("*/images/*.jpg"))
    for key in keywords:
        found = next((p for p in all_images if key in p.name), None)
        if found:
            picked.append((key.replace("_", " "), found))
    tiles = []
    for label, path in picked:
        img = cv2.imread(str(path))
        img = cv2.resize(img, (280, 280), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 245), (280, 280), (20, 20, 20), -1)
        cv2.putText(img, label, (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    if len(tiles) >= 6:
        panel = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])])
    else:
        panel = np.hstack(tiles)
    diversity = ASSETS / "dataset_quality_variants.jpg"
    cv2.imwrite(str(diversity), panel)
    return {"annotated": annotated, "diversity": diversity, "raw_sample": image_path}


def chart_dataset(counts: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    splits = counts["splits"]
    axes[0].bar(["Train", "Validation", "Test"], list(splits.values()), color=["#2F75B5", "#70AD47", "#ED7D31"])
    axes[0].set_title("Phân chia 900 ảnh")
    axes[0].set_ylabel("Số ảnh")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].bar_label(axes[0].containers[0], padding=3)
    classes = counts["classes"]
    axes[1].bar(["Cell", "Droplet"], list(classes.values()), color=["#C00000", "#7030A0"])
    axes[1].set_title("Bounding box trong bản export huấn luyện")
    axes[1].set_ylabel("Số box")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].bar_label(axes[1].containers[0], padding=3)
    fig.suptitle("Cấu trúc dataset 15 µm", fontsize=15, weight="bold")
    fig.tight_layout()
    return save_fig(fig, "dataset_distribution.png")


def chart_yolo_metrics(metrics: dict) -> Path:
    test = metrics["test"]
    f1 = 2 * test["precision"] * test["recall"] / (test["precision"] + test["recall"])
    labels = ["Precision", "Recall", "F1", "mAP@50", "mAP@50:95"]
    values = [test["precision"], test["recall"], f1, test["map50"], test["map50_95"]]
    fig, ax = plt.subplots(figsize=(9.4, 4.7))
    bars = ax.bar(labels, np.array(values) * 100, color=["#2F75B5", "#70AD47", "#5B9BD5", "#ED7D31", "#A5A5A5"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Phần trăm (%)")
    ax.set_title("YOLO11n trên tập test 15 µm")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{v*100:.2f}%" for v in values], padding=3)
    fig.tight_layout()
    return save_fig(fig, "yolo_test_metrics.png")


def chart_yolo_training() -> Path:
    rows = list(csv.DictReader((YOLO_RUN / "results.csv").open(encoding="utf-8-sig")))
    epochs = [int(float(r["epoch"])) for r in rows]
    p = [float(r["metrics/precision(B)"]) for r in rows]
    recall = [float(r["metrics/recall(B)"]) for r in rows]
    map50 = [float(r["metrics/mAP50(B)"]) for r in rows]
    map5095 = [float(r["metrics/mAP50-95(B)"]) for r in rows]
    train_box = [float(r["train/box_loss"]) for r in rows]
    val_box = [float(r["val/box_loss"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    axes[0].plot(epochs, p, label="Precision", lw=1.8)
    axes[0].plot(epochs, recall, label="Recall", lw=1.8)
    axes[0].plot(epochs, map50, label="mAP@50", lw=1.8)
    axes[0].plot(epochs, map5095, label="mAP@50:95", lw=1.8)
    axes[0].set_title("Metric validation theo epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].plot(epochs, train_box, label="Train box loss", lw=1.8)
    axes[1].plot(epochs, val_box, label="Validation box loss", lw=1.8)
    axes[1].set_title("Box loss")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Diễn biến huấn luyện YOLO11n", fontsize=15, weight="bold")
    fig.tight_layout()
    return save_fig(fig, "yolo_training_curves.png")


def chart_class_localization(metrics: dict) -> Path:
    test = metrics["test"]
    selected = metrics["test_at_selected_confidence"]
    labels = ["Cell", "Droplet"]
    base = [test["cell"]["map50_95"], test["droplet"]["map50_95"]]
    op = [selected["cell"]["map50_95"], selected["droplet"]["map50_95"]]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    b1 = ax.bar(x - 0.18, np.array(base) * 100, 0.36, label="Mặc định", color="#5B9BD5")
    b2 = ax.bar(x + 0.18, np.array(op) * 100, 0.36, label="Ngưỡng conf = 0,10", color="#ED7D31")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("mAP@50:95 (%)")
    ax.set_title("Khả năng định vị theo lớp")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.bar_label(b1, labels=[f"{v*100:.1f}%" for v in base], padding=2)
    ax.bar_label(b2, labels=[f"{v*100:.1f}%" for v in op], padding=2)
    fig.tight_layout()
    return save_fig(fig, "yolo_per_class_localization.png")


def chart_fpga_resources(fpga: dict) -> Path:
    r = fpga["resources"]
    labels = ["LUT", "FF", "BRAM", "DSP"]
    values = [r["slice_luts_percent"], r["slice_registers_percent"], r["bram_percent"], r["dsp_percent"]]
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    bars = ax.bar(labels, values, color=["#C00000", "#ED7D31", "#2F75B5", "#70AD47"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mức sử dụng (%)")
    ax.set_title("Tài nguyên QNN W4A6 trên Zybo Z7-10")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=3)
    fig.tight_layout()
    return save_fig(fig, "fpga_resource_utilization.png")


def chart_performance(fpga: dict, system: dict, pc_qnn: dict) -> Path:
    batch = fpga["hardware_benchmark"]
    video = fpga["corrected_video_demo"]
    cam = system["camera_live"]
    labels = ["Camera SDK\nlive", "PC QNN 96²\nbao gồm ghi video", "FPGA 1 ROI", "FPGA 2 ROI"]
    values = [cam["sdk_acquisition_fps"], pc_qnn["export_throughput_fps_including_video_write"], batch["roi_per_second"], batch["dual_roi_frames_per_second"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    b = axes[0].bar(labels, values, color=["#A5A5A5", "#5B9BD5", "#70AD47", "#ED7D31"])
    axes[0].set_ylabel("Thông lượng (FPS hoặc ROI/s)")
    axes[0].set_title("Thông lượng đo được theo phạm vi")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].bar_label(b, labels=[f"{v:.2f}" for v in values], padding=2)
    latency_labels = ["QNN batch/ROI", "QNN video/ROI", "UART 2 ROI"]
    latency_values = [1000 / batch["roi_per_second"], video["mean_qnn_ms_per_roi"], video["mean_dual_roi_uart_ms"]]
    b2 = axes[1].bar(latency_labels, latency_values, color=["#70AD47", "#5B9BD5", "#C00000"])
    axes[1].set_ylabel("Độ trễ (ms)")
    axes[1].set_title("Lõi QNN và nút thắt UART")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].bar_label(b2, labels=[f"{v:.2f}" for v in latency_values], padding=2)
    fig.suptitle("Phân rã hiệu năng hệ thống", fontsize=15, weight="bold")
    fig.tight_layout()
    return save_fig(fig, "system_performance_breakdown.png")


def chart_platform_comparison(fpga: dict, arty: dict) -> Path:
    labels = ["Arty S7\ncore detector", "Zybo 100 MHz\n2 ROI", "Zybo 104 MHz\n2 ROI"]
    values = [arty["accelerator"]["fps_mean"], fpga["baseline_100mhz"]["dual_roi_frames_per_second"], fpga["hardware_benchmark"]["dual_roi_frames_per_second"]]
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    bars = ax.bar(labels, values, color=["#5B9BD5", "#A5A5A5", "#70AD47"])
    ax.axhline(60, color="#C00000", ls="--", lw=1.5, label="Mục tiêu 60 FPS")
    ax.set_ylabel("FPS")
    ax.set_title("So sánh các mốc triển khai FPGA đã đo")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3)
    fig.tight_layout()
    return save_fig(fig, "fpga_platform_fps_comparison.png")


def chart_tracking_events() -> Path:
    rows = list(csv.DictReader(TRACK_FRAMES.open(encoding="utf-8-sig")))
    frame = [int(r["frame_index"]) for r in rows]
    cell = [int(r["confirmed_cell_total"]) for r in rows]
    droplet = [int(r["confirmed_droplet_total"]) for r in rows]
    uart = [float(r["dual_roi_uart_ms"]) for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 6.0), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})
    axes[0].step(frame, cell, where="post", label="Cell đã xác nhận", color="#C00000", lw=2)
    axes[0].step(frame, droplet, where="post", label="Droplet đã xác nhận", color="#7030A0", lw=2)
    axes[0].set_ylabel("Số sự kiện tích lũy")
    axes[0].set_title("Bộ đếm hai ROI: ROI 1 phát hiện, ROI 2 xác nhận")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(frame, uart, color="#ED7D31", lw=1)
    axes[1].axhline(np.mean(uart), color="#C00000", ls="--", label=f"Trung bình {np.mean(uart):.04f} ms")
    axes[1].set_xlabel("Frame nguồn")
    axes[1].set_ylabel("UART 2 ROI (ms)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    return save_fig(fig, "dual_roi_counting_timeline.png")


def chart_fpga_accuracy(arty: dict) -> Path:
    overall = arty["overall"]
    cls = arty["classes"]
    labels = ["Overall P", "Overall R", "Overall F1", "Cell F1", "Droplet F1", "mAP@50"]
    values = [overall["precision"], overall["recall"], overall["f1"], cls["cell"]["f1"], cls["droplet"]["f1"], arty["map50"]]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    bars = ax.bar(labels, np.array(values) * 100, color=["#2F75B5", "#70AD47", "#5B9BD5", "#C00000", "#7030A0", "#ED7D31"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Phần trăm (%)")
    ax.set_title("Kiểm thử detector QNN trên Arty S7-25, 30 ảnh (giai đoạn trước)")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{v*100:.1f}%" for v in values], padding=3)
    fig.tight_layout()
    return save_fig(fig, "arty_qnn_accuracy_reference.png")


def make_flow_diagram(name: str, title: str, nodes: list[str], highlight: set[int] | None = None) -> Path:
    highlight = highlight or set()
    fig, ax = plt.subplots(figsize=(12.0, 3.0))
    ax.axis("off")
    n = len(nodes)
    box_w = 0.82 / n
    for i, node in enumerate(nodes):
        x = 0.06 + i * (0.88 / n)
        color = "#DDEBF7" if i not in highlight else "#E2F0D9"
        rect = plt.Rectangle((x, 0.35), box_w * 0.86, 0.32, facecolor=color, edgecolor="#1F4E78", linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + box_w * 0.43, 0.51, node, ha="center", va="center", fontsize=9, wrap=True)
        if i < n - 1:
            ax.annotate("", xy=(x + 0.88 / n, 0.51), xytext=(x + box_w * 0.86, 0.51), arrowprops=dict(arrowstyle="->", lw=1.4, color="#1F4E78"))
    ax.text(0.5, 0.87, title, ha="center", va="center", fontsize=15, weight="bold", color="#1F4E78")
    return save_fig(fig, name)


def create_cnn_theory_diagrams() -> dict[str, Path]:
    """Create original diagrams so the theory matches the implemented network."""
    diagrams: dict[str, Path] = {}

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    ax = axes[0]
    ax.axis("off")
    inputs = [(0.12, 0.78, "x1"), (0.12, 0.58, "x2"), (0.12, 0.38, "x3"), (0.12, 0.18, "xn")]
    for x, y, label in inputs:
        ax.add_patch(plt.Circle((x, y), 0.045, facecolor="#DDEBF7", edgecolor="#1F4E78", lw=1.4))
        ax.text(x, y, label, ha="center", va="center", fontsize=10)
        ax.annotate("", xy=(0.48, 0.48), xytext=(x + 0.05, y), arrowprops=dict(arrowstyle="->", color="#1F4E78"))
    ax.add_patch(plt.Circle((0.53, 0.48), 0.10, facecolor="#FFF2CC", edgecolor="#BF9000", lw=1.6))
    ax.text(0.53, 0.48, "Σ", ha="center", va="center", fontsize=20, weight="bold")
    ax.annotate("", xy=(0.78, 0.48), xytext=(0.64, 0.48), arrowprops=dict(arrowstyle="->", color="#1F4E78", lw=1.5))
    ax.add_patch(plt.Circle((0.84, 0.48), 0.08, facecolor="#E2F0D9", edgecolor="#2E8B57", lw=1.6))
    ax.text(0.84, 0.48, "φ", ha="center", va="center", fontsize=18, weight="bold")
    ax.text(0.53, 0.08, "z = Σ wi xi + b", ha="center", fontsize=12)
    ax.text(0.84, 0.08, "a = φ(z)", ha="center", fontsize=12)
    ax.set_title("Neuron nhân tạo và lan truyền thuận", fontsize=13, weight="bold")

    ax = axes[1]
    ax.axis("off")
    stages = [
        (0.08, 0.62, "Loss L", "#FCE4D6"),
        (0.32, 0.62, "Activation a", "#E2F0D9"),
        (0.56, 0.62, "Pre-activation z", "#FFF2CC"),
        (0.80, 0.62, "Weight w", "#DDEBF7"),
    ]
    for x, y, label, color in stages:
        ax.add_patch(plt.Rectangle((x - 0.09, y - 0.08), 0.18, 0.16, facecolor=color, edgecolor="#1F4E78", lw=1.3))
        ax.text(x, y, label, ha="center", va="center", fontsize=9)
    for a, b in zip(stages[:-1], stages[1:]):
        ax.annotate("", xy=(b[0] - 0.10, 0.62), xytext=(a[0] + 0.10, 0.62), arrowprops=dict(arrowstyle="->", color="#C00000", lw=1.5))
    ax.text(0.50, 0.35, "∂L/∂w = (∂L/∂a)(∂a/∂z)(∂z/∂w)", ha="center", fontsize=13)
    ax.text(0.50, 0.18, "θ(t+1) = θ(t) - η∇θJ(θ)", ha="center", fontsize=13)
    ax.text(0.50, 0.07, "Gradient đi ngược qua chuỗi phép toán để cập nhật tham số", ha="center", fontsize=9, color="#666666")
    ax.set_title("Lan truyền ngược và tối ưu", fontsize=13, weight="bold")
    fig.suptitle("Cơ chế học của mạng neural", fontsize=16, weight="bold", color="#1F4E78")
    fig.tight_layout()
    diagrams["neuron"] = save_fig(fig, "cnn_neuron_backprop.png")

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.4))
    image = np.array([
        [0.15, 0.18, 0.18, 0.16, 0.14, 0.14],
        [0.17, 0.22, 0.38, 0.40, 0.22, 0.15],
        [0.16, 0.35, 0.82, 0.86, 0.38, 0.16],
        [0.14, 0.34, 0.80, 0.84, 0.36, 0.15],
        [0.13, 0.20, 0.36, 0.38, 0.20, 0.14],
        [0.12, 0.13, 0.14, 0.14, 0.13, 0.12],
    ])
    kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float)
    feature = np.array([[np.sum(image[i:i+3, j:j+3] * kernel) for j in range(4)] for i in range(4)])
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].add_patch(plt.Rectangle((1.5, 1.5), 3, 3, fill=False, edgecolor="#C00000", lw=2.2))
    axes[0].set_title("Ảnh/feature đầu vào X")
    axes[1].imshow(kernel, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_title("Kernel 3×3 W")
    axes[2].imshow(feature, cmap="viridis")
    axes[2].set_title("Feature map Y")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Phép tích chập: kernel trượt và tích lũy Multiply-Accumulate", fontsize=15, weight="bold", color="#1F4E78")
    fig.text(0.5, 0.02, "Mỗi điểm đầu ra là tổng có trọng số của lân cận; nhiều kernel học các mẫu biên, đốm sáng, cấu trúc vòng và texture khác nhau.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    diagrams["convolution"] = save_fig(fig, "cnn_convolution_feature_map.png")

    layers = [
        ("Input", "96×96×1", "UINT8", "-"),
        ("Conv1+BN+QReLU", "48×48×12", "W8/A6", "3×3, s=2"),
        ("Conv2+BN+QReLU", "24×24×16", "W4/A6", "3×3, s=2"),
        ("Conv3+BN+QReLU", "24×24×24", "W4/A6", "3×3, s=1"),
        ("Conv4+BN+QReLU", "24×24×24", "W4/A6", "3×3, s=1"),
        ("Raw head", "24×24×15", "W8/INT8", "1×1, s=1"),
    ]
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.axis("off")
    x_positions = np.linspace(0.05, 0.85, len(layers))
    heights = [0.62, 0.54, 0.46, 0.46, 0.46, 0.42]
    for i, ((name, shape, bits, op), x, h) in enumerate(zip(layers, x_positions, heights)):
        w = 0.125
        y = 0.47 - h / 2
        color = "#DDEBF7" if i == 0 else ("#E2F0D9" if i < 5 else "#FCE4D6")
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#1F4E78", lw=1.5))
        ax.text(x + w / 2, y + h * 0.70, name, ha="center", va="center", fontsize=8.6, weight="bold")
        ax.text(x + w / 2, y + h * 0.47, shape, ha="center", va="center", fontsize=9.3)
        ax.text(x + w / 2, y + h * 0.27, bits, ha="center", va="center", fontsize=8.5, color="#C00000")
        ax.text(x + w / 2, y + h * 0.10, op, ha="center", va="center", fontsize=7.8, color="#666666")
        if i < len(layers) - 1:
            ax.annotate("", xy=(x_positions[i + 1] - 0.006, 0.47), xytext=(x + w + 0.006, 0.47), arrowprops=dict(arrowstyle="->", lw=1.4, color="#1F4E78"))
    ax.text(0.50, 0.91, "TinyQuantDetector triển khai trên Zybo Z7-10", ha="center", fontsize=16, weight="bold", color="#1F4E78")
    ax.text(0.50, 0.07, "Head 15 kênh = (2 slot cell + 1 slot droplet) × 5 giá trị [objectness, tx, ty, tw, th]. Sigmoid, decode và NMS đặt ngoài lõi QNN.", ha="center", fontsize=9.5)
    diagrams["architecture"] = save_fig(fig, "tiny_qnn_exact_architecture.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    ax = axes[0]
    x = np.linspace(-1.25, 1.25, 500)
    scale = 0.18
    q = np.clip(np.round(x / scale), -8, 7)
    x_hat = scale * q
    ax.plot(x, x, color="#A5A5A5", lw=1.2, label="FP32 lý tưởng")
    ax.step(x, x_hat, where="mid", color="#C00000", lw=1.8, label="W4 dequantized")
    ax.set_title("Lượng tử hóa đều 4 bit có dấu")
    ax.set_xlabel("Giá trị FP32 x")
    ax.set_ylabel("Giá trị biểu diễn x̂")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.axis("off")
    stages = [
        (0.04, "FP32\nweights/activation", "#DDEBF7"),
        (0.27, "Fake quant\nround + clip", "#FFF2CC"),
        (0.50, "Integer MAC\nW4 × A6", "#E2F0D9"),
        (0.73, "INT16\naccumulator", "#FCE4D6"),
        (0.91, "INT8\nraw head", "#D9EAD3"),
    ]
    for i, (x0, label, color) in enumerate(stages):
        w = 0.16 if i < 4 else 0.08
        ax.add_patch(plt.Rectangle((x0, 0.48), w, 0.23, facecolor=color, edgecolor="#1F4E78", lw=1.3))
        ax.text(x0 + w / 2, 0.595, label, ha="center", va="center", fontsize=8.7)
        if i < len(stages) - 1:
            ax.annotate("", xy=(stages[i + 1][0] - 0.01, 0.595), xytext=(x0 + w + 0.01, 0.595), arrowprops=dict(arrowstyle="->", color="#1F4E78", lw=1.4))
    ax.text(0.50, 0.31, "QAT: forward dùng x̂ = s·clip(round(x/s)+z); backward dùng STE", ha="center", fontsize=10)
    ax.text(0.50, 0.16, "FINN biến tensor QONNX thành datapath số nguyên, folding PE/SIMD và bitstream", ha="center", fontsize=9.5, color="#666666")
    ax.set_title("Từ QAT tới phép toán phần cứng", fontsize=13, weight="bold")
    fig.suptitle("Cơ chế lượng tử hóa W4A6", fontsize=16, weight="bold", color="#1F4E78")
    fig.tight_layout()
    diagrams["quantization"] = save_fig(fig, "qnn_quantization_qat.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7))
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.set_aspect("equal")
        ax.axis("off")
    axes[0].add_patch(plt.Rectangle((1.0, 1.2), 5.2, 3.8, facecolor="#5B9BD5", alpha=0.28, edgecolor="#2F75B5", lw=2))
    axes[0].add_patch(plt.Rectangle((3.5, 2.0), 4.8, 3.7, facecolor="#ED7D31", alpha=0.28, edgecolor="#C65911", lw=2))
    axes[0].text(2.0, 4.5, "GT", color="#2F75B5", weight="bold")
    axes[0].text(7.2, 5.25, "Prediction", color="#C65911", weight="bold")
    axes[0].text(4.9, 3.0, "Intersection", ha="center", fontsize=9)
    axes[0].text(5.0, 0.35, "IoU = |B ∩ Bgt| / |B ∪ Bgt|", ha="center", fontsize=11)
    axes[0].set_title("Độ chồng lấp hộp", fontsize=13, weight="bold")

    boxes = [(1.0, 1.1, 4.2, 3.3, 0.93), (1.5, 1.4, 4.0, 3.1, 0.81), (5.0, 1.0, 3.5, 3.5, 0.76)]
    for i, (x0, y0, w, h, score) in enumerate(boxes):
        color = ["#2F75B5", "#70AD47", "#ED7D31"][i]
        axes[1].add_patch(plt.Rectangle((x0, y0), w, h, fill=False, edgecolor=color, lw=2))
        axes[1].text(x0, y0 + h + 0.18, f"s={score:.2f}", color=color, fontsize=9)
    axes[1].text(5.0, 0.35, "NMS giữ score cao nhất, loại hộp cùng lớp nếu IoU > τ", ha="center", fontsize=10)
    axes[1].set_title("Non-Maximum Suppression", fontsize=13, weight="bold")
    fig.suptitle("Hình học detection và loại hộp trùng", fontsize=16, weight="bold", color="#1F4E78")
    fig.tight_layout()
    diagrams["detection"] = save_fig(fig, "detection_iou_nms.png")

    return diagrams


def create_algorithm_diagrams() -> dict[str, Path]:
    research = make_flow_diagram(
        "hardware_algorithm_codesign.png",
        "Đồng thiết kế thuật toán - phần cứng",
        ["Ràng buộc\nvi lưu", "ROI cố định", "CNN nhỏ", "QAT\nW4A6", "FINN / Vivado", "FPGA\ndataflow", "Theo dõi\nhai ROI"],
        {1, 3, 5, 6},
    )
    deploy = make_flow_diagram(
        "qnn_deployment_pipeline.png",
        "Chuỗi chuyển đổi mô hình QNN",
        ["PyTorch /\nBrevitas", "QAT", "QONNX", "FINN\nstreamline", "Folding\nPE/SIMD", "Vivado", "Bitstream", "Zybo PL"],
        {2, 4, 7},
    )

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.axis("off")
    boxes = [
        (0.04, 0.41, 0.12, 0.20, "Frame\n1280×800", "#DDEBF7"),
        (0.21, 0.41, 0.12, 0.20, "ROI 1\n120×120", "#E2F0D9"),
        (0.38, 0.41, 0.12, 0.20, "QNN W4A6\n96×96", "#E2F0D9"),
        (0.55, 0.62, 0.14, 0.18, "Candidate\nclass + track ID", "#FFF2CC"),
        (0.55, 0.20, 0.14, 0.18, "ROI 2 QNN\nconfirmation", "#FCE4D6"),
        (0.75, 0.41, 0.12, 0.20, "Temporal\nmatching", "#D9EAD3"),
        (0.90, 0.41, 0.08, 0.20, "Count\nonce", "#C6E0B4"),
    ]
    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#1F4E78", linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    arrows = [((0.16, 0.51), (0.21, 0.51)), ((0.33, 0.51), (0.38, 0.51)), ((0.50, 0.51), (0.55, 0.71)), ((0.50, 0.51), (0.55, 0.29)), ((0.69, 0.71), (0.75, 0.55)), ((0.69, 0.29), (0.75, 0.47)), ((0.87, 0.51), (0.90, 0.51))]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.5, color="#1F4E78"))
    ax.text(0.5, 0.92, "Thuật toán đếm hai ROI có xác nhận", ha="center", va="center", fontsize=16, weight="bold", color="#1F4E78")
    ax.text(0.5, 0.08, "Một vật thể chỉ tăng bộ đếm khi candidate ROI 1 được ghép đúng lớp và đúng cửa sổ thời gian tại ROI 2.", ha="center", fontsize=10)
    dual = save_fig(fig, "dual_roi_algorithm.png")

    fig, ax = plt.subplots(figsize=(11.5, 5.3))
    ax.axis("off")
    ax.text(0.5, 0.93, "Kiến trúc tích hợp Camera - Zynq - QNN", ha="center", fontsize=16, weight="bold", color="#1F4E78")
    stages = [
        (0.04, "Phantom VEO\nEthernet"),
        (0.22, "ARM PS\nSDK / network"),
        (0.40, "DDR + DMA"),
        (0.58, "PL: ROI +\nQNN W4A6"),
        (0.76, "Tracking /\nCounting"),
        (0.90, "PC / storage\n/ display"),
    ]
    for i, (x, label) in enumerate(stages):
        w = 0.12 if i < 5 else 0.08
        rect = plt.Rectangle((x, 0.43), w, 0.25, facecolor="#DDEBF7" if i < 3 else "#E2F0D9", edgecolor="#1F4E78", linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + w / 2, 0.555, label, ha="center", va="center", fontsize=9)
        if i < len(stages) - 1:
            nx = stages[i + 1][0]
            ax.annotate("", xy=(nx, 0.555), xytext=(x + w, 0.555), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1F4E78"))
    ax.text(0.31, 0.29, "Đã kiểm thử: Camera → PC", ha="center", fontsize=10, color="#2E8B57", weight="bold")
    ax.text(0.68, 0.29, "Đã kiểm thử: PC ROI → Zybo PL → kết quả", ha="center", fontsize=10, color="#2E8B57", weight="bold")
    ax.text(0.50, 0.13, "Chưa hoàn thành: Camera Ethernet → Zybo → QNN → PC theo một luồng end-to-end", ha="center", fontsize=10, color="#C00000", weight="bold")
    system = save_fig(fig, "camera_zybo_system_architecture.png")
    return {"research": research, "deploy": deploy, "dual": dual, "system": system}


def shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_text(cell, text: str, bold: bool = False, color: str | None = None, size: float = 9.5) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(str(text))
    r.bold = bold
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(doc: Document, headers: list[str], rows: list[list[object]], widths: list[float] | None = None) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], header, bold=True, color="FFFFFF", size=9)
        shade_cell(table.rows[0].cells[i], COLORS["blue"])
    for ridx, row in enumerate(rows):
        cells = table.add_row().cells
        for i, value in enumerate(row):
            set_cell_text(cells[i], value, size=8.8)
            if ridx % 2:
                shade_cell(cells[i], "F5F8FA")
    if widths:
        for row in table.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Inches(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_hyperlink(paragraph, text: str, url: str, color: str = "0563C1", underline: bool = True) -> None:
    part = paragraph.part
    r_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    new_run = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")
    c = OxmlElement("w:color")
    c.set(qn("w:val"), color)
    r_pr.append(c)
    if underline:
        u = OxmlElement("w:u")
        u.set(qn("w:val"), "single")
        r_pr.append(u)
    new_run.append(r_pr)
    text_node = OxmlElement("w:t")
    text_node.text = text
    new_run.append(text_node)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = "PAGE"
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char1, instr_text, fld_char2])


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    h = doc.add_heading(text, level=level)
    h.paragraph_format.keep_with_next = True
    h.paragraph_format.space_before = Pt(10 if level == 1 else 7)
    h.paragraph_format.space_after = Pt(5)


def add_body(doc: Document, text: str, bold_prefix: str | None = None) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(1)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(5)
    if bold_prefix and text.startswith(bold_prefix):
        p.add_run(bold_prefix).bold = True
        p.add_run(text[len(bold_prefix) :])
    else:
        p.add_run(text)


def add_equation(doc: Document, equation: str, label: str | None = None, explanation: str | None = None) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(equation)
    run.font.name = "Cambria Math"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Cambria Math")
    run.font.size = Pt(11.5)
    if label:
        label_run = p.add_run(f"    {label}")
        label_run.font.name = "Times New Roman"
        label_run.font.size = Pt(10)
    if explanation:
        q = doc.add_paragraph(explanation)
        q.alignment = WD_ALIGN_PARAGRAPH.CENTER
        q.paragraph_format.space_after = Pt(5)
        for r in q.runs:
            r.italic = True
            r.font.size = Pt(9.5)
            r.font.color.rgb = RGBColor.from_string(COLORS["gray"])


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        p.add_run(item)


def add_picture(doc: Document, path: Path, caption: str, width: float = 6.55) -> None:
    if not path.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.keep_with_next = False
    cap.paragraph_format.space_after = Pt(6)
    r = cap.runs[0]
    r.italic = True
    r.font.size = Pt(9.5)


def add_video_link(doc: Document, title: str, url: str, note: str) -> None:
    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_cell_text(table.cell(0, 0), "VIDEO", bold=True, color="FFFFFF", size=10)
    shade_cell(table.cell(0, 0), COLORS["red"])
    cell = table.cell(0, 1)
    cell.text = ""
    p = cell.paragraphs[0]
    add_hyperlink(p, title, url)
    p.add_run(f"\n{note}").font.size = Pt(9)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_chapter_break(doc: Document, title: str) -> None:
    doc.add_page_break()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(80)
    r = p.add_run(title.upper())
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(20)
    r.font.color.rgb = RGBColor.from_string(COLORS["blue"])
    doc.add_paragraph()


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.8)
    section.right_margin = Cm(2.0)
    styles = doc.styles
    styles["Normal"].font.name = "Times New Roman"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    styles["Normal"].font.size = Pt(12)
    for name, size, color in [("Title", 22, COLORS["blue"]), ("Heading 1", 16, COLORS["blue"]), ("Heading 2", 13, COLORS["blue2"]), ("Heading 3", 12, COLORS["green"])]:
        style = styles[name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
    for section in doc.sections:
        add_page_number(section.footer.paragraphs[0])


def make_report(assets: dict[str, Path], counts: dict, yolo: dict, fpga: dict, system: dict, arty: dict, pc_qnn: dict) -> Path:
    doc = Document()
    configure_document(doc)
    yolo_test = yolo["test"]
    yolo_f1 = 2 * yolo_test["precision"] * yolo_test["recall"] / (yolo_test["precision"] + yolo_test["recall"])

    # Cover
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(42)
    r = p.add_run("BÁO CÁO TỔNG KẾT DỰ ÁN FPGA")
    r.bold = True
    r.font.size = Pt(20)
    r.font.color.rgb = RGBColor.from_string(COLORS["blue"])
    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_before = Pt(24)
    r = p2.add_run("THIẾT KẾ VÀ TRIỂN KHAI HỆ THỐNG\nPHÁT HIỆN CELL/DROPLET THỜI GIAN THỰC\nBẰNG CNN LƯỢNG TỬ HÓA TRÊN FPGA")
    r.bold = True
    r.font.size = Pt(18)
    r.font.color.rgb = RGBColor.from_string(COLORS["blue"])
    doc.add_paragraph()
    add_picture(doc, assets["system_arch"], "Kiến trúc nghiên cứu Camera - Zynq - QNN", width=6.4)
    for text in ["Sinh viên thực hiện: Dương Nhật Minh", "Nền tảng cuối: Digilent Zybo Z7-10 - XC7Z010-1CLG400C", "Ngày hoàn thiện báo cáo: 02/09/2026"]:
        q = doc.add_paragraph(text)
        q.alignment = WD_ALIGN_PARAGRAPH.CENTER
        q.runs[0].font.size = Pt(12)

    doc.add_page_break()
    add_heading(doc, "TÓM TẮT", 1)
    add_body(doc, "Báo cáo trình bày quá trình đồng thiết kế thuật toán và phần cứng cho bài toán phát hiện cell và droplet trong video kính hiển vi của hệ vi lưu. Điểm xuất phát là detector YOLO11n trên PC để kiểm chứng khả năng học dữ liệu. Sau đó, dự án khai thác tính cố định của kênh vi lưu để chuyển từ xử lý toàn frame 1280×800 sang hai ROI 120×120, thiết kế QNN W4A6 đầu vào 96×96 và triển khai suy luận trong Programmable Logic của Zybo Z7-10.")
    add_body(doc, f"Dataset cuối gồm {counts['images']} ảnh với {counts['boxes']:,} bounding box được sử dụng để huấn luyện hai lớp cell và droplet. YOLO11n đạt precision {yolo_test['precision']*100:.2f}%, recall {yolo_test['recall']*100:.2f}%, F1 {yolo_f1*100:.2f}% và mAP@50 {yolo_test['map50']*100:.2f}% trên tập test. Trên FPGA, lõi QNN xử lý 121,23 ROI/s; với hai ROI cho mỗi frame, thông lượng tương đương 60,61 FPS. Thiết kế đạt timing tại 104 MHz với WNS +0,586 ns, sử dụng 66,27% LUT, 41,97% FF, 9,17% BRAM và 13,75% DSP; công suất on-chip ước tính 1,872 W.")
    add_body(doc, "Thuật toán đếm sử dụng ROI 1 để tạo candidate và ROI 2 để xác nhận cùng lớp trong cửa sổ thời gian, nhờ đó một đối tượng chỉ tăng bộ đếm một lần. Thử nghiệm 300 frame trên video 3.4 xác nhận 8 cell và 8 droplet. Báo cáo cũng phân biệt rõ tốc độ lõi QNN với tốc độ đường kiểm thử UART: lõi đạt mục tiêu 60 FPS hai ROI, trong khi vòng PC-UART đồng bộ mất trung bình 127,04 ms/frame và chỉ dùng cho kiểm chứng. Camera Phantom đã truyền ảnh live tới PC qua Ethernet, nhưng luồng camera Ethernet đi thẳng vào Zybo chưa được tích hợp; trạng thái hệ thống là các phân hệ đã PASS, tích hợp end-to-end còn tiếp tục.")
    add_heading(doc, "Từ khóa", 2)
    doc.add_paragraph("Microfluidics; cell; droplet; CNN; QNN; QAT; FINN; FPGA; Zybo Z7-10; dual ROI; real-time.")

    add_heading(doc, "MỤC LỤC RÚT GỌN", 1)
    contents = [
        "Chương 1. Giới thiệu", "Chương 2. Cơ sở lý thuyết", "Chương 3. Dữ liệu và tiền xử lý", "Chương 4. Phương pháp CNN/QNN và thuật toán đếm", "Chương 5. Kiến trúc FPGA", "Chương 6. Thiết kế thực nghiệm", "Chương 7. Kết quả và phân tích", "Chương 8. Thảo luận, kết luận và hướng phát triển", "Tài liệu tham khảo", "Phụ lục minh chứng",
    ]
    add_bullets(doc, contents)

    add_chapter_break(doc, "CHƯƠNG 1. GIỚI THIỆU")
    add_heading(doc, "1.1. Bối cảnh", 2)
    add_body(doc, "Trong hệ vi lưu, cell và droplet chuyển động nhanh qua một vùng quan sát hẹp. Video nguồn có độ phân giải 1280×800 và tốc độ danh định xấp xỉ 100 FPS ở phía camera. Đối tượng nhỏ, có thể bám sát viền kênh và thay đổi hình dạng theo dòng chảy, nên detector phải đồng thời giữ recall cao và độ trễ thấp.")
    add_body(doc, "GPU phù hợp cho huấn luyện và xây dựng baseline nhưng không phải lúc nào cũng phù hợp với thiết bị edge. FPGA cho phép tạo datapath chuyên dụng, pipeline và xử lý song song với độ trễ xác định. Tuy vậy, tài nguyên của Spartan-7 và Zynq-7010 không đủ dư dả để đưa nguyên YOLO11n lên chip. Bởi vậy, dự án lựa chọn tư tưởng hardware-algorithm co-design: giảm miền quan sát trước, dùng CNN nhỏ, lượng tử hóa và thiết kế bộ đếm theo chuyển động vật lý của đối tượng.")
    add_picture(doc, assets["codesign"], "Hình 1.1. Tư tưởng đồng thiết kế thuật toán - phần cứng")
    add_heading(doc, "1.2. Mục tiêu và phạm vi", 2)
    add_bullets(doc, [
        "Xây dựng dataset cell/droplet có biến thể ánh sáng, nén và độ nét.",
        "Đánh giá YOLO11n trên PC làm teacher/baseline.",
        "Thiết kế QNN W4A6 nhỏ, tương thích FINN và FPGA.",
        "Đạt tối thiểu 60 FPS cho hai ROI trên phần cứng.",
        "Theo dõi và đếm một đối tượng đúng một lần bằng hai ROI xác nhận lẫn nhau.",
        "Kiểm chứng camera Ethernet, PL QNN, timing, tài nguyên, công suất và video thật.",
    ])
    add_body(doc, "FPGA chỉ thực hiện inference; huấn luyện diễn ra trên PC/Colab. Báo cáo không tuyên bố luồng camera → FPGA → PC đã hoàn thiện: camera live và QNN FPGA đã được kiểm thử độc lập, còn cầu nối frame Ethernet vào Zynq là công việc tích hợp kế tiếp.")

    add_chapter_break(doc, "CHƯƠNG 2. CƠ SỞ LÝ THUYẾT")
    add_heading(doc, "2.1. Từ neuron nhân tạo đến mạng neural nhiều lớp", 2)
    add_body(doc, "Neuron nhân tạo nhận vector đầu vào x = [x₁, x₂, …, xₙ], nhân từng phần tử với trọng số học được w, cộng độ lệch b rồi đưa qua hàm kích hoạt φ. Trọng số thể hiện mức ảnh hưởng của từng đặc trưng; bias cho phép biên quyết định dịch khỏi gốc tọa độ. Một lớp gồm nhiều neuron tạo thành phép biến đổi phi tuyến; ghép nhiều lớp cho phép mạng xấp xỉ các quan hệ phức tạp giữa ảnh và nhãn.")
    add_equation(doc, "zⱼ = Σᵢ wⱼᵢxᵢ + bⱼ", "(2.1)", "zⱼ là tổng có trọng số tại neuron j.")
    add_equation(doc, "aⱼ = φ(zⱼ)", "(2.2)", "aⱼ là đầu ra sau kích hoạt và trở thành đầu vào của lớp kế tiếp.")
    add_body(doc, "Trong học có giám sát, mô hình fθ(x) được tối ưu trên tập N mẫu bằng cách cực tiểu hóa rủi ro thực nghiệm. Thành phần regularization R(θ), thường là chuẩn L2 của trọng số, hạn chế mô hình ghi nhớ nhiễu của tập train.")
    add_equation(doc, "J(θ) = (1/N) Σₙ L(fθ(xₙ), yₙ) + λR(θ)", "(2.3)")
    add_body(doc, "Lan truyền ngược áp dụng quy tắc dây chuyền để tính đạo hàm của loss theo từng tham số. Bộ tối ưu cập nhật tham số theo hướng ngược gradient; Adam dùng thêm trung bình động bậc một và bậc hai, nhưng nguyên lý cơ sở vẫn là giảm J(θ).")
    add_equation(doc, "∂L/∂w = (∂L/∂a)(∂a/∂z)(∂z/∂w)", "(2.4)")
    add_equation(doc, "θₜ₊₁ = θₜ − η∇θJ(θₜ)", "(2.5)", "η là learning rate.")
    add_picture(doc, assets["cnn_neuron"], "Hình 2.1. Neuron, lan truyền thuận, lan truyền ngược và cập nhật trọng số")

    add_heading(doc, "2.2. Mạng tích chập CNN", 2)
    add_heading(doc, "2.2.1. Phép tích chập hai chiều", 3)
    add_body(doc, "Ảnh có cấu trúc không gian: các pixel lân cận tạo nên biên, đốm, vòng tròn và texture. CNN khai thác cấu trúc này bằng kernel dùng chung trên toàn ảnh; nguyên lý học đặc trưng cục bộ và chia sẻ trọng số đã được hệ thống hóa trong các kiến trúc CNN nền tảng [1]. Với tensor vào X có Cᵢₙ kênh, kernel W kích thước Kₕ×Kw, stride S và padding P, phần tử đầu ra tại kênh o được tính như sau:")
    add_equation(doc, "Y[o,i,j] = bₒ + Σc Σu Σv W[o,c,u,v]·X[c, iS+u−P, jS+v−P]", "(2.6)")
    add_body(doc, "Dùng chung W tại mọi vị trí làm số tham số không phụ thuộc trực tiếp vào chiều cao/rộng ảnh và tạo tính tương đương tịnh tiến: cùng một mẫu có thể được nhận ra ở các vị trí khác nhau. Kích thước đầu ra theo chiều cao và rộng là:")
    add_equation(doc, "Hₒᵤₜ = ⌊(Hᵢₙ + 2P − Kₕ)/S⌋ + 1;   Wₒᵤₜ = ⌊(Wᵢₙ + 2P − Kw)/S⌋ + 1", "(2.7)")
    add_equation(doc, "Nweight = KₕKwCᵢₙCₒᵤₜ", "(2.8)")
    add_equation(doc, "MAC = HₒᵤₜWₒᵤₜCₒᵤₜKₕKwCᵢₙ", "(2.9)", "Một MAC gồm một phép nhân và một phép cộng tích lũy.")
    add_picture(doc, assets["cnn_convolution"], "Hình 2.2. Kernel tích chập trượt trên ảnh và sinh feature map")
    add_heading(doc, "2.2.2. Feature map, receptive field và downsampling", 3)
    add_body(doc, "Mỗi kernel học một loại đặc trưng. Ở các lớp đầu, kernel thường phản ứng với biên sáng–tối hoặc đốm cục bộ; lớp sâu kết hợp chúng thành cấu trúc vòng của droplet, texture và chấm nhỏ của cell. Receptive field của một phần tử tăng theo độ sâu, nhờ đó lớp head có thể dùng ngữ cảnh rộng hơn mà vẫn giữ tọa độ không gian.")
    add_body(doc, "Downsampling giảm H×W, nhờ vậy giảm MAC của các lớp sau. Max-pooling chọn giá trị lớn nhất trong cửa sổ Ω; TinyQuantDetector hiện tại không dùng pooling mà dùng convolution stride 2 ở hai lớp đầu. Strided convolution cho phép kernel giảm kích thước đồng thời học phép biến đổi phù hợp với dữ liệu.")
    add_equation(doc, "Y[c,i,j] = max(u,v)∈Ω X[c, iS+u, jS+v]", "(2.10)", "Công thức max-pooling dùng để đối chiếu; model triển khai dùng convolution stride 2.")

    add_heading(doc, "2.2.3. Batch Normalization và hàm kích hoạt", 3)
    add_body(doc, "Batch Normalization chuẩn hóa activation theo mini-batch trong lúc train, sau đó học lại hệ số co γ và dịch β. Nó giúp gradient ổn định hơn và cho phép learning rate hữu dụng lớn hơn [2]. Với mini-batch m phần tử:")
    add_equation(doc, "μB = (1/m)Σᵢxᵢ;   σB² = (1/m)Σᵢ(xᵢ−μB)²", "(2.11)")
    add_equation(doc, "x̂ᵢ = (xᵢ−μB)/√(σB²+ε);   yᵢ = γx̂ᵢ+β", "(2.12)")
    add_body(doc, "Khi inference, mean và variance đã cố định nên BatchNorm có thể gộp vào convolution. Việc folding này loại bỏ một phép toán riêng trên FPGA:")
    add_equation(doc, "W′ = γW/√(σ²+ε);   b′ = β + γ(b−μ)/√(σ²+ε)", "(2.13)")
    add_body(doc, "ReLU đặt giá trị âm về 0 và giữ miền dương, tạo phi tuyến mà không cần phép hàm phức tạp [3]. Sigmoid ánh xạ logit về xác suất (0,1) và được dùng cho objectness cũng như offset tâm hộp trong decoder.")
    add_equation(doc, "ReLU(z) = max(0,z);   sigmoid(z) = 1/(1+e⁻ᶻ)", "(2.14)")

    add_heading(doc, "2.3. Object detection: hộp, confidence, loss và NMS", 2)
    add_heading(doc, "2.3.1. Biểu diễn bounding box", 3)
    add_body(doc, "Một bounding box có thể biểu diễn bởi hai góc (x₁,y₁,x₂,y₂) hoặc tâm, chiều rộng và chiều cao (xc,yc,w,h). Nhãn YOLO chuẩn hóa theo kích thước ảnh W×H, cho phép cùng định dạng dùng ở các độ phân giải khác nhau:")
    add_equation(doc, "xc = (xmin+xmax)/(2W);   yc = (ymin+ymax)/(2H)", "(2.15)")
    add_equation(doc, "w = (xmax−xmin)/W;   h = (ymax−ymin)/H", "(2.16)")
    add_heading(doc, "2.3.2. IoU và hàm mất mát", 3)
    add_body(doc, "Intersection over Union đo mức chồng lấp giữa hộp dự đoán B và hộp thật Bgt. IoU bằng 1 khi hai hộp trùng nhau và bằng 0 khi không giao nhau.")
    add_equation(doc, "IoU(B,Bgt) = |B∩Bgt| / |B∪Bgt|", "(2.17)")
    add_body(doc, "YOLO tối ưu đồng thời phân loại và định vị trong một detector một giai đoạn [4]. Có thể mô tả tổng quát loss của baseline trong nghiên cứu bằng ba thành phần: box loss dựa trên IoU/CIoU [5], Binary Cross Entropy cho lớp hoặc objectness, và Distribution Focal Loss cho phân phối khoảng cách biên hộp [6].")
    add_equation(doc, "LBCE = −[y·ln(p) + (1−y)·ln(1−p)]", "(2.18)")
    add_equation(doc, "LCIoU = 1−IoU + ρ²(b,bgt)/c² + αv", "(2.19)", "ρ là khoảng cách hai tâm; c là đường chéo hộp bao nhỏ nhất; v và α phạt sai khác tỷ lệ hình dạng.")
    add_equation(doc, "LDFL(y) = −[(yr−y)ln Sᵢ + (y−yl)ln Sᵢ₊₁]", "(2.20)", "yl và yr là hai bin lân cận của tọa độ liên tục y.")
    add_equation(doc, "LYOLO = λboxLCIoU + λclsLBCE + λdflLDFL", "(2.21)")
    add_heading(doc, "2.3.3. Confidence threshold và Non-Maximum Suppression", 3)
    add_body(doc, "Sau khi loại các dự đoán có confidence thấp, NMS sắp xếp hộp theo score, giữ hộp tốt nhất rồi loại các hộp cùng lớp có IoU lớn hơn ngưỡng τ. Confidence thấp làm recall tăng nhưng kéo theo false positive; τ NMS quá thấp có thể xóa hai đối tượng sát nhau, còn quá cao dễ giữ hộp trùng.")
    add_picture(doc, assets["detection_geometry"], "Hình 2.3. IoU và nguyên lý Non-Maximum Suppression")

    add_heading(doc, "2.4. Mạng neural lượng tử hóa QNN", 2)
    add_heading(doc, "2.4.1. Lượng tử hóa affine đều", 3)
    add_body(doc, "FP32 dùng 32 bit cho mỗi số thực. Trong QNN, một tensor được biểu diễn bằng số nguyên q, scale s và zero-point z. Phép quantize làm tròn và chặn về miền biểu diễn; dequantize ánh xạ trở lại giá trị gần đúng để mô phỏng sai số lượng tử.")
    add_equation(doc, "q = clip(round(x/s)+z, qmin, qmax)", "(2.22)")
    add_equation(doc, "x̂ = s(q−z)", "(2.23)")
    add_equation(doc, "s = (xmax−xmin)/(qmax−qmin)", "(2.24)")
    add_body(doc, "Với W4 signed, q∈[−8,7]; với A6 unsigned sau QuantReLU, q∈[0,63]. Input UINT8 nằm trong [0,255], còn raw head được requantize về INT8 [−128,127]. Trọng số W4 chỉ cần 1/8 bộ nhớ so với FP32; activation A6 chỉ cần 6/32 = 18,75% số bit. Đổi lại, bước lượng tử lớn hơn tạo sai số làm tròn và clipping.")
    add_heading(doc, "2.4.2. QAT và Straight-Through Estimator", 3)
    add_body(doc, "Post-Training Quantization chỉ hiệu chỉnh scale sau khi model FP32 đã học xong, thường suy giảm mạnh ở 4–6 bit. Quantization-Aware Training chèn fake-quantize vào forward ngay trong train theo nguyên lý integer-only inference [7]. Mạng nhìn thấy x̂ thay vì x và tự điều chỉnh trọng số để giảm loss trong miền số nguyên. Do round có đạo hàm bằng 0 hầu hết mọi nơi, backward dùng Straight-Through Estimator.")
    add_equation(doc, "xQAT = dequantize(quantize(x))", "(2.25)")
    add_equation(doc, "∂round(x)/∂x ≈ 1 trong miền không clipping; 0 ngoài miền", "(2.26)")
    add_body(doc, "Tích W4×A6 tạo tích có độ rộng lớn hơn; tổng nhiều tích phải dùng accumulator đủ rộng. Một cận thiết kế đơn giản cho kernel K×K và Cᵢₙ kênh là:")
    add_equation(doc, "Bacc ≳ Bw + Ba + ⌈log₂(K²Cᵢₙ)⌉", "(2.27)", "Cần thêm biên an toàn tùy signedness, bias và scale; thiết kế hiện tại dùng raw accumulator INT16 trước requantization.")
    add_picture(doc, assets["qnn_quantization"], "Hình 2.4. Fake quantization trong QAT và datapath số nguyên W4A6")

    add_heading(doc, "2.5. Ánh xạ CNN/QNN lên FPGA bằng FINN", 2)
    add_body(doc, "Convolution có thể biến đổi thành phép nhân ma trận–vector. FINN sinh một dataflow accelerator trong đó mỗi lớp là một IP streaming; các lớp làm việc đồng thời trên các mẫu khác nhau sau khi pipeline đầy [8], [11]. QONNX giữ thông tin quantization của từng tensor để các phép Streamline, folding và code generation không làm mất bit-width [10]; model QAT được xây dựng bằng Brevitas [9].")
    add_body(doc, "Trong MVAU của FINN, SIMD cho biết số phần tử đầu vào được nhân song song, còn PE là số neuron/kênh đầu ra tính song song. Với chiều ma trận MW = K²Cᵢₙ và MH = Cₒᵤₜ, số chu kỳ gần đúng cho một vector đầu ra là:")
    add_equation(doc, "Cyclesvector ≈ (MW/SIMD)·(MH/PE)", "(2.28)")
    add_body(doc, "SIMD phải chia hết số kênh hoặc chiều ma trận đầu vào phù hợp; PE phải chia hết số kênh đầu ra. Tăng PE/SIMD giảm folding và tăng throughput nhưng tiêu tốn LUT/BRAM, tăng fan-out và làm routing khó hơn. Bởi vậy mỗi lớp cần được cân bằng để initiation interval lớn nhất không trở thành nút thắt.")
    add_equation(doc, "Throughput ≈ fclk / maxₗ(IIₗ)", "(2.29)")
    add_equation(doc, "FPSdual-ROI = ROI/s ÷ 2", "(2.30)", "Mỗi frame chạy hai lần inference, một lần cho mỗi ROI.")
    add_picture(doc, assets["deploy"], "Hình 2.5. Chuỗi chuyển đổi QNN từ Brevitas/QAT qua QONNX, FINN và Vivado tới bitstream")

    add_heading(doc, "2.6. Kiến trúc QNN được dùng trong nghiên cứu", 2)
    add_picture(doc, assets["qnn_architecture"], "Hình 2.6. Cấu trúc chính xác của TinyQuantDetector W4A6 đầu vào 96×96")
    add_body(doc, "TinyQuantDetector dùng bốn khối Conv–BatchNorm–QuantReLU và một head 1×1. Hai lớp stride 2 đưa 96×96 về lưới 24×24; hai lớp stride 1 tăng chiều sâu đặc trưng mà không làm mất thêm tọa độ. Head 15 kênh gồm 3 slot, mỗi slot có 5 giá trị: objectness logit và bốn tham số hộp. Hai slot dành cho cell vì cell nhỏ, nhiều và có thể xuất hiện gần nhau; một slot dành cho droplet.")
    add_table(doc, ["Lớp", "Output", "Kernel/stride", "Bit-width", "Số weight", "MAC/ROI"], [
        ["Input QuantReLU", "96×96×1", "-", "UINT8", "0", "0"],
        ["Conv1 + BN + QReLU", "48×48×12", "3×3 / 2", "W8/A6", "108", "248.832"],
        ["Conv2 + BN + QReLU", "24×24×16", "3×3 / 2", "W4/A6", "1.728", "995.328"],
        ["Conv3 + BN + QReLU", "24×24×24", "3×3 / 1", "W4/A6", "3.456", "1.990.656"],
        ["Conv4 + BN + QReLU", "24×24×24", "3×3 / 1", "W4/A6", "5.184", "2.985.984"],
        ["Raw head", "24×24×15", "1×1 / 1", "W8/INT8", "360", "207.360"],
        ["Tổng convolution", "-", "-", "-", "10.836", "6.428.160"],
    ])
    add_body(doc, "Một ROI cần xấp xỉ 6,43 triệu MAC theo đếm toán học của các lớp convolution; hai ROI tương đương 12,86 triệu MAC cho mỗi frame. Số này mô tả độ phức tạp thuật toán, không đồng nhất với số chu kỳ FPGA vì accelerator thực hiện nhiều MAC song song và pipeline giữa các lớp.")
    add_body(doc, "Với lưới 24×24, tại ô (gx,gy), decoder biến raw output thành hộp chuẩn hóa. anchor (aw,ah) được học/ước lượng riêng theo lớp; exp bị clip để tránh hộp phát nổ số học:")
    add_equation(doc, "cx = [gx + sigmoid(tx)]/24;   cy = [gy + sigmoid(ty)]/24", "(2.31)")
    add_equation(doc, "w = aw·exp(clip(tw,−3,2));   h = ah·exp(clip(th,−3,2))", "(2.32)")
    add_equation(doc, "B = [cx−w/2, cy−h/2, cx+w/2, cy+h/2]", "(2.33)")
    add_body(doc, "Anchor của cell là (0,07407; 0,07447), anchor droplet là (0,42777; 0,49260) theo tỷ lệ ROI. Ngưỡng confidence vận hành lần lượt 0,96 cho cell và 0,88 cho droplet; NMS IoU 0,35 và 0,10. Đây là tham số hậu xử lý của checkpoint hiện tại, không phải hằng số phổ quát: chúng phải được hiệu chỉnh lại khi camera, ROI hoặc dataset thay đổi.")
    add_heading(doc, "2.7. Liên hệ lý thuyết với bài toán vi lưu", 2)
    add_body(doc, "Bài toán không chỉ là phân loại một ảnh có hay không có vật thể. Hệ thống phải định vị hai lớp có kích thước rất khác nhau, chấp nhận cell nằm trong droplet, duy trì ID qua nhiều frame và đếm đúng một lần. Bởi vậy mỗi khối lý thuyết đảm nhiệm một chức năng cụ thể trong pipeline:")
    add_table(doc, ["Khối lý thuyết", "Vai trò trong bài toán", "Quyết định thiết kế"], [
        ["Convolution", "Học biên vòng droplet và đốm/texture cell", "Kernel 3×3, bốn khối đặc trưng"],
        ["Stride/downsampling", "Giảm chi phí nhưng giữ lưới tọa độ", "96×96 → 24×24, không pooling"],
        ["Detection head", "Định vị cell và droplet cùng lúc", "2 slot cell + 1 slot droplet"],
        ["QAT", "Giảm lỗi khi W4/A6", "Fake quant trong train, integer inference"],
        ["NMS theo lớp", "Loại hộp trùng nhưng giữ cell gần nhau", "Ngưỡng riêng cell/droplet"],
        ["Hai ROI", "Xác nhận cùng một vật thể đi qua kênh", "ROI 1 candidate, ROI 2 confirmation"],
        ["Tracking", "Giữ ID khi box rung hoặc hụt ở biên", "Hit/miss, dự đoán ngắn hạn, consumed ID"],
        ["FPGA dataflow", "Đạt độ trễ xác định", "Folding PE/SIMD, clock 104 MHz"],
    ])
    add_body(doc, "Chuỗi quyết định này thể hiện hardware–algorithm co-design: ROI làm giảm lượng dữ liệu trước CNN; QNN làm giảm chi phí mỗi MAC; dataflow khai thác song song; tracking dùng thời gian và hướng chuyển động để bù cho detector lượng tử. Nếu chỉ tối ưu từng khối riêng lẻ, ví dụ hạ bit-width mà không QAT hoặc tăng confidence mà không dùng ROI 2, hệ thống có thể nhanh hơn nhưng sai lệch đếm tăng.")

    add_chapter_break(doc, "CHƯƠNG 3. DỮ LIỆU VÀ TIỀN XỬ LÝ")
    add_heading(doc, "3.1. Video nguồn và ROI", 2)
    add_picture(doc, assets["source_original"], "Hình 3.1. Frame nguồn 1280×800 từ video kính hiển vi 3.4")
    add_body(doc, "Đối tượng đi theo kênh ngang, vì vậy xử lý toàn frame chứa phần nền lớn không tạo thêm thông tin nhưng làm tăng đáng kể MAC và bộ nhớ. Hai ROI 120×120 được đặt liên tiếp tại (628, 410)–(748, 530) và (748, 410)–(868, 530). Mỗi ROI được resize về 96×96 trước QNN. So với full frame, một ROI chỉ còn 1,406% số pixel; hai ROI còn 2,8125% số pixel nguồn.")
    add_picture(doc, assets["roi_overlay"], "Hình 3.2. Vị trí hai ROI trên frame nguồn")
    add_picture(doc, assets["roi_crops"], "Hình 3.3. Nội dung ROI candidate và ROI confirmation")
    add_heading(doc, "3.2. Dataset và gán nhãn", 2)
    add_table(doc, ["Split", "Số ảnh", "Tỷ lệ"], [["Train", counts["splits"]["train"], "77,78%"], ["Validation", counts["splits"]["valid"], "11,11%"], ["Test", counts["splits"]["test"], "11,11%"]])
    add_body(doc, f"Bản export huấn luyện có {counts['images']} ảnh và {counts['boxes']:,} box: {counts['classes']['cell']:,} cell, {counts['classes']['droplet']:,} droplet. Roboflow từng ghi nhận thêm lớp uncertain dùng cho review; lớp này không có trong data.yaml cuối (nc=2) và không tham gia inference. Tên file còn mã video, frame, tile và biến thể chất lượng, nhờ đó có thể truy vết nguồn và kiểm tra leakage.")
    add_picture(doc, assets["dataset_chart"], "Hình 3.4. Phân bố ảnh và bounding box")
    add_picture(doc, assets["dataset_annotated"], "Hình 3.5. Mẫu gán nhãn cell và droplet trong bản export YOLO")
    add_heading(doc, "3.3. Đa dạng hóa chất lượng", 2)
    add_body(doc, "Các biến thể native, high-light, low-light, soft-blur, resolution-loss và sensor-JPEG được dùng để tăng khả năng chịu thay đổi camera. Augmentation không thay thế dữ liệu mới; nó chỉ tạo nhiễu có kiểm soát quanh phân bố đã quan sát. Các biến đổi hình học mạnh bị hạn chế vì hướng chuyển động và cấu trúc kênh có ý nghĩa vật lý.")
    add_picture(doc, assets["dataset_diversity"], "Hình 3.6. Các biến thể chất lượng ảnh trong dataset")
    add_heading(doc, "3.4. Rủi ro data leakage", 2)
    add_body(doc, "Các frame liên tiếp trong cùng video có tương quan rất cao. Dataset hiện được chia 700/100/100 ở mức ảnh; tên file cho phép kiểm tra theo sequence nhưng chưa chứng minh mọi video đã tách hoàn toàn giữa các split. Vì vậy, kết quả test được dùng như benchmark nội bộ, còn nghiên cứu tiếp theo cần split theo video/sequence để đo khả năng tổng quát hóa nghiêm ngặt hơn.")

    add_chapter_break(doc, "CHƯƠNG 4. PHƯƠNG PHÁP CNN/QNN VÀ THUẬT TOÁN ĐẾM")
    add_heading(doc, "4.1. Các hướng đã khảo sát", 2)
    add_table(doc, ["Phương pháp", "Vai trò", "Ưu điểm", "Giới hạn quan sát được"], [
        ["Threshold / connected component", "Candidate nhanh", "Rất nhẹ", "Nhạy nền và illumination"],
        ["Feature + cây quyết định", "Phân loại patch", "Giải thích được", "Bỏ sót hạt đen, dễ lấy nền"],
        ["YOLO11n", "Baseline/teacher", "Độ chính xác cao", "Khó đưa nguyên model lên FPGA nhỏ"],
        ["QNN W4A6", "Detector triển khai", "Nhẹ, số nguyên", "Cần QAT và hậu xử lý ổn định"],
        ["QNN + temporal tracking", "Đếm sự kiện", "Giảm double count", "Phụ thuộc ghép track và cửa sổ thời gian"],
    ])
    add_heading(doc, "4.2. YOLO11n baseline", 2)
    add_body(doc, "YOLO11n [12] được huấn luyện trên input 640×640 để trả lời câu hỏi đầu tiên: dữ liệu và cách gán nhãn hiện tại có đủ thông tin để phân biệt cell/droplet hay không. Backbone trích xuất đặc trưng đa mức, neck hợp nhất feature map và detection head dự đoán lớp cùng bounding box ở nhiều tỷ lệ. Mô hình cho accuracy tốt nhưng không được xem là kiến trúc đích của XC7Z010 vì số tham số, feature map và phép toán vượt xa ngân sách PL nhỏ.")
    add_body(doc, "Trong nghiên cứu, YOLO có ba vai trò: baseline độ chính xác trên PC; teacher để quan sát mẫu khó và phát hiện lỗi nhãn; mốc tham chiếu trước khi giảm model và bit-width. Không dùng metric YOLO để khẳng định trực tiếp accuracy của QNN FPGA, vì hai kiến trúc và hậu xử lý khác nhau.")
    add_picture(doc, YOLO_RUN / "results.png", "Hình 4.1. Log đồ thị huấn luyện YOLO11n")
    add_picture(doc, assets["yolo_training"], "Hình 4.2. Diễn biến metric và box loss theo epoch")
    add_heading(doc, "4.3. Tiền xử lý ROI cho QNN", 2)
    add_body(doc, "Frame nguồn 1280×800 không được đưa toàn bộ vào CNN. Dựa trên cấu trúc kênh cố định, hệ thống lấy hai cửa sổ 120×120 tại vị trí mà cell/droplet buộc phải đi qua. Mỗi cửa sổ giữ toàn bộ tiết diện kênh và một phần biên để không cắt mất vật thể bám viền, nhưng loại hơn 97% pixel không liên quan của frame.")
    add_equation(doc, "rROI = 2·120·120 / (1280·800) = 0,028125 = 2,8125%", "(4.1)")
    add_body(doc, "ROI màu được đổi sang grayscale rồi resize song tuyến tính về 96×96. Grayscale giảm ba kênh xuống một vì màu vàng chủ yếu do nguồn chiếu sáng; thông tin phân biệt chính nằm ở biên, độ sáng tương đối và texture. Công thức luminance chuẩn được mô tả gần đúng như sau:")
    add_equation(doc, "Igray = 0,299R + 0,587G + 0,114B", "(4.2)")
    add_body(doc, "Sau chuẩn hóa về [0,1], input được lượng tử thành UINT8. Scale input của model export là 0,0037305856; mỗi mức lượng tử tương ứng khoảng 0,373% toàn thang. Resize và quantize phải giống pipeline train/export, nếu khác interpolation, thứ tự kênh hoặc scale thì output FPGA vẫn chạy nhưng dự đoán sai.")
    add_equation(doc, "qin = clip(round(Igray/sin), 0, 255),   sin = 0,0037305856", "(4.3)")

    add_heading(doc, "4.4. QNN W4A6 96×96 và giải mã đầu ra", 2)
    add_body(doc, "QNN cuối sử dụng kiến trúc ở Hình 2.6. Conv1 giữ weight 8 bit để giảm mất thông tin ngay tại lớp tiếp xúc ảnh; ba khối thân dùng W4/A6; head 1×1 dùng weight 8 bit và xuất raw logit INT8. Đây là lượng tử hỗn hợp, không phải toàn bộ model đều 4 bit. Weight 4 bit ở phần thân giảm mạnh bộ nhớ, còn activation 6 bit giữ 64 mức dương để bảo toàn chấm cell nhỏ.")
    add_table(doc, ["Tensor/scale", "Giá trị", "Ý nghĩa"], [
        ["Input scale", "0,0037305856", "UINT8 → giá trị chuẩn hóa"],
        ["Feature scale", "0,0930200219", "Activation nội bộ A6"],
        ["Head-weight scale", "0,0028666302", "Weight head lượng tử"],
        ["Accumulator scale", "0,0002666540", "Raw INT16 accumulator"],
        ["Output scale", "0,0302868001", "Requantize raw head INT8"],
    ])
    add_body(doc, "Lõi FPGA nhận 9.216 byte cho một ROI 96×96×1. Raw internal stream trước requantization có 24×24×15 = 8.640 phần tử INT16; sau tầng requantization, host nhận raw head INT8. Sigmoid, anchor decode, hiệu chỉnh box và NMS được giữ ở phần mềm để accelerator gọn và dễ thay đổi ngưỡng mà không build lại bitstream.")
    add_body(doc, "Decoder trước hết áp sigmoid cho objectness, sau đó chỉ giữ cực đại cục bộ trong lân cận 3×3 của mỗi slot. Cơ chế này ngăn nhiều ô lưới xung quanh cùng một tâm tạo ra hàng loạt box. Sau lọc top-k, công thức (2.31)–(2.33) tạo box; tiếp theo áp giới hạn hình học và NMS riêng cho từng lớp.")
    add_equation(doc, "pobj = sigmoid(o);   keep ⇔ pobj≥τclass ∧ pobj là cực đại cục bộ 3×3", "(4.4)")
    add_table(doc, ["Lớp", "Slot", "Anchor (w,h)", "Conf", "NMS IoU"], [
        ["Cell", "2", "(0,07407; 0,07447)", "0,96", "0,35"],
        ["Droplet", "1", "(0,42777; 0,49260)", "0,88", "0,10"],
    ])
    add_body(doc, "Droplet có NMS IoU thấp vì một giọt thường sinh nhiều hộp gần trùng trên lưới; cell dùng hai slot và NMS cao hơn để giữ hai cell thật ở gần nhau. Width của box droplet được nhân 1,3 trong calibration để bọc sát vòng giọt hơn. Các tham số này được tách khỏi weight, vì vậy có thể hiệu chỉnh bằng validation mà không huấn luyện lại.")
    add_body(doc, "Kiểm thử 16 tensor video giữa bitstream 100 MHz và 104 MHz cho kết quả bit-exact, cùng SHA-256 và checksum CFC48C10. Build 140 MHz trả tensor rỗng/checksum 00000000 nên bị loại khỏi kết quả; báo cáo chỉ sử dụng build 104 MHz đã xác nhận. Bit-exact chứng minh hai build hợp lệ tính cùng hàm số nguyên, nhưng không tự chứng minh accuracy nếu nhãn test không đúng.")

    add_heading(doc, "4.5. Thuật toán tracking và đếm hai ROI", 2)
    add_picture(doc, assets["dual_algorithm"], "Hình 4.3. Luồng candidate–confirmation và tăng bộ đếm một lần")
    add_heading(doc, "4.5.1. Tracking trong từng ROI", 3)
    add_body(doc, "Detection theo từng frame không có định danh. Tracker gán ID bằng cách ghép detection mới với track cũ cùng lớp dựa trên khoảng cách tâm đã chuẩn hóa theo đường chéo ROI. Ngưỡng ghép của cell là 0,24 và droplet là 0,35; droplet được cho phép dịch xa hơn vì kích thước lớn và box có thể thay đổi khi chạm biên.")
    add_equation(doc, "dij = ‖ci(t)−ĉj(t)‖₂ / √(WROI²+HROI²)", "(4.5)")
    add_equation(doc, "match(i,j) ⇔ classi=classj ∧ dij≤τclass", "(4.6)")
    add_body(doc, "Một track chỉ trở thành candidate đủ tin cậy sau số hit tối thiểu: 3 frame cho cell và 2 frame cho droplet. Nếu detector hụt tạm thời, vị trí dự đoán được giữ tối đa 10 frame; max-miss của cell là 8 và droplet là 10. Cơ chế sticky/predicted box giúp không hủy ID chỉ vì vật thể đi sát mép ROI, nhưng predicted box không được tính như một detection mới.")
    add_equation(doc, "ĉ(t) = c(t−1) + v(t−1)", "(4.7)", "Mô hình vận tốc hằng dùng để dự đoán tâm ngắn hạn khi thiếu detection.")
    add_heading(doc, "4.5.2. Ghép sự kiện giữa ROI 1 và ROI 2", 3)
    add_body(doc, "ROI 1 là cổng candidate; ROI 2 là cổng confirmation. Hai ROI không có hai bộ đếm độc lập rồi cộng lại. Một candidate ở ROI 1 được đặt vào hàng chờ; chỉ khi ROI 2 quan sát đúng lớp, đúng hướng trái→phải, sai khác trục ngang kênh đủ nhỏ và nằm trong cửa sổ 4–60 frame thì cặp mới tạo một event.")
    add_equation(doc, "Δt = tROI2−tROI1;   4≤Δt≤60", "(4.8)")
    add_equation(doc, "|yROI2−yROI1|/HROI ≤ 0,25;   xROI2>xROI1", "(4.9)")
    add_equation(doc, "Nclass ← Nclass+1 chỉ khi candidate chưa được dùng và toàn bộ điều kiện ghép đúng", "(4.10)")
    add_body(doc, "Sau khi ghép, ID candidate và ID confirmation đều được đánh dấu consumed để không thể tăng bộ đếm lần thứ hai. Nếu nhiều candidate cùng lớp tồn tại, thuật toán ưu tiên cặp có delay và sai khác vị trí phù hợp nhất. Thiết kế này biến detection theo frame thành event theo chuyển động vật lý, phù hợp mục tiêu đếm vật thể đi qua kênh.")
    add_heading(doc, "4.5.3. Ý nghĩa đối với bài toán cell/droplet", 3)
    add_body(doc, "Droplet là vòng sáng lớn; cell là chấm/texture nhỏ và có thể nằm bên trong droplet. Vì vậy box cell và box droplet được phép chồng lấp, không loại nhau theo lớp. Tracker chạy riêng theo lớp, sau đó bộ ghép hai ROI xác nhận cell và droplet độc lập. Một droplet mang nhiều cell có thể tạo một event droplet và nhiều event cell, đúng với ý nghĩa vật lý thay vì ép hai lớp loại trừ nhau.")
    add_body(doc, "Trong 300 frame, hệ thống phát hành 18 candidate cell và 14 candidate droplet nhưng chỉ xác nhận 8 sự kiện mỗi lớp. Khoảng cách ROI tạo delay 5–48 frame trong các event đã ghép. Chênh lệch giữa candidate và confirmed thể hiện tác dụng của ROI 2 trong việc loại detection đơn lẻ; tuy nhiên chưa có ground truth event-level đầy đủ để coi 16 là độ chính xác đếm tuyệt đối.")

    add_chapter_break(doc, "CHƯƠNG 5. KIẾN TRÚC FPGA")
    add_heading(doc, "5.1. Quá trình chuyển nền tảng", 2)
    add_body(doc, "Arty S7-25 được dùng để chứng minh detector QNN và giao thức sparse-UART. Khi yêu cầu camera và quản lý frame trở nên rõ hơn, dự án chuyển sang Zybo Z7-10. Zynq-7010 bổ sung ARM Processing System và DDR, thuận lợi cho Ethernet, DMA và điều phối PL, dù tài nguyên PL vẫn hạn chế.")
    add_heading(doc, "5.2. Datapath QNN trên Zybo Z7-10", 2)
    add_picture(doc, assets["system_arch"], "Hình 5.1. Kiến trúc hệ thống camera và Zybo")
    add_body(doc, "Trong bài test hiện tại, PC cắt/resize ROI và truyền tensor; PS dùng DMA đưa dữ liệu vào accelerator FINN trong PL. PL thực hiện QNN, output requantization và tạo record sparse. PC nhận record để giải mã box, tracking, overlay và lưu video. Vì vậy kết quả video là suy luận QNN thật trên FPGA, còn I/O camera end-to-end chưa nằm hoàn toàn trên board.")
    add_heading(doc, "5.3. Timing, tài nguyên và công suất", 2)
    r = fpga["resources"]
    t = fpga["timing"]
    add_table(doc, ["Thông số", "Giá trị đo/report", "Kết luận"], [
        ["Clock PL", "104 MHz", "Build được chọn"],
        ["WNS / TNS", f"{t['wns_ns']:+.3f} ns / {t['tns_ns']:.3f} ns", "Setup timing PASS"],
        ["WHS / THS", f"{t['whs_ns']:+.3f} ns / {t['ths_ns']:.3f} ns", "Hold timing PASS"],
        ["LUT", f"{r['slice_luts']:,} / 17.600 ({r['slice_luts_percent']:.2f}%)", "Tài nguyên chi phối"],
        ["FF", f"{r['slice_registers']:,} / 35.200 ({r['slice_registers_percent']:.2f}%)", "Còn dư"],
        ["BRAM", f"{r['bram_tiles']} / 60 ({r['bram_percent']:.2f}%)", "Không phải nút thắt"],
        ["DSP", f"{r['dsps']} / 80 ({r['dsp_percent']:.2f}%)", "Không phải nút thắt"],
        ["Power", f"{r['estimated_total_on_chip_power_w']:.3f} W", "Ước tính Vivado, chưa đo điện ngoài"],
    ])
    add_picture(doc, assets["resource_chart"], "Hình 5.2. Mức sử dụng tài nguyên của build 104 MHz")

    add_chapter_break(doc, "CHƯƠNG 6. THIẾT KẾ THỰC NGHIỆM")
    add_heading(doc, "6.1. Các lớp kiểm thử", 2)
    add_table(doc, ["Lớp", "Đầu vào", "Đầu ra", "Tiêu chí PASS"], [
        ["Dataset", "900 ảnh", "train/valid/test", "Đủ ảnh và nhãn YOLO hợp lệ"],
        ["YOLO PC", "Test 100 ảnh", "P/R/F1/mAP", "Không lỗi, metric được lưu"],
        ["QNN equivalence", "16 tensor video", "Tensor output", "100 và 104 MHz bit-exact"],
        ["FPGA batch", "64 ROI", "Checksum + thời gian", "Checksum khớp, ≥120 ROI/s"],
        ["Video hai ROI", "300 frame video 3.4", "Box, track, event", "Kết quả đúng frame nguồn"],
        ["Camera live", "30 frame SDK", "30 hash", "Không lặp frame, Ethernet ổn định"],
    ])
    add_heading(doc, "6.2. Metric", 2)
    add_body(doc, "Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R). mAP@50 đánh giá detection tại IoU 0,5; mAP@50:95 khắt khe hơn vì trung bình nhiều ngưỡng IoU. Thông lượng hai ROI được tính bằng ROI/s chia 2. Hiệu suất năng lượng ước tính dùng FPS/Power, với lưu ý power lấy từ Vivado chứ chưa phải đồng hồ điện ngoài.")
    add_heading(doc, "6.3. Tách phạm vi thời gian đo", 2)
    add_bullets(doc, [
        "Core/batch FPGA: đo accelerator và DMA, dùng để kết luận khả năng 60 FPS hai ROI.",
        "Synchronous video: mỗi frame chờ hai ROI qua UART, dùng để kiểm tra box đúng frame; không đại diện tốc độ datapath cuối.",
        "Camera SDK: đo khả năng lấy frame live hiện tại trên PC, không phải giới hạn danh định 100 FPS của camera.",
        "Video encoding: tốc độ dựng file phụ thuộc CPU/codec và không phải FPS suy luận PL.",
    ])

    add_chapter_break(doc, "CHƯƠNG 7. KẾT QUẢ VÀ PHÂN TÍCH")
    add_heading(doc, "7.1. Kết quả YOLO11n", 2)
    test = yolo["test"]
    f1 = 2 * test["precision"] * test["recall"] / (test["precision"] + test["recall"])
    add_table(doc, ["Metric", "Validation", "Test"], [
        ["Precision", f"{yolo['valid']['precision']*100:.2f}%", f"{test['precision']*100:.2f}%"],
        ["Recall", f"{yolo['valid']['recall']*100:.2f}%", f"{test['recall']*100:.2f}%"],
        ["F1", f"{2*yolo['valid']['precision']*yolo['valid']['recall']/(yolo['valid']['precision']+yolo['valid']['recall'])*100:.2f}%", f"{f1*100:.2f}%"],
        ["mAP@50", f"{yolo['valid']['map50']*100:.2f}%", f"{test['map50']*100:.2f}%"],
        ["mAP@50:95", f"{yolo['valid']['map50_95']*100:.2f}%", f"{test['map50_95']*100:.2f}%"],
    ])
    add_picture(doc, assets["yolo_metrics"], "Hình 7.1. Metric YOLO11n trên tập test")
    add_picture(doc, YOLO_TEST / "confusion_matrix_normalized.png", "Hình 7.2. Confusion matrix YOLO11n trên tập test")
    add_picture(doc, YOLO_TEST / "val_batch0_pred.jpg", "Hình 7.3. Ví dụ dự đoán YOLO trên tập test")
    add_picture(doc, assets["class_chart"], "Hình 7.4. mAP@50:95 theo lớp")
    add_body(doc, "Recall 95,06% cao hơn precision 89,35%, phù hợp ưu tiên giảm bỏ sót. Tuy nhiên mAP@50:95 của cell chỉ 46,90%, thấp hơn droplet 84,68%. Điều này cho thấy classifier nhận ra cell khá tốt nhưng vị trí/box của cell khó ổn định hơn, đặc biệt khi cell nhỏ, sát viền hoặc nằm trong droplet. Ngưỡng confidence 0,10 tăng nhẹ mAP@50:95 tổng nhưng đánh đổi thêm false positive; ngưỡng vận hành nên hiệu chỉnh theo video và yêu cầu đếm.")
    add_heading(doc, "7.2. QNN detector giai đoạn Arty S7", 2)
    add_picture(doc, assets["arty_accuracy"], "Hình 7.5. Metric detector QNN trên 30 ảnh của bộ test giai đoạn trước")
    add_body(doc, "Trên 30 ảnh của dataset giai đoạn Arty, output QNN FPGA đạt precision 83,0%, recall 79,05%, F1 80,98% và mAP@50 77,73%. Droplet F1 90,32% cao hơn cell F1 75,29%. Vì đây không phải cùng test set 15 µm cuối, số liệu chỉ dùng làm mốc lịch sử, không dùng để tính mức suy giảm trực tiếp từ YOLO sang QNN.")
    add_heading(doc, "7.3. Hiệu năng QNN trên Zybo Z7-10", 2)
    batch = fpga["hardware_benchmark"]
    efficiency = batch["dual_roi_frames_per_second"] / r["estimated_total_on_chip_power_w"]
    add_table(doc, ["Chỉ số", "Giá trị", "Phạm vi đo"], [
        ["Tốc độ 1 ROI", f"{batch['roi_per_second']:.2f} ROI/s", "64 ROI batch, PL + DMA"],
        ["Tốc độ 2 ROI", f"{batch['dual_roi_frames_per_second']:.2f} FPS", "Hai lần inference/frame"],
        ["Latency batch", f"{1000/batch['roi_per_second']:.3f} ms/ROI", "Trung bình batch"],
        ["Latency video", f"{fpga['corrected_video_demo']['mean_qnn_ms_per_roi']:.3f} ms/ROI", "Tensor video thật"],
        ["UART synchronous", f"{fpga['corrected_video_demo']['mean_dual_roi_uart_ms']:.3f} ms/frame", "Vòng kiểm chứng PC-UART"],
        ["FPS/W ước tính", f"{efficiency:.2f} dual-ROI FPS/W", "Dùng power report 1,872 W"],
        ["Checksum", batch["checksum"], "Khớp expected"],
    ])
    add_picture(doc, assets["performance_chart"], "Hình 7.6. Thông lượng và nút thắt theo phạm vi đo")
    add_picture(doc, assets["platform_chart"], "Hình 7.7. Các mốc FPS FPGA đã kiểm chứng")
    add_body(doc, "Việc tăng clock từ 100 lên 104 MHz làm thông lượng hai ROI tăng từ 58,28 lên 60,61 FPS, tương đương 4,0%, gần tuyến tính với tỷ lệ clock. WNS vẫn dương 0,586 ns nên build đạt timing. LUT 66,27% là nguồn lực căng nhất; tăng song song tiếp có rủi ro routing/timing cao hơn, trong khi BRAM và DSP còn dư đáng kể.")
    add_heading(doc, "7.4. Video và bộ đếm hai ROI", 2)
    add_picture(doc, TRACK_DIR / "preview.jpg", "Hình 7.8. Frame kết quả QNN FPGA với hai ROI và tracking")
    add_picture(doc, assets["tracking_chart"], "Hình 7.9. Tiến trình xác nhận và đếm 300 frame")
    add_video_link(doc, "Mở video thực nghiệm chính trên Google Drive", VIDEO_MAIN_URL, "300 frame video 3.4; mỗi frame dùng output QNN của đúng frame nguồn; 8 cell và 8 droplet được xác nhận.")
    add_video_link(doc, "Mở video smoke test hệ thống trên Google Drive", VIDEO_SMOKE_URL, "30 frame dùng cho gói minh chứng camera/FPGA; 1 droplet được hai ROI xác nhận.")
    add_body(doc, "Video lưu ở 30 FPS để quan sát. Tốc độ tạo file 6,23 FPS là wall-clock của vòng đồng bộ UART, giải mã, tracking và encoding; không phủ định thông lượng 60,61 FPS của accelerator batch. Với tích hợp AXI DMA/DDR và overlay không qua UART, kiến trúc có cơ sở đạt 60 FPS; để đạt camera 100 FPS với hai ROI cần ≥200 ROI/s hoặc giảm số lần inference bằng event gating.")
    add_heading(doc, "7.5. Camera Ethernet và trạng thái end-to-end", 2)
    add_picture(doc, SYSTEM_EVIDENCE / "camera" / "phantom_live_frame_000.png", "Hình 7.10. Frame live đọc trực tiếp từ Phantom SDK")
    add_picture(doc, SYSTEM_EVIDENCE / "system_test_metrics.png", "Hình 7.11. Tổng hợp bài test camera, FPGA và tài nguyên")
    add_body(doc, "Camera Phantom VEO 710L tại 192.168.137.189 đạt 10/10 ping, mất gói 0%, link 1 Gbps. Phantom SDK đọc 30/30 frame có hash khác nhau, 1280×800, 48 bit, 11,739 FPS và 68,785 MiB/s trong cấu hình kiểm thử. Đây là bằng chứng camera live tới PC; chưa có receiver Ethernet/lwIP trên Zybo đưa frame trực tiếp vào PL.")

    add_chapter_break(doc, "CHƯƠNG 8. THẢO LUẬN, KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN")
    add_heading(doc, "8.1. Trade-off độ chính xác - tài nguyên", 2)
    add_body(doc, "YOLO11n là baseline chính xác nhưng không phù hợp với PL nhỏ. QNN W4A6 giảm mạnh bit-width và đạt throughput mục tiêu, đổi lại cần hậu xử lý và tracking để ổn định box. Do W8A8/W4A4/W2A2 chưa được đánh giá trên cùng test set cuối và cùng kiến trúc, báo cáo không dựng đường Pareto giả. Kết luận W4A6 là lựa chọn triển khai hiện tại dựa trên build đã hoạt động, không phải tuyên bố W4A6 tối ưu toàn cục.")
    add_heading(doc, "8.2. Các lỗi và bài học kỹ thuật", 2)
    add_table(doc, ["Vấn đề", "Nguyên nhân", "Xử lý đã áp dụng", "Còn lại"], [
        ["ROI lệch/quá lớn", "Chọn theo ảnh tổng thể", "Hiệu chỉnh theo quỹ đạo; 2 ROI 120×120", "Cần calibration tự động"],
        ["Box giật", "Output không đồng bộ frame", "Synchronous exact-frame + tracking", "UART vẫn chậm"],
        ["Vật thể ở biên", "Detector mất vài frame", "Sticky track tối đa 10 frame", "Cần kiểm thử event ground truth"],
        ["Double count", "Hai ROI cộng độc lập", "ROI 2 chỉ xác nhận ROI 1", "Ghép ID khi mật độ cao"],
        ["Build 140 MHz sai", "Tensor rỗng/checksum 0", "Loại build, giữ 104 MHz", "Tối ưu folding khác"],
        ["Camera chưa đi thẳng FPGA", "Thiếu network ingest", "Test camera và FPGA độc lập", "Tích hợp PS Ethernet + DMA"],
    ])
    add_heading(doc, "8.3. Kết luận", 2)
    add_body(doc, "Dự án đã hoàn thành dataset 15 µm, baseline YOLO11n, QNN W4A6, QONNX/FINN/Vivado, bitstream Zybo Z7-10, kiểm thử timing/tài nguyên/power, bài test batch 60,61 FPS hai ROI và video tracking–counting. Điểm mạnh chính không phải đưa một mạng lớn lên FPGA, mà là giảm dữ liệu từ đặc điểm vật lý của hệ vi lưu rồi đồng thiết kế model, datapath và thuật toán đếm.")
    add_body(doc, f"Kết quả tốt nhất hiện tại: YOLO11n test F1 {yolo_f1*100:.2f}% và mAP@50 {yolo_test['map50']*100:.2f}% trên PC; QNN W4A6 trên FPGA đạt 121,23 ROI/s, tương đương 60,61 FPS cho hai ROI, latency 8,248 ms/ROI ở batch, timing PASS và power ước tính 1,872 W. Hai ROI đã tạo 16 sự kiện xác nhận trong video 300 frame. Độ chính xác event-level của QNN 15 µm vẫn cần một test set video gán nhãn độc lập để định lượng.")
    add_heading(doc, "8.4. Hướng phát triển", 2)
    add_bullets(doc, [
        "Tạo ground truth theo event/track cho video 3.4, 3.5 và video mới; báo cáo precision/recall của bộ đếm, không chỉ detector.",
        "Split dataset theo video/sequence để loại leakage.",
        "So sánh FP32, W8A8, W4A6, W4A4 và W2A2 trên cùng test set; lập Pareto F1-LUT-BRAM-FPS.",
        "Thay UART frame transport bằng PS Ethernet/DDR/AXI DMA; chỉ gửi metadata sparse về PC.",
        "Calibration ROI từ frame đầu và giới hạn vùng hợp lệ theo channel mask.",
        "Dùng event gating hoặc một backbone dùng chung cho hai ROI để tiến tới 100 FPS camera.",
        "Đo công suất ngoài board thay vì chỉ dùng ước tính Vivado.",
    ])

    doc.add_page_break()
    add_heading(doc, "TÀI LIỆU THAM KHẢO", 1)
    refs = [
        ("[1] Y. LeCun, L. Bottou, Y. Bengio và P. Haffner, Gradient-Based Learning Applied to Document Recognition, Proceedings of the IEEE, 1998.", "https://bottou.org/papers/lecun-98h"),
        ("[2] S. Ioffe và C. Szegedy, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, ICML, 2015.", "https://proceedings.mlr.press/v37/ioffe15.html"),
        ("[3] X. Glorot, A. Bordes và Y. Bengio, Deep Sparse Rectifier Neural Networks, AISTATS, 2011.", "https://proceedings.mlr.press/v15/glorot11a.html"),
        ("[4] J. Redmon et al., You Only Look Once: Unified, Real-Time Object Detection, CVPR, 2016.", "https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html"),
        ("[5] Z. Zheng et al., Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression, AAAI, 2020.", "https://ojs.aaai.org/index.php/AAAI/article/view/6999"),
        ("[6] X. Li et al., Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection, NeurIPS, 2020.", "https://proceedings.neurips.cc/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html"),
        ("[7] B. Jacob et al., Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference, CVPR, 2018.", "https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html"),
        ("[8] Y. Umuroglu et al., FINN: A Framework for Fast, Scalable Binarized Neural Network Inference, FPGA, 2017.", "https://arxiv.org/abs/1612.07119"),
        ("[9] AMD/Xilinx Research Labs, Brevitas: neural-network quantization library for PyTorch.", "https://github.com/Xilinx/brevitas"),
        ("[10] M. Pappalardo et al., QONNX: Representing Arbitrary-Precision Quantized Neural Networks, 2022.", "https://arxiv.org/abs/2206.07527"),
        ("[11] AMD/Xilinx Research Labs, FINN documentation: network preparation, folding constraints and builder flow.", "https://finn.readthedocs.io/en/latest/nw_prep.html"),
        ("[12] Ultralytics, YOLO11 model and architecture documentation.", "https://docs.ultralytics.com/models/yolo11/"),
        ("[13] AMD, Zynq-7000 SoC Technical Reference Manual and XC7Z010 device documentation.", "https://docs.amd.com/r/en-US/ug585-zynq-7000-SoC-TRM"),
        ("[14] Vision Research, Phantom Camera Control / SDK 3.8 documentation supplied with the experimental system.", None),
        ("[15] Báo cáo JSON/CSV/RPT, source code, bitstream and video thực nghiệm lưu trong E:\\fpga\\final_results, E:\\fpga\\qnn, E:\\fpga\\finn và E:\\fpga\\reports.", None),
    ]
    for ref, url in refs:
        p = doc.add_paragraph()
        p.add_run(ref)
        if url:
            p.add_run(" ")
            add_hyperlink(p, "Nguồn trực tuyến", url)
        p.paragraph_format.left_indent = Cm(0.7)
        p.paragraph_format.first_line_indent = Cm(-0.7)
        p.paragraph_format.space_after = Pt(4)

    doc.add_page_break()
    add_heading(doc, "PHỤ LỤC A. BẢNG TỔNG HỢP KẾT QUẢ", 1)
    add_table(doc, ["Mốc", "Accuracy/F1", "FPS/latency", "Tài nguyên", "Ghi chú"], [
        ["YOLO11n 15 µm", f"F1 {f1*100:.2f}%; mAP50 {test['map50']*100:.2f}%", "PC baseline", "N/A", "Test 100 ảnh"],
        ["QNN Arty giai đoạn trước", f"F1 {arty['overall']['f1']*100:.2f}%; mAP50 {arty['map50']*100:.2f}%", f"{arty['accelerator']['fps_mean']:.2f} FPS core", "Arty S7-25", "Dataset khác, chỉ tham chiếu"],
        ["QNN W4A6 Zybo", "Chưa có event GT cuối", f"{batch['dual_roi_frames_per_second']:.2f} FPS 2 ROI", f"LUT {r['slice_luts_percent']:.2f}%; BRAM {r['bram_percent']:.2f}%", "Bit-exact, timing PASS"],
        ["Video 300 frame", "16 event xác nhận", f"{fpga['corrected_video_demo']['generation_wall_fps']:.2f} FPS dựng file", "Cùng bitstream", "8 cell + 8 droplet"],
        ["Camera SDK", "30/30 frame unique", f"{system['camera_live']['sdk_acquisition_fps']:.2f} FPS", "PC Ethernet", "Chưa nối trực tiếp Zybo"],
    ])
    add_heading(doc, "PHỤ LỤC B. LIÊN KẾT VÀ TỆP MINH CHỨNG", 1)
    p = doc.add_paragraph("Thư mục Google Drive của báo cáo: ")
    add_hyperlink(p, DRIVE_FOLDER_URL, DRIVE_FOLDER_URL)
    add_video_link(doc, "Video QNN FPGA hai ROI 300 frame", VIDEO_MAIN_URL, "Tệp chính để kiểm tra box, tracking và bộ đếm.")
    add_video_link(doc, "Video smoke test 30 frame", VIDEO_SMOKE_URL, "Tệp ngắn cho gói kiểm thử hệ thống.")
    add_bullets(doc, [
        str(FPGA_RESULTS), str(TRACK_REPORT), str(TRACK_EVENTS), str(TRACK_FRAMES), str(SYSTEM_SUMMARY), str(YOLO_METRICS), str(YOLO_RUN / "results.csv"), str(FPGA_ROOT / "artifacts" / "timing_summary.rpt"), str(FPGA_ROOT / "artifacts" / "utilization.rpt"),
    ])
    add_heading(doc, "PHỤ LỤC C. RANH GIỚI TUYÊN BỐ", 1)
    add_table(doc, ["Nội dung", "Trạng thái"], [
        ["YOLO PC trên dataset 15 µm", "Đã đo"],
        ["QNN W4A6 chạy trong PL", "Đã đo và checksum khớp"],
        ["60,61 FPS hai ROI", "Đã đo batch; không bao gồm UART video"],
        ["Video box đúng frame", "Đã kiểm tra bằng vòng synchronous"],
        ["Camera Phantom → PC", "Đã đo bằng SDK"],
        ["Camera Phantom → Zybo → QNN → PC", "Chưa tích hợp end-to-end"],
        ["QNN 15 µm event-level accuracy", "Chưa có ground truth độc lập đầy đủ"],
    ])

    output = OUT / "Bao_cao_tong_ket_FPGA_Cell_Droplet_CNN_QNN_chi_tiet_2026_09_02.docx"
    doc.save(output)
    return output


def write_manifest(output_doc: Path, counts: dict, fpga: dict, yolo: dict) -> None:
    test = yolo["test"]
    f1 = 2 * test["precision"] * test["recall"] / (test["precision"] + test["recall"])
    manifest = {
        "report": str(output_doc),
        "spec": str(SPEC),
        "dataset": counts,
        "yolo_test": {"precision": test["precision"], "recall": test["recall"], "f1": f1, "map50": test["map50"], "map50_95": test["map50_95"]},
        "fpga": fpga,
        "drive": {
            "folder": DRIVE_FOLDER_URL,
            "google_doc_detailed": GOOGLE_DOC_DETAILED_URL,
            "word_detailed": WORD_DETAILED_URL,
            "pdf_detailed": PDF_DETAILED_URL,
            "main_video": VIDEO_MAIN_URL,
            "smoke_video": VIDEO_SMOKE_URL,
        },
        "claim_boundary": "Camera-to-PC and PC-ROI-to-Zybo-PL are validated separately; direct Camera-to-Zybo-to-QNN-to-PC is not implemented.",
    }
    (OUT / "report_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Báo cáo tổng kết FPGA Cell/Droplet\n\n"
        f"- Word: `{output_doc.name}`\n"
        f"- Google Docs chi tiết: {GOOGLE_DOC_DETAILED_URL}\n"
        f"- Word trên Drive: {WORD_DETAILED_URL}\n"
        f"- PDF trên Drive: {PDF_DETAILED_URL}\n"
        f"- Thư mục Google Drive: {DRIVE_FOLDER_URL}\n"
        f"- Video chính: {VIDEO_MAIN_URL}\n"
        f"- Video smoke: {VIDEO_SMOKE_URL}\n\n"
        "Báo cáo chỉ sử dụng số liệu có nguồn JSON/CSV/RPT. Direct camera Ethernet -> Zybo -> QNN -> PC chưa được tuyên bố hoàn thành.\n",
        encoding="utf-8",
    )


def validate_docx(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        media = [n for n in names if n.startswith("word/media/")]
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    required = ["CHƯƠNG 1", "CHƯƠNG 4", "CHƯƠNG 7", "KẾT LUẬN", "Google Drive", "60,61"]
    missing = [x for x in required if x not in xml]
    result = {"docx": str(path), "size_bytes": path.stat().st_size, "embedded_media": len(media), "missing_markers": missing, "valid": not missing and len(media) >= 15}
    (OUT / "validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    counts = count_dataset()
    yolo = load_json(YOLO_METRICS)
    fpga = load_json(FPGA_RESULTS)
    system = load_json(SYSTEM_SUMMARY)
    arty = load_json(ARTY_ACCURACY)
    pc_qnn = load_json(PC_QNN_REPORT)

    source = make_source_and_roi_images()
    dataset = make_dataset_images()
    diagrams = create_algorithm_diagrams()
    theory = create_cnn_theory_diagrams()
    assets = {
        "source_original": source["original"], "roi_overlay": source["overlay"], "roi_crops": source["crops"],
        "dataset_annotated": dataset["annotated"], "dataset_diversity": dataset["diversity"],
        "dataset_chart": chart_dataset(counts), "yolo_metrics": chart_yolo_metrics(yolo), "yolo_training": chart_yolo_training(),
        "class_chart": chart_class_localization(yolo), "resource_chart": chart_fpga_resources(fpga),
        "performance_chart": chart_performance(fpga, system, pc_qnn), "platform_chart": chart_platform_comparison(fpga, arty),
        "tracking_chart": chart_tracking_events(), "arty_accuracy": chart_fpga_accuracy(arty),
        "codesign": diagrams["research"], "deploy": diagrams["deploy"], "dual_algorithm": diagrams["dual"], "system_arch": diagrams["system"],
        "cnn_neuron": theory["neuron"], "cnn_convolution": theory["convolution"],
        "qnn_architecture": theory["architecture"], "qnn_quantization": theory["quantization"],
        "detection_geometry": theory["detection"],
    }
    output = make_report(assets, counts, yolo, fpga, system, arty, pc_qnn)
    write_manifest(output, counts, fpga, yolo)
    validation = validate_docx(output)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
