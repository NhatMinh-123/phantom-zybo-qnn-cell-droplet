#!/usr/bin/env python3
"""Package the one-droplet microplastic experiments into one review folder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_if_present(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build_rows(summaries: list[tuple[str, str, dict]]) -> list[dict]:
    rows: list[dict] = []
    for key, display_name, summary in summaries:
        timing = summary["timing"]
        observations = summary["observations"]
        qnn = summary.get("qnn_classifier", {})
        rows.append(
            {
                "key": key,
                "configuration": display_name,
                "mean_latency_ms": timing["algorithm_mean_ms"],
                "p95_latency_ms": timing["algorithm_p95_ms"],
                "p99_latency_ms": timing["algorithm_p99_ms"],
                "mean_fps": timing["algorithm_mean_fps"],
                "p95_latency_fps": timing["algorithm_p95_latency_fps"],
                "droplet_sequences": observations["droplet_sequences"],
                "classical_confirmed_tracks": observations[
                    "particle_tracks_confirmed"
                ],
                "qnn_patches_evaluated": qnn.get("candidate_patches_evaluated", 0),
                "qnn_patches_accepted": qnn.get("candidate_patches_accepted", 0),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_comparison_plot(path: Path, rows: list[dict], patch_summary: dict) -> None:
    labels = [row["configuration"] for row in rows]
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "One-droplet microplastic pipeline: speed and bootstrap QNN quality",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.bar(
        x - width / 2,
        [row["mean_latency_ms"] for row in rows],
        width,
        label="Mean",
        color="#13795b",
    )
    ax.bar(
        x + width / 2,
        [row["p95_latency_ms"] for row in rows],
        width,
        label="P95",
        color="#f59f00",
    )
    ax.axhline(10.0, color="#c92a2a", linestyle="--", label="100 FPS limit")
    ax.set_ylabel("Latency (ms/frame)")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_title("Algorithm latency, lower is better")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    ax.bar(
        x - width / 2,
        [row["mean_fps"] for row in rows],
        width,
        label="From mean latency",
        color="#1971c2",
    )
    ax.bar(
        x + width / 2,
        [row["p95_latency_fps"] for row in rows],
        width,
        label="From P95 latency",
        color="#845ef7",
    )
    ax.axhline(100.0, color="#c92a2a", linestyle="--", label="Target")
    ax.set_ylabel("Algorithm FPS")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_title("Throughput estimate, higher is better")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    test = patch_summary["test"]
    metric_names = ["Precision", "Recall", "F1", "Accuracy", "Specificity"]
    metric_values = [
        test["precision"],
        test["recall"],
        test["f1"],
        test["accuracy"],
        test["specificity"],
    ]
    ax = axes[1, 0]
    bars = ax.bar(metric_names, metric_values, color="#12b886")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Tiny QNN on grouped bootstrap patch test")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, metric_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value * 100:.1f}%",
            ha="center",
            fontsize=9,
        )

    ax = axes[1, 1]
    phases = [
        "Full frame",
        "Acquisition ROI",
        "Processing ROI",
        "QNN patch",
    ]
    pixels = [1280 * 800, 180 * 260, 128 * 128, 32 * 32]
    bars = ax.bar(phases, pixels, color=["#495057", "#228be6", "#15aabf", "#e8590c"])
    ax.set_yscale("log")
    ax.set_ylabel("Pixels per image, log scale")
    ax.set_title("Spatial reduction before inference")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, pixels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.15,
            f"{value:,}",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_mermaid(path: Path) -> None:
    path.write_text(
        """flowchart LR
    A[Camera or video] --> B[Acquisition gate 180x260]
    B --> C[Background subtraction and droplet localization]
    C --> D[One-droplet crop 128x128]
    D --> E[Central core mask]
    E --> F[Black-hat and morphology]
    F --> G[Blob candidates]
    G --> H[Temporal min-hits gate]
    H --> I[Candidate patch 32x32]
    I --> J[Tiny QNN W4A6]
    J --> K[Track and count]
    K --> L[UART or display]
""",
        encoding="utf-8",
    )


def write_report(
    path: Path,
    rows: list[dict],
    patch_summary: dict,
    review_summary: dict,
    preserved_models: list[dict],
) -> None:
    row_map = {row["key"]: row for row in rows}
    broad = row_map["classical_broad"]
    small = row_map["classical_small"]
    direct = row_map["qnn_direct"]
    event = row_map["qnn_event_cpu"]
    test = patch_summary["test"]

    model_lines = "\n".join(
        f"- `{model['path']}`  \n  SHA-256: `{model['sha256']}`"
        for model in preserved_models
    )

    report = f"""# Báo cáo thử nghiệm nhận diện vi hạt trong một giọt

## Mục tiêu

Nhánh thử nghiệm này giữ nguyên các model hiện có và bổ sung pipeline lai để xử
lý vi hạt rất nhỏ, di chuyển nhanh trong kênh. Mỗi thời điểm chỉ xử lý một giọt:
ROI thu nhận `180 x 260`, vùng xử lý giọt `128 x 128`, và patch QNN `32 x 32`.

![So sánh benchmark](benchmark_comparison.png)

## Kết quả tốc độ

| Cấu hình | Mean ms | P95 ms | Mean FPS | FPS theo P95 |
|---|---:|---:|---:|---:|
| Classical, ROI 310x350 | {broad['mean_latency_ms']:.3f} | {broad['p95_latency_ms']:.3f} | {broad['mean_fps']:.1f} | {broad['p95_latency_fps']:.1f} |
| Classical, ROI 180x260 | {small['mean_latency_ms']:.3f} | {small['p95_latency_ms']:.3f} | {small['mean_fps']:.1f} | {small['p95_latency_fps']:.1f} |
| QNN trên mọi blob | {direct['mean_latency_ms']:.3f} | {direct['p95_latency_ms']:.3f} | {direct['mean_fps']:.1f} | {direct['p95_latency_fps']:.1f} |
| Temporal gate + QNN CPU | {event['mean_latency_ms']:.3f} | {event['p95_latency_ms']:.3f} | {event['mean_fps']:.1f} | {event['p95_latency_fps']:.1f} |

Cấu hình `180 x 260 -> 128 x 128` đạt mục tiêu 100 FPS ở cả trung bình và P95.
Chỉ `16.384` pixel được xử lý thay cho `1.024.000` pixel của toàn frame, giảm
`98,4%`. QNN không nên chạy trên mọi blob; temporal gate giảm số patch QNN từ
`{direct['qnn_patches_evaluated']}` xuống `{event['qnn_patches_evaluated']}` trên
cùng 300 frame.

Video nguồn có metadata `30 FPS`; vì vậy video xuất ra chỉ chứng minh chất lượng
hiển thị ở 30 FPS. Các con số trên đo riêng thời gian thuật toán và cho biết
pipeline có đủ ngân sách tính toán cho luồng camera 100 FPS.

## Kết quả Tiny QNN

- Kiến trúc: `TinyQuantPatchClassifier`, W4A6, đầu vào grayscale `32 x 32`.
- Số tham số: `{patch_summary['parameters']['total']:,}`.
- Test bootstrap: precision `{test['precision'] * 100:.2f}%`, recall
  `{test['recall'] * 100:.2f}%`, F1 `{test['f1'] * 100:.2f}%`, accuracy
  `{test['accuracy'] * 100:.2f}%`.
- Ngưỡng validation đã chọn: `{test['threshold']:.2f}`.

Các metric trên thuộc bộ patch bootstrap lấy từ nhãn `cell` cũ. Khi chạy trên
video vi hạt mới, QNN chấp nhận `0/{event['qnn_patches_evaluated']}` candidate.
Đây là domain shift, không phải bằng chứng video không có vi hạt. Không được hạ
ngưỡng QNN tùy ý để che vấn đề này.

## Độ chính xác end-to-end

Số track từ thuật toán cổ điển là candidate, không phải ground truth. Hiện chưa
có tập test video được duyệt thủ công nên chưa thể báo precision/recall chính
xác cho toàn pipeline.

Đã tạo hàng đợi active learning gồm `{review_summary['total_patches']}` patch:
`{review_summary['likely_particle']}` candidate ưu tiên và
`{review_summary['uncertain']}` candidate chưa chắc chắn. Cần điền
`reviewed_label` bằng `particle`, `background` hoặc `ignore` trước khi train lại.

## Pipeline đề xuất

```mermaid
flowchart LR
    A[Camera or video] --> B[Acquisition gate 180x260]
    B --> C[Background subtraction and droplet localization]
    C --> D[One-droplet crop 128x128]
    D --> E[Central core mask]
    E --> F[Black-hat and morphology]
    F --> G[Blob candidates]
    G --> H[Temporal min-hits gate]
    H --> I[Candidate patch 32x32]
    I --> J[Tiny QNN W4A6]
    J --> K[Track and count]
    K --> L[UART or display]
```

## Bước tiếp theo để tăng độ chính xác

1. Duyệt tối thiểu 300-500 patch mới, ưu tiên false positive, vật thể mờ và hạt
   sát biên; không dùng nhãn gợi ý làm ground truth.
2. Chia train/valid/test theo chuỗi giọt hoặc video, không chia ngẫu nhiên từng
   frame liên tiếp để tránh rò rỉ dữ liệu.
3. Train tiếp Tiny QNN W4A6 từ checkpoint hiện tại, sau đó hiệu chỉnh threshold
   trên validation mới với ưu tiên recall.
4. Đánh giá end-to-end trên video độc lập bằng TP/FP/FN theo track.
5. Chỉ khi đạt accuracy yêu cầu mới xuất trọng số integer và RTL cho Arty S7-25.

## Bảo toàn model

{model_lines}

Không checkpoint nào ở trên bị ghi đè trong thử nghiệm này.
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "final_results" / "microplastic_one_droplet_research_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    result_root = ROOT / "final_results"
    sources = [
        (
            "classical_broad",
            "Classical 310x350",
            result_root
            / "microplastic_hybrid_one_droplet_fpga_fast_benchmark"
            / "summary.json",
        ),
        (
            "classical_small",
            "Classical 180x260",
            result_root
            / "microplastic_hybrid_gate180x260_benchmark"
            / "summary.json",
        ),
        (
            "qnn_direct",
            "QNN every blob",
            result_root
            / "microplastic_hybrid_qnn_gate180x260_benchmark"
            / "summary.json",
        ),
        (
            "qnn_event_cpu",
            "Temporal gate + QNN",
            result_root
            / "microplastic_event_gated_qnn_cpu_gate180x260_benchmark"
            / "summary.json",
        ),
    ]
    loaded = [(key, name, load_json(path)) for key, name, path in sources]
    rows = build_rows(loaded)

    patch_model_dir = ROOT / "models" / "qnn_microplastic_patch32_w4a6_v1"
    patch_summary = load_json(patch_model_dir / "summary.json")
    review_dir = (
        result_root
        / "microplastic_active_learning_v1"
        / "review_package"
    )
    review_summary = load_json(review_dir / "review_summary.json")

    preserved_paths = [
        ROOT
        / "models"
        / "qnn_cell_droplet_v2_w4a6_square192_grouped"
        / "best.pt",
        ROOT
        / "models"
        / "qnn_cell_droplet_v2_w4a6_square192_quality_aug_v1_rc1"
        / "best.pt",
        patch_model_dir / "best.pt",
    ]
    preserved_models = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in preserved_paths
    ]

    write_csv(output / "benchmark_comparison.csv", rows)
    make_comparison_plot(
        output / "benchmark_comparison.png",
        rows,
        patch_summary,
    )
    write_mermaid(output / "pipeline.mmd")
    write_report(
        output / "experiment_summary.md",
        rows,
        patch_summary,
        review_summary,
        preserved_models,
    )

    (output / "model_preservation.json").write_text(
        json.dumps(preserved_models, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "experiment_data.json").write_text(
        json.dumps(
            {
                "benchmarks": rows,
                "patch_qnn": patch_summary,
                "active_learning": review_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    copy_if_present(
        patch_model_dir / "training_curves.png",
        output / "qnn_training_curves.png",
    )
    copy_if_present(
        patch_model_dir / "test_confusion_matrix.png",
        output / "qnn_test_confusion_matrix.png",
    )
    copy_if_present(
        review_dir / "contact_sheet_001.jpg",
        output / "active_learning_contact_sheet_001.jpg",
    )
    copy_if_present(
        result_root
        / "microplastic_hybrid_gate180x260_preview"
        / "contact_sheet.jpg",
        output / "one_droplet_preview_contact_sheet.jpg",
    )

    print(f"Packaged report: {output}")
    print(f"Preserved models: {len(preserved_models)}")


if __name__ == "__main__":
    main()
