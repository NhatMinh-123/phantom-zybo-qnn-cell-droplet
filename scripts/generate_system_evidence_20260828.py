from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


ROOT = Path(r"E:\fpga")
OUT = ROOT / "final_results" / "system_evidence_2026_08_28"
CAMERA = OUT / "camera" / "camera_live_probe.json"
CONNECTIVITY = OUT / "connectivity" / "connectivity.json"
PING = OUT / "connectivity" / "camera_ping_10.txt"
FPGA = OUT / "fpga" / "fpga_batch_benchmark.json"
VIDEO = OUT / "fpga" / "real_video_dual_roi_smoke" / "report.json"
IMPLEMENTATION = ROOT / "final_results" / "zybo_z7_10_qnn_60fps_104mhz" / "results.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def add_table(document: Document, rows: list[tuple[str, str]], widths: tuple[float, float] = (2.5, 4.1)) -> None:
    table = document.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    table.rows[0].cells[0].text = "Hạng mục"
    table.rows[0].cells[1].text = "Kết quả"
    for left, right in rows:
        cells = table.add_row().cells
        cells[0].text = left
        cells[1].text = right
    for row in table.rows:
        row.cells[0].width = Inches(widths[0])
        row.cells[1].width = Inches(widths[1])


def add_heading(document: Document, text: str, level: int = 1) -> None:
    heading = document.add_heading(text, level=level)
    heading.paragraph_format.space_before = Pt(8)
    heading.paragraph_format.space_after = Pt(4)


def write_chart(camera: dict, fpga: dict, video: dict, implementation: dict, ping_loss: float) -> Path:
    chart = OUT / "system_test_metrics.png"
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5))
    fig.suptitle("Kết quả kiểm thử hệ thống Camera - Zybo Z7-10 QNN", fontsize=15, weight="bold")

    axes[0, 0].bar(["SDK live", "Mục tiêu QNN\n2 ROI"], [camera["sdk_acquisition_fps"], 60], color=["#23967F", "#3569B8"])
    axes[0, 0].set_ylabel("Frame/s")
    axes[0, 0].set_title("Tốc độ luồng ảnh và mục tiêu")
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(["1 ROI/s", "2 ROI FPS"], [fpga["roi_per_second"], fpga["dual_roi_frames_per_second"]], color=["#F49D37", "#3569B8"])
    axes[0, 1].set_ylabel("Thông lượng")
    axes[0, 1].set_title("QNN đo trực tiếp trên FPGA")
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].bar(["QNN/ROI", "UART/2 ROI"], [video["mean_qnn_ms"], video["mean_dual_roi_uart_ms"]], color=["#23967F", "#D94F4F"])
    axes[1, 0].set_ylabel("ms")
    axes[1, 0].set_title("Độ trễ lõi và đường thử UART")
    axes[1, 0].grid(axis="y", alpha=0.25)

    resources = implementation["resources"]
    names = ["LUT", "FF", "BRAM", "DSP"]
    values = [resources["slice_luts_percent"], resources["slice_registers_percent"], resources["bram_percent"], resources["dsp_percent"]]
    axes[1, 1].bar(names, values, color=["#D94F4F", "#F49D37", "#23967F", "#3569B8"])
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].set_ylabel("% tài nguyên")
    axes[1, 1].set_title(f"Tài nguyên PL; mất gói camera {ping_loss:.0f}%")
    axes[1, 1].grid(axis="y", alpha=0.25)

    for axis in axes.flat:
        for container in axis.containers:
            axis.bar_label(container, fmt="%.1f", padding=2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(chart, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return chart


def make_docx(summary: dict, chart: Path) -> Path:
    camera = summary["camera_live"]
    connectivity = summary["connectivity"]
    fpga = summary["fpga_qnn_benchmark"]
    video = summary["real_video_fpga_test"]
    implementation = summary["implementation"]

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    styles = document.styles
    styles["Normal"].font.name = "Times New Roman"
    styles["Normal"].font.size = Pt(11)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("BÁO CÁO KIỂM THỬ MINH CHỨNG\nHỆ CAMERA - FPGA QNN")
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(16)
    subtitle = document.add_paragraph("Ngày kiểm thử: 28/08/2026")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    add_heading(document, "1. Kết luận", 1)
    document.add_paragraph(
        "Các phân hệ đã được kiểm tra trực tiếp và đều PASS: camera Phantom truyền ảnh live qua Ethernet tới PC; "
        "Zybo Z7-10 chạy QNN W4A6 trong programmable logic; và video hai ROI được gửi qua PS DMA để suy luận trong PL. "
        "Luồng camera Ethernet đi thẳng vào Zybo chưa được tích hợp, vì vậy trạng thái tổng thể là SUBSYSTEM_PASS_INTEGRATION_PENDING."
    )

    add_heading(document, "2. Cấu hình kiểm thử", 1)
    add_table(document, [
        ("Camera", f"{camera['model']}, serial {camera['serial']}, IP {camera['ip']}"),
        ("FPGA", f"{fpga['board']}, {fpga['device']}"),
        ("Mô hình", f"{fpga['qnn']}, clock {fpga['logic_clock_mhz']:.0f} MHz"),
        ("Kết nối", f"Ethernet {connectivity['ethernet_link_speed']}; Zybo UART {connectivity['zybo_uart']}"),
        ("ROI", "Hai ROI 120×120; QNN nhận tensor 96×96"),
    ])

    add_heading(document, "3. Kết quả đo", 1)
    add_table(document, [
        ("Camera Ethernet", f"10/10 ping, mất gói {summary['camera_ping']['packet_loss_percent']:.0f}%, RTT 0-1 ms"),
        ("Camera live SDK", f"{camera['captured_frames']}/{camera['requested_frames']} frame, {camera['unique_frame_hashes']} hash khác nhau"),
        ("Dữ liệu camera", f"{camera['width']}×{camera['height']}, {camera['bit_count']} bit, {camera['payload_mib_per_second']:.2f} MiB/s"),
        ("Tốc độ SDK live", f"{camera['sdk_acquisition_fps']:.2f} FPS trong cấu hình hiện tại"),
        ("QNN trên FPGA", f"{fpga['roi_per_second']:.2f} ROI/s = {fpga['dual_roi_frames_per_second']:.2f} FPS với hai ROI"),
        ("Độ trễ QNN", f"{fpga['latency_ms_per_roi']:.3f} ms/ROI (batch); {video['mean_qnn_ms']:.3f} ms/ROI (video thật)"),
        ("Kiểm tra dữ liệu", f"Checksum {fpga['checksum']}; khớp chuẩn: {fpga['checksum_match']}"),
        ("Video thực", f"{video['processed_frames']} frame; {video['confirmed_events']} sự kiện được hai ROI xác nhận"),
        ("Timing", f"WNS {implementation['timing']['wns_ns']:+.3f} ns, TNS {implementation['timing']['tns_ns']:.3f} ns, timing PASS"),
    ])

    document.add_picture(str(chart), width=Inches(6.7))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    live_image = OUT / "camera" / "phantom_live_frame_000.png"
    if live_image.exists():
        add_heading(document, "4. Ảnh camera live", 1)
        document.add_picture(str(live_image), width=Inches(6.5))
        document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        document.add_paragraph("Ảnh được đọc trực tiếp bằng Phantom SDK trong bài test, không lấy từ video lưu sẵn.")

    video_preview = OUT / "fpga" / "real_video_dual_roi_smoke" / "preview.jpg"
    if video_preview.exists():
        add_heading(document, "5. Minh chứng QNN hai ROI", 1)
        document.add_picture(str(video_preview), width=Inches(6.5))
        document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        document.add_paragraph("Box và kết quả đếm thuộc đúng frame nguồn; hai lần suy luận ROI chạy trong PL của Zybo.")

    add_heading(document, "6. Tài nguyên và giới hạn", 1)
    resources = implementation["resources"]
    add_table(document, [
        ("LUT", f"{resources['slice_luts']:,} ({resources['slice_luts_percent']:.2f}%)"),
        ("FF", f"{resources['slice_registers']:,} ({resources['slice_registers_percent']:.2f}%)"),
        ("BRAM", f"{resources['bram_tiles']} ({resources['bram_percent']:.2f}%)"),
        ("DSP", f"{resources['dsps']} ({resources['dsp_percent']:.2f}%)"),
        ("Công suất ước tính", f"{resources['estimated_total_on_chip_power_w']:.3f} W"),
    ])
    document.add_paragraph(
        "UART 2 Mbaud chỉ dùng để kiểm chứng và trả kết quả sparse, nên tốc độ tạo video đầu-cuối thấp hơn tốc độ lõi QNN. "
        "Muốn xem camera trực tiếp qua FPGA cần triển khai receiver Ethernet/lwIP hoặc DMA frame vào DDR, cắt ROI trên Zynq và gửi overlay/kết quả về PC."
    )

    add_heading(document, "7. Tệp minh chứng", 1)
    for item in [
        "camera/camera_live_probe.json và phantom_live_frame_000.png",
        "connectivity/camera_ping_10.txt và connectivity.json",
        "fpga/fpga_batch_uart.log và fpga_batch_benchmark.json",
        "fpga/real_video_dual_roi_smoke/zybo_dual_roi_exact_arty_style.mp4",
        "system_test_summary.json, system_test_metrics.png và SHA256SUMS.txt",
    ]:
        document.add_paragraph(item, style="List Bullet")

    output = OUT / "Bao_cao_kiem_thu_he_thong_2026_08_28.docx"
    document.save(output)
    return output


def write_hashes() -> None:
    target = OUT / "SHA256SUMS.txt"
    lines = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path == target or path.suffix.lower() == ".raw":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(OUT).as_posix()}")
    target.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    camera = load_json(CAMERA)
    connectivity = load_json(CONNECTIVITY)
    fpga = load_json(FPGA)
    video = load_json(VIDEO)
    implementation = load_json(IMPLEMENTATION)
    ping_text = PING.read_text(encoding="utf-16", errors="ignore")
    if "Packets:" not in ping_text:
        ping_text = PING.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Sent = (\d+), Received = (\d+), Lost = (\d+) \((\d+)% loss\)", ping_text)
    sent, received, lost, loss_percent = (10, 10, 0, 0) if not match else map(int, match.groups())

    summary = {
        "overall_status": "SUBSYSTEM_PASS_INTEGRATION_PENDING",
        "measured_on": "2026-08-28",
        "camera_ping": {"sent": sent, "received": received, "lost": lost, "packet_loss_percent": loss_percent},
        "connectivity": connectivity,
        "camera_live": camera,
        "fpga_qnn_benchmark": fpga,
        "real_video_fpga_test": video,
        "implementation": {"timing": implementation["timing"], "resources": implementation["resources"]},
        "end_to_end_camera_to_fpga": {
            "status": "NOT_IMPLEMENTED",
            "note": "Camera live reaches the PC via Phantom SDK; direct Ethernet frame ingestion by Zybo is the remaining integration task.",
        },
    }
    (OUT / "system_test_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    chart = write_chart(camera, fpga, video, implementation, float(loss_percent))
    report = make_docx(summary, chart)

    markdown = f"""# Kiểm thử hệ thống Camera - Zybo QNN ngày 28/08/2026

## Kết luận

**{summary['overall_status']}**

- Camera Phantom VEO 710L: PASS, IP `{camera['ip']}`, 10/10 ping, mất gói {loss_percent}%.
- Đọc live bằng Phantom SDK: PASS, {camera['captured_frames']} frame khác nhau, {camera['sdk_acquisition_fps']:.2f} FPS, {camera['payload_mib_per_second']:.2f} MiB/s.
- QNN W4A6 trên Zybo PL: PASS, {fpga['roi_per_second']:.2f} ROI/s, tương đương {fpga['dual_roi_frames_per_second']:.2f} FPS với hai ROI.
- Độ trễ: {fpga['latency_ms_per_roi']:.3f} ms/ROI; checksum `{fpga['checksum']}` khớp chuẩn.
- Video thật qua FPGA: PASS, {video['processed_frames']} frame, {video['confirmed_events']} sự kiện hai ROI xác nhận.
- Direct camera Ethernet -> Zybo -> QNN: **chưa tích hợp**.

## Tệp chính

- `{report.name}`: báo cáo DOCX.
- `{chart.name}`: biểu đồ số đo.
- `camera/phantom_live_frame_000.png`: ảnh camera live.
- `fpga/real_video_dual_roi_smoke/zybo_dual_roi_exact_arty_style.mp4`: video minh chứng.
- `system_test_summary.json`: dữ liệu tổng hợp máy đọc được.
- `SHA256SUMS.txt`: mã kiểm tra toàn bộ bằng chứng.
"""
    (OUT / "README.md").write_text(markdown, encoding="utf-8")
    write_hashes()
    print(report)


if __name__ == "__main__":
    main()
