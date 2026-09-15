"""Measure real Ethernet/PL inference using labeled-video-era ROI geometry."""
import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qnn.fpga_io import load_manifest, prepare_image, pack_input_axis, decode_output_tensor
from scripts.run_phantom_zybo_qnn_live import sparse_tensor, scaled_roi, map_detections
from scripts.zybo_qnn_udp_protocol import ZyboQnnUdpClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--save-frames', action='store_true')
    parser.add_argument('--video', type=Path, default=ROOT / 'data/raw/09_07_2026/09_07_2026/3.4.mp4')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/ethernet_live_20260910/video_benchmark')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(ROOT / 'exports/15micro_qnn_w4a6_96_roi120_v1/fpga_manifest_raw_core.json')
    config = json.loads((ROOT / 'configs/15micro_dual_qnn_fpga.json').read_text())
    classes = manifest['postprocessing']['decoder']['class_names']
    cap = cv2.VideoCapture(str(args.video))
    rows, times, images, repeated = [], [], [], {}
    reference_payloads = []
    try:
        with ZyboQnnUdpClient(timeout=1) as client:
            hello = client.hello()
            # Warm-up before the timed run.
            client.infer(0, 0, bytes(9216))
            started = time.perf_counter()
            for index in range(args.frames):
                frame_start = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                output = frame.copy()
                for roi_id, key in enumerate(('left', 'right')):
                    roi = scaled_roi(width, height, config['rois'][key])
                    x1, y1, x2, y2 = roi
                    payload = pack_input_axis(prepare_image(frame[y1:y2, x1:x2], manifest, array_is_bgr=True), manifest)
                    response = client.infer(index + 1, roi_id, payload)
                    if index < 4:
                        reference_payloads.append((payload, response.records))
                    decoded = decode_output_tensor(sparse_tensor(response.records, manifest), manifest)[0]
                    detections = map_detections(decoded, classes, roi)
                    row = dict(frame=index, roi=roi_id, qnn_us=response.qnn_us,
                               roundtrip_ms=response.round_trip_seconds * 1000,
                               record_count=response.record_count,
                               detections=len(detections),
                               droplet=sum(d.class_name == 'droplet' for d in detections),
                               cell=sum(d.class_name == 'cell' for d in detections),
                               output_sha256=hashlib.sha256(response.records).hexdigest())
                    rows.append(row)
                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 0), 1)
                    for item in detections:
                        a,b,c,d = item.box
                        color = (255,100,0) if item.class_name == 'droplet' else (0,0,255)
                        cv2.rectangle(output, (a,b), (c,d), color, 1)
                        cv2.putText(output, f'{item.class_name} {item.confidence:.2f}', (a,max(12,b-3)), 0, .35, color, 1)
                times.append(time.perf_counter() - frame_start)
                if index in (0, 30, 60, 90):
                    cv2.imwrite(str(args.output / f'frame_{index:04d}.png'), output)
                images.append(output)
            elapsed = time.perf_counter() - started
            for index, (payload, expected) in enumerate(reference_payloads):
                actual = client.infer(100000 + index, 0, payload)
                repeated[str(index)] = actual.records == expected
    finally:
        cap.release()
    if not rows:
        raise RuntimeError('No source frames processed')
    fps = len(times) / elapsed
    export_started = time.perf_counter()
    frame_files = 0
    if args.save_frames:
        frame_dir = args.output / 'annotated_frames'
        frame_dir.mkdir(parents=True, exist_ok=True)
        for index, output in enumerate(images):
            if not cv2.imwrite(
                str(frame_dir / f'frame_{index:06d}.jpg'),
                output,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            ):
                raise RuntimeError(f'Could not save annotated frame {index}')
            frame_files += 1
    writer = cv2.VideoWriter(str(args.output / 'fpga_udp_detection.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError('Video writer failed')
    for output in images:
        writer.write(output)
    writer.release()
    export_elapsed = time.perf_counter() - export_started
    report = dict(status='MEASURED', transport='Ethernet UDP -> PS DMA -> PL QNN', hello=hello,
                  source=str(args.video), source_frames=len(times), inferences=len(rows),
                  elapsed_seconds=elapsed, dual_roi_pipeline_fps=fps,
                  mean_frame_ms=statistics.mean(times)*1000,
                  p95_frame_ms=sorted(times)[int(.95*(len(times)-1))]*1000,
                  mean_qnn_ms_per_roi=statistics.mean(r['qnn_us'] for r in rows)/1000,
                  mean_udp_roundtrip_ms_per_roi=statistics.mean(r['roundtrip_ms'] for r in rows),
                  droplet_boxes=sum(r['droplet'] for r in rows), cell_boxes=sum(r['cell'] for r in rows),
                  boxes_are_not_unique_object_counts=True,
                  repeated_inputs_bit_exact=repeated,
                  accuracy='Not measured against ground truth',
                  timing_scope='Video decode, ROI preprocessing, UDP, PL inference, decode and overlay; excludes MP4 encoding',
                  camera_live=False,
                  saved_annotated_frames=frame_files,
                  frame_and_video_export_seconds=export_elapsed,
                  frame_and_video_export_fps=len(images)/max(export_elapsed, 1e-9),
                  processing_plus_export_fps=len(images)/max(elapsed+export_elapsed, 1e-9))
    with (args.output / 'per_roi.csv').open('w', newline='') as file:
        csv_writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
