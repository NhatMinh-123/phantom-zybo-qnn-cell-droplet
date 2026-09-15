#!/usr/bin/env python3
"""Compare the Arty S7 feature gate against the Python reference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import serial


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bright_round_feature_gate import evaluate_bright_round_candidate
from scripts.bright_round_feature_uart_protocol import (
    BAUD,
    BrightRoundFeatureUartClient,
)


DEFAULT_CANDIDATES = (
    ROOT
    / "dataset"
    / "droplet_microplastic_features_unlabeled_v1"
    / "particle_candidate_features.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Python/RTL parity on real extracted candidate features."
    )
    parser.add_argument("--port", default="COM12")
    parser.add_argument("--baud", type=int, default=BAUD)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--video-slug", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "bright_round_feature_uart_hardware.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    with args.candidates.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if args.video_slug and row.get("video_slug") != args.video_slug:
                continue
            rows.append(row)
            if args.limit and len(rows) >= args.limit:
                break
    if not rows:
        raise RuntimeError("No candidate rows matched the requested filters")

    mismatches: list[dict[str, object]] = []
    pass_count = 0
    start = time.perf_counter()
    with serial.Serial(
        args.port,
        args.baud,
        timeout=0.01,
        write_timeout=1.0,
    ) as port:
        time.sleep(0.2)
        port.reset_input_buffer()
        client = BrightRoundFeatureUartClient(port, timeout_seconds=0.15)
        for index, row in enumerate(rows):
            expected = evaluate_bright_round_candidate(row)
            actual = client.classify(row)
            pass_count += int(actual.decision)
            if actual.decision != expected.passed:
                mismatches.append(
                    {
                        "index": index,
                        "candidate_id": row.get("candidate_id", ""),
                        "expected": expected.passed,
                        "actual": actual.decision,
                        "software_reasons": list(expected.reasons),
                        "fpga_reasons": list(actual.reasons),
                    }
                )
                if len(mismatches) <= 10:
                    print(json.dumps(mismatches[-1], ensure_ascii=True))

    elapsed = time.perf_counter() - start
    summary = {
        "port": args.port,
        "baud": args.baud,
        "candidate_file": str(args.candidates.resolve()),
        "video_slug": args.video_slug,
        "vectors": len(rows),
        "fpga_pass": pass_count,
        "mismatches": len(mismatches),
        "parity_accuracy": 1.0 - len(mismatches) / len(rows),
        "elapsed_seconds": elapsed,
        "transactions_per_second": len(rows) / max(elapsed, 1e-9),
        "first_mismatches": mismatches[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
