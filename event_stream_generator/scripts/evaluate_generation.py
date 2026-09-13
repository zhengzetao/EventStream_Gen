#!/usr/bin/env python3
"""Evaluate generic and calibrated synthetic streams against real events."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from event_stream_generator.evaluation.compare import compare_generation_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-csv", required=True)
    parser.add_argument("--generic-jsonl", required=True)
    parser.add_argument("--calibrated-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--timestamp-col", required=True)
    parser.add_argument("--stream-id-col", required=True)
    parser.add_argument("--event-type-col", required=True)
    args = parser.parse_args()

    report, rows = compare_generation_outputs(
        real_csv=args.real_csv,
        generic_jsonl=args.generic_jsonl,
        calibrated_jsonl=args.calibrated_jsonl,
        timestamp_col=args.timestamp_col,
        stream_id_col=args.stream_id_col,
        event_type_col=args.event_type_col,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "comparison", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"evaluated generic={args.generic_jsonl} calibrated={args.calibrated_jsonl} "
        f"real={args.real_csv} output_json={output_json} output_csv={output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
