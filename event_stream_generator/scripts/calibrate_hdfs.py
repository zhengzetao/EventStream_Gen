#!/usr/bin/env python3
"""Calibrate HDFS transition priors from a structured HDFS event CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from event_stream_generator.calibration.hdfs import calibrate_hdfs_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--stream-id-col", default="block_id")
    parser.add_argument("--event-type-col", default="template_id")
    parser.add_argument("--min-transition-samples", type=int, default=5)
    args = parser.parse_args()

    prior = calibrate_hdfs_csv(
        args.input,
        timestamp_col=args.timestamp_col,
        stream_id_col=args.stream_id_col,
        event_type_col=args.event_type_col,
        min_transition_samples=args.min_transition_samples,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(prior, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"calibrated domain=HDFS streams={prior['stream_count']} "
        f"transitions={len(prior['transition_priors'])} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
