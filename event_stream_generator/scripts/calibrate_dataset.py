#!/usr/bin/env python3
"""Run adapter-based calibration from a YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from event_stream_generator.calibration.pipeline import run_calibration_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Calibration dataset YAML config.")
    args = parser.parse_args()

    result = run_calibration_pipeline(args.config)
    report = result["report"]
    summary = report["prior_summary"]
    print(
        f"calibrated dataset={report['dataset_name']} adapter={report['adapter']} "
        f"streams={summary['stream_count']} events={summary['event_count']} "
        f"transitions={summary['transition_prior_count']} prior={report['prior_output']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
