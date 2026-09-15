#!/usr/bin/env python3
"""Run adapter-based calibration and compare generic vs calibrated generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from event_stream_generator.calibration.evaluation_pipeline import (
    run_calibration_evaluation_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Calibration evaluation YAML config.")
    args = parser.parse_args()

    summary = run_calibration_evaluation_pipeline(args.config)
    print(
        f"evaluated calibration dataset={summary['dataset_name']} "
        f"comparison={summary['comparison_json']} summary={summary['summary_output']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
