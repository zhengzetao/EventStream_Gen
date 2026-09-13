#!/usr/bin/env python3
"""Export accepted event streams into LLM training JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from event_stream_generator.export.training_export import SUPPORTED_FORMATS, export_training_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json")
    parser.add_argument("--format", choices=sorted(SUPPORTED_FORMATS), default="instruction_jsonl")
    args = parser.parse_args()

    report = export_training_data(
        accepted_jsonl=args.accepted_jsonl,
        output_jsonl=args.output_jsonl,
        output_format=args.format,
    )
    if args.report_json:
        output = Path(args.report_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    print(
        f"exported training examples={report['num_training_examples']} "
        f"format={args.format} output_jsonl={args.output_jsonl}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
