#!/usr/bin/env python3
"""Build a complete dataset run directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from event_stream_generator.dataset_builder import build_dataset
from event_stream_generator.core.io import load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--semantic-mode", choices=["none", "rule", "llm"])
    parser.add_argument("--qa-mode", choices=["none", "template"])
    parser.add_argument("--llm-concurrency", type=int)
    parser.add_argument("--llm-cache")
    parser.add_argument("--progress-path")
    args = parser.parse_args()

    config = load_yaml(args.config)
    for key, value in (
        ("output_dir", args.output_dir),
        ("num_samples", args.num_samples),
        ("seed", args.seed),
        ("semantic_mode", args.semantic_mode),
        ("qa_mode", args.qa_mode),
        ("llm_concurrency", args.llm_concurrency),
        ("llm_cache", args.llm_cache),
        ("progress_path", args.progress_path),
    ):
        if value is not None:
            config[key] = value
    build_dataset(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
