#!/usr/bin/env python3
"""Evaluate LLM semantic instantiation quality from generated JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from event_stream_generator.evaluation.llm_semantic_quality import evaluate_llm_semantics
from event_stream_generator.semantic.llm_client import OpenAIResponsesClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-jsonl", required=True)
    parser.add_argument("--rejected-jsonl")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--sample-preview-count", type=int, default=5)
    parser.add_argument("--judge-mode", choices=["none", "llm"], default="none")
    parser.add_argument("--judge-sample-limit", type=int)
    args = parser.parse_args()

    judge_client = OpenAIResponsesClient.from_env() if args.judge_mode == "llm" else None
    report = evaluate_llm_semantics(
        accepted_jsonl=args.accepted_jsonl,
        rejected_jsonl=args.rejected_jsonl,
        sample_preview_count=args.sample_preview_count,
        judge_client=judge_client,
        judge_sample_limit=args.judge_sample_limit,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"evaluated llm semantics accepted={args.accepted_jsonl} "
        f"rejected={args.rejected_jsonl} output_json={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
