"""Quality summary for LLM semantic instantiation outputs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from event_stream_generator.judge.semantic_quality import summarize_semantic_quality
from event_stream_generator.judge.semantic_judge import (
    JudgeClient,
    judge_semantic_scenario,
    summarize_judge_results,
)


def evaluate_llm_semantics(
    *,
    accepted_jsonl: str | Path,
    rejected_jsonl: str | Path | None = None,
    sample_preview_count: int = 5,
    judge_client: JudgeClient | None = None,
    judge_sample_limit: int | None = None,
) -> Dict[str, Any]:
    accepted = _read_jsonl(accepted_jsonl)
    rejected = _read_jsonl(rejected_jsonl) if rejected_jsonl else []
    semantic_results = [
        row.get("validation", {}).get("semantic", {})
        for row in accepted
        if row.get("metadata", {}).get("semantic_method") == "llm"
    ]
    approved_count = sum(1 for result in semantic_results if result.get("approved"))
    attempt_counts = [
        int(row.get("metadata", {}).get("llm_attempt_count", 0))
        for row in accepted
        if row.get("metadata", {}).get("semantic_method") == "llm"
    ]
    rejection_stage_counts = Counter(
        str(row.get("stage", "unknown")) for row in rejected
    )
    rejection_error_counts = Counter()
    for row in rejected:
        for error in row.get("errors", []) or []:
            rejection_error_counts[str(error)] += 1
    accepted_count = len(accepted)
    report = {
        "accepted_count": accepted_count,
        "rejected_count": len(rejected),
        "llm_semantic_count": len(semantic_results),
        "semantic_pass_rate": (
            round(approved_count / len(semantic_results), 6)
            if semantic_results
            else 0.0
        ),
        "average_llm_attempt_count": (
            round(sum(attempt_counts) / len(attempt_counts), 6)
            if attempt_counts
            else 0.0
        ),
        "max_llm_attempt_count": max(attempt_counts) if attempt_counts else 0,
        "domain_counts": dict(sorted(Counter(row.get("domain") for row in accepted).items())),
        "rejection_stage_counts": dict(sorted(rejection_stage_counts.items())),
        "rejection_error_counts": dict(sorted(rejection_error_counts.items())),
        "semantic_quality": summarize_semantic_quality(accepted),
        "sample_previews": _sample_previews(accepted, sample_preview_count),
    }
    if judge_client is not None:
        judged_rows = accepted[:judge_sample_limit] if judge_sample_limit else accepted
        judge_results = []
        for row in judged_rows:
            if not row.get("semantic_scenario"):
                continue
            try:
                judge_results.append(judge_semantic_scenario(row, judge_client))
            except Exception as exc:
                judge_results.append(
                    {
                        "approved": False,
                        "score": 0.0,
                        "issues": [f"judge failed: {exc}"],
                        "suggestions": [],
                    }
                )
        report["semantic_judge"] = summarize_judge_results(judge_results)
    return report


def _sample_previews(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    previews = []
    for row in rows[: max(0, limit)]:
        scenario = row.get("semantic_scenario", {})
        previews.append(
            {
                "stream_id": row.get("stream_id"),
                "domain": row.get("domain"),
                "scenario_description": scenario.get("scenario_description", ""),
                "event_mapping": scenario.get("event_mapping", {}),
                "llm_attempt_count": row.get("metadata", {}).get("llm_attempt_count"),
            }
        )
    return previews


def _read_jsonl(path: str | Path | None) -> List[Dict[str, Any]]:
    if path is None:
        return []
    target = Path(path)
    if not target.exists():
        return []
    return [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
