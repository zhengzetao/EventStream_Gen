"""Dataset-level audit for semantic event-stream instantiation quality."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from event_stream_generator.judge.semantic_judge import (
    JudgeClient,
    judge_semantic_scenario,
    summarize_judge_results,
)
from event_stream_generator.judge.semantic_quality import (
    score_semantic_quality,
    summarize_semantic_quality,
)


def audit_semantic_dataset(
    *,
    accepted_jsonl: str | Path,
    rejected_jsonl: str | Path | None = None,
    low_quality_limit: int = 20,
    judge_client: JudgeClient | None = None,
    judge_sample_limit: int | None = None,
) -> Dict[str, Any]:
    accepted = _read_jsonl(accepted_jsonl)
    rejected = _read_jsonl(rejected_jsonl) if rejected_jsonl else []
    scored_rows = _scored_rows(accepted)
    report: Dict[str, Any] = {
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "llm_semantic_count": sum(
            1
            for row in accepted
            if row.get("metadata", {}).get("semantic_method") == "llm"
        ),
        "overall_quality": summarize_semantic_quality(accepted),
        "issue_counts": _issue_counts(scored_rows),
        "groups": {
            "domain": _group_quality(accepted, "domain"),
            "mechanism_type": _group_quality(accepted, "mechanism_type"),
            "topology": _group_quality(accepted, "topology"),
            "temporal_regime": _regime_group_quality(accepted),
            "semantic_method": _semantic_method_group_quality(accepted),
        },
        "low_quality_samples": _low_quality_samples(
            scored_rows,
            limit=low_quality_limit,
        ),
        "rejection_stage_counts": dict(
            sorted(Counter(str(row.get("stage", "unknown")) for row in rejected).items())
        ),
        "rejection_error_counts": _rejection_error_counts(rejected),
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


def _scored_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scored = []
    for row in rows:
        if not row.get("semantic_scenario"):
            continue
        result = score_semantic_quality(row)
        scored.append({"row": row, "quality": result})
    return scored


def _issue_counts(scored_rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter(
        issue for item in scored_rows for issue in item["quality"].get("issues", [])
    )
    return dict(sorted(counts.items()))


def _group_quality(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    return {
        group: summarize_semantic_quality(group_rows)
        for group, group_rows in sorted(grouped.items())
    }


def _regime_group_quality(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for regime in _row_regimes(row):
            grouped[regime].append(row)
    return {
        regime: summarize_semantic_quality(group_rows)
        for regime, group_rows in sorted(grouped.items())
    }


def _semantic_method_group_quality(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("metadata", {}).get("semantic_method", "unknown"))].append(row)
    return {
        method: summarize_semantic_quality(group_rows)
        for method, group_rows in sorted(grouped.items())
    }


def _row_regimes(row: Dict[str, Any]) -> List[str]:
    regimes = {
        str(edge.get("regime"))
        for edge in row.get("mechanism", {}).get("edges", [])
        if isinstance(edge, dict) and edge.get("regime")
    }
    return sorted(regimes) if regimes else ["unknown"]


def _low_quality_samples(
    scored_rows: List[Dict[str, Any]],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    ranked = sorted(
        scored_rows,
        key=lambda item: (
            float(item["quality"].get("score", 0.0)),
            str(item["row"].get("stream_id", "")),
        ),
    )
    samples = []
    for item in ranked[: max(0, limit)]:
        row = item["row"]
        scenario = row.get("semantic_scenario", {}) or {}
        samples.append(
            {
                "stream_id": row.get("stream_id"),
                "domain": row.get("domain"),
                "mechanism_type": row.get("mechanism_type"),
                "topology": row.get("topology"),
                "regimes": _row_regimes(row),
                "score": item["quality"].get("score", 0.0),
                "issues": item["quality"].get("issues", []),
                "scenario_description": scenario.get("scenario_description", ""),
                "event_mapping": scenario.get("event_mapping", {}),
            }
        )
    return samples


def _rejection_error_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    for row in rows:
        for error in row.get("errors", []) or []:
            counts[str(error)] += 1
    return dict(sorted(counts.items()))


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
