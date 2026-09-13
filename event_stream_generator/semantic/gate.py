"""Deterministic semantic quality gate for final dataset outputs."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

from ..judge.semantic_quality import score_semantic_quality, summarize_semantic_quality


DEFAULT_SEMANTIC_GATE_CONFIG = {
    "enabled": False,
    "min_rule_score": 0.7,
    "reject_on_rule_issues": True,
}


def apply_semantic_quality_gate(
    accepted: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    config: Dict[str, Any] | None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    gate_config = {**DEFAULT_SEMANTIC_GATE_CONFIG, **(config or {})}
    if not gate_config["enabled"]:
        return accepted, rejected, _report(
            enabled=False,
            accepted_before=len(accepted),
            accepted_after=len(accepted),
            rejected_by_gate=[],
            kept=accepted,
        )

    min_score = float(gate_config.get("min_rule_score", 0.7))
    reject_on_issues = bool(gate_config.get("reject_on_rule_issues", True))
    kept: List[Dict[str, Any]] = []
    gate_rejected: List[Dict[str, Any]] = []
    for row in accepted:
        quality = score_semantic_quality(row)
        errors = _gate_errors(
            quality,
            min_score=min_score,
            reject_on_issues=reject_on_issues,
        )
        row.setdefault("validation", {})["semantic_quality"] = quality
        if errors:
            gate_rejected.append(
                {
                    "stream_id": f"{row.get('stream_id', 'unknown')}_semantic_quality_gate",
                    "stage": "semantic_quality_gate",
                    "approved": False,
                    "errors": errors,
                    "sample": row,
                }
            )
        else:
            kept.append(row)
    return kept, [*rejected, *gate_rejected], _report(
        enabled=True,
        accepted_before=len(accepted),
        accepted_after=len(kept),
        rejected_by_gate=gate_rejected,
        kept=kept,
    )


def _gate_errors(
    quality: Dict[str, Any],
    *,
    min_score: float,
    reject_on_issues: bool,
) -> List[str]:
    errors = []
    score = float(quality.get("score", 0.0))
    if score < min_score:
        errors.append("semantic quality score below threshold")
    if reject_on_issues:
        errors.extend(str(issue) for issue in quality.get("issues", []) or [])
    return errors


def _report(
    *,
    enabled: bool,
    accepted_before: int,
    accepted_after: int,
    rejected_by_gate: List[Dict[str, Any]],
    kept: List[Dict[str, Any]],
) -> Dict[str, Any]:
    rejection_errors = Counter()
    for row in rejected_by_gate:
        for error in row.get("errors", []) or []:
            rejection_errors[str(error)] += 1
    return {
        "enabled": enabled,
        "accepted_before_gate": accepted_before,
        "accepted_after_gate": accepted_after,
        "rejected_by_gate": len(rejected_by_gate),
        "quality_summary": summarize_semantic_quality(kept),
        "rejection_error_counts": dict(sorted(rejection_errors.items())),
    }
