"""Rule-based semantic quality scoring for instantiated scenarios."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List


GENERIC_TERMS = {
    "event",
    "update",
    "issue",
    "problem",
    "alert",
    "incident",
    "thing",
    "something",
    "state",
}


def score_semantic_quality(row: Dict[str, Any]) -> Dict[str, Any]:
    scenario = row.get("semantic_scenario", {}) or {}
    mapping = scenario.get("event_mapping", {}) or {}
    relation_explanations = scenario.get("relation_explanations", []) or []
    event_phrases = [str(value).strip() for value in mapping.values()]
    relation_texts = [
        str(item.get("explanation", "")).strip()
        for item in relation_explanations
        if isinstance(item, dict)
    ]
    description = str(scenario.get("scenario_description", "")).strip()

    short_event_count = sum(1 for phrase in event_phrases if _word_count(phrase) < 3)
    duplicate_event_count = _duplicate_count(_normalize(phrase) for phrase in event_phrases)
    generic_event_count = sum(1 for phrase in event_phrases if _is_generic_phrase(phrase))
    weak_relation_count = sum(
        1 for text in relation_texts if _word_count(text) < 5 or _is_generic_relation(text)
    )
    short_description = _word_count(description) < 6
    issues: List[str] = []
    if short_event_count:
        issues.append("event mapping contains short phrases")
    if duplicate_event_count:
        issues.append("event mapping contains duplicate phrases")
    if generic_event_count:
        issues.append("event mapping contains generic phrases")
    if weak_relation_count:
        issues.append("relation explanations are too short or generic")
    if short_description:
        issues.append("scenario description is too short")

    penalties = (
        short_event_count * 0.12
        + duplicate_event_count * 0.18
        + generic_event_count * 0.10
        + weak_relation_count * 0.12
        + (0.10 if short_description else 0.0)
    )
    score = max(0.0, round(1.0 - penalties, 6))
    return {
        "approved": score >= 0.7 and not issues,
        "score": score,
        "issues": issues,
        "metrics": {
            "event_count": len(event_phrases),
            "average_event_phrase_words": _average(_word_count(text) for text in event_phrases),
            "short_event_phrase_count": short_event_count,
            "duplicate_event_phrase_count": duplicate_event_count,
            "generic_event_phrase_count": generic_event_count,
            "weak_relation_explanation_count": weak_relation_count,
            "scenario_description_words": _word_count(description),
        },
    }


def summarize_semantic_quality(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [score_semantic_quality(row) for row in rows if row.get("semantic_scenario")]
    issue_counts = Counter(issue for result in scored for issue in result["issues"])
    scores = [float(result["score"]) for result in scored]
    return {
        "sample_count": len(scored),
        "approved_count": sum(1 for result in scored if result["approved"]),
        "average_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "issue_counts": dict(sorted(issue_counts.items())),
    }


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _duplicate_count(values: Iterable[str]) -> int:
    counts = Counter(value for value in values if value)
    return sum(count - 1 for count in counts.values() if count > 1)


def _is_generic_phrase(text: str) -> bool:
    words = {_normalize(word) for word in re.findall(r"[A-Za-z]+", text)}
    return bool(words) and words <= GENERIC_TERMS


def _is_generic_relation(text: str) -> bool:
    normalized = _normalize(text)
    return normalized in {"related", "causes", "a causes b", "triggered", "leads to"}


def _average(values: Iterable[int]) -> float:
    rows = list(values)
    return round(sum(rows) / len(rows), 6) if rows else 0.0
