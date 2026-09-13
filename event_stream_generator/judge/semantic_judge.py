"""Optional LLM judge for semantic scenario quality."""

from __future__ import annotations

import json
from typing import Any, Dict, Protocol


class JudgeClient(Protocol):
    def complete(self, prompt: str) -> str: ...


def judge_semantic_scenario(
    row: Dict[str, Any],
    client: JudgeClient,
    approval_threshold: float = 0.7,
) -> Dict[str, Any]:
    prompt = _judge_prompt(row)
    text = client.complete(prompt)
    payload = _parse_json_object(text)
    score = float(payload.get("score", 0.0))
    issues = [str(item) for item in payload.get("issues", []) or []]
    suggestions = [str(item) for item in payload.get("suggestions", []) or []]
    approved = bool(payload.get("approved", False)) and score >= approval_threshold
    if score < approval_threshold and "score below threshold" not in issues:
        issues.append("score below threshold")
    return {
        "approved": approved,
        "score": round(score, 6),
        "issues": issues,
        "suggestions": suggestions,
    }


def summarize_judge_results(results: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "sample_count": 0,
            "approved_count": 0,
            "pass_rate": 0.0,
            "average_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "issue_counts": {},
        }
    issue_counts: Dict[str, int] = {}
    for result in results:
        for issue in result.get("issues", []) or []:
            issue_counts[str(issue)] = issue_counts.get(str(issue), 0) + 1
    approved_count = sum(1 for result in results if result.get("approved"))
    scores = [float(result.get("score", 0.0)) for result in results]
    return {
        "sample_count": len(results),
        "approved_count": approved_count,
        "pass_rate": round(approved_count / len(results), 6),
        "average_score": round(sum(scores) / len(scores), 6),
        "min_score": min(scores),
        "max_score": max(scores),
        "issue_counts": dict(sorted(issue_counts.items())),
    }


def _judge_prompt(row: Dict[str, Any]) -> str:
    payload = {
        "stream_id": row.get("stream_id"),
        "domain": row.get("domain"),
        "topology": row.get("topology"),
        "mechanism_type": row.get("mechanism_type"),
        "mechanism": {
            "nodes": row.get("mechanism", {}).get("nodes", []),
            "edges": row.get("mechanism", {}).get("edges", []),
        },
        "semantic_scenario": row.get("semantic_scenario", {}),
    }
    return (
        "You are a semantic quality judge for synthetic event-stream data.\n"
        "Do not modify or infer ground-truth answers. Do not output parent ids, "
        "root ids, timestamps, regimes, or labels.\n"
        "Judge whether the semantic scenario is natural, domain-consistent, "
        "causally directional, and aligned with the given mechanism.\n"
        "Return strict JSON only with this schema:\n"
        '{"approved": true, "score": 0.0, "issues": [], "suggestions": []}\n'
        "Use score in [0, 1]. Approve only if the scenario is coherent and the "
        "event meanings are specific rather than generic.\n"
        "Input JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def _parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("judge response does not contain a JSON object")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("judge response JSON must be an object")
    return payload
