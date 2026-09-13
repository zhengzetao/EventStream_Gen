"""Dataset-level coverage report for generated samples."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List


def build_coverage_report(samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(samples)
    event_counts = [len(row.get("events", [])) for row in rows]
    background_ratios = []
    chain_lengths = Counter()
    tree_node_counts = Counter()
    for row in rows:
        events = row.get("events", [])
        bg = sum(1 for event in events if event.get("is_background"))
        background_ratios.append(bg / len(events) if events else 0.0)
        if row.get("topology") == "chain":
            chain_lengths[str(len(row.get("mechanism", {}).get("nodes", [])))] += 1
        if row.get("topology") == "tree":
            tree_node_counts[str(len(row.get("mechanism", {}).get("nodes", [])))] += 1
    return {
        "sample_count": len(rows),
        "domain_counts": dict(Counter(row.get("domain") for row in rows)),
        "topology_counts": dict(Counter(row.get("topology") for row in rows)),
        "mechanism_type_counts": dict(Counter(row.get("mechanism_type") for row in rows)),
        "regime_counts": dict(Counter(row.get("primary_regime") for row in rows)),
        "chain_length_distribution": dict(chain_lengths),
        "tree_node_count_distribution": dict(tree_node_counts),
        "background_density_counts": dict(
            Counter(row.get("metadata", {}).get("background_density") for row in rows)
        ),
        "event_count_quantiles": _quantiles(event_counts),
        "background_ratio_quantiles": _quantiles(background_ratios),
    }


def _quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "min": round(float(ordered[0]), 6),
        "p25": round(float(_pick(ordered, 0.25)), 6),
        "p50": round(float(_pick(ordered, 0.50)), 6),
        "p75": round(float(_pick(ordered, 0.75)), 6),
        "max": round(float(ordered[-1]), 6),
    }


def _pick(ordered: List[float], q: float) -> float:
    index = int(round((len(ordered) - 1) * q))
    return ordered[index]
