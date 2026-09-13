"""Dataset-level diversity validation and summary metrics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List


def validate_dataset_diversity(
    samples: Iterable[Dict[str, Any]], config: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    rows = list(samples)
    config = config or {}
    errors: List[str] = []
    if not rows:
        errors.append("dataset contains no accepted samples")

    expected_domains = _as_list(config.get("domains"))
    expected_topologies = _as_list(config.get("topologies"))
    expected_mechanisms = _as_list(config.get("mechanism_types"))
    expected_regimes = _as_list(config.get("temporal_regimes"))

    domain_distribution = _distribution(row.get("domain") for row in rows)
    topology_distribution = _distribution(row.get("topology") for row in rows)
    mechanism_distribution = _distribution(row.get("mechanism_type") for row in rows)
    regime_distribution = _distribution(row.get("primary_regime") for row in rows)

    _check_expected_coverage("domain", domain_distribution, expected_domains, len(rows), errors)
    _check_expected_coverage("topology", topology_distribution, expected_topologies, len(rows), errors)
    _check_expected_coverage(
        "mechanism_type", mechanism_distribution, expected_mechanisms, len(rows), errors
    )
    _check_expected_coverage("regime", regime_distribution, expected_regimes, len(rows), errors)

    event_counts = [len(row.get("events", [])) for row in rows]
    cascade_depths = [
        int(event.get("cascade_depth", 0))
        for row in rows
        for event in row.get("events", [])
        if not event.get("is_background")
    ]
    background_ratios = []
    delta_ts = []
    qa_types = Counter()
    for row in rows:
        events = row.get("events", [])
        if events:
            background_count = sum(1 for event in events if event.get("is_background"))
            background_ratios.append(background_count / len(events))
        for event in events:
            delta_t = event.get("delta_t_from_parent")
            if delta_t is not None:
                delta_ts.append(float(delta_t))
        for qa in row.get("qa_pairs", []) or []:
            qa_type = qa.get("qa_type")
            if qa_type:
                qa_types[str(qa_type)] += 1

    metrics = {
        "domain_distribution": domain_distribution,
        "topology_distribution": topology_distribution,
        "mechanism_distribution": mechanism_distribution,
        "regime_distribution": regime_distribution,
        "event_count_quantiles": _quantiles(event_counts),
        "cascade_depth_quantiles": _quantiles(cascade_depths),
        "background_ratio_quantiles": _quantiles(background_ratios),
        "delta_t_quantiles": _quantiles(delta_ts),
        "qa_type_distribution": dict(sorted(qa_types.items())),
    }
    return {
        "approved": not errors,
        "errors": errors,
        "sample_count": len(rows),
        "domain_distribution": domain_distribution,
        "topology_distribution": topology_distribution,
        "mechanism_distribution": mechanism_distribution,
        "regime_distribution": regime_distribution,
        "metrics": metrics,
    }


def _distribution(values: Iterable[Any]) -> Dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values if value is not None).items()))


def _check_expected_coverage(
    name: str,
    observed: Dict[str, int],
    expected: List[str],
    sample_count: int,
    errors: List[str],
) -> None:
    if len(expected) <= 1 or sample_count < len(expected):
        return
    missing = sorted(set(expected) - set(observed))
    if missing:
        errors.append(f"missing configured {name} values: {missing}")


def _quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)
    return {
        "min": round(ordered[0], 6),
        "p25": round(_pick(ordered, 0.25), 6),
        "p50": round(_pick(ordered, 0.50), 6),
        "p75": round(_pick(ordered, 0.75), 6),
        "max": round(ordered[-1], 6),
    }


def _pick(ordered: List[float], q: float) -> float:
    index = int(round((len(ordered) - 1) * q))
    return ordered[index]


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]
