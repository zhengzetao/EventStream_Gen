"""Distribution-distance metrics for generation evaluation."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, Mapping


def ks_statistic(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = sorted(float(value) for value in left)
    right_values = sorted(float(value) for value in right)
    if not left_values or not right_values:
        return 0.0
    points = sorted(set(left_values + right_values))
    max_diff = 0.0
    left_index = 0
    right_index = 0
    for point in points:
        while left_index < len(left_values) and left_values[left_index] <= point:
            left_index += 1
        while right_index < len(right_values) and right_values[right_index] <= point:
            right_index += 1
        diff = abs(left_index / len(left_values) - right_index / len(right_values))
        max_diff = max(max_diff, diff)
    return round(max_diff, 9)


def wasserstein_distance(left: Iterable[float], right: Iterable[float]) -> float:
    try:
        from scipy.stats import wasserstein_distance as scipy_wasserstein
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for Wasserstein evaluation. Install with: pip install scipy"
        ) from exc
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if not left_values or not right_values:
        return 0.0
    return round(float(scipy_wasserstein(left_values, right_values)), 9)


def l1_frequency_distance(
    left: Mapping[str, int | float], right: Mapping[str, int | float]
) -> float:
    left_prob = _normalize(left)
    right_prob = _normalize(right)
    keys = set(left_prob) | set(right_prob)
    return round(sum(abs(left_prob.get(key, 0.0) - right_prob.get(key, 0.0)) for key in keys), 9)


def repeated_event_ratio(streams: Iterable[list[str]]) -> float:
    ratios = []
    for stream in streams:
        if not stream:
            continue
        counts = Counter(stream)
        repeated = sum(count for count in counts.values() if count > 1)
        ratios.append(repeated / len(stream))
    if not ratios:
        return 0.0
    return round(sum(ratios) / len(ratios), 9)


def burstiness(deltas: Iterable[float]) -> float:
    values = [float(value) for value in deltas if float(value) > 0.0]
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = variance ** 0.5
    if std + mean == 0.0:
        return 0.0
    return round((std - mean) / (std + mean), 9)


def _normalize(values: Mapping[str, int | float]) -> Dict[str, float]:
    total = sum(float(value) for value in values.values())
    if total <= 0.0:
        return {}
    return {str(key): float(value) / total for key, value in values.items()}
