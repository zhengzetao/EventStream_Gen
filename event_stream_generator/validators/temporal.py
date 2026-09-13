"""Temporal validation and quality metrics."""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Optional

from event_stream_generator.core.models import EventStream, GeneratedEvent, ValidationResult
from event_stream_generator.regimes import get_regime


def validate_temporal(
    stream: EventStream, config: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    errors: List[str] = []
    config = config or {}
    event_by_id: Dict[int, GeneratedEvent] = {event.event_id: event for event in stream.events}
    deltas = []
    causal_count = 0
    background_count = 0
    for event in stream.events:
        if event.is_background:
            background_count += 1
            continue
        causal_count += 1
        if event.parent_id is None:
            continue
        parent = event_by_id.get(event.parent_id)
        if parent is None:
            continue
        if event.delta_t_from_parent is None or event.delta_t_from_parent <= 0.0:
            errors.append(f"event {event.event_id} has non-positive delta_t")
            continue
        actual = round(event.timestamp - parent.timestamp, 9)
        if abs(actual - event.delta_t_from_parent) > 1e-6:
            errors.append(f"event {event.event_id} delta_t mismatch")
        if event.regime is None:
            errors.append(f"event {event.event_id} has no regime")
        elif event.regime == "deterministic_backoff":
            value = event.regime_params.get("value")
            if not isinstance(value, (int, float)) or value <= 0.0:
                errors.append(f"event {event.event_id} has invalid deterministic params")
        else:
            try:
                regime = get_regime(event.regime)
                if not regime.validate_params(event.regime_params):
                    errors.append(f"event {event.event_id} has invalid regime params")
            except ValueError as exc:
                errors.append(str(exc))
        if event.delta_t_from_parent > 1_000_000.0:
            errors.append(f"event {event.event_id} has extreme delta_t")
        deltas.append(event.delta_t_from_parent)
    metrics = {
        "event_count": len(stream.events),
        "causal_event_count": causal_count,
        "background_event_count": background_count,
        "background_ratio": (
            round(background_count / len(stream.events), 6) if stream.events else 0.0
        ),
        "delta_t_mean": round(mean(deltas), 9) if deltas else 0.0,
        "delta_t_min": round(min(deltas), 9) if deltas else 0.0,
        "delta_t_max": round(max(deltas), 9) if deltas else 0.0,
    }
    timestamps = [event.timestamp for event in stream.events]
    if timestamps != sorted(timestamps):
        errors.append("events are not sorted by timestamp")
    if len(stream.events) >= 2:
        gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
        if gaps and min(gaps) < 1e-12:
            errors.append("sequence is too dense")
    _validate_event_count(stream, config, errors)
    _validate_background_density(metrics["background_ratio"], stream, config, errors)
    return ValidationResult(approved=not errors, errors=errors, metrics=metrics)


def _validate_event_count(
    stream: EventStream, config: Dict[str, Any], errors: List[str]
) -> None:
    stream_config = config.get("stream", {})
    min_events = stream_config.get("min_events")
    max_events = stream_config.get("max_events")
    count = len(stream.events)
    if min_events is not None and count < int(min_events):
        errors.append("event count outside configured range")
    if max_events is not None and count > int(max_events):
        errors.append("event count outside configured range")


def _validate_background_density(
    background_ratio: float,
    stream: EventStream,
    config: Dict[str, Any],
    errors: List[str],
) -> None:
    background_config = config.get("background", {})
    density = str(
        stream.metadata.get("background_density")
        or background_config.get("density", "medium")
    )
    ranges = background_config.get("density_ranges", {})
    if density not in ranges:
        return
    lower, upper = ranges[density]
    if background_ratio < float(lower) or background_ratio > float(upper):
        errors.append("background ratio outside configured range")
