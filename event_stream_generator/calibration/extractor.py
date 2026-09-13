"""Generic event-table calibration extractor."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .fit_regimes import fit_best_regime


def calibrate_event_csv(
    path: str | Path,
    *,
    timestamp_col: str,
    stream_id_col: str,
    event_type_col: str,
    domain: str,
    min_transition_samples: int = 5,
) -> Dict[str, Any]:
    rows = _read_rows(path, timestamp_col, stream_id_col, event_type_col)
    streams = _group_streams(rows)
    event_frequencies: Counter[str] = Counter()
    first_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    second_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    transition_deltas: Dict[str, List[float]] = defaultdict(list)
    sequence_lengths: Counter[str] = Counter()
    for events in streams.values():
        ordered = sorted(events, key=lambda item: (item[0], item[1]))
        sequence_lengths[str(len(ordered))] += 1
        event_frequencies.update(event_type for _, event_type in ordered)
        for left, right in zip(ordered, ordered[1:]):
            delta = right[0] - left[0]
            if delta <= 0.0:
                continue
            source = left[1]
            target = right[1]
            first_counts[source][target] += 1
            transition_deltas[f"{source}->{target}"].append(delta)
        for first, second, third in zip(ordered, ordered[1:], ordered[2:]):
            if second[0] <= first[0] or third[0] <= second[0]:
                continue
            second_counts[f"{first[1]}|{second[1]}"][third[1]] += 1
    transition_priors = {}
    first_order = _normalize_nested_counts(first_counts)
    total_out = {
        source: sum(targets.values()) for source, targets in first_counts.items()
    }
    for key, samples in sorted(transition_deltas.items()):
        source, target = key.split("->", 1)
        fit = fit_best_regime(samples, min_samples=min_transition_samples)
        transition_priors[key] = {
            "probability": round(first_counts[source][target] / total_out[source], 9),
            "regime": fit["regime"],
            "params": fit["params"],
            "sample_count": fit["sample_count"],
            "fit": fit["fit"],
            "delta_t_summary": _summary(samples),
        }
    return {
        "domain": domain,
        "calibration_type": "transition_interevent_time",
        "input_schema": {
            "timestamp_col": timestamp_col,
            "stream_id_col": stream_id_col,
            "event_type_col": event_type_col,
        },
        "stream_count": len(streams),
        "event_count": sum(event_frequencies.values()),
        "event_frequencies": dict(sorted(event_frequencies.items())),
        "first_order_transitions": first_order,
        "second_order_transitions": _normalize_nested_counts(second_counts),
        "transition_priors": transition_priors,
        "sequence_length_distribution": dict(sorted(sequence_lengths.items())),
        "backoff": {
            "min_transition_samples": min_transition_samples,
            "order": [
                "specific_transition",
                "event_type",
                "domain",
                "generic",
            ],
        },
    }


def _read_rows(
    path: str | Path,
    timestamp_col: str,
    stream_id_col: str,
    event_type_col: str,
) -> List[Tuple[float, str, str, int]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [
            column
            for column in (timestamp_col, stream_id_col, event_type_col)
            if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {missing}")
        rows = []
        for index, row in enumerate(reader):
            stream_id = str(row[stream_id_col]).strip()
            event_type = str(row[event_type_col]).strip()
            if not stream_id or not event_type:
                continue
            rows.append((float(row[timestamp_col]), stream_id, event_type, index))
    return rows


def _group_streams(
    rows: Iterable[Tuple[float, str, str, int]]
) -> Dict[str, List[Tuple[float, str]]]:
    streams: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    for timestamp, stream_id, event_type, order in rows:
        streams[stream_id].append((timestamp, event_type))
    return dict(streams)


def _normalize_nested_counts(
    counts: Dict[str, Counter[str]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    output: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for source, targets in sorted(counts.items()):
        total = sum(targets.values())
        output[source] = {
            target: {
                "count": count,
                "probability": round(count / total, 9) if total else 0.0,
            }
            for target, count in sorted(targets.items())
        }
    return output


def _summary(samples: List[float]) -> Dict[str, float]:
    ordered = sorted(samples)
    if not ordered:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": round(ordered[0], 9),
        "mean": round(sum(ordered) / len(ordered), 9),
        "max": round(ordered[-1], 9),
    }
