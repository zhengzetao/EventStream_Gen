"""Compare generic and calibrated synthetic streams against real event data."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .metrics import (
    burstiness,
    ks_statistic,
    l1_frequency_distance,
    repeated_event_ratio,
    wasserstein_distance,
)


def compare_generation_outputs(
    *,
    real_csv: str | Path,
    generic_jsonl: str | Path,
    calibrated_jsonl: str | Path,
    timestamp_col: str,
    stream_id_col: str,
    event_type_col: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    real = load_real_csv(
        real_csv,
        timestamp_col=timestamp_col,
        stream_id_col=stream_id_col,
        event_type_col=event_type_col,
    )
    generic = load_synthetic_jsonl(generic_jsonl)
    calibrated = load_synthetic_jsonl(calibrated_jsonl)
    report = {
        "inter_event_time": _continuous_comparison(
            real["inter_event_times"],
            generic["inter_event_times"],
            calibrated["inter_event_times"],
        ),
        "event_type_frequency": _frequency_comparison(
            real["event_type_counts"],
            generic["event_type_counts"],
            calibrated["event_type_counts"],
        ),
        "transition_frequency": _frequency_comparison(
            real["transition_counts"],
            generic["transition_counts"],
            calibrated["transition_counts"],
        ),
        "sequence_length": _continuous_comparison(
            real["sequence_lengths"],
            generic["sequence_lengths"],
            calibrated["sequence_lengths"],
        ),
        "burstiness": {
            "real": burstiness(real["inter_event_times"]),
            "generic": burstiness(generic["inter_event_times"]),
            "calibrated": burstiness(calibrated["inter_event_times"]),
        },
        "repeated_event_ratio": {
            "real": repeated_event_ratio(real["event_type_sequences"]),
            "generic": repeated_event_ratio(generic["event_type_sequences"]),
            "calibrated": repeated_event_ratio(calibrated["event_type_sequences"]),
        },
        "counts": {
            "real_streams": len(real["event_type_sequences"]),
            "generic_streams": len(generic["event_type_sequences"]),
            "calibrated_streams": len(calibrated["event_type_sequences"]),
        },
    }
    return report, flatten_report(report)


def load_real_csv(
    path: str | Path,
    *,
    timestamp_col: str,
    stream_id_col: str,
    event_type_col: str,
) -> Dict[str, Any]:
    streams: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [
            column
            for column in (timestamp_col, stream_id_col, event_type_col)
            if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {missing}")
        for row in reader:
            stream_id = str(row[stream_id_col]).strip()
            event_type = str(row[event_type_col]).strip()
            if stream_id and event_type:
                streams[stream_id].append((float(row[timestamp_col]), event_type))
    return _summarize_streams(streams.values())


def load_synthetic_jsonl(path: str | Path) -> Dict[str, Any]:
    streams = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            events = [
                (
                    float(event["timestamp"]),
                    str(event["abstract_type"]),
                )
                for event in row.get("events", [])
                if not event.get("is_background", False)
            ]
            streams.append(events)
    return _summarize_streams(streams)


def flatten_report(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric_name in (
        "inter_event_time",
        "event_type_frequency",
        "transition_frequency",
        "sequence_length",
    ):
        metric = report[metric_name]
        for comparison, values in metric.items():
            for name, value in values.items():
                rows.append(
                    {
                        "metric": f"{metric_name}_{name}",
                        "comparison": comparison,
                        "value": value,
                    }
                )
    for metric_name in ("burstiness", "repeated_event_ratio"):
        for comparison, value in report[metric_name].items():
            rows.append(
                {
                    "metric": metric_name,
                    "comparison": comparison,
                    "value": value,
                }
            )
    return rows


def _summarize_streams(streams: Iterable[List[Tuple[float, str]]]) -> Dict[str, Any]:
    event_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    inter_event_times: List[float] = []
    sequence_lengths: List[int] = []
    event_type_sequences: List[List[str]] = []
    for stream in streams:
        ordered = sorted(stream, key=lambda item: (item[0], item[1]))
        if not ordered:
            continue
        types = [event_type for _, event_type in ordered]
        times = [timestamp for timestamp, _ in ordered]
        event_type_sequences.append(types)
        sequence_lengths.append(len(types))
        event_counts.update(types)
        for left_type, right_type in zip(types, types[1:]):
            transition_counts[f"{left_type}->{right_type}"] += 1
        for left, right in zip(times, times[1:]):
            if right > left:
                inter_event_times.append(right - left)
    return {
        "event_type_counts": dict(event_counts),
        "transition_counts": dict(transition_counts),
        "inter_event_times": inter_event_times,
        "sequence_lengths": sequence_lengths,
        "event_type_sequences": event_type_sequences,
    }


def _continuous_comparison(
    real: List[float], generic: List[float], calibrated: List[float]
) -> Dict[str, Dict[str, float]]:
    return {
        "generic_vs_real": {
            "ks": ks_statistic(generic, real),
            "wasserstein": wasserstein_distance(generic, real),
        },
        "calibrated_vs_real": {
            "ks": ks_statistic(calibrated, real),
            "wasserstein": wasserstein_distance(calibrated, real),
        },
    }


def _frequency_comparison(
    real: Dict[str, int], generic: Dict[str, int], calibrated: Dict[str, int]
) -> Dict[str, Dict[str, float]]:
    return {
        "generic_vs_real": {"l1": l1_frequency_distance(generic, real)},
        "calibrated_vs_real": {"l1": l1_frequency_distance(calibrated, real)},
    }
