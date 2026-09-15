"""Adapter for HDFS Event_traces.csv sequence rows."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .base import CalibrationAdapter, STANDARD_COLUMNS, format_timestamp


class HdfsEventTracesAdapter(CalibrationAdapter):
    name = "hdfs_event_traces"

    def prepare(
        self,
        input_path: str | Path,
        output_path: str | Path,
        options: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        options = options or {}
        max_streams = _optional_int(options.get("max_streams"))
        source = Path(input_path)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        stream_count = 0
        event_count = 0
        skipped_rows = 0
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=STANDARD_COLUMNS)
            writer.writeheader()
            for block_id, features, intervals in _iter_hdfs_rows(source):
                if max_streams is not None and stream_count >= max_streams:
                    break
                n_events = min(len(features), len(intervals))
                if not block_id or n_events == 0:
                    skipped_rows += 1
                    continue
                timestamp = 0.0
                for index in range(n_events):
                    if index > 0:
                        timestamp += intervals[index]
                    event_type = features[index].strip()
                    if not event_type:
                        continue
                    writer.writerow(
                        {
                            "timestamp": format_timestamp(timestamp),
                            "stream_id": block_id,
                            "event_type": event_type,
                        }
                    )
                    event_count += 1
                stream_count += 1
        return {
            "adapter": self.name,
            "input_path": str(source),
            "output_path": str(target),
            "stream_count": stream_count,
            "event_count": event_count,
            "skipped_rows": skipped_rows,
            "max_streams": max_streams,
        }


def _iter_hdfs_rows(path: Path) -> Iterable[Tuple[str, List[str], List[float]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if None in raw:
                parsed = _parse_malformed_hdfs_line(",".join([raw.get("BlockId", ""), *raw[None]]))
                if parsed is not None:
                    yield parsed
                continue
            block_id = str(raw.get("BlockId") or raw.get("block_id") or "").strip()
            features = _parse_bracket_list(str(raw.get("Features") or ""))
            intervals = [_to_float(item) for item in _parse_bracket_list(str(raw.get("TimeInterval") or ""))]
            yield block_id, features, intervals


def _parse_malformed_hdfs_line(line: str) -> Tuple[str, List[str], List[float]] | None:
    first_comma = line.find(",")
    if first_comma < 0:
        return None
    block_id = line[:first_comma].strip()
    lists = []
    start = first_comma + 1
    while len(lists) < 2:
        left = line.find("[", start)
        right = line.find("]", left + 1)
        if left < 0 or right < 0:
            return None
        lists.append(line[left : right + 1])
        start = right + 1
    return (
        block_id,
        _parse_bracket_list(lists[0]),
        [_to_float(item) for item in _parse_bracket_list(lists[1])],
    )


def _parse_bracket_list(value: str) -> List[str]:
    text = str(value).strip().strip('"')
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    return [item.strip().strip("'\"") for item in text.split(",") if item.strip()]


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
