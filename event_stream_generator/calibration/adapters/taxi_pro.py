"""Adapter for Taxi-Pro driver JSON event streams."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base import CalibrationAdapter, STANDARD_COLUMNS, format_timestamp


class TaxiProAdapter(CalibrationAdapter):
    name = "taxi_pro"

    def prepare(
        self,
        input_path: str | Path,
        output_path: str | Path,
        options: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        options = options or {}
        max_streams = _optional_int(options.get("max_streams"))
        source = Path(input_path)
        files = [source] if source.is_file() else sorted(source.glob("driver_*.json"))
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        stream_count = 0
        event_count = 0
        skipped_files = 0
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=STANDARD_COLUMNS)
            writer.writeheader()
            for file_path in files:
                if max_streams is not None and stream_count >= max_streams:
                    break
                events = _read_driver_events(file_path)
                if not events:
                    skipped_files += 1
                    continue
                stream_id = file_path.stem
                for timestamp, event_type in sorted(events):
                    writer.writerow(
                        {
                            "timestamp": format_timestamp(timestamp),
                            "stream_id": stream_id,
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
            "skipped_files": skipped_files,
            "max_streams": max_streams,
        }


def _read_driver_events(path: Path) -> List[Tuple[float, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    events = []
    for item in payload if isinstance(payload, list) else []:
        try:
            timestamp = float(item["time"])
        except (KeyError, TypeError, ValueError):
            continue
        attrs = item.get("TPP_attribute") or {}
        raw_type = attrs.get("event_type", item.get("event_type"))
        if raw_type in (None, ""):
            continue
        events.append((timestamp, f"taxi_event_{raw_type}"))
    return events


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
