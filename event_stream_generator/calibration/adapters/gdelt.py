"""Adapter for GDELT-style tab-separated political event files."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base import CalibrationAdapter, STANDARD_COLUMNS, format_timestamp


class GdeltAdapter(CalibrationAdapter):
    name = "gdelt"

    def prepare(
        self,
        input_path: str | Path,
        output_path: str | Path,
        options: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        options = options or {}
        min_events = int(options.get("min_events_per_stream", 3))
        max_events = _optional_int(options.get("max_events_per_stream"))
        max_streams = _optional_int(options.get("max_streams"))
        source = Path(input_path)
        streams: Dict[str, List[Tuple[float, int, str]]] = defaultdict(list)
        skipped_rows = 0
        for row_index, row in enumerate(_iter_rows(source)):
            if len(row) <= 26:
                skipped_rows += 1
                continue
            try:
                timestamp = float(row[4])
            except (TypeError, ValueError):
                skipped_rows += 1
                continue
            actor1 = (row[5] or row[7] or "UNK1").strip()
            actor2 = (row[15] or row[17] or "UNK2").strip()
            event_code = row[26].strip()
            if not actor1 or not actor2 or not event_code:
                skipped_rows += 1
                continue
            streams[f"{actor1}->{actor2}"].append((timestamp, row_index, f"cameo_{event_code}"))
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        stream_count = 0
        event_count = 0
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=STANDARD_COLUMNS)
            writer.writeheader()
            for stream_id, events in sorted(streams.items()):
                if len(events) < min_events:
                    continue
                if max_streams is not None and stream_count >= max_streams:
                    break
                capped = sorted(events)[:max_events]
                for local_index, (timestamp, _row_index, event_type) in enumerate(capped):
                    writer.writerow(
                        {
                            "timestamp": format_timestamp(timestamp + local_index * 1e-6),
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
            "skipped_rows": skipped_rows,
            "min_events_per_stream": min_events,
            "max_events_per_stream": max_events,
            "max_streams": max_streams,
        }


def _iter_rows(path: Path):
    files = [path] if path.is_file() else sorted(path.glob("*"))
    for file_path in files:
        if not file_path.is_file():
            continue
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            yield from reader


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
