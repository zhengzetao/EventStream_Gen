"""Adapter for already-normalized event CSV files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict

from .base import CalibrationAdapter, STANDARD_COLUMNS


class StandardCsvAdapter(CalibrationAdapter):
    name = "standard_csv"

    def prepare(
        self,
        input_path: str | Path,
        output_path: str | Path,
        options: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        del options
        source = Path(input_path)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        stream_ids = set()
        event_count = 0
        with source.open("r", encoding="utf-8", newline="") as src, target.open(
            "w", encoding="utf-8", newline=""
        ) as dst:
            reader = csv.DictReader(src)
            missing = [col for col in STANDARD_COLUMNS if col not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"Standard CSV is missing required columns: {missing}")
            writer = csv.DictWriter(dst, fieldnames=STANDARD_COLUMNS)
            writer.writeheader()
            for row in reader:
                stream_id = str(row["stream_id"]).strip()
                event_type = str(row["event_type"]).strip()
                if not stream_id or not event_type:
                    continue
                writer.writerow(
                    {
                        "timestamp": str(row["timestamp"]).strip(),
                        "stream_id": stream_id,
                        "event_type": event_type,
                    }
                )
                stream_ids.add(stream_id)
                event_count += 1
        return {
            "adapter": self.name,
            "input_path": str(source),
            "output_path": str(target),
            "stream_count": len(stream_ids),
            "event_count": event_count,
        }
