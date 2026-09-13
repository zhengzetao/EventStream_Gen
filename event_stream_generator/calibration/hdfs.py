"""HDFS convenience wrapper around generic event calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .extractor import calibrate_event_csv


def calibrate_hdfs_csv(
    path: str | Path,
    *,
    timestamp_col: str = "timestamp",
    stream_id_col: str = "block_id",
    event_type_col: str = "template_id",
    min_transition_samples: int = 5,
) -> Dict[str, Any]:
    return calibrate_event_csv(
        path,
        timestamp_col=timestamp_col,
        stream_id_col=stream_id_col,
        event_type_col=event_type_col,
        domain="HDFS",
        min_transition_samples=min_transition_samples,
    )
