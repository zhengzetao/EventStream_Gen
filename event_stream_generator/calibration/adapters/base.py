"""Dataset adapter interface for calibration inputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


STANDARD_COLUMNS = ("timestamp", "stream_id", "event_type")


class CalibrationAdapter(ABC):
    """Normalize a dataset-specific raw file into the standard event CSV."""

    name: str

    @abstractmethod
    def prepare(
        self,
        input_path: str | Path,
        output_path: str | Path,
        options: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Write standard CSV and return preparation summary."""


def format_timestamp(value: float) -> str:
    return f"{float(value):.12g}"
