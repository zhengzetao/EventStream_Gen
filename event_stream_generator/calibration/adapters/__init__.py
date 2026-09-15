"""Calibration dataset adapters."""

from __future__ import annotations

from typing import Dict, Type

from .base import CalibrationAdapter
from .csv_adapter import StandardCsvAdapter
from .gdelt import GdeltAdapter
from .hdfs_event_traces import HdfsEventTracesAdapter
from .taxi_pro import TaxiProAdapter


_ADAPTERS: Dict[str, Type[CalibrationAdapter]] = {
    StandardCsvAdapter.name: StandardCsvAdapter,
    HdfsEventTracesAdapter.name: HdfsEventTracesAdapter,
    TaxiProAdapter.name: TaxiProAdapter,
    GdeltAdapter.name: GdeltAdapter,
}


def get_adapter(name: str) -> CalibrationAdapter:
    try:
        return _ADAPTERS[name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown calibration adapter {name!r}. Available adapters: {sorted(_ADAPTERS)}"
        ) from exc


def available_adapters() -> Dict[str, Type[CalibrationAdapter]]:
    return dict(_ADAPTERS)
