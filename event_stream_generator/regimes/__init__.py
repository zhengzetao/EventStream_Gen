"""Temporal regime registry."""

from __future__ import annotations

from typing import Dict

from .base import TemporalRegime
from .exponential import ExponentialRegime
from .gamma import GammaRegime
from .lognormal import LogNormalRegime
from .weibull import WeibullRegime


_REGIMES: Dict[str, TemporalRegime] = {
    item.name: item
    for item in (
        ExponentialRegime(),
        GammaRegime(),
        LogNormalRegime(),
        WeibullRegime(),
    )
}


def get_regime(name: str) -> TemporalRegime:
    try:
        return _REGIMES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown temporal regime: {name}") from exc


def regime_names() -> list[str]:
    return sorted(_REGIMES)
