"""Temporal regime interface."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict

from event_stream_generator.temporal.history import HistoryState


class TemporalRegime(ABC):
    name: str

    @abstractmethod
    def sample(
        self, params: Dict[str, Any], history_state: HistoryState, rng: random.Random
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def describe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"name": self.name, "params": dict(params)}


def positive_number(params: Dict[str, Any], key: str) -> bool:
    value = params.get(key)
    return isinstance(value, (int, float)) and value > 0.0
