"""Exponential inter-event time regime."""

from __future__ import annotations

import random
from typing import Any, Dict

from .base import TemporalRegime, positive_number
from event_stream_generator.temporal.history import HistoryState


class ExponentialRegime(TemporalRegime):
    name = "exponential"

    def sample(
        self, params: Dict[str, Any], history_state: HistoryState, rng: random.Random
    ) -> float:
        del history_state
        if not self.validate_params(params):
            raise ValueError("exponential regime requires lambda > 0")
        return max(rng.expovariate(float(params["lambda"])), 1e-9)

    def validate_params(self, params: Dict[str, Any]) -> bool:
        return positive_number(params, "lambda")
