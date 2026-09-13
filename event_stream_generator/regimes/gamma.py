"""Gamma inter-event time regime."""

from __future__ import annotations

import random
from typing import Any, Dict

from .base import TemporalRegime, positive_number
from event_stream_generator.temporal.history import HistoryState


class GammaRegime(TemporalRegime):
    name = "gamma"

    def sample(
        self, params: Dict[str, Any], history_state: HistoryState, rng: random.Random
    ) -> float:
        del history_state
        if not self.validate_params(params):
            raise ValueError("gamma regime requires shape > 0 and scale > 0")
        return max(
            rng.gammavariate(float(params["shape"]), float(params["scale"])),
            1e-9,
        )

    def validate_params(self, params: Dict[str, Any]) -> bool:
        return positive_number(params, "shape") and positive_number(params, "scale")
