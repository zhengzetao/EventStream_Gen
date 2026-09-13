"""Log-normal inter-event time regime."""

from __future__ import annotations

import random
from typing import Any, Dict

from .base import TemporalRegime, positive_number
from event_stream_generator.temporal.history import HistoryState


class LogNormalRegime(TemporalRegime):
    name = "lognormal"

    def sample(
        self, params: Dict[str, Any], history_state: HistoryState, rng: random.Random
    ) -> float:
        del history_state
        if not self.validate_params(params):
            raise ValueError("lognormal regime requires numeric mu and sigma > 0")
        return max(rng.lognormvariate(float(params["mu"]), float(params["sigma"])), 1e-9)

    def validate_params(self, params: Dict[str, Any]) -> bool:
        return isinstance(params.get("mu"), (int, float)) and positive_number(
            params, "sigma"
        )
