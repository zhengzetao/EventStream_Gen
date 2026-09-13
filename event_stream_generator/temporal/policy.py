"""Rule-based temporal-regime selection and parameter adjustment."""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, Tuple

from .history import HistoryState


def select_regime(
    relation_type: str,
    history_state: HistoryState,
    prior: Dict[str, Any],
    rng: random.Random,
    preferred_regime: str | None = None,
) -> Tuple[str, Dict[str, Any]]:
    defaults = prior.get("temporal_policy", {}).get("relation_defaults", {})
    relation_spec = defaults.get(relation_type)
    if not relation_spec:
        raise ValueError(f"No temporal policy configured for relation_type={relation_type}")
    weights = relation_spec.get("regimes", {})
    if not weights:
        raise ValueError(f"No regimes configured for relation_type={relation_type}")
    regime_name = (
        preferred_regime if preferred_regime in weights else _weighted_choice(weights, rng)
    )
    params = copy.deepcopy(prior.get("regime_params", {}).get(regime_name, {}))
    if not params:
        raise ValueError(f"No parameters configured for regime={regime_name}")
    return regime_name, adjust_params(regime_name, params, relation_type, history_state, prior)


def adjust_params(
    regime_name: str,
    params: Dict[str, Any],
    relation_type: str,
    history_state: HistoryState,
    prior: Dict[str, Any],
) -> Dict[str, Any]:
    adjusted = dict(params)
    multipliers = prior.get("temporal_policy", {}).get("adjustments", {})
    scale_multiplier = 1.0
    if history_state.is_bursty:
        scale_multiplier *= float(multipliers.get("bursty_scale_multiplier", 0.65))
    if history_state.cascade_depth > 0:
        scale_multiplier *= 1.0 + float(
            multipliers.get("depth_delay_multiplier", 0.08)
        ) * history_state.cascade_depth
    if relation_type == "delayed_trigger":
        scale_multiplier *= float(multipliers.get("delayed_trigger_multiplier", 2.0))
    elif relation_type == "weak_trigger":
        scale_multiplier *= float(multipliers.get("weak_trigger_multiplier", 1.4))
    elif relation_type == "recovery":
        scale_multiplier *= float(multipliers.get("recovery_multiplier", 2.5))

    if regime_name == "exponential":
        adjusted["lambda"] = max(1e-9, float(adjusted["lambda"]) / scale_multiplier)
    elif regime_name in {"gamma", "weibull"}:
        adjusted["scale"] = max(1e-9, float(adjusted["scale"]) * scale_multiplier)
    elif regime_name == "lognormal":
        adjusted["mu"] = float(adjusted["mu"]) + _safe_log(scale_multiplier)
    return adjusted


def _weighted_choice(weights: Dict[str, float], rng: random.Random) -> str:
    total = sum(float(value) for value in weights.values())
    if total <= 0.0:
        raise ValueError("regime weights must sum to a positive value")
    draw = rng.random() * total
    cumulative = 0.0
    for name, weight in weights.items():
        cumulative += float(weight)
        if draw <= cumulative:
            return name
    return next(reversed(weights))


def _safe_log(value: float) -> float:
    import math

    return math.log(max(value, 1e-9))
