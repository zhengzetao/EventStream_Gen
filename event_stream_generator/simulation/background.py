"""Background and distractor event generation."""

from __future__ import annotations

import random
from typing import Dict, List

from ..core.models import GeneratedEvent


DISTRACTOR_TYPES = (
    "near_time_unrelated",
    "similar_type_unrelated",
    "invalid_trigger_timing",
    "post_target_non_cause",
)


def generate_background_events(
    *,
    stream_id: str,
    start: float,
    end: float,
    density: str,
    rates: Dict[str, float],
    max_background_events: int,
    rng: random.Random,
    starting_event_id: int = 0,
) -> List[GeneratedEvent]:
    del stream_id
    if end <= start or max_background_events <= 0:
        return []
    rate = float(rates.get(density, 0.0))
    if rate <= 0.0:
        return []
    events: List[GeneratedEvent] = []
    current = start + rng.expovariate(rate)
    while current < end and len(events) < max_background_events:
        index = len(events)
        events.append(
            GeneratedEvent(
                event_id=starting_event_id + index,
                timestamp=round(current, 9),
                abstract_type=f"BG{index}",
                semantic_type=None,
                parent_id=None,
                root_id=None,
                relation_to_parent=None,
                delta_t_from_parent=None,
                regime="exponential",
                regime_params={"lambda": rate},
                is_background=True,
                cascade_depth=0,
                mechanism_node_id=None,
                distractor_type=DISTRACTOR_TYPES[index % len(DISTRACTOR_TYPES)],
            )
        )
        current += rng.expovariate(rate)
    return events
