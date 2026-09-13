"""History-state compression used by temporal policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from ..core.models import GeneratedEvent


@dataclass(frozen=True)
class HistoryState:
    previous_event_type: Optional[str]
    previous_two_event_types: Tuple[str, ...]
    cascade_depth: int
    time_since_root: float
    recent_event_count: int
    recent_same_type_count: int
    is_bursty: bool
    active_source_count: int

    @classmethod
    def empty(cls) -> "HistoryState":
        return cls(None, (), 0, 0.0, 0, 0, False, 1)


def build_history_state(
    events: Iterable[GeneratedEvent],
    *,
    parent_event: GeneratedEvent,
    root_timestamp: float,
    recent_window: float = 2.0,
    burst_threshold: int = 3,
) -> HistoryState:
    ordered = sorted(events, key=lambda item: (item.timestamp, item.event_id))
    previous = ordered[-1] if ordered else None
    previous_two = tuple(event.abstract_type for event in ordered[-2:])
    now = parent_event.timestamp
    recent = [event for event in ordered if 0.0 <= now - event.timestamp <= recent_window]
    same_type = [
        event for event in recent if event.abstract_type == parent_event.abstract_type
    ]
    active_sources = len(
        {
            event.root_id
            for event in ordered
            if not event.is_background and event.root_id is not None
        }
    ) or 1
    return HistoryState(
        previous_event_type=previous.abstract_type if previous else None,
        previous_two_event_types=previous_two,
        cascade_depth=parent_event.cascade_depth,
        time_since_root=max(0.0, now - root_timestamp),
        recent_event_count=len(recent),
        recent_same_type_count=len(same_type),
        is_bursty=len(recent) >= burst_threshold,
        active_source_count=active_sources,
    )
