"""Shared helpers for template QA generation."""

from __future__ import annotations

from typing import Optional

from event_stream_generator.core.models import EventStream, GeneratedEvent


def causal_events(stream: EventStream) -> list[GeneratedEvent]:
    return [event for event in stream.events if not event.is_background]


def event_label(event: GeneratedEvent) -> str:
    semantic = f" ({event.semantic_type})" if event.semantic_type else ""
    return f"event {event.event_id} [{event.abstract_type}]{semantic}"


def deepest_causal_event(stream: EventStream) -> Optional[GeneratedEvent]:
    events = causal_events(stream)
    if not events:
        return None
    return max(events, key=lambda event: (event.cascade_depth, event.timestamp, event.event_id))


def first_parented_event(stream: EventStream) -> Optional[GeneratedEvent]:
    candidates = [
        event for event in causal_events(stream) if event.parent_id is not None
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda event: (event.cascade_depth, event.timestamp))[0]
