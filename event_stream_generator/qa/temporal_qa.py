"""Temporal QA generated only from timestamps."""

from __future__ import annotations

from typing import Any, Dict, List

from event_stream_generator.core.models import EventStream
from .common import first_parented_event


def generate_temporal_qa(stream: EventStream) -> List[Dict[str, Any]]:
    ordered_ids = [event.event_id for event in stream.events]
    qas: List[Dict[str, Any]] = [
        {
            "qa_id": f"{stream.stream_id}_qa_temporal_order",
            "qa_type": "temporal_order",
            "question": "List all event IDs in ascending timestamp order.",
            "target_event_id": None,
            "answer": ordered_ids,
            "answer_source": "events.timestamp",
        }
    ]
    target = first_parented_event(stream)
    if target is not None:
        qas.append(
            {
                "qa_id": f"{stream.stream_id}_qa_time_delay_{target.event_id}",
                "qa_type": "time_delay",
                "question": (
                    f"What is the time delay from event {target.parent_id} "
                    f"to event {target.event_id}?"
                ),
                "target_event_id": target.event_id,
                "source_event_id": target.parent_id,
                "answer": target.delta_t_from_parent,
                "answer_source": "events.timestamp_difference",
            }
        )
    return qas
