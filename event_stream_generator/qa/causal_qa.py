"""Causal QA generated only from ground truth."""

from __future__ import annotations

from typing import Any, Dict, List

from event_stream_generator.core.models import EventStream
from .common import deepest_causal_event, event_label


def generate_causal_qa(stream: EventStream) -> List[Dict[str, Any]]:
    target = deepest_causal_event(stream)
    if target is None or target.parent_id is None:
        return []
    return [
        {
            "qa_id": f"{stream.stream_id}_qa_direct_parent_{target.event_id}",
            "qa_type": "direct_parent",
            "question": f"What is the direct parent of {event_label(target)}?",
            "target_event_id": target.event_id,
            "answer": stream.ground_truth["parents"][str(target.event_id)],
            "answer_source": "ground_truth.parents",
        },
        {
            "qa_id": f"{stream.stream_id}_qa_root_cause_{target.event_id}",
            "qa_type": "root_cause",
            "question": f"What is the root cause event of {event_label(target)}?",
            "target_event_id": target.event_id,
            "answer": stream.ground_truth["roots"][str(target.event_id)],
            "answer_source": "ground_truth.roots",
        },
    ]
