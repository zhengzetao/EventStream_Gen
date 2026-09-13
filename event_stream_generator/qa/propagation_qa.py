"""Propagation-path QA generated only from ground truth."""

from __future__ import annotations

from typing import Any, Dict, List

from event_stream_generator.core.models import EventStream
from .common import deepest_causal_event, event_label


def generate_propagation_qa(stream: EventStream) -> List[Dict[str, Any]]:
    target = deepest_causal_event(stream)
    if target is None:
        return []
    path = stream.ground_truth.get("paths", {}).get(str(target.event_id))
    if not path:
        return []
    return [
        {
            "qa_id": f"{stream.stream_id}_qa_causal_path_{target.event_id}",
            "qa_type": "causal_path",
            "question": f"What is the causal path from root to {event_label(target)}?",
            "target_event_id": target.event_id,
            "answer": path,
            "answer_source": "ground_truth.paths",
        }
    ]
