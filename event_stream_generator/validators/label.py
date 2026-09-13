"""Validate QA labels against simulator ground truth."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from event_stream_generator.core.models import EventStream, GeneratedEvent, ValidationResult


def validate_qa_ground_truth(
    stream: EventStream, qa: Dict[str, Any]
) -> ValidationResult:
    expected = _expected_answer(stream, qa)
    if qa.get("answer") != expected:
        return ValidationResult(
            approved=False,
            errors=[
                f"answer mismatch for {qa.get('qa_id', '<unknown>')}: "
                f"expected {expected}, got {qa.get('answer')}"
            ],
        )
    return ValidationResult(approved=True, errors=[])


def validate_qa_pairs(
    stream: EventStream, qa_pairs: Iterable[Dict[str, Any]]
) -> ValidationResult:
    errors: List[str] = []
    count = 0
    for qa in qa_pairs:
        count += 1
        result = validate_qa_ground_truth(stream, qa)
        errors.extend(result.errors)
    if count == 0:
        errors.append("no QA pairs generated")
    return ValidationResult(
        approved=not errors,
        errors=errors,
        metrics={"qa_count": count},
    )


def _expected_answer(stream: EventStream, qa: Dict[str, Any]) -> Any:
    qa_type = qa.get("qa_type")
    target = qa.get("target_event_id")
    if qa_type == "direct_parent":
        return stream.ground_truth["parents"][str(target)]
    if qa_type == "root_cause":
        return stream.ground_truth["roots"][str(target)]
    if qa_type == "causal_path":
        return stream.ground_truth["paths"][str(target)]
    if qa_type == "temporal_order":
        return [event.event_id for event in sorted(stream.events, key=_event_sort_key)]
    if qa_type == "time_delay":
        target_event = _event_by_id(stream)[int(target)]
        source_event = _event_by_id(stream)[int(qa["source_event_id"])]
        return round(target_event.timestamp - source_event.timestamp, 9)
    raise ValueError(f"Unsupported qa_type: {qa_type}")


def _event_by_id(stream: EventStream) -> Dict[int, GeneratedEvent]:
    return {event.event_id: event for event in stream.events}


def _event_sort_key(event: GeneratedEvent):
    return (event.timestamp, event.event_id)
