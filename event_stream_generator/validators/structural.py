"""Deterministic structural validation for generated streams."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional

from event_stream_generator.core.models import EventStream, GeneratedEvent, ValidationResult


VALID_RELATION_TYPES = {
    "direct_trigger",
    "delayed_trigger",
    "weak_trigger",
    "recovery",
}


def validate_structure(
    stream: EventStream, config: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    errors: List[str] = []
    config = config or {}
    events = stream.events
    event_by_id: Dict[int, GeneratedEvent] = {}
    for event in events:
        if event.event_id in event_by_id:
            errors.append(f"duplicate event_id: {event.event_id}")
        event_by_id[event.event_id] = event
        if not math.isfinite(float(event.timestamp)) or event.timestamp < 0.0:
            errors.append(f"invalid timestamp for event {event.event_id}")
        if event.relation_to_parent and event.relation_to_parent not in VALID_RELATION_TYPES:
            errors.append(f"invalid relation type for event {event.event_id}: {event.relation_to_parent}")
    _validate_mechanism_alignment(stream, errors)
    _validate_tree_limits(stream, config, errors)

    for event in events:
        if event.is_background:
            if event.parent_id is not None or event.root_id is not None:
                errors.append(f"background event {event.event_id} has causal ids")
            continue
        if event.parent_id is None:
            if event.root_id != event.event_id:
                errors.append(f"root event {event.event_id} has incorrect root_id")
            if event.cascade_depth != 0:
                errors.append(f"root event {event.event_id} has nonzero cascade depth")
            continue
        parent = event_by_id.get(event.parent_id)
        if parent is None:
            errors.append(f"event {event.event_id} has missing parent {event.parent_id}")
            continue
        if parent.is_background:
            errors.append(f"event {event.event_id} uses background event as parent")
        if event.timestamp <= parent.timestamp:
            errors.append(f"event {event.event_id} is not later than parent {parent.event_id}")
        expected_root = parent.root_id if parent.parent_id is not None else parent.event_id
        if event.root_id != expected_root:
            errors.append(f"event {event.event_id} has incorrect root_id")
        if event.cascade_depth != parent.cascade_depth + 1:
            errors.append(f"event {event.event_id} has incorrect cascade depth")
        _detect_parent_cycle(event, event_by_id, errors)

    if stream.ground_truth.get("parents"):
        expected = _derive_ground_truth(events)
        for key in ("parents", "roots", "paths", "root_event_ids", "background_event_ids"):
            if stream.ground_truth.get(key) != expected.get(key):
                errors.append(f"ground_truth {key} does not match events")
    return ValidationResult(approved=not errors, errors=errors)


def _validate_mechanism_alignment(stream: EventStream, errors: List[str]) -> None:
    mechanism = stream.mechanism or {}
    nodes = mechanism.get("nodes") or []
    edges = mechanism.get("edges") or []
    mechanism_node_ids = {
        str(node.get("node_id")) for node in nodes if node.get("node_id") is not None
    }
    event_node_ids = {
        str(event.mechanism_node_id)
        for event in stream.events
        if not event.is_background and event.mechanism_node_id is not None
    }
    for node_id in sorted(mechanism_node_ids - event_node_ids):
        errors.append(f"mechanism node {node_id} has no event")
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        relation = edge.get("relation_type")
        if source not in mechanism_node_ids:
            errors.append(f"mechanism edge source {source} has no node")
        if target not in mechanism_node_ids:
            errors.append(f"mechanism edge target {target} has no node")
        if relation not in VALID_RELATION_TYPES:
            errors.append(f"mechanism edge {source}->{target} has invalid relation type")


def _validate_tree_limits(
    stream: EventStream, config: Dict[str, Any], errors: List[str]
) -> None:
    if stream.topology != "tree":
        return
    tree_config = config.get("mechanism", {}).get("tree", {})
    max_children = tree_config.get("max_children_per_node")
    max_depth = tree_config.get("max_depth")
    edges = stream.mechanism.get("edges", []) if stream.mechanism else []
    if max_children is not None:
        counts = Counter(edge.get("source") for edge in edges)
        for source, count in counts.items():
            if count > int(max_children):
                errors.append(f"too many tree children for {source}: {count}")
    if max_depth is not None:
        for event in stream.events:
            if not event.is_background and event.cascade_depth > int(max_depth):
                errors.append(f"tree event {event.event_id} exceeds max depth")


def _detect_parent_cycle(
    event: GeneratedEvent,
    event_by_id: Dict[int, GeneratedEvent],
    errors: List[str],
) -> None:
    seen = set()
    current = event
    while current.parent_id is not None:
        if current.event_id in seen:
            errors.append(f"cycle detected at event {event.event_id}")
            return
        seen.add(current.event_id)
        parent = event_by_id.get(current.parent_id)
        if parent is None:
            return
        current = parent


def _derive_ground_truth(events: List[GeneratedEvent]) -> Dict[str, object]:
    event_by_id = {event.event_id: event for event in events}
    parents = {}
    roots = {}
    paths = {}
    root_event_ids = []
    background_event_ids = []
    for event in events:
        if event.is_background:
            background_event_ids.append(event.event_id)
            continue
        if event.parent_id is None:
            root_event_ids.append(event.event_id)
        else:
            parents[str(event.event_id)] = event.parent_id
        roots[str(event.event_id)] = event.root_id
        paths[str(event.event_id)] = _path(event, event_by_id)
    return {
        "parents": parents,
        "roots": roots,
        "paths": paths,
        "root_event_ids": sorted(root_event_ids),
        "background_event_ids": sorted(background_event_ids),
    }


def _path(event: GeneratedEvent, event_by_id: Dict[int, GeneratedEvent]) -> List[int]:
    path = [event.event_id]
    current = event
    while current.parent_id is not None and current.parent_id in event_by_id:
        current = event_by_id[current.parent_id]
        path.append(current.event_id)
    return list(reversed(path))
