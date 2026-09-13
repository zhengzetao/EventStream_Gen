"""Rule-based semantic validation."""

from __future__ import annotations

from typing import Any, Dict, List

from event_stream_generator.core.models import EventStream, ValidationResult
from event_stream_generator.skeleton.semantic_skeleton import build_semantic_skeleton


FORBIDDEN_SEMANTIC_FIELDS = {
    "timestamp",
    "delta_t",
    "delta_t_from_parent",
    "parent_id",
    "root_id",
    "ground_truth",
    "regime",
    "regime_params",
}


def validate_semantic(stream: EventStream) -> ValidationResult:
    errors: List[str] = []
    scenario = stream.semantic_scenario
    if scenario is None:
        return ValidationResult(False, ["semantic_scenario is missing"])
    _check_forbidden_fields(scenario, errors)
    if not str(scenario.get("scenario_description", "")).strip():
        errors.append("scenario_description is empty")
    if stream.metadata.get("semantic_status") != "instantiated":
        errors.append("metadata.semantic_status is not instantiated")

    mechanism_nodes = {
        str(node.get("node_id"))
        for node in stream.mechanism.get("nodes", [])
        if str(node.get("node_id", "")).startswith("E")
    }
    mapping = scenario.get("event_mapping", {})
    mapping_keys = set(mapping)
    for node_id in sorted(mechanism_nodes - mapping_keys):
        errors.append(f"missing semantic mapping for {node_id}")
    for node_id in sorted(mapping_keys - mechanism_nodes):
        errors.append(f"extra semantic mapping for {node_id}")
    for event in stream.events:
        if event.is_background:
            if event.mechanism_node_id in mapping_keys:
                errors.append(f"background event {event.event_id} appears in event_mapping")
            continue
        expected = mapping.get(str(event.mechanism_node_id))
        if event.semantic_type != expected:
            errors.append(f"event {event.event_id} semantic_type does not match mapping")
    _validate_relation_explanations(stream, scenario, errors)
    _validate_skeleton_constraints(stream, scenario, errors)
    return ValidationResult(approved=not errors, errors=errors)


def _validate_relation_explanations(
    stream: EventStream, scenario: Dict[str, Any], errors: List[str]
) -> None:
    expected = [
        (
            str(edge.get("source")),
            str(edge.get("target")),
            str(edge.get("relation_type")),
        )
        for edge in stream.mechanism.get("edges", [])
    ]
    actual = [
        (
            str(item.get("source")),
            str(item.get("target")),
            str(item.get("relation_type")),
        )
        for item in scenario.get("relation_explanations", [])
    ]
    if len(actual) != len(expected):
        errors.append("relation explanation count mismatch")
        return
    for left, right in zip(actual, expected):
        if left != right:
            errors.append("relation explanation mismatch")


def _check_forbidden_fields(value: Any, errors: List[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_SEMANTIC_FIELDS:
                errors.append(f"forbidden semantic field: {key}")
            _check_forbidden_fields(child, errors)
    elif isinstance(value, list):
        for child in value:
            _check_forbidden_fields(child, errors)


def _validate_skeleton_constraints(
    stream: EventStream,
    scenario: Dict[str, Any],
    errors: List[str],
) -> None:
    skeleton = build_semantic_skeleton(stream)
    mapping = scenario.get("event_mapping", {}) or {}
    explanations = scenario.get("relation_explanations", []) or []
    explanation_by_target = {
        str(item.get("target")): str(item.get("explanation", ""))
        for item in explanations
        if isinstance(item, dict)
    }
    for constraint in skeleton.get("edge_constraints", []):
        if constraint.get("relation_type") != "recovery":
            continue
        target = str(constraint.get("target"))
        target_text = str(mapping.get(target, ""))
        explanation = explanation_by_target.get(target, "")
        combined = f"{target_text} {explanation}".lower()
        for disallowed in constraint.get("disallowed_effect_types", []):
            if _contains_disallowed_effect(combined, str(disallowed)):
                errors.append(
                    f"recovery target introduces disallowed effect: {target} contains {disallowed}"
                )
                break


def _contains_disallowed_effect(text: str, effect: str) -> bool:
    aliases = {
        "new fault": ["new fault", "new failure"],
        "new outage": ["new outage"],
        "new blockage": ["new blockage", "blockage", "blocks", "blocked"],
        "risk escalation": ["risk escalation", "risk increases", "escalates risk"],
    }
    return any(alias in text for alias in aliases.get(effect, [effect]))
