"""Programmatic semantic skeletons for constrained LLM instantiation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from ..core.models import EventStream


SKELETON_VERSION = "v2"

DOMAIN_ARTIFACT_HINTS = {
    "Cybersecurity": ["account", "session", "policy", "SIEM alert"],
    "Finance": ["order", "position", "limit", "risk ticket"],
    "Healthcare": ["patient record", "clinical workflow", "care team", "safety check"],
    "IT/System": ["service", "database", "queue", "monitoring alert"],
    "Industrial": ["sensor", "controller", "maintenance crew", "inspection record"],
    "Scientific Process": ["sample", "instrument reading", "assay control", "analysis file"],
    "Supply Chain": ["shipment", "warehouse task", "inventory record", "carrier update"],
    "Transportation": ["vehicle", "dispatcher", "signal system", "passenger notice"],
}

DOMAIN_CAUSAL_CHANNEL_HINTS = {
    "Cybersecurity": ["same account/session", "same host", "same SIEM case", "same access policy"],
    "Finance": ["same order", "same position", "same account", "same risk ticket"],
    "Healthcare": ["same patient episode", "same clinical criterion", "same care-team workflow", "same lab result"],
    "IT/System": ["same service dependency", "same request path", "same queue", "same monitoring alert"],
    "Industrial": ["same machine", "same sensor channel", "same control loop", "same maintenance ticket"],
    "Scientific Process": ["same sample", "same assay run", "same instrument channel", "same analysis file"],
    "Supply Chain": ["same shipment", "same warehouse order", "same inventory item", "same carrier route"],
    "Transportation": ["same vehicle", "same route segment", "same dispatcher decision", "same signal-control context"],
}

EFFECT_TYPES_BY_MECHANISM = {
    "direct_trigger": [
        "immediate operational effect",
        "detected downstream state change",
        "triggered workflow action",
    ],
    "delayed_trigger": [
        "accumulated condition",
        "delayed review outcome",
        "threshold-crossing effect",
    ],
    "multi_hop_propagation": [
        "propagated state transition",
        "intermediate dependency activation",
        "downstream impact",
    ],
    "recovery": [
        "diagnosis",
        "mitigation",
        "repair action",
        "reroute action",
        "restoration verification",
    ],
}

BRIDGE_BY_RELATION = {
    "direct_trigger": "Explain the immediate mechanism that makes the target follow from the source.",
    "delayed_trigger": "Explain the waiting condition, accumulation, or review latency between source and target.",
    "weak_trigger": "Explain why the source increases the likelihood of the target without making it inevitable.",
    "recovery": "Explain how the target mitigates, repairs, reroutes, restores, or verifies recovery from the source.",
}

DISALLOWED_BY_RELATION = {
    "recovery": ["new fault", "new outage", "new blockage", "risk escalation"],
    "direct_trigger": ["unexplained hidden intermediate", "unrelated administrative follow-up"],
    "delayed_trigger": ["instantaneous effect", "unexplained direct jump"],
    "weak_trigger": ["deterministic guarantee", "unrelated event"],
}

GROUNDING_REQUIREMENTS = [
    "Reuse at least one concrete domain object from the source event in the target event or explanation.",
    "Name the observable signal, resource, artifact, or workflow handoff that carries the effect.",
    "Avoid making a child depend on evidence that appears only in a sibling event.",
]

RECOVERY_STAGES = [
    "fault detection or diagnosis",
    "containment or routing mitigation",
    "repair action or restoration verification",
]


def build_semantic_skeleton(stream: EventStream) -> Dict[str, Any]:
    nodes = [
        node
        for node in stream.mechanism.get("nodes", [])
        if str(node.get("node_id", "")).startswith("E")
    ]
    edges = [
        edge
        for edge in stream.mechanism.get("edges", [])
        if str(edge.get("source", "")).startswith("E")
        and str(edge.get("target", "")).startswith("E")
    ]
    children_by_parent: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        children_by_parent[str(edge.get("source"))].append(str(edge.get("target")))

    event_slots = {}
    for index, node in enumerate(nodes):
        node_id = str(node.get("node_id"))
        sibling_group = _sibling_group(node_id, children_by_parent)
        event_slots[node_id] = {
            "role": str(node.get("role", "event")),
            "slot_index": index,
            "semantic_role": _semantic_role(node, stream.mechanism_type),
            "preferred_effect_type": _preferred_effect_type(
                stream.mechanism_type,
                index,
                sibling_group=sibling_group,
            ),
            "recovery_stage": _recovery_stage(node, index, stream.mechanism_type),
            "domain_artifact_hints": DOMAIN_ARTIFACT_HINTS.get(stream.domain, ["domain artifact"]),
            "sibling_group": sibling_group,
        }

    return {
        "version": SKELETON_VERSION,
        "domain": stream.domain,
        "topology": stream.topology,
        "mechanism_type": stream.mechanism_type,
        "event_slots": event_slots,
        "edge_constraints": [
            _edge_constraint(edge, stream.mechanism_type, stream.domain)
            for edge in edges
        ],
        "forbidden_patterns": [
            "changing topology or relation direction",
            "adding hidden causal events",
            "using identical or overlapping sibling meanings",
            "including simulator labels or hidden causal metadata",
        ],
    }


def _semantic_role(node: Dict[str, Any], mechanism_type: str) -> str:
    role = str(node.get("role", "event"))
    if role == "root":
        return "initial condition or triggering observation"
    if mechanism_type == "recovery":
        return "recovery step or recovery outcome"
    if role == "intermediate":
        return "intermediate state transition"
    if role == "target":
        return "observable downstream consequence"
    return "event state"


def _preferred_effect_type(
    mechanism_type: str,
    index: int,
    *,
    sibling_group: str | None,
) -> str:
    choices = EFFECT_TYPES_BY_MECHANISM.get(
        mechanism_type,
        ["domain-specific state change", "workflow action", "observable consequence"],
    )
    offset = _stable_offset(sibling_group) if sibling_group else 0
    return choices[(index + offset) % len(choices)]


def _sibling_group(
    node_id: str,
    children_by_parent: Dict[str, List[str]],
) -> str | None:
    for parent, children in children_by_parent.items():
        if node_id in children and len(children) > 1:
            return f"children_of_{parent}"
    return None


def _edge_constraint(
    edge: Dict[str, Any],
    mechanism_type: str,
    domain: str,
) -> Dict[str, Any]:
    relation_type = str(edge.get("relation_type", mechanism_type))
    return {
        "source": str(edge.get("source")),
        "target": str(edge.get("target")),
        "relation_type": relation_type,
        "required_causal_bridge": BRIDGE_BY_RELATION.get(
            relation_type,
            "Explain the concrete cause-effect bridge for this edge.",
        ),
        "allowed_effect_types": EFFECT_TYPES_BY_MECHANISM.get(
            mechanism_type,
            ["domain-specific state change", "workflow action", "observable consequence"],
        ),
        "disallowed_effect_types": DISALLOWED_BY_RELATION.get(
            relation_type,
            ["unrelated event", "duplicated sibling meaning"],
        ),
        "grounding_requirements": list(GROUNDING_REQUIREMENTS),
        "domain_causal_channel_hints": DOMAIN_CAUSAL_CHANNEL_HINTS.get(
            domain,
            ["same domain object", "same workflow", "same operational signal"],
        ),
    }


def _stable_offset(value: str | None) -> int:
    if not value:
        return 0
    return sum(ord(char) for char in value) % 7


def _recovery_stage(
    node: Dict[str, Any],
    index: int,
    mechanism_type: str,
) -> str | None:
    if mechanism_type != "recovery":
        return None
    if str(node.get("role", "")) == "root" or index == 0:
        return RECOVERY_STAGES[0]
    return RECOVERY_STAGES[min(index, len(RECOVERY_STAGES) - 1)]
