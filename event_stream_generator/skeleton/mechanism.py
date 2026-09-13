"""Abstract event mechanisms for generic event-stream generation."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EventNode:
    node_id: str
    role: str
    abstract_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EventEdge:
    source: str
    target: str
    relation_type: str
    regime: Optional[str]
    regime_params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventMechanism:
    mechanism_id: str
    nodes: List[EventNode]
    edges: List[EventEdge]
    topology: str
    domain_hint: str
    mechanism_type: str

    def __post_init__(self) -> None:
        self.node_by_id = {node.node_id: node for node in self.nodes}
        self.parent_by_target = {edge.target: edge.source for edge in self.edges}
        self.edge_by_target = {edge.target: edge for edge in self.edges}

    def get_parent(self, node_id: str) -> Optional[str]:
        return self.parent_by_target.get(node_id)

    def get_root(self, node_id: str) -> str:
        current = node_id
        seen = set()
        while current in self.parent_by_target:
            if current in seen:
                raise ValueError("cycle detected while resolving root")
            seen.add(current)
            current = self.parent_by_target[current]
        return current

    def get_path(self, node_id: str) -> List[str]:
        path = [node_id]
        current = node_id
        seen = set()
        while current in self.parent_by_target:
            if current in seen:
                raise ValueError("cycle detected while resolving path")
            seen.add(current)
            current = self.parent_by_target[current]
            path.append(current)
        return list(reversed(path))

    def depth(self, node_id: str) -> int:
        return len(self.get_path(node_id)) - 1

    def validate_dag(self) -> bool:
        try:
            for node in self.nodes:
                self.get_path(node.node_id)
        except ValueError:
            return False
        return len(self.parent_by_target) == len(self.edges)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "topology": self.topology,
            "domain_hint": self.domain_hint,
            "mechanism_type": self.mechanism_type,
        }


def build_chain_mechanism(
    *,
    stream_id: str,
    domain: str,
    mechanism_type: str,
    length: int,
    relation_type: str,
    primary_regime: str,
    rng: random.Random,
) -> EventMechanism:
    del rng
    if length < 2:
        raise ValueError("chain length must be at least 2")
    nodes = [
        EventNode(
            node_id=f"E{index}",
            role=("root" if index == 0 else "target" if index == length - 1 else "intermediate"),
            abstract_type=f"E{index}",
        )
        for index in range(length)
    ]
    edges = [
        EventEdge(
            source=f"E{index}",
            target=f"E{index + 1}",
            relation_type=relation_type,
            regime=primary_regime,
            regime_params={},
        )
        for index in range(length - 1)
    ]
    return EventMechanism(
        mechanism_id=f"{stream_id}_mechanism",
        nodes=nodes,
        edges=edges,
        topology="chain",
        domain_hint=domain,
        mechanism_type=mechanism_type,
    )


def build_tree_mechanism(
    *,
    stream_id: str,
    domain: str,
    mechanism_type: str,
    node_count: int,
    max_depth: int,
    max_children_per_node: int,
    relation_type: str,
    primary_regime: str,
    rng: random.Random,
) -> EventMechanism:
    if node_count < 2:
        raise ValueError("tree node_count must be at least 2")
    if max_depth < 1 or max_children_per_node < 1:
        raise ValueError("tree max_depth and max_children_per_node must be positive")
    nodes = [EventNode("E0", "root", "E0")]
    edges: List[EventEdge] = []
    child_counts = {"E0": 0}
    depths = {"E0": 0}
    for index in range(1, node_count):
        candidates = [
            node.node_id
            for node in nodes
            if depths[node.node_id] < max_depth
            and child_counts.get(node.node_id, 0) < max_children_per_node
        ]
        if not candidates:
            break
        parent = rng.choice(candidates)
        node_id = f"E{index}"
        child_counts[parent] = child_counts.get(parent, 0) + 1
        child_counts[node_id] = 0
        depths[node_id] = depths[parent] + 1
        nodes.append(EventNode(node_id, "target", node_id))
        edges.append(
            EventEdge(parent, node_id, relation_type, primary_regime, {})
        )
    intermediate_ids = {edge.source for edge in edges}
    nodes = [
        EventNode(
            node.node_id,
            "root" if node.node_id == "E0" else "intermediate" if node.node_id in intermediate_ids else "target",
            node.abstract_type,
        )
        for node in nodes
    ]
    return EventMechanism(
        mechanism_id=f"{stream_id}_mechanism",
        nodes=nodes,
        edges=edges,
        topology="tree",
        domain_hint=domain,
        mechanism_type=mechanism_type,
    )
