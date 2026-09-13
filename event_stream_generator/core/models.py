"""Core records for generic synthetic event streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class GeneratedEvent:
    event_id: int
    timestamp: float
    abstract_type: str
    semantic_type: Optional[str]
    parent_id: Optional[int]
    root_id: Optional[int]
    relation_to_parent: Optional[str]
    delta_t_from_parent: Optional[float]
    regime: Optional[str]
    regime_params: Dict[str, Any]
    is_background: bool
    cascade_depth: int
    mechanism_node_id: Optional[str] = None
    distractor_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventStream:
    stream_id: str
    domain: str
    topology: str
    mechanism_type: str
    primary_regime: str
    seed: int
    mechanism: Dict[str, Any]
    events: List[GeneratedEvent]
    ground_truth: Dict[str, Any]
    validation: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    semantic_scenario: Optional[Dict[str, Any]] = None
    qa_pairs: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "stream_id": self.stream_id,
            "domain": self.domain,
            "topology": self.topology,
            "mechanism_type": self.mechanism_type,
            "primary_regime": self.primary_regime,
            "seed": self.seed,
            "mechanism": self.mechanism,
            "events": [event.to_dict() for event in self.events],
            "ground_truth": self.ground_truth,
            "validation": self.validation,
            "metadata": self.metadata,
        }
        if self.semantic_scenario is not None:
            payload["semantic_scenario"] = self.semantic_scenario
        if self.qa_pairs is not None:
            payload["qa_pairs"] = self.qa_pairs
        return payload


@dataclass(frozen=True)
class ValidationResult:
    approved: bool
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
