"""Calibrated generation using transition priors from real event tables."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from ..skeleton.mechanism import EventEdge, EventMechanism, EventNode
from .simulator import generate_event_stream_from_mechanism


def generate_calibrated_event_stream(
    *,
    stream_id: str,
    calibration_prior: Dict[str, Any],
    seed: int,
    config: Dict[str, Any],
    topology: str = "chain",
    mechanism_type: str = "multi_hop_propagation",
):
    rng = random.Random(seed)
    mechanism = build_calibrated_mechanism(
        stream_id=stream_id,
        calibration_prior=calibration_prior,
        config=config,
        topology=topology,
        mechanism_type=mechanism_type,
        rng=rng,
    )
    stream = generate_event_stream_from_mechanism(
        stream_id=stream_id,
        domain=str(calibration_prior.get("domain", "Calibrated")),
        topology=topology,
        mechanism_type=mechanism_type,
        primary_regime="calibrated_transition",
        seed=seed,
        config=config,
        mechanism=mechanism,
        rng=rng,
    )
    stream.metadata.update(
        {
            "generation_mode": "calibrated",
            "calibration_domain": calibration_prior.get("domain", "Calibrated"),
            "calibration_type": calibration_prior.get("calibration_type"),
            "calibration_profile": {
                "min_positive_delta_t": calibration_prior.get("delta_t_profile", {}).get("min", 0.0),
                "p95_delta_t": calibration_prior.get("delta_t_profile", {}).get("p95", 0.0),
                "event_count": calibration_prior.get("event_count", 0),
                "stream_count": calibration_prior.get("stream_count", 0),
            },
        }
    )
    return stream


def build_calibrated_mechanism(
    *,
    stream_id: str,
    calibration_prior: Dict[str, Any],
    config: Dict[str, Any],
    topology: str,
    mechanism_type: str,
    rng: random.Random,
) -> EventMechanism:
    if topology == "chain":
        length = _sample_chain_length(calibration_prior, config, rng)
        event_types = _sample_event_type_chain(
            calibration_prior,
            length,
            rng,
            config.get("calibrated_generation", {}),
        )
        return _chain_from_event_types(
            stream_id, calibration_prior, event_types, mechanism_type
        )
    if topology == "tree":
        event_types, edges = _sample_event_type_tree(calibration_prior, config, rng)
        return _tree_from_event_types(
            stream_id, calibration_prior, event_types, edges, mechanism_type
        )
    raise ValueError(f"Unsupported calibrated topology: {topology}")


def _sample_chain_length(
    calibration_prior: Dict[str, Any], config: Dict[str, Any], rng: random.Random
) -> int:
    distribution = calibration_prior.get("sequence_length_distribution", {})
    if distribution:
        return int(_weighted_choice({str(k): float(v) for k, v in distribution.items()}, rng))
    chain = config.get("mechanism", {}).get("chain", {})
    return rng.randint(int(chain.get("min_length", 4)), int(chain.get("max_length", 12)))


def _sample_event_type_chain(
    calibration_prior: Dict[str, Any],
    length: int,
    rng: random.Random,
    options: Dict[str, Any] | None = None,
) -> List[str]:
    options = options or {}
    transitions = calibration_prior.get("first_order_transitions", {})
    if not transitions:
        raise ValueError("calibration prior has no first_order_transitions")
    current = _sample_start_event(calibration_prior, rng)
    sequence = [current]
    while len(sequence) < length:
        previous = sequence[-2] if len(sequence) >= 2 else None
        current = _sample_next_event(
            transitions,
            current,
            rng,
            options,
            previous_event=previous,
            second_order_transitions=calibration_prior.get("second_order_transitions", {}),
        )
        sequence.append(current)
    return sequence


def _sample_start_event(calibration_prior: Dict[str, Any], rng: random.Random) -> str:
    transitions = calibration_prior.get("first_order_transitions", {})
    targets = {
        target
        for outgoing in transitions.values()
        for target in outgoing
    }
    root_like = [event_type for event_type in transitions if event_type not in targets]
    if root_like:
        transitions = {event_type: transitions[event_type] for event_type in root_like}
    candidates = {
        event_type: float(calibration_prior.get("event_frequencies", {}).get(event_type, 1.0))
        for event_type in transitions
    }
    return _weighted_choice(candidates, rng)


def _sample_next_event(
    transitions: Dict[str, Dict[str, Dict[str, Any]]],
    current: str,
    rng: random.Random,
    options: Dict[str, Any] | None = None,
    *,
    previous_event: str | None = None,
    second_order_transitions: Dict[str, Dict[str, Dict[str, Any]]] | None = None,
) -> str:
    options = options or {}
    if (
        options.get("use_second_order", True)
        and previous_event is not None
        and second_order_transitions
    ):
        second_key = f"{previous_event}|{current}"
        if second_order_transitions.get(second_key):
            return _weighted_choice(
                _transition_weights(
                    second_order_transitions[second_key],
                    all_event_types=transitions.keys(),
                    smoothing_alpha=float(options.get("smoothing_alpha", 0.0)),
                    temperature=float(options.get("temperature", 1.0)),
                ),
                rng,
            )
    targets = transitions.get(current)
    if not targets:
        return _weighted_choice({key: 1.0 for key in transitions}, rng)
    return _weighted_choice(
        _transition_weights(
            targets,
            all_event_types=transitions.keys(),
            smoothing_alpha=float(options.get("smoothing_alpha", 0.0)),
            temperature=float(options.get("temperature", 1.0)),
        ),
        rng,
    )


def _chain_from_event_types(
    stream_id: str,
    calibration_prior: Dict[str, Any],
    event_types: List[str],
    mechanism_type: str,
) -> EventMechanism:
    nodes = [
        EventNode(
            node_id=f"E{index}",
            role=("root" if index == 0 else "target" if index == len(event_types) - 1 else "intermediate"),
            abstract_type=event_type,
        )
        for index, event_type in enumerate(event_types)
    ]
    edges = [
        _edge_from_transition(
            source_node=f"E{index}",
            target_node=f"E{index + 1}",
            source_type=event_types[index],
            target_type=event_types[index + 1],
            calibration_prior=calibration_prior,
        )
        for index in range(len(event_types) - 1)
    ]
    return EventMechanism(
        mechanism_id=f"{stream_id}_calibrated_mechanism",
        nodes=nodes,
        edges=edges,
        topology="chain",
        domain_hint=str(calibration_prior.get("domain", "Calibrated")),
        mechanism_type=mechanism_type,
    )


def _sample_event_type_tree(
    calibration_prior: Dict[str, Any],
    config: Dict[str, Any],
    rng: random.Random,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    tree = config.get("mechanism", {}).get("tree", {})
    node_count = rng.randint(int(tree.get("min_nodes", 6)), int(tree.get("max_nodes", 30)))
    max_depth = int(tree.get("max_depth", 6))
    max_children = int(tree.get("max_children_per_node", 3))
    transitions = calibration_prior.get("first_order_transitions", {})
    event_types = [_sample_start_event(calibration_prior, rng)]
    edges: List[Tuple[int, int]] = []
    child_counts = {0: 0}
    depths = {0: 0}
    for index in range(1, node_count):
        candidates = [
            parent
            for parent in range(len(event_types))
            if depths[parent] < max_depth and child_counts.get(parent, 0) < max_children
        ]
        if not candidates:
            break
        parent = rng.choice(candidates)
        event_types.append(
            _sample_next_event(
                transitions,
                event_types[parent],
                rng,
                config.get("calibrated_generation", {}),
            )
        )
        edges.append((parent, index))
        child_counts[parent] = child_counts.get(parent, 0) + 1
        child_counts[index] = 0
        depths[index] = depths[parent] + 1
    return event_types, edges


def _tree_from_event_types(
    stream_id: str,
    calibration_prior: Dict[str, Any],
    event_types: List[str],
    edge_indexes: List[Tuple[int, int]],
    mechanism_type: str,
) -> EventMechanism:
    intermediate = {source for source, _ in edge_indexes}
    nodes = [
        EventNode(
            node_id=f"E{index}",
            role=("root" if index == 0 else "intermediate" if index in intermediate else "target"),
            abstract_type=event_type,
        )
        for index, event_type in enumerate(event_types)
    ]
    edges = [
        _edge_from_transition(
            source_node=f"E{source}",
            target_node=f"E{target}",
            source_type=event_types[source],
            target_type=event_types[target],
            calibration_prior=calibration_prior,
        )
        for source, target in edge_indexes
    ]
    return EventMechanism(
        mechanism_id=f"{stream_id}_calibrated_mechanism",
        nodes=nodes,
        edges=edges,
        topology="tree",
        domain_hint=str(calibration_prior.get("domain", "Calibrated")),
        mechanism_type=mechanism_type,
    )


def _edge_from_transition(
    *,
    source_node: str,
    target_node: str,
    source_type: str,
    target_type: str,
    calibration_prior: Dict[str, Any],
) -> EventEdge:
    transition = calibration_prior.get("transition_priors", {}).get(
        f"{source_type}->{target_type}", {}
    )
    return EventEdge(
        source=source_node,
        target=target_node,
        relation_type="direct_trigger",
        regime=transition.get("effective_regime", transition.get("regime", "generic_backoff")),
        regime_params=dict(transition.get("effective_params", transition.get("params", {}))),
    )


def _transition_weights(
    targets: Dict[str, Dict[str, Any]],
    *,
    all_event_types,
    smoothing_alpha: float,
    temperature: float,
) -> Dict[str, float]:
    weights = {
        target: float(record.get("probability", record.get("count", 1.0)))
        for target, record in targets.items()
    }
    if smoothing_alpha > 0.0:
        for event_type in all_event_types:
            weights[event_type] = weights.get(event_type, 0.0) + smoothing_alpha
    if temperature and temperature > 0.0 and abs(temperature - 1.0) > 1e-12:
        exponent = 1.0 / temperature
        weights = {
            event_type: max(weight, 0.0) ** exponent
            for event_type, weight in weights.items()
        }
    return weights


def _weighted_choice(weights: Dict[str, float], rng: random.Random) -> str:
    total = sum(float(value) for value in weights.values())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    draw = rng.random() * total
    cumulative = 0.0
    for key, weight in weights.items():
        cumulative += float(weight)
        if draw <= cumulative:
            return key
    return next(reversed(weights))
