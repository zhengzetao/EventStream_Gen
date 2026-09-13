"""Generic stochastic event-stream simulator."""

from __future__ import annotations

import random
from typing import Any, Dict, List

from ..core.models import EventStream, GeneratedEvent
from ..regimes import get_regime
from ..skeleton.mechanism import (
    EventMechanism,
    build_chain_mechanism,
    build_tree_mechanism,
)
from ..temporal.history import HistoryState, build_history_state
from ..temporal.policy import select_regime
from .background import generate_background_events


def generate_event_stream(
    *,
    stream_id: str,
    domain: str,
    topology: str,
    mechanism_type: str,
    primary_regime: str,
    seed: int,
    config: Dict[str, Any],
) -> EventStream:
    rng = random.Random(seed)
    mechanism = _build_mechanism(
        stream_id, domain, topology, mechanism_type, primary_regime, config, rng
    )
    return generate_event_stream_from_mechanism(
        stream_id=stream_id,
        domain=domain,
        topology=topology,
        mechanism_type=mechanism_type,
        primary_regime=primary_regime,
        seed=seed,
        config=config,
        mechanism=mechanism,
        rng=rng,
    )


def generate_event_stream_from_mechanism(
    *,
    stream_id: str,
    domain: str,
    topology: str,
    mechanism_type: str,
    primary_regime: str,
    seed: int,
    config: Dict[str, Any],
    mechanism: EventMechanism,
    rng: random.Random,
) -> EventStream:
    causal_events = _sample_causal_events(mechanism, primary_regime, config, rng)
    stream_end = max((event.timestamp for event in causal_events), default=1.0)
    stream_end = max(stream_end + 3.0, float(config.get("stream", {}).get("min_window", 10.0)))
    background_config = config.get("background", {})
    max_events = int(config.get("stream", {}).get("max_events", 0))
    configured_bg_max = int(background_config.get("max_background_events", 0))
    if max_events > 0:
        background_limit = max(0, min(configured_bg_max, max_events - len(causal_events)))
    else:
        background_limit = configured_bg_max
    background = generate_background_events(
        stream_id=stream_id,
        start=0.0,
        end=stream_end,
        density=str(background_config.get("density", "medium")),
        rates=background_config.get("rates", {}),
        max_background_events=background_limit,
        rng=rng,
        starting_event_id=len(causal_events),
    )
    background = _ensure_interleaving_background(
        causal_events, background, background_config
    )
    events = sorted(
        [*causal_events, *background],
        key=lambda event: (event.timestamp, event.event_id),
    )
    return EventStream(
        stream_id=stream_id,
        domain=domain,
        topology=topology,
        mechanism_type=mechanism_type,
        primary_regime=primary_regime,
        seed=seed,
        mechanism=mechanism.to_dict(),
        events=events,
        ground_truth=_build_ground_truth(events),
        validation={},
        metadata={
            "semantic_status": "not_instantiated",
            "time_unit": "abstract_float",
            "event_id_order_independent_of_timestamp": True,
        },
    )


def _ensure_interleaving_background(
    causal_events: List[GeneratedEvent],
    background: List[GeneratedEvent],
    background_config: Dict[str, Any],
) -> List[GeneratedEvent]:
    density = str(background_config.get("density", "medium"))
    max_background = int(background_config.get("max_background_events", 0))
    if density == "none" or max_background <= 0 or len(causal_events) < 2:
        return background
    causal_ids = {event.event_id for event in causal_events}
    combined = sorted(
        [*causal_events, *background],
        key=lambda event: (event.timestamp, event.event_id),
    )
    event_ids = [event.event_id for event in combined]
    if event_ids != sorted(event_ids):
        return background
    if len(background) >= max_background:
        return background
    first_child = causal_events[1]
    root = causal_events[0]
    timestamp = round((root.timestamp + first_child.timestamp) / 2.0, 9)
    if timestamp <= root.timestamp:
        timestamp = round(root.timestamp + 1e-6, 9)
    next_bg_index = len(background)
    background.append(
        GeneratedEvent(
            event_id=len(causal_events) + next_bg_index,
            timestamp=timestamp,
            abstract_type=f"BG{next_bg_index}",
            semantic_type=None,
            parent_id=None,
            root_id=None,
            relation_to_parent=None,
            delta_t_from_parent=None,
            regime="exponential",
            regime_params={
                "lambda": float(
                    background_config.get("rates", {}).get(density, 0.0)
                )
            },
            is_background=True,
            cascade_depth=0,
            mechanism_node_id=None,
            distractor_type="near_time_unrelated",
        )
    )
    return background


def _build_mechanism(
    stream_id: str,
    domain: str,
    topology: str,
    mechanism_type: str,
    primary_regime: str,
    config: Dict[str, Any],
    rng: random.Random,
) -> EventMechanism:
    mechanism_config = config.get("mechanism", {})
    relation_type = _relation_for_mechanism(mechanism_type)
    if topology == "chain":
        chain = mechanism_config.get("chain", {})
        length = rng.randint(int(chain.get("min_length", 4)), int(chain.get("max_length", 12)))
        return build_chain_mechanism(
            stream_id=stream_id,
            domain=domain,
            mechanism_type=mechanism_type,
            length=length,
            relation_type=relation_type,
            primary_regime=primary_regime,
            rng=rng,
        )
    if topology == "tree":
        tree = mechanism_config.get("tree", {})
        node_count = rng.randint(int(tree.get("min_nodes", 6)), int(tree.get("max_nodes", 30)))
        return build_tree_mechanism(
            stream_id=stream_id,
            domain=domain,
            mechanism_type=mechanism_type,
            node_count=node_count,
            max_depth=int(tree.get("max_depth", 6)),
            max_children_per_node=int(tree.get("max_children_per_node", 3)),
            relation_type=relation_type,
            primary_regime=primary_regime,
            rng=rng,
        )
    raise ValueError(f"Unsupported topology: {topology}")


def _relation_for_mechanism(mechanism_type: str) -> str:
    mapping = {
        "direct_trigger": "direct_trigger",
        "delayed_trigger": "delayed_trigger",
        "multi_hop_propagation": "direct_trigger",
        "burst": "weak_trigger",
        "recovery": "recovery",
    }
    return mapping.get(mechanism_type, "direct_trigger")


def _sample_causal_events(
    mechanism: EventMechanism,
    primary_regime: str,
    config: Dict[str, Any],
    rng: random.Random,
) -> List[GeneratedEvent]:
    root_node = next(
        (node for node in mechanism.nodes if mechanism.get_parent(node.node_id) is None),
        mechanism.nodes[0],
    )
    events = [
        GeneratedEvent(
            event_id=0,
            timestamp=0.0,
            abstract_type=root_node.abstract_type,
            semantic_type=None,
            parent_id=None,
            root_id=0,
            relation_to_parent=None,
            delta_t_from_parent=None,
            regime=None,
            regime_params={},
            is_background=False,
            cascade_depth=0,
            mechanism_node_id=root_node.node_id,
        )
    ]
    event_by_node = {root_node.node_id: events[0]}
    next_id = 1
    for edge in _topological_edges(mechanism):
        parent = event_by_node[edge.source]
        root = event_by_node[mechanism.get_root(edge.target)]
        history = (
            build_history_state(
                events,
                parent_event=parent,
                root_timestamp=root.timestamp,
            )
            if events
            else HistoryState.empty()
        )
        regime_name, params, delta_t = _sample_edge_delta(
            edge, history, config, rng, primary_regime
        )
        target_node = mechanism.node_by_id[edge.target]
        event = GeneratedEvent(
            event_id=next_id,
            timestamp=round(parent.timestamp + delta_t, 9),
            abstract_type=target_node.abstract_type,
            semantic_type=None,
            parent_id=parent.event_id,
            root_id=root.event_id,
            relation_to_parent=edge.relation_type,
            delta_t_from_parent=round(delta_t, 9),
            regime=regime_name,
            regime_params=params,
            is_background=False,
            cascade_depth=mechanism.depth(edge.target),
            mechanism_node_id=edge.target,
        )
        event_by_node[edge.target] = event
        events.append(event)
        next_id += 1
    return events


def _sample_edge_delta(
    edge,
    history: HistoryState,
    config: Dict[str, Any],
    rng: random.Random,
    primary_regime: str,
):
    if edge.regime == "deterministic_backoff":
        value = float(edge.regime_params["value"])
        if value <= 0.0:
            raise ValueError("deterministic_backoff requires value > 0")
        return edge.regime, dict(edge.regime_params), value
    if edge.regime and edge.regime != "generic_backoff" and edge.regime_params:
        params = dict(edge.regime_params)
        return edge.regime, params, get_regime(edge.regime).sample(params, history, rng)
    regime_name, params = select_regime(
        edge.relation_type,
        history,
        config,
        rng,
        preferred_regime=primary_regime,
    )
    return regime_name, params, get_regime(regime_name).sample(params, history, rng)


def _topological_edges(mechanism: EventMechanism):
    return sorted(
        mechanism.edges,
        key=lambda edge: (mechanism.depth(edge.source), edge.source, edge.target),
    )


def _build_ground_truth(events: List[GeneratedEvent]) -> Dict[str, Any]:
    event_by_id = {event.event_id: event for event in events}
    parents = {}
    roots = {}
    paths = {}
    root_event_ids = []
    background_event_ids = []
    for event in events:
        key = str(event.event_id)
        if event.is_background:
            background_event_ids.append(event.event_id)
            continue
        if event.parent_id is None:
            root_event_ids.append(event.event_id)
        else:
            parents[key] = event.parent_id
        roots[key] = event.root_id
        paths[key] = _path_for_event(event, event_by_id)
    return {
        "parents": parents,
        "roots": roots,
        "paths": paths,
        "root_event_ids": sorted(root_event_ids),
        "background_event_ids": sorted(background_event_ids),
    }


def _path_for_event(
    event: GeneratedEvent, event_by_id: Dict[int, GeneratedEvent]
) -> List[int]:
    path = [event.event_id]
    current = event
    seen = set()
    while current.parent_id is not None:
        if current.event_id in seen:
            raise ValueError("cycle detected while building event path")
        seen.add(current.event_id)
        current = event_by_id[current.parent_id]
        path.append(current.event_id)
    return list(reversed(path))
