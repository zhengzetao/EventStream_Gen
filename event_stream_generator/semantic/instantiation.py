"""Rule-based semantic instantiation for abstract event streams."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Any, Dict, List, Protocol

from event_stream_generator.core.models import EventStream, GeneratedEvent
from event_stream_generator.skeleton.semantic_skeleton import (
    SKELETON_VERSION,
    build_semantic_skeleton,
)
from event_stream_generator.validators.semantic import validate_semantic


class SemanticClient(Protocol):
    def complete(self, prompt: str) -> str: ...


def instantiate_scenario(
    stream: EventStream,
    templates: Dict[str, Any],
    seed: int,
) -> EventStream:
    del seed
    domain_templates = _domain_templates(stream.domain, templates)
    event_phrases = list(domain_templates.get("event_phrases", []))
    background_phrases = list(domain_templates.get("background_phrases", []))
    if not event_phrases:
        raise ValueError(f"No event_phrases configured for domain={stream.domain}")
    if not background_phrases:
        raise ValueError(f"No background_phrases configured for domain={stream.domain}")

    mechanism_nodes = [
        node["node_id"]
        for node in stream.mechanism.get("nodes", [])
        if str(node.get("node_id", "")).startswith("E")
    ]
    mapping = {
        node_id: _phrase(event_phrases, index)
        for index, node_id in enumerate(mechanism_nodes)
    }
    background_mapping: Dict[str, str] = {}
    updated_events: List[GeneratedEvent] = []
    background_index = 0
    for event in stream.events:
        if event.is_background:
            phrase = _phrase(background_phrases, background_index)
            background_mapping[str(event.event_id)] = phrase
            background_index += 1
        else:
            phrase = mapping.get(str(event.mechanism_node_id))
        updated_events.append(replace(event, semantic_type=phrase))

    relation_explanations = [
        _relation_explanation(edge, mapping)
        for edge in stream.mechanism.get("edges", [])
    ]
    scenario = {
        "scenario_description": _scenario_description(stream, mapping),
        "domain": stream.domain,
        "event_mapping": mapping,
        "background_mapping": background_mapping,
        "relation_explanations": relation_explanations,
        "instantiation_method": "rule_based_template_v1",
    }
    return EventStream(
        stream_id=stream.stream_id,
        domain=stream.domain,
        topology=stream.topology,
        mechanism_type=stream.mechanism_type,
        primary_regime=stream.primary_regime,
        seed=stream.seed,
        mechanism=copy.deepcopy(stream.mechanism),
        events=updated_events,
        ground_truth=copy.deepcopy(stream.ground_truth),
        validation=copy.deepcopy(stream.validation),
        metadata={**copy.deepcopy(stream.metadata), "semantic_status": "instantiated"},
        semantic_scenario=scenario,
    )


def instantiate_scenario_llm(
    stream: EventStream,
    client: SemanticClient,
    max_retry: int = 3,
) -> EventStream:
    last_error = ""
    for attempt in range(max_retry):
        prompt = _llm_prompt(stream, last_error=last_error)
        text = client.complete(prompt)
        try:
            scenario = _parse_llm_json(text)
            updated = _apply_semantic_scenario(
                stream,
                scenario,
                semantic_method="llm",
                llm_attempt_count=attempt + 1,
            )
        except Exception as exc:
            last_error = f"Previous response failed to parse/apply: {exc}"
            continue
        validation = validate_semantic(updated)
        if validation.approved:
            return updated
        last_error = "Previous response failed semantic validation: " + "; ".join(
            validation.errors
        )
    raise RuntimeError(f"LLM semantic instantiation failed after {max_retry} attempts: {last_error}")


def _apply_semantic_scenario(
    stream: EventStream,
    scenario: Dict[str, Any],
    semantic_method: str,
    llm_attempt_count: int | None = None,
) -> EventStream:
    mapping = scenario.get("event_mapping")
    if not isinstance(mapping, dict):
        raise ValueError("event_mapping must be a JSON object")
    background_mapping = scenario.get("background_mapping", {})
    if not isinstance(background_mapping, dict):
        background_mapping = {}
    updated_events: List[GeneratedEvent] = []
    for event in stream.events:
        if event.is_background:
            semantic_type = background_mapping.get(str(event.event_id), event.semantic_type)
        else:
            semantic_type = mapping.get(str(event.mechanism_node_id))
        updated_events.append(replace(event, semantic_type=semantic_type))
    scenario = copy.deepcopy(scenario)
    scenario["relation_explanations"] = _canonical_relation_explanations(
        stream,
        scenario.get("relation_explanations", []),
    )
    scenario["domain"] = stream.domain
    scenario["instantiation_method"] = (
        "llm_responses_api_v1" if semantic_method == "llm" else semantic_method
    )
    metadata = {
        **copy.deepcopy(stream.metadata),
        "semantic_status": "instantiated",
        "semantic_method": semantic_method,
        "semantic_skeleton_version": SKELETON_VERSION,
    }
    if llm_attempt_count is not None:
        metadata["llm_attempt_count"] = llm_attempt_count
    return EventStream(
        stream_id=stream.stream_id,
        domain=stream.domain,
        topology=stream.topology,
        mechanism_type=stream.mechanism_type,
        primary_regime=stream.primary_regime,
        seed=stream.seed,
        mechanism=copy.deepcopy(stream.mechanism),
        events=updated_events,
        ground_truth=copy.deepcopy(stream.ground_truth),
        validation=copy.deepcopy(stream.validation),
        metadata=metadata,
        semantic_scenario=scenario,
        qa_pairs=copy.deepcopy(stream.qa_pairs),
    )


def _canonical_relation_explanations(
    stream: EventStream, llm_explanations: Any
) -> List[Dict[str, str]]:
    provided = llm_explanations if isinstance(llm_explanations, list) else []
    canonical = []
    for index, edge in enumerate(stream.mechanism.get("edges", [])):
        explanation = ""
        if index < len(provided) and isinstance(provided[index], dict):
            explanation = str(provided[index].get("explanation", "")).strip()
        if not explanation:
            source = str(edge.get("source"))
            target = str(edge.get("target"))
            explanation = f"{source} has a {edge.get('relation_type')} relation to {target}."
        canonical.append(
            {
                "source": str(edge.get("source")),
                "target": str(edge.get("target")),
                "relation_type": str(edge.get("relation_type")),
                "explanation": explanation,
            }
        )
    return canonical


def _parse_llm_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response does not contain a JSON object")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON must be an object")
    return payload


def _llm_prompt(stream: EventStream, last_error: str = "") -> str:
    mechanism_nodes = [
        str(node.get("node_id"))
        for node in stream.mechanism.get("nodes", [])
        if str(node.get("node_id", "")).startswith("E")
    ]
    mechanism = {
        "domain": stream.domain,
        "mechanism": stream.mechanism_type,
        "topology": stream.topology,
        "events": mechanism_nodes,
        "relations": [
            {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "relation_type": edge.get("relation_type"),
            }
            for edge in stream.mechanism.get("edges", [])
        ],
    }
    relation_count = len(mechanism["relations"])
    skeleton = build_semantic_skeleton(stream)
    guidance = _prompt_guidance(stream)
    retry_note = f"\nFix this previous issue: {last_error}\n" if last_error else ""
    return (
        "You instantiate abstract event mechanisms into a realistic semantic scenario.\n"
        "Return compact valid JSON only. Do not include markdown.\n"
        "You must not create, remove, rename, or reorder abstract events.\n"
        "You must not mention or alter timestamps, parent ids, root ids, regimes, "
        "delta_t, or ground truth.\n"
        f"Return exactly {len(mechanism_nodes)} event_mapping entries and exactly "
        f"{relation_count} relation_explanations.\n"
        "Each semantic event phrase should be 3-8 words. "
        "Each explanation must be one short sentence with no nested JSON.\n"
        "Each relation explanation must name the source event phrase and target event phrase.\n"
        "The source event must be sufficient to make the target event plausible without hidden intermediate events.\n"
        "Every child event must reuse or explicitly reference a concrete object, signal, artifact, or workflow handoff from its parent.\n"
        "Do not make a child depend on information that only appears in another branch or later event.\n"
        "Use the domain_causal_channel_hints in the Semantic Skeleton to keep each parent-child link on the same object, workflow, or operational signal.\n"
        "Use concrete domain-specific actors, systems, artifacts, or measurements.\n"
        f"{guidance}"
        "Required JSON schema:\n"
        "{\n"
        '  "scenario_description": "one concise scenario description",\n'
        '  "event_mapping": {"E0": "semantic event phrase", "...": "..."},\n'
        '  "relation_explanations": [\n'
        '    {"source": "E0", "target": "E1", "relation_type": "direct_trigger", '
        '"explanation": "natural language explanation"}\n'
        "  ]\n"
        "}\n"
        f"{retry_note}"
        "Mechanism JSON:\n"
        f"{json.dumps(mechanism, ensure_ascii=False, sort_keys=True)}\n"
        "Semantic Skeleton JSON:\n"
        f"{json.dumps(skeleton, ensure_ascii=False, sort_keys=True)}"
    )


def _prompt_guidance(stream: EventStream) -> str:
    parts = []
    if stream.topology == "tree":
        parts.append(
            "Tree guidance: Sibling child events must be distinct downstream consequences. "
            "Do not map siblings to overlapping actions, duplicated workflow steps, or the same outcome. "
            "For tree branches, explain why this exact parent can produce each child without borrowing evidence from a sibling."
        )
    elif stream.topology == "chain":
        parts.append(
            "Chain guidance: Each event must be the next concrete state transition, "
            "not a restatement of an earlier step."
        )

    mechanism_type = stream.mechanism_type
    if mechanism_type == "recovery":
        parts.append(
            "Recovery guidance: Use mitigation, repair, failover, rerouting, or restoration actions. "
            "For recovery, order meanings as diagnosis, containment or mitigation, repair, then verification when those roles exist. "
            "Do not say the asset repairs itself unless the event phrase names an automated recovery system. "
            "Never make a recovery child introduce a new fault, blockage, outage, or risk escalation."
        )
    elif mechanism_type == "delayed_trigger":
        parts.append(
            "Delayed trigger guidance: Explain the waiting condition or accumulation process that makes the effect delayed."
        )
    elif mechanism_type == "multi_hop_propagation":
        parts.append(
            "Multi-hop guidance: Each hop must transform the system state and make the next event more likely."
        )
    elif mechanism_type == "direct_trigger":
        parts.append(
            "Direct trigger guidance: Explain the immediate operational mechanism connecting cause and effect."
        )
    return "\n".join(parts) + ("\n" if parts else "")


def _domain_templates(domain: str, templates: Dict[str, Any]) -> Dict[str, Any]:
    domains = templates.get("domains", {})
    if domain in domains:
        return domains[domain]
    fallback = templates.get("fallback_domain")
    if fallback and fallback in domains:
        return domains[fallback]
    raise ValueError(f"No semantic templates configured for domain={domain}")


def _phrase(phrases: List[str], index: int) -> str:
    phrase = phrases[index % len(phrases)]
    repeat = index // len(phrases)
    return phrase if repeat == 0 else f"{phrase} #{repeat + 1}"


def _relation_explanation(
    edge: Dict[str, Any], mapping: Dict[str, str]
) -> Dict[str, str]:
    source = str(edge["source"])
    target = str(edge["target"])
    relation = str(edge["relation_type"])
    source_phrase = mapping.get(source, source)
    target_phrase = mapping.get(target, target)
    return {
        "source": source,
        "target": target,
        "relation_type": relation,
        "explanation": (
            f"{source_phrase} has a {relation.replace('_', ' ')} relation "
            f"to {target_phrase}."
        ),
    }


def _scenario_description(stream: EventStream, mapping: Dict[str, str]) -> str:
    ordered = [mapping[key] for key in sorted(mapping, key=_node_sort_key)]
    return (
        f"In the {stream.domain} domain, a {stream.topology} "
        f"{stream.mechanism_type} event stream links "
        f"{'; '.join(ordered)}."
    )


def _node_sort_key(node_id: str) -> int:
    try:
        return int(node_id[1:])
    except ValueError:
        return 0
