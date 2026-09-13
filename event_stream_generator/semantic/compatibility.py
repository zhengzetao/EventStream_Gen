"""Semantic-compatible mechanism sampling constraints."""

from __future__ import annotations

import copy
from typing import Any, Dict, Tuple


SEMANTIC_COMPATIBILITY_VERSION = "v2"
PROCESS_LIKE_DOMAINS = {"Finance", "Healthcare", "IT/System", "Scientific Process"}


def apply_semantic_compatible_sampling(
    base_config: Dict[str, Any],
    combo: Dict[str, str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config = copy.deepcopy(base_config)
    metadata = {
        "enabled": True,
        "version": SEMANTIC_COMPATIBILITY_VERSION,
        "rules_applied": [],
    }
    topology = combo.get("topology", "")
    mechanism_type = combo.get("mechanism_type", "")
    domain = combo.get("domain", "")

    if topology == "tree":
        _limit_tree(config, max_nodes=10, max_depth=3, max_children=2)
        metadata["rules_applied"].append("limited generic tree branching")
        metadata["rules_applied"].append("limited tree semantic branch load")
        if domain in PROCESS_LIKE_DOMAINS:
            _limit_tree(config, max_nodes=10, max_depth=3, max_children=2)
            metadata["rules_applied"].append("limited process-like tree depth")
        if mechanism_type == "recovery":
            _limit_tree(config, max_nodes=8, max_depth=3, max_children=2)
            metadata["rules_applied"].append("limited recovery tree size")

    if topology == "chain":
        _limit_chain(config, max_length=8)
        metadata["rules_applied"].append("limited chain semantic span")
        if mechanism_type == "recovery":
            _limit_chain(config, max_length=6)
            metadata["rules_applied"].append("limited recovery chain length")

    if mechanism_type == "recovery":
        stream = config.setdefault("stream", {})
        if int(stream.get("max_events", 0)) > 0:
            stream["max_events"] = min(int(stream["max_events"]), 45)
            metadata["rules_applied"].append("limited recovery total events")

    return config, metadata


def disabled_semantic_compatibility_metadata() -> Dict[str, Any]:
    return {
        "enabled": False,
        "version": SEMANTIC_COMPATIBILITY_VERSION,
        "rules_applied": [],
    }


def _limit_tree(
    config: Dict[str, Any],
    *,
    max_nodes: int,
    max_depth: int,
    max_children: int,
) -> None:
    tree = config.setdefault("mechanism", {}).setdefault("tree", {})
    tree["max_nodes"] = min(int(tree.get("max_nodes", max_nodes)), max_nodes)
    tree["max_depth"] = min(int(tree.get("max_depth", max_depth)), max_depth)
    tree["max_children_per_node"] = min(
        int(tree.get("max_children_per_node", max_children)),
        max_children,
    )
    min_nodes = int(tree.get("min_nodes", 2))
    if min_nodes > int(tree["max_nodes"]):
        tree["min_nodes"] = int(tree["max_nodes"])


def _limit_chain(config: Dict[str, Any], *, max_length: int) -> None:
    chain = config.setdefault("mechanism", {}).setdefault("chain", {})
    chain["max_length"] = min(int(chain.get("max_length", max_length)), max_length)
    min_length = int(chain.get("min_length", 2))
    if min_length > int(chain["max_length"]):
        chain["min_length"] = int(chain["max_length"])
