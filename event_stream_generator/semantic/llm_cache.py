"""Persistent cache for LLM semantic instantiation results."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from event_stream_generator.core.models import EventStream
from event_stream_generator.skeleton.semantic_skeleton import SKELETON_VERSION


CACHE_VERSION = "semantic_cache_v1"


def make_semantic_cache_key(
    stream: EventStream,
    *,
    prompt_version: str = "llm_semantic_prompt_v1",
) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "prompt_version": prompt_version,
        "skeleton_version": SKELETON_VERSION,
        "stream_id": stream.stream_id,
        "seed": stream.seed,
        "domain": stream.domain,
        "topology": stream.topology,
        "mechanism_type": stream.mechanism_type,
        "primary_regime": stream.primary_regime,
        "mechanism": stream.mechanism,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SemanticCache:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._items: Dict[str, Dict[str, Any]] = {}
        if self.path and self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._items = {
                    str(key): value
                    for key, value in payload.get("items", {}).items()
                    if isinstance(value, dict)
                }

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        value = self._items.get(key)
        return copy.deepcopy(value) if value is not None else None

    def set(self, key: str, scenario: Dict[str, Any]) -> None:
        self._items[key] = copy.deepcopy(scenario)

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_version": CACHE_VERSION,
            "items": self._items,
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
