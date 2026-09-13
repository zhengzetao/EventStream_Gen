"""I/O helpers for generator configuration and JSONL outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def load_yaml(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read generator YAML config. Install with: pip install pyyaml"
        ) from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_yaml(path: str | Path, payload: Dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to write generator YAML config. Install with: pip install pyyaml"
        ) from exc
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True, allow_unicode=True)
