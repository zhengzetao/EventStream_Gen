"""Export accepted event streams into LLM training JSONL formats."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..core.io import write_jsonl


SUPPORTED_FORMATS = {"instruction_jsonl", "chat_jsonl"}
SYSTEM_PROMPT = (
    "You answer event-stream understanding questions using only the visible "
    "event IDs, timestamps, and event descriptions."
)


def export_training_data(
    *,
    accepted_jsonl: str | Path,
    output_jsonl: str | Path,
    output_format: str = "instruction_jsonl",
) -> Dict[str, Any]:
    if output_format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported training export format: {output_format}")
    streams = _read_jsonl(accepted_jsonl)
    examples = []
    for row in streams:
        examples.extend(_examples_for_stream(row, output_format))
    write_jsonl(output_jsonl, examples)
    return _report(streams, examples, output_jsonl, output_format)


def render_stream_input(row: Dict[str, Any]) -> str:
    lines = [
        "Event stream:",
        f"domain={row.get('domain', 'unknown')}",
        f"topology={row.get('topology', 'unknown')}",
        "events in timestamp order:",
    ]
    events = sorted(
        row.get("events", []) or [],
        key=lambda event: (float(event.get("timestamp", 0.0)), int(event.get("event_id", 0))),
    )
    for event in events:
        event_id = event.get("event_id")
        timestamp = event.get("timestamp")
        semantic = event.get("semantic_type") or event.get("abstract_type") or "event"
        lines.append(f"- event_id={event_id}; timestamp={timestamp}; event={semantic}")
    return "\n".join(lines)


def _examples_for_stream(row: Dict[str, Any], output_format: str) -> List[Dict[str, Any]]:
    stream_input = render_stream_input(row)
    examples = []
    for qa in row.get("qa_pairs", []) or []:
        if output_format == "instruction_jsonl":
            examples.append(_instruction_example(row, qa, stream_input))
        elif output_format == "chat_jsonl":
            examples.append(_chat_example(row, qa, stream_input))
    return examples


def _instruction_example(
    row: Dict[str, Any],
    qa: Dict[str, Any],
    stream_input: str,
) -> Dict[str, Any]:
    return {
        "id": _example_id(row, qa),
        "source_stream_id": row.get("stream_id"),
        "qa_id": qa.get("qa_id"),
        "task_type": qa.get("qa_type"),
        "instruction": qa.get("question", ""),
        "input": stream_input,
        "output": _answer_text(qa.get("answer")),
        "answer": qa.get("answer"),
        "answer_source": qa.get("answer_source"),
        "metadata": _metadata(row),
    }


def _chat_example(
    row: Dict[str, Any],
    qa: Dict[str, Any],
    stream_input: str,
) -> Dict[str, Any]:
    return {
        "id": _example_id(row, qa),
        "source_stream_id": row.get("stream_id"),
        "qa_id": qa.get("qa_id"),
        "task_type": qa.get("qa_type"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{stream_input}\n\nQuestion: {qa.get('question', '')}",
            },
            {"role": "assistant", "content": _answer_text(qa.get("answer"))},
        ],
        "answer": qa.get("answer"),
        "answer_source": qa.get("answer_source"),
        "metadata": _metadata(row),
    }


def _metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    metadata = row.get("metadata", {}) or {}
    return {
        "domain": row.get("domain"),
        "topology": row.get("topology"),
        "mechanism_type": row.get("mechanism_type"),
        "primary_regime": row.get("primary_regime"),
        "semantic_method": metadata.get("semantic_method"),
        "semantic_skeleton_version": metadata.get("semantic_skeleton_version"),
        "semantic_compatibility": metadata.get("semantic_compatibility"),
    }


def _example_id(row: Dict[str, Any], qa: Dict[str, Any]) -> str:
    return f"{row.get('stream_id')}:{qa.get('qa_id')}"


def _answer_text(answer: Any) -> str:
    if isinstance(answer, (list, dict)):
        return json.dumps(answer, ensure_ascii=False)
    return str(answer)


def _report(
    streams: List[Dict[str, Any]],
    examples: List[Dict[str, Any]],
    output_jsonl: str | Path,
    output_format: str,
) -> Dict[str, Any]:
    task_counts = Counter(str(example.get("task_type")) for example in examples)
    return {
        "output_format": output_format,
        "output_jsonl": str(output_jsonl),
        "num_input_streams": len(streams),
        "num_training_examples": len(examples),
        "task_type_counts": dict(sorted(task_counts.items())),
    }


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    return [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
