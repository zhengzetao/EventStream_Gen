"""Generic calibration pipeline with dataset adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from event_stream_generator.calibration.adapters import get_adapter
from event_stream_generator.calibration.extractor import calibrate_event_csv
from event_stream_generator.core.io import load_yaml, write_json


def run_calibration_pipeline(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    config = load_yaml(path)
    dataset_name = str(config.get("dataset_name") or config.get("adapter") or "dataset")
    adapter_name = str(config["adapter"])
    domain = str(config["domain"])
    input_path = _resolve_path(path, config["input_path"])
    prepared_output = _resolve_path(path, config["prepared_output"])
    prior_output = _resolve_path(path, config["prior_output"])
    report_output = _resolve_path(
        path,
        config.get("report_output") or f"{prior_output.with_suffix('')}_report.json",
    )
    adapter_options = dict(config.get("adapter_options") or {})
    min_transition_samples = int(config.get("min_transition_samples", 5))

    adapter_summary = get_adapter(adapter_name).prepare(
        input_path,
        prepared_output,
        adapter_options,
    )
    prior = calibrate_event_csv(
        prepared_output,
        timestamp_col="timestamp",
        stream_id_col="stream_id",
        event_type_col="event_type",
        domain=domain,
        min_transition_samples=min_transition_samples,
    )
    prior["calibration_dataset"] = {
        "dataset_name": dataset_name,
        "adapter": adapter_name,
        "source_input_path": str(input_path),
        "prepared_output": str(prepared_output),
    }
    write_json(prior_output, prior)
    report = {
        "dataset_name": dataset_name,
        "adapter": adapter_name,
        "domain": domain,
        "input_path": str(input_path),
        "prepared_output": str(prepared_output),
        "prior_output": str(prior_output),
        "adapter_summary": adapter_summary,
        "prior_summary": {
            "stream_count": prior["stream_count"],
            "event_count": prior["event_count"],
            "event_type_count": len(prior["event_frequencies"]),
            "transition_prior_count": len(prior["transition_priors"]),
            "min_transition_samples": min_transition_samples,
        },
    }
    write_json(report_output, report)
    return {
        "dataset_name": dataset_name,
        "adapter_summary": adapter_summary,
        "prior": prior,
        "report": report,
    }


def _resolve_path(config_path: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    del config_path
    return Path.cwd() / path
