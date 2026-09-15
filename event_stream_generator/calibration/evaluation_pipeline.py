"""End-to-end calibration evaluation pipeline."""

from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List

from event_stream_generator.calibration.pipeline import run_calibration_pipeline
from event_stream_generator.core.io import load_yaml, write_json
from event_stream_generator.evaluation.compare import compare_generation_outputs


def run_calibration_evaluation_pipeline(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    config = load_yaml(path)
    evaluation = dict(config.get("evaluation") or {})
    dataset_name = str(config.get("dataset_name") or config.get("adapter") or "dataset")
    output_dir = _resolve_path(evaluation.get("output_dir", f"data/calibration/evaluation/{dataset_name}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_result = run_calibration_pipeline(path)
    prior_output = Path(calibration_result["report"]["prior_output"])
    prepared_output = Path(calibration_result["report"]["prepared_output"])

    num_samples = int(evaluation.get("num_samples", 100))
    seed = int(evaluation.get("seed", 0))
    topology = str(evaluation.get("topology", "chain"))
    mechanism_type = str(evaluation.get("mechanism_type", "multi_hop_propagation"))
    synthetic_domain = _generation_domain(config, evaluation)
    semantic_mode = str(evaluation.get("semantic_mode", "none"))
    qa_mode = str(evaluation.get("qa_mode", "none"))
    calibrated_generation = dict(evaluation.get("calibrated_generation") or {})
    generic_output = _resolve_path(evaluation.get("generic_output", output_dir / "generic.jsonl"))
    calibrated_output = _resolve_path(
        evaluation.get("calibrated_output", output_dir / "calibrated.jsonl")
    )
    comparison_json = _resolve_path(evaluation.get("comparison_json", output_dir / "comparison.json"))
    comparison_csv = _resolve_path(evaluation.get("comparison_csv", output_dir / "comparison.csv"))
    summary_output = _resolve_path(evaluation.get("summary_output", output_dir / "summary.json"))

    _run_script(
        "event_stream_generator.scripts.generate_generic",
        [
            "--num-samples",
            str(num_samples),
            "--output",
            str(generic_output),
            "--seed",
            str(seed),
            "--domains",
            synthetic_domain,
            "--topologies",
            topology,
            "--mechanism-types",
            mechanism_type,
            "--semantic-mode",
            semantic_mode,
            "--qa-mode",
            qa_mode,
        ],
    )
    _run_script(
        "event_stream_generator.scripts.generate_calibrated",
        [
            "--calibration-prior",
            str(prior_output),
            "--num-samples",
            str(num_samples),
            "--output",
            str(calibrated_output),
            "--seed",
            str(seed + 1000003),
            "--topology",
            topology,
            "--mechanism-type",
            mechanism_type,
            "--semantic-mode",
            semantic_mode,
            "--qa-mode",
            qa_mode,
            *_calibrated_generation_args(calibrated_generation),
        ],
    )
    comparison, rows = compare_generation_outputs(
        real_csv=prepared_output,
        generic_jsonl=generic_output,
        calibrated_jsonl=calibrated_output,
        timestamp_col="timestamp",
        stream_id_col="stream_id",
        event_type_col="event_type",
    )
    write_json(comparison_json, comparison)
    _write_rows(comparison_csv, rows)
    effect = summarize_calibration_effect(comparison)
    summary = {
        "dataset_name": dataset_name,
        "domain": config["domain"],
        "synthetic_domain": synthetic_domain,
        "calibration_report": calibration_result["report"],
        "generation": {
            "num_samples": num_samples,
            "seed": seed,
            "topology": topology,
            "mechanism_type": mechanism_type,
            "generic_output": str(generic_output),
            "calibrated_output": str(calibrated_output),
            "calibrated_generation": calibrated_generation,
        },
        "comparison_json": str(comparison_json),
        "comparison_csv": str(comparison_csv),
        "calibration_effect": effect,
    }
    write_json(summary_output, summary)
    summary["summary_output"] = str(summary_output)
    return summary


def summarize_calibration_effect(report: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    effect: Dict[str, Dict[str, float]] = {}
    metric_map = {
        "inter_event_time_ks": ("inter_event_time", "ks"),
        "inter_event_time_wasserstein": ("inter_event_time", "wasserstein"),
        "sequence_length_ks": ("sequence_length", "ks"),
        "sequence_length_wasserstein": ("sequence_length", "wasserstein"),
        "event_type_frequency_l1": ("event_type_frequency", "l1"),
        "transition_frequency_l1": ("transition_frequency", "l1"),
    }
    for name, (section, metric) in metric_map.items():
        generic = float(report[section]["generic_vs_real"][metric])
        calibrated = float(report[section]["calibrated_vs_real"][metric])
        absolute = generic - calibrated
        relative = absolute / generic if generic else 0.0
        effect[name] = {
            "generic": round(generic, 9),
            "calibrated": round(calibrated, 9),
            "absolute_improvement": round(absolute, 9),
            "relative_improvement": round(relative, 9),
            "improved": calibrated < generic,
        }
    return effect


def _generation_domain(config: Dict[str, Any], evaluation: Dict[str, Any]) -> str:
    return str(
        evaluation.get("synthetic_domain")
        or config.get("synthetic_domain")
        or config["domain"]
    )


def _run_script(module_name: str, forwarded_args: List[str]) -> None:
    module = importlib.import_module(module_name)
    previous_argv = sys.argv[:]
    sys.argv = [module_name, *forwarded_args]
    try:
        result = module.main()
    finally:
        sys.argv = previous_argv
    if int(result or 0) != 0:
        raise RuntimeError(f"{module_name} failed with exit code {result}")


def _calibrated_generation_args(options: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    if "use_second_order" in options:
        args.extend(
            [
                "--calibrated-use-second-order",
                "true" if bool(options["use_second_order"]) else "false",
            ]
        )
    if "smoothing_alpha" in options:
        args.extend(["--calibrated-smoothing-alpha", str(options["smoothing_alpha"])])
    if "temperature" in options:
        args.extend(["--calibrated-temperature", str(options["temperature"])])
    return args


def _write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "comparison", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _resolve_path(raw_path: Any) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return Path.cwd() / path
