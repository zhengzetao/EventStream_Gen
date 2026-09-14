"""Unified command-line dispatcher for the event-stream generator."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Callable, List, Sequence

from event_stream_generator.core.io import load_yaml

ScriptRunner = Callable[[str, List[str]], int]


MODE_TO_MODULE = {
    "audit-semantics": "event_stream_generator.scripts.audit_semantics",
    "build-dataset": "event_stream_generator.scripts.build_dataset",
    "generic": "event_stream_generator.scripts.generate_generic",
    "calibrated": "event_stream_generator.scripts.generate_calibrated",
    "calibrate": "event_stream_generator.scripts.calibrate_events",
    "calibrate-hdfs": "event_stream_generator.scripts.calibrate_hdfs",
    "evaluate": "event_stream_generator.scripts.evaluate_generation",
    "evaluate-llm-semantics": "event_stream_generator.scripts.evaluate_llm_semantics",
    "export-training": "event_stream_generator.scripts.export_training",
}


def main(argv: Sequence[str] | None = None, runner: ScriptRunner | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Unified entry point for synthetic event-stream generation."
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_TO_MODULE),
        help=(
            "Operation to run: generic, calibrated, calibrate, calibrate-hdfs, "
            "or evaluate."
        ),
    )
    parser.add_argument(
        "--config",
        help="Optional YAML config that supplies mode and default CLI arguments.",
    )
    parsed, forwarded_args = parser.parse_known_args(args)
    if parsed.config:
        if parsed.mode == "build-dataset" or (
            parsed.mode is None and _config_mode(Path(parsed.config)) == "build-dataset"
        ):
            dispatch = runner or run_script_main
            return dispatch(MODE_TO_MODULE["build-dataset"], ["--config", parsed.config, *forwarded_args])
        config_args = _args_from_config(Path(parsed.config))
        if parsed.mode:
            config_args = ["--mode", parsed.mode, *config_args]
        args = [*config_args, *forwarded_args]
        parsed, forwarded_args = parser.parse_known_args(args)
    if not parsed.mode:
        parser.error("--mode is required unless supplied by --config")
    dispatch = runner or run_script_main
    return dispatch(MODE_TO_MODULE[parsed.mode], forwarded_args)


def run_script_main(module_name: str, forwarded_args: List[str]) -> int:
    module = importlib.import_module(module_name)
    previous_argv = sys.argv[:]
    sys.argv = [module_name, *forwarded_args]
    try:
        result = module.main()
    finally:
        sys.argv = previous_argv
    return int(result or 0)


def _args_from_config(path: Path) -> List[str]:
    config = load_yaml(path)
    mode = str(config.get("mode", "generic"))
    args = ["--mode", mode]
    key_to_arg = {
        "num_samples": "--num-samples",
        "output": "--output",
        "seed": "--seed",
        "semantic_mode": "--semantic-mode",
        "semantic_templates": "--semantic-templates",
        "qa_mode": "--qa-mode",
        "prior": "--prior",
        "taxonomy": "--taxonomy",
        "calibration_prior": "--calibration-prior",
        "input": "--input",
        "timestamp_col": "--timestamp-col",
        "stream_id_col": "--stream-id-col",
        "event_type_col": "--event-type-col",
        "domain": "--domain",
        "min_transition_samples": "--min-transition-samples",
        "real_csv": "--real-csv",
        "generic_jsonl": "--generic-jsonl",
        "calibrated_jsonl": "--calibrated-jsonl",
        "output_json": "--output-json",
        "output_csv": "--output-csv",
        "topology": "--topology",
        "mechanism_type": "--mechanism-type",
        "llm_concurrency": "--llm-concurrency",
        "llm_cache": "--llm-cache",
        "progress_path": "--progress-path",
    }
    list_key_to_arg = {
        "domains": "--domains",
        "topologies": "--topologies",
        "mechanism_types": "--mechanism-types",
        "temporal_regimes": "--temporal-regimes",
    }
    for key, arg_name in key_to_arg.items():
        if key in config and config[key] is not None:
            args.extend([arg_name, str(config[key])])
    for key, arg_name in list_key_to_arg.items():
        if key in config and config[key] is not None:
            args.append(arg_name)
            args.extend(str(item) for item in config[key])
    return args


def _config_mode(path: Path) -> str:
    return str(load_yaml(path).get("mode", "generic"))
