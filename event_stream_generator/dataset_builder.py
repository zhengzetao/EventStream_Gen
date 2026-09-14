"""Dataset build pipeline for accepted/rejected event-stream outputs."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .cli import run_script_main
from .coverage_report import build_coverage_report
from .core.io import load_yaml, write_json, write_jsonl, write_yaml
from .export.training_export import export_training_data
from .semantic.gate import apply_semantic_quality_gate
from .validators.diversity import validate_dataset_diversity


def build_dataset(config: Dict[str, Any] | str | Path) -> Dict[str, Any]:
    resolved = load_yaml(config) if isinstance(config, (str, Path)) else dict(config)
    generation_mode = str(resolved.get("generation_mode", "generic"))
    if generation_mode not in {"generic", "calibrated"}:
        raise ValueError("generation_mode must be 'generic' or 'calibrated'")
    output_dir = Path(str(resolved["output_dir"]))
    accepted_dir = output_dir / "accepted"
    rejected_dir = output_dir / "rejected"
    reports_dir = output_dir / "reports"
    configs_dir = output_dir / "configs"
    training_dir = output_dir / "training"
    for directory in (accepted_dir, rejected_dir, reports_dir, configs_dir, training_dir):
        directory.mkdir(parents=True, exist_ok=True)

    accepted_path = accepted_dir / "samples.jsonl"
    args = _generation_args(resolved, generation_mode, accepted_path)
    module_name = (
        "event_stream_generator.scripts.generate_generic"
        if generation_mode == "generic"
        else "event_stream_generator.scripts.generate_calibrated"
    )
    exit_code = run_script_main(module_name, args)
    if exit_code != 0:
        raise RuntimeError(f"{generation_mode} generation failed with exit_code={exit_code}")

    generated_rejected = accepted_path.with_name("samples_rejected.jsonl")
    generated_coverage = accepted_path.with_name("samples.coverage.json")
    rejected_path = rejected_dir / "rejected.jsonl"
    coverage_path = reports_dir / "coverage.json"
    diversity_path = reports_dir / "diversity.json"
    semantic_gate_path = reports_dir / "semantic_quality_gate.json"
    training_path = training_dir / "train.jsonl"
    training_report_path = reports_dir / "training_export.json"
    _move_if_exists(generated_rejected, rejected_path)
    _move_if_exists(generated_coverage, coverage_path)
    accepted_samples = _read_jsonl(accepted_path)
    rejected_samples = _read_jsonl(rejected_path)
    accepted_samples, rejected_samples, semantic_gate = apply_semantic_quality_gate(
        accepted_samples,
        rejected_samples,
        resolved.get("semantic_quality", {}),
    )
    write_jsonl(accepted_path, accepted_samples)
    write_jsonl(rejected_path, rejected_samples)
    write_json(semantic_gate_path, semantic_gate)
    write_json(coverage_path, build_coverage_report(accepted_samples))
    diversity = validate_dataset_diversity(accepted_samples, resolved)
    write_json(diversity_path, diversity)
    training_export_config = resolved.get("training_export", {}) or {}
    if training_export_config.get("enabled", False):
        training_report = export_training_data(
            accepted_jsonl=accepted_path,
            output_jsonl=training_path,
            output_format=str(training_export_config.get("format", "instruction_jsonl")),
        )
        write_json(training_report_path, training_report)
    else:
        training_path.write_text("", encoding="utf-8")
        write_json(
            training_report_path,
            {
                "enabled": False,
                "num_input_streams": len(accepted_samples),
                "num_training_examples": 0,
            },
        )

    resolved_config_path = configs_dir / "resolved_config.yaml"
    write_yaml(resolved_config_path, resolved)
    manifest = _build_manifest(
        config=resolved,
        generation_mode=generation_mode,
        entry_command=_entry_command(resolved),
        accepted_path=accepted_path,
        rejected_path=rejected_path,
        coverage_path=coverage_path,
        diversity_path=diversity_path,
        semantic_gate_path=semantic_gate_path,
        training_path=training_path,
        training_report_path=training_report_path,
        resolved_config_path=resolved_config_path,
    )
    manifest_path = reports_dir / "manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_hash"] = _file_hash(manifest_path)
    write_json(manifest_path, manifest)
    print(
        f"built dataset mode={generation_mode} accepted={manifest['num_accepted']} "
        f"rejected={manifest['num_rejected']} output_dir={output_dir}"
    )
    return manifest


def _generation_args(
    config: Dict[str, Any], generation_mode: str, accepted_path: Path
) -> List[str]:
    args = [
        "--num-samples",
        str(config["num_samples"]),
        "--output",
        str(accepted_path),
        "--seed",
        str(config.get("seed", 0)),
        "--semantic-mode",
        str(config.get("semantic_mode", "none")),
        "--qa-mode",
        str(config.get("qa_mode", "none")),
    ]
    scalar_keys = {
        "prior": "--prior",
        "taxonomy": "--taxonomy",
        "semantic_templates": "--semantic-templates",
        "topology": "--topology",
        "mechanism_type": "--mechanism-type",
        "llm_cache": "--llm-cache",
        "progress_path": "--progress-path",
        "llm_concurrency": "--llm-concurrency",
    }
    list_keys = {
        "domains": "--domains",
        "topologies": "--topologies",
        "mechanism_types": "--mechanism-types",
        "temporal_regimes": "--temporal-regimes",
    }
    if generation_mode == "calibrated":
        args.extend(["--calibration-prior", str(config["calibration_prior"])])
    for key, flag in scalar_keys.items():
        if key in config and config[key] is not None:
            args.extend([flag, str(config[key])])
    if generation_mode == "generic":
        for key, flag in list_keys.items():
            if key in config and config[key] is not None:
                args.append(flag)
                args.extend(str(item) for item in _as_list(config[key]))
    return args


def _build_manifest(
    *,
    config: Dict[str, Any],
    generation_mode: str,
    entry_command: List[str],
    accepted_path: Path,
    rejected_path: Path,
    coverage_path: Path,
    diversity_path: Path,
    semantic_gate_path: Path,
    training_path: Path,
    training_report_path: Path,
    resolved_config_path: Path,
) -> Dict[str, Any]:
    num_accepted = _count_jsonl(accepted_path)
    num_rejected = _count_jsonl(rejected_path)
    outputs = {
        "accepted": str(accepted_path),
        "rejected": str(rejected_path),
        "coverage": str(coverage_path),
        "diversity": str(diversity_path),
        "semantic_quality_gate": str(semantic_gate_path),
        "training": str(training_path),
        "training_export": str(training_report_path),
        "resolved_config": str(resolved_config_path),
    }
    return {
        "run_id": str(config.get("run_id") or accepted_path.parents[1].name),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator_version": "event_stream_generator_v1",
        "python_version": sys.version.split()[0],
        "entry_command": entry_command,
        "generation_mode": generation_mode,
        "seed": int(config.get("seed", 0)),
        "num_requested": int(config["num_samples"]),
        "num_accepted": num_accepted,
        "num_rejected": num_rejected,
        "config_hash": _stable_hash(config),
        "input_hashes": _input_hashes(config),
        "output_hashes": _output_hashes(outputs),
        "outputs": outputs,
    }


def _move_if_exists(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.move(str(source), str(target))
    else:
        target.write_text("", encoding="utf-8")


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stable_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_hashes(config: Dict[str, Any]) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for key in ("calibration_prior", "prior", "taxonomy", "semantic_templates"):
        value = config.get(key)
        if value is None:
            continue
        path = Path(str(value))
        if path.exists():
            hashes[key] = _file_hash(path)
    return hashes


def _output_hashes(outputs: Dict[str, str]) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for key, value in outputs.items():
        path = Path(value)
        if path.exists() and path.is_file():
            hashes[key] = _file_hash(path)
    return hashes


def _entry_command(config: Dict[str, Any]) -> List[str]:
    command = ["build_dataset"]
    for key in ("generation_mode", "output_dir", "num_samples", "seed", "semantic_mode", "qa_mode"):
        if key in config:
            command.extend([f"--{key.replace('_', '-')}", str(config[key])])
    if config.get("generation_mode") == "calibrated" and config.get("calibration_prior"):
        command.extend(["--calibration-prior", str(config["calibration_prior"])])
    return command


def _as_list(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        return value
    return [value]
