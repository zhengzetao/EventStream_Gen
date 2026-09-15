#!/usr/bin/env python3
"""Generate synthetic streams from a calibrated transition prior."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from event_stream_generator.coverage_report import build_coverage_report
from event_stream_generator.core.io import load_yaml, write_json, write_jsonl
from event_stream_generator.qa import generate_all_qa
from event_stream_generator.semantic.compatibility import (
    apply_semantic_compatible_sampling,
    disabled_semantic_compatibility_metadata,
)
from event_stream_generator.semantic.instantiation import (
    instantiate_scenario,
    instantiate_scenario_llm,
    instantiate_scenario_llm_judge_guided,
)
from event_stream_generator.semantic.llm_client import OpenAIResponsesClient
from event_stream_generator.simulation.calibrated import generate_calibrated_event_stream
from event_stream_generator.validators.label import validate_qa_pairs
from event_stream_generator.validators.semantic import validate_semantic
from event_stream_generator.validators.structural import validate_structure
from event_stream_generator.validators.temporal import validate_temporal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_config_dir = Path(__file__).resolve().parents[1] / "config"
    parser.add_argument("--calibration-prior", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prior", default=str(default_config_dir / "generic_prior.yaml"))
    parser.add_argument("--topology", choices=["chain", "tree"], default="chain")
    parser.add_argument("--mechanism-type", default="multi_hop_propagation")
    parser.add_argument(
        "--calibrated-use-second-order",
        choices=["true", "false"],
        help="Override calibrated_generation.use_second_order from the prior config.",
    )
    parser.add_argument(
        "--calibrated-smoothing-alpha",
        type=float,
        help="Override calibrated_generation.smoothing_alpha from the prior config.",
    )
    parser.add_argument(
        "--calibrated-temperature",
        type=float,
        help="Override calibrated_generation.temperature from the prior config.",
    )
    parser.add_argument("--semantic-mode", choices=["none", "rule", "llm"], default="none")
    parser.add_argument(
        "--semantic-judge-mode",
        choices=["none", "llm"],
        default="none",
        help="Optionally require an LLM semantic judge to approve LLM-instantiated scenarios.",
    )
    parser.add_argument(
        "--semantic-judge-threshold",
        type=float,
        default=0.8,
        help="Minimum LLM judge score required when --semantic-judge-mode=llm.",
    )
    parser.add_argument(
        "--semantic-judge-max-retry",
        type=int,
        default=3,
        help="Maximum semantic regeneration attempts after LLM judge rejection.",
    )
    parser.add_argument(
        "--semantic-templates",
        default=str(default_config_dir / "semantic_templates.yaml"),
    )
    parser.add_argument("--qa-mode", choices=["none", "template"], default="none")
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.semantic_judge_max_retry <= 0:
        raise ValueError("--semantic-judge-max-retry must be positive")
    if args.semantic_judge_mode == "llm" and args.semantic_mode != "llm":
        parser.error("--semantic-judge-mode=llm requires --semantic-mode=llm")
    config = _adapt_config_for_calibration(
        load_yaml(args.prior),
        _load_calibration_prior(args.calibration_prior),
    )
    _apply_calibrated_generation_overrides(config, args)
    calibration_prior = _load_calibration_prior(args.calibration_prior)
    semantic_templates = (
        load_yaml(args.semantic_templates) if args.semantic_mode == "rule" else {}
    )
    llm_client = OpenAIResponsesClient.from_env() if args.semantic_mode == "llm" else None
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    max_retry = int(config.get("stream", {}).get("max_generation_retry", 5))

    for index in range(args.num_samples):
        accepted_sample = None
        for attempt in range(max_retry):
            sample_seed = args.seed + index * 1009 + attempt * 104729
            sample_config, compatibility_metadata = _config_for_sample(
                config,
                calibration_prior,
                topology=args.topology,
                mechanism_type=args.mechanism_type,
                semantic_mode=args.semantic_mode,
            )
            stream = generate_calibrated_event_stream(
                stream_id=f"calibrated_{index:06d}",
                calibration_prior=calibration_prior,
                seed=sample_seed,
                config=sample_config,
                topology=args.topology,
                mechanism_type=args.mechanism_type,
            )
            stream.metadata["background_density"] = sample_config.get("background", {}).get(
                "density", "medium"
            )
            stream.metadata["semantic_compatibility"] = compatibility_metadata
            structural = validate_structure(stream, config=sample_config)
            temporal = validate_temporal(stream, config=sample_config)
            semantic = None
            label = None
            if structural.approved and temporal.approved:
                if args.semantic_mode == "rule":
                    stream = instantiate_scenario(
                        stream,
                        semantic_templates,
                        seed=sample_seed + 17,
                    )
                    stream.metadata.update(
                        {
                            "generation_mode": "calibrated",
                            "calibration_domain": calibration_prior.get("domain", "Calibrated"),
                            "calibration_type": calibration_prior.get("calibration_type"),
                        }
                    )
                    semantic = validate_semantic(stream)
                elif args.semantic_mode == "llm":
                    try:
                        if args.semantic_judge_mode == "llm":
                            stream = instantiate_scenario_llm_judge_guided(
                                stream,
                                semantic_client=llm_client,
                                judge_client=llm_client,
                                max_retry=args.semantic_judge_max_retry,
                                approval_threshold=args.semantic_judge_threshold,
                            )
                        else:
                            stream = instantiate_scenario_llm(
                                stream,
                                llm_client,
                                max_retry=3,
                            )
                        stream.metadata.update(
                            {
                                "generation_mode": "calibrated",
                                "calibration_domain": calibration_prior.get("domain", "Calibrated"),
                                "calibration_type": calibration_prior.get("calibration_type"),
                            }
                        )
                        semantic = validate_semantic(stream)
                    except Exception as exc:
                        semantic = validate_semantic(stream)
                        semantic.errors.append(str(exc))
            if (
                structural.approved
                and temporal.approved
                and (semantic is None or semantic.approved)
                and args.qa_mode == "template"
            ):
                stream.qa_pairs = generate_all_qa(stream)
                label = validate_qa_pairs(stream, stream.qa_pairs)
            stream.validation = {
                "structural": structural.to_dict(),
                "temporal": temporal.to_dict(),
            }
            if semantic is not None:
                stream.validation["semantic"] = semantic.to_dict()
            if label is not None:
                stream.validation["label"] = label.to_dict()
            if (
                structural.approved
                and temporal.approved
                and (semantic is None or semantic.approved)
                and (label is None or label.approved)
            ):
                accepted_sample = stream.to_dict()
                break
            rejected.append(
                {
                    "stream_id": f"calibrated_{index:06d}_attempt_{attempt + 1}",
                    "stage": _stage(structural, temporal, semantic, label),
                    "approved": False,
                    "errors": (
                        structural.errors
                        + temporal.errors
                        + (semantic.errors if semantic is not None else [])
                        + (label.errors if label is not None else [])
                    ),
                    "sample": stream.to_dict(),
                }
            )
        if accepted_sample is None:
            return 1
        accepted.append(accepted_sample)

    output = Path(args.output)
    write_jsonl(output, accepted)
    write_jsonl(output.with_name(f"{output.stem}_rejected{output.suffix}"), rejected)
    write_json(output.with_name(f"{output.stem}.coverage.json"), build_coverage_report(accepted))
    print(
        f"generated calibrated accepted={len(accepted)} rejected={len(rejected)} output={output}"
    )
    return 0


def _load_calibration_prior(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _apply_calibrated_generation_overrides(config: Dict[str, Any], args) -> None:
    options = config.setdefault("calibrated_generation", {})
    if args.calibrated_use_second_order is not None:
        options["use_second_order"] = args.calibrated_use_second_order == "true"
    if args.calibrated_smoothing_alpha is not None:
        options["smoothing_alpha"] = args.calibrated_smoothing_alpha
    if args.calibrated_temperature is not None:
        options["temperature"] = args.calibrated_temperature


def _adapt_config_for_calibration(
    config: Dict[str, Any], calibration_prior: Dict[str, Any]
) -> Dict[str, Any]:
    adapted = copy.deepcopy(config)
    lengths = [
        int(length)
        for length in calibration_prior.get("sequence_length_distribution", {})
    ]
    if lengths:
        stream = adapted.setdefault("stream", {})
        current_min = int(stream.get("min_events", 1))
        stream["min_events"] = min(current_min, min(lengths))
    return adapted


def _config_for_sample(
    config: Dict[str, Any],
    calibration_prior: Dict[str, Any],
    *,
    topology: str,
    mechanism_type: str,
    semantic_mode: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if semantic_mode != "llm":
        return config, disabled_semantic_compatibility_metadata()
    combo = {
        "domain": str(calibration_prior.get("domain", "Calibrated")),
        "topology": topology,
        "mechanism_type": mechanism_type,
        "primary_regime": "",
    }
    return apply_semantic_compatible_sampling(config, combo)


def _stage(structural, temporal, semantic, label) -> str:
    if not structural.approved:
        return "structural_validation"
    if not temporal.approved:
        return "temporal_validation"
    if semantic is not None and not semantic.approved:
        return "semantic_validation"
    if label is not None and not label.approved:
        return "label_validation"
    return "unknown_validation"


if __name__ == "__main__":
    raise SystemExit(main())
