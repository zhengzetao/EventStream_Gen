#!/usr/bin/env python3
"""Generate generic synthetic event-stream samples."""

from __future__ import annotations

import argparse
import copy
import itertools
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
    apply_semantic_scenario,
    instantiate_scenario,
    instantiate_scenario_llm,
)
from event_stream_generator.semantic.llm_cache import (
    SemanticCache,
    make_semantic_cache_key,
)
from event_stream_generator.semantic.llm_client import OpenAIResponsesClient
from event_stream_generator.simulation.simulator import generate_event_stream
from event_stream_generator.validators.label import validate_qa_pairs
from event_stream_generator.validators.semantic import validate_semantic
from event_stream_generator.validators.structural import validate_structure
from event_stream_generator.validators.temporal import validate_temporal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_config_dir = Path(__file__).resolve().parents[1] / "config"
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--taxonomy", default=str(default_config_dir / "taxonomy.yaml"))
    parser.add_argument("--prior", default=str(default_config_dir / "generic_prior.yaml"))
    parser.add_argument(
        "--semantic-mode",
        choices=["none", "rule", "llm"],
        default="none",
        help="Semantic instantiation mode. 'none' keeps abstract events only.",
    )
    parser.add_argument(
        "--semantic-templates",
        default=str(default_config_dir / "semantic_templates.yaml"),
    )
    parser.add_argument(
        "--qa-mode",
        choices=["none", "template"],
        default="none",
        help="Generate template QA pairs from simulator ground truth.",
    )
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--min-events", type=int)
    parser.add_argument("--max-chain-length", type=int)
    parser.add_argument("--min-chain-length", type=int)
    parser.add_argument("--max-tree-nodes", type=int)
    parser.add_argument("--min-tree-nodes", type=int)
    parser.add_argument("--max-tree-depth", type=int)
    parser.add_argument("--max-children-per-node", type=int)
    parser.add_argument("--domains", nargs="+")
    parser.add_argument("--topologies", nargs="+")
    parser.add_argument("--mechanism-types", nargs="+")
    parser.add_argument("--temporal-regimes", nargs="+")
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=1,
        help="Number of concurrent LLM semantic instantiation requests.",
    )
    parser.add_argument(
        "--llm-cache",
        help="Optional JSON cache path for LLM semantic instantiation results.",
    )
    parser.add_argument(
        "--progress-path",
        help="Optional JSON progress file updated during generation.",
    )
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.llm_concurrency <= 0:
        raise ValueError("--llm-concurrency must be positive")
    taxonomy = load_yaml(args.taxonomy)
    _apply_taxonomy_filters(taxonomy, args)
    prior = load_yaml(args.prior)
    semantic_templates = (
        load_yaml(args.semantic_templates) if args.semantic_mode == "rule" else {}
    )
    llm_client = OpenAIResponsesClient.from_env() if args.semantic_mode == "llm" else None
    _apply_overrides(prior, args)

    combinations = _stratified_combinations(taxonomy)
    max_retry = int(prior.get("stream", {}).get("max_generation_retry", 5))
    output = Path(args.output)
    progress_path = Path(args.progress_path) if args.progress_path else None
    cache = SemanticCache(args.llm_cache) if args.semantic_mode == "llm" else None
    cache_lock = threading.Lock()

    worker = lambda index: _generate_one_sample(
        index=index,
        args=args,
        prior=copy.deepcopy(prior),
        combinations=combinations,
        semantic_templates=semantic_templates,
        llm_client=llm_client,
        max_retry=max_retry,
        cache=cache,
        cache_lock=cache_lock,
    )
    _write_progress(
        progress_path,
        total=args.num_samples,
        completed=0,
        accepted=0,
        rejected=0,
        status="running",
    )
    if args.semantic_mode == "llm" and args.llm_concurrency > 1:
        accepted, rejected = _run_llm_tasks_concurrently(
            indexes=list(range(args.num_samples)),
            worker=worker,
            concurrency=args.llm_concurrency,
            progress_path=progress_path,
        )
    else:
        accepted = []
        rejected = []
        for index in range(args.num_samples):
            _, accepted_sample, rejected_attempts = worker(index)
            rejected.extend(rejected_attempts)
            if accepted_sample is None:
                _write_outputs(output, accepted, rejected)
                _write_progress(
                    progress_path,
                    total=args.num_samples,
                    completed=index + 1,
                    accepted=len(accepted),
                    rejected=len(rejected),
                    status="failed",
                )
                print(
                    f"generation failed stream_index={index} accepted={len(accepted)} "
                    f"rejected={len(rejected)} output={output}"
                )
                return 1
            accepted.append(accepted_sample)
            _write_progress(
                progress_path,
                total=args.num_samples,
                completed=index + 1,
                accepted=len(accepted),
                rejected=len(rejected),
                status="running",
            )

    if cache is not None:
        cache.save()
    if len(accepted) != args.num_samples:
        _write_outputs(output, accepted, rejected)
        _write_progress(
            progress_path,
            total=args.num_samples,
            completed=args.num_samples,
            accepted=len(accepted),
            rejected=len(rejected),
            status="failed",
        )
        print(
            f"generation failed accepted={len(accepted)} rejected={len(rejected)} "
            f"output={output}"
        )
        return 1

    _write_outputs(output, accepted, rejected)
    _write_progress(
        progress_path,
        total=args.num_samples,
        completed=args.num_samples,
        accepted=len(accepted),
        rejected=len(rejected),
        status="completed",
    )
    print(
        f"generated accepted={len(accepted)} rejected={len(rejected)} output={output}"
    )
    return 0


def _generate_one_sample(
    *,
    index: int,
    args: argparse.Namespace,
    prior: Dict[str, Any],
    combinations: Sequence[Dict[str, str]],
    semantic_templates: Dict[str, Any],
    llm_client: Any,
    max_retry: int,
    cache: Optional[SemanticCache],
    cache_lock: threading.Lock,
) -> Tuple[int, Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    combo = combinations[index % len(combinations)]
    rejected: List[Dict[str, Any]] = []
    for attempt in range(max_retry):
        sample_seed = args.seed + index * 1009 + attempt * 104729
        sample_config, compatibility_metadata = _config_for_combo(
            prior,
            combo,
            semantic_mode=args.semantic_mode,
        )
        stream = generate_event_stream(
            stream_id=f"stream_{index:06d}",
            seed=sample_seed,
            config=sample_config,
            **combo,
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
                semantic = validate_semantic(stream)
            elif args.semantic_mode == "llm":
                try:
                    stream = _instantiate_llm_with_cache(stream, llm_client, cache, cache_lock)
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
            return index, stream.to_dict(), rejected
        stage = "label_validation"
        if not structural.approved:
            stage = "structural_validation"
        elif not temporal.approved:
            stage = "temporal_validation"
        elif semantic is not None and not semantic.approved:
            stage = "semantic_validation"
        rejected.append(
            {
                "stream_id": f"stream_{index:06d}_attempt_{attempt + 1}",
                "stage": stage,
                "approved": False,
                "errors": (
                    structural.errors
                    + temporal.errors
                    + (label.errors if label is not None else [])
                    + (semantic.errors if semantic is not None else [])
                ),
                "sample": stream.to_dict(),
            }
        )
    return index, None, rejected


def _instantiate_llm_with_cache(
    stream: Any,
    llm_client: Any,
    cache: Optional[SemanticCache],
    cache_lock: threading.Lock,
) -> Any:
    if cache is None:
        return instantiate_scenario_llm(stream, llm_client, max_retry=3)
    cache_key = make_semantic_cache_key(stream)
    with cache_lock:
        cached = cache.get(cache_key)
    if cached is not None:
        updated = apply_semantic_scenario(
            stream,
            cached,
            semantic_method="llm",
            llm_attempt_count=0,
        )
        updated.metadata["llm_cache_hit"] = True
        return updated
    updated = instantiate_scenario_llm(stream, llm_client, max_retry=3)
    with cache_lock:
        cache.set(cache_key, updated.semantic_scenario or {})
    updated.metadata["llm_cache_hit"] = False
    return updated


def _run_llm_tasks_concurrently(
    *,
    indexes: Sequence[int],
    worker: Callable[[int], Tuple[int, Optional[Dict[str, Any]], List[Dict[str, Any]]]],
    concurrency: int,
    progress_path: Optional[Path],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    results: Dict[int, Dict[str, Any]] = {}
    rejected: List[Dict[str, Any]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(worker, index): index for index in indexes}
        for future in as_completed(futures):
            index, accepted_sample, rejected_attempts = future.result()
            completed += 1
            rejected.extend(rejected_attempts)
            if accepted_sample is not None:
                results[index] = accepted_sample
            _write_progress(
                progress_path,
                total=len(indexes),
                completed=completed,
                accepted=len(results),
                rejected=len(rejected),
                status="running",
            )
    accepted = [results[index] for index in sorted(results)]
    return accepted, rejected


def _write_progress(
    path: Optional[Path],
    *,
    total: int,
    completed: int,
    accepted: int,
    rejected: int,
    status: str,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "total": total,
        "completed": completed,
        "accepted": accepted,
        "rejected": rejected,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(path, payload)


def _write_outputs(
    output: Path, accepted: List[Dict[str, Any]], rejected: List[Dict[str, Any]]
) -> None:
    rejected_path = output.with_name(f"{output.stem}_rejected{output.suffix}")
    coverage_path = output.with_name(f"{output.stem}.coverage.json")
    write_jsonl(output, accepted)
    write_jsonl(rejected_path, rejected)
    write_json(coverage_path, build_coverage_report(accepted))


def _apply_overrides(prior: Dict[str, Any], args: argparse.Namespace) -> None:
    stream = prior.setdefault("stream", {})
    chain = prior.setdefault("mechanism", {}).setdefault("chain", {})
    tree = prior.setdefault("mechanism", {}).setdefault("tree", {})
    for arg_name, target, key in (
        ("max_events", stream, "max_events"),
        ("min_events", stream, "min_events"),
        ("max_chain_length", chain, "max_length"),
        ("min_chain_length", chain, "min_length"),
        ("max_tree_nodes", tree, "max_nodes"),
        ("min_tree_nodes", tree, "min_nodes"),
        ("max_tree_depth", tree, "max_depth"),
        ("max_children_per_node", tree, "max_children_per_node"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            target[key] = value


def _config_for_combo(
    prior: Dict[str, Any],
    combo: Dict[str, str],
    *,
    semantic_mode: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if semantic_mode == "llm":
        return apply_semantic_compatible_sampling(prior, combo)
    return prior, disabled_semantic_compatibility_metadata()


def _stratified_combinations(taxonomy: Dict[str, Any]) -> List[Dict[str, str]]:
    domains = taxonomy.get("domains", [])
    topologies = taxonomy.get("topologies", [])
    mechanism_types = taxonomy.get("mechanism_types", [])
    regimes = taxonomy.get("temporal_regimes", [])
    combos = [
        {
            "domain": domain,
            "topology": topology,
            "mechanism_type": mechanism_type,
            "primary_regime": regime,
        }
        for mechanism_type, regime, topology, domain in itertools.product(
            mechanism_types, regimes, topologies, domains
        )
    ]
    if not combos:
        raise ValueError("taxonomy does not define any generation combinations")
    return _balanced_prefix_order(combos)


def _balanced_prefix_order(combos: List[Dict[str, str]]) -> List[Dict[str, str]]:
    remaining = list(combos)
    ordered: List[Dict[str, str]] = []
    counts = {
        "domain": {},
        "topology": {},
        "mechanism_type": {},
        "primary_regime": {},
    }
    while remaining:
        best_index = min(
            range(len(remaining)),
            key=lambda index: (
                _coverage_score(remaining[index], counts),
                _repeat_score(remaining[index], ordered),
                remaining[index]["domain"],
                remaining[index]["topology"],
                remaining[index]["mechanism_type"],
                remaining[index]["primary_regime"],
            ),
        )
        combo = remaining.pop(best_index)
        ordered.append(combo)
        for key, bucket in counts.items():
            value = combo[key]
            bucket[value] = bucket.get(value, 0) + 1
    return ordered


def _coverage_score(combo: Dict[str, str], counts: Dict[str, Dict[str, int]]) -> int:
    return sum(counts[key].get(combo[key], 0) for key in counts)


def _repeat_score(combo: Dict[str, str], ordered: List[Dict[str, str]]) -> int:
    if not ordered:
        return 0
    previous = ordered[-1]
    return sum(
        1
        for key in ("domain", "topology", "mechanism_type", "primary_regime")
        if previous[key] == combo[key]
    )


def _apply_taxonomy_filters(taxonomy: Dict[str, Any], args: argparse.Namespace) -> None:
    for arg_name, taxonomy_key in (
        ("domains", "domains"),
        ("topologies", "topologies"),
        ("mechanism_types", "mechanism_types"),
        ("temporal_regimes", "temporal_regimes"),
    ):
        selected = getattr(args, arg_name)
        if selected is None:
            continue
        available = set(taxonomy.get(taxonomy_key, []))
        unknown = set(selected) - available
        if unknown:
            raise ValueError(
                f"Unknown {taxonomy_key}: {sorted(unknown)}. "
                f"Available: {sorted(available)}"
            )
        taxonomy[taxonomy_key] = list(selected)


if __name__ == "__main__":
    raise SystemExit(main())
