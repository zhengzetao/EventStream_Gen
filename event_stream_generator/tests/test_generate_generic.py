import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from event_stream_generator.dataset_builder import _generation_args
from event_stream_generator.core.models import EventStream
from event_stream_generator.scripts import generate_generic
from event_stream_generator.scripts.generate_generic import _stratified_combinations
from event_stream_generator.semantic.llm_cache import SemanticCache, make_semantic_cache_key


class GenerateGenericCliTests(unittest.TestCase):
    def test_stratified_combinations_interleave_topologies_for_small_samples(self):
        taxonomy = {
            "domains": ["IT/System"],
            "topologies": ["chain", "tree"],
            "mechanism_types": ["direct_trigger", "delayed_trigger"],
            "temporal_regimes": ["exponential", "gamma"],
        }

        first_four = _stratified_combinations(taxonomy)[:4]

        self.assertEqual(
            [item["topology"] for item in first_four],
            ["chain", "tree", "chain", "tree"],
        )

    def test_cli_writes_accepted_rejected_and_coverage_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "generic.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    "event_stream_generator/scripts/generate_generic.py",
                    "--num-samples",
                    "12",
                    "--output",
                    str(output),
                    "--seed",
                    "0",
                ],
                check=True,
            )

            rejected = output.with_name("generic_rejected.jsonl")
            coverage = output.with_name("generic.coverage.json")
            self.assertTrue(output.exists())
            self.assertTrue(rejected.exists())
            self.assertTrue(coverage.exists())
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 12)
            self.assertIn("domain_counts", json.loads(coverage.read_text(encoding="utf-8")))

    def test_cli_writes_rejected_file_when_generation_cannot_accept_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "generic_failed.jsonl"
            result = subprocess.run(
                [
                    sys.executable,
                    "event_stream_generator/scripts/generate_generic.py",
                    "--num-samples",
                    "1",
                    "--output",
                    str(output),
                    "--seed",
                    "0",
                    "--domains",
                    "IT/System",
                    "--topologies",
                    "chain",
                    "--mechanism-types",
                    "direct_trigger",
                    "--temporal-regimes",
                    "gamma",
                    "--min-chain-length",
                    "3",
                    "--max-chain-length",
                    "3",
                    "--min-events",
                    "999",
                ],
                check=False,
            )

            rejected = output.with_name("generic_failed_rejected.jsonl")
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(output.exists())
            self.assertTrue(rejected.exists())
            accepted_rows = [
                line for line in output.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
            rejected_rows = [
                json.loads(line)
                for line in rejected.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(accepted_rows, [])
            self.assertTrue(rejected_rows)
            self.assertEqual(rejected_rows[0]["stage"], "temporal_validation")

    def test_dataset_builder_forwards_llm_efficiency_options(self):
        args = _generation_args(
            {
                "num_samples": 3,
                "seed": 0,
                "semantic_mode": "llm",
                "qa_mode": "template",
                "llm_concurrency": 4,
                "llm_cache": "data/cache/semantic.json",
                "progress_path": "data/runs/progress.json",
                "semantic_judge_mode": "llm",
                "semantic_judge_threshold": 0.8,
                "semantic_judge_max_retry": 2,
            },
            "generic",
            Path("data/out.jsonl"),
        )

        self.assertIn("--llm-concurrency", args)
        self.assertIn("4", args)
        self.assertIn("--llm-cache", args)
        self.assertIn("data/cache/semantic.json", args)
        self.assertIn("--progress-path", args)
        self.assertIn("data/runs/progress.json", args)
        self.assertIn("--semantic-judge-mode", args)
        self.assertIn("llm", args)
        self.assertIn("--semantic-judge-threshold", args)
        self.assertIn("0.8", args)
        self.assertIn("--semantic-judge-max-retry", args)
        self.assertIn("2", args)

    def test_progress_writer_records_completed_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            progress = Path(tmp) / "progress.json"

            generate_generic._write_progress(
                progress,
                total=10,
                completed=3,
                accepted=2,
                rejected=1,
                status="running",
            )

            payload = json.loads(progress.read_text(encoding="utf-8"))
            self.assertEqual(payload["total"], 10)
            self.assertEqual(payload["completed"], 3)
            self.assertEqual(payload["accepted"], 2)
            self.assertEqual(payload["rejected"], 1)
            self.assertEqual(payload["status"], "running")

    def test_llm_task_runner_uses_concurrency(self):
        def worker(index: int):
            time.sleep(0.2)
            return index, {"stream_id": f"stream_{index:06d}"}, []

        started = time.perf_counter()
        accepted, rejected = generate_generic._run_llm_tasks_concurrently(
            indexes=[0, 1, 2, 3],
            worker=worker,
            concurrency=4,
            progress_path=None,
        )
        elapsed = time.perf_counter() - started

        self.assertEqual([row["stream_id"] for row in accepted], [
            "stream_000000",
            "stream_000001",
            "stream_000002",
            "stream_000003",
        ])
        self.assertEqual(rejected, [])
        self.assertLess(elapsed, 0.6)

    def test_semantic_cache_round_trips_by_stable_stream_key(self):
        stream = EventStream(
            stream_id="stream_cache",
            domain="IT/System",
            topology="chain",
            mechanism_type="direct_trigger",
            primary_regime="gamma",
            seed=11,
            mechanism={
                "nodes": [{"node_id": "E0"}, {"node_id": "E1"}],
                "edges": [{"source": "E0", "target": "E1", "relation_type": "direct_trigger"}],
            },
            events=[],
            ground_truth={},
        )
        key = make_semantic_cache_key(stream)
        scenario = {
            "scenario_description": "A concrete cache test scenario.",
            "event_mapping": {"E0": "database queue pressure rises", "E1": "gateway timeout alert fires"},
            "relation_explanations": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "semantic_cache.json"
            cache = SemanticCache(path)
            cache.set(key, scenario)
            cache.save()

            loaded = SemanticCache(path)
            self.assertEqual(loaded.get(key), scenario)

    def test_semantic_cache_key_changes_for_prompt_version(self):
        stream = EventStream(
            stream_id="stream_cache",
            domain="IT/System",
            topology="chain",
            mechanism_type="direct_trigger",
            primary_regime="gamma",
            seed=11,
            mechanism={
                "nodes": [{"node_id": "E0"}, {"node_id": "E1"}],
                "edges": [{"source": "E0", "target": "E1", "relation_type": "direct_trigger"}],
            },
            events=[],
            ground_truth={},
        )

        self.assertNotEqual(
            make_semantic_cache_key(stream, prompt_version="semantic-only-v1"),
            make_semantic_cache_key(stream, prompt_version="judge-guided-v1-threshold-0.8"),
        )


if __name__ == "__main__":
    unittest.main()
