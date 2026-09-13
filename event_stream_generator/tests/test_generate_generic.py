import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from event_stream_generator.scripts.generate_generic import _stratified_combinations


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


if __name__ == "__main__":
    unittest.main()
