from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from event_stream_generator.dataset_builder import build_dataset
from event_stream_generator.tests.test_calibrated_generation import CALIBRATION_PRIOR


class DatasetBuilderTests(unittest.TestCase):
    def test_build_generic_dataset_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run_generic"
            manifest = build_dataset(
                {
                    "generation_mode": "generic",
                    "output_dir": str(output_dir),
                    "num_samples": 6,
                    "seed": 13,
                    "semantic_mode": "rule",
                    "qa_mode": "template",
                    "domains": ["Transportation"],
                    "topologies": ["chain", "tree"],
                }
            )

            accepted = output_dir / "accepted" / "samples.jsonl"
            rejected = output_dir / "rejected" / "rejected.jsonl"
            coverage = output_dir / "reports" / "coverage.json"
            diversity = output_dir / "reports" / "diversity.json"
            manifest_path = output_dir / "reports" / "manifest.json"
            resolved_config = output_dir / "configs" / "resolved_config.yaml"

            self.assertTrue(accepted.exists())
            self.assertTrue(rejected.exists())
            self.assertTrue(coverage.exists())
            self.assertTrue(diversity.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(resolved_config.exists())
            rows = [
                json.loads(line)
                for line in accepted.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["domain"] for row in rows}, {"Transportation"})
            self.assertEqual(manifest["num_requested"], 6)
            self.assertEqual(manifest["num_accepted"], 6)
            self.assertEqual(manifest["num_rejected"], 0)
            self.assertEqual(manifest["generation_mode"], "generic")
            self.assertIn("diversity", manifest["outputs"])
            self.assertEqual(manifest["generator_version"], "event_stream_generator_v1")
            self.assertIn("python_version", manifest)
            self.assertEqual(manifest["entry_command"][0], "build_dataset")
            self.assertEqual(manifest["input_hashes"], {})
            self.assertEqual(
                manifest["output_hashes"]["accepted"],
                _sha256(accepted),
            )
            self.assertEqual(
                manifest["output_hashes"]["diversity"],
                _sha256(diversity),
            )

    def test_build_generic_dataset_balances_small_multidomain_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run_balanced"
            build_dataset(
                {
                    "generation_mode": "generic",
                    "output_dir": str(output_dir),
                    "num_samples": 16,
                    "seed": 23,
                    "semantic_mode": "none",
                    "qa_mode": "none",
                    "domains": ["IT/System", "Finance", "Healthcare", "Transportation"],
                    "topologies": ["chain", "tree"],
                    "mechanism_types": ["direct_trigger", "delayed_trigger"],
                    "temporal_regimes": ["exponential", "gamma"],
                }
            )

            rows = [
                json.loads(line)
                for line in (output_dir / "accepted" / "samples.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            diversity = json.loads(
                (output_dir / "reports" / "diversity.json").read_text(encoding="utf-8")
            )

            self.assertEqual(
                {row["domain"] for row in rows},
                {"IT/System", "Finance", "Healthcare", "Transportation"},
            )
            self.assertEqual(set(diversity["domain_distribution"]), {
                "IT/System",
                "Finance",
                "Healthcare",
                "Transportation",
            })
            self.assertTrue(diversity["approved"])
            self.assertIn("delta_t_quantiles", diversity["metrics"])
            self.assertIn("qa_type_distribution", diversity["metrics"])

    def test_build_dataset_cli_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run_cli"
            config = Path(tmp) / "dataset.yaml"
            config.write_text(
                "\n".join(
                    [
                        "mode: build-dataset",
                        "generation_mode: generic",
                        f"output_dir: {output_dir}",
                        "num_samples: 4",
                        "seed: 17",
                        "semantic_mode: none",
                        "qa_mode: none",
                        "domains:",
                        "  - IT/System",
                        "topologies:",
                        "  - chain",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    "run_data_generation.py",
                    "--config",
                    str(config),
                ],
                check=True,
            )

            manifest = json.loads(
                (output_dir / "reports" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["generation_mode"], "generic")
            self.assertEqual(manifest["num_accepted"], 4)

    def test_build_calibrated_dataset_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run_calibrated"
            prior_path = Path(tmp) / "prior.json"
            prior_path.write_text(json.dumps(CALIBRATION_PRIOR), encoding="utf-8")

            manifest = build_dataset(
                {
                    "generation_mode": "calibrated",
                    "output_dir": str(output_dir),
                    "calibration_prior": str(prior_path),
                    "num_samples": 5,
                    "seed": 19,
                    "semantic_mode": "rule",
                    "qa_mode": "template",
                }
            )

            accepted = output_dir / "accepted" / "samples.jsonl"
            rejected = output_dir / "rejected" / "rejected.jsonl"
            coverage = output_dir / "reports" / "coverage.json"
            diversity = output_dir / "reports" / "diversity.json"
            self.assertTrue(accepted.exists())
            self.assertTrue(rejected.exists())
            self.assertTrue(coverage.exists())
            self.assertTrue(diversity.exists())
            rows = [
                json.loads(line)
                for line in accepted.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 5)
            self.assertEqual({row["domain"] for row in rows}, {"HDFS"})
            self.assertEqual(manifest["generation_mode"], "calibrated")
            self.assertEqual(manifest["num_accepted"], 5)
            self.assertEqual(
                manifest["input_hashes"]["calibration_prior"],
                _sha256(prior_path),
            )
            self.assertEqual(
                manifest["output_hashes"]["accepted"],
                _sha256(accepted),
            )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
