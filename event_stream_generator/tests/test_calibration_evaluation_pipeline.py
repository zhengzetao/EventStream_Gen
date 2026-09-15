import csv
import json
import tempfile
import unittest
from pathlib import Path

from event_stream_generator.calibration.evaluation_pipeline import (
    _generation_domain,
    run_calibration_evaluation_pipeline,
)
from event_stream_generator import cli


class CalibrationEvaluationPipelineTests(unittest.TestCase):
    def test_generation_domain_can_differ_from_calibration_domain(self):
        self.assertEqual(
            _generation_domain(
                {"domain": "HDFS"},
                {"synthetic_domain": "IT/System"},
            ),
            "IT/System",
        )

    def test_pipeline_calibrates_generates_compares_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "events.csv"
            with raw.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["timestamp", "stream_id", "event_type"],
                )
                writer.writeheader()
                for stream_id, delay in [("s1", 1.0), ("s2", 2.0), ("s3", 3.0)]:
                    writer.writerow(
                        {"timestamp": 0.0, "stream_id": stream_id, "event_type": "A"}
                    )
                    writer.writerow(
                        {"timestamp": delay, "stream_id": stream_id, "event_type": "B"}
                    )
            config = root / "eval.yaml"
            output_dir = root / "eval"
            config.write_text(
                "\n".join(
                    [
                        "mode: evaluate-calibration-dataset",
                        "dataset_name: standard-test",
                        "adapter: standard_csv",
                        "domain: IT/System",
                        f"input_path: {raw}",
                        f"prepared_output: {root / 'prepared.csv'}",
                        f"prior_output: {root / 'prior.json'}",
                        "min_transition_samples: 2",
                        "evaluation:",
                        f"  output_dir: {output_dir}",
                        "  num_samples: 3",
                        "  seed: 11",
                        "  topology: chain",
                        "  mechanism_type: multi_hop_propagation",
                    ]
                ),
                encoding="utf-8",
            )

            result = run_calibration_evaluation_pipeline(config)

            self.assertEqual(result["dataset_name"], "standard-test")
            self.assertTrue((output_dir / "generic.jsonl").exists())
            self.assertTrue((output_dir / "calibrated.jsonl").exists())
            self.assertTrue((output_dir / "comparison.json").exists())
            self.assertTrue((output_dir / "comparison.csv").exists())
            comparison = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
            self.assertIn("inter_event_time", comparison)
            self.assertIn("calibration_effect", result)
            self.assertIn("inter_event_time_ks", result["calibration_effect"])

    def test_pipeline_forwards_calibrated_generation_options_to_generated_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "events.csv"
            with raw.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["timestamp", "stream_id", "event_type"],
                )
                writer.writeheader()
                for stream_id in ["s1", "s2"]:
                    writer.writerow(
                        {"timestamp": 0.0, "stream_id": stream_id, "event_type": "A"}
                    )
                    writer.writerow(
                        {"timestamp": 1.0, "stream_id": stream_id, "event_type": "B"}
                    )
            config = root / "eval.yaml"
            output_dir = root / "eval"
            config.write_text(
                "\n".join(
                    [
                        "mode: evaluate-calibration-dataset",
                        "dataset_name: option-test",
                        "adapter: standard_csv",
                        "domain: IT/System",
                        f"input_path: {raw}",
                        f"prepared_output: {root / 'prepared.csv'}",
                        f"prior_output: {root / 'prior.json'}",
                        "min_transition_samples: 2",
                        "evaluation:",
                        f"  output_dir: {output_dir}",
                        "  num_samples: 2",
                        "  seed: 17",
                        "  calibrated_generation:",
                        "    use_second_order: false",
                        "    smoothing_alpha: 0.0",
                    ]
                ),
                encoding="utf-8",
            )

            result = run_calibration_evaluation_pipeline(config)

            self.assertEqual(
                result["generation"]["calibrated_generation"]["use_second_order"],
                False,
            )
            self.assertEqual(
                result["generation"]["calibrated_generation"]["smoothing_alpha"],
                0.0,
            )

    def test_cli_dispatches_evaluate_calibration_dataset_config(self):
        called = []

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "eval.yaml"
            config_path.write_text(
                "mode: evaluate-calibration-dataset\n",
                encoding="utf-8",
            )

            result = cli.main(
                ["--mode", "evaluate-calibration-dataset", "--config", str(config_path)],
                runner=lambda module_name, forwarded_args: called.append(
                    (module_name, forwarded_args)
                )
                or 0,
            )

        self.assertEqual(result, 0)
        self.assertEqual(
            called[0][0],
            "event_stream_generator.scripts.evaluate_calibration_dataset",
        )
        self.assertEqual(called[0][1], ["--config", str(config_path)])


if __name__ == "__main__":
    unittest.main()
