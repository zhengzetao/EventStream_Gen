from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from event_stream_generator import cli


class CliDispatchTests(unittest.TestCase):
    def test_generic_mode_dispatches_to_generic_script(self) -> None:
        called = []

        def fake_runner(module_name, forwarded_args):
            called.append((module_name, forwarded_args))
            return 0

        result = cli.main(
            [
                "--mode",
                "generic",
                "--num-samples",
                "3",
                "--output",
                "data/generic.jsonl",
                "--seed",
                "7",
            ],
            runner=fake_runner,
        )

        self.assertEqual(result, 0)
        self.assertEqual(
            called,
            [
                (
                    "event_stream_generator.scripts.generate_generic",
                    [
                        "--num-samples",
                        "3",
                        "--output",
                        "data/generic.jsonl",
                        "--seed",
                        "7",
                    ],
                )
            ],
        )

    def test_calibrated_mode_dispatches_to_calibrated_script(self) -> None:
        called = []

        result = cli.main(
            [
                "--mode",
                "calibrated",
                "--calibration-prior",
                "data/prior.json",
                "--num-samples",
                "3",
                "--output",
                "data/calibrated.jsonl",
            ],
            runner=lambda module_name, forwarded_args: called.append(
                (module_name, forwarded_args)
            )
            or 0,
        )

        self.assertEqual(result, 0)
        self.assertEqual(called[0][0], "event_stream_generator.scripts.generate_calibrated")
        self.assertIn("--calibration-prior", called[0][1])

    def test_calibrate_mode_dispatches_to_generic_calibrator(self) -> None:
        called = []

        result = cli.main(
            [
                "--mode",
                "calibrate",
                "--input",
                "data/events.csv",
                "--output",
                "data/prior.json",
                "--timestamp-col",
                "ts",
                "--stream-id-col",
                "sid",
                "--event-type-col",
                "etype",
                "--domain",
                "Healthcare",
            ],
            runner=lambda module_name, forwarded_args: called.append(
                (module_name, forwarded_args)
            )
            or 0,
        )

        self.assertEqual(result, 0)
        self.assertEqual(called[0][0], "event_stream_generator.scripts.calibrate_events")
        self.assertIn("--domain", called[0][1])

    def test_evaluate_mode_dispatches_to_evaluator(self) -> None:
        called = []

        result = cli.main(
            [
                "--mode",
                "evaluate",
                "--real-csv",
                "data/real.csv",
                "--generic-jsonl",
                "data/generic.jsonl",
                "--calibrated-jsonl",
                "data/calibrated.jsonl",
                "--output-json",
                "data/report.json",
                "--output-csv",
                "data/metrics.csv",
                "--timestamp-col",
                "timestamp",
                "--stream-id-col",
                "block_id",
                "--event-type-col",
                "template_id",
            ],
            runner=lambda module_name, forwarded_args: called.append(
                (module_name, forwarded_args)
            )
            or 0,
        )

        self.assertEqual(result, 0)
        self.assertEqual(called[0][0], "event_stream_generator.scripts.evaluate_generation")
        self.assertIn("--output-csv", called[0][1])

    def test_default_runner_restores_sys_argv(self) -> None:
        original_argv = list(sys.argv)

        class FakeModule:
            @staticmethod
            def main() -> int:
                self.assertEqual(sys.argv, ["fake.module", "--x", "1"])
                return 9

        with patch.dict(sys.modules, {"fake.module": FakeModule}):
            result = cli.run_script_main("fake.module", ["--x", "1"])

        self.assertEqual(result, 9)
        self.assertEqual(sys.argv, original_argv)

    def test_config_file_expands_to_forwarded_arguments(self) -> None:
        called = []

        config_path = (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "config"
            / "examples"
            / "transportation_accident.yaml"
        )
        result = cli.main(
            ["--config", str(config_path), "--output", "data/override.jsonl"],
            runner=lambda module_name, forwarded_args: called.append(
                (module_name, forwarded_args)
            )
            or 0,
        )

        self.assertEqual(result, 0)
        self.assertEqual(called[0][0], "event_stream_generator.scripts.generate_generic")
        self.assertIn("--domains", called[0][1])
        self.assertIn("Transportation", called[0][1])
        self.assertIn("--output", called[0][1])
        self.assertIn("data/override.jsonl", called[0][1])


if __name__ == "__main__":
    unittest.main()
