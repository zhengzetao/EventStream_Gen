from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class Phase3DConvergenceTests(unittest.TestCase):
    def test_root_run_data_generation_is_the_only_cli_file_entry(self) -> None:
        self.assertTrue((ROOT / "run_data_generation.py").exists())
        self.assertFalse((ROOT / "data_generation/run_data_generation.py").exists())

    def test_legacy_generation_stack_has_been_removed(self) -> None:
        removed_paths = [
            "domain_packages",
            "generation_core",
            "eventflow_v1",
            "tests/test_generation_core.py",
            "run_eventflow_v1.py",
            "run_eventflow_calibration.py",
            "run_eventflow_judge_challenge.py",
            "run_eventflow_stability.py",
            "run_parameter_calibration.py",
            "output_eventflow_v140_llm_100",
            "output_transportation_new",
        ]
        existing = [path for path in removed_paths if (ROOT / path).exists()]

        self.assertEqual(existing, [])

    def test_unified_entries_do_not_import_legacy_stack(self) -> None:
        forbidden = ("domain_packages", "generation_core", "eventflow_v1")
        for relative_path in [
            "run_data_generation.py",
            "event_stream_generator/cli/__init__.py",
            "event_stream_generator/cli/main.py",
        ]:
            text = (ROOT / relative_path).read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(f"import {token}", text)
                self.assertNotIn(f"from {token}", text)


if __name__ == "__main__":
    unittest.main()
