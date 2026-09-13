from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class LegacyLayoutTests(unittest.TestCase):
    def test_legacy_assets_are_not_part_of_active_repository(self) -> None:
        self.assertFalse((ROOT / "legacy").exists())

    def test_legacy_assets_do_not_shadow_main_packages(self) -> None:
        self.assertFalse((ROOT / "stpp_v27_release").exists())
        self.assertFalse((ROOT / "prompts").exists())
        self.assertFalse((ROOT / "run_pipeline.py").exists())
        self.assertFalse((ROOT / "llm_client.py").exists())
        self.assertFalse((ROOT / "demo_sts_sde.py").exists())
        self.assertFalse((ROOT / "generate_alignment_QA.py").exists())
        self.assertFalse((ROOT / "generate_cot.py").exists())
        self.assertFalse((ROOT / "generate_file_list.py").exists())
        self.assertFalse((ROOT / "generate_reasoning_QA.py").exists())
        self.assertFalse((ROOT / "generate_reasoning_forecasting_QA.py").exists())
        self.assertFalse((ROOT / "event_stream_generator" / "semantic.py").exists())


if __name__ == "__main__":
    unittest.main()
