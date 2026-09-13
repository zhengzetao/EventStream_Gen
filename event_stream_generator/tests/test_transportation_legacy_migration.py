from __future__ import annotations

import unittest
from pathlib import Path

from event_stream_generator.core.io import load_yaml


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


class TransportationLegacyMigrationTests(unittest.TestCase):
    def test_transportation_semantic_templates_include_legacy_event_vocabulary(self) -> None:
        templates = load_yaml(CONFIG_DIR / "semantic_templates.yaml")
        transportation = templates["domains"]["Transportation"]
        phrases = set(transportation["event_phrases"])

        expected_phrases = {
            "vehicle collision",
            "congestion onset",
            "normal traffic flow restoration",
            "heavy rain onset",
            "heavy rain clearance",
            "planned road closure",
            "planned road reopening",
        }
        self.assertTrue(expected_phrases <= phrases)

    def test_transportation_example_config_preserves_legacy_scenario_weights(self) -> None:
        config = load_yaml(CONFIG_DIR / "examples" / "transportation_accident.yaml")

        self.assertEqual(config["domain"], "Transportation")
        self.assertEqual(config["mode"], "generic")
        self.assertEqual(config["semantic_mode"], "rule")
        self.assertEqual(config["qa_mode"], "template")
        self.assertEqual(
            config["legacy_source"]["path"],
            "domain_packages/transportation/config/default.json",
        )
        self.assertEqual(
            config["legacy_source"]["scenario_weights"],
            {
                "accident_propagation": 0.35,
                "weather_disruption": 0.25,
                "planned_closure": 0.2,
                "compound_weather_accident": 0.2,
            },
        )
        self.assertEqual(set(config["topologies"]), {"chain", "tree"})
        self.assertIn("Transportation", config["domains"])

    def test_legacy_audit_marks_transportation_package_as_migrate_not_delete(self) -> None:
        audit = load_yaml(CONFIG_DIR / "legacy_module_audit.yaml")
        by_path = {item["path"]: item for item in audit["modules"]}

        package = by_path["domain_packages/transportation/package.py"]
        self.assertEqual(package["decision"], "migrate_domain_knowledge")
        self.assertIn("event vocabulary", package["reusable_assets"])
        self.assertIn("scenario families", package["reusable_assets"])

        root_entry = by_path["run_data_generation.py"]
        self.assertEqual(root_entry["decision"], "replaced_by_unified_entry")


if __name__ == "__main__":
    unittest.main()
