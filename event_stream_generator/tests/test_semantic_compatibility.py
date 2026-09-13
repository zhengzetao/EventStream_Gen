from __future__ import annotations

import copy
import unittest

from event_stream_generator.semantic.compatibility import (
    SEMANTIC_COMPATIBILITY_VERSION,
    apply_semantic_compatible_sampling,
)
from event_stream_generator.scripts.generate_calibrated import _config_for_sample
from event_stream_generator.scripts.generate_generic import _config_for_combo


BASE_CONFIG = {
    "mechanism": {
        "chain": {"min_length": 4, "max_length": 12},
        "tree": {"min_nodes": 6, "max_nodes": 30, "max_depth": 6, "max_children_per_node": 3},
    },
    "stream": {"min_events": 5, "max_events": 80},
    "background": {"density": "medium"},
}


class SemanticCompatibilityTests(unittest.TestCase):
    def test_recovery_tree_sampling_is_constrained_without_mutating_base_config(self) -> None:
        original = copy.deepcopy(BASE_CONFIG)
        combo = {
            "domain": "Transportation",
            "topology": "tree",
            "mechanism_type": "recovery",
            "primary_regime": "gamma",
        }

        config, metadata = apply_semantic_compatible_sampling(BASE_CONFIG, combo)

        self.assertEqual(BASE_CONFIG, original)
        self.assertEqual(metadata["version"], SEMANTIC_COMPATIBILITY_VERSION)
        self.assertTrue(metadata["enabled"])
        self.assertLessEqual(config["mechanism"]["tree"]["max_nodes"], 8)
        self.assertLessEqual(config["mechanism"]["tree"]["max_depth"], 3)
        self.assertLessEqual(config["mechanism"]["tree"]["max_children_per_node"], 2)
        self.assertIn("limited recovery tree size", metadata["rules_applied"])

    def test_process_like_tree_sampling_is_kept_shallow(self) -> None:
        combo = {
            "domain": "Scientific Process",
            "topology": "tree",
            "mechanism_type": "multi_hop_propagation",
            "primary_regime": "lognormal",
        }

        config, metadata = apply_semantic_compatible_sampling(BASE_CONFIG, combo)

        self.assertLessEqual(config["mechanism"]["tree"]["max_nodes"], 12)
        self.assertLessEqual(config["mechanism"]["tree"]["max_depth"], 3)
        self.assertIn("limited process-like tree depth", metadata["rules_applied"])

    def test_general_llm_tree_sampling_limits_semantic_branch_load(self) -> None:
        combo = {
            "domain": "Transportation",
            "topology": "tree",
            "mechanism_type": "multi_hop_propagation",
            "primary_regime": "gamma",
        }

        config, metadata = apply_semantic_compatible_sampling(BASE_CONFIG, combo)

        self.assertLessEqual(config["mechanism"]["tree"]["max_nodes"], 10)
        self.assertLessEqual(config["mechanism"]["tree"]["max_depth"], 3)
        self.assertIn("limited tree semantic branch load", metadata["rules_applied"])

    def test_generate_generic_only_applies_compatibility_for_llm_semantics(self) -> None:
        combo = {
            "domain": "Finance",
            "topology": "tree",
            "mechanism_type": "recovery",
            "primary_regime": "gamma",
        }

        plain_config, plain_metadata = _config_for_combo(BASE_CONFIG, combo, semantic_mode="none")
        llm_config, llm_metadata = _config_for_combo(BASE_CONFIG, combo, semantic_mode="llm")

        self.assertEqual(plain_config["mechanism"]["tree"]["max_nodes"], 30)
        self.assertFalse(plain_metadata["enabled"])
        self.assertLessEqual(llm_config["mechanism"]["tree"]["max_nodes"], 8)
        self.assertTrue(llm_metadata["enabled"])

    def test_generate_calibrated_uses_same_compatibility_for_llm_semantics(self) -> None:
        calibration_prior = {"domain": "IT/System", "calibration_type": "generic_event"}

        config, metadata = _config_for_sample(
            BASE_CONFIG,
            calibration_prior,
            topology="tree",
            mechanism_type="recovery",
            semantic_mode="llm",
        )

        self.assertTrue(metadata["enabled"])
        self.assertLessEqual(config["mechanism"]["tree"]["max_nodes"], 8)
        self.assertIn("limited recovery tree size", metadata["rules_applied"])


if __name__ == "__main__":
    unittest.main()
