from __future__ import annotations

import unittest
from pathlib import Path
import re

import yaml

from event_stream_generator.judge.semantic_quality import score_semantic_quality


ROOT = Path(__file__).resolve().parents[2]


class SemanticQualityTests(unittest.TestCase):
    def test_scores_specific_non_repetitive_semantics_high(self) -> None:
        row = {
            "domain": "Cybersecurity",
            "semantic_scenario": {
                "scenario_description": "A phishing email compromises an employee account and triggers containment.",
                "event_mapping": {
                    "E0": "phishing email compromises employee mailbox",
                    "E1": "attacker escalates privileges on internal dashboard",
                    "E2": "security team activates account containment policy",
                },
                "relation_explanations": [
                    {"explanation": "The compromised mailbox gives the attacker credentials for the dashboard."},
                    {"explanation": "The privilege escalation triggers containment by the security team."},
                ],
            },
        }

        result = score_semantic_quality(row)

        self.assertTrue(result["approved"])
        self.assertGreaterEqual(result["score"], 0.8)
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["metrics"]["duplicate_event_phrase_count"], 0)

    def test_flags_short_generic_duplicate_semantics(self) -> None:
        row = {
            "domain": "Finance",
            "semantic_scenario": {
                "scenario_description": "Scenario.",
                "event_mapping": {
                    "E0": "event",
                    "E1": "event",
                    "E2": "update",
                },
                "relation_explanations": [
                    {"explanation": "A causes B."},
                    {"explanation": "related"},
                ],
            },
        }

        result = score_semantic_quality(row)

        self.assertFalse(result["approved"])
        self.assertLess(result["score"], 0.7)
        self.assertIn("event mapping contains short phrases", result["issues"])
        self.assertIn("event mapping contains duplicate phrases", result["issues"])
        self.assertIn("relation explanations are too short or generic", result["issues"])

    def test_configured_rule_based_templates_are_specific_enough_for_quality_gate(self) -> None:
        templates = yaml.safe_load(
            (ROOT / "event_stream_generator/config/semantic_templates.yaml").read_text(
                encoding="utf-8"
            )
        )
        failures = []
        for domain, domain_templates in templates["domains"].items():
            phrases = [
                *domain_templates.get("event_phrases", []),
                *domain_templates.get("background_phrases", []),
            ]
            for index, phrase in enumerate(phrases):
                row = {
                    "domain": domain,
                    "semantic_scenario": {
                        "scenario_description": (
                            f"A concrete {domain} scenario links a source condition "
                            "to a downstream operational response."
                        ),
                        "event_mapping": {"E0": phrase},
                        "relation_explanations": [],
                    },
                }
                result = score_semantic_quality(row)
                if result["issues"]:
                    failures.append((domain, index, phrase, result["issues"]))
                word_count = len(
                    re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", phrase)
                )
                if not 4 <= word_count <= 8:
                    failures.append((domain, index, phrase, [f"word_count={word_count}"]))

        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
