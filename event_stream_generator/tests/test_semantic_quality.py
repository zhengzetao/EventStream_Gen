from __future__ import annotations

import unittest

from event_stream_generator.judge.semantic_quality import score_semantic_quality


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


if __name__ == "__main__":
    unittest.main()
