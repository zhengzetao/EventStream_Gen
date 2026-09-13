from __future__ import annotations

import json
import unittest

from event_stream_generator.judge.semantic_judge import judge_semantic_scenario


class FakeJudgeClient:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class SemanticJudgeTests(unittest.TestCase):
    def test_judge_accepts_strict_json_and_preserves_ground_truth_boundary(self) -> None:
        row = {
            "stream_id": "s1",
            "domain": "Finance",
            "mechanism": {"nodes": [{"node_id": "E0"}, {"node_id": "E1"}], "edges": []},
            "semantic_scenario": {
                "scenario_description": "A market event triggers a risk review.",
                "event_mapping": {"E0": "market volatility spike", "E1": "risk desk review"},
                "relation_explanations": [],
            },
            "ground_truth": {"parents": {"1": 0}},
        }
        client = FakeJudgeClient(
            json.dumps(
                {
                    "approved": True,
                    "score": 0.92,
                    "issues": [],
                    "suggestions": ["Keep the causal wording concise."],
                }
            )
        )

        result = judge_semantic_scenario(row, client)

        self.assertTrue(result["approved"])
        self.assertEqual(result["score"], 0.92)
        self.assertEqual(result["issues"], [])
        self.assertEqual(len(client.prompts), 1)
        self.assertIn("Do not modify or infer ground-truth answers", client.prompts[0])
        self.assertNotIn("parents", client.prompts[0])

    def test_judge_rejects_invalid_or_low_score_result(self) -> None:
        row = {
            "stream_id": "s2",
            "domain": "Healthcare",
            "mechanism": {"nodes": [{"node_id": "E0"}], "edges": []},
            "semantic_scenario": {"scenario_description": "Vague.", "event_mapping": {"E0": "event"}},
        }
        client = FakeJudgeClient(
            json.dumps(
                {
                    "approved": True,
                    "score": 0.4,
                    "issues": ["too vague"],
                    "suggestions": [],
                }
            )
        )

        result = judge_semantic_scenario(row, client, approval_threshold=0.7)

        self.assertFalse(result["approved"])
        self.assertEqual(result["score"], 0.4)
        self.assertIn("score below threshold", result["issues"])


if __name__ == "__main__":
    unittest.main()
