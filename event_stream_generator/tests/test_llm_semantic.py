from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from event_stream_generator.semantic.llm_client import OpenAIResponsesClient, extract_response_text
from event_stream_generator.semantic import (
    instantiate_scenario_llm,
    instantiate_scenario_llm_judge_guided,
)
from event_stream_generator.tests.test_semantic import _stream
from event_stream_generator.validators.semantic import validate_semantic


def _tree_recovery_stream():
    stream = _stream()
    stream.topology = "tree"
    stream.mechanism_type = "recovery"
    stream.domain = "Transportation"
    stream.mechanism = {
        "nodes": [
            {"node_id": "E0", "role": "root", "abstract_type": "E0"},
            {"node_id": "E1", "role": "target", "abstract_type": "E1"},
            {"node_id": "E2", "role": "target", "abstract_type": "E2"},
        ],
        "edges": [
            {"source": "E0", "target": "E1", "relation_type": "recovery"},
            {"source": "E0", "target": "E2", "relation_type": "recovery"},
        ],
    }
    return stream


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("fake client has no reply left")
        return self.replies.pop(0)


class FakeJudgeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("fake judge has no reply left")
        return self.replies.pop(0)


class LlmSemanticTests(unittest.TestCase):
    def test_extract_response_text_reads_responses_api_output(self) -> None:
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "hello"},
                    ],
                }
            ]
        }

        self.assertEqual(extract_response_text(response), "hello")

    def test_openai_client_reads_environment_without_exposing_key(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "secret-key",
                "OPENAI_BASE_URL": "https://example.test/api/v1",
                "OPENAI_MODEL": "model-x",
                "OPENAI_TIMEOUT_SECONDS": "90",
                "OPENAI_MAX_OUTPUT_TOKENS": "1600",
            },
        ):
            client = OpenAIResponsesClient.from_env()

        self.assertEqual(client.base_url, "https://example.test/api/v1")
        self.assertEqual(client.model, "model-x")
        self.assertEqual(client.timeout_seconds, 90)
        self.assertEqual(client.max_output_tokens, 1600)
        self.assertNotIn("secret-key", repr(client))

    def test_llm_instantiation_maps_events_and_preserves_ground_truth(self) -> None:
        stream = _stream()
        before_ground_truth = json.loads(json.dumps(stream.ground_truth))
        reply = json.dumps(
            {
                "scenario_description": "A service overload cascade in an IT system.",
                "event_mapping": {
                    "E0": "database connection pool exhaustion",
                    "E1": "API request queue expansion",
                    "E2": "customer-facing latency alert",
                },
                "relation_explanations": [
                    {
                        "source": "E0",
                        "target": "E1",
                        "relation_type": "direct_trigger",
                        "explanation": "The database bottleneck directly triggers API queue growth.",
                    },
                    {
                        "source": "E1",
                        "target": "E2",
                        "relation_type": "delayed_trigger",
                        "explanation": "The sustained queue later causes a latency alert.",
                    },
                ],
            }
        )

        updated = instantiate_scenario_llm(stream, FakeClient([reply]), max_retry=1)

        self.assertEqual(updated.ground_truth, before_ground_truth)
        self.assertEqual(updated.metadata["semantic_status"], "instantiated")
        self.assertEqual(updated.metadata["semantic_method"], "llm")
        self.assertTrue(validate_semantic(updated).approved)
        event_by_id = {event.event_id: event for event in updated.events}
        self.assertEqual(event_by_id[0].semantic_type, "database connection pool exhaustion")
        self.assertEqual(event_by_id[1].semantic_type, "API request queue expansion")

    def test_llm_instantiation_retries_invalid_json(self) -> None:
        valid = json.dumps(
            {
                "scenario_description": "A valid retry scenario.",
                "event_mapping": {"E0": "A", "E1": "B", "E2": "C"},
                "relation_explanations": [
                    {"source": "E0", "target": "E1", "relation_type": "direct_trigger", "explanation": "A causes B."},
                    {"source": "E1", "target": "E2", "relation_type": "delayed_trigger", "explanation": "B later causes C."},
                ],
            }
        )
        client = FakeClient(["not json", valid])

        updated = instantiate_scenario_llm(_stream(), client, max_retry=2)

        self.assertTrue(validate_semantic(updated).approved)
        self.assertEqual(len(client.prompts), 2)

    def test_judge_guided_instantiation_retries_with_judge_feedback(self) -> None:
        weak = json.dumps(
            {
                "scenario_description": "A weak semantic scenario.",
                "event_mapping": {
                    "E0": "database connection pool exhaustion",
                    "E1": "API request queue expansion",
                    "E2": "customer-facing latency alert",
                },
                "relation_explanations": [
                    {
                        "source": "E0",
                        "target": "E1",
                        "relation_type": "direct_trigger",
                        "explanation": "The database bottleneck causes queue growth.",
                    },
                    {
                        "source": "E1",
                        "target": "E2",
                        "relation_type": "delayed_trigger",
                        "explanation": "The queue later causes an alert.",
                    },
                ],
            }
        )
        strong = json.dumps(
            {
                "scenario_description": "A checkout service overload produces a queue-driven latency incident.",
                "event_mapping": {
                    "E0": "checkout database pool exhausts",
                    "E1": "checkout API workers queue requests",
                    "E2": "checkout latency alert fires",
                },
                "relation_explanations": [
                    {
                        "source": "E0",
                        "target": "E1",
                        "relation_type": "direct_trigger",
                        "explanation": "The exhausted checkout database pool prevents API workers from clearing requests.",
                    },
                    {
                        "source": "E1",
                        "target": "E2",
                        "relation_type": "delayed_trigger",
                        "explanation": "The same checkout queue persists until the latency threshold is crossed.",
                    },
                ],
            }
        )
        semantic_client = FakeClient([weak, strong])
        judge_client = FakeJudgeClient(
            [
                json.dumps(
                    {
                        "approved": False,
                        "score": 0.62,
                        "issues": ["causal direction is too generic"],
                        "suggestions": ["Tie E1 and E2 to the same checkout service queue."],
                    }
                ),
                json.dumps(
                    {
                        "approved": True,
                        "score": 0.91,
                        "issues": [],
                        "suggestions": [],
                    }
                ),
            ]
        )

        updated = instantiate_scenario_llm_judge_guided(
            _stream(),
            semantic_client=semantic_client,
            judge_client=judge_client,
            max_retry=2,
            approval_threshold=0.8,
        )

        self.assertTrue(validate_semantic(updated).approved)
        self.assertEqual(updated.metadata["llm_attempt_count"], 2)
        self.assertEqual(updated.metadata["semantic_judge"]["score"], 0.91)
        self.assertEqual(len(semantic_client.prompts), 2)
        self.assertEqual(len(judge_client.prompts), 2)
        self.assertIn("causal direction is too generic", semantic_client.prompts[1])
        self.assertIn("Tie E1 and E2 to the same checkout service queue", semantic_client.prompts[1])

    def test_judge_guided_instantiation_rejects_after_max_retry(self) -> None:
        reply = json.dumps(
            {
                "scenario_description": "A weak semantic scenario.",
                "event_mapping": {"E0": "A", "E1": "B", "E2": "C"},
                "relation_explanations": [
                    {"source": "E0", "target": "E1", "relation_type": "direct_trigger", "explanation": "A causes B."},
                    {"source": "E1", "target": "E2", "relation_type": "delayed_trigger", "explanation": "B later causes C."},
                ],
            }
        )
        judge_client = FakeJudgeClient(
            [
                json.dumps(
                    {
                        "approved": False,
                        "score": 0.5,
                        "issues": ["weak edge alignment"],
                        "suggestions": [],
                    }
                )
            ]
        )

        with self.assertRaises(RuntimeError) as context:
            instantiate_scenario_llm_judge_guided(
                _stream(),
                semantic_client=FakeClient([reply]),
                judge_client=judge_client,
                max_retry=1,
                approval_threshold=0.8,
            )

        self.assertIn("LLM judge rejected semantic scenario", str(context.exception))

    def test_llm_prompt_constrains_output_size_and_relation_count(self) -> None:
        valid = json.dumps(
            {
                "scenario_description": "A compact valid retry scenario.",
                "event_mapping": {"E0": "A", "E1": "B", "E2": "C"},
                "relation_explanations": [
                    {"source": "E0", "target": "E1", "relation_type": "direct_trigger", "explanation": "A causes B."},
                    {"source": "E1", "target": "E2", "relation_type": "delayed_trigger", "explanation": "B later causes C."},
                ],
            }
        )
        client = FakeClient([valid])

        instantiate_scenario_llm(_stream(), client, max_retry=1)

        self.assertIn("exactly 2 relation_explanations", client.prompts[0])
        self.assertIn("compact valid JSON", client.prompts[0])
        self.assertIn("Each explanation must be one short sentence", client.prompts[0])

    def test_llm_prompt_hardens_tree_branch_and_recovery_semantics(self) -> None:
        valid = json.dumps(
            {
                "scenario_description": "A compact tree recovery scenario.",
                "event_mapping": {
                    "E0": "transit control detects signal failure",
                    "E1": "dispatcher activates train holding plan",
                    "E2": "maintenance crew resets signal relay",
                },
                "relation_explanations": [
                    {
                        "source": "E0",
                        "target": "E1",
                        "relation_type": "recovery",
                        "explanation": "Detection gives dispatch enough evidence to hold trains safely.",
                    },
                    {
                        "source": "E0",
                        "target": "E2",
                        "relation_type": "recovery",
                        "explanation": "Detection directs maintenance crews to reset the affected relay.",
                    },
                ],
            }
        )
        client = FakeClient([valid])

        instantiate_scenario_llm(_tree_recovery_stream(), client, max_retry=1)

        prompt = client.prompts[0]
        self.assertIn("Tree guidance:", prompt)
        self.assertIn("Sibling child events must be distinct downstream consequences", prompt)
        self.assertIn("Do not map siblings to overlapping actions", prompt)
        self.assertIn("Recovery guidance:", prompt)
        self.assertIn("Do not say the asset repairs itself", prompt)
        self.assertIn("Each relation explanation must name the source event phrase and target event phrase", prompt)
        self.assertIn("The source event must be sufficient to make the target event plausible without hidden intermediate events", prompt)
        self.assertIn("Never make a recovery child introduce a new fault, blockage, outage, or risk escalation", prompt)
        self.assertIn("Use concrete domain-specific actors, systems, artifacts, or measurements", prompt)
        self.assertIn("Every child event must reuse or explicitly reference a concrete object", prompt)
        self.assertIn("For tree branches, explain why this exact parent can produce each child without borrowing evidence from a sibling", prompt)
        self.assertIn("For recovery, order meanings as diagnosis, containment or mitigation, repair, then verification when those roles exist", prompt)


    def test_llm_relation_metadata_is_canonicalized_from_mechanism(self) -> None:
        reply = json.dumps(
            {
                "scenario_description": "A semantic scenario with noisy relation metadata.",
                "event_mapping": {"E0": "A", "E1": "B", "E2": "C"},
                "relation_explanations": [
                    {
                        "source": "wrong",
                        "target": "wrong",
                        "relation_type": "wrong",
                        "explanation": "A directly causes B.",
                    },
                    {
                        "source": "wrong",
                        "target": "wrong",
                        "relation_type": "wrong",
                        "explanation": "B later causes C.",
                    },
                ],
            }
        )

        updated = instantiate_scenario_llm(_stream(), FakeClient([reply]), max_retry=1)

        self.assertTrue(validate_semantic(updated).approved)
        self.assertEqual(
            [
                (
                    item["source"],
                    item["target"],
                    item["relation_type"],
                    item["explanation"],
                )
                for item in updated.semantic_scenario["relation_explanations"]
            ],
            [
                ("E0", "E1", "direct_trigger", "A directly causes B."),
                ("E1", "E2", "delayed_trigger", "B later causes C."),
            ],
        )


if __name__ == "__main__":
    unittest.main()
