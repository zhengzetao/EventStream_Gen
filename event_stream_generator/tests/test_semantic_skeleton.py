from __future__ import annotations

import json
import unittest

from event_stream_generator.semantic import instantiate_scenario_llm
from event_stream_generator.skeleton.semantic_skeleton import build_semantic_skeleton
from event_stream_generator.tests.test_llm_semantic import FakeClient, _tree_recovery_stream
from event_stream_generator.tests.test_semantic import _stream


class SemanticSkeletonTests(unittest.TestCase):
    def test_skeleton_preserves_ground_truth_boundary(self) -> None:
        skeleton = build_semantic_skeleton(_stream())
        encoded = json.dumps(skeleton, sort_keys=True)

        self.assertEqual(skeleton["domain"], "IT/System")
        self.assertEqual(skeleton["topology"], "chain")
        self.assertIn("event_slots", skeleton)
        self.assertIn("edge_constraints", skeleton)
        self.assertNotIn("ground_truth", encoded)
        self.assertNotIn("parents", encoded)
        self.assertNotIn("root_id", encoded)
        self.assertNotIn("timestamp", encoded)

    def test_tree_sibling_slots_are_distinct(self) -> None:
        skeleton = build_semantic_skeleton(_tree_recovery_stream())
        slots = skeleton["event_slots"]
        edge_constraints = skeleton["edge_constraints"]

        self.assertEqual(slots["E1"]["sibling_group"], "children_of_E0")
        self.assertEqual(slots["E2"]["sibling_group"], "children_of_E0")
        self.assertNotEqual(
            slots["E1"]["preferred_effect_type"],
            slots["E2"]["preferred_effect_type"],
        )
        self.assertTrue(
            all(
                "new fault" in constraint["disallowed_effect_types"]
                for constraint in edge_constraints
            )
        )

    def test_llm_prompt_includes_semantic_skeleton(self) -> None:
        reply = json.dumps(
            {
                "scenario_description": "A skeleton-constrained IT incident.",
                "event_mapping": {
                    "E0": "database pool reaches saturation",
                    "E1": "api queue expands rapidly",
                    "E2": "latency alert is emitted",
                },
                "relation_explanations": [
                    {
                        "source": "E0",
                        "target": "E1",
                        "relation_type": "direct_trigger",
                        "explanation": "The saturated database pool prevents API workers from clearing queued requests.",
                    },
                    {
                        "source": "E1",
                        "target": "E2",
                        "relation_type": "delayed_trigger",
                        "explanation": "The growing API queue sustains latency long enough to emit an alert.",
                    },
                ],
            }
        )
        client = FakeClient([reply])

        updated = instantiate_scenario_llm(_stream(), client, max_retry=1)

        self.assertEqual(updated.metadata["semantic_skeleton_version"], "v2")
        self.assertIn("Semantic Skeleton JSON:", client.prompts[0])
        self.assertIn("preferred_effect_type", client.prompts[0])
        self.assertIn("required_causal_bridge", client.prompts[0])

    def test_skeleton_includes_edge_grounding_requirements(self) -> None:
        skeleton = build_semantic_skeleton(_stream())

        first_edge = skeleton["edge_constraints"][0]

        self.assertIn("grounding_requirements", first_edge)
        self.assertIn(
            "Reuse at least one concrete domain object from the source event in the target event or explanation.",
            first_edge["grounding_requirements"],
        )
        self.assertIn(
            "Name the observable signal, resource, artifact, or workflow handoff that carries the effect.",
            first_edge["grounding_requirements"],
        )
        self.assertIn("domain_causal_channel_hints", first_edge)
        self.assertIn("same service dependency", first_edge["domain_causal_channel_hints"])

    def test_recovery_skeleton_assigns_ordered_recovery_stages(self) -> None:
        skeleton = build_semantic_skeleton(_tree_recovery_stream())
        stages = {
            node_id: slot["recovery_stage"]
            for node_id, slot in skeleton["event_slots"].items()
        }

        self.assertEqual(stages["E0"], "fault detection or diagnosis")
        self.assertEqual(stages["E1"], "containment or routing mitigation")
        self.assertEqual(stages["E2"], "repair action or restoration verification")

    def test_semantic_validation_rejects_recovery_child_as_new_failure(self) -> None:
        reply = json.dumps(
            {
                "scenario_description": "A problematic recovery scenario.",
                "event_mapping": {
                    "E0": "transit control detects signal failure",
                    "E1": "new switch outage blocks trains",
                    "E2": "maintenance crew resets signal relay",
                },
                "relation_explanations": [
                    {
                        "source": "E0",
                        "target": "E1",
                        "relation_type": "recovery",
                        "explanation": "The detected signal failure causes a new switch outage.",
                    },
                    {
                        "source": "E0",
                        "target": "E2",
                        "relation_type": "recovery",
                        "explanation": "The detected signal failure sends maintenance crews to reset the relay.",
                    },
                ],
            }
        )

        with self.assertRaises(RuntimeError) as context:
            instantiate_scenario_llm(_tree_recovery_stream(), FakeClient([reply]), max_retry=1)

        self.assertIn("recovery target introduces disallowed effect", str(context.exception))


if __name__ == "__main__":
    unittest.main()
