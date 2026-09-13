import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from event_stream_generator.core.models import EventStream, GeneratedEvent
from event_stream_generator.semantic import instantiate_scenario
from event_stream_generator.validators.semantic import validate_semantic


TEMPLATES = {
    "domains": {
        "IT/System": {
            "event_phrases": [
                "server CPU saturation",
                "request queue buildup",
                "API latency spike",
            ],
            "background_phrases": [
                "unrelated monitoring warning",
                "routine health-check log",
            ],
        }
    }
}


def _stream() -> EventStream:
    return EventStream(
        stream_id="stream_semantic",
        domain="IT/System",
        topology="chain",
        mechanism_type="multi_hop_propagation",
        primary_regime="gamma",
        seed=4,
        mechanism={
            "nodes": [
                {"node_id": "E0", "role": "root", "abstract_type": "E0"},
                {"node_id": "E1", "role": "target", "abstract_type": "E1"},
                {"node_id": "E2", "role": "target", "abstract_type": "E2"},
            ],
            "edges": [
                {"source": "E0", "target": "E1", "relation_type": "direct_trigger"},
                {"source": "E1", "target": "E2", "relation_type": "delayed_trigger"},
            ],
        },
        events=[
            GeneratedEvent(0, 0.0, "E0", None, None, 0, None, None, None, {}, False, 0, "E0"),
            GeneratedEvent(3, 0.5, "BG0", None, None, None, None, None, "exponential", {"lambda": 0.2}, True, 0),
            GeneratedEvent(1, 1.0, "E1", None, 0, 0, "direct_trigger", 1.0, "gamma", {"shape": 2.0, "scale": 0.5}, False, 1, "E1"),
            GeneratedEvent(2, 3.0, "E2", None, 1, 0, "delayed_trigger", 2.0, "gamma", {"shape": 2.0, "scale": 0.5}, False, 2, "E2"),
        ],
        ground_truth={
            "parents": {"1": 0, "2": 1},
            "roots": {"0": 0, "1": 0, "2": 0},
            "paths": {"0": [0], "1": [0, 1], "2": [0, 1, 2]},
            "root_event_ids": [0],
            "background_event_ids": [3],
        },
        metadata={"semantic_status": "not_instantiated", "time_unit": "abstract_float"},
    )


class SemanticTests(unittest.TestCase):
    def test_rule_based_instantiation_fills_causal_and_background_semantics(self):
        stream = instantiate_scenario(_stream(), TEMPLATES, seed=10)

        event_by_id = {event.event_id: event for event in stream.events}
        self.assertEqual(event_by_id[0].semantic_type, "server CPU saturation")
        self.assertEqual(event_by_id[1].semantic_type, "request queue buildup")
        self.assertEqual(event_by_id[2].semantic_type, "API latency spike")
        self.assertEqual(event_by_id[3].semantic_type, "unrelated monitoring warning")
        self.assertEqual(stream.metadata["semantic_status"], "instantiated")
        self.assertEqual(
            set(stream.semantic_scenario["event_mapping"]),
            {"E0", "E1", "E2"},
        )
        self.assertEqual(len(stream.semantic_scenario["relation_explanations"]), 2)
        self.assertTrue(validate_semantic(stream).approved)

    def test_semantic_validator_rejects_missing_mapping(self):
        stream = instantiate_scenario(_stream(), TEMPLATES, seed=10)
        del stream.semantic_scenario["event_mapping"]["E1"]

        result = validate_semantic(stream)

        self.assertFalse(result.approved)
        self.assertTrue(any("missing semantic mapping" in error for error in result.errors))

    def test_semantic_validator_rejects_extra_mapping(self):
        stream = instantiate_scenario(_stream(), TEMPLATES, seed=10)
        stream.semantic_scenario["event_mapping"]["E99"] = "invented event"

        result = validate_semantic(stream)

        self.assertFalse(result.approved)
        self.assertTrue(any("extra semantic mapping" in error for error in result.errors))

    def test_semantic_validator_rejects_relation_mismatch(self):
        stream = instantiate_scenario(_stream(), TEMPLATES, seed=10)
        stream.semantic_scenario["relation_explanations"][0]["target"] = "E2"

        result = validate_semantic(stream)

        self.assertFalse(result.approved)
        self.assertTrue(any("relation explanation mismatch" in error for error in result.errors))

    def test_semantic_validator_rejects_forbidden_ground_truth_fields(self):
        stream = instantiate_scenario(_stream(), TEMPLATES, seed=10)
        stream.semantic_scenario["ground_truth"] = {"parents": {"1": 0}}

        result = validate_semantic(stream)

        self.assertFalse(result.approved)
        self.assertTrue(any("forbidden semantic field" in error for error in result.errors))

    def test_cli_semantic_mode_rule_writes_semantic_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "generic_semantic.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    "event_stream_generator/scripts/generate_generic.py",
                    "--num-samples",
                    "8",
                    "--output",
                    str(output),
                    "--seed",
                    "5",
                    "--semantic-mode",
                    "rule",
                ],
                check=True,
            )
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 8)
            self.assertIn("semantic_scenario", rows[0])
            self.assertTrue(rows[0]["validation"]["semantic"]["approved"])
            causal_events = [event for event in rows[0]["events"] if not event["is_background"]]
            self.assertTrue(all(event["semantic_type"] for event in causal_events))

    def test_cli_default_semantic_mode_keeps_phase1_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "generic_none.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    "event_stream_generator/scripts/generate_generic.py",
                    "--num-samples",
                    "4",
                    "--output",
                    str(output),
                    "--seed",
                    "5",
                ],
                check=True,
            )
            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertNotIn("semantic_scenario", row)
            self.assertEqual(row["metadata"]["semantic_status"], "not_instantiated")


if __name__ == "__main__":
    unittest.main()
