import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from event_stream_generator.core.models import EventStream, GeneratedEvent
from event_stream_generator.qa.causal_qa import generate_causal_qa
from event_stream_generator.qa.temporal_qa import generate_temporal_qa
from event_stream_generator.qa.propagation_qa import generate_propagation_qa
from event_stream_generator.validators.label import validate_qa_ground_truth


def _stream() -> EventStream:
    return EventStream(
        stream_id="stream_qa",
        domain="IT/System",
        topology="chain",
        mechanism_type="multi_hop_propagation",
        primary_regime="gamma",
        seed=5,
        mechanism={"nodes": [], "edges": []},
        events=[
            GeneratedEvent(0, 0.0, "E0", "server CPU saturation", None, 0, None, None, None, {}, False, 0, "E0"),
            GeneratedEvent(3, 0.2, "BG0", "routine health-check log", None, None, None, None, "exponential", {"lambda": 0.2}, True, 0),
            GeneratedEvent(1, 1.5, "E1", "request queue buildup", 0, 0, "direct_trigger", 1.5, "gamma", {"shape": 2.0, "scale": 0.5}, False, 1, "E1"),
            GeneratedEvent(2, 3.0, "E2", "API latency spike", 1, 0, "delayed_trigger", 1.5, "gamma", {"shape": 2.0, "scale": 0.5}, False, 2, "E2"),
        ],
        ground_truth={
            "parents": {"1": 0, "2": 1},
            "roots": {"0": 0, "1": 0, "2": 0},
            "paths": {"0": [0], "1": [0, 1], "2": [0, 1, 2]},
            "root_event_ids": [0],
            "background_event_ids": [3],
        },
        validation={},
        metadata={"semantic_status": "instantiated", "time_unit": "abstract_float"},
    )


class QAGenerationTests(unittest.TestCase):
    def test_causal_qa_answers_come_from_ground_truth(self):
        qas = generate_causal_qa(_stream())

        by_type = {qa["qa_type"]: qa for qa in qas}
        self.assertEqual(by_type["direct_parent"]["answer"], 1)
        self.assertEqual(by_type["root_cause"]["answer"], 0)
        self.assertTrue(all(validate_qa_ground_truth(_stream(), qa).approved for qa in qas))

    def test_temporal_qa_answers_come_from_timestamps(self):
        qas = generate_temporal_qa(_stream())

        by_type = {qa["qa_type"]: qa for qa in qas}
        self.assertEqual(by_type["temporal_order"]["answer"], [0, 3, 1, 2])
        self.assertEqual(by_type["time_delay"]["answer"], 1.5)
        self.assertTrue(all(validate_qa_ground_truth(_stream(), qa).approved for qa in qas))

    def test_propagation_qa_answers_path_from_ground_truth(self):
        qas = generate_propagation_qa(_stream())

        self.assertEqual(qas[0]["qa_type"], "causal_path")
        self.assertEqual(qas[0]["answer"], [0, 1, 2])
        self.assertTrue(validate_qa_ground_truth(_stream(), qas[0]).approved)

    def test_label_validator_rejects_tampered_answer(self):
        qa = generate_causal_qa(_stream())[0]
        qa["answer"] = 99

        result = validate_qa_ground_truth(_stream(), qa)

        self.assertFalse(result.approved)
        self.assertTrue(any("answer mismatch" in error for error in result.errors))

    def test_cli_writes_qa_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "generic_with_qa.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    "event_stream_generator/scripts/generate_generic.py",
                    "--num-samples",
                    "6",
                    "--output",
                    str(output),
                    "--seed",
                    "7",
                    "--semantic-mode",
                    "rule",
                    "--qa-mode",
                    "template",
                ],
                check=True,
            )
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 6)
            self.assertTrue(rows[0]["qa_pairs"])
            self.assertTrue(rows[0]["validation"]["label"]["approved"])

    def test_cli_default_qa_mode_keeps_no_qa_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "generic_no_qa.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    "event_stream_generator/scripts/generate_generic.py",
                    "--num-samples",
                    "3",
                    "--output",
                    str(output),
                    "--seed",
                    "7",
                ],
                check=True,
            )
            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertNotIn("qa_pairs", row)


if __name__ == "__main__":
    unittest.main()
