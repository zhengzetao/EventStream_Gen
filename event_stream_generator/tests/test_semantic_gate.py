from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from event_stream_generator.dataset_builder import build_dataset
from event_stream_generator.semantic.gate import apply_semantic_quality_gate


def _sample(stream_id: str, mapping: dict[str, str]) -> dict:
    return {
        "stream_id": stream_id,
        "domain": "IT/System",
        "topology": "chain",
        "mechanism_type": "direct_trigger",
        "primary_regime": "gamma",
        "seed": 1,
        "mechanism": {
            "nodes": [{"node_id": "E0"}, {"node_id": "E1"}],
            "edges": [
                {"source": "E0", "target": "E1", "relation_type": "direct_trigger"}
            ],
        },
        "events": [],
        "ground_truth": {},
        "validation": {"semantic": {"approved": True, "errors": [], "metrics": {}}},
        "metadata": {"semantic_method": "llm"},
        "semantic_scenario": {
            "scenario_description": "A concrete system incident creates a downstream operational alert.",
            "event_mapping": mapping,
            "relation_explanations": [
                {
                    "source": "E0",
                    "target": "E1",
                    "relation_type": "direct_trigger",
                    "explanation": "The upstream system condition directly creates the downstream alert state.",
                }
            ],
        },
    }


class SemanticGateTests(unittest.TestCase):
    def test_gate_moves_low_quality_samples_to_rejected(self) -> None:
        accepted = [
            _sample(
                "good",
                {
                    "E0": "database connection pool saturation",
                    "E1": "api timeout alert emitted",
                },
            ),
            _sample("bad", {"E0": "event", "E1": "issue"}),
        ]
        rejected = []

        kept, new_rejected, report = apply_semantic_quality_gate(
            accepted,
            rejected,
            {"enabled": True, "min_rule_score": 0.7, "reject_on_rule_issues": True},
        )

        self.assertEqual([row["stream_id"] for row in kept], ["good"])
        self.assertEqual(len(new_rejected), 1)
        self.assertEqual(new_rejected[0]["stage"], "semantic_quality_gate")
        self.assertEqual(new_rejected[0]["sample"]["stream_id"], "bad")
        self.assertIn("semantic quality score below threshold", new_rejected[0]["errors"])
        self.assertEqual(report["enabled"], True)
        self.assertEqual(report["accepted_after_gate"], 1)
        self.assertEqual(report["rejected_by_gate"], 1)

    def test_gate_is_disabled_by_default(self) -> None:
        accepted = [_sample("bad", {"E0": "event", "E1": "issue"})]
        kept, new_rejected, report = apply_semantic_quality_gate(accepted, [], {})

        self.assertEqual(kept, accepted)
        self.assertEqual(new_rejected, [])
        self.assertFalse(report["enabled"])

    def test_dataset_builder_writes_semantic_quality_gate_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run_gate"
            manifest = build_dataset(
                {
                    "generation_mode": "generic",
                    "output_dir": str(output_dir),
                    "num_samples": 4,
                    "seed": 31,
                    "semantic_mode": "rule",
                    "qa_mode": "template",
                    "domains": ["Transportation"],
                    "topologies": ["chain"],
                    "semantic_quality": {
                        "enabled": True,
                        "min_rule_score": 0.7,
                        "reject_on_rule_issues": True,
                    },
                }
            )

            gate_report = output_dir / "reports" / "semantic_quality_gate.json"
            coverage = output_dir / "reports" / "coverage.json"
            self.assertTrue(gate_report.exists())
            report = json.loads(gate_report.read_text(encoding="utf-8"))
            coverage_report = json.loads(coverage.read_text(encoding="utf-8"))
            self.assertTrue(report["enabled"])
            self.assertEqual(report["accepted_before_gate"], 4)
            self.assertEqual(report["accepted_after_gate"], manifest["num_accepted"])
            self.assertEqual(sum(coverage_report["domain_counts"].values()), manifest["num_accepted"])
            self.assertIn("semantic_quality_gate", manifest["outputs"])


if __name__ == "__main__":
    unittest.main()
