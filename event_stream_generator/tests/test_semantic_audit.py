from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from event_stream_generator.evaluation.semantic_audit import audit_semantic_dataset


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(
    stream_id: str,
    domain: str,
    mechanism_type: str,
    topology: str,
    regime: str,
    event_mapping: dict[str, str],
    relation_explanation: str,
) -> dict:
    return {
        "stream_id": stream_id,
        "domain": domain,
        "mechanism_type": mechanism_type,
        "topology": topology,
        "metadata": {"semantic_method": "llm", "llm_attempt_count": 1},
        "mechanism": {
            "nodes": [{"node_id": "E0"}, {"node_id": "E1"}],
            "edges": [
                {
                    "source": "E0",
                    "target": "E1",
                    "relation_type": "direct_trigger",
                    "regime": regime,
                }
            ],
        },
        "validation": {"semantic": {"approved": True, "errors": []}},
        "semantic_scenario": {
            "scenario_description": "A concrete domain event produces a downstream operational response.",
            "event_mapping": event_mapping,
            "relation_explanations": [
                {
                    "source": "E0",
                    "target": "E1",
                    "relation_type": "direct_trigger",
                    "explanation": relation_explanation,
                }
            ],
        },
    }


class SemanticAuditTests(unittest.TestCase):
    def test_audit_semantic_dataset_reports_group_quality_and_low_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            accepted = Path(tmp) / "accepted.jsonl"
            rejected = Path(tmp) / "rejected.jsonl"
            _write_jsonl(
                accepted,
                [
                    _row(
                        "s1",
                        "Finance",
                        "direct_trigger",
                        "chain",
                        "gamma",
                        {"E0": "market liquidity shock", "E1": "risk desk exposure review"},
                        "The liquidity shock increases exposure uncertainty, so the risk desk starts a targeted review.",
                    ),
                    _row(
                        "s2",
                        "Healthcare",
                        "delayed_trigger",
                        "tree",
                        "lognormal",
                        {"E0": "event", "E1": "event"},
                        "causes",
                    ),
                ],
            )
            _write_jsonl(
                rejected,
                [
                    {
                        "stream_id": "s3",
                        "stage": "semantic_validation",
                        "errors": ["missing event mapping"],
                    }
                ],
            )

            report = audit_semantic_dataset(
                accepted_jsonl=accepted,
                rejected_jsonl=rejected,
                low_quality_limit=2,
            )

            self.assertEqual(report["accepted_count"], 2)
            self.assertEqual(report["rejected_count"], 1)
            self.assertEqual(report["overall_quality"]["sample_count"], 2)
            self.assertEqual(report["groups"]["domain"]["Finance"]["sample_count"], 1)
            self.assertEqual(report["groups"]["domain"]["Healthcare"]["sample_count"], 1)
            self.assertEqual(report["groups"]["topology"]["chain"]["sample_count"], 1)
            self.assertEqual(report["groups"]["mechanism_type"]["delayed_trigger"]["sample_count"], 1)
            self.assertEqual(report["groups"]["temporal_regime"]["gamma"]["sample_count"], 1)
            self.assertEqual(report["rejection_stage_counts"], {"semantic_validation": 1})
            self.assertEqual(report["low_quality_samples"][0]["stream_id"], "s2")
            self.assertLess(report["low_quality_samples"][0]["score"], 0.7)
            self.assertIn("event mapping contains generic phrases", report["issue_counts"])

    def test_audit_semantic_dataset_can_include_optional_judge_summary(self) -> None:
        class JudgeClient:
            def complete(self, prompt: str) -> str:
                return json.dumps(
                    {
                        "approved": True,
                        "score": 0.88,
                        "issues": [],
                        "suggestions": [],
                    }
                )

        with tempfile.TemporaryDirectory() as tmp:
            accepted = Path(tmp) / "accepted.jsonl"
            _write_jsonl(
                accepted,
                [
                    _row(
                        "s1",
                        "Transportation",
                        "direct_trigger",
                        "chain",
                        "weibull",
                        {"E0": "signal controller outage", "E1": "intersection queue spillback"},
                        "The controller outage disrupts timing and creates a queue that spills back into upstream lanes.",
                    )
                ],
            )

            report = audit_semantic_dataset(
                accepted_jsonl=accepted,
                judge_client=JudgeClient(),
                judge_sample_limit=1,
            )

            self.assertEqual(report["semantic_judge"]["sample_count"], 1)
            self.assertEqual(report["semantic_judge"]["approved_count"], 1)
            self.assertEqual(report["semantic_judge"]["average_score"], 0.88)

    def test_semantic_audit_cli_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            accepted = Path(tmp) / "accepted.jsonl"
            output = Path(tmp) / "audit.json"
            _write_jsonl(
                accepted,
                [
                    _row(
                        "s1",
                        "IT/System",
                        "direct_trigger",
                        "chain",
                        "exponential",
                        {"E0": "database connection pool saturation", "E1": "api request timeout burst"},
                        "The saturated connection pool prevents requests from acquiring database sessions, leading to timeout bursts.",
                    )
                ],
            )

            subprocess.run(
                [
                    sys.executable,
                    "data_generation/run_data_generation.py",
                    "--mode",
                    "audit-semantics",
                    "--accepted-jsonl",
                    str(accepted),
                    "--output-json",
                    str(output),
                    "--low-quality-limit",
                    "1",
                    "--judge-mode",
                    "none",
                ],
                check=True,
            )

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["accepted_count"], 1)
            self.assertIn("groups", report)
            self.assertIn("overall_quality", report)


if __name__ == "__main__":
    unittest.main()
