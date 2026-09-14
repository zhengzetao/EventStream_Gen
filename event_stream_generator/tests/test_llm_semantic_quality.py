from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from event_stream_generator.evaluation.llm_semantic_quality import evaluate_llm_semantics


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class LlmSemanticQualityTests(unittest.TestCase):
    def test_evaluate_llm_semantics_summarizes_pass_rate_and_retries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            accepted = Path(tmp) / "accepted.jsonl"
            rejected = Path(tmp) / "rejected.jsonl"
            _write_jsonl(
                accepted,
                [
                    {
                        "stream_id": "s1",
                        "domain": "Finance",
                        "metadata": {"semantic_method": "llm", "llm_attempt_count": 1},
                        "validation": {"semantic": {"approved": True, "errors": []}},
                        "semantic_scenario": {
                            "scenario_description": "Scenario one.",
                            "event_mapping": {"E0": "A", "E1": "B"},
                        },
                    },
                    {
                        "stream_id": "s2",
                        "domain": "Healthcare",
                        "metadata": {"semantic_method": "llm", "llm_attempt_count": 3},
                        "validation": {"semantic": {"approved": True, "errors": []}},
                        "semantic_scenario": {
                            "scenario_description": "Scenario two.",
                            "event_mapping": {"E0": "C", "E1": "D"},
                        },
                    },
                ],
            )
            _write_jsonl(
                rejected,
                [
                    {
                        "stream_id": "s3",
                        "stage": "semantic_validation",
                        "errors": ["relation explanation mismatch"],
                    }
                ],
            )

            report = evaluate_llm_semantics(
                accepted_jsonl=accepted,
                rejected_jsonl=rejected,
                sample_preview_count=1,
            )

            self.assertEqual(report["accepted_count"], 2)
            self.assertEqual(report["rejected_count"], 1)
            self.assertEqual(report["semantic_pass_rate"], 1.0)
            self.assertEqual(report["average_llm_attempt_count"], 2.0)
            self.assertEqual(report["domain_counts"], {"Finance": 1, "Healthcare": 1})
            self.assertEqual(
                report["rejection_stage_counts"],
                {"semantic_validation": 1},
            )
            self.assertEqual(
                report["rejection_error_counts"],
                {"relation explanation mismatch": 1},
            )
            self.assertIn("semantic_quality", report)
            self.assertIn("average_score", report["semantic_quality"])
            self.assertIn("issue_counts", report["semantic_quality"])
            self.assertNotIn("semantic_judge", report)
            self.assertEqual(len(report["sample_previews"]), 1)

    def test_evaluate_llm_semantics_can_run_optional_judge(self) -> None:
        class JudgeClient:
            def complete(self, prompt: str) -> str:
                return json.dumps(
                    {
                        "approved": True,
                        "score": 0.85,
                        "issues": [],
                        "suggestions": [],
                    }
                )

        with tempfile.TemporaryDirectory() as tmp:
            accepted = Path(tmp) / "accepted.jsonl"
            _write_jsonl(
                accepted,
                [
                    {
                        "stream_id": "s1",
                        "domain": "Transportation",
                        "mechanism": {"nodes": [{"node_id": "E0"}], "edges": []},
                        "metadata": {"semantic_method": "llm", "llm_attempt_count": 1},
                        "validation": {"semantic": {"approved": True, "errors": []}},
                        "semantic_scenario": {
                            "scenario_description": "A stalled bus causes dispatch review.",
                            "event_mapping": {"E0": "bus service disruption"},
                            "relation_explanations": [],
                        },
                    }
                ],
            )

            report = evaluate_llm_semantics(
                accepted_jsonl=accepted,
                sample_preview_count=1,
                judge_client=JudgeClient(),
            )

            self.assertEqual(report["semantic_judge"]["sample_count"], 1)
            self.assertEqual(report["semantic_judge"]["approved_count"], 1)
            self.assertEqual(report["semantic_judge"]["average_score"], 0.85)
            self.assertEqual(report["semantic_judge"]["pass_rate"], 1.0)

    def test_evaluate_llm_semantics_records_judge_failures(self) -> None:
        class BrokenJudgeClient:
            def complete(self, prompt: str) -> str:
                return "not json"

        with tempfile.TemporaryDirectory() as tmp:
            accepted = Path(tmp) / "accepted.jsonl"
            _write_jsonl(
                accepted,
                [
                    {
                        "stream_id": "s1",
                        "domain": "Finance",
                        "mechanism": {"nodes": [{"node_id": "E0"}], "edges": []},
                        "metadata": {"semantic_method": "llm", "llm_attempt_count": 1},
                        "validation": {"semantic": {"approved": True, "errors": []}},
                        "semantic_scenario": {
                            "scenario_description": "A market shock triggers review.",
                            "event_mapping": {"E0": "market shock"},
                            "relation_explanations": [],
                        },
                    }
                ],
            )

            report = evaluate_llm_semantics(
                accepted_jsonl=accepted,
                judge_client=BrokenJudgeClient(),
            )

            self.assertEqual(report["semantic_judge"]["sample_count"], 1)
            self.assertEqual(report["semantic_judge"]["approved_count"], 0)
            self.assertEqual(report["semantic_judge"]["min_score"], 0.0)
            self.assertTrue(
                any(
                    issue.startswith("judge failed:")
                    for issue in report["semantic_judge"]["issue_counts"]
                )
            )

    def test_llm_semantic_quality_cli_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            accepted = Path(tmp) / "accepted.jsonl"
            rejected = Path(tmp) / "rejected.jsonl"
            output = Path(tmp) / "report.json"
            _write_jsonl(
                accepted,
                [
                    {
                        "stream_id": "s1",
                        "domain": "IT/System",
                        "metadata": {"semantic_method": "llm", "llm_attempt_count": 1},
                        "validation": {"semantic": {"approved": True, "errors": []}},
                        "semantic_scenario": {
                            "scenario_description": "A system incident.",
                            "event_mapping": {"E0": "A"},
                        },
                    }
                ],
            )
            _write_jsonl(rejected, [])

            subprocess.run(
                [
                    sys.executable,
                    "run_data_generation.py",
                    "--mode",
                    "evaluate-llm-semantics",
                    "--accepted-jsonl",
                    str(accepted),
                    "--rejected-jsonl",
                    str(rejected),
                    "--output-json",
                    str(output),
                    "--judge-mode",
                    "none",
                ],
                check=True,
            )

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["accepted_count"], 1)
            self.assertEqual(report["semantic_pass_rate"], 1.0)
            self.assertIn("semantic_quality", report)


if __name__ == "__main__":
    unittest.main()
