from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from event_stream_generator.export.training_export import export_training_data, render_stream_input
from event_stream_generator.dataset_builder import build_dataset


def _row() -> dict:
    return {
        "stream_id": "stream_train",
        "domain": "IT/System",
        "topology": "chain",
        "mechanism_type": "multi_hop_propagation",
        "primary_regime": "gamma",
        "seed": 9,
        "events": [
            {
                "event_id": 0,
                "timestamp": 0.0,
                "abstract_type": "E0",
                "semantic_type": "database pool saturation",
                "parent_id": None,
                "root_id": 0,
                "relation_to_parent": None,
                "delta_t_from_parent": None,
                "regime": None,
                "is_background": False,
            },
            {
                "event_id": 2,
                "timestamp": 0.4,
                "abstract_type": "BG0",
                "semantic_type": "routine health check",
                "parent_id": None,
                "root_id": None,
                "relation_to_parent": None,
                "delta_t_from_parent": None,
                "regime": "exponential",
                "is_background": True,
            },
            {
                "event_id": 1,
                "timestamp": 1.2,
                "abstract_type": "E1",
                "semantic_type": "api timeout alert",
                "parent_id": 0,
                "root_id": 0,
                "relation_to_parent": "direct_trigger",
                "delta_t_from_parent": 1.2,
                "regime": "gamma",
                "is_background": False,
            },
        ],
        "ground_truth": {"parents": {"1": 0}},
        "metadata": {"semantic_method": "llm", "semantic_skeleton_version": "v2"},
        "qa_pairs": [
            {
                "qa_id": "stream_train_qa_direct_parent_1",
                "qa_type": "direct_parent",
                "question": "What is the direct parent of event 1?",
                "target_event_id": 1,
                "answer": 0,
                "answer_source": "ground_truth.parents",
            },
            {
                "qa_id": "stream_train_qa_temporal_order",
                "qa_type": "temporal_order",
                "question": "List all event IDs in ascending timestamp order.",
                "target_event_id": None,
                "answer": [0, 2, 1],
                "answer_source": "events.timestamp",
            },
        ],
    }


class TrainingExportTests(unittest.TestCase):
    def test_render_stream_input_hides_ground_truth_fields(self) -> None:
        text = render_stream_input(_row())

        self.assertIn("event_id=0", text)
        self.assertIn("timestamp=1.2", text)
        self.assertIn("api timeout alert", text)
        self.assertNotIn("ground_truth", text)
        self.assertNotIn("parent_id", text)
        self.assertNotIn("root_id", text)
        self.assertNotIn("relation_to_parent", text)
        self.assertNotIn("regime", text)
        self.assertNotIn("is_background", text)

    def test_export_training_data_expands_qa_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            accepted = Path(tmp) / "accepted.jsonl"
            output = Path(tmp) / "train.jsonl"
            accepted.write_text(json.dumps(_row(), sort_keys=True) + "\n", encoding="utf-8")

            report = export_training_data(
                accepted_jsonl=accepted,
                output_jsonl=output,
                output_format="instruction_jsonl",
            )

            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(report["num_training_examples"], 2)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["id"], "stream_train:stream_train_qa_direct_parent_1")
            self.assertEqual(rows[0]["task_type"], "direct_parent")
            self.assertEqual(rows[0]["instruction"], "What is the direct parent of event 1?")
            self.assertEqual(rows[0]["output"], "0")
            self.assertEqual(rows[0]["answer"], 0)
            self.assertEqual(rows[0]["answer_source"], "ground_truth.parents")
            self.assertNotIn("ground_truth", rows[0]["input"])

    def test_export_training_data_can_write_chat_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            accepted = Path(tmp) / "accepted.jsonl"
            output = Path(tmp) / "chat.jsonl"
            accepted.write_text(json.dumps(_row(), sort_keys=True) + "\n", encoding="utf-8")

            export_training_data(
                accepted_jsonl=accepted,
                output_jsonl=output,
                output_format="chat_jsonl",
            )

            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["id"], "stream_train:stream_train_qa_direct_parent_1")
            self.assertEqual([message["role"] for message in row["messages"]], ["system", "user", "assistant"])
            self.assertEqual(row["messages"][-1]["content"], "0")

    def test_training_export_cli_writes_jsonl_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            accepted = Path(tmp) / "accepted.jsonl"
            output = Path(tmp) / "train.jsonl"
            report = Path(tmp) / "report.json"
            accepted.write_text(json.dumps(_row(), sort_keys=True) + "\n", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    "data_generation/run_data_generation.py",
                    "--mode",
                    "export-training",
                    "--accepted-jsonl",
                    str(accepted),
                    "--output-jsonl",
                    str(output),
                    "--report-json",
                    str(report),
                    "--format",
                    "instruction_jsonl",
                ],
                check=True,
            )

            self.assertTrue(output.exists())
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["num_input_streams"], 1)
            self.assertEqual(payload["num_training_examples"], 2)

    def test_dataset_builder_can_export_training_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run_export"
            manifest = build_dataset(
                {
                    "generation_mode": "generic",
                    "output_dir": str(output_dir),
                    "num_samples": 3,
                    "seed": 41,
                    "semantic_mode": "rule",
                    "qa_mode": "template",
                    "domains": ["IT/System"],
                    "topologies": ["chain"],
                    "training_export": {
                        "enabled": True,
                        "format": "instruction_jsonl",
                    },
                }
            )

            train_path = output_dir / "training" / "train.jsonl"
            report_path = output_dir / "reports" / "training_export.json"
            self.assertTrue(train_path.exists())
            self.assertTrue(report_path.exists())
            self.assertIn("training", manifest["outputs"])
            self.assertIn("training_export", manifest["outputs"])
            train_rows = [
                json.loads(line)
                for line in train_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreater(len(train_rows), 0)
            self.assertNotIn("ground_truth", train_rows[0]["input"])


if __name__ == "__main__":
    unittest.main()
