import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from event_stream_generator.evaluation.compare import compare_generation_outputs
from event_stream_generator.evaluation.metrics import l1_frequency_distance


def _write_real_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "block_id", "template_id"])
        writer.writeheader()
        for index in range(6):
            base = float(index * 10)
            block = f"blk_{index}"
            writer.writerow({"timestamp": base, "block_id": block, "template_id": "E5"})
            writer.writerow({"timestamp": base + 0.5, "block_id": block, "template_id": "E11"})
            writer.writerow({"timestamp": base + 1.5, "block_id": block, "template_id": "E21"})


def _sample(stream_id: str, event_types, timestamps):
    return {
        "stream_id": stream_id,
        "domain": "HDFS",
        "events": [
            {
                "event_id": index,
                "timestamp": timestamp,
                "abstract_type": event_type,
                "is_background": False,
            }
            for index, (event_type, timestamp) in enumerate(zip(event_types, timestamps))
        ],
    }


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class EvaluationTests(unittest.TestCase):
    def test_l1_frequency_distance_compares_normalized_counts(self):
        self.assertEqual(l1_frequency_distance({"A": 2}, {"A": 1}), 0.0)
        self.assertEqual(l1_frequency_distance({"A": 1}, {"B": 1}), 2.0)

    def test_compare_generation_outputs_reports_calibrated_closer_than_generic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_csv = root / "real.csv"
            generic = root / "generic.jsonl"
            calibrated = root / "calibrated.jsonl"
            _write_real_csv(real_csv)
            _write_jsonl(
                generic,
                [
                    _sample(f"g{i}", ["A", "B", "C"], [0.0, 4.0, 9.0])
                    for i in range(6)
                ],
            )
            _write_jsonl(
                calibrated,
                [
                    _sample(f"c{i}", ["E5", "E11", "E21"], [0.0, 0.5, 1.5])
                    for i in range(6)
                ],
            )

            report, rows = compare_generation_outputs(
                real_csv=real_csv,
                generic_jsonl=generic,
                calibrated_jsonl=calibrated,
                timestamp_col="timestamp",
                stream_id_col="block_id",
                event_type_col="template_id",
            )

            self.assertLess(
                report["inter_event_time"]["calibrated_vs_real"]["wasserstein"],
                report["inter_event_time"]["generic_vs_real"]["wasserstein"],
            )
            self.assertLess(
                report["event_type_frequency"]["calibrated_vs_real"]["l1"],
                report["event_type_frequency"]["generic_vs_real"]["l1"],
            )
            self.assertTrue(any(row["metric"] == "transition_frequency_l1" for row in rows))

    def test_evaluate_generation_cli_writes_json_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_csv = root / "real.csv"
            generic = root / "generic.jsonl"
            calibrated = root / "calibrated.jsonl"
            out_json = root / "report.json"
            out_csv = root / "metrics.csv"
            _write_real_csv(real_csv)
            _write_jsonl(generic, [_sample("g0", ["A", "B", "C"], [0.0, 4.0, 9.0])])
            _write_jsonl(calibrated, [_sample("c0", ["E5", "E11", "E21"], [0.0, 0.5, 1.5])])

            subprocess.run(
                [
                    sys.executable,
                    "event_stream_generator/scripts/evaluate_generation.py",
                    "--real-csv",
                    str(real_csv),
                    "--generic-jsonl",
                    str(generic),
                    "--calibrated-jsonl",
                    str(calibrated),
                    "--output-json",
                    str(out_json),
                    "--output-csv",
                    str(out_csv),
                    "--timestamp-col",
                    "timestamp",
                    "--stream-id-col",
                    "block_id",
                    "--event-type-col",
                    "template_id",
                ],
                check=True,
            )

            self.assertTrue(out_json.exists())
            self.assertTrue(out_csv.exists())
            report = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertIn("inter_event_time", report)
            self.assertIn("metric,comparison,value", out_csv.read_text(encoding="utf-8").splitlines()[0])


if __name__ == "__main__":
    unittest.main()
