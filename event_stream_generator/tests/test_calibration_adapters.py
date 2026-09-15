import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from event_stream_generator.calibration.adapters import get_adapter
from event_stream_generator.calibration.pipeline import run_calibration_pipeline


def _read_prepared(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class CalibrationAdapterTests(unittest.TestCase):
    def test_hdfs_event_traces_adapter_expands_feature_sequences(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "Event_traces.csv"
            raw.write_text(
                "BlockId,Features,TimeInterval\n"
                "blk_1,\"[E1,E2,E3]\",\"[0,0.5,1.5]\"\n"
                "blk_2,\"[E1,E3]\",\"[0,2.0]\"\n",
                encoding="utf-8",
            )
            prepared = Path(tmp) / "prepared.csv"

            summary = get_adapter("hdfs_event_traces").prepare(
                raw,
                prepared,
                {"max_streams": 1},
            )

            rows = _read_prepared(prepared)
            self.assertEqual(summary["stream_count"], 1)
            self.assertEqual(summary["event_count"], 3)
            self.assertEqual(
                rows,
                [
                    {"timestamp": "0", "stream_id": "blk_1", "event_type": "E1"},
                    {"timestamp": "0.5", "stream_id": "blk_1", "event_type": "E2"},
                    {"timestamp": "2", "stream_id": "blk_1", "event_type": "E3"},
                ],
            )

    def test_taxi_pro_adapter_reads_driver_json_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "taxi"
            raw_dir.mkdir()
            (raw_dir / "driver_1.json").write_text(
                json.dumps(
                    [
                        {"time": 2.0, "TPP_attribute": {"event_type": 4}},
                        {"time": 1.0, "TPP_attribute": {"event_type": 3}},
                    ]
                ),
                encoding="utf-8",
            )
            prepared = Path(tmp) / "prepared.csv"

            summary = get_adapter("taxi_pro").prepare(raw_dir, prepared, {})

            rows = _read_prepared(prepared)
            self.assertEqual(summary["stream_count"], 1)
            self.assertEqual(rows[0]["stream_id"], "driver_1")
            self.assertEqual(rows[0]["event_type"], "taxi_event_3")
            self.assertEqual(rows[1]["event_type"], "taxi_event_4")

    def test_gdelt_adapter_builds_actor_pair_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "gdelt.tsv"
            columns = [""] * 58
            rows = []
            for index, code in enumerate(["010", "020", "030"]):
                row = list(columns)
                row[4] = str(2026.01 + index * 0.001)
                row[5] = "USA"
                row[15] = "CHN"
                row[26] = code
                rows.append("\t".join(row))
            raw.write_text("\n".join(rows) + "\n", encoding="utf-8")
            prepared = Path(tmp) / "prepared.csv"

            summary = get_adapter("gdelt").prepare(
                raw,
                prepared,
                {"min_events_per_stream": 3},
            )

            rows = _read_prepared(prepared)
            self.assertEqual(summary["stream_count"], 1)
            self.assertEqual(summary["event_count"], 3)
            self.assertEqual(rows[0]["stream_id"], "USA->CHN")
            self.assertEqual(rows[0]["event_type"], "cameo_010")

    def test_pipeline_writes_prepared_prior_and_summary_from_yaml_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "taxi"
            raw_dir.mkdir()
            (raw_dir / "driver_1.json").write_text(
                json.dumps(
                    [
                        {"time": 0.0, "TPP_attribute": {"event_type": 1}},
                        {"time": 1.0, "TPP_attribute": {"event_type": 2}},
                    ]
                ),
                encoding="utf-8",
            )
            (raw_dir / "driver_2.json").write_text(
                json.dumps(
                    [
                        {"time": 0.0, "TPP_attribute": {"event_type": 1}},
                        {"time": 2.0, "TPP_attribute": {"event_type": 2}},
                    ]
                ),
                encoding="utf-8",
            )
            config = tmp_path / "config.yaml"
            prepared = tmp_path / "prepared.csv"
            prior = tmp_path / "prior.json"
            report = tmp_path / "report.json"
            config.write_text(
                "\n".join(
                    [
                        "mode: calibrate-dataset",
                        "dataset_name: taxi-test",
                        "adapter: taxi_pro",
                        "domain: Transportation",
                        f"input_path: {raw_dir}",
                        f"prepared_output: {prepared}",
                        f"prior_output: {prior}",
                        f"report_output: {report}",
                        "min_transition_samples: 2",
                    ]
                ),
                encoding="utf-8",
            )

            result = run_calibration_pipeline(config)

            self.assertTrue(prepared.exists())
            self.assertTrue(prior.exists())
            self.assertTrue(report.exists())
            self.assertEqual(result["dataset_name"], "taxi-test")
            self.assertEqual(result["prior"]["stream_count"], 2)
            self.assertIn("taxi_event_1->taxi_event_2", result["prior"]["transition_priors"])

    def test_calibrate_dataset_cli_accepts_yaml_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "Event_traces.csv"
            raw.write_text(
                "BlockId,Features,TimeInterval\n"
                "blk_1,\"[E1,E2]\",\"[0,1.0]\"\n"
                "blk_2,\"[E1,E2]\",\"[0,2.0]\"\n",
                encoding="utf-8",
            )
            config = tmp_path / "config.yaml"
            prior = tmp_path / "prior.json"
            config.write_text(
                "\n".join(
                    [
                        "mode: calibrate-dataset",
                        "dataset_name: hdfs-test",
                        "adapter: hdfs_event_traces",
                        "domain: HDFS",
                        f"input_path: {raw}",
                        f"prepared_output: {tmp_path / 'prepared.csv'}",
                        f"prior_output: {prior}",
                        "min_transition_samples: 2",
                    ]
                ),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    "event_stream_generator/scripts/calibrate_dataset.py",
                    "--config",
                    str(config),
                ],
                check=True,
            )

            payload = json.loads(prior.read_text(encoding="utf-8"))
            self.assertEqual(payload["domain"], "HDFS")
            self.assertIn("E1->E2", payload["transition_priors"])


if __name__ == "__main__":
    unittest.main()
