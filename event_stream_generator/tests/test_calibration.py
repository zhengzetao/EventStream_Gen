import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from event_stream_generator.calibration.extractor import calibrate_event_csv
from event_stream_generator.calibration.fit_regimes import fit_best_regime
from event_stream_generator.calibration.hdfs import calibrate_hdfs_csv


def _write_rows(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


class CalibrationTests(unittest.TestCase):
    def test_generic_extractor_calibrates_field_mapped_event_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            _write_rows(
                path,
                [
                    {"ts": "0.0", "case": "s1", "code": "A"},
                    {"ts": "1.0", "case": "s1", "code": "B"},
                    {"ts": "3.0", "case": "s1", "code": "C"},
                    {"ts": "0.0", "case": "s2", "code": "A"},
                    {"ts": "2.0", "case": "s2", "code": "B"},
                    {"ts": "5.0", "case": "s2", "code": "C"},
                ],
            )

            prior = calibrate_event_csv(
                path,
                timestamp_col="ts",
                stream_id_col="case",
                event_type_col="code",
                domain="Healthcare",
                min_transition_samples=2,
            )

            self.assertEqual(prior["domain"], "Healthcare")
            self.assertEqual(prior["stream_count"], 2)
            self.assertEqual(prior["event_frequencies"], {"A": 2, "B": 2, "C": 2})
            self.assertEqual(prior["first_order_transitions"]["A"]["B"]["count"], 2)
            self.assertEqual(prior["first_order_transitions"]["A"]["B"]["probability"], 1.0)
            self.assertEqual(prior["second_order_transitions"]["A|B"]["C"]["count"], 2)
            self.assertIn("A->B", prior["transition_priors"])
            self.assertEqual(prior["transition_priors"]["A->B"]["sample_count"], 2)
            self.assertIn(prior["transition_priors"]["A->B"]["regime"], {"exponential", "gamma", "lognormal", "weibull"})

    def test_hdfs_wrapper_uses_block_id_and_template_id_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hdfs.csv"
            _write_rows(
                path,
                [
                    {"timestamp": "0", "block_id": "blk_1", "template_id": "E5"},
                    {"timestamp": "0.5", "block_id": "blk_1", "template_id": "E11"},
                    {"timestamp": "0", "block_id": "blk_2", "template_id": "E5"},
                    {"timestamp": "0.7", "block_id": "blk_2", "template_id": "E11"},
                ],
            )

            prior = calibrate_hdfs_csv(path, min_transition_samples=2)

            self.assertEqual(prior["domain"], "HDFS")
            self.assertEqual(prior["input_schema"]["stream_id_col"], "block_id")
            self.assertEqual(prior["input_schema"]["event_type_col"], "template_id")
            self.assertEqual(prior["transition_priors"]["E5->E11"]["sample_count"], 2)

    def test_fit_best_regime_uses_backoff_when_samples_are_insufficient(self):
        fit = fit_best_regime([0.5, 0.7], min_samples=5)

        self.assertEqual(fit["regime"], "generic_backoff")
        self.assertEqual(fit["sample_count"], 2)
        self.assertEqual(fit["fit"]["status"], "insufficient_samples")

    def test_fit_best_regime_handles_zero_variance_samples(self):
        fit = fit_best_regime([1.0, 1.0, 1.0, 1.0, 1.0], min_samples=5)

        self.assertEqual(fit["regime"], "deterministic_backoff")
        self.assertEqual(fit["params"], {"value": 1.0})
        self.assertEqual(fit["sample_count"], 5)
        self.assertEqual(fit["fit"]["status"], "zero_variance_samples")

    def test_calibrate_events_cli_writes_prior_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            out = Path(tmp) / "prior.json"
            _write_rows(
                path,
                [
                    {"timestamp": "0", "session": "s1", "event": "A"},
                    {"timestamp": "1", "session": "s1", "event": "B"},
                    {"timestamp": "0", "session": "s2", "event": "A"},
                    {"timestamp": "2", "session": "s2", "event": "B"},
                ],
            )

            subprocess.run(
                [
                    sys.executable,
                    "event_stream_generator/scripts/calibrate_events.py",
                    "--input",
                    str(path),
                    "--output",
                    str(out),
                    "--timestamp-col",
                    "timestamp",
                    "--stream-id-col",
                    "session",
                    "--event-type-col",
                    "event",
                    "--domain",
                    "GenericTest",
                    "--min-transition-samples",
                    "2",
                ],
                check=True,
            )

            prior = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(prior["domain"], "GenericTest")
            self.assertIn("A->B", prior["transition_priors"])

    def test_calibrate_hdfs_cli_writes_prior_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hdfs.csv"
            out = Path(tmp) / "hdfs_prior.json"
            _write_rows(
                path,
                [
                    {"timestamp": "0", "block_id": "blk_1", "template_id": "E1"},
                    {"timestamp": "1", "block_id": "blk_1", "template_id": "E2"},
                    {"timestamp": "0", "block_id": "blk_2", "template_id": "E1"},
                    {"timestamp": "2", "block_id": "blk_2", "template_id": "E2"},
                ],
            )

            subprocess.run(
                [
                    sys.executable,
                    "event_stream_generator/scripts/calibrate_hdfs.py",
                    "--input",
                    str(path),
                    "--output",
                    str(out),
                    "--min-transition-samples",
                    "2",
                ],
                check=True,
            )

            prior = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(prior["domain"], "HDFS")
            self.assertEqual(prior["input_schema"]["stream_id_col"], "block_id")


if __name__ == "__main__":
    unittest.main()
