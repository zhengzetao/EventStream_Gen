import csv
import tempfile
import unittest
from pathlib import Path

from event_stream_generator.calibration.extractor import calibrate_event_csv
from event_stream_generator.core.models import EventStream, GeneratedEvent
from event_stream_generator.simulation.calibrated import (
    _sample_event_type_chain,
    _sample_next_event,
)
from event_stream_generator.validators.temporal import validate_temporal


def _write_rows(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "stream_id", "event_type"])
        writer.writeheader()
        writer.writerows(rows)


class CalibrationHardeningTests(unittest.TestCase):
    def test_extractor_adds_domain_delta_profile_and_backoff_priors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            _write_rows(
                path,
                [
                    {"timestamp": "0", "stream_id": "s1", "event_type": "A"},
                    {"timestamp": "1", "stream_id": "s1", "event_type": "B"},
                    {"timestamp": "3", "stream_id": "s1", "event_type": "C"},
                    {"timestamp": "0", "stream_id": "s2", "event_type": "A"},
                    {"timestamp": "1.5", "stream_id": "s2", "event_type": "B"},
                    {"timestamp": "4.5", "stream_id": "s2", "event_type": "C"},
                    {"timestamp": "0", "stream_id": "s3", "event_type": "A"},
                    {"timestamp": "9", "stream_id": "s3", "event_type": "D"},
                ],
            )

            prior = calibrate_event_csv(
                path,
                timestamp_col="timestamp",
                stream_id_col="stream_id",
                event_type_col="event_type",
                domain="TestDomain",
                min_transition_samples=3,
            )

            self.assertIn("delta_t_profile", prior)
            self.assertEqual(prior["delta_t_profile"]["count"], 5)
            self.assertIn("p95", prior["delta_t_profile"])
            self.assertIn("event_type_delta_priors", prior)
            self.assertIn("A", prior["event_type_delta_priors"])
            self.assertIn("domain_delta_prior", prior)
            self.assertEqual(
                prior["transition_priors"]["A->D"]["fit"]["status"],
                "insufficient_samples",
            )
            self.assertEqual(
                prior["transition_priors"]["A->D"]["backoff_source"],
                "event_type",
            )
            self.assertIn("effective_regime", prior["transition_priors"]["A->D"])

    def test_generator_uses_second_order_transition_when_available(self):
        prior = {
            "event_frequencies": {"A": 10, "B": 10, "C": 10, "D": 10},
            "first_order_transitions": {
                "A": {"B": {"probability": 1.0}},
                "B": {"C": {"probability": 0.01}, "D": {"probability": 0.99}},
                "C": {"C": {"probability": 1.0}},
                "D": {"D": {"probability": 1.0}},
            },
            "second_order_transitions": {
                "A|B": {"C": {"probability": 1.0}},
            },
        }

        sequence = _sample_event_type_chain(
            prior,
            3,
            __import__("random").Random(4),
            {"use_second_order": True},
        )

        self.assertEqual(sequence, ["A", "B", "C"])

    def test_generator_smoothing_keeps_rare_transition_sampleable(self):
        transitions = {
            "A": {
                "B": {"probability": 1.0},
            },
            "B": {
                "B": {"probability": 1.0},
            },
        }
        next_event = _sample_next_event(
            transitions,
            "A",
            __import__("random").Random(1),
            {"smoothing_alpha": 0.5},
        )

        self.assertIn(next_event, {"A", "B"})

    def test_temporal_validator_uses_calibration_aware_dense_threshold(self):
        stream = EventStream(
            stream_id="calibrated_dense",
            domain="Politics",
            topology="chain",
            mechanism_type="multi_hop_propagation",
            primary_regime="calibrated_transition",
            seed=1,
            mechanism={"nodes": [], "edges": []},
            events=[
                GeneratedEvent(0, 0.0, "A", None, None, 0, None, None, None, {}, False, 0),
                GeneratedEvent(1, 1e-13, "B", None, 0, 0, "direct_trigger", 1e-13, "deterministic_backoff", {"value": 1e-13}, False, 1),
            ],
            ground_truth={},
            validation={},
            metadata={
                "generation_mode": "calibrated",
                "calibration_profile": {"min_positive_delta_t": 1e-13},
            },
        )

        result = validate_temporal(
            stream,
            config={
                "stream": {"min_events": 2, "max_events": 2},
                "validation": {"temporal": {"calibration_aware": True}},
            },
        )

        self.assertTrue(result.approved)
        self.assertEqual(result.metrics["min_gap_threshold"], 1e-14)

    def test_calibration_aware_dense_threshold_never_becomes_stricter(self):
        stream = EventStream(
            stream_id="calibrated_normal_min_gap",
            domain="HDFS",
            topology="chain",
            mechanism_type="multi_hop_propagation",
            primary_regime="calibrated_transition",
            seed=1,
            mechanism={"nodes": [], "edges": []},
            events=[
                GeneratedEvent(0, 0.0, "A", None, None, 0, None, None, None, {}, False, 0),
                GeneratedEvent(1, 1e-6, "B", None, 0, 0, "direct_trigger", 1e-6, "deterministic_backoff", {"value": 1e-6}, False, 1),
            ],
            ground_truth={},
            validation={},
            metadata={
                "generation_mode": "calibrated",
                "calibration_profile": {"min_positive_delta_t": 1.0},
            },
        )

        result = validate_temporal(
            stream,
            config={
                "stream": {"min_events": 2, "max_events": 2},
                "validation": {"temporal": {"calibration_aware": True}},
            },
        )

        self.assertTrue(result.approved)
        self.assertEqual(result.metrics["min_gap_threshold"], 1e-12)


if __name__ == "__main__":
    unittest.main()
