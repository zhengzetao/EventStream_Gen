import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from event_stream_generator.simulation.calibrated import generate_calibrated_event_stream


CALIBRATION_PRIOR = {
    "domain": "HDFS",
    "calibration_type": "transition_interevent_time",
    "stream_count": 10,
    "event_count": 30,
    "event_frequencies": {"E5": 10, "E11": 10, "E21": 10},
    "first_order_transitions": {
        "E5": {"E11": {"count": 10, "probability": 1.0}},
        "E11": {"E21": {"count": 10, "probability": 1.0}},
    },
    "second_order_transitions": {
        "E5|E11": {"E21": {"count": 10, "probability": 1.0}}
    },
    "transition_priors": {
        "E5->E11": {
            "probability": 1.0,
            "regime": "deterministic_backoff",
            "params": {"value": 0.5},
            "sample_count": 10,
            "fit": {"status": "zero_variance_samples"},
        },
        "E11->E21": {
            "probability": 1.0,
            "regime": "deterministic_backoff",
            "params": {"value": 1.25},
            "sample_count": 10,
            "fit": {"status": "zero_variance_samples"},
        },
    },
    "sequence_length_distribution": {"3": 10},
    "backoff": {
        "min_transition_samples": 5,
        "order": ["specific_transition", "event_type", "domain", "generic"],
    },
}


GENERIC_CONFIG = {
    "mechanism": {
        "chain": {"min_length": 3, "max_length": 3},
        "tree": {
            "min_nodes": 5,
            "max_nodes": 5,
            "max_depth": 3,
            "max_children_per_node": 2,
        },
    },
    "stream": {"min_events": 3, "max_events": 20, "min_window": 5.0, "max_generation_retry": 5},
    "background": {
        "density": "low",
        "rates": {"low": 0.05, "medium": 0.15, "high": 0.35},
        "density_ranges": {"low": [0.0, 0.6]},
        "max_background_events": 5,
    },
    "temporal_policy": {
        "relation_defaults": {
            "direct_trigger": {"regimes": {"exponential": 1.0}},
            "delayed_trigger": {"regimes": {"gamma": 1.0}},
            "weak_trigger": {"regimes": {"lognormal": 1.0}},
            "recovery": {"regimes": {"weibull": 1.0}},
        }
    },
    "regime_params": {
        "exponential": {"lambda": 1.0},
        "gamma": {"shape": 2.0, "scale": 0.8},
        "lognormal": {"mu": 0.2, "sigma": 0.5},
        "weibull": {"shape": 1.5, "scale": 1.0},
    },
}


class CalibratedGenerationTests(unittest.TestCase):
    def test_calibrated_chain_uses_event_vocabulary_and_transition_delta(self):
        stream = generate_calibrated_event_stream(
            stream_id="calibrated_000000",
            calibration_prior=CALIBRATION_PRIOR,
            seed=3,
            config=GENERIC_CONFIG,
            topology="chain",
            mechanism_type="multi_hop_propagation",
        )

        causal = [event for event in stream.events if not event.is_background]
        self.assertEqual([event.abstract_type for event in causal], ["E5", "E11", "E21"])
        self.assertEqual(causal[1].delta_t_from_parent, 0.5)
        self.assertEqual(causal[2].delta_t_from_parent, 1.25)
        self.assertEqual(causal[1].regime, "deterministic_backoff")
        self.assertEqual(stream.metadata["generation_mode"], "calibrated")
        self.assertEqual(stream.metadata["calibration_domain"], "HDFS")

    def test_generate_calibrated_cli_writes_valid_samples_with_qa(self):
        with tempfile.TemporaryDirectory() as tmp:
            prior_path = Path(tmp) / "hdfs_prior.json"
            output_path = Path(tmp) / "hdfs_synthetic.jsonl"
            prior_path.write_text(json.dumps(CALIBRATION_PRIOR), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    "event_stream_generator/scripts/generate_calibrated.py",
                    "--calibration-prior",
                    str(prior_path),
                    "--num-samples",
                    "6",
                    "--output",
                    str(output_path),
                    "--seed",
                    "0",
                    "--semantic-mode",
                    "rule",
                    "--qa-mode",
                    "template",
                ],
                check=True,
            )

            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rejected = [
                json.loads(line)
                for line in output_path.with_name("hdfs_synthetic_rejected.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 6)
            self.assertEqual(rejected, [])
            self.assertEqual(rows[0]["metadata"]["generation_mode"], "calibrated")
            self.assertEqual(rows[0]["domain"], "HDFS")
            self.assertTrue(rows[0]["validation"]["label"]["approved"])
            self.assertTrue(rows[0]["qa_pairs"])
            causal_types = [
                event["abstract_type"]
                for event in rows[0]["events"]
                if not event["is_background"]
            ]
            self.assertEqual(causal_types, ["E5", "E11", "E21"])


if __name__ == "__main__":
    unittest.main()
