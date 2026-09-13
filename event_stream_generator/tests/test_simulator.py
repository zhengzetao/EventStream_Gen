import unittest

from event_stream_generator.simulation.simulator import generate_event_stream


class SimulatorTests(unittest.TestCase):
    def test_generation_is_reproducible_for_fixed_seed(self):
        config = {
            "mechanism": {
                "chain": {"min_length": 5, "max_length": 5},
                "tree": {"min_nodes": 6, "max_nodes": 6, "max_depth": 3, "max_children_per_node": 2},
            },
            "stream": {"min_events": 1, "max_events": 40, "max_generation_retry": 5},
            "background": {
                "density": "medium",
                "rates": {"low": 0.05, "medium": 0.2, "high": 0.6},
                "max_background_events": 20,
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
        kwargs = {
            "stream_id": "stream_000000",
            "domain": "IT/System",
            "topology": "chain",
            "mechanism_type": "multi_hop_propagation",
            "primary_regime": "exponential",
            "seed": 42,
            "config": config,
        }

        left = generate_event_stream(**kwargs).to_dict()
        right = generate_event_stream(**kwargs).to_dict()

        self.assertEqual(left, right)
        timestamps = [event["timestamp"] for event in left["events"]]
        self.assertEqual(timestamps, sorted(timestamps))
        event_ids = [event["event_id"] for event in left["events"]]
        self.assertNotEqual(event_ids, sorted(event_ids))
        self.assertEqual(left["metadata"]["semantic_status"], "not_instantiated")


if __name__ == "__main__":
    unittest.main()
