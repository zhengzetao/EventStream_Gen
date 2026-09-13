import unittest

from event_stream_generator.core.models import EventStream, GeneratedEvent
from event_stream_generator.validators.structural import validate_structure
from event_stream_generator.validators.temporal import validate_temporal


class ValidatorTests(unittest.TestCase):
    def test_structural_validator_rejects_missing_parent(self):
        stream = EventStream(
            stream_id="bad_parent",
            domain="IT/System",
            topology="chain",
            mechanism_type="direct_trigger",
            primary_regime="exponential",
            seed=1,
            mechanism={"nodes": [], "edges": []},
            events=[
                GeneratedEvent(0, 0.0, "E0", None, None, 0, None, None, None, None, {}, False, 0),
                GeneratedEvent(1, 1.0, "E1", None, 99, 0, "direct_trigger", 1.0, "exponential", {"lambda": 1.0}, False, 1),
            ],
            ground_truth={},
            validation={},
            metadata={"time_unit": "abstract_float"},
        )

        result = validate_structure(stream)
        self.assertFalse(result.approved)
        self.assertTrue(any("missing parent" in error for error in result.errors))

    def test_structural_validator_rejects_mechanism_event_mismatch(self):
        stream = EventStream(
            stream_id="bad_mechanism_alignment",
            domain="IT/System",
            topology="chain",
            mechanism_type="direct_trigger",
            primary_regime="exponential",
            seed=1,
            mechanism={
                "nodes": [
                    {"node_id": "E0", "role": "root", "abstract_type": "E0"},
                    {"node_id": "E1", "role": "target", "abstract_type": "E1"},
                ],
                "edges": [
                    {
                        "source": "E0",
                        "target": "E1",
                        "relation_type": "direct_trigger",
                        "regime": "exponential",
                        "regime_params": {},
                    }
                ],
            },
            events=[
                GeneratedEvent(0, 0.0, "E0", None, None, 0, None, None, None, None, {}, False, 0),
            ],
            ground_truth={},
            validation={},
            metadata={"time_unit": "abstract_float"},
        )

        result = validate_structure(stream)
        self.assertFalse(result.approved)
        self.assertTrue(any("mechanism node E1 has no event" in error for error in result.errors))

    def test_structural_validator_rejects_tree_branching_over_config_limit(self):
        stream = EventStream(
            stream_id="bad_tree_branching",
            domain="IT/System",
            topology="tree",
            mechanism_type="direct_trigger",
            primary_regime="exponential",
            seed=1,
            mechanism={
                "nodes": [
                    {"node_id": "E0", "role": "root", "abstract_type": "E0"},
                    {"node_id": "E1", "role": "target", "abstract_type": "E1"},
                    {"node_id": "E2", "role": "target", "abstract_type": "E2"},
                    {"node_id": "E3", "role": "target", "abstract_type": "E3"},
                ],
                "edges": [
                    {"source": "E0", "target": "E1", "relation_type": "direct_trigger"},
                    {"source": "E0", "target": "E2", "relation_type": "direct_trigger"},
                    {"source": "E0", "target": "E3", "relation_type": "direct_trigger"},
                ],
            },
            events=[
                GeneratedEvent(0, 0.0, "E0", None, None, 0, None, None, None, None, {}, False, 0, "E0"),
                GeneratedEvent(1, 1.0, "E1", None, 0, 0, "direct_trigger", 1.0, "exponential", {"lambda": 1.0}, False, 1, "E1"),
                GeneratedEvent(2, 2.0, "E2", None, 0, 0, "direct_trigger", 2.0, "exponential", {"lambda": 1.0}, False, 1, "E2"),
                GeneratedEvent(3, 3.0, "E3", None, 0, 0, "direct_trigger", 3.0, "exponential", {"lambda": 1.0}, False, 1, "E3"),
            ],
            ground_truth={},
            validation={},
            metadata={"time_unit": "abstract_float"},
        )

        result = validate_structure(
            stream,
            config={"mechanism": {"tree": {"max_children_per_node": 2, "max_depth": 3}}},
        )
        self.assertFalse(result.approved)
        self.assertTrue(any("too many tree children" in error for error in result.errors))

    def test_temporal_validator_rejects_inconsistent_delta(self):
        stream = EventStream(
            stream_id="bad_delta",
            domain="IT/System",
            topology="chain",
            mechanism_type="direct_trigger",
            primary_regime="exponential",
            seed=1,
            mechanism={"nodes": [], "edges": []},
            events=[
                GeneratedEvent(0, 0.0, "E0", None, None, 0, None, None, None, None, {}, False, 0),
                GeneratedEvent(1, 2.0, "E1", None, 0, 0, "direct_trigger", 1.0, "exponential", {"lambda": 1.0}, False, 1),
            ],
            ground_truth={},
            validation={},
            metadata={"time_unit": "abstract_float"},
        )

        result = validate_temporal(stream)
        self.assertFalse(result.approved)
        self.assertTrue(any("delta_t mismatch" in error for error in result.errors))

    def test_temporal_validator_rejects_event_count_outside_config_range(self):
        stream = EventStream(
            stream_id="too_short",
            domain="IT/System",
            topology="chain",
            mechanism_type="direct_trigger",
            primary_regime="exponential",
            seed=1,
            mechanism={"nodes": [], "edges": []},
            events=[
                GeneratedEvent(0, 0.0, "E0", None, None, 0, None, None, None, None, {}, False, 0),
            ],
            ground_truth={},
            validation={},
            metadata={"time_unit": "abstract_float"},
        )

        result = validate_temporal(stream, config={"stream": {"min_events": 3, "max_events": 5}})
        self.assertFalse(result.approved)
        self.assertTrue(any("event count outside configured range" in error for error in result.errors))

    def test_temporal_validator_rejects_background_ratio_outside_density_range(self):
        stream = EventStream(
            stream_id="too_many_background",
            domain="IT/System",
            topology="chain",
            mechanism_type="direct_trigger",
            primary_regime="exponential",
            seed=1,
            mechanism={"nodes": [], "edges": []},
            events=[
                GeneratedEvent(0, 0.0, "E0", None, None, 0, None, None, None, None, {}, False, 0),
                GeneratedEvent(1, 1.0, "BG0", None, None, None, None, None, "exponential", {"lambda": 1.0}, True, 0),
                GeneratedEvent(2, 2.0, "BG1", None, None, None, None, None, "exponential", {"lambda": 1.0}, True, 0),
                GeneratedEvent(3, 3.0, "BG2", None, None, None, None, None, "exponential", {"lambda": 1.0}, True, 0),
            ],
            ground_truth={},
            validation={},
            metadata={"time_unit": "abstract_float", "background_density": "low"},
        )

        result = validate_temporal(
            stream,
            config={"background": {"density_ranges": {"low": [0.0, 0.5]}}},
        )
        self.assertFalse(result.approved)
        self.assertTrue(any("background ratio outside configured range" in error for error in result.errors))


if __name__ == "__main__":
    unittest.main()
