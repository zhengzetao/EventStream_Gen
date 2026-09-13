import random
import unittest

from event_stream_generator.skeleton.mechanism import build_chain_mechanism, build_tree_mechanism


class MechanismTests(unittest.TestCase):
    def test_chain_parent_root_and_path_are_recoverable(self):
        mechanism = build_chain_mechanism(
            stream_id="stream_x",
            domain="IT/System",
            mechanism_type="multi_hop_propagation",
            length=4,
            relation_type="direct_trigger",
            primary_regime="gamma",
            rng=random.Random(7),
        )

        self.assertEqual(mechanism.get_parent("E3"), "E2")
        self.assertEqual(mechanism.get_root("E3"), "E0")
        self.assertEqual(mechanism.get_path("E3"), ["E0", "E1", "E2", "E3"])
        self.assertTrue(mechanism.validate_dag())

    def test_tree_nodes_have_single_parent_and_root_path(self):
        mechanism = build_tree_mechanism(
            stream_id="stream_tree",
            domain="Finance",
            mechanism_type="direct_trigger",
            node_count=8,
            max_depth=3,
            max_children_per_node=3,
            relation_type="weak_trigger",
            primary_regime="exponential",
            rng=random.Random(11),
        )

        self.assertEqual(len(mechanism.nodes), 8)
        for node in mechanism.nodes:
            if node.node_id == "E0":
                continue
            self.assertIsNotNone(mechanism.get_parent(node.node_id))
            self.assertEqual(mechanism.get_root(node.node_id), "E0")
            self.assertEqual(mechanism.get_path(node.node_id)[0], "E0")
        self.assertTrue(mechanism.validate_dag())


if __name__ == "__main__":
    unittest.main()
