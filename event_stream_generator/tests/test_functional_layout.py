from __future__ import annotations

import unittest


class FunctionalLayoutTests(unittest.TestCase):
    def test_new_functional_namespaces_export_core_objects(self) -> None:
        from event_stream_generator.core.models import EventStream
        from event_stream_generator.skeleton.mechanism import build_chain_mechanism
        from event_stream_generator.skeleton.semantic_skeleton import build_semantic_skeleton
        from event_stream_generator.simulation.simulator import generate_event_stream
        from event_stream_generator.temporal.history import HistoryState
        from event_stream_generator.semantic.instantiation import instantiate_scenario_llm
        from event_stream_generator.judge.semantic_judge import judge_semantic_scenario
        from event_stream_generator.export.training_export import render_stream_input

        self.assertEqual(EventStream.__name__, "EventStream")
        self.assertTrue(callable(build_chain_mechanism))
        self.assertTrue(callable(build_semantic_skeleton))
        self.assertTrue(callable(generate_event_stream))
        self.assertEqual(HistoryState.__name__, "HistoryState")
        self.assertTrue(callable(instantiate_scenario_llm))
        self.assertTrue(callable(judge_semantic_scenario))
        self.assertTrue(callable(render_stream_input))

    def test_cli_package_exports_main_dispatcher(self) -> None:
        from event_stream_generator.cli import main as package_main
        from event_stream_generator.cli.main import main as module_main

        self.assertIs(package_main, module_main)


if __name__ == "__main__":
    unittest.main()
