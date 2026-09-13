from __future__ import annotations

import ast
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class SecondRoundLayoutTests(unittest.TestCase):
    def test_functional_modules_contain_real_implementations(self) -> None:
        for relative_path, required_symbol in {
            "core/models.py": "EventStream",
            "core/io.py": "load_yaml",
            "simulation/simulator.py": "generate_event_stream",
            "simulation/background.py": "generate_background_events",
            "simulation/calibrated.py": "generate_calibrated_event_stream",
            "skeleton/mechanism.py": "build_chain_mechanism",
            "skeleton/semantic_skeleton.py": "build_semantic_skeleton",
            "temporal/history.py": "build_history_state",
            "temporal/policy.py": "select_regime",
            "export/training_export.py": "export_training_data",
            "semantic/compatibility.py": "apply_semantic_compatible_sampling",
            "semantic/gate.py": "apply_semantic_quality_gate",
            "semantic/llm_client.py": "OpenAIResponsesClient",
        }.items():
            path = PACKAGE_ROOT / relative_path
            tree = ast.parse(path.read_text(encoding="utf-8"))
            defined_symbols = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            self.assertIn(required_symbol, defined_symbols, relative_path)

    def test_package_root_does_not_contain_function_wrapper_modules(self) -> None:
        root_modules = {
            path.name for path in PACKAGE_ROOT.glob("*.py") if path.name != "__init__.py"
        }
        self.assertEqual({"coverage_report.py", "dataset_builder.py"}, root_modules)


if __name__ == "__main__":
    unittest.main()
