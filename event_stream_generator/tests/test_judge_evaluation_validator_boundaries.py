from __future__ import annotations

import ast
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class JudgeEvaluationValidatorBoundaryTests(unittest.TestCase):
    def test_judge_modules_own_semantic_judge_and_quality_implementations(self) -> None:
        for relative_path, required_symbol in {
            "judge/semantic_judge.py": "judge_semantic_scenario",
            "judge/semantic_quality.py": "score_semantic_quality",
        }.items():
            symbols = _defined_symbols(PACKAGE_ROOT / relative_path)
            self.assertIn(required_symbol, symbols, relative_path)

    def test_validators_do_not_contain_judge_wrappers(self) -> None:
        for relative_path in [
            "validators/semantic_judge.py",
            "validators/semantic_quality.py",
        ]:
            path = PACKAGE_ROOT / relative_path
            self.assertFalse(path.exists(), relative_path)

    def test_judge_does_not_contain_dataset_audit_wrapper(self) -> None:
        self.assertFalse((PACKAGE_ROOT / "judge" / "semantic_audit.py").exists())

    def test_dataset_level_semantic_evaluators_use_judge_namespace(self) -> None:
        for relative_path in [
            "evaluation/semantic_audit.py",
            "evaluation/llm_semantic_quality.py",
        ]:
            source = (PACKAGE_ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn("event_stream_generator.judge.semantic_judge", source)
            self.assertIn("event_stream_generator.judge.semantic_quality", source)
            self.assertNotIn("event_stream_generator.validators.semantic_judge", source)
            self.assertNotIn("event_stream_generator.validators.semantic_quality", source)


def _defined_symbols(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]


if __name__ == "__main__":
    unittest.main()
