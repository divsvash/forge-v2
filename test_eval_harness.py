import contextlib
import io
import unittest
from pathlib import Path

import eval_harness
import forge
from eval_harness import (
    DEFAULT_CASES, run_case, run_suite, summarize, compare_baseline_vs_candidate,
)


class TestEvalHarness(unittest.TestCase):
    def setUp(self):
        self._original_messages = list(forge.messages)
        self._stdout = contextlib.redirect_stdout(io.StringIO())
        self._stdout.__enter__()

    def tearDown(self):
        self._stdout.__exit__(None, None, None)
        forge.messages[:] = self._original_messages

    def test_default_cases_pass_against_real_forge(self):
        results = run_suite(DEFAULT_CASES)
        failures = [(r.case_id, r.reason, r.error) for r in results if not r.passed]
        self.assertEqual(failures, [], f"unexpected failures: {failures}")

    def test_summarize_reports_pass_rate(self):
        results = run_suite(DEFAULT_CASES)
        summary = summarize(results)
        self.assertEqual(summary["total"], len(DEFAULT_CASES))
        self.assertEqual(summary["passed"], len(DEFAULT_CASES))
        self.assertEqual(summary["pass_rate"], 1.0)

    def test_case_run_isolated_in_its_own_workspace(self):
        seen_workspaces = []

        def setup(workspace: Path):
            seen_workspaces.append(workspace)
            (workspace / "x.py").write_text("x = 1\n")

        def check(result, workspace):
            return True, "ok"

        case = eval_harness.TaskCase(
            case_id="isolated",
            description="just checks workspace isolation",
            prompt="inspect x.py",
            setup=setup,
            check=check,
            scripted_responses=[
                '{"tool":"read_file","arguments":{"filename":"x.py"}}',
                "x.py sets x = 1.",
            ],
        )
        result = run_case(case)
        self.assertTrue(result.passed)
        # The workspace must not be the real repo workspace.
        self.assertNotEqual(seen_workspaces[0], forge.WORKSPACE)

    def test_comparison_detects_regression(self):
        def passing_run_task(prompt, max_steps=10, max_stalls=3):
            return {
                "completion": {"success": True, "missing": []},
                "evidence": {
                    "relevant file contents": [{"arguments": {"filename": "a.py"}}],
                    "test execution result": [{"x": 1}],
                    "failed command execution": [{"x": 1}],
                },
                "answer": "ok", "steps": 1, "tool_history": [], "run_id": "r1",
            }

        def regressing_run_task(prompt, max_steps=10, max_stalls=3):
            return {
                "completion": {"success": False, "missing": ["x"]},
                "evidence": {
                    "relevant file contents": [],
                    "test execution result": [],
                    "failed command execution": [],
                },
                "answer": "fail", "steps": 1, "tool_history": [], "run_id": "r2",
            }

        comparison = compare_baseline_vs_candidate(
            DEFAULT_CASES, passing_run_task, regressing_run_task
        )
        self.assertFalse(comparison.is_safe_to_promote)
        self.assertEqual(set(comparison.regressions), {c.case_id for c in DEFAULT_CASES})

    def test_comparison_allows_promotion_with_no_regressions(self):
        comparison = compare_baseline_vs_candidate(
            DEFAULT_CASES, forge.run_task, forge.run_task
        )
        self.assertTrue(comparison.is_safe_to_promote)
        self.assertEqual(comparison.regressions, [])


if __name__ == "__main__":
    unittest.main()
