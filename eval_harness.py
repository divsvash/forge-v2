"""
eval_harness.py — repeatable task cases and baseline-vs-candidate evaluation.

This is the "Forge needs a baseline before it can credibly claim a change
makes it smarter" piece of the foundations slice. It does three things:

  1. Defines TaskCase: a fixed, reproducible scenario — a scratch
     workspace, a prompt, and an *observable* success check (not "the
     model said it worked", but a check against the actual workspace
     and/or the actual run result).
  2. Runs a case (or a suite of cases) against a given `run_task`
     implementation, in an isolated temporary workspace, and reports
     pass/fail, step count and the full run record for each.
  3. Compares two implementations (baseline vs candidate) across the same
     suite and reports which cases regressed, improved, or held steady.
     A candidate is only worth keeping if this comparison says so.

Because this environment has no live Ollama model to call, every case
drives the model deterministically via `scripted_responses` — a fixed
sequence of model outputs — rather than mocking "success". This keeps
the *tool execution* and *completion checking* real while making the
case reproducible without a live LLM. Swap in a live `ask_model` and
drop `scripted_responses` to run the same cases against the real model.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from unittest.mock import patch

import forge
import run_record
import memory


@dataclass
class TaskCase:
    case_id: str
    description: str
    prompt: str
    setup: Callable[[Path], None]
    check: Callable[[dict, Path], tuple[bool, str]]
    scripted_responses: Optional[list[str]] = None
    max_steps: int = 10


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    reason: str
    steps: int
    run_id: Optional[str]
    error: Optional[str] = None


def run_case(case: TaskCase, run_task_fn=None) -> CaseResult:
    run_task_fn = run_task_fn or forge.run_task

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        case.setup(workspace)

        patches = [
            patch.object(forge, "WORKSPACE", workspace),
            # Keep each case's run record and project memory inside its
            # own scratch dir — otherwise every eval run would write
            # real run records / memory entries into the repo's own
            # runs/ directory and forge_memory.db.
            patch.object(run_record, "RUNS_DIR", workspace / "_runs"),
            patch.object(memory, "DB_PATH", workspace / "_memory.db"),
        ]
        if case.scripted_responses is not None:
            responses = iter(case.scripted_responses)
            patches.append(
                patch.object(forge, "ask_model", side_effect=lambda: next(responses))
            )

        for p in patches:
            p.start()
        try:
            try:
                result = run_task_fn(case.prompt, max_steps=case.max_steps)
            except Exception as error:
                return CaseResult(
                    case_id=case.case_id, passed=False,
                    reason="run_task raised an exception",
                    steps=0, run_id=None, error=str(error),
                )

            passed, reason = case.check(result, workspace)
            return CaseResult(
                case_id=case.case_id, passed=passed, reason=reason,
                steps=result.get("steps", 0), run_id=result.get("run_id"),
            )
        finally:
            for p in reversed(patches):
                p.stop()


def run_suite(cases: list[TaskCase], run_task_fn=None) -> list[CaseResult]:
    return [run_case(case, run_task_fn=run_task_fn) for case in cases]


def summarize(results: list[CaseResult]) -> dict:
    passed = [r for r in results if r.passed]
    return {
        "total": len(results),
        "passed": len(passed),
        "failed": len(results) - len(passed),
        "pass_rate": (len(passed) / len(results)) if results else 0.0,
    }


@dataclass
class ComparisonResult:
    baseline: list[CaseResult]
    candidate: list[CaseResult]
    regressions: list[str]   # case_ids: passed on baseline, failed on candidate
    improvements: list[str]  # case_ids: failed on baseline, passed on candidate
    unchanged: list[str]

    @property
    def is_safe_to_promote(self) -> bool:
        """A candidate is only promotable if it introduces zero regressions.
        Improvements with no regressions is the bar; equal-or-better alone
        (e.g. same pass count via a different failing case) is not enough,
        because that can hide a real regression behind an unrelated fix."""
        return len(self.regressions) == 0


def compare_baseline_vs_candidate(
    cases: list[TaskCase], baseline_fn, candidate_fn
) -> ComparisonResult:
    baseline_results = run_suite(cases, run_task_fn=baseline_fn)
    candidate_results = run_suite(cases, run_task_fn=candidate_fn)

    baseline_by_id = {r.case_id: r for r in baseline_results}
    candidate_by_id = {r.case_id: r for r in candidate_results}

    regressions, improvements, unchanged = [], [], []
    for case_id in baseline_by_id:
        before = baseline_by_id[case_id].passed
        after = candidate_by_id[case_id].passed
        if before and not after:
            regressions.append(case_id)
        elif not before and after:
            improvements.append(case_id)
        else:
            unchanged.append(case_id)

    return ComparisonResult(
        baseline=baseline_results,
        candidate=candidate_results,
        regressions=regressions,
        improvements=improvements,
        unchanged=unchanged,
    )


# ---------- DEFAULT BENCHMARK CASES ----------
# Small, fast, reproducible cases that exercise the real tool pipeline
# (actual files, actual run_command through the real sandbox backend,
# actual completion checking) against a scripted model, so the harness
# itself can be exercised and regression-tested without a live LLM.

def _setup_two_files(workspace: Path) -> None:
    (workspace / "a.py").write_text("def add(x, y):\n    return x + y\n")
    (workspace / "b.py").write_text("# unrelated\n")


def _check_inspected_a(result: dict, workspace: Path) -> tuple[bool, str]:
    if not result["completion"]["success"]:
        return False, f"not verified complete: {result['completion']['missing']}"
    inspected = {
        record["arguments"].get("filename")
        for record in result["evidence"]["relevant file contents"]
    }
    if "a.py" not in inspected:
        return False, f"a.py never inspected via read_file; saw {inspected}"
    return True, "a.py inspected and task marked complete"


CASE_INSPECT_FILE = TaskCase(
    case_id="inspect_single_file",
    description="Read a named file and report on it.",
    prompt="inspect the contents of a.py",
    setup=_setup_two_files,
    check=_check_inspected_a,
    scripted_responses=[
        '{"tool":"read_file","arguments":{"filename":"a.py"}}',
        "a.py defines add(x, y), which returns x + y.",
    ],
)


def _setup_passing_test(workspace: Path) -> None:
    (workspace / "calc.py").write_text("def add(x, y):\n    return x + y\n")
    (workspace / "test_calc.py").write_text(
        "import unittest\n"
        "from calc import add\n\n"
        "class TestAdd(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(1, 2), 3)\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )


def _check_test_ran(result: dict, workspace: Path) -> tuple[bool, str]:
    test_evidence = result["evidence"].get("test execution result", [])
    if not test_evidence:
        return False, "no successful test execution recorded"
    return True, "test command executed and recorded as evidence"


CASE_RUN_PASSING_TEST = TaskCase(
    case_id="run_passing_test",
    description="Run a test file that should pass; evidence must show it ran.",
    prompt="run the tests in test_calc.py",
    setup=_setup_passing_test,
    check=_check_test_ran,
    scripted_responses=[
        # unittest rather than pytest so the case has no extra dependency.
        '{"tool":"run_command","arguments":{"command":"python3 -m unittest test_calc"}}',
        "The test suite ran and test_add passed.",
    ],
    max_steps=5,
)


def _setup_failing_command(workspace: Path) -> None:
    (workspace / "broken.py").write_text("import sys\nsys.exit(1)\n")


def _check_failure_retained(result: dict, workspace: Path) -> tuple[bool, str]:
    failed = result["evidence"].get("failed command execution", [])
    if not failed:
        return False, "failed command was not retained as evidence"
    return True, "failed command retained as evidence (not silently dropped)"


CASE_FAILED_COMMAND_RETAINED = TaskCase(
    case_id="failed_command_retained_as_evidence",
    description=(
        "A failing command must still be recorded as evidence, not dropped "
        "(this is the gap called out explicitly in the foundations plan)."
    ),
    prompt="run broken.py and tell me what happens",
    setup=_setup_failing_command,
    check=_check_failure_retained,
    scripted_responses=[
        '{"tool":"run_command","arguments":{"command":"python3 broken.py"}}',
        "broken.py exits with status 1.",
    ],
    max_steps=5,
)


DEFAULT_CASES = [
    CASE_INSPECT_FILE,
    CASE_RUN_PASSING_TEST,
    CASE_FAILED_COMMAND_RETAINED,
]
