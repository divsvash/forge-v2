"""
test_workflows.py — unit tests for workflows.py

Tests are isolated: every case gets its own TemporaryDirectory for the
SQLite DB (via patch.object) so no state leaks between tests and nothing
is ever written to the real forge_memory.db.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock

import workflows
import memory


class TestWorkflowCRUD(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test_wf.db"
        self._ws = TemporaryDirectory()
        # DB_PATH must be patched on BOTH modules since workflows.py
        # re-exports it from memory.DB_PATH at import time.
        with patch.object(memory, "DB_PATH", self.db_path), \
             patch.object(workflows, "DB_PATH", self.db_path):
            self.project_id = memory.get_or_create_project(Path(self._ws.name))

    def tearDown(self):
        self._tmp.cleanup()
        self._ws.cleanup()

    def _save(self, name="my-wf", steps=None, **kwargs):
        if steps is None:
            steps = [
                workflows.WorkflowStep("step1", "do {thing}"),
                workflows.WorkflowStep("step2", "verify {thing}"),
            ]
        return workflows.save_workflow(
            self.project_id, name, steps, db_path=self.db_path, **kwargs
        )

    # --- save / get / list ---

    def test_save_and_get_round_trip(self):
        wf_id = self._save("review", description="A review workflow")
        wf = workflows.get_workflow(self.project_id, "review", db_path=self.db_path)
        self.assertIsNotNone(wf)
        self.assertEqual(wf.id, wf_id)
        self.assertEqual(wf.name, "review")
        self.assertEqual(wf.description, "A review workflow")
        self.assertEqual(len(wf.steps), 2)

    def test_get_returns_none_for_unknown_name(self):
        wf = workflows.get_workflow(self.project_id, "nonexistent", db_path=self.db_path)
        self.assertIsNone(wf)

    def test_save_is_idempotent_on_name(self):
        """Saving the same name twice updates, not duplicates."""
        self._save("my-wf")
        updated_steps = [workflows.WorkflowStep("only-step", "do everything")]
        self._save("my-wf", steps=updated_steps, description="updated")

        wfs = workflows.list_workflows(self.project_id, db_path=self.db_path)
        self.assertEqual(len(wfs), 1)
        self.assertEqual(len(wfs[0].steps), 1)
        self.assertEqual(wfs[0].description, "updated")

    def test_list_workflows_alphabetical(self):
        self._save("zebra")
        self._save("apple")
        self._save("mango")
        wfs = workflows.list_workflows(self.project_id, db_path=self.db_path)
        self.assertEqual([w.name for w in wfs], ["apple", "mango", "zebra"])

    def test_list_workflows_empty_for_new_project(self):
        with TemporaryDirectory() as ws2:
            with patch.object(memory, "DB_PATH", self.db_path), \
                 patch.object(workflows, "DB_PATH", self.db_path):
                pid2 = memory.get_or_create_project(Path(ws2))
            wfs = workflows.list_workflows(pid2, db_path=self.db_path)
            self.assertEqual(wfs, [])

    def test_delete_workflow(self):
        self._save("to-delete")
        deleted = workflows.delete_workflow(
            self.project_id, "to-delete", db_path=self.db_path
        )
        self.assertTrue(deleted)
        wf = workflows.get_workflow(self.project_id, "to-delete", db_path=self.db_path)
        self.assertIsNone(wf)

    def test_delete_nonexistent_returns_false(self):
        result = workflows.delete_workflow(
            self.project_id, "ghost", db_path=self.db_path
        )
        self.assertFalse(result)

    def test_workflows_scoped_per_project(self):
        with TemporaryDirectory() as ws2:
            with patch.object(memory, "DB_PATH", self.db_path), \
                 patch.object(workflows, "DB_PATH", self.db_path):
                pid2 = memory.get_or_create_project(Path(ws2))
            self._save("shared-name")
            # Same name, different project — independent rows
            workflows.save_workflow(
                pid2, "shared-name",
                [workflows.WorkflowStep("x", "do x")],
                db_path=self.db_path,
            )
            wf1 = workflows.get_workflow(self.project_id, "shared-name", db_path=self.db_path)
            wf2 = workflows.get_workflow(pid2, "shared-name", db_path=self.db_path)
            self.assertNotEqual(wf1.id, wf2.id)
            self.assertEqual(len(wf1.steps), 2)
            self.assertEqual(len(wf2.steps), 1)

    def test_stop_on_failure_persists(self):
        self._save("halt-wf", stop_on_failure=True)
        wf = workflows.get_workflow(self.project_id, "halt-wf", db_path=self.db_path)
        self.assertTrue(wf.stop_on_failure)


class TestWorkflowStep(unittest.TestCase):
    def test_render_fills_placeholders(self):
        step = workflows.WorkflowStep("s", "inspect {filename} in branch {branch}")
        rendered = step.render({"filename": "calc.py", "branch": "main"})
        self.assertEqual(rendered, "inspect calc.py in branch main")

    def test_render_raises_on_missing_placeholder(self):
        step = workflows.WorkflowStep("s", "do {missing}")
        with self.assertRaises(KeyError):
            step.render({})

    def test_render_with_no_placeholders(self):
        step = workflows.WorkflowStep("s", "list all files")
        self.assertEqual(step.render({}), "list all files")

    def test_to_dict_round_trip(self):
        step = workflows.WorkflowStep("my-step", "prompt {x}")
        d = step.to_dict()
        restored = workflows.WorkflowStep.from_dict(d)
        self.assertEqual(restored.name, step.name)
        self.assertEqual(restored.prompt_template, step.prompt_template)


class TestRunWorkflow(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test_run_wf.db"
        self._ws = TemporaryDirectory()
        with patch.object(memory, "DB_PATH", self.db_path), \
             patch.object(workflows, "DB_PATH", self.db_path):
            self.project_id = memory.get_or_create_project(Path(self._ws.name))

        # Save a 2-step workflow with a placeholder
        workflows.save_workflow(
            self.project_id,
            "review",
            [
                workflows.WorkflowStep("inspect", "inspect {filename}"),
                workflows.WorkflowStep("test",    "test {filename}"),
            ],
            db_path=self.db_path,
        )

    def tearDown(self):
        self._tmp.cleanup()
        self._ws.cleanup()

    def _fake_run_task(self, prompt):
        return {
            "run_id": f"fake-{prompt[:8].replace(' ', '-')}",
            "final_status": "complete",
            "answer": f"done: {prompt}",
        }

    def test_run_workflow_calls_run_task_per_step(self):
        calls = []

        def capturing_run_task(prompt):
            calls.append(prompt)
            return {"run_id": "r1", "final_status": "complete", "answer": "ok"}

        result = workflows.run_workflow(
            self.project_id, "review",
            params={"filename": "calc.py"},
            run_task_fn=capturing_run_task,
            db_path=self.db_path,
        )
        self.assertEqual(len(calls), 2)
        self.assertIn("calc.py", calls[0])
        self.assertIn("calc.py", calls[1])
        self.assertEqual(result.status, "complete")

    def test_run_workflow_result_has_step_results(self):
        result = workflows.run_workflow(
            self.project_id, "review",
            params={"filename": "calc.py"},
            run_task_fn=self._fake_run_task,
            db_path=self.db_path,
        )
        self.assertEqual(len(result.step_results), 2)
        self.assertEqual(result.step_results[0].step_name, "inspect")
        self.assertEqual(result.step_results[1].step_name, "test")

    def test_run_nonexistent_workflow_returns_error(self):
        result = workflows.run_workflow(
            self.project_id, "no-such-workflow",
            run_task_fn=self._fake_run_task,
            db_path=self.db_path,
        )
        self.assertEqual(result.status, "error")
        self.assertIn("no-such-workflow", result.step_results[0].error)

    def test_missing_placeholder_captured_as_error_step(self):
        result = workflows.run_workflow(
            self.project_id, "review",
            params={},  # filename not supplied
            run_task_fn=self._fake_run_task,
            db_path=self.db_path,
        )
        self.assertEqual(result.step_results[0].status, "error")
        self.assertIn("Missing workflow parameter", result.step_results[0].error)

    def test_stop_on_failure_skips_later_steps(self):
        workflows.save_workflow(
            self.project_id,
            "halt-wf",
            [
                workflows.WorkflowStep("step1", "step one"),
                workflows.WorkflowStep("step2", "step two"),
                workflows.WorkflowStep("step3", "step three"),
            ],
            stop_on_failure=True,
            db_path=self.db_path,
        )

        call_count = [0]

        def failing_first(prompt):
            call_count[0] += 1
            if "one" in prompt:
                return {"run_id": "r1", "final_status": "error", "answer": None}
            return {"run_id": "r2", "final_status": "complete", "answer": "ok"}

        result = workflows.run_workflow(
            self.project_id, "halt-wf",
            run_task_fn=failing_first,
            db_path=self.db_path,
        )
        # Only the first step should have been called
        self.assertEqual(call_count[0], 1)
        self.assertEqual(result.step_results[1].status, "skipped")
        self.assertEqual(result.step_results[2].status, "skipped")
        self.assertEqual(result.status, "partial")

    def test_run_records_saved_to_db(self):
        workflows.run_workflow(
            self.project_id, "review",
            params={"filename": "calc.py"},
            run_task_fn=self._fake_run_task,
            db_path=self.db_path,
        )
        runs = workflows.list_workflow_runs(
            self.project_id, db_path=self.db_path
        )
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["workflow_name"], "review")
        self.assertEqual(runs[0]["status"], "complete")
        self.assertEqual(len(runs[0]["step_results"]), 2)

    def test_list_workflow_runs_filters_by_name(self):
        workflows.save_workflow(
            self.project_id, "other",
            [workflows.WorkflowStep("x", "do x")],
            db_path=self.db_path,
        )
        workflows.run_workflow(
            self.project_id, "review",
            params={"filename": "f.py"},
            run_task_fn=self._fake_run_task,
            db_path=self.db_path,
        )
        workflows.run_workflow(
            self.project_id, "other",
            run_task_fn=self._fake_run_task,
            db_path=self.db_path,
        )

        runs = workflows.list_workflow_runs(
            self.project_id, workflow_name="review", db_path=self.db_path
        )
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["workflow_name"], "review")

    def test_workflow_result_to_dict_is_serialisable(self):
        import json as _json
        result = workflows.run_workflow(
            self.project_id, "review",
            params={"filename": "calc.py"},
            run_task_fn=self._fake_run_task,
            db_path=self.db_path,
        )
        # Must not raise
        serialised = _json.dumps(result.to_dict())
        self.assertIn("review", serialised)


class TestWorkflowToolsInForge(unittest.TestCase):
    """
    End-to-end: forge.py's save_workflow_tool / list_workflows_tool /
    run_workflow_tool / recall_workflow_runs_tool wired up correctly.
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "forge_wf.db"
        self._ws = TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()
        self._ws.cleanup()

    def test_save_and_list_via_forge_tools(self):
        import forge

        with patch.object(forge, "WORKSPACE", Path(self._ws.name)), \
             patch.object(memory, "DB_PATH", self.db_path), \
             patch.object(workflows, "DB_PATH", self.db_path):
            save_result = forge.save_workflow_tool(
                name="hello",
                steps=[{"name": "greet", "prompt_template": "say hello"}],
                description="a greeting",
            )
            self.assertEqual(save_result["name"], "hello")

            listed = forge.list_workflows_tool()
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["name"], "hello")

    def test_run_workflow_via_forge_tool(self):
        import forge

        calls = []

        def mock_run_task(prompt):
            calls.append(prompt)
            return {"run_id": "r1", "final_status": "complete", "answer": "done"}

        with patch.object(forge, "WORKSPACE", Path(self._ws.name)), \
             patch.object(memory, "DB_PATH", self.db_path), \
             patch.object(workflows, "DB_PATH", self.db_path), \
             patch.object(forge, "run_task", mock_run_task):

            forge.save_workflow_tool(
                name="greet",
                steps=[{"name": "say", "prompt_template": "hello {who}"}],
            )
            result = forge.run_workflow_tool("greet", params={"who": "world"})

        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(calls), 1)
        self.assertIn("world", calls[0])

    def test_recall_workflow_runs_via_forge_tool(self):
        import forge

        def mock_run_task(prompt):
            return {"run_id": "r1", "final_status": "complete", "answer": "done"}

        with patch.object(forge, "WORKSPACE", Path(self._ws.name)), \
             patch.object(memory, "DB_PATH", self.db_path), \
             patch.object(workflows, "DB_PATH", self.db_path), \
             patch.object(forge, "run_task", mock_run_task):

            forge.save_workflow_tool(
                name="greet",
                steps=[{"name": "say", "prompt_template": "hello"}],
            )
            forge.run_workflow_tool("greet")
            runs = forge.recall_workflow_runs_tool(workflow_name="greet")

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
