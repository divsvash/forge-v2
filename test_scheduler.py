"""
test_scheduler.py — unit tests for scheduler.py

Every test uses an isolated TemporaryDirectory for the SQLite DB via
patch.object on scheduler.DB_PATH, memory.DB_PATH, and workflows.DB_PATH
so the real forge_memory.db is never touched.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import memory
import workflows
import scheduler


def _patch_db(db_path):
    """Return a combined context that patches all three DB_PATH refs."""
    import contextlib
    return contextlib.ExitStack()  # caller stacks the patches manually


class SchedulerTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test_sched.db"
        self._ws = TemporaryDirectory()

        # Patch all three modules so they share the isolated DB.
        self._p_mem = patch.object(memory, "DB_PATH", self.db_path)
        self._p_wf = patch.object(workflows, "DB_PATH", self.db_path)
        self._p_sc = patch.object(scheduler, "DB_PATH", self.db_path)
        self._p_mem.start()
        self._p_wf.start()
        self._p_sc.start()

        self.project_id = memory.get_or_create_project(Path(self._ws.name))

    def tearDown(self):
        self._p_sc.stop()
        self._p_wf.stop()
        self._p_mem.stop()
        self._tmp.cleanup()
        self._ws.cleanup()


class TestScheduleAndList(SchedulerTestBase):
    def test_schedule_task_returns_job_id(self):
        job_id = scheduler.schedule_task(
            self.project_id, "lint",
            "run linter",
            trigger_type="one_shot",
            trigger_config={"at": 9999999999.0},
            db_path=self.db_path,
        )
        self.assertIsInstance(job_id, int)
        self.assertGreater(job_id, 0)

    def test_get_job_round_trip(self):
        job_id = scheduler.schedule_task(
            self.project_id, "lint", "run linter",
            trigger_type="one_shot",
            trigger_config={"at": 9999999999.0},
            db_path=self.db_path,
        )
        job = scheduler.get_job(job_id, db_path=self.db_path)
        self.assertIsNotNone(job)
        self.assertEqual(job.name, "lint")
        self.assertEqual(job.kind, "task")
        self.assertEqual(job.trigger_type, "one_shot")
        self.assertEqual(job.payload["prompt"], "run linter")
        self.assertEqual(job.status, "active")

    def test_get_job_returns_none_for_unknown_id(self):
        self.assertIsNone(scheduler.get_job(99999, db_path=self.db_path))

    def test_list_jobs_returns_scheduled(self):
        scheduler.schedule_task(
            self.project_id, "j1", "do thing 1",
            trigger_type="interval", trigger_config={"seconds": 60},
            db_path=self.db_path,
        )
        scheduler.schedule_task(
            self.project_id, "j2", "do thing 2",
            trigger_type="interval", trigger_config={"seconds": 120},
            db_path=self.db_path,
        )
        jobs = scheduler.list_jobs(self.project_id, db_path=self.db_path)
        self.assertEqual(len(jobs), 2)

    def test_list_jobs_filters_by_status(self):
        job_id = scheduler.schedule_task(
            self.project_id, "j1", "p",
            trigger_type="interval", trigger_config={"seconds": 60},
            db_path=self.db_path,
        )
        scheduler.pause_job(job_id, db_path=self.db_path)
        active = scheduler.list_jobs(self.project_id, status="active", db_path=self.db_path)
        paused = scheduler.list_jobs(self.project_id, status="paused", db_path=self.db_path)
        self.assertEqual(len(active), 0)
        self.assertEqual(len(paused), 1)

    def test_schedule_workflow_job(self):
        # Save a workflow first
        workflows.save_workflow(
            self.project_id, "review",
            [workflows.WorkflowStep("step", "inspect {f}")],
            db_path=self.db_path,
        )
        job_id = scheduler.schedule_workflow(
            self.project_id, "daily-review", "review",
            params={"f": "calc.py"},
            trigger_type="interval",
            trigger_config={"seconds": 86400},
            db_path=self.db_path,
        )
        job = scheduler.get_job(job_id, db_path=self.db_path)
        self.assertEqual(job.kind, "workflow")
        self.assertEqual(job.payload["workflow_name"], "review")
        self.assertEqual(job.payload["params"]["f"], "calc.py")

    def test_jobs_scoped_per_project(self):
        with TemporaryDirectory() as ws2:
            pid2 = memory.get_or_create_project(Path(ws2))
        scheduler.schedule_task(
            self.project_id, "mine", "p",
            trigger_type="interval", trigger_config={"seconds": 1},
            db_path=self.db_path,
        )
        scheduler.schedule_task(
            pid2, "theirs", "p",
            trigger_type="interval", trigger_config={"seconds": 1},
            db_path=self.db_path,
        )
        mine = scheduler.list_jobs(self.project_id, db_path=self.db_path)
        self.assertEqual(len(mine), 1)
        self.assertEqual(mine[0].name, "mine")


class TestPauseResumeCancelJob(SchedulerTestBase):
    def _new_job(self):
        return scheduler.schedule_task(
            self.project_id, "j", "p",
            trigger_type="interval", trigger_config={"seconds": 60},
            db_path=self.db_path,
        )

    def test_pause_active_job(self):
        job_id = self._new_job()
        result = scheduler.pause_job(job_id, db_path=self.db_path)
        self.assertTrue(result)
        self.assertEqual(scheduler.get_job(job_id, db_path=self.db_path).status, "paused")

    def test_pause_nonexistent_returns_false(self):
        self.assertFalse(scheduler.pause_job(9999, db_path=self.db_path))

    def test_resume_paused_job(self):
        job_id = self._new_job()
        scheduler.pause_job(job_id, db_path=self.db_path)
        result = scheduler.resume_job(job_id, db_path=self.db_path)
        self.assertTrue(result)
        self.assertEqual(scheduler.get_job(job_id, db_path=self.db_path).status, "active")

    def test_cancel_job_marks_done(self):
        job_id = self._new_job()
        result = scheduler.cancel_job(job_id, db_path=self.db_path)
        self.assertTrue(result)
        self.assertEqual(scheduler.get_job(job_id, db_path=self.db_path).status, "done")

    def test_pause_already_paused_returns_false(self):
        job_id = self._new_job()
        scheduler.pause_job(job_id, db_path=self.db_path)
        # Pausing again should return False (already paused)
        result = scheduler.pause_job(job_id, db_path=self.db_path)
        self.assertFalse(result)


class TestTriggerHelpers(SchedulerTestBase):
    def test_one_shot_next_run_is_the_at_value(self):
        at = 1700000000.0
        nxt = scheduler._next_run_for_trigger("one_shot", {"at": at}, after=0)
        self.assertEqual(nxt, at)

    def test_one_shot_past_at_returns_none(self):
        at = 1000.0
        nxt = scheduler._next_run_for_trigger("one_shot", {"at": at}, after=2000.0)
        self.assertIsNone(nxt)

    def test_interval_adds_seconds(self):
        nxt = scheduler._next_run_for_trigger("interval", {"seconds": 300}, after=1000.0)
        self.assertEqual(nxt, 1300.0)

    def test_unknown_trigger_type_raises(self):
        with self.assertRaises(ValueError):
            scheduler._next_run_for_trigger("unknown", {}, after=0)

    def test_initial_next_run_one_shot(self):
        at = 9999999999.0
        nxt = scheduler._initial_next_run("one_shot", {"at": at})
        self.assertEqual(nxt, at)

    def test_initial_next_run_interval_is_in_the_future(self):
        import time
        nxt = scheduler._initial_next_run("interval", {"seconds": 60})
        self.assertGreater(nxt, time.time())


class TestTick(SchedulerTestBase):
    """Core execution engine tests — no real run_task calls."""

    def _mock_run_task(self, status="complete"):
        calls = []

        def run_task(prompt):
            calls.append(prompt)
            return {"run_id": "r1", "final_status": status, "answer": "done"}

        return run_task, calls

    def test_tick_runs_due_job(self):
        run_task, calls = self._mock_run_task()
        job_id = scheduler.schedule_task(
            self.project_id, "j", "say hello",
            trigger_type="one_shot",
            trigger_config={"at": 1000.0},
            db_path=self.db_path,
        )
        summaries = scheduler.tick(
            self.project_id,
            run_task_fn=run_task,
            db_path=self.db_path,
            _now=2000.0,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], "say hello")
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["run_status"], "complete")

    def test_tick_does_not_run_future_job(self):
        run_task, calls = self._mock_run_task()
        scheduler.schedule_task(
            self.project_id, "j", "p",
            trigger_type="one_shot",
            trigger_config={"at": 9999999999.0},
            db_path=self.db_path,
        )
        scheduler.tick(
            self.project_id,
            run_task_fn=run_task,
            db_path=self.db_path,
            _now=1000.0,
        )
        self.assertEqual(len(calls), 0)

    def test_tick_does_not_run_paused_job(self):
        run_task, calls = self._mock_run_task()
        job_id = scheduler.schedule_task(
            self.project_id, "j", "p",
            trigger_type="one_shot",
            trigger_config={"at": 1000.0},
            db_path=self.db_path,
        )
        scheduler.pause_job(job_id, db_path=self.db_path)
        scheduler.tick(
            self.project_id,
            run_task_fn=run_task,
            db_path=self.db_path,
            _now=2000.0,
        )
        self.assertEqual(len(calls), 0)

    def test_one_shot_job_becomes_done_after_success(self):
        run_task, _ = self._mock_run_task()
        job_id = scheduler.schedule_task(
            self.project_id, "j", "p",
            trigger_type="one_shot",
            trigger_config={"at": 1000.0},
            db_path=self.db_path,
        )
        scheduler.tick(
            self.project_id, run_task_fn=run_task,
            db_path=self.db_path, _now=2000.0,
        )
        job = scheduler.get_job(job_id, db_path=self.db_path)
        self.assertEqual(job.status, "done")

    def test_interval_job_rescheduled_after_success(self):
        run_task, _ = self._mock_run_task()
        job_id = scheduler.schedule_task(
            self.project_id, "j", "p",
            trigger_type="interval",
            trigger_config={"seconds": 300},
            db_path=self.db_path,
        )
        # Force next_run_at to be in the past
        with scheduler._open(self.db_path) as conn:
            conn.execute("UPDATE jobs SET next_run_at=1000.0 WHERE id=?", (job_id,))

        scheduler.tick(
            self.project_id, run_task_fn=run_task,
            db_path=self.db_path, _now=2000.0,
        )
        job = scheduler.get_job(job_id, db_path=self.db_path)
        self.assertEqual(job.status, "active")
        # next_run should be ~300s after the tick ended (somewhere > 2000)
        self.assertGreater(job.next_run_at, 2000.0)

    def test_failed_job_retried_with_backoff(self):
        run_task, _ = self._mock_run_task(status="error")
        job_id = scheduler.schedule_task(
            self.project_id, "j", "p",
            trigger_type="one_shot",
            trigger_config={"at": 1000.0},
            max_retries=3,
            retry_delay_secs=60.0,
            db_path=self.db_path,
        )
        scheduler.tick(
            self.project_id, run_task_fn=run_task,
            db_path=self.db_path, _now=2000.0,
        )
        job = scheduler.get_job(job_id, db_path=self.db_path)
        # Still active, attempt incremented, next_run rescheduled
        self.assertEqual(job.status, "active")
        self.assertEqual(job.attempt, 1)
        self.assertGreater(job.next_run_at, 2000.0)

    def test_job_fails_permanently_after_max_retries(self):
        run_task, _ = self._mock_run_task(status="error")
        job_id = scheduler.schedule_task(
            self.project_id, "j", "p",
            trigger_type="one_shot",
            trigger_config={"at": 1000.0},
            max_retries=1,
            retry_delay_secs=1.0,
            db_path=self.db_path,
        )
        # Tick once — sets attempt=1, reschedules
        scheduler.tick(
            self.project_id, run_task_fn=run_task,
            db_path=self.db_path, _now=2000.0,
        )
        # Tick again at a time past the retry deadline
        scheduler.tick(
            self.project_id, run_task_fn=run_task,
            db_path=self.db_path, _now=9999.0,
        )
        job = scheduler.get_job(job_id, db_path=self.db_path)
        self.assertEqual(job.status, "failed")

    def test_tick_records_job_run(self):
        run_task, _ = self._mock_run_task()
        job_id = scheduler.schedule_task(
            self.project_id, "j", "p",
            trigger_type="one_shot",
            trigger_config={"at": 1000.0},
            db_path=self.db_path,
        )
        scheduler.tick(
            self.project_id, run_task_fn=run_task,
            db_path=self.db_path, _now=2000.0,
        )
        runs = scheduler.list_job_runs(job_id, db_path=self.db_path)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].status, "complete")

    def test_tick_captures_exception_without_raising(self):
        def exploding_run_task(prompt):
            raise RuntimeError("boom")

        job_id = scheduler.schedule_task(
            self.project_id, "j", "p",
            trigger_type="one_shot",
            trigger_config={"at": 1000.0},
            db_path=self.db_path,
        )
        # Must not raise
        summaries = scheduler.tick(
            self.project_id, run_task_fn=exploding_run_task,
            db_path=self.db_path, _now=2000.0,
        )
        self.assertEqual(summaries[0]["run_status"], "error")
        self.assertIn("boom", summaries[0]["error"])

    def test_tick_returns_empty_when_no_due_jobs(self):
        run_task, _ = self._mock_run_task()
        summaries = scheduler.tick(
            self.project_id, run_task_fn=run_task,
            db_path=self.db_path, _now=1000.0,
        )
        self.assertEqual(summaries, [])


class TestSchedulerForgeTools(SchedulerTestBase):
    """Smoke tests for forge.py's scheduler tool wrappers."""

    def test_schedule_task_tool_and_list_jobs_tool(self):
        import forge

        with patch.object(forge, "WORKSPACE", Path(self._ws.name)), \
             patch.object(memory, "DB_PATH", self.db_path), \
             patch.object(workflows, "DB_PATH", self.db_path), \
             patch.object(scheduler, "DB_PATH", self.db_path):

            result = forge.schedule_task_tool(
                name="nightly",
                prompt="run linter",
                trigger_type="interval",
                trigger_config={"seconds": 3600},
            )
            self.assertIn("job_id", result)

            jobs = forge.list_jobs_tool()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["name"], "nightly")

    def test_pause_resume_cancel_via_forge_tools(self):
        import forge

        with patch.object(forge, "WORKSPACE", Path(self._ws.name)), \
             patch.object(memory, "DB_PATH", self.db_path), \
             patch.object(workflows, "DB_PATH", self.db_path), \
             patch.object(scheduler, "DB_PATH", self.db_path):

            r = forge.schedule_task_tool(
                "j", "p", "interval", {"seconds": 60}
            )
            job_id = r["job_id"]

            pause_r = forge.pause_job_tool(job_id)
            self.assertTrue(pause_r["paused"])

            resume_r = forge.resume_job_tool(job_id)
            self.assertTrue(resume_r["resumed"])

            cancel_r = forge.cancel_job_tool(job_id)
            self.assertTrue(cancel_r["cancelled"])

    def test_recall_job_runs_via_forge_tool(self):
        import forge

        def mock_run_task(prompt):
            return {"run_id": "r1", "final_status": "complete", "answer": "done"}

        with patch.object(forge, "WORKSPACE", Path(self._ws.name)), \
             patch.object(memory, "DB_PATH", self.db_path), \
             patch.object(workflows, "DB_PATH", self.db_path), \
             patch.object(scheduler, "DB_PATH", self.db_path):

            r = forge.schedule_task_tool("j", "say hello", "one_shot", {"at": 1.0})
            job_id = r["job_id"]

            scheduler.tick(
                self.project_id,
                run_task_fn=mock_run_task,
                db_path=self.db_path,
                _now=9999.0,
            )

            runs = forge.recall_job_runs_tool(job_id)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
