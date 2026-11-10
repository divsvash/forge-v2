import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import memory


class TestMemory(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test_memory.db"

    def tearDown(self):
        self._tmp.cleanup()

    def test_same_workspace_maps_to_same_project_across_calls(self):
        with TemporaryDirectory() as workspace:
            first = memory.get_or_create_project(Path(workspace), db_path=self.db_path)
            second = memory.get_or_create_project(Path(workspace), db_path=self.db_path)
            self.assertEqual(first, second)

    def test_different_workspaces_get_different_projects(self):
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            project_a = memory.get_or_create_project(Path(a), db_path=self.db_path)
            project_b = memory.get_or_create_project(Path(b), db_path=self.db_path)
            self.assertNotEqual(project_a, project_b)

    def test_remember_rejects_unknown_kind(self):
        with TemporaryDirectory() as workspace:
            project_id = memory.get_or_create_project(Path(workspace), db_path=self.db_path)
            with self.assertRaises(ValueError):
                memory.remember(
                    project_id, "opinion", "forge is cool", source="user",
                    db_path=self.db_path,
                )

    def test_recall_returns_only_active_entries_by_default(self):
        with TemporaryDirectory() as workspace:
            project_id = memory.get_or_create_project(Path(workspace), db_path=self.db_path)
            entry_id = memory.remember(
                project_id, "fact", "uses Python 3.11", source="user",
                db_path=self.db_path,
            )
            memory.supersede(
                entry_id, "uses Python 3.13", source="user", db_path=self.db_path,
            )

            active = memory.recall(project_id, db_path=self.db_path)
            contents = [entry.content for entry in active]
            self.assertIn("uses Python 3.13", contents)
            self.assertNotIn("uses Python 3.11", contents)

    def test_supersede_preserves_old_entry_in_history(self):
        with TemporaryDirectory() as workspace:
            project_id = memory.get_or_create_project(Path(workspace), db_path=self.db_path)
            entry_id = memory.remember(
                project_id, "fact", "uses Python 3.11", source="user",
                db_path=self.db_path,
            )
            new_id = memory.supersede(
                entry_id, "uses Python 3.13", source="user", db_path=self.db_path,
            )

            full_history = memory.history(project_id, db_path=self.db_path)
            contents = {entry.content for entry in full_history}
            self.assertIn("uses Python 3.11", contents)
            self.assertIn("uses Python 3.13", contents)

            old_entry = next(e for e in full_history if e.id == entry_id)
            self.assertFalse(old_entry.active)
            self.assertEqual(old_entry.superseded_by, new_id)

    def test_recall_filters_by_kind(self):
        with TemporaryDirectory() as workspace:
            project_id = memory.get_or_create_project(Path(workspace), db_path=self.db_path)
            memory.remember(project_id, "fact", "fact one", source="user", db_path=self.db_path)
            memory.remember(
                project_id, "assumption", "assumption one", source="forge",
                db_path=self.db_path,
            )

            facts = memory.recall(project_id, kind="fact", db_path=self.db_path)
            self.assertEqual(len(facts), 1)
            self.assertEqual(facts[0].content, "fact one")

    def test_recall_orders_most_recent_first(self):
        with TemporaryDirectory() as workspace:
            project_id = memory.get_or_create_project(Path(workspace), db_path=self.db_path)
            memory.remember(project_id, "fact", "first", source="user", db_path=self.db_path)
            memory.remember(project_id, "fact", "second", source="user", db_path=self.db_path)

            entries = memory.recall(project_id, db_path=self.db_path)
            self.assertEqual(entries[0].content, "second")

    def test_search_matches_substring(self):
        with TemporaryDirectory() as workspace:
            project_id = memory.get_or_create_project(Path(workspace), db_path=self.db_path)
            memory.remember(
                project_id, "decision", "chose SQLite over Postgres for simplicity",
                source="forge", db_path=self.db_path,
            )
            memory.remember(
                project_id, "fact", "unrelated entry", source="user", db_path=self.db_path,
            )

            results = memory.search(project_id, "SQLite", db_path=self.db_path)
            self.assertEqual(len(results), 1)
            self.assertIn("SQLite", results[0].content)

    def test_memory_persists_across_separate_connections(self):
        """The whole point is durability across restarts — simulate that
        by never holding a connection open between calls (which is how
        the module already works: _open() opens and closes every time)."""
        with TemporaryDirectory() as workspace:
            project_id = memory.get_or_create_project(Path(workspace), db_path=self.db_path)
            memory.remember(
                project_id, "fact", "persisted fact", source="user", db_path=self.db_path,
            )

        # Fresh lookup, same db_path, no shared state except the file on disk.
        project_id_again = memory.get_or_create_project(Path(workspace), db_path=self.db_path)
        entries = memory.recall(project_id_again, db_path=self.db_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].content, "persisted fact")

    def test_patching_module_db_path_affects_calls_using_the_default(self):
        """Regression guard: db_path must default to None and resolve
        memory.DB_PATH inside the function body, never bind to DB_PATH's
        value directly in the signature. A direct `db_path: Path =
        DB_PATH` default is captured once at function-definition time,
        so patch.object(memory, "DB_PATH", ...) in a test — or any
        runtime override — would silently do nothing for every call
        that relies on the default, and entries would land in the real
        forge_memory.db instead of the isolated one. This test calls
        every public function with NO db_path argument at all, the way
        forge.py's tool wrappers do, after patching the module
        attribute, exactly as forge.py's real callers would."""
        from unittest.mock import patch

        with TemporaryDirectory() as workspace:
            with patch.object(memory, "DB_PATH", self.db_path):
                project_id = memory.get_or_create_project(Path(workspace))
                entry_id = memory.remember(project_id, "fact", "x", source="user")
                memory.supersede(entry_id, "y", source="user")
                memory.recall(project_id)
                memory.search(project_id, "y")
                memory.history(project_id)

            # Every one of those calls must have landed in self.db_path,
            # not in the module's real default DB_PATH.
            self.assertTrue(self.db_path.exists())

        real_default_db = memory.DB_PATH
        self.assertFalse(
            real_default_db.exists(),
            "a call relying on the db_path default wrote to the real "
            "forge_memory.db instead of the patched path",
        )

    def test_format_context_labels_each_entry_by_kind(self):
        with TemporaryDirectory() as workspace:
            project_id = memory.get_or_create_project(Path(workspace), db_path=self.db_path)
            memory.remember(project_id, "fact", "a fact", source="user", db_path=self.db_path)
            memory.remember(
                project_id, "assumption", "a guess", source="forge", db_path=self.db_path,
            )

            entries = memory.recall(project_id, db_path=self.db_path)
            text = memory.format_context(entries)
            self.assertIn("[FACT] a fact", text)
            self.assertIn("[ASSUMPTION] a guess", text)

    def test_format_context_empty_for_no_entries(self):
        self.assertEqual(memory.format_context([]), "")


class TestRunStorage(unittest.TestCase):
    """Structured, queryable task log — extends the SQLite layer so run
    history (what run_record.py already captures per-task) can be
    queried across a project's runs, not just read one JSON file at a
    time."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test_runs.db"
        self._workspace = TemporaryDirectory()
        self.project_id = memory.get_or_create_project(
            Path(self._workspace.name), db_path=self.db_path
        )

    def tearDown(self):
        self._tmp.cleanup()
        self._workspace.cleanup()

    def _run_dict(self, run_id="run-1", task="do the thing", final_status="complete",
                   final_answer="did the thing", steps_taken=3, stalled=False,
                   error=None, tool_calls=None):
        return {
            "run_id": run_id,
            "task": task,
            "model": "qwen2.5-coder:3b",
            "started_at": 1000.0,
            "ended_at": 1010.0,
            "final_status": final_status,
            "final_answer": final_answer,
            "steps_taken": steps_taken,
            "stalled": stalled,
            "error": error,
            "tool_calls": tool_calls or [
                {"step": 1, "tool": "read_file", "status": "success"},
            ],
        }

    def test_save_and_get_run_round_trips_tool_calls(self):
        memory.save_run(self.project_id, self._run_dict(), db_path=self.db_path)
        run = memory.get_run("run-1", db_path=self.db_path)
        self.assertIsNotNone(run)
        self.assertEqual(run["task"], "do the thing")
        self.assertEqual(len(run["tool_calls"]), 1)
        self.assertEqual(run["tool_calls"][0]["tool"], "read_file")

    def test_get_run_returns_none_for_unknown_id(self):
        self.assertIsNone(memory.get_run("no-such-run", db_path=self.db_path))

    def test_save_run_is_idempotent_on_run_id(self):
        """A run saved once on error and again at natural completion
        must update the same row, not create a duplicate."""
        memory.save_run(
            self.project_id,
            self._run_dict(final_status="error", final_answer=None),
            db_path=self.db_path,
        )
        memory.save_run(
            self.project_id,
            self._run_dict(final_status="complete", final_answer="done"),
            db_path=self.db_path,
        )

        runs = memory.list_runs(self.project_id, db_path=self.db_path)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].final_status, "complete")

    def test_list_runs_orders_most_recent_first(self):
        memory.save_run(
            self.project_id,
            self._run_dict(run_id="run-old", task="first"),
            db_path=self.db_path,
        )
        newer = self._run_dict(run_id="run-new", task="second")
        newer["started_at"] = 2000.0
        memory.save_run(self.project_id, newer, db_path=self.db_path)

        runs = memory.list_runs(self.project_id, db_path=self.db_path)
        self.assertEqual(runs[0].run_id, "run-new")
        self.assertEqual(runs[1].run_id, "run-old")

    def test_list_runs_does_not_include_tool_calls(self):
        """Listing must stay cheap — no tool_calls payload on a summary."""
        memory.save_run(self.project_id, self._run_dict(), db_path=self.db_path)
        runs = memory.list_runs(self.project_id, db_path=self.db_path)
        self.assertFalse(hasattr(runs[0], "tool_calls"))

    def test_list_runs_respects_limit(self):
        for i in range(5):
            run = self._run_dict(run_id=f"run-{i}", task=f"task {i}")
            run["started_at"] = float(i)
            memory.save_run(self.project_id, run, db_path=self.db_path)

        runs = memory.list_runs(self.project_id, limit=2, db_path=self.db_path)
        self.assertEqual(len(runs), 2)

    def test_search_runs_matches_task_text(self):
        memory.save_run(
            self.project_id, self._run_dict(run_id="r1", task="fix the auth bug"),
            db_path=self.db_path,
        )
        memory.save_run(
            self.project_id, self._run_dict(run_id="r2", task="add a new endpoint"),
            db_path=self.db_path,
        )

        results = memory.search_runs(self.project_id, "auth", db_path=self.db_path)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].run_id, "r1")

    def test_search_runs_matches_final_answer_text(self):
        memory.save_run(
            self.project_id,
            self._run_dict(
                run_id="r1", task="unrelated task",
                final_answer="rolled back the broken edit to calc.py",
            ),
            db_path=self.db_path,
        )
        results = memory.search_runs(self.project_id, "calc.py", db_path=self.db_path)
        self.assertEqual(len(results), 1)

    def test_runs_scoped_per_project(self):
        with TemporaryDirectory() as other_workspace:
            other_project_id = memory.get_or_create_project(
                Path(other_workspace), db_path=self.db_path
            )
            memory.save_run(
                self.project_id, self._run_dict(run_id="mine"), db_path=self.db_path,
            )
            memory.save_run(
                other_project_id, self._run_dict(run_id="theirs"), db_path=self.db_path,
            )

            mine = memory.list_runs(self.project_id, db_path=self.db_path)
            self.assertEqual([r.run_id for r in mine], ["mine"])

    def test_stalled_flag_round_trips_as_bool(self):
        memory.save_run(
            self.project_id, self._run_dict(stalled=True), db_path=self.db_path,
        )
        run = memory.get_run("run-1", db_path=self.db_path)
        self.assertIs(run["stalled"], True)


if __name__ == "__main__":
    unittest.main()
