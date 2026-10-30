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


if __name__ == "__main__":
    unittest.main()
