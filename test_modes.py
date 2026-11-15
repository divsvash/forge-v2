"""
test_modes.py — unit tests for modes.py and mode-aware forge.py behaviour.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import memory
import modes
import workflows
import scheduler


class TestModeRegistry(unittest.TestCase):
    def test_all_six_modes_defined(self):
        expected = {"researcher", "planner", "coding_agent",
                    "mini_divs", "intern", "growth"}
        self.assertEqual(set(modes.MODES.keys()), expected)

    def test_every_mode_has_required_fields(self):
        for mid, mode in modes.MODES.items():
            self.assertEqual(mode.id, mid)
            self.assertIsInstance(mode.display_name, str)
            self.assertIsInstance(mode.description, str)
            self.assertIsInstance(mode.system_prompt_prefix, str)
            self.assertGreater(len(mode.system_prompt_prefix), 10)
            self.assertGreater(mode.default_max_steps, 0)
            self.assertGreater(mode.default_max_stalls, 0)

    def test_researcher_and_growth_have_restricted_tools(self):
        self.assertIsNotNone(modes.MODES["researcher"].allowed_tools)
        self.assertIsNotNone(modes.MODES["growth"].allowed_tools)

    def test_coding_agent_planner_mini_divs_intern_have_all_tools(self):
        for mid in ("coding_agent", "planner", "mini_divs", "intern"):
            self.assertIsNone(modes.MODES[mid].allowed_tools,
                              f"{mid} should have full tool access")

    def test_researcher_tools_exclude_write_operations(self):
        researcher_tools = modes.MODES["researcher"].allowed_tools
        for write_tool in ("write_file", "run_command", "apply_change_with_test"):
            self.assertNotIn(write_tool, researcher_tools)

    def test_list_modes_returns_all_six(self):
        result = modes.list_modes()
        self.assertEqual(len(result), 6)
        ids = {m["id"] for m in result}
        self.assertEqual(ids, set(modes.MODES.keys()))

    def test_list_modes_shows_restricted_for_researcher(self):
        result = modes.list_modes()
        researcher = next(m for m in result if m["id"] == "researcher")
        self.assertIn("restricted", researcher["tool_access"])

    def test_list_modes_shows_all_tools_for_coding_agent(self):
        result = modes.list_modes()
        coding = next(m for m in result if m["id"] == "coding_agent")
        self.assertEqual(coding["tool_access"], "all tools")


class TestActiveModeStorage(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test_modes.db"
        self._ws = TemporaryDirectory()
        with patch.object(memory, "DB_PATH", self.db_path):
            self.project_id = memory.get_or_create_project(Path(self._ws.name))

    def tearDown(self):
        self._tmp.cleanup()
        self._ws.cleanup()

    def test_default_mode_is_coding_agent(self):
        mode = modes.get_active_mode(self.project_id, db_path=self.db_path)
        self.assertEqual(mode.id, "coding_agent")

    def test_set_mode_persists(self):
        modes.set_active_mode(self.project_id, "researcher", db_path=self.db_path)
        mode = modes.get_active_mode(self.project_id, db_path=self.db_path)
        self.assertEqual(mode.id, "researcher")

    def test_set_mode_can_be_changed(self):
        modes.set_active_mode(self.project_id, "planner", db_path=self.db_path)
        modes.set_active_mode(self.project_id, "intern", db_path=self.db_path)
        mode = modes.get_active_mode(self.project_id, db_path=self.db_path)
        self.assertEqual(mode.id, "intern")

    def test_set_mode_rejects_unknown_id(self):
        with self.assertRaises(ValueError):
            modes.set_active_mode(self.project_id, "wizard", db_path=self.db_path)

    def test_mode_stored_in_decision_history(self):
        modes.set_active_mode(self.project_id, "mini_divs", db_path=self.db_path)
        decisions = memory.recall(
            self.project_id, kind="decision", db_path=self.db_path
        )
        contents = [d.content for d in decisions]
        self.assertTrue(any("mini_divs" in c for c in contents))

    def test_old_mode_superseded_not_duplicated(self):
        modes.set_active_mode(self.project_id, "planner", db_path=self.db_path)
        modes.set_active_mode(self.project_id, "growth", db_path=self.db_path)
        # Only one active mode entry should exist
        active = [
            e for e in memory.recall(self.project_id, db_path=self.db_path)
            if e.source == "forge-mode"
        ]
        self.assertEqual(len(active), 1)
        self.assertIn("growth", active[0].content)

    def test_modes_scoped_per_project(self):
        with TemporaryDirectory() as ws2:
            with patch.object(memory, "DB_PATH", self.db_path):
                pid2 = memory.get_or_create_project(Path(ws2))
        modes.set_active_mode(self.project_id, "intern", db_path=self.db_path)
        # project 2 should still be on default
        mode2 = modes.get_active_mode(pid2, db_path=self.db_path)
        self.assertEqual(mode2.id, "coding_agent")


class TestModeAwareRunTask(unittest.TestCase):
    """
    forge.run_task() mode-injection tests — no real LLM calls.
    We verify the system message and step/stall limits are shaped by the
    active mode without running the full agent loop.
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "forge_modes.db"
        self._ws = TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()
        self._ws.cleanup()

    def _run_with_mode(self, mode_id):
        """
        Set a mode, call run_task with a scripted one-shot answer,
        and return the system message that was built + the run result.
        """
        import forge

        captured_system_msg = {}

        def capturing_ask_model():
            # Capture first system message then return a done answer
            if forge.messages:
                captured_system_msg["content"] = forge.messages[0].get("content", "")
            return '{"answer": "done"}'

        with patch.object(forge, "WORKSPACE", Path(self._ws.name)), \
             patch.object(memory, "DB_PATH", self.db_path), \
             patch.object(forge, "ask_model", capturing_ask_model):

            # Set mode via memory directly — db_path is already patched above
            pid = memory.get_or_create_project(Path(self._ws.name))
            modes.set_active_mode(pid, mode_id, db_path=self.db_path)

            result = forge.run_task("list all files")

        return captured_system_msg.get("content", ""), result

    def test_coding_agent_mode_prefix_in_system_message(self):
        sys_content, _ = self._run_with_mode("coding_agent")
        self.assertIn("CODING AGENT", sys_content)

    def test_researcher_mode_prefix_in_system_message(self):
        sys_content, _ = self._run_with_mode("researcher")
        self.assertIn("RESEARCHER", sys_content)

    def test_mini_divs_mode_prefix_in_system_message(self):
        sys_content, _ = self._run_with_mode("mini_divs")
        self.assertIn("MINI DIVS", sys_content)

    def test_mode_prefix_prepended_before_shared_prompt(self):
        sys_content, _ = self._run_with_mode("planner")
        planner_idx = sys_content.find("PLANNER")
        forge_idx = sys_content.find("list_files")  # from shared tool docs
        self.assertGreater(planner_idx, -1)
        self.assertGreater(forge_idx, -1)
        self.assertLess(planner_idx, forge_idx,
                        "mode prefix should appear before shared prompt content")


class TestModeForgeTools(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "forge_mode_tools.db"
        self._ws = TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()
        self._ws.cleanup()

    def test_list_modes_tool(self):
        import forge
        with patch.object(forge, "WORKSPACE", Path(self._ws.name)), \
             patch.object(memory, "DB_PATH", self.db_path):
            result = forge.list_modes_tool()
        self.assertEqual(len(result), 6)
        self.assertTrue(all("id" in m for m in result))

    def test_set_mode_tool_returns_mode_info(self):
        import forge
        with patch.object(forge, "WORKSPACE", Path(self._ws.name)), \
             patch.object(memory, "DB_PATH", self.db_path):
            result = forge.set_mode_tool("researcher")
        self.assertEqual(result["mode"], "researcher")
        self.assertIn("display_name", result)
        self.assertIn("restricted", result["tool_access"])

    def test_set_mode_tool_rejects_unknown(self):
        import forge
        with patch.object(forge, "WORKSPACE", Path(self._ws.name)), \
             patch.object(memory, "DB_PATH", self.db_path):
            with self.assertRaises(ValueError):
                forge.set_mode_tool("wizard")


if __name__ == "__main__":
    unittest.main()
