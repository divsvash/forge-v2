import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from run_record import new_run, ToolCallRecord


class TestRunRecord(unittest.TestCase):
    def test_new_run_has_unique_id(self):
        a = new_run(task="do x", model="m1")
        b = new_run(task="do x", model="m1")
        self.assertNotEqual(a.run_id, b.run_id)

    def test_tool_calls_recorded_in_order(self):
        run = new_run(task="do x", model="m1")
        run.add_tool_call(ToolCallRecord(
            step=1, tool="list_files", arguments={}, status="success",
            result=["a.py"],
        ))
        run.add_tool_call(ToolCallRecord(
            step=2, tool="read_file", arguments={"filename": "a.py"},
            status="error", error="boom",
        ))
        self.assertEqual(len(run.tool_calls), 2)
        self.assertEqual(run.tool_calls[0]["tool"], "list_files")
        self.assertEqual(run.tool_calls[1]["status"], "error")
        self.assertEqual(run.tool_calls[1]["error"], "boom")

    def test_failed_tool_call_is_retained_not_dropped(self):
        run = new_run(task="do x", model="m1")
        run.add_tool_call(ToolCallRecord(
            step=1, tool="run_command", arguments={"command": "false"},
            status="error", error="exit 1",
        ))
        self.assertEqual(len(run.tool_calls), 1)

    def test_save_writes_durable_json(self):
        run = new_run(task="do x", model="m1")
        run.add_tool_call(ToolCallRecord(
            step=1, tool="list_files", arguments={}, status="success",
            result=["a.py"],
        ))
        run.finish(
            status="complete",
            completion={"success": True, "missing": []},
            answer="done",
            steps=1,
        )
        with TemporaryDirectory() as tmp:
            path = run.save(Path(tmp))
            self.assertTrue(path.exists())
            data = json.loads(path.read_text())
            self.assertEqual(data["run_id"], run.run_id)
            self.assertEqual(data["final_status"], "complete")
            self.assertEqual(len(data["tool_calls"]), 1)

    def test_finish_sets_final_status(self):
        run = new_run(task="do x", model="m1")
        run.finish(
            status="incomplete",
            completion={"success": False, "missing": ["x"]},
            answer="partial",
            steps=3,
        )
        self.assertEqual(run.final_status, "incomplete")
        self.assertEqual(run.steps_taken, 3)
        self.assertIsNotNone(run.ended_at)


if __name__ == "__main__":
    unittest.main()
