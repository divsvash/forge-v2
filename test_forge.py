import unittest
from unittest.mock import patch
import forge

from forge import check_file_coverage, check_success, build_continuation_message,  build_evidence_summary



class TestForgeEvidence(unittest.TestCase):

    def setUp(self):
        self.evidence = {
            "search results": [
                {
                    "result": [
                        "hello.py:1:print(1)",
                        "hello2.py:1:print(2)"
                    ]
                }
            ],
            "relevant file contents": [
                {
                    "arguments": {"filename": "hello.py"},
                    "result": "print(1)"
                }
            ],
            "command execution result": [],
            "test execution result": [],
            "workspace file listing": []
        }

        self.success_definition = {
            "goal": "find and inspect hello files",
            "required_evidence": [
                "search results",
                "relevant file contents"
            ]
        }

    def test_detects_missing_inspection(self):
        coverage = check_file_coverage(self.evidence)

        self.assertEqual(
            coverage["missing_inspections"],
            ["hello2.py"]
        )

    def test_rejects_incomplete_coverage(self):
        completion = check_success(
            self.success_definition,
            self.evidence
        )

        self.assertFalse(completion["success"])
        self.assertIn(
            "uninspected files: hello2.py",
            completion["missing"]
        )

    def test_accepts_complete_coverage(self):
        self.evidence["relevant file contents"].append({
            "arguments": {"filename": "hello2.py"},
            "result": "print(2)"
        })

        completion = check_success(
            self.success_definition,
            self.evidence
        )

        self.assertTrue(completion["success"])
        self.assertEqual(completion["missing"], [])

    def test_execution_not_required_for_inspection(self):
        self.evidence["relevant file contents"].append({
            "arguments": {"filename": "hello2.py"},
            "result": "print(2)"
        })

        completion = check_success(
            self.success_definition,
            self.evidence
        )

        self.assertTrue(completion["success"])
        self.assertEqual(
            self.evidence["command execution result"],
            []
        )

    def test_continuation_names_missing_files(self):
        completion = {
            "success": False,
            "missing": [
                "uninspected files: hello.py, hello2.py"
            ]
            }

        message = build_continuation_message(completion)

        self.assertIn("- hello.py", message)
        self.assertIn("- hello2.py", message)
        self.assertIn("read_file", message)

    def test_continuation_handles_generic_evidence(self):
         completion = {
            "success": False,
             "missing": ["test execution result"]
            }

         message = build_continuation_message(completion)

         self.assertIn("test execution result", message)
         self.assertIn("Continue working", message)

    def test_evidence_summary_preserves_execution_output(self):
        self.evidence["command execution result"].append({
             "tool": "run_command",
             "arguments": {"command": "python broken3.py"},
             "result": {
                "exit_code": 0,
                  "stdout": "",
                  "stderr": ""
             }
        })

        summary = build_evidence_summary(self.evidence)

        self.assertIn("EXECUTED COMMAND: python broken3.py", summary)
        self.assertIn("Exit code: 0", summary)
        self.assertIn("Stdout: ''", summary)
        self.assertIn(
            "does not by itself prove that tests passed",
            summary
         )

    def test_grounded_answer_uses_temporary_context(self):
        original_messages = list(forge.messages)

        completion = {
            "success": True,
            "missing": []
        }

        try:
            with patch.object(
            forge,
                "ask_model",
                return_value="Verified explanation"
            ) as mock_model:

              answer = forge.generate_grounded_answer(
                    "Explain hello.py",
                    self.evidence,
                    completion
                    )

            self.assertEqual(
                    answer,
                    "Verified explanation"
                )

            mock_model.assert_called_once()

            self.assertEqual(
                    forge.messages,
                    original_messages
                    )

        finally:
            forge.messages[:] = original_messages

class TestForgeIntegration(unittest.TestCase):
    def test_productive_continuation_exceeds_three_turns(self):
        from unittest.mock import patch
        from tempfile import TemporaryDirectory
        from pathlib import Path
        import contextlib
        import io

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            for filename in ("one.py", "two.py", "three.py"):
                (workspace / filename).write_text("print('hello')\n")
            responses = iter([
                '{"tool":"search_files","arguments":{"query":"hello"}}',
                "I found three files.",
                '{"tool":"read_file","arguments":{"filename":"one.py"}}',
                "Read one.",
                '{"tool":"read_file","arguments":{"filename":"two.py"}}',
                "Read two.",
                '{"tool":"read_file","arguments":{"filename":"three.py"}}',
                "Read three.",
                "Verified: three files contain hello.",
            ])
            original_messages = list(forge.messages)
            try:
                with patch.object(forge, "WORKSPACE", workspace), patch.object(
                    forge, "ask_model", side_effect=lambda: next(responses)
                ), patch("run_record.RUNS_DIR", workspace / "_runs"), \
                        contextlib.redirect_stdout(io.StringIO()):
                    result = forge.run_task(
                        "find where hello is used, inspect the relevant files", max_steps=20
                    )
                self.assertTrue(result["completion"]["success"])
                self.assertEqual(result["steps"], 4)
                self.assertIn("Verified: three files", result["answer"])
            finally:
                forge.messages[:] = original_messages

    def test_test_evidence_requires_test_command(self):
        evidence = {k: [] for k in (
            "search results", "relevant file contents", "test execution result",
            "command execution result", "workspace file listing"
        )}
        forge.record_evidence(evidence, "run_command", {"command": "python hello.py"},
                              {"exit_code": 0, "stdout": "all tests passed", "stderr": ""})
        self.assertFalse(evidence["test execution result"])
        forge.record_evidence(evidence, "run_command", {"command": "python -m unittest"},
                              {"exit_code": 0, "stdout": "OK", "stderr": ""})
        self.assertEqual(len(evidence["test execution result"]), 1)

    def test_unknown_request_is_not_auto_verified(self):
        evidence = {k: [] for k in (
            "search results", "relevant file contents", "test execution result",
            "command execution result", "workspace file listing"
        )}
        result = forge.check_success(forge.define_success("build an application"), evidence)
        self.assertFalse(result["success"])


class TestFoundationsSlice(unittest.TestCase):
    """Covers the foundations slice: failure-as-evidence, no message leak
    across tasks, and a durable run record for every run_task() call."""

    def test_failed_command_is_recorded_not_dropped(self):
        evidence = {}
        forge.record_evidence(
            evidence, "run_command", {"command": "python broken.py"},
            {"exit_code": 1, "stdout": "", "stderr": "Traceback..."},
        )
        self.assertEqual(len(evidence["failed command execution"]), 1)
        self.assertNotIn("command execution result", evidence)

    def test_failed_test_command_lands_in_failed_test_bucket(self):
        evidence = {}
        forge.record_evidence(
            evidence, "run_command", {"command": "pytest"},
            {"exit_code": 1, "stdout": "1 failed", "stderr": ""},
        )
        self.assertEqual(len(evidence["failed test execution"]), 1)
        self.assertNotIn("test execution result", evidence)

    def test_successful_command_still_goes_to_original_buckets(self):
        evidence = {}
        forge.record_evidence(
            evidence, "run_command", {"command": "pytest"},
            {"exit_code": 0, "stdout": "ok", "stderr": ""},
        )
        self.assertEqual(len(evidence["command execution result"]), 1)
        self.assertEqual(len(evidence["test execution result"]), 1)
        self.assertNotIn("failed command execution", evidence)

    def test_messages_do_not_leak_between_tasks(self):
        import contextlib
        import io
        from unittest.mock import patch
        from pathlib import Path
        from tempfile import TemporaryDirectory

        original_messages = list(forge.messages)
        try:
            with TemporaryDirectory() as tmp:
                with patch.object(forge, "WORKSPACE", Path(tmp)), patch.object(
                    forge, "ask_model", return_value="Noted."
                ), patch("run_record.RUNS_DIR", Path(tmp) / "runs"), \
                        contextlib.redirect_stdout(io.StringIO()):
                    forge.run_task(
                        "remember the secret word banana", max_steps=5, max_stalls=1
                    )
                    forge.run_task(
                        "what did I just ask about?", max_steps=5, max_stalls=1
                    )

            task_contents = [
                m["content"] for m in forge.messages if m["role"] == "user"
            ]
            self.assertTrue(
                any("what did I just ask" in c for c in task_contents)
            )
            self.assertFalse(
                any("secret word banana" in c for c in task_contents)
            )
        finally:
            forge.messages[:] = original_messages

    def test_run_task_writes_a_durable_run_record(self):
        import contextlib
        import io
        import json
        from unittest.mock import patch
        from pathlib import Path
        from tempfile import TemporaryDirectory

        original_messages = list(forge.messages)
        try:
            with TemporaryDirectory() as tmp:
                runs_dir = Path(tmp) / "runs"
                with patch.object(forge, "WORKSPACE", Path(tmp)), patch.object(
                    forge, "ask_model", return_value="Final answer."
                ), patch("run_record.RUNS_DIR", runs_dir), \
                        contextlib.redirect_stdout(io.StringIO()):
                    result = forge.run_task("a trivial task")

                self.assertIn("run_id", result)
                record_path = Path(result["run_record_path"])
                self.assertTrue(record_path.exists())

                data = json.loads(record_path.read_text())
                self.assertEqual(data["task"], "a trivial task")
                self.assertEqual(data["run_id"], result["run_id"])
                self.assertIn(data["final_status"], ("complete", "incomplete"))
        finally:
            forge.messages[:] = original_messages

    def test_run_record_keeps_failed_tool_calls(self):
        import contextlib
        import io
        from unittest.mock import patch
        from pathlib import Path
        from tempfile import TemporaryDirectory

        responses = iter([
            '{"tool":"read_file","arguments":{"filename":"missing.py"}}',
            "I could not find that file.",
        ])
        original_messages = list(forge.messages)
        try:
            with TemporaryDirectory() as tmp:
                with patch.object(forge, "WORKSPACE", Path(tmp)), patch.object(
                    forge, "ask_model", side_effect=lambda: next(responses)
                ), patch("run_record.RUNS_DIR", Path(tmp) / "runs"), \
                        contextlib.redirect_stdout(io.StringIO()):
                    result = forge.run_task("read missing.py", max_steps=5)

                import json
                data = json.loads(Path(result["run_record_path"]).read_text())
                statuses = [call["status"] for call in data["tool_calls"]]
                self.assertIn("error", statuses)
        finally:
            forge.messages[:] = original_messages

    def test_redundant_tool_call_is_not_re_executed(self):
        """An exact repeat of a prior call (same tool, same arguments) in
        the same task must be served from cache, not run again — and must
        not be double-counted as evidence (which would falsely look like
        progress to the stall detector)."""
        import contextlib
        import io
        import json
        from unittest.mock import patch
        from pathlib import Path
        from tempfile import TemporaryDirectory

        responses = iter([
            '{"tool":"read_file","arguments":{"filename":"one.py"}}',
            '{"tool":"read_file","arguments":{"filename":"one.py"}}',
            "one.py contains a print statement.",
        ])
        original_messages = list(forge.messages)
        try:
            with TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                (workspace / "one.py").write_text("print('hi')\n")

                original_read_file = forge.TOOLS["read_file"].func
                with patch.object(
                    forge.TOOLS["read_file"], "func", wraps=original_read_file
                ) as wrapped_read_file, \
                        patch.object(forge, "WORKSPACE", workspace), \
                        patch.object(
                            forge, "ask_model", side_effect=lambda: next(responses)
                        ), \
                        patch("run_record.RUNS_DIR", workspace / "_runs"), \
                        contextlib.redirect_stdout(io.StringIO()):
                    result = forge.run_task(
                        "inspect the contents of one.py", max_steps=10
                    )

                    # The underlying tool function only actually ran once,
                    # even though the model asked for it twice.
                    self.assertEqual(wrapped_read_file.call_count, 1)

                # Evidence wasn't double-counted on the cached repeat.
                self.assertEqual(
                    len(result["evidence"]["relevant file contents"]), 1
                )

                data = json.loads(Path(result["run_record_path"]).read_text())
                statuses = [call["status"] for call in data["tool_calls"]]
                self.assertIn("cached", statuses)
        finally:
            forge.messages[:] = original_messages

    def test_apply_change_with_test_keeps_passing_change_end_to_end(self):
        import contextlib
        import io
        from unittest.mock import patch
        from pathlib import Path
        from tempfile import TemporaryDirectory

        responses = iter([
            '{"tool":"apply_change_with_test","arguments":'
            '{"filename":"calc.py","content":"def add(x, y):\\n    return x + y\\n",'
            '"test_command":"python3 -m unittest test_calc"}}',
            "The change was applied and the test passed.",
        ])
        original_messages = list(forge.messages)
        try:
            with TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                (workspace / "test_calc.py").write_text(
                    "import unittest\nfrom calc import add\n\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_add(self):\n"
                    "        self.assertEqual(add(1, 2), 3)\n"
                )
                with patch.object(forge, "WORKSPACE", workspace), patch.object(
                    forge, "ask_model", side_effect=lambda: next(responses)
                ), patch("run_record.RUNS_DIR", workspace / "_runs"), \
                        contextlib.redirect_stdout(io.StringIO()):
                    result = forge.run_task(
                        "add a working add() function to calc.py and verify it",
                        max_steps=5,
                    )

                self.assertEqual(
                    (workspace / "calc.py").read_text(),
                    "def add(x, y):\n    return x + y\n",
                )
                self.assertEqual(len(result["evidence"]["transactional edits"]), 1)
                edit = result["evidence"]["transactional edits"][0]["result"]
                self.assertTrue(edit["kept"])
        finally:
            forge.messages[:] = original_messages

    def test_apply_change_with_test_rolls_back_failing_change_end_to_end(self):
        import contextlib
        import io
        from unittest.mock import patch
        from pathlib import Path
        from tempfile import TemporaryDirectory

        responses = iter([
            '{"tool":"apply_change_with_test","arguments":'
            '{"filename":"calc.py","content":"def add(x, y):\\n    return x - y\\n",'
            '"test_command":"python3 -m unittest test_calc"}}',
            "The change broke the test and was rolled back.",
        ])
        original_messages = list(forge.messages)
        try:
            with TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                (workspace / "calc.py").write_text(
                    "def add(x, y):\n    return x + y\n"
                )
                (workspace / "test_calc.py").write_text(
                    "import unittest\nfrom calc import add\n\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_add(self):\n"
                    "        self.assertEqual(add(1, 2), 3)\n"
                )
                with patch.object(forge, "WORKSPACE", workspace), patch.object(
                    forge, "ask_model", side_effect=lambda: next(responses)
                ), patch("run_record.RUNS_DIR", workspace / "_runs"), \
                        contextlib.redirect_stdout(io.StringIO()):
                    result = forge.run_task(
                        "fix add() in calc.py and verify it", max_steps=5,
                    )

                # The broken "fix" must never survive — original content
                # must be exactly restored.
                self.assertEqual(
                    (workspace / "calc.py").read_text(),
                    "def add(x, y):\n    return x + y\n",
                )
                edit = result["evidence"]["transactional edits"][0]["result"]
                self.assertFalse(edit["kept"])
                self.assertTrue(edit["rolled_back"])
                self.assertEqual(len(result["evidence"]["failed test execution"]), 1)

                # The grounded summary must say it was rolled back, not kept.
                summary = forge.build_evidence_summary(result["evidence"])
                self.assertIn("ROLLED BACK", summary)
        finally:
            forge.messages[:] = original_messages


if __name__ == "__main__":
    unittest.main()
