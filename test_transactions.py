import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import sandbox
import transactions


class TestApplyChangeWithTest(unittest.TestCase):
    def setUp(self):
        self.backend = sandbox.RestrictedLocalBackend()

    def test_keeps_change_when_test_passes(self):
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            result = transactions.apply_change_with_test(
                workspace, "a.py", "x = 1\n", "exit 0", self.backend,
            )
            self.assertTrue(result["kept"])
            self.assertFalse(result["rolled_back"])
            self.assertEqual((workspace / "a.py").read_text(), "x = 1\n")

    def test_rolls_back_existing_file_when_test_fails(self):
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "a.py").write_text("x = 0\n")

            result = transactions.apply_change_with_test(
                workspace, "a.py", "x = 1\n", "exit 1", self.backend,
            )

            self.assertFalse(result["kept"])
            self.assertTrue(result["rolled_back"])
            self.assertTrue(result["previous_existed"])
            # File must be restored to exactly what it was before.
            self.assertEqual((workspace / "a.py").read_text(), "x = 0\n")

    def test_deletes_new_file_when_test_fails(self):
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            # a.py does not exist yet.
            result = transactions.apply_change_with_test(
                workspace, "a.py", "x = 1\n", "exit 1", self.backend,
            )

            self.assertFalse(result["kept"])
            self.assertFalse(result["previous_existed"])
            self.assertFalse((workspace / "a.py").exists())

    def test_rejects_path_outside_workspace(self):
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            with self.assertRaises(ValueError):
                transactions.apply_change_with_test(
                    workspace, "../escape.py", "x = 1\n", "exit 0", self.backend,
                )

    def test_reports_actual_exit_code_and_output(self):
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            result = transactions.apply_change_with_test(
                workspace, "a.py", "x = 1\n",
                "python3 -c \"import sys; print('boom'); sys.exit(3)\"",
                self.backend,
            )
            self.assertFalse(result["kept"])
            self.assertEqual(result["test_exit_code"], 3)
            self.assertIn("boom", result["test_stdout"])

    def test_real_failing_test_suite_triggers_rollback(self):
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

            # A "fix" that actually breaks the function.
            result = transactions.apply_change_with_test(
                workspace, "calc.py", "def add(x, y):\n    return x - y\n",
                "python3 -m unittest test_calc", self.backend,
            )

            self.assertFalse(result["kept"])
            self.assertEqual(
                (workspace / "calc.py").read_text(),
                "def add(x, y):\n    return x + y\n",
            )


if __name__ == "__main__":
    unittest.main()
