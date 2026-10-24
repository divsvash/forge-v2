import platform
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import sandbox


class TestRestrictedLocalBackend(unittest.TestCase):
    def test_runs_command_in_workspace(self):
        backend = sandbox.RestrictedLocalBackend()
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "marker.txt").write_text("hello")
            result = backend.run("ls", workspace, timeout=10)
            self.assertEqual(result.exit_code, 0)
            self.assertIn("marker.txt", result.stdout)
            self.assertFalse(result.timed_out)

    def test_capabilities_are_honest_about_no_isolation(self):
        backend = sandbox.RestrictedLocalBackend()
        caps = backend.capabilities()
        self.assertFalse(caps.filesystem_isolated)
        self.assertFalse(caps.network_blocked)

    def test_nonzero_exit_code_is_reported_not_swallowed(self):
        backend = sandbox.RestrictedLocalBackend()
        with TemporaryDirectory() as tmp:
            result = backend.run("exit 7", Path(tmp), timeout=10)
            self.assertEqual(result.exit_code, 7)

    def test_timeout_is_reported(self):
        backend = sandbox.RestrictedLocalBackend()
        with TemporaryDirectory() as tmp:
            result = backend.run("sleep 5", Path(tmp), timeout=1)
            self.assertTrue(result.timed_out)
            self.assertEqual(result.exit_code, -1)

    def test_environment_is_scrubbed(self):
        backend = sandbox.RestrictedLocalBackend()
        with TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"FORGE_TEST_SECRET": "leak-me"}):
                result = backend.run(
                    "echo $FORGE_TEST_SECRET", Path(tmp), timeout=10
                )
            self.assertNotIn("leak-me", result.stdout)

    def test_output_is_truncated(self):
        backend = sandbox.RestrictedLocalBackend()
        with TemporaryDirectory() as tmp:
            # Ask for far more output than the cap and confirm it's capped.
            original_limit = sandbox.MAX_OUTPUT_CHARS
            with patch.object(sandbox, "MAX_OUTPUT_CHARS", 100):
                result = backend.run(
                    "python3 -c \"print('x' * 1000)\"", Path(tmp), timeout=10
                )
            self.assertLess(len(result.stdout), 1000)
            self.assertIn("truncated", result.stdout)


class TestSelectBackend(unittest.TestCase):
    def test_falls_back_to_local_when_docker_unreachable(self):
        with patch.object(sandbox, "_docker_daemon_reachable", return_value=False):
            backend = sandbox.select_backend()
            self.assertIsInstance(backend, sandbox.RestrictedLocalBackend)

    def test_uses_docker_when_reachable(self):
        with patch.object(sandbox, "_docker_daemon_reachable", return_value=True):
            backend = sandbox.select_backend()
            self.assertIsInstance(backend, sandbox.DockerBackend)

    def test_explicit_local_never_probes_docker(self):
        with patch.object(sandbox, "_docker_daemon_reachable") as probe:
            backend = sandbox.select_backend(prefer="local")
            self.assertIsInstance(backend, sandbox.RestrictedLocalBackend)
            probe.assert_not_called()

    def test_explicit_docker_raises_when_unavailable(self):
        with patch.object(sandbox, "_docker_daemon_reachable", return_value=False):
            with self.assertRaises(RuntimeError):
                sandbox.select_backend(prefer="docker")


if __name__ == "__main__":
    unittest.main()
