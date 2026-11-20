"""
test_github_tools.py — unit tests for github_tools.py and forge.py git wiring.

Tests use real git repos created in temp directories so we can verify actual
subprocess behaviour without mocking git itself.  The only things mocked are:
- github_tools.REPO_PATH  (so we can point at a temp repo)
- _gh_available()          (so create_pull_request tests run without gh CLI)
"""

import unittest
import subprocess
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import github_tools as _gh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_repo(path: Path, bare: bool = False) -> Path:
    """Create a minimal git repo with one commit."""
    path.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "t@t.com",
           "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "t@t.com"}
    subprocess.run(["git", "init"], cwd=str(path), check=True,
                   capture_output=True, env=env)
    subprocess.run(["git", "config", "user.email", "t@t.com"],
                   cwd=str(path), check=True, capture_output=True, env=env)
    subprocess.run(["git", "config", "user.name", "Test"],
                   cwd=str(path), check=True, capture_output=True, env=env)
    # Initial commit
    readme = path / "README.md"
    readme.write_text("# test\n")
    subprocess.run(["git", "add", "."], cwd=str(path), check=True,
                   capture_output=True, env=env)
    subprocess.run(["git", "commit", "-m", "init"],
                   cwd=str(path), check=True, capture_output=True, env=env)
    return path


# ---------------------------------------------------------------------------
# TestGitStatus
# ---------------------------------------------------------------------------

class TestGitStatus(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.repo = _init_repo(Path(self._tmp.name) / "repo")

    def tearDown(self):
        self._tmp.cleanup()

    def test_clean_repo_has_no_changes(self):
        result = _gh.git_status(repo_path=self.repo)
        self.assertTrue(result["clean"])
        self.assertEqual(result["staged"], [])
        self.assertEqual(result["unstaged"], [])
        self.assertEqual(result["untracked"], [])

    def test_new_untracked_file_appears(self):
        (self.repo / "new.py").write_text("x = 1\n")
        result = _gh.git_status(repo_path=self.repo)
        self.assertFalse(result["clean"])
        self.assertIn("new.py", result["untracked"])

    def test_staged_file_appears(self):
        (self.repo / "staged.py").write_text("y = 2\n")
        subprocess.run(["git", "add", "staged.py"], cwd=str(self.repo),
                       check=True, capture_output=True)
        result = _gh.git_status(repo_path=self.repo)
        self.assertIn("staged.py", result["staged"])

    def test_branch_name_is_returned(self):
        result = _gh.git_status(repo_path=self.repo)
        self.assertIsInstance(result["branch"], str)
        self.assertGreater(len(result["branch"]), 0)

    def test_resolve_error_when_repo_path_none(self):
        with patch.object(_gh, "REPO_PATH", None):
            with self.assertRaises(ValueError):
                _gh.git_status(repo_path=None)


# ---------------------------------------------------------------------------
# TestGitDiff
# ---------------------------------------------------------------------------

class TestGitDiff(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.repo = _init_repo(Path(self._tmp.name) / "repo")

    def tearDown(self):
        self._tmp.cleanup()

    def test_no_diff_on_clean_repo(self):
        result = _gh.git_diff(repo_path=self.repo)
        self.assertEqual(result["diff"], "")
        self.assertEqual(result["files_changed"], [])

    def test_unstaged_diff_shows_modification(self):
        (self.repo / "README.md").write_text("# changed\n")
        result = _gh.git_diff(repo_path=self.repo)
        self.assertIn("README.md", result["diff"])

    def test_staged_diff_shows_staged_file(self):
        (self.repo / "new.py").write_text("z = 3\n")
        subprocess.run(["git", "add", "new.py"], cwd=str(self.repo),
                       check=True, capture_output=True)
        result = _gh.git_diff(repo_path=self.repo, staged=True)
        self.assertIn("new.py", result["diff"])
        self.assertTrue(result["staged"])


# ---------------------------------------------------------------------------
# TestGitLog
# ---------------------------------------------------------------------------

class TestGitLog(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.repo = _init_repo(Path(self._tmp.name) / "repo")

    def tearDown(self):
        self._tmp.cleanup()

    def test_log_returns_list_of_commits(self):
        result = _gh.git_log(repo_path=self.repo, n=5)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_commit_has_expected_keys(self):
        commits = _gh.git_log(repo_path=self.repo)
        for c in commits:
            self.assertIn("hash", c)
            self.assertIn("author", c)
            self.assertIn("date", c)
            self.assertIn("subject", c)

    def test_initial_commit_message_in_log(self):
        commits = _gh.git_log(repo_path=self.repo)
        subjects = [c["subject"] for c in commits]
        self.assertIn("init", subjects)


# ---------------------------------------------------------------------------
# TestGitBranch
# ---------------------------------------------------------------------------

class TestGitBranch(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.repo = _init_repo(Path(self._tmp.name) / "repo")

    def tearDown(self):
        self._tmp.cleanup()

    def test_branch_returns_current(self):
        result = _gh.git_branch(repo_path=self.repo)
        self.assertIsNotNone(result["current"])
        self.assertIn(result["current"], result["branches"])

    def test_new_branch_appears_in_list(self):
        subprocess.run(["git", "branch", "feature/x"], cwd=str(self.repo),
                       check=True, capture_output=True)
        result = _gh.git_branch(repo_path=self.repo)
        self.assertIn("feature/x", result["branches"])


# ---------------------------------------------------------------------------
# TestGitCheckout
# ---------------------------------------------------------------------------

class TestGitCheckout(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.repo = _init_repo(Path(self._tmp.name) / "repo")

    def tearDown(self):
        self._tmp.cleanup()

    def test_checkout_creates_new_branch(self):
        result = _gh.git_checkout("feature/new", create=True,
                                   repo_path=self.repo)
        self.assertEqual(result["branch"], "feature/new")
        self.assertTrue(result["created"])
        # Verify we're on it
        status = _gh.git_status(repo_path=self.repo)
        self.assertEqual(status["branch"], "feature/new")

    def test_checkout_existing_branch(self):
        subprocess.run(["git", "branch", "other"], cwd=str(self.repo),
                       check=True, capture_output=True)
        result = _gh.git_checkout("other", create=False, repo_path=self.repo)
        self.assertEqual(result["branch"], "other")
        self.assertFalse(result["created"])

    def test_checkout_nonexistent_branch_returns_error(self):
        result = _gh.git_checkout("does-not-exist", repo_path=self.repo)
        self.assertIn("error", result)


# ---------------------------------------------------------------------------
# TestGitAddAndCommit
# ---------------------------------------------------------------------------

class TestGitAddAndCommit(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.repo = _init_repo(Path(self._tmp.name) / "repo")
        # git needs an identity
        env = {**os.environ}
        subprocess.run(["git", "config", "user.email", "t@t.com"],
                       cwd=str(self.repo), capture_output=True, env=env)
        subprocess.run(["git", "config", "user.name", "Test"],
                       cwd=str(self.repo), capture_output=True, env=env)

    def tearDown(self):
        self._tmp.cleanup()

    def test_add_all_stages_new_file(self):
        (self.repo / "foo.py").write_text("x=1\n")
        result = _gh.git_add(repo_path=self.repo)
        self.assertIn("foo.py", result["staged"])

    def test_add_specific_file(self):
        (self.repo / "a.py").write_text("a=1\n")
        (self.repo / "b.py").write_text("b=2\n")
        result = _gh.git_add(paths=["a.py"], repo_path=self.repo)
        self.assertIn("a.py", result["staged"])
        # b.py should still be untracked, not staged
        self.assertNotIn("b.py", result["staged"])

    def test_commit_nothing_staged_returns_false(self):
        result = _gh.git_commit("empty commit", repo_path=self.repo)
        self.assertFalse(result["committed"])
        self.assertIn("reason", result)

    def test_commit_after_add_succeeds(self):
        (self.repo / "new.py").write_text("n=1\n")
        _gh.git_add(repo_path=self.repo)
        result = _gh.git_commit("add new.py", repo_path=self.repo)
        self.assertTrue(result["committed"])
        self.assertIsInstance(result["hash"], str)
        self.assertGreater(len(result["hash"]), 0)

    def test_commit_hash_appears_in_log(self):
        (self.repo / "z.py").write_text("z=9\n")
        _gh.git_add(repo_path=self.repo)
        commit_result = _gh.git_commit("add z", repo_path=self.repo)
        log = _gh.git_log(repo_path=self.repo, n=1)
        self.assertIn(commit_result["hash"], log[0]["hash"])


# ---------------------------------------------------------------------------
# TestGitClone
# ---------------------------------------------------------------------------

class TestGitClone(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.source = _init_repo(Path(self._tmp.name) / "source")

    def tearDown(self):
        self._tmp.cleanup()

    def test_clone_local_repo(self):
        dest = Path(self._tmp.name) / "cloned"
        with patch.object(_gh, "REPO_PATH", self.source):
            result = _gh.git_clone(url=str(self.source), dest=str(dest))
        self.assertTrue(result["cloned"])
        self.assertTrue(dest.exists())
        self.assertTrue((dest / "README.md").exists())

    def test_clone_to_existing_dest_returns_false(self):
        dest = Path(self._tmp.name) / "already_exists"
        dest.mkdir()
        with patch.object(_gh, "REPO_PATH", self.source):
            result = _gh.git_clone(url=str(self.source), dest=str(dest))
        self.assertFalse(result["cloned"])
        self.assertIn("reason", result)


# ---------------------------------------------------------------------------
# TestCreatePullRequest
# ---------------------------------------------------------------------------

class TestCreatePullRequest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.repo = _init_repo(Path(self._tmp.name) / "repo")

    def tearDown(self):
        self._tmp.cleanup()

    def test_fallback_when_gh_not_available(self):
        with patch.object(_gh, "_gh_available", return_value=False):
            result = _gh.create_pull_request(
                title="Test PR",
                body="body",
                base="main",
                repo_path=self.repo,
            )
        self.assertFalse(result["created"])
        self.assertIn("url", result)
        self.assertIn("reason", result)

    def test_fallback_url_contains_base_branch(self):
        with patch.object(_gh, "_gh_available", return_value=False):
            result = _gh.create_pull_request(
                title="Test PR",
                base="develop",
                repo_path=self.repo,
            )
        self.assertIn("develop", result["url"])


# ---------------------------------------------------------------------------
# TestGitToolsWiredInForge
# ---------------------------------------------------------------------------

class TestGitToolsWiredInForge(unittest.TestCase):
    """
    Verify that git tool wrapper functions and ToolSpec entries exist in
    forge.py and are callable.  Uses a real temp repo but doesn't run the
    full LLM loop.
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.repo = _init_repo(Path(self._tmp.name) / "repo")

    def tearDown(self):
        self._tmp.cleanup()

    def test_all_git_tools_in_forge_tools_dict(self):
        import forge
        expected = {
            "git_status", "git_diff", "git_log", "git_branch",
            "git_checkout", "git_add", "git_commit", "git_push",
            "git_clone", "create_pull_request",
        }
        self.assertTrue(expected.issubset(set(forge.TOOLS.keys())))

    def test_git_status_tool_callable(self):
        import forge
        with patch.object(forge, "WORKSPACE", self.repo), \
             patch.object(_gh, "REPO_PATH", self.repo):
            result = forge.git_status_tool()
        self.assertIn("branch", result)
        self.assertIn("clean", result)

    def test_git_log_tool_callable(self):
        import forge
        with patch.object(forge, "WORKSPACE", self.repo), \
             patch.object(_gh, "REPO_PATH", self.repo):
            result = forge.git_log_tool(n=3)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_github_operator_mode_in_forge_tools_set(self):
        """
        github_operator mode's allowed_tools set should reference only tools
        that actually exist in forge.TOOLS.
        """
        import forge
        import modes
        github_mode = modes.MODES["github_operator"]
        for tool_name in github_mode.allowed_tools:
            self.assertIn(
                tool_name, forge.TOOLS,
                f"github_operator mode references '{tool_name}' which is not in forge.TOOLS",
            )

    def test_git_tools_not_accessible_in_researcher_mode(self):
        import modes
        researcher = modes.MODES["researcher"]
        for git_tool in ("git_commit", "git_push", "create_pull_request"):
            self.assertNotIn(git_tool, researcher.allowed_tools)

    def test_system_prompt_contains_git_tool_docs(self):
        import forge
        content = forge._SYSTEM_MESSAGE["content"]
        for tool_name in ("git_status", "git_diff", "git_commit",
                          "create_pull_request"):
            self.assertIn(tool_name, content,
                          f"System prompt missing docs for {tool_name}")


if __name__ == "__main__":
    unittest.main()
