"""
github_tools.py — GitHub / Git operator tools for Forge.

Phase 04 integrations: gives Forge the ability to work with Git repos
directly — clone, inspect status/diff, branch, commit, push, open PRs.
These are the tools the Coding Agent and Mini Divs modes use when Forge
is asked to "open a PR", "commit my changes", or "review what changed".

Design decisions
----------------

subprocess, not GitPython
  No new dependencies. Every operation is a plain `git` subprocess call.
  Results are structured dicts, not raw CLI output, so callers can reason
  about them without parsing.

Explicit working-directory scoping
  Every function takes an optional `repo_path` that defaults to
  `REPO_PATH` (the module-level default, which forge.py sets from
  WORKSPACE or a configured clone directory). Tests override it via
  patch.object — same pattern as DB_PATH in memory.py.

Safe-by-default
  - `git_push` requires explicit `force=True` to force-push; default is
    always a normal push.
  - `git_commit` will not commit if the working tree is clean (returns a
    clear message rather than an error).
  - `create_pull_request` uses the GitHub CLI (`gh`) when available,
    falls back to returning a URL template the user can open manually.
  - No credential storage: tokens come from the environment
    (GITHUB_TOKEN / GH_TOKEN), which is where `gh` already looks.

Operations
----------
  git_status(repo_path)         → {branch, clean, staged, unstaged,
                                    untracked}
  git_diff(repo_path, staged,   → {diff: str, files_changed: [str]}
           file_path)
  git_log(repo_path, n, branch) → [{hash, author, date, subject}]
  git_branch(repo_path)         → {current, branches: [str]}
  git_checkout(repo_path, name, → {branch, created}
               create)
  git_add(repo_path, paths)     → {staged: [str]}
  git_commit(repo_path, message)→ {committed, hash, message}
  git_push(repo_path, remote,   → {pushed, branch, remote, output}
           branch, force)
  git_clone(url, dest,          → {cloned, path, url}
            repo_path)
  create_pull_request(          → {created, url, title, number}
    repo_path, title, body,
    base, head, draft)
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

# Default repo path — forge.py sets this to WORKSPACE (or a sibling
# directory for clones). Tests override via patch.object.
REPO_PATH: Optional[Path] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run(args: list[str], cwd: Path, env: Optional[dict] = None,
         timeout: int = 30) -> tuple[int, str, str]:
    """Run a git command; return (returncode, stdout, stderr)."""
    result = subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env or {**os.environ},
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _resolve(repo_path: Optional[Path]) -> Path:
    """Resolve repo_path, falling back to module REPO_PATH."""
    p = repo_path if repo_path is not None else REPO_PATH
    if p is None:
        raise ValueError(
            "No repo_path provided and REPO_PATH is not set. "
            "Pass repo_path explicitly or set github_tools.REPO_PATH."
        )
    return Path(p)


def _gh_available() -> bool:
    try:
        rc, _, _ = _run(["gh", "--version"], cwd=Path("/tmp"))
        return rc == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def git_status(repo_path: Optional[Path] = None) -> dict:
    """
    Return the working-tree status of the repo.
    staged: files with changes staged for commit.
    unstaged: modified tracked files not yet staged.
    untracked: new files not tracked by git.
    """
    repo = _resolve(repo_path)
    rc, out, err = _run(["git", "status", "--porcelain=v1", "-b"], cwd=repo)
    if rc != 0:
        return {"error": err or "git status failed", "repo": str(repo)}

    lines = out.splitlines()
    branch = "unknown"
    staged, unstaged, untracked = [], [], []

    for line in lines:
        if line.startswith("## "):
            # ## main...origin/main [ahead 1]
            branch_part = line[3:].split("...")[0].split(" ")[0]
            branch = branch_part
            continue
        if len(line) < 2:
            continue
        xy, path = line[:2], line[3:]
        # Rename: "old -> new"
        if " -> " in path:
            path = path.split(" -> ")[-1]
        x, y = xy[0], xy[1]
        if x != " " and x != "?":
            staged.append(path)
        if y not in (" ", "?"):
            unstaged.append(path)
        if xy == "??":
            untracked.append(path)

    return {
        "branch": branch,
        "clean": not staged and not unstaged and not untracked,
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked,
        "repo": str(repo),
    }


def git_diff(repo_path: Optional[Path] = None,
             staged: bool = False,
             file_path: Optional[str] = None) -> dict:
    """
    Return the diff for the working tree (or staged changes).
    Pass staged=True for `git diff --cached`.
    Pass file_path to limit to one file.
    """
    repo = _resolve(repo_path)
    args = ["git", "diff"]
    if staged:
        args.append("--cached")
    args += ["--stat", "--patch"]
    if file_path:
        args += ["--", file_path]

    rc, out, err = _run(args, cwd=repo)
    if rc != 0:
        return {"error": err or "git diff failed", "diff": ""}

    # Extract changed file names from the stat block.
    files_changed = []
    for line in out.splitlines():
        if "|" in line:
            files_changed.append(line.split("|")[0].strip())

    return {
        "diff": out,
        "files_changed": files_changed,
        "staged": staged,
        "repo": str(repo),
    }


def git_log(repo_path: Optional[Path] = None,
            n: int = 10,
            branch: Optional[str] = None) -> list[dict]:
    """
    Return the last n commits as structured dicts.
    """
    repo = _resolve(repo_path)
    fmt = "%H%x1f%an%x1f%ad%x1f%s"   # hash, author, date, subject
    args = ["git", "log", f"-{n}", f"--format={fmt}", "--date=short"]
    if branch:
        args.append(branch)

    rc, out, err = _run(args, cwd=repo)
    if rc != 0:
        return [{"error": err or "git log failed"}]

    commits = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\x1f")
        if len(parts) == 4:
            commits.append({
                "hash": parts[0][:12],
                "author": parts[1],
                "date": parts[2],
                "subject": parts[3],
            })
    return commits


def git_branch(repo_path: Optional[Path] = None) -> dict:
    """List branches; mark the current one."""
    repo = _resolve(repo_path)
    rc, out, err = _run(["git", "branch", "--list"], cwd=repo)
    if rc != 0:
        return {"error": err or "git branch failed"}

    branches = []
    current = None
    for line in out.splitlines():
        name = line.strip().lstrip("* ").strip()
        if line.startswith("*"):
            current = name
        branches.append(name)

    return {"current": current, "branches": branches, "repo": str(repo)}


def git_checkout(name: str,
                 create: bool = False,
                 repo_path: Optional[Path] = None) -> dict:
    """
    Check out a branch. Pass create=True to create it (-b).
    """
    repo = _resolve(repo_path)
    args = ["git", "checkout"]
    if create:
        args.append("-b")
    args.append(name)

    rc, out, err = _run(args, cwd=repo)
    if rc != 0:
        return {"error": err or f"git checkout {name!r} failed", "branch": name}

    return {"branch": name, "created": create, "repo": str(repo)}


def git_add(paths: Optional[list[str]] = None,
            repo_path: Optional[Path] = None) -> dict:
    """
    Stage files. Pass paths=None or paths=['.'] to stage everything.
    """
    repo = _resolve(repo_path)
    targets = paths if paths else ["."]
    args = ["git", "add", "--"] + targets

    rc, out, err = _run(args, cwd=repo)
    if rc != 0:
        return {"error": err or "git add failed", "staged": []}

    # Confirm what ended up staged
    status = git_status(repo_path=repo)
    return {"staged": status.get("staged", []), "repo": str(repo)}


def git_commit(message: str,
               repo_path: Optional[Path] = None) -> dict:
    """
    Commit staged changes. Returns early (committed=False) if nothing
    is staged, to avoid an empty-commit error.
    """
    repo = _resolve(repo_path)
    status = git_status(repo_path=repo)
    if not status.get("staged"):
        return {
            "committed": False,
            "message": message,
            "reason": "nothing staged — use git_add first",
            "repo": str(repo),
        }

    rc, out, err = _run(["git", "commit", "-m", message], cwd=repo)
    if rc != 0:
        return {"committed": False, "error": err or "git commit failed"}

    # Extract the new commit hash from the output line like
    # "[main abc1234] commit message"
    commit_hash = ""
    for line in out.splitlines():
        if line.startswith("["):
            parts = line.split()
            if len(parts) >= 2:
                commit_hash = parts[1].rstrip("]")
            break

    return {
        "committed": True,
        "hash": commit_hash,
        "message": message,
        "repo": str(repo),
    }


def git_push(remote: str = "origin",
             branch: Optional[str] = None,
             force: bool = False,
             repo_path: Optional[Path] = None) -> dict:
    """
    Push to remote. force=True adds --force-with-lease (safer than
    --force: fails if the remote has commits the local doesn't have).
    """
    repo = _resolve(repo_path)
    args = ["git", "push", remote]
    if branch:
        args.append(branch)
    if force:
        args.append("--force-with-lease")

    rc, out, err = _run(args, cwd=repo, timeout=60)
    output = (out + "\n" + err).strip()
    if rc != 0:
        return {"pushed": False, "error": output, "remote": remote}

    return {
        "pushed": True,
        "remote": remote,
        "branch": branch,
        "output": output,
        "repo": str(repo),
    }


def git_clone(url: str,
              dest: Optional[str] = None,
              repo_path: Optional[Path] = None) -> dict:
    """
    Clone a repository. dest is a directory name relative to the
    parent of repo_path (or an absolute path). Returns the path cloned to.
    """
    # Determine where to clone.
    base = _resolve(repo_path).parent if repo_path is not None or REPO_PATH is not None else Path.cwd()
    if dest:
        clone_to = Path(dest) if Path(dest).is_absolute() else base / dest
    else:
        # Derive from URL: "https://github.com/user/repo.git" → "repo"
        name = url.rstrip("/").split("/")[-1]
        if name.endswith(".git"):
            name = name[:-4]
        clone_to = base / name

    if clone_to.exists():
        return {
            "cloned": False,
            "path": str(clone_to),
            "reason": "destination already exists",
            "url": url,
        }

    rc, out, err = _run(
        ["git", "clone", url, str(clone_to)],
        cwd=base,
        timeout=120,
    )
    if rc != 0:
        return {"cloned": False, "error": err or "git clone failed", "url": url}

    return {"cloned": True, "path": str(clone_to), "url": url}


def create_pull_request(title: str,
                        body: str = "",
                        base: str = "main",
                        head: Optional[str] = None,
                        draft: bool = False,
                        repo_path: Optional[Path] = None) -> dict:
    """
    Create a GitHub PR using the `gh` CLI when available.
    Falls back to a URL template the user can open manually.

    Requires GITHUB_TOKEN or GH_TOKEN in the environment, or the user
    to have run `gh auth login` already.
    """
    repo = _resolve(repo_path)

    if not _gh_available():
        # Return a manual fallback — the user can click the URL
        status = git_status(repo_path=repo)
        current_branch = head or status.get("branch", "HEAD")
        url_template = (
            f"https://github.com/<owner>/<repo>/compare/{base}...{current_branch}"
            "?expand=1"
        )
        return {
            "created": False,
            "reason": "gh CLI not available — open the URL to create the PR manually",
            "url": url_template,
            "title": title,
        }

    args = [
        "gh", "pr", "create",
        "--title", title,
        "--body", body,
        "--base", base,
    ]
    if head:
        args += ["--head", head]
    if draft:
        args.append("--draft")

    rc, out, err = _run(args, cwd=repo, timeout=60)
    if rc != 0:
        return {"created": False, "error": err or out or "gh pr create failed"}

    # `gh pr create` outputs the PR URL on success
    pr_url = out.strip()
    pr_number = ""
    if pr_url:
        pr_number = pr_url.rstrip("/").split("/")[-1]

    return {
        "created": True,
        "url": pr_url,
        "number": pr_number,
        "title": title,
        "base": base,
        "draft": draft,
        "repo": str(repo),
    }
