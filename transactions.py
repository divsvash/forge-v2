"""
transactions.py — transactional code editing: edit -> test -> keep/rollback.

This closes the last open item from Phase 01's known limitations:
"transactional code editing (edit -> test -> keep/rollback)".

Before this, a code change went through Forge as two or three
independent tool calls (write_file, then maybe run_command). Nothing
guaranteed the model would actually run the verifying command, and if
the change broke something, nothing undid it — a bad edit just sat in
the workspace as the new, broken state, with "rollback" being a thing
a human would have to notice and do by hand.

`apply_change_with_test` makes edit+verify+rollback a single atomic
runtime operation instead of a multi-step pattern the model has to be
trusted to follow correctly:

  1. Snapshot the file's current content (or note that it didn't exist).
  2. Write the new content.
  3. Run the verifying command through the real sandbox backend.
  4. If it exits 0: keep the change.
     If it doesn't: restore the exact previous content (or delete the
     file, if it didn't exist before this call) and report the failure.

The caller (forge.py) supplies the already-selected sandbox backend, so
this module stays backend-agnostic and easy to test without touching
sandbox selection at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def apply_change_with_test(
    workspace: Path,
    filename: str,
    content: str,
    test_command: str,
    backend,
    timeout: int = 30,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    path = (workspace / filename).resolve()

    if not path.is_relative_to(workspace):
        raise ValueError("Access outside workspace is not allowed")

    previous_existed = path.is_file()
    previous_content = path.read_text(encoding="utf-8") if previous_existed else None

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    test_result = backend.run(test_command, workspace, timeout=timeout)
    kept = test_result.exit_code == 0

    if not kept:
        # Roll back to exactly what was there before — restore the prior
        # content, or remove the file entirely if this call created it.
        if previous_existed:
            path.write_text(previous_content, encoding="utf-8")
        else:
            path.unlink(missing_ok=True)

    return {
        "filename": filename,
        "kept": kept,
        "rolled_back": not kept,
        "previous_existed": previous_existed,
        "test_command": test_command,
        "test_exit_code": test_result.exit_code,
        "test_stdout": test_result.stdout,
        "test_stderr": test_result.stderr,
        "test_timed_out": test_result.timed_out,
        "backend": test_result.backend,
    }
