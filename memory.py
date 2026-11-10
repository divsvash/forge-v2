"""
memory.py — durable, project-scoped memory backed by SQLite.

Phase 02 ("Forge's Brain" / "Everyday agent") starts here. Before this,
Forge's only state was the in-memory `messages` list, reset at the start
of every run_task() call (correctly — that fixed a context-leak bug) and
the `evidence` dict, discarded entirely when the process exits. Nothing
survived a restart, and nothing distinguished "the user told me this" or
"Forge concluded this from evidence" from "the user is still deciding" —
all of that just lived as prose in a conversation that got thrown away.

This module gives Forge a durable record, per project, of:

  - facts        things established as true (from the user or from
                  verified evidence — never from an unverified guess)
  - assumptions   things Forge is working from but hasn't confirmed
  - decisions     choices made about how to proceed, with a reason
  - task_summary   what happened in a completed run_task() call

Every entry has provenance: who/what said it (`source`) and, when it
came from a run, which run_id. Memory is never silently overwritten —
when new evidence contradicts an old entry, the old one is marked
superseded (kept, not deleted) and a new entry points back at it, so the
project has an actual decision history, not just a current snapshot.

A project is identified by the resolved path of its workspace, so the
same workspace always maps to the same project across restarts.

It also holds the structured, queryable task log: every run_task() call
in full (its ordered tool calls, completion check, final status), keyed
by project. run_record.py already wrote this to a per-run JSON file for
human/debug reading (RUNS_DIR) — that stays as-is. What's new here is
making run history actually queryable: "how many of my last 20 runs on
this project failed", "show me what happened in the run where I fixed
the auth bug", without opening files one at a time.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "forge_memory.db"

VALID_KINDS = {"fact", "assumption", "decision", "task_summary"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_path TEXT UNIQUE NOT NULL,
    name TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    run_id TEXT,
    created_at REAL NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    superseded_by INTEGER REFERENCES memory_entries(id)
);

CREATE INDEX IF NOT EXISTS idx_memory_project_kind
    ON memory_entries(project_id, kind, active);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    task TEXT NOT NULL,
    model TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    final_status TEXT,
    final_answer TEXT,
    steps_taken INTEGER,
    stalled INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    tool_calls_json TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_runs_project_started
    ON runs(project_id, started_at);
"""


@dataclass
class MemoryEntry:
    id: int
    project_id: int
    kind: str
    content: str
    source: str
    run_id: Optional[str]
    created_at: float
    active: bool
    superseded_by: Optional[int]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "MemoryEntry":
        return cls(
            id=row["id"],
            project_id=row["project_id"],
            kind=row["kind"],
            content=row["content"],
            source=row["source"],
            run_id=row["run_id"],
            created_at=row["created_at"],
            active=bool(row["active"]),
            superseded_by=row["superseded_by"],
        )


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def _open(db_path: Path):
    conn = _connect(db_path)
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_or_create_project(
    workspace_path: Path, name: Optional[str] = None, db_path: Optional[Path] = None
) -> int:
    # db_path deliberately defaults to None, not DB_PATH: a default bound
    # to DB_PATH's value at function-definition time would freeze that
    # value forever, so `patch.object(memory, "DB_PATH", ...)` in tests
    # (and any future runtime override) would silently have no effect on
    # every call site that relies on the default. Resolving module.DB_PATH
    # here, at call time, keeps it a live reference.
    db_path = db_path if db_path is not None else DB_PATH
    resolved = str(Path(workspace_path).resolve())
    with _open(db_path) as conn:
        row = conn.execute(
            "SELECT id FROM projects WHERE workspace_path = ?", (resolved,)
        ).fetchone()
        if row:
            return row["id"]

        cursor = conn.execute(
            "INSERT INTO projects (workspace_path, name, created_at) "
            "VALUES (?, ?, ?)",
            (resolved, name or Path(resolved).name, time.time()),
        )
        return cursor.lastrowid


def remember(
    project_id: int,
    kind: str,
    content: str,
    source: str,
    run_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    db_path = db_path if db_path is not None else DB_PATH
    if kind not in VALID_KINDS:
        raise ValueError(
            f"Unknown memory kind: {kind!r}. Must be one of {sorted(VALID_KINDS)}."
        )

    with _open(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO memory_entries "
            "(project_id, kind, content, source, run_id, created_at, active) "
            "VALUES (?, ?, ?, ?, ?, ?, 1)",
            (project_id, kind, content, source, run_id, time.time()),
        )
        return cursor.lastrowid


def supersede(
    entry_id: int,
    new_content: str,
    source: str,
    run_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Replace an entry because new evidence contradicts it. The old
    entry is kept (active=0, superseded_by=<new id>) rather than deleted
    or overwritten in place — the point of a decision history is being
    able to see that the understanding changed, not just what it is now."""
    db_path = db_path if db_path is not None else DB_PATH
    with _open(db_path) as conn:
        old = conn.execute(
            "SELECT * FROM memory_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        if old is None:
            raise ValueError(f"No memory entry with id {entry_id}")

        cursor = conn.execute(
            "INSERT INTO memory_entries "
            "(project_id, kind, content, source, run_id, created_at, active) "
            "VALUES (?, ?, ?, ?, ?, ?, 1)",
            (old["project_id"], old["kind"], new_content, source, run_id, time.time()),
        )
        new_id = cursor.lastrowid

        conn.execute(
            "UPDATE memory_entries SET active = 0, superseded_by = ? WHERE id = ?",
            (new_id, entry_id),
        )
        return new_id


def recall(
    project_id: int,
    kind: Optional[str] = None,
    limit: Optional[int] = None,
    include_superseded: bool = False,
    db_path: Optional[Path] = None,
) -> list[MemoryEntry]:
    db_path = db_path if db_path is not None else DB_PATH
    query = "SELECT * FROM memory_entries WHERE project_id = ?"
    params: list = [project_id]

    if not include_superseded:
        query += " AND active = 1"

    if kind is not None:
        if kind not in VALID_KINDS:
            raise ValueError(f"Unknown memory kind: {kind!r}")
        query += " AND kind = ?"
        params.append(kind)

    query += " ORDER BY created_at DESC"

    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    with _open(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [MemoryEntry.from_row(row) for row in rows]


def search(
    project_id: int, text: str, db_path: Optional[Path] = None
) -> list[MemoryEntry]:
    """Simple substring search over active entries' content — enough for
    now; a future slice can add FTS if recall quality demands it."""
    db_path = db_path if db_path is not None else DB_PATH
    with _open(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM memory_entries "
            "WHERE project_id = ? AND active = 1 AND content LIKE ? "
            "ORDER BY created_at DESC",
            (project_id, f"%{text}%"),
        ).fetchall()
        return [MemoryEntry.from_row(row) for row in rows]


def history(project_id: int, db_path: Optional[Path] = None) -> list[MemoryEntry]:
    """Full history for a project, including superseded entries, in the
    order they were created — the actual decision/fact timeline."""
    db_path = db_path if db_path is not None else DB_PATH
    with _open(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM memory_entries WHERE project_id = ? "
            "ORDER BY created_at ASC",
            (project_id,),
        ).fetchall()
        return [MemoryEntry.from_row(row) for row in rows]


@dataclass
class RunSummary:
    """A run without its tool_calls payload — cheap to list many of."""
    run_id: str
    project_id: int
    task: str
    model: Optional[str]
    started_at: float
    ended_at: Optional[float]
    final_status: Optional[str]
    final_answer: Optional[str]
    steps_taken: Optional[int]
    stalled: bool
    error: Optional[str]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "RunSummary":
        return cls(
            run_id=row["run_id"],
            project_id=row["project_id"],
            task=row["task"],
            model=row["model"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            final_status=row["final_status"],
            final_answer=row["final_answer"],
            steps_taken=row["steps_taken"],
            stalled=bool(row["stalled"]),
            error=row["error"],
        )


def save_run(
    project_id: int, run_dict: dict, db_path: Optional[Path] = None
) -> None:
    """Persist a run_record.RunRecord.to_dict() as a queryable row.
    Idempotent on run_id (REPLACE) so saving the same run twice — e.g.
    once on error, once at natural completion — just updates it rather
    than duplicating or failing."""
    db_path = db_path if db_path is not None else DB_PATH
    with _open(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs "
            "(run_id, project_id, task, model, started_at, ended_at, "
            " final_status, final_answer, steps_taken, stalled, error, "
            " tool_calls_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_dict["run_id"],
                project_id,
                run_dict.get("task"),
                run_dict.get("model"),
                run_dict.get("started_at"),
                run_dict.get("ended_at"),
                run_dict.get("final_status"),
                run_dict.get("final_answer"),
                run_dict.get("steps_taken"),
                int(bool(run_dict.get("stalled"))),
                run_dict.get("error"),
                json.dumps(run_dict.get("tool_calls", []), default=str),
            ),
        )


def get_run(run_id: str, db_path: Optional[Path] = None) -> Optional[dict]:
    """Full run, including its tool call history, as a plain dict."""
    db_path = db_path if db_path is not None else DB_PATH
    with _open(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None

        data = dict(row)
        data["stalled"] = bool(data["stalled"])
        data["tool_calls"] = json.loads(data.pop("tool_calls_json"))
        return data


def list_runs(
    project_id: int, limit: Optional[int] = 20, db_path: Optional[Path] = None
) -> list[RunSummary]:
    """Most-recent-first run summaries for a project, without the
    (potentially large) tool_calls payload — cheap enough to list."""
    db_path = db_path if db_path is not None else DB_PATH
    query = (
        "SELECT run_id, project_id, task, model, started_at, ended_at, "
        "final_status, final_answer, steps_taken, stalled, error "
        "FROM runs WHERE project_id = ? ORDER BY started_at DESC"
    )
    params: list = [project_id]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    with _open(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [RunSummary.from_row(row) for row in rows]


def search_runs(
    project_id: int, text: str, db_path: Optional[Path] = None
) -> list[RunSummary]:
    """Runs for this project whose task or final answer mentions `text`."""
    db_path = db_path if db_path is not None else DB_PATH
    with _open(db_path) as conn:
        rows = conn.execute(
            "SELECT run_id, project_id, task, model, started_at, ended_at, "
            "final_status, final_answer, steps_taken, stalled, error "
            "FROM runs WHERE project_id = ? "
            "AND (task LIKE ? OR final_answer LIKE ?) "
            "ORDER BY started_at DESC",
            (project_id, f"%{text}%", f"%{text}%"),
        ).fetchall()
        return [RunSummary.from_row(row) for row in rows]


def format_context(entries: list[MemoryEntry]) -> str:
    """Render recalled entries as a block Forge can prepend to a new
    task's context, with each entry's kind kept visible so facts are
    never presented with the same confidence as assumptions."""
    if not entries:
        return ""

    lines = ["PROJECT MEMORY (from prior sessions):"]
    for entry in entries:
        lines.append(f"- [{entry.kind.upper()}] {entry.content} (source: {entry.source})")
    return "\n".join(lines)
