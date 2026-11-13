"""
workflows.py — named, parameterized, storable step sequences for Forge.

Phase 02 ("Everyday agent") includes "reusable workflows": things Forge
does often enough that they shouldn't be re-planned from scratch every
time. A workflow is a named list of prompt templates, each of which
becomes one run_task() call when the workflow is executed.

Design principles:

  Workflows are data, not code
    A workflow is a JSON-serialisable dict stored in SQLite (same DB as
    memory.py, same project scope). No code gets stored; a "step" is a
    prompt template string with {placeholder} variables that are filled
    at run time.

  Separation of definition and execution
    Defining a workflow never calls run_task(). Executing it calls
    run_task() once per step, in order, passing each step's rendered
    prompt. The caller (forge.py) owns run_task; this module owns the
    definition and rendering layer.

  Parameterised via simple string interpolation
    Placeholders use Python's str.format_map(), so {filename}, {branch},
    {ticket} etc. are supported. Unknown placeholders raise a clear
    KeyError rather than silently producing broken prompts.

  Steps can be skipped on prior-step failure
    By default all steps run regardless. Set `stop_on_failure=True` on
    the workflow to halt as soon as a step's run_task() returns a
    non-"complete" status.

  Project-scoped
    Workflows live under a project id (same as memory entries) so a
    workflow named "code-review" for project A is distinct from one for
    project B.

  Idempotent upsert on name + project_id
    Saving a workflow with the same name and project updates it rather
    than duplicating it.

Schema added to memory.py's existing DB:

    workflows (
        id INTEGER PK,
        project_id INTEGER FK → projects(id),
        name TEXT,
        description TEXT,
        steps_json TEXT,       -- JSON list of {name, prompt_template}
        stop_on_failure INTEGER DEFAULT 0,
        created_at REAL,
        updated_at REAL,
        UNIQUE(project_id, name)
    )

    workflow_runs (
        id INTEGER PK,
        workflow_id INTEGER FK → workflows(id),
        project_id INTEGER FK → projects(id),
        params_json TEXT,      -- the {placeholder: value} dict used
        started_at REAL,
        ended_at REAL,
        status TEXT,           -- 'complete' | 'partial' | 'error'
        step_results_json TEXT -- list of {step_name, run_id, status, answer}
    )
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import memory as _mem

# Re-use memory.py's DB_PATH so workflows live alongside memory entries.
DB_PATH = _mem.DB_PATH

WORKFLOW_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    name TEXT NOT NULL,
    description TEXT,
    steps_json TEXT NOT NULL DEFAULT '[]',
    stop_on_failure INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(project_id, name)
);

CREATE INDEX IF NOT EXISTS idx_workflows_project
    ON workflows(project_id);

CREATE TABLE IF NOT EXISTS workflow_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id INTEGER NOT NULL REFERENCES workflows(id),
    project_id INTEGER NOT NULL REFERENCES projects(id),
    params_json TEXT NOT NULL DEFAULT '{}',
    started_at REAL NOT NULL,
    ended_at REAL,
    status TEXT,
    step_results_json TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_workflow_runs_project
    ON workflow_runs(project_id, started_at);
"""


@dataclass
class WorkflowStep:
    name: str
    prompt_template: str

    def render(self, params: dict) -> str:
        """Fill placeholders using params. Raises KeyError for missing keys."""
        return self.prompt_template.format_map(params)

    def to_dict(self) -> dict:
        return {"name": self.name, "prompt_template": self.prompt_template}

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowStep":
        return cls(name=d["name"], prompt_template=d["prompt_template"])


@dataclass
class Workflow:
    id: int
    project_id: int
    name: str
    description: str
    steps: list[WorkflowStep]
    stop_on_failure: bool
    created_at: float
    updated_at: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "stop_on_failure": self.stop_on_failure,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Workflow":
        return cls(
            id=row["id"],
            project_id=row["project_id"],
            name=row["name"],
            description=row["description"] or "",
            steps=[WorkflowStep.from_dict(s) for s in json.loads(row["steps_json"])],
            stop_on_failure=bool(row["stop_on_failure"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass
class StepResult:
    step_name: str
    run_id: Optional[str]
    status: str          # 'complete' | 'error' | 'stalled' | 'skipped'
    answer: Optional[str]
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "step_name": self.step_name,
            "run_id": self.run_id,
            "status": self.status,
            "answer": self.answer,
            "error": self.error,
        }


@dataclass
class WorkflowResult:
    workflow_name: str
    params: dict
    status: str          # 'complete' | 'partial' | 'error'
    step_results: list[StepResult] = field(default_factory=list)
    workflow_run_id: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "workflow_name": self.workflow_name,
            "params": self.params,
            "status": self.status,
            "step_results": [r.to_dict() for r in self.step_results],
            "workflow_run_id": self.workflow_run_id,
        }


# ---------------------------------------------------------------------------
# DB helpers (mirror memory.py's pattern exactly so patch.object works)
# ---------------------------------------------------------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def _open(db_path: Path):
    conn = _connect(db_path)
    try:
        # Ensure memory schema exists first (projects table must exist
        # before our FK to it), then apply our own schema.
        conn.executescript(_mem.SCHEMA)
        conn.executescript(WORKFLOW_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_workflow(
    project_id: int,
    name: str,
    steps: list[WorkflowStep],
    description: str = "",
    stop_on_failure: bool = False,
    db_path: Optional[Path] = None,
) -> int:
    """Create or update (upsert) a workflow. Returns its id."""
    db_path = db_path if db_path is not None else DB_PATH
    now = time.time()
    steps_json = json.dumps([s.to_dict() for s in steps])
    with _open(db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM workflows WHERE project_id = ? AND name = ?",
            (project_id, name),
        ).fetchone()

        if existing:
            conn.execute(
                "UPDATE workflows SET description=?, steps_json=?, "
                "stop_on_failure=?, updated_at=? WHERE id=?",
                (description, steps_json, int(stop_on_failure), now, existing["id"]),
            )
            return existing["id"]

        cursor = conn.execute(
            "INSERT INTO workflows "
            "(project_id, name, description, steps_json, stop_on_failure, "
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (project_id, name, description, steps_json,
             int(stop_on_failure), now, now),
        )
        return cursor.lastrowid


def get_workflow(
    project_id: int,
    name: str,
    db_path: Optional[Path] = None,
) -> Optional[Workflow]:
    """Look up a workflow by project + name. Returns None if not found."""
    db_path = db_path if db_path is not None else DB_PATH
    with _open(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM workflows WHERE project_id = ? AND name = ?",
            (project_id, name),
        ).fetchone()
        return Workflow.from_row(row) if row else None


def list_workflows(
    project_id: int,
    db_path: Optional[Path] = None,
) -> list[Workflow]:
    """All workflows for a project, alphabetically by name."""
    db_path = db_path if db_path is not None else DB_PATH
    with _open(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM workflows WHERE project_id = ? ORDER BY name ASC",
            (project_id,),
        ).fetchall()
        return [Workflow.from_row(row) for row in rows]


def delete_workflow(
    project_id: int,
    name: str,
    db_path: Optional[Path] = None,
) -> bool:
    """Delete a workflow by name. Returns True if it existed."""
    db_path = db_path if db_path is not None else DB_PATH
    with _open(db_path) as conn:
        row = conn.execute(
            "SELECT id FROM workflows WHERE project_id = ? AND name = ?",
            (project_id, name),
        ).fetchone()
        if not row:
            return False
        conn.execute("DELETE FROM workflows WHERE id = ?", (row["id"],))
        return True


def run_workflow(
    project_id: int,
    name: str,
    params: Optional[dict] = None,
    run_task_fn: Optional[Callable] = None,
    db_path: Optional[Path] = None,
) -> WorkflowResult:
    """
    Execute a named workflow by running each step's rendered prompt
    through run_task_fn (defaults to forge.run_task when None — the
    import is deferred so workflows.py doesn't create a circular import
    at module load time).

    Returns a WorkflowResult with per-step outcomes. Never raises: errors
    are captured into StepResult.error so the caller gets a full picture
    of what ran and what failed.
    """
    db_path = db_path if db_path is not None else DB_PATH
    params = params or {}

    if run_task_fn is None:
        import forge as _forge  # deferred to avoid circular import
        run_task_fn = _forge.run_task

    workflow = get_workflow(project_id, name, db_path=db_path)
    if workflow is None:
        return WorkflowResult(
            workflow_name=name,
            params=params,
            status="error",
            step_results=[
                StepResult(
                    step_name="<lookup>",
                    run_id=None,
                    status="error",
                    answer=None,
                    error=f"No workflow named {name!r} found for this project.",
                )
            ],
        )

    # Record the run start
    run_id = _start_workflow_run(project_id, workflow.id, params, db_path)
    step_results: list[StepResult] = []
    overall_status = "complete"
    started_at = time.time()

    for step in workflow.steps:
        # Render the prompt; catch missing placeholders.
        try:
            prompt = step.render(params)
        except KeyError as exc:
            sr = StepResult(
                step_name=step.name,
                run_id=None,
                status="error",
                answer=None,
                error=f"Missing workflow parameter: {exc}",
            )
            step_results.append(sr)
            overall_status = "partial" if step_results else "error"
            if workflow.stop_on_failure:
                break
            continue

        # Execute the step.
        try:
            result = run_task_fn(prompt)
            step_status = result.get("final_status", "error")
            sr = StepResult(
                step_name=step.name,
                run_id=result.get("run_id"),
                status=step_status,
                answer=result.get("answer"),
            )
        except Exception as exc:  # noqa: BLE001
            sr = StepResult(
                step_name=step.name,
                run_id=None,
                status="error",
                answer=None,
                error=str(exc),
            )

        step_results.append(sr)

        if sr.status != "complete":
            overall_status = "partial"
            if workflow.stop_on_failure:
                # Mark remaining steps as skipped.
                for remaining in workflow.steps[len(step_results):]:
                    step_results.append(StepResult(
                        step_name=remaining.name,
                        run_id=None,
                        status="skipped",
                        answer=None,
                    ))
                break

    _finish_workflow_run(run_id, overall_status, step_results, db_path)

    return WorkflowResult(
        workflow_name=name,
        params=params,
        status=overall_status,
        step_results=step_results,
        workflow_run_id=run_id,
    )


def list_workflow_runs(
    project_id: int,
    workflow_name: Optional[str] = None,
    limit: int = 20,
    db_path: Optional[Path] = None,
) -> list[dict]:
    """
    Recent workflow run summaries for a project, most-recent-first.
    Optionally filtered to a specific workflow name.
    """
    db_path = db_path if db_path is not None else DB_PATH
    with _open(db_path) as conn:
        if workflow_name:
            wf_row = conn.execute(
                "SELECT id FROM workflows WHERE project_id = ? AND name = ?",
                (project_id, workflow_name),
            ).fetchone()
            if not wf_row:
                return []
            rows = conn.execute(
                "SELECT wr.*, w.name as workflow_name FROM workflow_runs wr "
                "JOIN workflows w ON w.id = wr.workflow_id "
                "WHERE wr.project_id = ? AND wr.workflow_id = ? "
                "ORDER BY wr.started_at DESC LIMIT ?",
                (project_id, wf_row["id"], limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT wr.*, w.name as workflow_name FROM workflow_runs wr "
                "JOIN workflows w ON w.id = wr.workflow_id "
                "WHERE wr.project_id = ? ORDER BY wr.started_at DESC LIMIT ?",
                (project_id, limit),
            ).fetchall()

        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "workflow_name": row["workflow_name"],
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
                "status": row["status"],
                "params": json.loads(row["params_json"]),
                "step_results": json.loads(row["step_results_json"]),
            })
        return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _start_workflow_run(
    project_id: int,
    workflow_id: int,
    params: dict,
    db_path: Path,
) -> int:
    with _open(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO workflow_runs "
            "(workflow_id, project_id, params_json, started_at, "
            " status, step_results_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (workflow_id, project_id, json.dumps(params), time.time(),
             "running", "[]"),
        )
        return cursor.lastrowid


def _finish_workflow_run(
    run_id: int,
    status: str,
    step_results: list[StepResult],
    db_path: Path,
) -> None:
    with _open(db_path) as conn:
        conn.execute(
            "UPDATE workflow_runs SET ended_at=?, status=?, step_results_json=? "
            "WHERE id=?",
            (time.time(), status,
             json.dumps([r.to_dict() for r in step_results]), run_id),
        )
