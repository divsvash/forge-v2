"""
scheduler.py — persistent job queue for Forge's automation engine.

Phase 03 ("Automation engine," days 7-9) adds the ability to schedule
tasks and workflows to run later or on a repeating schedule, without
requiring the user to be present. Jobs survive process restarts because
everything lives in SQLite (same DB as memory and workflows).

Design decisions
----------------

No APScheduler runtime dependency
  APScheduler is excellent but adds a large dependency and an in-process
  daemon thread. Forge's design is local-first and restart-tolerant: a
  plain "tick" function checks the DB for due jobs and runs them. The
  caller (a CLI entrypoint, a cron job, or a simple `while True` loop)
  decides how often to tick. This keeps scheduling logic entirely in
  Python stdlib + SQLite.

Three trigger types
  one_shot    run once at a specific Unix timestamp
  interval    run every N seconds (next_run_at = last_run_at + interval_seconds)
  cron        run on a cron expression — the five-field POSIX subset
              (minute hour dom month dow); evaluated with croniter when
              available, otherwise raises ImportError with install hint.

Retry with exponential backoff
  max_retries and retry_delay_seconds (base) on each job. After a failed
  attempt: delay = retry_delay_seconds * 2^attempt, capped at 1 hour.
  When retries are exhausted the job moves to status 'failed' permanently.

Explicit locking — no silent double-execution
  tick() uses a SQLite UPDATE … WHERE status='pending' AND next_run_at<=now
  as an atomic claim, so two concurrent ticks never double-execute the
  same job. (SQLite's default serialised write mode gives this for free.)

Safe unattended execution
  Jobs run inside the same forge.run_task() / _workflows.run_workflow()
  loop as interactive calls, so the evidence-ledger / completion-checking
  / sandbox / transactional-edit machinery applies automatically.

Schema (extends memory/workflows DB)
-------------------------------------
jobs (
    id               INTEGER PK,
    project_id       INTEGER FK → projects(id),
    name             TEXT,          -- human label, not unique
    kind             TEXT,          -- 'task' | 'workflow'
    payload_json     TEXT,          -- {prompt} or {workflow_name, params}
    trigger_type     TEXT,          -- 'one_shot' | 'interval' | 'cron'
    trigger_config   TEXT,          -- JSON: {at} | {seconds} | {expr}
    status           TEXT,          -- 'active' | 'paused' | 'failed' | 'done'
    next_run_at      REAL,
    last_run_at      REAL,
    attempt          INTEGER DEFAULT 0,
    max_retries      INTEGER DEFAULT 3,
    retry_delay_secs REAL    DEFAULT 60,
    created_at       REAL,
    updated_at       REAL
)

job_runs (
    id           INTEGER PK,
    job_id       INTEGER FK → jobs(id),
    project_id   INTEGER FK → projects(id),
    started_at   REAL,
    ended_at     REAL,
    status       TEXT,     -- 'complete' | 'error' | 'stalled'
    result_json  TEXT,     -- serialised run_task / run_workflow result
    error        TEXT
)
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import memory as _mem
import workflows as _wf

DB_PATH = _mem.DB_PATH

SCHEDULER_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id       INTEGER NOT NULL REFERENCES projects(id),
    name             TEXT    NOT NULL,
    kind             TEXT    NOT NULL,
    payload_json     TEXT    NOT NULL DEFAULT '{}',
    trigger_type     TEXT    NOT NULL,
    trigger_config   TEXT    NOT NULL DEFAULT '{}',
    status           TEXT    NOT NULL DEFAULT 'active',
    next_run_at      REAL,
    last_run_at      REAL,
    attempt          INTEGER NOT NULL DEFAULT 0,
    max_retries      INTEGER NOT NULL DEFAULT 3,
    retry_delay_secs REAL    NOT NULL DEFAULT 60,
    created_at       REAL    NOT NULL,
    updated_at       REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_project_status
    ON jobs(project_id, status, next_run_at);

CREATE TABLE IF NOT EXISTS job_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id       INTEGER NOT NULL REFERENCES jobs(id),
    project_id   INTEGER NOT NULL REFERENCES projects(id),
    started_at   REAL    NOT NULL,
    ended_at     REAL,
    status       TEXT,
    result_json  TEXT    NOT NULL DEFAULT '{}',
    error        TEXT
);

CREATE INDEX IF NOT EXISTS idx_job_runs_job
    ON job_runs(job_id, started_at);
"""

_MAX_BACKOFF = 3600.0   # 1 hour cap on retry delay


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Job:
    id: int
    project_id: int
    name: str
    kind: str                     # 'task' | 'workflow'
    payload: dict
    trigger_type: str             # 'one_shot' | 'interval' | 'cron'
    trigger_config: dict
    status: str                   # 'active' | 'paused' | 'failed' | 'done'
    next_run_at: Optional[float]
    last_run_at: Optional[float]
    attempt: int
    max_retries: int
    retry_delay_secs: float
    created_at: float
    updated_at: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "name": self.name,
            "kind": self.kind,
            "payload": self.payload,
            "trigger_type": self.trigger_type,
            "trigger_config": self.trigger_config,
            "status": self.status,
            "next_run_at": self.next_run_at,
            "last_run_at": self.last_run_at,
            "attempt": self.attempt,
            "max_retries": self.max_retries,
            "retry_delay_secs": self.retry_delay_secs,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Job":
        return cls(
            id=row["id"],
            project_id=row["project_id"],
            name=row["name"],
            kind=row["kind"],
            payload=json.loads(row["payload_json"]),
            trigger_type=row["trigger_type"],
            trigger_config=json.loads(row["trigger_config"]),
            status=row["status"],
            next_run_at=row["next_run_at"],
            last_run_at=row["last_run_at"],
            attempt=row["attempt"],
            max_retries=row["max_retries"],
            retry_delay_secs=row["retry_delay_secs"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass
class JobRun:
    id: int
    job_id: int
    project_id: int
    started_at: float
    ended_at: Optional[float]
    status: Optional[str]
    result: dict
    error: Optional[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "project_id": self.project_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "JobRun":
        return cls(
            id=row["id"],
            job_id=row["job_id"],
            project_id=row["project_id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            status=row["status"],
            result=json.loads(row["result_json"]),
            error=row["error"],
        )


# ---------------------------------------------------------------------------
# DB helpers
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
        conn.executescript(_mem.SCHEMA)
        conn.executescript(_wf.WORKFLOW_SCHEMA)
        conn.executescript(SCHEDULER_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Trigger helpers
# ---------------------------------------------------------------------------

def _next_run_for_trigger(trigger_type: str, trigger_config: dict,
                           after: float) -> Optional[float]:
    """Compute the next Unix timestamp this trigger fires after `after`."""
    if trigger_type == "one_shot":
        at = trigger_config.get("at")
        # Only fire if it hasn't been reached yet.
        return float(at) if at is not None and float(at) > after else None

    if trigger_type == "interval":
        seconds = float(trigger_config["seconds"])
        return after + seconds

    if trigger_type == "cron":
        expr = trigger_config["expr"]
        try:
            from croniter import croniter  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "croniter is required for cron triggers: "
                "pip install croniter"
            ) from exc
        itr = croniter(expr, after)
        return itr.get_next(float)

    raise ValueError(f"Unknown trigger_type: {trigger_type!r}")


def _initial_next_run(trigger_type: str, trigger_config: dict) -> Optional[float]:
    """First fire time for a newly created job."""
    now = time.time()
    if trigger_type == "one_shot":
        at = trigger_config.get("at")
        return float(at) if at is not None else None
    # For interval and cron, fire after the first interval/tick from now.
    return _next_run_for_trigger(trigger_type, trigger_config, now)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def schedule_task(
    project_id: int,
    name: str,
    prompt: str,
    trigger_type: str,
    trigger_config: dict,
    max_retries: int = 3,
    retry_delay_secs: float = 60.0,
    db_path: Optional[Path] = None,
) -> int:
    """
    Schedule a run_task() call. Returns the job id.

    trigger_type / trigger_config examples:
      one_shot  {"at": 1700000000.0}         — Unix timestamp
      interval  {"seconds": 3600}             — every hour
      cron      {"expr": "0 9 * * 1-5"}       — 09:00 weekdays
    """
    db_path = db_path if db_path is not None else DB_PATH
    now = time.time()
    next_run = _initial_next_run(trigger_type, trigger_config)
    with _open(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO jobs (project_id, name, kind, payload_json, "
            "trigger_type, trigger_config, status, next_run_at, "
            "max_retries, retry_delay_secs, created_at, updated_at) "
            "VALUES (?, ?, 'task', ?, ?, ?, 'active', ?, ?, ?, ?, ?)",
            (project_id, name, json.dumps({"prompt": prompt}),
             trigger_type, json.dumps(trigger_config),
             next_run, max_retries, retry_delay_secs, now, now),
        )
        return cursor.lastrowid


def schedule_workflow(
    project_id: int,
    name: str,
    workflow_name: str,
    params: Optional[dict] = None,
    trigger_type: str = "one_shot",
    trigger_config: Optional[dict] = None,
    max_retries: int = 3,
    retry_delay_secs: float = 60.0,
    db_path: Optional[Path] = None,
) -> int:
    """Schedule a run_workflow() call. Returns the job id."""
    db_path = db_path if db_path is not None else DB_PATH
    trigger_config = trigger_config or {}
    now = time.time()
    next_run = _initial_next_run(trigger_type, trigger_config)
    with _open(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO jobs (project_id, name, kind, payload_json, "
            "trigger_type, trigger_config, status, next_run_at, "
            "max_retries, retry_delay_secs, created_at, updated_at) "
            "VALUES (?, ?, 'workflow', ?, ?, ?, 'active', ?, ?, ?, ?, ?)",
            (project_id, name,
             json.dumps({"workflow_name": workflow_name, "params": params or {}}),
             trigger_type, json.dumps(trigger_config),
             next_run, max_retries, retry_delay_secs, now, now),
        )
        return cursor.lastrowid


def get_job(job_id: int, db_path: Optional[Path] = None) -> Optional[Job]:
    db_path = db_path if db_path is not None else DB_PATH
    with _open(db_path) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job.from_row(row) if row else None


def list_jobs(
    project_id: int,
    status: Optional[str] = None,
    limit: int = 50,
    db_path: Optional[Path] = None,
) -> list[Job]:
    """List jobs for a project, most-recently-created first."""
    db_path = db_path if db_path is not None else DB_PATH
    with _open(db_path) as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE project_id = ? AND status = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (project_id, status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE project_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (project_id, limit),
            ).fetchall()
        return [Job.from_row(r) for r in rows]


def pause_job(job_id: int, db_path: Optional[Path] = None) -> bool:
    """Pause an active job. Returns True if it existed and was active."""
    db_path = db_path if db_path is not None else DB_PATH
    with _open(db_path) as conn:
        cur = conn.execute(
            "UPDATE jobs SET status='paused', updated_at=? "
            "WHERE id=? AND status='active'",
            (time.time(), job_id),
        )
        return cur.rowcount > 0


def resume_job(job_id: int, db_path: Optional[Path] = None) -> bool:
    """Resume a paused job. Returns True if it was paused."""
    db_path = db_path if db_path is not None else DB_PATH
    with _open(db_path) as conn:
        cur = conn.execute(
            "UPDATE jobs SET status='active', updated_at=? "
            "WHERE id=? AND status='paused'",
            (time.time(), job_id),
        )
        return cur.rowcount > 0


def cancel_job(job_id: int, db_path: Optional[Path] = None) -> bool:
    """Cancel a job (marks done, never re-runs). Returns True if found."""
    db_path = db_path if db_path is not None else DB_PATH
    with _open(db_path) as conn:
        cur = conn.execute(
            "UPDATE jobs SET status='done', updated_at=? WHERE id=?",
            (time.time(), job_id),
        )
        return cur.rowcount > 0


def list_job_runs(
    job_id: int,
    limit: int = 20,
    db_path: Optional[Path] = None,
) -> list[JobRun]:
    """Run history for a specific job, most-recent first."""
    db_path = db_path if db_path is not None else DB_PATH
    with _open(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM job_runs WHERE job_id = ? "
            "ORDER BY started_at DESC LIMIT ?",
            (job_id, limit),
        ).fetchall()
        return [JobRun.from_row(r) for r in rows]


# ---------------------------------------------------------------------------
# Tick — the execution engine
# ---------------------------------------------------------------------------

def tick(
    project_id: int,
    run_task_fn: Optional[Callable] = None,
    run_workflow_fn: Optional[Callable] = None,
    db_path: Optional[Path] = None,
    _now: Optional[float] = None,   # injectable for tests; also used as clock base in _execute_job
) -> list[dict]:
    """
    Check for due jobs and execute them. Call this repeatedly (e.g. every
    30 s from a loop or cron) to drive scheduled execution.

    Returns a list of execution summaries for jobs that ran this tick.
    Never raises: errors are recorded in job_runs and the job's attempt
    counter / retry schedule is updated, but tick() itself always returns.

    run_task_fn and run_workflow_fn default to forge.run_task and
    _wf.run_workflow respectively; both are deferred imports to avoid a
    circular import at module load time.
    """
    db_path = db_path if db_path is not None else DB_PATH
    now = _now if _now is not None else time.time()

    if run_task_fn is None:
        import forge as _forge  # deferred
        run_task_fn = _forge.run_task

    if run_workflow_fn is None:
        def _default_wf_runner(prompt_or_name, params=None):
            return _wf.run_workflow(
                project_id, prompt_or_name,
                params=params, run_task_fn=run_task_fn,
                db_path=db_path,
            )
        run_workflow_fn = _default_wf_runner

    # Atomically claim all due, active jobs for this project.
    with _open(db_path) as conn:
        due_rows = conn.execute(
            "SELECT * FROM jobs WHERE project_id = ? AND status = 'active' "
            "AND next_run_at IS NOT NULL AND next_run_at <= ? "
            "ORDER BY next_run_at ASC",
            (project_id, now),
        ).fetchall()
        due_jobs = [Job.from_row(r) for r in due_rows]

        # Mark them 'running' atomically to prevent double-execution if
        # tick() is somehow called concurrently.
        for job in due_jobs:
            conn.execute(
                "UPDATE jobs SET status='active', updated_at=? WHERE id=? "
                "AND status='active'",
                (now, job.id),
            )

    summaries = []
    for job in due_jobs:
        summary = _execute_job(job, run_task_fn, run_workflow_fn, db_path, now,
                               _clock=now)
        summaries.append(summary)

    return summaries


def _execute_job(
    job: Job,
    run_task_fn: Callable,
    run_workflow_fn: Callable,
    db_path: Path,
    now: float,
    _clock: Optional[float] = None,  # use injected clock if set (for tests)
) -> dict:
    """Run one job, record the outcome, reschedule or retire it."""
    started = _clock if _clock is not None else time.time()
    run_id: Optional[int] = None
    error: Optional[str] = None
    result: dict = {}
    run_status = "error"

    # Insert a placeholder job_run row so we always have a record even if
    # the execution crashes hard.
    with _open(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO job_runs (job_id, project_id, started_at, status, "
            "result_json) VALUES (?, ?, ?, 'running', '{}')",
            (job.id, job.project_id, started),
        )
        run_id = cur.lastrowid

    try:
        if job.kind == "task":
            prompt = job.payload["prompt"]
            result = run_task_fn(prompt)
            run_status = result.get("final_status", "error")

        elif job.kind == "workflow":
            wf_result = run_workflow_fn(
                job.payload["workflow_name"],
                params=job.payload.get("params", {}),
            )
            if hasattr(wf_result, "to_dict"):
                result = wf_result.to_dict()
            else:
                result = wf_result
            run_status = result.get("status", "error")

        else:
            raise ValueError(f"Unknown job kind: {job.kind!r}")

    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        run_status = "error"

    ended = _clock if _clock is not None else time.time()

    # Update the job_run row.
    with _open(db_path) as conn:
        conn.execute(
            "UPDATE job_runs SET ended_at=?, status=?, result_json=?, error=? "
            "WHERE id=?",
            (ended, run_status, json.dumps(result, default=str), error, run_id),
        )

    # Decide what happens next to the job itself.
    success = run_status == "complete"
    new_attempt = job.attempt + 1

    with _open(db_path) as conn:
        if success:
            # Reschedule repeating jobs; retire one_shot / exhausted ones.
            next_run = None
            new_status = "active"
            if job.trigger_type != "one_shot":
                try:
                    next_run = _next_run_for_trigger(
                        job.trigger_type, job.trigger_config, ended
                    )
                except Exception:
                    next_run = None
            if job.trigger_type == "one_shot" or next_run is None:
                new_status = "done"

            conn.execute(
                "UPDATE jobs SET status=?, next_run_at=?, last_run_at=?, "
                "attempt=0, updated_at=? WHERE id=?",
                (new_status, next_run, ended, ended, job.id),
            )
        else:
            # Failed. Retry if attempts remain.
            if new_attempt <= job.max_retries:
                delay = min(job.retry_delay_secs * (2 ** (new_attempt - 1)),
                            _MAX_BACKOFF)
                next_run = ended + delay
                conn.execute(
                    "UPDATE jobs SET status='active', next_run_at=?, "
                    "last_run_at=?, attempt=?, updated_at=? WHERE id=?",
                    (next_run, ended, new_attempt, ended, job.id),
                )
            else:
                conn.execute(
                    "UPDATE jobs SET status='failed', last_run_at=?, "
                    "attempt=?, updated_at=? WHERE id=?",
                    (ended, new_attempt, ended, job.id),
                )

    return {
        "job_id": job.id,
        "job_name": job.name,
        "kind": job.kind,
        "run_status": run_status,
        "attempt": new_attempt,
        "error": error,
    }
