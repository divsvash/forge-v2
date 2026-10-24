"""
run_record.py — a durable, structured record of one Forge task.

The previous runtime only had print() statements and an in-memory
`evidence` dict that vanished when the process exited. This gives each
`run_task()` call a unique id and a JSON record on disk containing the
task input, model/config info, every tool call in order (arguments,
result, error, timestamps, duration), the completion check, and the
final status. Failed tool calls are recorded as evidence here, not
discarded — a failed run is exactly the kind of thing the learning
harness (next slice) needs to see.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

RUNS_DIR = Path(__file__).parent / "runs"


@dataclass
class ToolCallRecord:
    step: int
    tool: str
    arguments: dict
    status: str  # "success" | "error"
    result: Any = None
    error: str | None = None
    started_at: float = 0.0
    duration_seconds: float = 0.0
    redundant: bool = False


@dataclass
class RunRecord:
    run_id: str
    task: str
    model: str
    started_at: float
    ended_at: float | None = None
    tool_calls: list = field(default_factory=list)
    completion: dict | None = None
    final_status: str = "running"  # "running" | "complete" | "incomplete" | "error"
    final_answer: str | None = None
    steps_taken: int = 0
    stalled: bool = False
    error: str | None = None

    def add_tool_call(self, record: "ToolCallRecord") -> None:
        self.tool_calls.append(asdict(record))

    def finish(self, *, status: str, completion: dict | None, answer: str | None,
               steps: int, stalled: bool = False, error: str | None = None) -> None:
        self.ended_at = time.time()
        self.final_status = status
        self.completion = completion
        self.final_answer = answer
        self.steps_taken = steps
        self.stalled = stalled
        self.error = error

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tool_calls"] = self.tool_calls
        return data

    def save(self, directory: Path | None = None) -> Path:
        directory = directory or RUNS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return path


def new_run(task: str, model: str) -> RunRecord:
    return RunRecord(
        run_id=str(uuid.uuid4()),
        task=task,
        model=model,
        started_at=time.time(),
    )


def load_run(run_id: str, directory: Path | None = None) -> dict:
    directory = directory or RUNS_DIR
    path = directory / f"{run_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def list_runs(directory: Path | None = None) -> list[str]:
    directory = directory or RUNS_DIR
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.json"))
