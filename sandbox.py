"""
sandbox.py — execution backends for Forge's run_command tool.

The previous implementation ran model-supplied shell strings with
``subprocess.run(..., shell=True)`` and only a cwd change. That is not
isolation: a command can still read, write, or delete anything the host
process can touch, and can still reach the network. This module makes the
isolation level explicit and honest instead of implicit and assumed.

Two backends:

  RestrictedLocalBackend
      Always available. Removes the shell-metacharacter and environment-
      leak risks (scrubbed env, explicit timeout, output size caps,
      resource limits on POSIX via RLIMIT) and confines the *working
      directory* to the workspace. It does NOT stop a command from
      touching files elsewhere on disk that the OS lets the host process
      touch, and it does NOT block outbound network access. It must never
      be reported as "sandboxed" — only as "restricted, unisolated".

  DockerBackend
      Used only when a Docker daemon is actually reachable. Runs the
      command inside a short-lived container: workspace bind-mounted
      read-write, everything else is the container's own filesystem,
      ``--network none``, memory/pids/cpu limits, no new privileges, auto
      removed. This is the backend that can honestly be called isolated.

``select_backend()`` probes for Docker and falls back to the restricted
backend when it isn't reachable, returning the backend together with a
``Capabilities`` description. Callers must persist which backend actually
ran a command (the run record does this) rather than assuming the strong
backend was used.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# POSIX-only resource limits (CPU time, memory, process count). On Windows
# this module is simply unavailable and the restricted backend runs
# without those limits — that fact is recorded in Capabilities.notes, not
# hidden.
try:
    import resource as _resource  # type: ignore
    _HAS_RESOURCE = True
except ImportError:  # Windows
    _resource = None
    _HAS_RESOURCE = False

MAX_OUTPUT_CHARS = 200_000

# Environment variables passed through to the child process. Everything
# else — including anything that looks like a credential — is stripped so
# a model-supplied command can't exfiltrate the host's secrets via `env`
# or similar.
_ENV_ALLOWLIST = {
    "PATH", "HOME", "LANG", "LC_ALL", "PYTHONIOENCODING",
    "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "COMSPEC",
}


@dataclass
class Capabilities:
    """What a backend actually guarantees. Never assume; always report."""
    backend: str
    filesystem_isolated: bool
    network_blocked: bool
    resource_limited: bool
    notes: str


@dataclass
class ExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    backend: str
    capabilities: Capabilities
    command: str = ""

    def as_dict(self) -> dict:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "backend": self.backend,
            "capabilities": {
                "filesystem_isolated": self.capabilities.filesystem_isolated,
                "network_blocked": self.capabilities.network_blocked,
                "resource_limited": self.capabilities.resource_limited,
                "notes": self.capabilities.notes,
            },
        }


def _truncate(text: str, limit: int | None = None) -> str:
    # Read MAX_OUTPUT_CHARS at call time (not as a default-arg value bound
    # at import time) so tests/configuration can override it at runtime.
    if limit is None:
        limit = MAX_OUTPUT_CHARS
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more chars]"


def _scrubbed_env() -> dict:
    env = {key: os.environ[key] for key in _ENV_ALLOWLIST if key in os.environ}
    # Make sure PATH exists even if nothing matched (shouldn't happen, but
    # a command with no PATH is a confusing failure mode).
    env.setdefault("PATH", os.defpath)
    return env


class RestrictedLocalBackend:
    """Best-effort guardrails, explicitly not isolation."""

    name = "restricted-local"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            backend=self.name,
            filesystem_isolated=False,
            network_blocked=False,
            resource_limited=_HAS_RESOURCE,
            notes=(
                "No filesystem or network isolation. Scrubbed environment, "
                "capped output, enforced timeout"
                + (", POSIX CPU/memory/process-count rlimits." if _HAS_RESOURCE
                   else " (no rlimits available on this platform).")
            ),
        )

    def _preexec(self):
        if not _HAS_RESOURCE:
            return None

        def _limit():
            # 30s CPU time, 512MB address space, max 64 child processes.
            _resource.setrlimit(_resource.RLIMIT_CPU, (30, 30))
            try:
                _resource.setrlimit(
                    _resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024)
                )
            except (ValueError, OSError):
                pass  # Some platforms (e.g. macOS) restrict RLIMIT_AS.
            try:
                _resource.setrlimit(_resource.RLIMIT_NPROC, (64, 64))
            except (ValueError, OSError):
                pass

        return _limit

    def run(self, command: str, workspace: Path, timeout: int) -> ExecutionResult:
        caps = self.capabilities()
        kwargs = dict(
            cwd=workspace,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_scrubbed_env(),
        )
        if _HAS_RESOURCE:
            kwargs["preexec_fn"] = self._preexec()

        try:
            result = subprocess.run(command, **kwargs)
            return ExecutionResult(
                exit_code=result.returncode,
                stdout=_truncate(result.stdout),
                stderr=_truncate(result.stderr),
                timed_out=False,
                backend=self.name,
                capabilities=caps,
                command=command,
            )
        except subprocess.TimeoutExpired as error:
            return ExecutionResult(
                exit_code=-1,
                stdout=_truncate(error.stdout or ""),
                stderr=_truncate((error.stderr or "") + "\n[killed: timeout]"),
                timed_out=True,
                backend=self.name,
                capabilities=caps,
                command=command,
            )


class DockerBackend:
    """Real isolation: short-lived container, no network, resource caps."""

    name = "docker"
    default_image = "python:3.11-slim"

    def __init__(self, image: str | None = None):
        self.image = image or self.default_image

    def capabilities(self) -> Capabilities:
        return Capabilities(
            backend=self.name,
            filesystem_isolated=True,
            network_blocked=True,
            resource_limited=True,
            notes=(
                f"Container ({self.image}), workspace bind-mounted "
                "read-write, --network none, memory/pids/cpu capped, "
                "no-new-privileges, auto-removed."
            ),
        )

    def run(self, command: str, workspace: Path, timeout: int) -> ExecutionResult:
        caps = self.capabilities()
        container_name = f"forge-{uuid.uuid4().hex[:12]}"
        docker_cmd = [
            "docker", "run", "--rm",
            "--name", container_name,
            "-v", f"{workspace.resolve()}:/workspace",
            "-w", "/workspace",
            "--network", "none",
            "--memory", "256m",
            "--pids-limit", "128",
            "--cpus", "1",
            "--security-opt", "no-new-privileges",
            self.image,
            "sh", "-c", command,
        ]

        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecutionResult(
                exit_code=result.returncode,
                stdout=_truncate(result.stdout),
                stderr=_truncate(result.stderr),
                timed_out=False,
                backend=self.name,
                capabilities=caps,
                command=command,
            )
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["docker", "kill", container_name],
                capture_output=True, timeout=10,
            )
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr="[killed: timeout]",
                timed_out=True,
                backend=self.name,
                capabilities=caps,
                command=command,
            )


def _docker_daemon_reachable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def select_backend(prefer: str = "auto", docker_image: str | None = None):
    """Pick the strongest backend actually available right now.

    prefer="auto"   -> Docker if the daemon is reachable, else restricted-local.
    prefer="docker" -> Docker, raising if the daemon isn't reachable.
    prefer="local"  -> always the restricted local backend.
    """
    if prefer == "local":
        return RestrictedLocalBackend()

    if prefer == "docker":
        if not _docker_daemon_reachable():
            raise RuntimeError(
                "Docker backend requested but the Docker daemon is not "
                "reachable. Refusing to silently fall back for an explicit "
                "request — pass prefer='auto' to allow fallback."
            )
        return DockerBackend(image=docker_image)

    if _docker_daemon_reachable():
        return DockerBackend(image=docker_image)

    return RestrictedLocalBackend()
