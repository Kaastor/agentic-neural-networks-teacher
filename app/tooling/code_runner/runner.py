"""Sandboxed code execution utilities."""

from __future__ import annotations

import math
import resource
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from app.tooling.code_runner.models import CodeArtifact, CodeRunRequest, CodeRunResponse

_MAX_ARTIFACTS = 16
_MAX_ARTIFACT_SIZE_BYTES = 10 * 1024 * 1024  # 10 MiB per artifact snapshot.
_BLOCKED_ENV_PREFIXES = ("LD_", "PYTHONPATH", "VIRTUAL_ENV", "POETRY", "PATH")


def _is_allowed_env_key(key: str) -> bool:
    """Return True when an environment variable key is allowed to propagate."""

    if any(key.startswith(prefix) for prefix in _BLOCKED_ENV_PREFIXES):
        return False
    return key.isupper() and key.replace("_", "").isalnum()


def _truncate_output(stream: str, limit_kb: int) -> tuple[str, bool]:
    """Truncate a stream to the configured byte limit while preserving UTF-8 integrity."""

    limit_bytes = limit_kb * 1024
    encoded = stream.encode("utf-8", errors="replace")
    if len(encoded) <= limit_bytes:
        return stream, False
    truncated = encoded[:limit_bytes]
    decoded = truncated.decode("utf-8", errors="ignore")
    return f"{decoded}\n[truncated]", True


def _set_resource_limits(request: CodeRunRequest) -> Callable[[], None]:
    """Return a pre-exec callable that applies resource limits inside the child process."""

    limits = request.limits
    memory_bytes = limits.memory_limit_mb * 1024 * 1024
    cpu_seconds = max(1, math.ceil(limits.cpu_seconds))

    def _apply_limits() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (memory_bytes, memory_bytes))
        resource.setrlimit(
            resource.RLIMIT_FSIZE,
            (_MAX_ARTIFACT_SIZE_BYTES, _MAX_ARTIFACT_SIZE_BYTES),
        )
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))

    return _apply_limits


def _collect_artifacts(root: Path) -> list[CodeArtifact]:
    """Return artifact metadata for files created during execution."""

    artifacts: list[CodeArtifact] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root.parent)
        size = path.stat().st_size
        if size > _MAX_ARTIFACT_SIZE_BYTES:
            continue
        artifacts.append(CodeArtifact(path=str(relative), size_bytes=size))
        if len(artifacts) >= _MAX_ARTIFACTS:
            break
    return artifacts


def run_code(request: CodeRunRequest) -> CodeRunResponse:
    """Execute Python code in a constrained subprocess sandbox."""

    if request.language != "python":
        raise ValueError("Only Python execution is supported at this time.")

    limits = request.limits
    env = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    for key, value in request.environment.items():
        if _is_allowed_env_key(key):
            env[key] = value

    with TemporaryDirectory(prefix="code-runner-") as tmp:
        workdir = Path(tmp)
        artifacts_dir = workdir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        env["ARTIFACTS_DIR"] = str(artifacts_dir)

        submission_path = workdir / "submission.py"
        submission_path.write_text(request.code, encoding="utf-8")

        command = [sys.executable, "-I", "-u", str(submission_path), *request.args]

        start = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=limits.timeout_seconds,
                env=env,
                preexec_fn=_set_resource_limits(request),
            )
            runtime = time.monotonic() - start
            stdout, stdout_truncated = _truncate_output(completed.stdout or "", limits.stdout_limit_kb)
            stderr, stderr_truncated = _truncate_output(completed.stderr or "", limits.stdout_limit_kb)
            artifacts = _collect_artifacts(artifacts_dir)
            return CodeRunResponse(
                exit_code=completed.returncode,
                runtime_seconds=runtime,
                timed_out=False,
                stdout=stdout,
                stderr=stderr,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
                artifacts=artifacts,
            )
        except subprocess.TimeoutExpired as exc:
            runtime = time.monotonic() - start
            stdout, stdout_truncated = _truncate_output(exc.stdout or "", limits.stdout_limit_kb)
            stderr_message = (exc.stderr or "") + "\n[terminated: wall-clock timeout exceeded]"
            stderr, stderr_truncated = _truncate_output(stderr_message, limits.stdout_limit_kb)
            artifacts = _collect_artifacts(artifacts_dir)
            return CodeRunResponse(
                exit_code=None,
                runtime_seconds=runtime,
                timed_out=True,
                stdout=stdout,
                stderr=stderr,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
                artifacts=artifacts,
            )
