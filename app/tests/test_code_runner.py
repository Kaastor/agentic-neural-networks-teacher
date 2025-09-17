"""Unit tests for the sandboxed code runner."""

from __future__ import annotations

from app.tooling.code_runner.models import CodeRunRequest, ExecutionLimits
from app.tooling.code_runner.runner import run_code


def test_code_runner_executes_program_and_collects_artifacts() -> None:
    """Runner executes code, captures output, and records artifacts."""

    code = (
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "artifact_dir = Path(os.environ['ARTIFACTS_DIR'])\n"
        "(artifact_dir / 'metrics.json').write_text(json.dumps({'accuracy': 0.95}))\n"
        "print('hello sandbox')\n"
    )
    request = CodeRunRequest(code=code)
    result = run_code(request)

    assert result.exit_code == 0
    assert result.timed_out is False
    assert "hello sandbox" in result.stdout
    assert any(artifact.path.endswith("metrics.json") for artifact in result.artifacts)


def test_code_runner_enforces_timeout() -> None:
    """Runner terminates executions that exceed the configured timeout."""

    code = "while True: pass"
    limits = ExecutionLimits(timeout_seconds=0.5, cpu_seconds=0.5)
    request = CodeRunRequest(code=code, limits=limits)
    result = run_code(request)

    assert result.timed_out is True
    assert result.exit_code is None


def test_code_runner_blocks_process_creation() -> None:
    """Runner prevents subprocess or fork attempts via RLIMIT_NPROC."""

    code = (
        "import errno\n"
        "import os\n"
        "import sys\n"
        "try:\n"
        "    os.fork()\n"
        "except OSError as exc:\n"
        "    if exc.errno == errno.EAGAIN:\n"
        "        print('fork blocked')\n"
        "        sys.exit(0)\n"
        "    raise\n"
        "else:\n"
        "    sys.exit(1)\n"
    )
    request = CodeRunRequest(code=code)
    result = run_code(request)

    assert result.exit_code == 0
    assert "fork blocked" in result.stdout
