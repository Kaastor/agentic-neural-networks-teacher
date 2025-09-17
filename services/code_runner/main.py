"""FastAPI surface for the sandboxed code runner."""

from __future__ import annotations

from fastapi import FastAPI

from app.tooling.code_runner.models import CodeRunRequest, CodeRunResponse
from app.tooling.code_runner.runner import run_code

app = FastAPI(
    title="Code Runner Service",
    description="Sandboxed execution utility for coding labs and graders.",
    version="0.1.0",
)


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    """Return readiness information for orchestrator tooling."""

    return {"status": "ok"}


@app.post("/run", response_model=CodeRunResponse)
def run(request: CodeRunRequest) -> CodeRunResponse:
    """Execute supplied code inside the sandbox and return structured results."""

    return run_code(request)
