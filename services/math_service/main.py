"""FastAPI surface for math verification tooling."""

from __future__ import annotations

from fastapi import FastAPI

from app.tooling.math.checks import check_jacobian, check_symbolic_equality
from app.tooling.math.models import (
    DerivativeCheckRequest,
    DerivativeCheckResponse,
    EqualityCheckRequest,
    EqualityCheckResponse,
)

app = FastAPI(
    title="Math Verification Service",
    description="Symbolic math utilities for the agentic tutor stack.",
    version="0.1.0",
)


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    """Return readiness information for orchestration tooling."""

    return {"status": "ok"}


@app.post("/check/equality", response_model=EqualityCheckResponse)
def check_equality(request: EqualityCheckRequest) -> EqualityCheckResponse:
    """Return verdict for symbolic equality between canonical and candidate expressions."""

    return check_symbolic_equality(request)


@app.post("/check/derivative", response_model=DerivativeCheckResponse)
def check_derivative(request: DerivativeCheckRequest) -> DerivativeCheckResponse:
    """Return verdict for whether a supplied Jacobian matches the true Jacobian."""

    return check_jacobian(request)
