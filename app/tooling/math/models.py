"""Pydantic models supporting the math verification service."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class EqualityCheckRequest(BaseModel):
    """Payload for verifying whether two symbolic expressions are equivalent."""

    canonical: str = Field(..., description="Reference expression considered correct.")
    candidate: str = Field(..., description="Expression supplied by the learner or agent.")
    symbols: list[str] = Field(
        default_factory=list,
        description="Optional ordered list of symbols used in the expressions.",
    )
    random_trials: int = Field(
        default=3,
        ge=0,
        le=8,
        description="Number of random numeric evaluations used as a fallback check.",
    )


class EqualityCheckResponse(BaseModel):
    """Result of the symbolic equality verification."""

    equivalent: bool = Field(..., description="True when the expressions are mathematically equal.")
    detail: str = Field(..., description="Human-oriented explanation of the verification outcome.")


FunctionVector = Annotated[list[str], Field(min_length=1)]


class DerivativeCheckRequest(BaseModel):
    """Payload describing a derivative (Jacobian) verification task."""

    function: FunctionVector = Field(
        ..., description="Components of the function being differentiated."
    )
    variables: list[str] = Field(
        ..., min_length=1, description="Ordered variables the Jacobian is taken with respect to."
    )
    candidate: list[list[str]] = Field(
        ..., description="Learner-supplied Jacobian components aligned with function × variables.",
    )
    evaluation_point: dict[str, float] | None = Field(
        default=None,
        description="Optional point at which to numerically evaluate the Jacobian difference.",
    )
    random_trials: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Fallback random evaluations to detect subtle mismatches.",
    )

    @model_validator(mode="after")
    def validate_candidate_shape(self) -> "DerivativeCheckRequest":
        """Ensure the candidate Jacobian matches the expected dimensions."""

        expected_rows = len(self.function)
        expected_cols = len(self.variables)
        if len(self.candidate) != expected_rows:
            msg = (
                "Candidate Jacobian row count does not match the function dimension: expected "
                f"{expected_rows}, received {len(self.candidate)}."
            )
            raise ValueError(msg)
        for row in self.candidate:
            if len(row) != expected_cols:
                msg = (
                    "Candidate Jacobian column count does not match variable dimension: expected "
                    f"{expected_cols}, received {len(row)}."
                )
                raise ValueError(msg)
        return self


class DerivativeCheckResponse(BaseModel):
    """Result of the Jacobian verification."""

    equivalent: bool = Field(..., description="True when the supplied Jacobian matches the reference.")
    detail: str = Field(..., description="Explanation of the verification verdict.")
    symbolic_difference: list[list[str]] = Field(
        default_factory=list,
        description="Simplified symbolic difference between reference and candidate Jacobian.",
    )
    evaluated_difference: list[list[float]] | None = Field(
        default=None,
        description="Numeric evaluation of the difference, when an evaluation point was provided.",
    )
