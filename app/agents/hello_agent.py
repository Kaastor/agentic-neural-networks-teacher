"""Prototype hello agent used to validate the scaffolding for the agent stack."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HelloAgentRequest(BaseModel):
    """Input payload for the hello agent."""

    user_name: str = Field(..., description="Name of the learner engaging with the agent.")
    objective: str | None = Field(
        default=None,
        description="Optional learning objective that frames the agent response.",
    )


class HelloAgentResponse(BaseModel):
    """Structured response emitted by the hello agent."""

    agent_name: str = Field(
        default="planner-tutor-prototype", description="Identifier for the agent."
    )
    message: str = Field(..., description="Natural language response from the agent.")
    objective: str | None = Field(
        default=None,
        description="Echo of the learner objective for observability and downstream routing.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context such as version and confidence signals.",
    )


class HelloAgent:
    """Simple deterministic agent that returns a structured greeting."""

    def __init__(
        self, *, agent_name: str = "planner-tutor-prototype", version: str = "0.1.0"
    ) -> None:
        self._agent_name = agent_name
        self._version = version

    def run(self, request: HelloAgentRequest) -> HelloAgentResponse:
        """Return a JSON-serializable greeting for the supplied learner."""

        message = (
            f"Hello, {request.user_name}. "
            f"I am {self._agent_name}. We'll use this pipeline to orchestrate personalized NN study plans."
        )
        metadata = {"version": self._version}
        if request.objective:
            metadata["objective"] = request.objective
        return HelloAgentResponse(
            agent_name=self._agent_name,
            message=message,
            objective=request.objective,
            metadata=metadata,
        )


hello_agent = HelloAgent()
"""Module-level instance for simple integrations."""
