"""FastAPI service exposing the milestone 0 hello agent."""

from __future__ import annotations

from fastapi import FastAPI

from app.agents.hello_agent import HelloAgent, HelloAgentRequest, HelloAgentResponse

app = FastAPI(
    title="Agentic Neural Networks Tutor",
    description="API surface for orchestrating the neural networks tutoring agents.",
    version="0.1.0",
)

_hello_agent = HelloAgent()


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    """Return a minimal readiness signal for orchestrator tooling."""

    return {"status": "ok"}


@app.post("/hello-agent", response_model=HelloAgentResponse)
def run_hello_agent(request: HelloAgentRequest) -> HelloAgentResponse:
    """Execute the hello agent and return its structured response."""

    return _hello_agent.run(request)
