"""Unit tests for the milestone 0 hello agent."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.agents.hello_agent import HelloAgent, HelloAgentRequest
from services.api.main import app


def test_hello_agent_response_structure() -> None:
    """The agent should produce deterministic, structured JSON output."""

    agent = HelloAgent(agent_name="test-agent", version="9.9.9")
    request = HelloAgentRequest(user_name="Ada", objective="diagnostic")

    response = agent.run(request)

    assert response.agent_name == "test-agent"
    assert "personalized NN study plans" in response.message
    assert response.metadata["version"] == "9.9.9"
    assert response.metadata["objective"] == "diagnostic"


def test_http_route_returns_agent_payload() -> None:
    """The HTTP surface should wrap the agent and return the same payload."""

    client = TestClient(app)
    payload = {"user_name": "Grace", "objective": "warmup"}

    response = client.post("/hello-agent", json=payload, timeout=5.0)

    assert response.status_code == 200
    data = response.json()
    assert data["agent_name"] == "planner-tutor-prototype"
    assert data["objective"] == "warmup"
    assert data["metadata"]["version"] == "0.1.0"
