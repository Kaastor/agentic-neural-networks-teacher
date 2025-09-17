"""Integration tests for the content-service HTTP endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from services.api.main import app

client = TestClient(app)


def test_get_concept_returns_backprop_module() -> None:
    """The content endpoint should return the fully expanded backpropagation concept."""

    response = client.get("/concept/concept-backpropagation", timeout=5.0)
    assert response.status_code == 200
    payload = response.json()
    assert payload["concept"]["title"] == "Backpropagation"
    assert len(payload["concept"]["sections"]) >= 12
    assert len(payload["problem_templates"]) >= 8


def test_get_concept_returns_404_for_unknown_concept() -> None:
    """Unknown concept identifiers should raise a 404 indicating absence."""

    response = client.get("/concept/concept-unknown", timeout=5.0)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_canonical_facts_returns_expected_subset() -> None:
    """Facts endpoint should return the subset that exists while rejecting unknown sets."""

    response = client.get(
        "/facts",
        params=[("ids", "fact-chain-rule"), ("ids", "fact-softmax-gradient")],
        timeout=5.0,
    )
    assert response.status_code == 200
    payload = response.json()
    returned_ids = {fact["id"] for fact in payload}
    assert returned_ids == {"fact-chain-rule", "fact-softmax-gradient"}


def test_get_canonical_facts_returns_404_for_empty_match() -> None:
    """Supplying only unknown fact identifiers should result in a 404."""

    response = client.get("/facts", params=[("ids", "fact-unknown")], timeout=5.0)
    assert response.status_code == 404
