"""FastAPI service exposing the milestone 0 hello agent."""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, HTTPException, Query

from app.agents.hello_agent import HelloAgent, HelloAgentRequest, HelloAgentResponse
from app.agents.session import LearnerProfile, SessionOrchestrator, SessionTranscript
from app.content import CONTENT_REPOSITORY, CanonicalFact, ConceptDetail

app = FastAPI(
    title="Agentic Neural Networks Tutor",
    description="API surface for orchestrating the neural networks tutoring agents.",
    version="0.1.0",
)

_hello_agent = HelloAgent()
_session_orchestrator = SessionOrchestrator()


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    """Return a minimal readiness signal for orchestrator tooling."""

    return {"status": "ok"}


@app.post("/hello-agent", response_model=HelloAgentResponse)
def run_hello_agent(request: HelloAgentRequest) -> HelloAgentResponse:
    """Execute the hello agent and return its structured response."""

    return _hello_agent.run(request)


@app.get("/concept/{concept_id}", response_model=ConceptDetail)
def get_concept(concept_id: str) -> ConceptDetail:
    """Return the fully expanded concept payload for curriculum consumption."""

    detail = CONTENT_REPOSITORY.get_concept_detail(concept_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Concept '{concept_id}' not found.")
    return detail


FactsQuery = Annotated[list[str], Query(min_items=1)]


@app.get("/facts", response_model=list[CanonicalFact])
def get_canonical_facts(ids: FactsQuery) -> list[CanonicalFact]:
    """Return canonical facts for the supplied identifiers."""

    facts = CONTENT_REPOSITORY.get_canonical_facts(ids)
    if not facts:
        raise HTTPException(status_code=404, detail="No canonical facts found for supplied ids.")
    return facts


@app.post("/session/thin-slice", response_model=SessionTranscript)
def run_thin_slice_session(profile: LearnerProfile) -> SessionTranscript:
    """Execute the milestone three thin-slice session and return the transcript."""

    return _session_orchestrator.run_thin_slice(profile)
