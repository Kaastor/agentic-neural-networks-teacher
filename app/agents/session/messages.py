"""Typed message schemas underpinning the session agent graph."""

from __future__ import annotations

from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from app.content.schema import CanonicalFact, ProblemInstance
from app.tooling.code_runner.models import CodeRunResponse


class AgentNode(str, Enum):
    """Identifiers for the core agents participating in the session graph."""

    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    TUTOR = "tutor"
    ASSESSOR = "assessor"
    CRITIC = "critic"


SessionAction = Literal["explain", "derive", "quiz", "code"]


class BaseMessage(BaseModel):
    """Base wire model including metadata for idempotent retries."""

    message_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for correlating requests and responses.",
    )
    attempt: int = Field(
        default=1,
        ge=1,
        description="1-indexed retry counter supporting idempotent handling.",
    )


class LearnerProfile(BaseModel):
    """Static profile information about the learner engaging with the tutor."""

    learner_id: str = Field(..., description="Stable identifier for the learner.")
    name: str = Field(..., description="Human-readable learner name for personalization.")
    target_concept_id: str = Field(
        ..., description="Concept that anchors the current tutoring session."
    )


class LearnerState(BaseModel):
    """Mutable learner state tracked across the session turns."""

    concept_id: str = Field(..., description="Concept currently being studied.")
    completed_actions: list[SessionAction] = Field(
        default_factory=list,
        description="Ordered list of actions already completed in the session.",
    )
    retrieved_facts: dict[str, CanonicalFact] = Field(
        default_factory=dict,
        description="Cache of canonical facts fetched so far keyed by fact identifier.",
    )

    def record_action(self, action: SessionAction) -> None:
        """Append a completed action to the learner history."""

        self.completed_actions.append(action)

    def cache_facts(self, facts: list[CanonicalFact]) -> None:
        """Merge canonical facts into the cache, preserving the latest copies."""

        for fact in facts:
            self.retrieved_facts[fact.id] = fact


class PlannerRequest(BaseMessage):
    """Planner input containing the learner profile and current session state."""

    sender: AgentNode = Field(default=AgentNode.ORCHESTRATOR)
    learner: LearnerProfile = Field(..., description="Profile data for personalization.")
    state: LearnerState = Field(..., description="Current learner state representation.")


class DerivationTarget(BaseModel):
    """Expected derivation outcome used for symbolic verification."""

    canonical_expression: str = Field(
        ..., description="Canonical symbolic expression representing the correct derivation."
    )
    symbols: list[str] = Field(
        default_factory=list,
        description="Ordered symbol names used when parsing the expressions.",
    )


class PlannerDecision(BaseMessage):
    """Planner output describing the next action in the session graph."""

    sender: AgentNode = Field(default=AgentNode.PLANNER)
    action: SessionAction = Field(..., description="Selected pedagogical action.")
    learning_objective_id: str = Field(
        ..., description="Learning objective targeted by the action."
    )
    canonical_fact_ids: list[str] = Field(
        default_factory=list,
        description="Canonical facts the downstream tutor should ground to.",
    )
    prompt: str = Field(
        ..., description="High-level instruction describing what the tutor should achieve."
    )
    problem_template_id: str | None = Field(
        default=None,
        description="Problem template identifier when the action requires assessment.",
    )
    variant_id: str | None = Field(
        default=None, description="Deterministic problem variant identifier."
    )
    derivation_target: DerivationTarget | None = Field(
        default=None,
        description="Target used when verifying symbolic derivations.",
    )


class TutorRequest(BaseMessage):
    """Tutor input describing the planner directive and available grounding facts."""

    sender: AgentNode = Field(default=AgentNode.ORCHESTRATOR)
    decision: PlannerDecision = Field(..., description="Planner directive to execute.")
    canonical_facts: list[CanonicalFact] = Field(
        default_factory=list,
        description="Canonical facts fetched by the orchestrator for grounding.",
    )


TutorResponseType = Literal["needs_facts", "socratic_turn"]


class TutorResponse(BaseMessage):
    """Structured tutor output ensuring explicit tool usage and citations."""

    sender: AgentNode = Field(default=AgentNode.TUTOR)
    response_type: TutorResponseType = Field(
        ...,
        description="Indicates whether the tutor performed a turn or requested facts.",
    )
    prompt: str | None = Field(
        default=None,
        description="Socratic question posed to the learner when delivering a turn.",
    )
    explanation: str | None = Field(
        default=None,
        description="Concise explanation scaffolding the learner response.",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Canonical fact identifiers cited in the explanation.",
    )
    requested_fact_ids: list[str] = Field(
        default_factory=list,
        description="Fact identifiers required before the tutor can proceed.",
    )
    candidate_expression: str | None = Field(
        default=None,
        description="Learner-facing derivation result proposed during a derive action.",
    )


class AssessmentRequest(BaseMessage):
    """Assessor input bundling the problem template selection."""

    sender: AgentNode = Field(default=AgentNode.ORCHESTRATOR)
    decision: PlannerDecision = Field(..., description="Planner directive requiring assessment.")


class GradingDetail(BaseModel):
    """Fine-grained grading metadata for the critic and telemetry."""

    rubric_applied: str | None = Field(
        default=None,
        description="Rubric text used during evaluation when applicable.",
    )
    notes: str | None = Field(default=None, description="Assessor free-form notes or hints.")


class AssessmentResult(BaseMessage):
    """Assessor output including instantiated problems and grading outcomes."""

    sender: AgentNode = Field(default=AgentNode.ASSESSOR)
    problem: ProblemInstance = Field(..., description="Instantiated deterministic problem.")
    passed: bool = Field(..., description="Indicates whether the submission met the rubric.")
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized score assigned by the assessor."
    )
    feedback: str = Field(..., description="Feedback shared with the learner.")
    grading: GradingDetail = Field(
        default_factory=GradingDetail,
        description="Supplementary grader metadata for logging and critics.",
    )
    code_run: CodeRunResponse | None = Field(
        default=None,
        description="Code runner output when the assessment executes code.",
    )


class CriticRequest(BaseMessage):
    """Critic input referencing the planner decision and downstream agent output."""

    sender: AgentNode = Field(default=AgentNode.ORCHESTRATOR)
    decision: PlannerDecision = Field(..., description="Planner directive under review.")
    tutor_response: TutorResponse | None = Field(
        default=None,
        description="Tutor output to be audited by the critic when applicable.",
    )
    assessment_result: AssessmentResult | None = Field(
        default=None,
        description="Assessment outcome to audit during quiz or code actions.",
    )


class CriticVerdict(BaseMessage):
    """Critic verdict consolidating math/code checks and faithfulness review."""

    sender: AgentNode = Field(default=AgentNode.CRITIC)
    approved: bool = Field(..., description="True when the critic accepts the turn.")
    issues: list[str] = Field(
        default_factory=list,
        description="Detected issues requiring retries or human escalation.",
    )
    detail: str = Field(..., description="Concise rationale for the approval decision.")


class SessionTurn(BaseModel):
    """Single orchestrated step capturing planner, executor, and critic artefacts."""

    decision: PlannerDecision = Field(..., description="Planner directive for the turn.")
    tutor_response: TutorResponse | None = Field(
        default=None, description="Tutor output when the turn involved tutoring."
    )
    assessment: AssessmentResult | None = Field(
        default=None, description="Assessment outcome when applicable."
    )
    critic_verdict: CriticVerdict = Field(..., description="Final critic verdict for the turn.")


class SessionTranscript(BaseModel):
    """Aggregate transcript for a thin-slice tutoring session."""

    learner: LearnerProfile = Field(..., description="Learner engaging with the session.")
    turns: list[SessionTurn] = Field(
        default_factory=list, description="Ordered collection of completed turns."
    )
    final_state: LearnerState = Field(..., description="Learner state after the session.")
