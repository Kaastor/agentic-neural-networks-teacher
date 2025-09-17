"""Session orchestrator wiring planner, tutor, assessor, and critic agents."""

from __future__ import annotations

from app.content import CONTENT_REPOSITORY, ContentRepository

from .assessor import AssessorAgent
from .critic import CriticAgent
from .messages import (
    AssessmentRequest,
    CriticRequest,
    LearnerProfile,
    LearnerState,
    PlannerDecision,
    PlannerRequest,
    SessionTranscript,
    SessionTurn,
    TutorRequest,
    TutorResponse,
)
from .planner import PlannerAgent
from .tutor import TutorAgent


class SessionOrchestrator:
    """Deterministic orchestrator executing the milestone thin-slice flow."""

    def __init__(
        self,
        *,
        content_repository: ContentRepository = CONTENT_REPOSITORY,
        planner: PlannerAgent | None = None,
        tutor: TutorAgent | None = None,
        assessor: AssessorAgent | None = None,
        critic: CriticAgent | None = None,
    ) -> None:
        self._repository = content_repository
        self._planner = planner or PlannerAgent(content_repository=content_repository)
        self._tutor = tutor or TutorAgent()
        self._assessor = assessor or AssessorAgent(content_repository=content_repository)
        self._critic = critic or CriticAgent(content_repository=content_repository)

    def run_thin_slice(self, learner: LearnerProfile) -> SessionTranscript:
        """Execute the milestone three thin-slice tutoring session."""

        state = LearnerState(concept_id=learner.target_concept_id)
        turns: list[SessionTurn] = []

        while True:
            planner_request = PlannerRequest(learner=learner, state=state)
            decision = self._planner.decide(planner_request)
            if decision is None:
                break

            tutor_response = None
            assessment_result = None

            if decision.action in {"explain", "derive"}:
                tutor_response = self._run_tutor(decision, state)
            elif decision.action in {"quiz", "code"}:
                assessment_request = AssessmentRequest(decision=decision)
                assessment_result = self._assessor.evaluate(assessment_request)
            else:  # pragma: no cover - defensive
                raise ValueError(f"Unsupported action '{decision.action}'.")

            critic_request = CriticRequest(
                decision=decision,
                tutor_response=tutor_response,
                assessment_result=assessment_result,
            )
            critic_verdict = self._critic.review(critic_request)
            if not critic_verdict.approved:
                issues = ", ".join(critic_verdict.issues)
                raise RuntimeError(f"Critic rejected turn '{decision.action}': {issues}")

            state.record_action(decision.action)
            turns.append(
                SessionTurn(
                    decision=decision,
                    tutor_response=tutor_response,
                    assessment=assessment_result,
                    critic_verdict=critic_verdict,
                )
            )

        return SessionTranscript(learner=learner, turns=turns, final_state=state)

    def _run_tutor(self, decision: PlannerDecision, state: LearnerState) -> TutorResponse:
        required_ids = decision.canonical_fact_ids
        cached_facts = [
            state.retrieved_facts[fact_id]
            for fact_id in required_ids
            if fact_id in state.retrieved_facts
        ]
        tutor_request = TutorRequest(decision=decision, canonical_facts=cached_facts)
        response = self._tutor.respond(tutor_request)
        if response.response_type == "needs_facts":
            fact_ids = response.requested_fact_ids or required_ids
            fetched = self._repository.get_canonical_facts(fact_ids)
            state.cache_facts(fetched)
            all_facts = [
                state.retrieved_facts[fact_id]
                for fact_id in required_ids
                if fact_id in state.retrieved_facts
            ]
            response = self._tutor.respond(
                TutorRequest(decision=decision, canonical_facts=all_facts)
            )
        else:
            state.cache_facts(cached_facts)

        if response.response_type == "needs_facts":
            missing = response.requested_fact_ids or required_ids
            raise RuntimeError(
                "Tutor continued to request canonical facts after fetch cycle: "
                + ", ".join(missing)
            )
        return response
