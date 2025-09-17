"""Integration tests for the milestone three agent graph thin slice."""

from __future__ import annotations

import re

from app.agents.session import (
    LearnerProfile,
    LearnerState,
    PlannerAgent,
    PlannerRequest,
    SessionOrchestrator,
    TutorAgent,
    TutorRequest,
)


def _learner_profile() -> LearnerProfile:
    return LearnerProfile(
        learner_id="learner-test",
        name="Test Learner",
        target_concept_id="concept-backpropagation",
    )


def test_planner_sequence_for_thin_slice() -> None:
    """Planner should emit the expected milestone sequence before terminating."""

    planner = PlannerAgent()
    profile = _learner_profile()
    state = LearnerState(concept_id=profile.target_concept_id)

    actions: list[str] = []
    while True:
        decision = planner.decide(PlannerRequest(learner=profile, state=state))
        if decision is None:
            break
        actions.append(decision.action)
        state.record_action(decision.action)

    assert actions == ["explain", "derive", "quiz", "code"]


def test_tutor_requests_canonical_facts_before_responding() -> None:
    """Tutor must request canonical facts before producing a Socratic turn."""

    planner = PlannerAgent()
    profile = _learner_profile()
    state = LearnerState(concept_id=profile.target_concept_id)
    decision = planner.decide(PlannerRequest(learner=profile, state=state))
    assert decision is not None

    tutor = TutorAgent()
    response = tutor.respond(TutorRequest(decision=decision, canonical_facts=[]))

    assert response.response_type == "needs_facts"
    assert "fact-chain-rule" in response.requested_fact_ids


def test_session_orchestrator_generates_verified_transcript() -> None:
    """End-to-end thin slice should yield four approved turns with verified tooling."""

    orchestrator = SessionOrchestrator()
    profile = _learner_profile()

    transcript = orchestrator.run_thin_slice(profile)

    assert transcript.learner.learner_id == profile.learner_id
    assert len(transcript.turns) == 4
    assert transcript.final_state.completed_actions == [
        "explain",
        "derive",
        "quiz",
        "code",
    ]

    actions = [turn.decision.action for turn in transcript.turns]
    assert actions == ["explain", "derive", "quiz", "code"]

    for turn in transcript.turns:
        assert turn.critic_verdict.approved, turn.critic_verdict.issues

    derive_turn = transcript.turns[1]
    assert derive_turn.tutor_response is not None
    assert derive_turn.tutor_response.candidate_expression == "y - t"

    quiz_turn = transcript.turns[2]
    assert quiz_turn.assessment is not None
    assert quiz_turn.assessment.passed
    assert quiz_turn.assessment.score == 1.0

    code_turn = transcript.turns[3]
    assert code_turn.assessment is not None
    run = code_turn.assessment.code_run
    assert run is not None
    assert run.exit_code == 0
    assert "MAX_REL_ERROR" in run.stdout
    match = re.search(r"MAX_REL_ERROR=([0-9eE.+-]+)", run.stdout)
    assert match is not None
    assert float(match.group(1)) < 1e-4

    # Tutor facts should be cached for reuse in later turns.
    assert "fact-chain-rule" in transcript.final_state.retrieved_facts

