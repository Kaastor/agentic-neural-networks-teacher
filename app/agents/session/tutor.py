"""Tutor agent implementing a Socratic, fact-grounded dialogue."""

from __future__ import annotations

from app.content.schema import CanonicalFact

from .messages import TutorRequest, TutorResponse


class TutorAgent:
    """Deterministic tutor that references canonical facts before responding."""

    def respond(self, request: TutorRequest) -> TutorResponse:
        """Return a Socratic tutor response or request additional canonical facts."""

        decision = request.decision
        required_facts = set(decision.canonical_fact_ids)
        provided_fact_ids = {fact.id for fact in request.canonical_facts}

        missing_facts = sorted(required_facts - provided_fact_ids)
        if missing_facts:
            return TutorResponse(
                response_type="needs_facts",
                requested_fact_ids=missing_facts,
            )

        if decision.action == "explain":
            fact = request.canonical_facts[0]
            prompt = (
                "Looking at the multivariable chain rule, how do vector-Jacobian products allow us "
                "to reuse intermediate computations during the backward pass?"
            )
            explanation = _summarize_fact(fact)
            return TutorResponse(
                response_type="socratic_turn",
                prompt=prompt,
                explanation=explanation,
                citations=[fact.id],
            )

        if decision.action == "derive":
            fact = request.canonical_facts[0]
            prompt = (
                "If L = 0.5 * (y - t)^2, what gradient with respect to y does the chain rule imply?"
            )
            explanation = (
                "The derivative follows directly from the scalar chain rule: the loss sensitivity to the "
                "prediction equals the prediction error."
            )
            return TutorResponse(
                response_type="socratic_turn",
                prompt=prompt,
                explanation=explanation,
                citations=[fact.id],
                candidate_expression="y - t",
            )

        raise ValueError(f"Unsupported tutor action: {decision.action}")


def _summarize_fact(fact: CanonicalFact) -> str:
    """Return a concise restatement of a canonical fact for the learner."""

    return (
        f"{fact.title}: {fact.statement} This relationship is what lets us propagate gradients "
        "by composing local Jacobians."
    )
