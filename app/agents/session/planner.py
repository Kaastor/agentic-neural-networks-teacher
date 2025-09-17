"""Planner agent responsible for sequencing actions inside the session graph."""

from __future__ import annotations

from dataclasses import dataclass

from app.content import CONTENT_REPOSITORY, ContentRepository

from .messages import DerivationTarget, PlannerDecision, PlannerRequest, SessionAction


@dataclass(frozen=True)
class _PlannedStep:
    """Immutable configuration describing a planner step."""

    action: SessionAction
    learning_objective_id: str
    canonical_fact_ids: tuple[str, ...]
    prompt: str
    problem_template_id: str | None = None
    variant_id: str | None = None
    derivation_target: DerivationTarget | None = None


class PlannerAgent:
    """Deterministic planner implementing the milestone thin-slice sequence."""

    def __init__(
        self,
        *,
        content_repository: ContentRepository = CONTENT_REPOSITORY,
        concept_id: str = "concept-backpropagation",
    ) -> None:
        self._repository = content_repository
        if self._repository.get_concept(concept_id) is None:
            raise ValueError(f"Concept '{concept_id}' not found in repository.")
        self._steps: list[_PlannedStep] = [
            _PlannedStep(
                action="explain",
                learning_objective_id="lo-bp-jacobian-intuition",
                canonical_fact_ids=("fact-chain-rule",),
                prompt=(
                    "Guide the learner through why vector-Jacobian products make reverse-mode automatic differentiation efficient."
                ),
            ),
            _PlannedStep(
                action="derive",
                learning_objective_id="lo-bp-derive-two-layer",
                canonical_fact_ids=("fact-chain-rule",),
                prompt=(
                    "Work with the learner to derive the gradient of a scalar MSE loss with respect to the prediction."
                ),
                derivation_target=DerivationTarget(
                    canonical_expression="y - t", symbols=["y", "t"]
                ),
            ),
            _PlannedStep(
                action="quiz",
                learning_objective_id="lo-bp-derive-two-layer",
                canonical_fact_ids=("fact-chain-rule",),
                prompt="Assess whether the learner can restate the linear model gradient derivation.",
                problem_template_id="pt-linear-mse-gradient",
                variant_id="v1",
            ),
            _PlannedStep(
                action="code",
                learning_objective_id="lo-bp-gradient-check",
                canonical_fact_ids=("fact-gradient-check", "fact-delta-recursion"),
                prompt=(
                    "Validate an implementation of the two-layer MLP backward pass with a gradient check."
                ),
                problem_template_id="pt-code-two-layer",
                variant_id="v1",
            ),
        ]

    def decide(self, request: PlannerRequest) -> PlannerDecision | None:
        """Return the next planner decision or None when the session is complete."""

        completed = len(request.state.completed_actions)
        if completed >= len(self._steps):
            return None

        step = self._steps[completed]
        return PlannerDecision(
            action=step.action,
            learning_objective_id=step.learning_objective_id,
            canonical_fact_ids=list(step.canonical_fact_ids),
            prompt=step.prompt,
            problem_template_id=step.problem_template_id,
            variant_id=step.variant_id,
            derivation_target=step.derivation_target,
        )
