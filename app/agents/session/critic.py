"""Critic agent validating tutor turns, assessments, and tooling results."""

from __future__ import annotations

from app.content import CONTENT_REPOSITORY, ContentRepository
from app.tooling.code_runner.models import CodeRunResponse
from app.tooling.math.checks import check_symbolic_equality
from app.tooling.math.models import EqualityCheckRequest

from .messages import CriticRequest, CriticVerdict, PlannerDecision, TutorResponse


class CriticAgent:
    """Critic enforcing fact grounding and math/code verification."""

    def __init__(self, *, content_repository: ContentRepository = CONTENT_REPOSITORY) -> None:
        self._repository = content_repository

    def review(self, request: CriticRequest) -> CriticVerdict:
        """Audit the downstream agent output and return a structured verdict."""

        decision = request.decision
        issues: list[str] = []

        if decision.action in {"explain", "derive"}:
            tutor = request.tutor_response
            if tutor is None:
                issues.append("Tutor response missing for planner directive.")
            else:
                issues.extend(self._validate_tutor(decision.canonical_fact_ids, tutor))
                if decision.action == "derive":
                    issues.extend(self._validate_derivation(decision, tutor))
        elif decision.action == "quiz":
            assessment = request.assessment_result
            if assessment is None:
                issues.append("Assessment result missing for quiz action.")
            elif not assessment.passed:
                issues.append("Assessment did not pass canonical rubric.")
        elif decision.action == "code":
            assessment = request.assessment_result
            if assessment is None:
                issues.append("Assessment result missing for code action.")
            else:
                issues.extend(self._validate_code_run(assessment.code_run))
        else:  # pragma: no cover - defensive
            issues.append(f"Unsupported planner action '{decision.action}' for critic review.")

        if issues:
            detail = "Critic detected issues requiring correction."
            return CriticVerdict(approved=False, issues=issues, detail=detail)

        detail = f"Turn for action '{decision.action}' approved by critic."
        return CriticVerdict(approved=True, issues=[], detail=detail)

    def _validate_tutor(
        self, required_fact_ids: list[str], tutor_response: TutorResponse
    ) -> list[str]:
        issues: list[str] = []
        citations = set(tutor_response.citations)
        required = set(required_fact_ids)
        missing = required - citations
        if missing:
            issues.append(
                "Tutor response omitted required canonical fact citations: "
                + ", ".join(sorted(missing))
            )
        if tutor_response.prompt is None or not tutor_response.prompt.endswith("?"):
            issues.append("Tutor prompt must end with a Socratic question.")
        if tutor_response.response_type != "socratic_turn":
            issues.append("Tutor must deliver a Socratic turn after facts are provided.")
        available_fact_ids = {fact.id for fact in self._repository.get_canonical_facts(required)}
        unknown = citations - available_fact_ids
        if unknown:
            issues.append("Tutor cited unknown canonical facts: " + ", ".join(sorted(unknown)))
        if not tutor_response.explanation:
            issues.append("Tutor explanation is required for pedagogical scaffolding.")
        return issues

    def _validate_derivation(
        self, decision: PlannerDecision, tutor_response: TutorResponse
    ) -> list[str]:
        issues: list[str] = []
        target = decision.derivation_target
        if target is None:
            issues.append("Planner decision missing derivation target for symbolic check.")
            return issues
        candidate = tutor_response.candidate_expression
        if not candidate:
            issues.append("Tutor response missing candidate expression for derivation.")
            return issues
        check = check_symbolic_equality(
            EqualityCheckRequest(
                canonical=target.canonical_expression,
                candidate=candidate,
                symbols=target.symbols,
            )
        )
        if not check.equivalent:
            issues.append(f"Symbolic equality check failed: {check.detail}")
        return issues

    def _validate_code_run(self, code_run: CodeRunResponse | None) -> list[str]:
        issues: list[str] = []
        if code_run is None:
            issues.append("Code runner response missing for coding assessment.")
            return issues
        if code_run.timed_out:
            issues.append("Code execution exceeded the configured timeout.")
        if code_run.exit_code is None:
            issues.append("Code execution was terminated before completion.")
        elif code_run.exit_code != 0:
            issues.append(f"Code execution failed with exit code {code_run.exit_code}.")
        if "MAX_REL_ERROR" not in code_run.stdout:
            issues.append("Gradient check summary missing from code execution output.")
        return issues
