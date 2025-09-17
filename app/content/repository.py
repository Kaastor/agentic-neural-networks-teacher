"""In-memory repository providing access to canonical curriculum content."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from app.content.backprop_module import (
    CANONICAL_FACTS,
    CONCEPT,
    LEARNING_OBJECTIVES,
    PROBLEM_TEMPLATES,
    WORKED_EXAMPLES,
)
from app.content.schema import (
    CanonicalFact,
    Concept,
    ConceptDetail,
    LearningObjective,
    ProblemInstance,
    ProblemTemplate,
    WorkedExample,
)


class ContentRepository:
    """Read-only repository backed by static canonical content."""

    def __init__(self) -> None:
        self._concepts: dict[str, Concept] = {CONCEPT.id: CONCEPT}
        self._learning_objectives: dict[str, LearningObjective] = {
            lo.id: lo for lo in LEARNING_OBJECTIVES
        }
        self._facts: dict[str, CanonicalFact] = {fact.id: fact for fact in CANONICAL_FACTS}
        self._worked_examples: dict[str, WorkedExample] = {
            example.id: example for example in WORKED_EXAMPLES
        }
        self._problem_templates: dict[str, ProblemTemplate] = {
            template.metadata.id: template for template in PROBLEM_TEMPLATES
        }

    # Concept operations -------------------------------------------------

    def list_concepts(self) -> list[Concept]:
        """Return a shallow copy of all concept metadata."""

        return [concept.model_copy(deep=True) for concept in self._concepts.values()]

    def get_concept(self, concept_id: str) -> Concept | None:
        """Fetch concept metadata by identifier."""

        concept = self._concepts.get(concept_id)
        return concept.model_copy(deep=True) if concept else None

    def get_concept_detail(self, concept_id: str) -> ConceptDetail | None:
        """Return fully expanded concept detail with associated resources."""

        concept = self._concepts.get(concept_id)
        if concept is None:
            return None
        learning_objectives = [
            self._learning_objectives[lo_id].model_copy(deep=True)
            for lo_id in concept.primary_objectives
            if lo_id in self._learning_objectives
        ]
        canonical_facts = [
            self._facts[fact_id].model_copy(deep=True)
            for fact_id in concept.canonical_fact_ids
            if fact_id in self._facts
        ]
        worked_examples = [
            self._worked_examples[example_id].model_copy(deep=True)
            for example_id in concept.worked_example_ids
            if example_id in self._worked_examples
        ]
        problem_templates = [
            self._problem_templates[template_id].model_copy(deep=True)
            for template_id in concept.problem_template_ids
            if template_id in self._problem_templates
        ]
        return ConceptDetail(
            concept=concept.model_copy(deep=True),
            learning_objectives=learning_objectives,
            canonical_facts=canonical_facts,
            worked_examples=worked_examples,
            problem_templates=problem_templates,
        )

    # Canonical facts ----------------------------------------------------

    def get_canonical_facts(self, fact_ids: Iterable[str]) -> list[CanonicalFact]:
        """Return canonical facts for the provided identifiers."""

        results: list[CanonicalFact] = []
        for fact_id in fact_ids:
            fact = self._facts.get(fact_id)
            if fact is not None:
                results.append(fact.model_copy(deep=True))
        return results

    # Problem templates --------------------------------------------------

    def get_problem_template(self, template_id: str) -> ProblemTemplate | None:
        """Fetch a problem template by identifier."""

        template = self._problem_templates.get(template_id)
        return template.model_copy(deep=True) if template else None

    def instantiate_problem(self, template_id: str, variant_id: str) -> ProblemInstance | None:
        """Instantiate a deterministic problem variant."""

        template = self._problem_templates.get(template_id)
        if template is None:
            return None
        try:
            return template.instantiate(variant_id)
        except KeyError:
            return None

    # Debug helpers ------------------------------------------------------

    def _snapshot(self) -> Mapping[str, object]:  # pragma: no cover - internal use only
        """Expose raw dictionaries to aid debugging and future persistence layers."""

        return {
            "concepts": self._concepts,
            "learning_objectives": self._learning_objectives,
            "facts": self._facts,
            "worked_examples": self._worked_examples,
            "problem_templates": self._problem_templates,
        }


CONTENT_REPOSITORY = ContentRepository()
"""Default in-memory repository instance used by application services."""
