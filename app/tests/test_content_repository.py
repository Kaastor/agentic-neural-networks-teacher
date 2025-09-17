"""Tests for the canonical content repository and templates."""

from __future__ import annotations

import pytest

from app.content import CONTENT_REPOSITORY


def test_concept_detail_includes_required_artifacts() -> None:
    """Backpropagation concept should expose rich content and resources."""

    detail = CONTENT_REPOSITORY.get_concept_detail("concept-backpropagation")
    assert detail is not None
    assert detail.concept.title == "Backpropagation"
    assert len(detail.concept.sections) >= 12
    assert len(detail.learning_objectives) >= 5
    assert len(detail.canonical_facts) >= 5
    assert len(detail.worked_examples) >= 3
    assert len(detail.problem_templates) >= 8

    coding_templates = [
        template
        for template in detail.problem_templates
        if template.metadata.problem_type == "coding"
    ]
    assert len(coding_templates) >= 2


@pytest.mark.parametrize(
    "template_id,variant_id",
    [
        ("pt-linear-mse-gradient", "v1"),
        ("pt-gradient-check", "v1"),
        ("pt-code-two-layer", "v1"),
    ],
)
def test_problem_template_variants_are_deterministic(template_id: str, variant_id: str) -> None:
    """Repeated instantiation should return equal-but-not-identical problem instances."""

    first = CONTENT_REPOSITORY.instantiate_problem(template_id, variant_id)
    second = CONTENT_REPOSITORY.instantiate_problem(template_id, variant_id)

    assert first is not None
    assert second is not None
    assert first == second
    assert first is not second


def test_get_canonical_facts_returns_only_known_ids() -> None:
    """Unknown fact identifiers should be silently ignored to keep the API read-only."""

    facts = CONTENT_REPOSITORY.get_canonical_facts(["fact-chain-rule", "fact-unknown"])
    assert len(facts) == 1
    assert facts[0].id == "fact-chain-rule"
