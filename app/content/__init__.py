"""Content module housing canonical curriculum data for neural networks."""

from .repository import CONTENT_REPOSITORY, ContentRepository
from .schema import (
    CanonicalFact,
    Concept,
    ConceptDetail,
    ContentSection,
    LearningObjective,
    ProblemInstance,
    ProblemTemplate,
    ProblemTemplateMetadata,
    WorkedExample,
)

__all__ = [
    "CanonicalFact",
    "Concept",
    "ConceptDetail",
    "ContentSection",
    "LearningObjective",
    "ProblemInstance",
    "ProblemTemplate",
    "ProblemTemplateMetadata",
    "WorkedExample",
    "CONTENT_REPOSITORY",
    "ContentRepository",
]
