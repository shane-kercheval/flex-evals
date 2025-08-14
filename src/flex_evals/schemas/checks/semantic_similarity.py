"""SemanticSimilarityCheck schema class for type-safe semantic similarity check definitions."""

from typing import ClassVar
from collections.abc import Callable

from pydantic import BaseModel, Field

from ...constants import CheckType, SimilarityMetric
from ..check import Check, SchemaCheck, OptionalJSONPath


class ThresholdConfig(BaseModel):
    """Threshold configuration for semantic similarity check."""

    min_value: float | None = Field(None, description="Minimum similarity score")
    max_value: float | None = Field(None, description="Maximum similarity score")
    min_inclusive: bool = Field(True, description="Whether min threshold is inclusive")
    max_inclusive: bool = Field(True, description="Whether max threshold is inclusive")
    negate: bool = Field(False, description="Whether to negate the threshold logic")


class SemanticSimilarityCheck(SchemaCheck):
    """Computes semantic similarity between two texts using embeddings."""

    VERSION: ClassVar[str] = "1.0.0"
    CHECK_TYPE: ClassVar[CheckType] = CheckType.SEMANTIC_SIMILARITY

    text: str = OptionalJSONPath(
        "first text to compare or JSONPath expression pointing to the text",
    )
    reference: str = OptionalJSONPath(
        "second text to compare against or JSONPath expression pointing to the text",
    )
    embedding_function: Callable = Field(..., description="User-provided function to generate embeddings")  # noqa: E501
    similarity_metric: SimilarityMetric = Field(SimilarityMetric.COSINE, description="Similarity calculation method")  # noqa: E501
    threshold: ThresholdConfig | None = Field(None, description="Optional threshold configuration for pass/fail determination")  # noqa: E501

    model_config = {"arbitrary_types_allowed": True}  # noqa: RUF012

    def to_check(self) -> Check:
        """Convert to generic Check object for execution."""
        arguments = {
            "text": self.text,
            "reference": self.reference,
            "embedding_function": self.embedding_function,
            "similarity_metric": self.similarity_metric.value,
        }

        if self.threshold is not None:
            arguments["threshold"] = self.threshold.model_dump(exclude_none=True)

        return Check(
            type=self.check_type,
            arguments=arguments,
            version=self.VERSION,
        )
