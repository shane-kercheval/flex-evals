"""
Combined SemanticSimilarityCheck implementation for FEP.

Combines schema validation with execution logic in a single class.
"""

import asyncio
from typing import Any
from collections.abc import Callable
from pydantic import Field, BaseModel, field_validator

from .base import BaseAsyncCheck, JSONPath, _convert_to_jsonpath
from ..registry import register
from ..exceptions import ValidationError, CheckExecutionError
from ..constants import CheckType, SimilarityMetric


class ThresholdConfig(BaseModel):
    """Threshold configuration for similarity score evaluation."""

    min_value: float | None = Field(None, description='Minimum similarity score')
    max_value: float | None = Field(None, description='Maximum similarity score')
    min_inclusive: bool = Field(True, description='Whether min threshold is inclusive')
    max_inclusive: bool = Field(True, description='Whether max threshold is inclusive')
    negate: bool = Field(False, description='Whether to negate the threshold logic')


@register(CheckType.SEMANTIC_SIMILARITY, version='1.0.0')
class SemanticSimilarityCheck(BaseAsyncCheck):
    """Computes semantic similarity between two texts using embeddings."""

    # Pydantic fields with validation - can be literals or JSONPath objects
    text: str | JSONPath = Field(
        ...,
        description="First text to compare or JSONPath expression",
    )
    reference: str | JSONPath = Field(
        ...,
        description="Second text to compare against or JSONPath expression",
    )
    threshold: ThresholdConfig | JSONPath | None = Field(
        None,
        description="Threshold configuration for pass/fail determination",
    )
    embedding_function: Any = Field(
        ...,
        description="User-provided function to generate embeddings",
    )
    similarity_metric: SimilarityMetric | JSONPath = Field(
        SimilarityMetric.COSINE,
        description="Similarity calculation method",
    )

    @field_validator('text', 'reference', 'threshold', 'similarity_metric', mode='before')
    @classmethod
    def convert_jsonpath(cls, value: object) -> object | JSONPath:
        """Convert JSONPath-like strings to JSONPath objects."""
        return _convert_to_jsonpath(value)

    @field_validator('embedding_function')
    @classmethod
    def validate_embedding_function(cls, v: Any) -> Any:  # noqa: ANN401
        """Validate that embedding_function is callable."""
        if not callable(v):
            raise ValueError("embedding_function must be callable")
        return v

    async def __call__(self) -> dict[str, Any]:  # noqa: PLR0912
        """
        Execute semantic similarity check using resolved Pydantic fields.

        All JSONPath objects should have been resolved by execute() before this is called.

        Returns:
            Dictionary with 'score' key and optional 'passed' key

        Raises:
            RuntimeError: If any field contains unresolved JSONPath objects
        """
        # Validate that all fields are resolved (no JSONPath objects remain)
        if isinstance(self.text, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'text' field: {self.text}")
        if isinstance(self.reference, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'reference' field: {self.reference}")
        if isinstance(self.threshold, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'threshold' field: {self.threshold}")
        if isinstance(self.similarity_metric, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'similarity_metric' field: {self.similarity_metric}")  # noqa: E501

        # Validate embedding function is callable
        if not callable(self.embedding_function):
            raise ValidationError("embedding_function must be callable")

        # Handle similarity metric conversion
        if isinstance(self.similarity_metric, str):
            try:
                similarity_metric = SimilarityMetric(self.similarity_metric)
            except ValueError:
                valid_metrics = [m.value for m in SimilarityMetric]
                raise ValidationError(f"Unsupported similarity metric: {self.similarity_metric}. Valid options: {valid_metrics}")  # noqa: E501
        else:
            similarity_metric = self.similarity_metric

        # Handle threshold conversion
        threshold_obj = None
        if self.threshold is not None:
            if isinstance(self.threshold, dict):
                threshold_obj = ThresholdConfig(**self.threshold)
            elif isinstance(self.threshold, ThresholdConfig):
                threshold_obj = self.threshold
            else:
                raise ValidationError("threshold must be a dict or ThresholdConfig")

        # Convert to strings for embedding
        text_str = str(self.text) if self.text is not None else ""
        reference_str = str(self.reference) if self.reference is not None else ""

        try:
            # Get embeddings for both texts
            text_embedding = await self._call_embedding_function(self.embedding_function, text_str)
            reference_embedding = await self._call_embedding_function(self.embedding_function, reference_str)  # noqa: E501

            # Calculate similarity score
            score = self._calculate_similarity(text_embedding, reference_embedding, similarity_metric.value)  # noqa: E501

            # Ensure score is in valid range [0, 1]
            score = max(0.0, min(1.0, score))

            # Prepare result
            result = {'score': score}

            # Apply threshold if provided
            if threshold_obj is not None:
                passed = self._evaluate_threshold(score, threshold_obj)
                result['passed'] = passed

            return result

        except Exception as e:
            raise CheckExecutionError(f"Error in semantic similarity computation: {e!s}") from e

    async def _call_embedding_function(
            self,
            embedding_function: Callable,
            text: str,
        ) -> list[float]:
        """Call the user-provided embedding function with error handling."""
        try:
            # Handle both sync and async embedding functions
            if asyncio.iscoroutinefunction(embedding_function):
                result = await embedding_function(text)
            else:
                result = embedding_function(text)

            # Validate result is a list of numbers
            if not isinstance(result, list | tuple):
                raise ValueError("Embedding function must return a list or tuple of numbers")

            embedding = [float(x) for x in result]
            if len(embedding) == 0:
                raise ValueError("Embedding function returned empty vector")

            return embedding

        except Exception as e:
            raise CheckExecutionError(f"Embedding function failed: {e!s}") from e

    def _calculate_similarity(
            self,
            embedding1: list[float],
            embedding2: list[float],
            metric: str,
        ) -> float:
        """Calculate similarity between two embeddings."""
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same dimensionality")

        if metric == 'cosine':
            return self._cosine_similarity(embedding1, embedding2)
        if metric == 'dot':
            return self._dot_product_similarity(embedding1, embedding2)
        if metric == 'euclidean':
            return self._euclidean_similarity(embedding1, embedding2)
        raise ValueError(f"Unsupported similarity metric: {metric}")

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _dot_product_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate normalized dot product similarity."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        # Normalize to [0, 1] range
        max_possible = (sum(a * a for a in vec1) * sum(b * b for b in vec2)) ** 0.5
        if max_possible == 0:
            return 0.0
        return max(0.0, dot_product / max_possible)

    def _euclidean_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate similarity based on euclidean distance (converted to [0,1] range)."""
        distance = sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5
        # Convert distance to similarity in [0, 1] range
        # Using 1 / (1 + distance) formula
        return 1.0 / (1.0 + distance)

    def _evaluate_threshold(self, score: float, threshold: ThresholdConfig) -> bool:
        """Evaluate whether score passes the threshold criteria."""
        passed = True

        # Check minimum threshold
        if threshold.min_value is not None:
            if threshold.min_inclusive:
                passed = passed and score >= threshold.min_value
            else:
                passed = passed and score > threshold.min_value

        # Check maximum threshold
        if threshold.max_value is not None:
            if threshold.max_inclusive:
                passed = passed and score <= threshold.max_value
            else:
                passed = passed and score < threshold.max_value

        # Apply negation
        if threshold.negate:
            passed = not passed

        return passed
