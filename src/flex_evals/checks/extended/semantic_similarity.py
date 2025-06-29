"""
Semantic Similarity check implementation for FEP.

Computes semantic similarity between two texts using user-provided embedding functions.
"""

import asyncio
from typing import Any
from collections.abc import Callable

from ..base import BaseAsyncCheck
from ...registry import register
from ...exceptions import ValidationError, CheckExecutionError
from ...constants import CheckType, SimilarityMetric


@register(CheckType.SEMANTIC_SIMILARITY, version="1.0.0")
class SemanticSimilarityCheck(BaseAsyncCheck):
    """
    Computes semantic similarity between two texts using embeddings.

    Arguments Schema:
    - text: string | JSONPath - First text to compare
    - reference: string | JSONPath - Second text to compare against
    - threshold: object (optional) - Threshold configuration for pass/fail determination
      - min_value: number (optional) - Minimum similarity score
      - max_value: number (optional) - Maximum similarity score
      - min_inclusive: boolean (default: true) - Whether min threshold is inclusive
      - max_inclusive: boolean (default: true) - Whether max threshold is inclusive
      - negate: boolean (default: false) - Whether to negate the threshold logic
    - embedding_function: async callable - User-provided function to generate embeddings
    - similarity_metric: string (default: 'cosine') - Similarity calculation method

    Results Schema:
    - score: number[0,1] - Similarity score between the texts
    - passed: boolean (optional) - Present only when threshold is provided
    """

    async def __call__(
        self,
        text: str,
        reference: str,
        embedding_function: Callable,
        threshold: dict | None = None,
        similarity_metric: str = 'cosine',
    ) -> dict[str, Any]:
        """Execute semantic similarity check with direct arguments."""
# Validate embedding function is callable
        if not callable(embedding_function):
            raise ValidationError("embedding_function must be callable")

        # Validate similarity metric
        valid_metrics = [member.value for member in SimilarityMetric]
        if similarity_metric not in valid_metrics:
            raise ValidationError(f"Unsupported similarity metric: {similarity_metric}. Valid options: {valid_metrics}")  # noqa: E501

        # Convert to strings for embedding
        text_str = str(text) if text is not None else ""
        reference_str = str(reference) if reference is not None else ""

        try:
            # Get embeddings for both texts
            text_embedding = await self._call_embedding_function(embedding_function, text_str)
            reference_embedding = await self._call_embedding_function(
                embedding_function, reference_str,
            )

            # Calculate similarity score
            score = self._calculate_similarity(
                text_embedding, reference_embedding, similarity_metric,
            )

            # Ensure score is in valid range [0, 1]
            score = max(0.0, min(1.0, score))

            # Prepare result
            result = {"score": score}

            # Apply threshold if provided
            if threshold is not None:
                passed = self._evaluate_threshold(score, threshold)
                result["passed"] = passed

            return result

        except Exception as e:
            raise CheckExecutionError(
                f"Error in semantic similarity computation: {e!s}",
            ) from e

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
            raise CheckExecutionError(
                f"Embedding function failed: {e!s}",
            ) from e

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

    def _evaluate_threshold(self, score: float, threshold: dict[str, Any]) -> bool:
        """Evaluate whether score passes the threshold criteria."""
        min_value = threshold.get("min_value")
        max_value = threshold.get("max_value")
        min_inclusive = threshold.get("min_inclusive", True)
        max_inclusive = threshold.get("max_inclusive", True)
        negate = threshold.get("negate", False)

        passed = True

        # Check minimum threshold
        if min_value is not None:
            if min_inclusive:
                passed = passed and score >= min_value
            else:
                passed = passed and score > min_value

        # Check maximum threshold
        if max_value is not None:
            if max_inclusive:
                passed = passed and score <= max_value
            else:
                passed = passed and score < max_value

        # Apply negation
        if negate:
            passed = not passed

        return passed
