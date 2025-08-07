"""
Constants and enums for flex-evals.

This module defines enums for commonly used string constants throughout the
codebase. All enums inherit from str to ensure they serialize properly to JSON
and maintain compatibility with the FEP protocol.
"""

from enum import Enum


class CheckType(str, Enum):
    """
    Standard check types available in flex-evals.

    These enums can be used interchangeably with their string values.
    Custom check types can still be registered using arbitrary strings.
    """

    ATTRIBUTE_EXISTS = 'attribute_exists'
    CONTAINS = 'contains'
    EXACT_MATCH = 'exact_match'
    IS_EMPTY = 'is_empty'
    REGEX = 'regex'
    THRESHOLD = 'threshold'
    SEMANTIC_SIMILARITY = 'semantic_similarity'
    LLM_JUDGE = 'llm_judge'
    CUSTOM_FUNCTION = 'custom_function'

    def __str__(self) -> str:
        """Return the enum value as string."""
        return str(self.value)


class Status(str, Enum):
    """Evaluation and check result status values."""

    COMPLETED = 'completed'
    ERROR = 'error'
    SKIP = 'skip'

    def __str__(self) -> str:
        """Return the enum value as string."""
        return str(self.value)


class ErrorType(str, Enum):
    """Check error types for detailed error reporting."""

    JSONPATH_ERROR = 'jsonpath_error'
    VALIDATION_ERROR = 'validation_error'
    TIMEOUT_ERROR = 'timeout_error'
    UNKNOWN_ERROR = 'unknown_error'

    def __str__(self) -> str:
        """Return the enum value as string."""
        return str(self.value)


class SimilarityMetric(str, Enum):
    """Similarity calculation methods for semantic similarity checks."""

    COSINE = 'cosine'
    DOT = 'dot'
    EUCLIDEAN = 'euclidean'

    def __str__(self) -> str:
        """Return the enum value as string."""
        return str(self.value)
