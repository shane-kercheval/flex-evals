"""
Combined ExactMatch check implementation for FEP.

Combines schema validation with execution logic in a single class.
"""

from typing import Any
from pydantic import field_validator

from .base import BaseCheck, JSONPath, _convert_to_jsonpath
from pydantic import Field
from ..registry import register
from ..constants import CheckType


@register(CheckType.EXACT_MATCH, version='1.0.0')
class ExactMatchCheck(BaseCheck):
    """Compares two text values for exact equality."""

    # Pydantic fields with validation - can be literals or JSONPath objects
    actual: Any | JSONPath = Field(
        ..., description="Value to check or JSONPath expression pointing to the value",
    )
    expected: Any | JSONPath = Field(
        ..., description="Expected value or JSONPath expression pointing to the value",
    )
    case_sensitive: bool | JSONPath = Field(
        True, description="Whether string comparison is case-sensitive",
    )
    negate: bool | JSONPath = Field(
        False, description="If true, passes when values don't match",
    )

    @field_validator('actual', 'expected', 'case_sensitive', 'negate', mode='before')
    @classmethod
    def convert_jsonpath(cls, value: object) -> object | JSONPath:
        """Convert JSONPath-like strings to JSONPath objects."""
        return _convert_to_jsonpath(value)

    def __call__(self) -> dict[str, Any]:
        """
        Execute exact match check using resolved Pydantic fields.

        All JSONPath objects should have been resolved by execute() before this is called.

        Returns:
            Dictionary with 'passed' key indicating check result

        Raises:
            RuntimeError: If any field contains unresolved JSONPath objects
        """
        # Validate that all fields are resolved (no JSONPath objects remain)
        if isinstance(self.actual, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'actual' field: {self.actual}")
        if isinstance(self.expected, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'expected' field: {self.expected}")
        if isinstance(self.case_sensitive, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'case_sensitive' field: {self.case_sensitive}")  # noqa: E501
        if isinstance(self.negate, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'negate' field: {self.negate}")

        # Convert to strings for comparison
        actual_str = str(self.actual) if self.actual is not None else ''
        expected_str = str(self.expected) if self.expected is not None else ''

        # Apply case sensitivity
        if not self.case_sensitive:
            actual_str = actual_str.lower()
            expected_str = expected_str.lower()

        # Perform comparison
        match = actual_str == expected_str

        # Apply negation
        passed = not match if self.negate else match

        return {'passed': passed}
