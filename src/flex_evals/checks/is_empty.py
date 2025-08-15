"""
Combined IsEmptyCheck implementation for FEP.

Combines schema validation with execution logic in a single class.
"""

from typing import Any
from pydantic import field_validator, Field

from .base import BaseCheck, JSONPath, _convert_to_jsonpath
from ..registry import register
from ..constants import CheckType


@register(CheckType.IS_EMPTY, version='1.0.0')
class IsEmptyCheck(BaseCheck):
    """Tests whether a value is empty (None, empty string, whitespace-only, or any empty collection that supports len())."""  # noqa: E501

    # Pydantic fields with validation - can be literals or JSONPath objects
    value: str | list | dict | set | tuple | int | float | bool | None | JSONPath = Field(
        ...,
        description="Value to test for emptiness or JSONPath expression",
    )
    negate: bool | JSONPath = Field(
        False,
        description="If true, passes when value is not empty",
    )
    strip_whitespace: bool | JSONPath = Field(
        True,
        description="If true, strips whitespace before checking (strings only)",
    )

    @field_validator('value', 'negate', 'strip_whitespace', mode='before')
    @classmethod
    def convert_jsonpath(cls, value: object) -> object | JSONPath:
        """Convert JSONPath-like strings to JSONPath objects."""
        return _convert_to_jsonpath(value)

    def __call__(self) -> dict[str, Any]:
        """
        Execute emptiness check with resolved arguments.

        Args:
            value: Resolved value to test for emptiness
            negate: If true, passes when value is not empty
            strip_whitespace: If true, strips whitespace before checking (strings only)

        Returns:
            Dictionary with 'passed' key indicating check result
        """
        # Validate that all fields are resolved (no JSONPath objects remain)
        if isinstance(self.value, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'value' field: {self.value}")
        if isinstance(self.negate, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'negate' field: {self.negate}")
        if isinstance(self.strip_whitespace, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'strip_whitespace' field: {self.strip_whitespace}")  # noqa: E501

        # Handle None directly
        if self.value is None:
            is_empty = True
        # Handle strings with optional whitespace stripping
        elif isinstance(self.value, str):
            is_empty = self.value.strip() == "" if self.strip_whitespace else self.value == ""
        # Handle any object that supports len()
        elif hasattr(self.value, '__len__'):
            is_empty = len(self.value) == 0
        # All other types are considered non-empty
        else:
            is_empty = False

        # Apply negation
        passed = not is_empty if self.negate else is_empty

        return {'passed': passed}
