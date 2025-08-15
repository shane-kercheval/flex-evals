"""
Combined EqualsCheck implementation for FEP.

Combines schema validation with execution logic in a single class.
"""

from typing import Any
from pydantic import field_validator, Field

from .base import BaseCheck, JSONPath, _convert_to_jsonpath
from ..registry import register
from ..constants import CheckType


@register(CheckType.EQUALS, version='1.0.0')
class EqualsCheck(BaseCheck):
    """Tests whether two values of any type are equal using Python's == operator."""

    # Pydantic fields with validation - can be literals or JSONPath objects
    actual: str | list | dict | set | tuple | int | float | bool | None | JSONPath = Field(
        ..., description="Value to check or JSONPath expression pointing to the value",
    )
    expected: str | list | dict | set | tuple | int | float | bool | None | JSONPath = Field(
        ..., description="Expected value or JSONPath expression pointing to the value",
    )
    negate: bool | JSONPath = Field(
        False, description="If true, passes when values don't match",
    )

    @field_validator('actual', 'expected', 'negate', mode='before')
    @classmethod
    def convert_jsonpath(cls, value: object) -> object | JSONPath:
        """Convert JSONPath-like strings to JSONPath objects."""
        return _convert_to_jsonpath(value)

    def __call__(self) -> dict[str, Any]:
        """
        Execute equals check using resolved Pydantic fields.

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
        if isinstance(self.negate, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'negate' field: {self.negate}")

        # Perform direct equality comparison
        # Python's == operator handles different types appropriately
        equal = self.actual == self.expected

        # Apply negation
        passed = not equal if self.negate else equal

        return {'passed': passed}
