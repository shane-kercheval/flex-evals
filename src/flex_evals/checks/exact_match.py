"""
Combined ExactMatch check implementation for FEP.

Combines schema validation with execution logic in a single class.
"""

from typing import Any
from pydantic import Field

from .base import BaseCheck, OptionalJSONPath
from ..registry import register
from ..constants import CheckType


@register(CheckType.EXACT_MATCH, version='1.0.0')
class ExactMatchCheck(BaseCheck):
    """Compares two text values for exact equality."""

    # Pydantic fields with validation
    actual: str = OptionalJSONPath(
        'Value to check or JSONPath expression pointing to the value',
    )
    expected: str = OptionalJSONPath(
        'Expected value or JSONPath expression pointing to the value',
    )
    case_sensitive: bool = Field(True, description='Whether string comparison is case-sensitive')
    negate: bool = Field(False, description='If true, passes when values don\'t match')

    def __call__(
            self,
            actual: str,
            expected: str,
            case_sensitive: bool = True,
            negate: bool = False,
        ) -> dict[str, Any]:
        """
        Execute exact match check with resolved arguments.

        Args:
            actual: Resolved value to check
            expected: Resolved expected value
            case_sensitive: Whether comparison is case-sensitive
            negate: If true, passes when values don't match

        Returns:
            Dictionary with 'passed' key indicating check result
        """
        # Convert to strings for comparison
        actual_str = str(actual) if actual is not None else ''
        expected_str = str(expected) if expected is not None else ''

        # Apply case sensitivity
        if not case_sensitive:
            actual_str = actual_str.lower()
            expected_str = expected_str.lower()

        # Perform comparison
        match = actual_str == expected_str

        # Apply negation
        passed = not match if negate else match

        return {'passed': passed}
