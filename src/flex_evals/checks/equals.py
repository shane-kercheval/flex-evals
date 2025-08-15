"""
Combined EqualsCheck implementation for FEP.

Combines schema validation with execution logic in a single class.
"""

from typing import Any
from pydantic import Field

from .base import BaseCheck, OptionalJSONPath
from ..registry import register
from ..constants import CheckType


@register(CheckType.EQUALS, version='1.0.0')
class EqualsCheck(BaseCheck):
    """Tests whether two values of any type are equal using Python's == operator."""

    # Pydantic fields with validation
    actual: Any = OptionalJSONPath('Value to check or JSONPath expression pointing to the value')
    expected: Any = OptionalJSONPath('Expected value or JSONPath expression pointing to the value')
    negate: bool = Field(False, description='If true, passes when values don\'t match')

    def __call__(
        self,
        actual: Any,  # noqa: ANN401
        expected: Any,  # noqa: ANN401
        negate: bool = False,
    ) -> dict[str, Any]:
        """
        Execute equality check with resolved arguments.

        Args:
            actual: Resolved value to check
            expected: Resolved expected value
            negate: If true, passes when values don't match

        Returns:
            Dictionary with 'passed' key indicating check result
        """
        # Perform direct equality comparison
        # Python's == operator handles different types appropriately
        equal = actual == expected

        # Apply negation
        passed = not equal if negate else equal

        return {'passed': passed}
