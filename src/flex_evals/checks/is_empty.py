"""
Combined IsEmptyCheck implementation for FEP.

Combines schema validation with execution logic in a single class.
"""

from typing import Any
from pydantic import Field

from .base import BaseCheck, OptionalJSONPath
from ..registry import register
from ..constants import CheckType


@register(CheckType.IS_EMPTY, version='1.0.0')
class IsEmptyCheck(BaseCheck):
    """Tests whether a value is empty (None, empty string, whitespace-only, or any empty collection that supports len())."""  # noqa: E501

    # Pydantic fields with validation
    value: Any = OptionalJSONPath('Value to test for emptiness or JSONPath expression')
    negate: bool = Field(False, description='If true, passes when value is not empty')
    strip_whitespace: bool = Field(True, description='If true, strips whitespace before checking (strings only)')

    def __call__(
        self,
        value: Any,  # noqa: ANN401
        negate: bool = False,
        strip_whitespace: bool = True,
    ) -> dict[str, Any]:
        """
        Execute emptiness check with resolved arguments.

        Args:
            value: Resolved value to test for emptiness
            negate: If true, passes when value is not empty
            strip_whitespace: If true, strips whitespace before checking (strings only)

        Returns:
            Dictionary with 'passed' key indicating check result
        """
        # Handle None directly
        if value is None:
            is_empty = True
        # Handle strings with optional whitespace stripping
        elif isinstance(value, str):
            is_empty = value.strip() == "" if strip_whitespace else value == ""
        # Handle any object that supports len()
        elif hasattr(value, '__len__'):
            is_empty = len(value) == 0
        # All other types are considered non-empty
        else:
            is_empty = False

        # Apply negation
        passed = not is_empty if negate else is_empty

        return {'passed': passed}
