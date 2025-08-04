"""
Is Empty check implementation for FEP.

Tests whether a value is empty (None, empty string, or whitespace-only).
"""

from typing import Any

from ..base import BaseCheck
from ...registry import register
from ...constants import CheckType


@register(CheckType.IS_EMPTY, version="1.0.0")
class IsEmptyCheck(BaseCheck):
    """
    Tests whether a value is empty.

    Arguments Schema:
    - value: string | JSONPath - Value to test for emptiness
    - negate: boolean (default: false) - If true, passes when value is not empty
    - strip_whitespace: boolean (default: true) - If true, strips whitespace before checking

    Results Schema:
    - passed: boolean - Whether the empty check passed
    """

    def __call__(  # noqa: D102
            self,
            value: str,
            negate: bool = False,
            strip_whitespace: bool = True,
        ) -> dict[str, Any]:

        # Convert to string, handling None
        value_str = str(value) if value is not None else ""

        # Apply whitespace stripping if requested
        if strip_whitespace:
            value_str = value_str.strip()

        # Check if empty
        is_empty = value_str == ""

        # Apply negation
        passed = not is_empty if negate else is_empty

        return {"passed": passed}
