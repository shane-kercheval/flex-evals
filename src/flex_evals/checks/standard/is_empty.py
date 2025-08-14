"""
Is Empty check implementation for FEP.

Tests whether a value is empty (None, empty string, whitespace-only, empty list, or empty dict).
"""

from typing import Any

from ..base import BaseCheck
from ...registry import register
from ...constants import CheckType


@register(CheckType.IS_EMPTY, version="1.0.0")
class IsEmptyCheck_v1_0_0(BaseCheck):  # noqa: N801
    """
    Tests whether a value is empty.

    Arguments Schema:
    - value: any | JSONPath - Value to test for emptiness
    - negate: boolean (default: false) - If true, passes when value is not empty
    - strip_whitespace: boolean (default: true) - If true, strips whitespace before checking
        (strings only)

    Empty values include:
    - None
    - Empty strings ("")
    - Whitespace-only strings (when strip_whitespace=True)
    - Any empty collection that supports len() (lists, dicts, sets, tuples, etc.)

    Results Schema:
    - passed: boolean - Whether the empty check passed
    """

    def __call__(  # noqa: D102
            self,
            value: Any,  # noqa: ANN401
            negate: bool = False,
            strip_whitespace: bool = True,
        ) -> dict[str, Any]:

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

        return {"passed": passed}
