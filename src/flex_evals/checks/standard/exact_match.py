"""
Exact Match check implementation for FEP.

Compares two text values for exact equality with configurable case sensitivity and negation.
"""

from typing import Any

from ..base import BaseCheck
from ...registry import register
from ...constants import CheckType


@register(CheckType.EXACT_MATCH, version="1.0.0")
class ExactMatchCheck_v1_0_0(BaseCheck):  # noqa: N801
    """
    Compares two text values for exact equality.

    Arguments Schema:
    - actual: string | JSONPath - Value to check
    - expected: string | JSONPath - Value to compare against
    - negate: boolean (default: false) - If true, passes when values don't match
    - case_sensitive: boolean (default: true) - Whether string comparison is case-sensitive

    Results Schema:
    - passed: boolean - Whether the exact match check passed
    """

    def __call__(  # noqa: D102
            self,
            actual: str,
            expected: str,
            case_sensitive: bool = True,
            negate: bool = False,
        ) -> dict[str, Any]:

        # Convert to strings for comparison
        actual_str = str(actual) if actual is not None else ""
        expected_str = str(expected) if expected is not None else ""

        # Apply case sensitivity
        if not case_sensitive:
            actual_str = actual_str.lower()
            expected_str = expected_str.lower()

        # Perform comparison
        match = actual_str == expected_str

        # Apply negation
        passed = not match if negate else match

        return {"passed": passed}
