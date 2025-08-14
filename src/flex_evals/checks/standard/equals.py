"""
Equals check implementation for FEP.

Compares two values of any type for equality with configurable negation.
"""

from typing import Any

from ..base import BaseCheck
from ...registry import register
from ...constants import CheckType


@register(CheckType.EQUALS, version="1.0.0")
class EqualsCheck_v1_0_0(BaseCheck):  # noqa: N801
    """
    Compares two values of any type for equality.

    Arguments Schema:
    - actual: any | JSONPath - Value to check
    - expected: any | JSONPath - Value to compare against
    - negate: boolean (default: false) - If true, passes when values don't match

    Supported value types:
    - Strings, integers, floats, booleans
    - Lists, dictionaries, tuples, sets
    - None values
    - Any objects that support equality comparison

    Results Schema:
    - passed: boolean - Whether the equality check passed
    """

    def __call__(  # noqa: D102
            self,
            actual: Any,  # noqa: ANN401
            expected: Any,  # noqa: ANN401
            negate: bool = False,
        ) -> dict[str, Any]:

        # Perform direct equality comparison
        # Python's == operator handles different types appropriately
        equal = actual == expected

        # Apply negation
        passed = not equal if negate else equal

        return {"passed": passed}
