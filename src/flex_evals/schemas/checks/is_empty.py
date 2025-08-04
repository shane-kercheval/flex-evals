"""IsEmptyCheck schema class for type-safe empty check definitions."""

from typing import Any
from pydantic import Field

from ...constants import CheckType
from ..check import SchemaCheck


class IsEmptyCheck(SchemaCheck):
    """
    Type-safe schema for empty check.

    Tests whether a value is empty (None, empty string, or whitespace-only).

    Fields:
    - value: value to check or JSONPath expression pointing to the value
    - negate: If true, passes when value is not empty (default: False)
    - strip_whitespace: If true, strips whitespace before checking strings only (default: True)
    - version: Optional version string for the check\
    """

    value: Any = Field(..., description="value to check or JSONPath expression pointing to the value")  # noqa: E501
    negate: bool = Field(False, description="If true, passes when value is not empty")
    strip_whitespace: bool = Field(
        True, description="If true, strips whitespace before checking strings only",
    )

    @property
    def check_type(self) -> CheckType:
        """Return the CheckType for this check."""
        return CheckType.IS_EMPTY
