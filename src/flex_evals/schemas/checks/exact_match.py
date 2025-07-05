"""ExactMatchCheck schema class for type-safe exact match check definitions."""

from pydantic import Field

from ...constants import CheckType
from ..check import SchemaCheck


class ExactMatchCheck(SchemaCheck):
    """
    Type-safe schema for exact match check.

    Compares two text values for exact equality with configurable case sensitivity and negation.

    Fields:
    - actual: value to check or JSONPath expression pointing to the value
    - expected: expected value or JSONPath expression pointing to the value
    - case_sensitive: Whether string comparison is case-sensitive (default: True)
    - negate: If true, passes when values don't match (default: False)
    - version: Optional version string for the check
    """

    actual: str = Field(..., min_length=1, description="value to check or JSONPath expression pointing to the value")  # noqa: E501
    expected: str = Field(..., min_length=1, description="expected value or JSONPath expression pointing to the value")  # noqa: E501
    case_sensitive: bool = Field(True, description="Whether string comparison is case-sensitive")
    negate: bool = Field(False, description="If true, passes when values don't match")

    @property
    def check_type(self) -> CheckType:
        """Return the CheckType for this check."""
        return CheckType.EXACT_MATCH
