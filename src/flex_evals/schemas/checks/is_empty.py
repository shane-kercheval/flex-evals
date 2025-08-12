"""IsEmptyCheck schema class for type-safe empty check definitions."""

from typing import ClassVar
from pydantic import Field

from ...constants import CheckType
from ..check import SchemaCheck, OptionalJSONPath


class IsEmptyCheck(SchemaCheck):
    """
    Type-safe schema for empty check.

    Tests whether a value is empty (None, empty string, or whitespace-only).

    Fields:
    - value: value to check or JSONPath expression pointing to the value
    - negate: If true, passes when value is not empty (default: False)
    - strip_whitespace: If true, strips whitespace before checking strings only (default: True)
    - version: Optional version string for the check
    """

    VERSION: ClassVar[str] = "1.0.0"
    CHECK_TYPE: ClassVar[CheckType] = CheckType.IS_EMPTY

    value: str = OptionalJSONPath("value to check or JSONPath expression pointing to the value")
    negate: bool = Field(False, description="If true, passes when value is not empty")
    strip_whitespace: bool = Field(
        True, description="If true, strips whitespace before checking strings only",
    )
