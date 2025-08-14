"""ExactMatchCheck schema class for type-safe exact match check definitions."""

from typing import ClassVar
from pydantic import Field

from ...constants import CheckType
from ..check import SchemaCheck, OptionalJSONPath


class ExactMatchCheck(SchemaCheck):
    """Compares two text values for exact equality."""

    VERSION: ClassVar[str] = "1.0.0"
    CHECK_TYPE: ClassVar[CheckType] = CheckType.EXACT_MATCH

    actual: str = OptionalJSONPath(
        "value to check or JSONPath expression pointing to the value", min_length=1,
    )
    expected: str = OptionalJSONPath(
        "expected value or JSONPath expression pointing to the value", min_length=1,
    )
    case_sensitive: bool = Field(True, description="Whether string comparison is case-sensitive")
    negate: bool = Field(False, description="If true, passes when values don't match")
