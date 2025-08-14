"""EqualsCheck schema class for type-safe equality check definitions."""

from typing import ClassVar, Any
from pydantic import Field

from ...constants import CheckType
from ..check import SchemaCheck, OptionalJSONPath


class EqualsCheck(SchemaCheck):
    """Tests whether two values of any type are equal using Python's == operator."""

    VERSION: ClassVar[str] = "1.0.0"
    CHECK_TYPE: ClassVar[CheckType] = CheckType.EQUALS

    actual: str | list | dict | set | tuple | int | float | bool | None | Any = OptionalJSONPath(
        "value to check or JSONPath expression pointing to the value", min_length=1,
    )
    expected: str | list | dict | set | tuple | int | float | bool | None | Any = OptionalJSONPath(
        "expected value or JSONPath expression pointing to the value", min_length=1,
    )
    negate: bool = Field(False, description="If true, passes when values don't match")
