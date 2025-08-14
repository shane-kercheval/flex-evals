"""IsEmptyCheck schema class for type-safe empty check definitions."""

from typing import ClassVar, Any
from pydantic import Field

from ...constants import CheckType
from ..check import SchemaCheck, OptionalJSONPath


class IsEmptyCheck(SchemaCheck):
    """Tests whether a value is empty (None, empty string, whitespace-only, or any empty collection that supports len())."""  # noqa: E501

    VERSION: ClassVar[str] = "1.0.0"
    CHECK_TYPE: ClassVar[CheckType] = CheckType.IS_EMPTY

    value: str | list | dict | set | tuple | int | float | bool | None | Any = OptionalJSONPath(
        "value to check or JSONPath expression pointing to the value",
    )
    negate: bool = Field(False, description="If true, passes when value is not empty")
    strip_whitespace: bool = Field(
        True, description="If true, strips whitespace before checking strings only",
    )
