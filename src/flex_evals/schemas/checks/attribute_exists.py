"""AttributeExistsCheck schema class for type-safe attribute existence check definitions."""

from typing import ClassVar
from pydantic import Field

from ...constants import CheckType
from ..check import SchemaCheck, RequiredJSONPath


class AttributeExistsCheck(SchemaCheck):
    """
    Type-safe schema for attribute existence check.

    Tests whether an attribute/field exists in the evaluation context, useful for checking
    if optional fields like errors, metadata, or dynamic properties are present without
    failing if they don't exist (unlike JSONPath resolution which throws errors).

    Fields:
    - path: JSONPath expression pointing to the attribute to check for existence
    - negate: If true, passes when attribute does not exist (default: False)
    - version: Optional version string for the check
    """

    VERSION: ClassVar[str] = "1.0.0"
    CHECK_TYPE: ClassVar[CheckType] = CheckType.ATTRIBUTE_EXISTS

    path: str = RequiredJSONPath(
        "JSONPath expression pointing to the attribute to check for existence",
    )
    negate: bool = Field(False, description="If true, passes when attribute does not exist")
