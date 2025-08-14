"""AttributeExistsCheck schema class for type-safe attribute existence check definitions."""

from typing import ClassVar
from pydantic import Field

from ...constants import CheckType
from ..check import SchemaCheck, RequiredJSONPath


class AttributeExistsCheck(SchemaCheck):
    """Tests whether an attribute/field exists as defined by the JSONPath; useful for checking if optional fields like errors, metadata, or dynamic properties are present."""  # noqa: E501

    VERSION: ClassVar[str] = "1.0.0"
    CHECK_TYPE: ClassVar[CheckType] = CheckType.ATTRIBUTE_EXISTS

    path: str = RequiredJSONPath(
        "JSONPath expression pointing to the attribute to check for existence",
    )
    negate: bool = Field(False, description="If true, passes when attribute does not exist")
