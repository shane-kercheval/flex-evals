"""AttributeExistsCheck schema class for type-safe attribute existence check definitions."""

from pydantic import Field

from ...constants import CheckType
from ..check import SchemaCheck


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

    path: str = Field(..., description="JSONPath expression pointing to the attribute to check for existence")  # noqa: E501
    negate: bool = Field(False, description="If true, passes when attribute does not exist")

    @property
    def check_type(self) -> CheckType:
        """Return the CheckType for this check."""
        return CheckType.ATTRIBUTE_EXISTS
