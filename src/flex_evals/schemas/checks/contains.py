"""ContainsCheck schema class for type-safe contains check definitions."""

from typing import ClassVar
from pydantic import Field, field_validator

from ...constants import CheckType
from ..check import SchemaCheck, OptionalJSONPath


class ContainsCheck(SchemaCheck):
    """
    Type-safe schema for contains check.

    Checks if text contains all specified phrases or patterns.

    Fields:
    - text: text to search or JSONPath expression pointing to the text to search
    - phrases: List of strings that must be present in the text
    - case_sensitive: Whether phrase matching is case-sensitive (default: True)
    - negate: If true, passes when text contains none of the phrases.
             If false, passes when text contains all of the phrases (default: False)
    - version: Optional version string for the check
    """

    VERSION: ClassVar[str] = "1.0.0"
    CHECK_TYPE: ClassVar[CheckType] = CheckType.CONTAINS

    text: str = OptionalJSONPath(
        "text to search or JSONPath expression pointing to the text to search",
        min_length=1,
    )
    phrases: list[str] = Field(..., min_length=1, description="List of strings that must be present in the text")  # noqa: E501
    case_sensitive: bool = Field(True, description="Whether phrase matching is case-sensitive")
    negate: bool = Field(False, description="If true, passes when text contains none of the phrases")  # noqa: E501


    @field_validator('phrases')
    @classmethod
    def validate_phrases(cls, v: list[str]) -> list[str]:
        """Validate that all phrases are non-empty strings."""
        if not v:
            raise ValueError("phrases must not be empty")
        for phrase in v:
            if not isinstance(phrase, str) or not phrase:
                raise ValueError("all phrases must be non-empty strings")
        return v

