"""ContainsCheck schema class for type-safe contains check definitions."""

from typing import ClassVar
from pydantic import Field, field_validator

from ...constants import CheckType
from ..check import SchemaCheck, OptionalJSONPath


class ContainsCheck(SchemaCheck):
    """Checks if the string value contains all specified phrases."""

    VERSION: ClassVar[str] = "1.0.0"
    CHECK_TYPE: ClassVar[CheckType] = CheckType.CONTAINS

    text: str = OptionalJSONPath(
        "Text to search or JSONPath expression pointing to the text to search",
    )
    phrases: str | list[str] = OptionalJSONPath(
        "String or list of strings that must be present in the text, or JSONPath expression pointing to single string",  # noqa: E501
    )
    case_sensitive: bool = Field(True, description="Whether phrase matching is case-sensitive")
    negate: bool = Field(False, description="If true, passes when text contains none of the phrases")  # noqa: E501


    @field_validator('phrases')
    @classmethod
    def validate_phrases(cls, v: str | list[str]) -> str | list[str]:
        """Validate phrases field."""
        if isinstance(v, str):
            return v  # Empty strings are now allowed
        if isinstance(v, list):
            if not v:
                raise ValueError("phrases list must not be empty")
            for phrase in v:
                if not isinstance(phrase, str):
                    raise ValueError("all phrases must be strings")
            return v

        raise ValueError("phrases must be a string or list of strings")

