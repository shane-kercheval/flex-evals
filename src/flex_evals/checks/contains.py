"""
Combined ContainsCheck implementation for FEP.

Combines schema validation with execution logic in a single class.
"""

from typing import Any
from pydantic import field_validator, Field

from .base import BaseCheck, JSONPath, _convert_to_jsonpath
from ..registry import register
from ..exceptions import ValidationError
from ..constants import CheckType


@register(CheckType.CONTAINS, version='1.0.0')
class ContainsCheck(BaseCheck):
    """Checks if the string value contains all specified phrases."""

    # Pydantic fields with validation - can be literals or JSONPath objects
    text: str | JSONPath = Field(
        ...,
        description="Text to search or JSONPath expression pointing to the text",
    )
    phrases: str | list[str] | JSONPath = Field(
        ...,
        description="Single string or array of strings that must be present",
    )
    case_sensitive: bool | JSONPath = Field(
        True,
        description="Whether phrase matching is case-sensitive",
    )
    negate: bool | JSONPath = Field(
        False,
        description="If true, passes when text contains none of the phrases",
    )

    @field_validator('text', 'phrases', 'case_sensitive', 'negate', mode='before')
    @classmethod
    def convert_jsonpath(cls, value: object) -> object | JSONPath:
        """Convert JSONPath-like strings to JSONPath objects."""
        return _convert_to_jsonpath(value)

    def __call__(self) -> dict[str, Any]:  # noqa: PLR0912
        """
        Execute contains check using resolved Pydantic fields.

        All JSONPath objects should have been resolved by execute() before this is called.

        Returns:
            Dictionary with 'passed' key indicating check result

        Raises:
            RuntimeError: If any field contains unresolved JSONPath objects
        """
        # Validate that all fields are resolved (no JSONPath objects remain)
        if isinstance(self.text, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'text' field: {self.text}")
        if isinstance(self.phrases, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'phrases' field: {self.phrases}")
        if isinstance(self.case_sensitive, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'case_sensitive' field: {self.case_sensitive}")  # noqa: E501
        if isinstance(self.negate, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'negate' field: {self.negate}")

        # Convert single string to list
        phrases = self.phrases
        if isinstance(phrases, str):
            if not phrases:  # Empty string
                raise ValidationError("Contains check 'phrases' argument must not be empty")
            phrases = [phrases]
        elif isinstance(phrases, list):
            if len(phrases) == 0:
                raise ValidationError("Contains check 'phrases' argument must not be empty")
        else:
            raise ValidationError("Contains check 'phrases' argument must be a string or list")

        # Convert text to string
        text_str = str(self.text) if self.text is not None else ""

        # Apply case sensitivity
        if not self.case_sensitive:
            text_str = text_str.lower()
            phrases = [str(phrase).lower() for phrase in phrases]
        else:
            phrases = [str(phrase) for phrase in phrases]

        # Check phrase presence
        found_count = 0
        for phrase in phrases:
            if phrase in text_str:
                found_count += 1

        if self.negate:  # noqa: SIM108
            # Pass if NONE of the phrases are found
            passed = found_count == 0
        else:
            # Pass if ALL phrases are found
            passed = found_count == len(phrases)

        return {'passed': passed}
