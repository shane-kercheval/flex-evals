"""
Combined ContainsCheck implementation for FEP.

Combines schema validation with execution logic in a single class.
"""

from typing import Any
from pydantic import Field, field_validator

from .base import BaseCheck, OptionalJSONPath
from ..registry import register
from ..exceptions import ValidationError
from ..constants import CheckType


@register(CheckType.CONTAINS, version='1.0.0')
class ContainsCheck(BaseCheck):
    """Checks if the string value contains all specified phrases."""

    # Pydantic fields with validation
    text: str = OptionalJSONPath('Text to search or JSONPath expression pointing to the text')
    phrases: str | list[str] = Field(
        ...,
        description='Single string or array of strings that must be present',
    )
    case_sensitive: bool = Field(True, description='Whether phrase matching is case-sensitive')
    negate: bool = Field(False, description='If true, passes when text contains none of the phrases')

    @field_validator('phrases')
    @classmethod
    def validate_phrases(cls, v: str | list[str]) -> str | list[str]:
        """Validate that phrases is not an empty list."""
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("phrases cannot be an empty list")
        return v

    def __call__(
        self,
        text: str,
        phrases: str | list[str],
        case_sensitive: bool = True,
        negate: bool = False,
    ) -> dict[str, Any]:
        """
        Execute contains check with resolved arguments.

        Args:
            text: Resolved text to search
            phrases: Resolved phrases to find (string or list of strings)
            case_sensitive: Whether phrase matching is case-sensitive
            negate: If true, passes when text contains none of the phrases

        Returns:
            Dictionary with 'passed' key indicating check result
        """
        # Convert single string to list
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
        text_str = str(text) if text is not None else ""

        # Apply case sensitivity
        if not case_sensitive:
            text_str = text_str.lower()
            phrases = [str(phrase).lower() for phrase in phrases]
        else:
            phrases = [str(phrase) for phrase in phrases]

        # Check phrase presence
        found_count = 0
        for phrase in phrases:
            if phrase in text_str:
                found_count += 1

        if negate:  # noqa: SIM108
            # Pass if NONE of the phrases are found
            passed = found_count == 0
        else:
            # Pass if ALL phrases are found
            passed = found_count == len(phrases)

        return {'passed': passed}
