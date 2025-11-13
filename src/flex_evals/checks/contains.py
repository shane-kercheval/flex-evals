"""Contains Check - checks if a string contains a string or specified phrases."""

from typing import Any
from pydantic import field_validator, Field

from .base import BaseCheck, JSONPath, _convert_to_jsonpath
from ..registry import register
from ..exceptions import ValidationError
from ..constants import CheckType


@register(CheckType.CONTAINS, version='1.0.0')
class ContainsCheck(BaseCheck):
    """Checks if the string value contains specified phrases."""

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
    match_all: bool | JSONPath = Field(
        True,
        description="If true, all phrases must be present; if false, at least one phrase must be present",  # noqa: E501
    )

    @field_validator('text', 'phrases', 'case_sensitive', 'negate', 'match_all', mode='before')
    @classmethod
    def convert_jsonpath(cls, value: object) -> object | JSONPath:
        """Convert JSONPath-like strings to JSONPath objects."""
        return _convert_to_jsonpath(value)

    @property
    def default_results(self) -> dict[str, Any]:
        """Return default results structure for contains checks on error."""
        return {'passed': False, 'found': []}

    def __call__(self) -> dict[str, Any]:  # noqa: PLR0912
        """
        Execute contains check using resolved Pydantic fields.

        All JSONPath objects should have been resolved by execute() before this is called.

        Returns:
            Dictionary with 'passed' (bool) and 'found' (list[str]) keys.
            'passed' indicates check result, 'found' contains the list of phrases that were found.

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
        if isinstance(self.match_all, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'match_all' field: {self.match_all}")

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

        # Convert phrases to strings and store originals
        original_phrases = [str(phrase) for phrase in phrases]

        # Apply case sensitivity
        if not self.case_sensitive:
            text_str = text_str.lower()
            search_phrases = [phrase.lower() for phrase in original_phrases]
        else:
            search_phrases = original_phrases

        # Check phrase presence and track found phrases
        found_phrases: list[str] = []
        for i, phrase in enumerate(search_phrases):
            if phrase in text_str:
                found_phrases.append(original_phrases[i])

        # Determine pass criteria based on match_all and negate
        found_count = len(found_phrases)
        if self.match_all:
            passed = (found_count == 0) if self.negate else (found_count == len(phrases))
        else:  # match any
            passed = (found_count < len(phrases)) if self.negate else (found_count > 0)

        return {'passed': passed, 'found': found_phrases}
