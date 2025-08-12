"""
Contains check implementation for FEP.

Checks if text contains all specified phrases or patterns.
"""

from typing import Any

from ..base import BaseCheck
from ...registry import register
from ...exceptions import ValidationError
from ...constants import CheckType


@register(CheckType.CONTAINS, version="1.0.0")
class ContainsCheck_v1_0_0(BaseCheck):  # noqa: N801
    """
    Checks if text contains all specific phrases or patterns.

    Arguments Schema:
    - text: string | JSONPath - Text to search
    - phrases: array[string] - Array of strings that must be present in the text
    - negate: boolean (default: false) - If true, passes when text contains none of the phrases.
                                        If false, passes when text contains all of the phrases
    - case_sensitive: boolean (default: true) - Whether phrase matching is case-sensitive

    Results Schema:
    - passed: boolean - Whether the contains check passed
    """

    def __call__(  # noqa: D102
            self,
            text: str,
            phrases: list[str],
            case_sensitive: bool = True,
            negate: bool = False,
        ) -> dict[str, Any]:
        # Validate phrases is a list
        if not isinstance(phrases, list):
            raise ValidationError("Contains check 'phrases' argument must be a list")

        if len(phrases) == 0:
            raise ValidationError("Contains check 'phrases' argument must not be empty")

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

        return {"passed": passed}
