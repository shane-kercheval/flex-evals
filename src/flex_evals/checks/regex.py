"""
Combined RegexCheck implementation for FEP.

Combines schema validation with execution logic in a single class.
"""

import re
from typing import Any
from pydantic import Field, BaseModel

from .base import BaseCheck, OptionalJSONPath
from ..registry import register
from ..exceptions import ValidationError
from ..constants import CheckType


class RegexFlags(BaseModel):
    """Regex matching options configuration."""

    case_insensitive: bool = Field(False, description='If true, ignores case when matching')
    multiline: bool = Field(False, description='If true, ^ and $ match line boundaries')
    dot_all: bool = Field(False, description='If true, . matches newline characters')


@register(CheckType.REGEX, version='1.0.0')
class RegexCheck(BaseCheck):
    """Checks if a text value matches a specified regular expression pattern."""

    # Pydantic fields with validation
    text: str = OptionalJSONPath('Text to test against the pattern or JSONPath expression')
    pattern: str = Field(..., description='Regular expression pattern to match against the text')
    negate: bool = Field(False, description='If true, passes when pattern doesn\'t match')
    flags: RegexFlags = Field(default_factory=RegexFlags, description='Regex matching options')

    def __call__(
        self,
        text: str,
        pattern: str,
        negate: bool = False,
        flags: dict | RegexFlags | None = None,
    ) -> dict[str, Any]:
        """
        Execute regex check with resolved arguments.

        Args:
            text: Resolved text to test against pattern
            pattern: Regular expression pattern to match
            negate: If true, passes when pattern doesn't match
            flags: Regex matching options (dict or RegexFlags instance)

        Returns:
            Dictionary with 'passed' key indicating check result
        """
        # Validate pattern is a string
        if not isinstance(pattern, str):
            raise ValidationError("Regex check 'pattern' argument must be a string")

        # Handle flags conversion
        if flags is None:
            flags_obj = RegexFlags()
        elif isinstance(flags, dict):
            flags_obj = RegexFlags(**flags)
        elif isinstance(flags, RegexFlags):
            flags_obj = flags
        else:
            raise ValidationError("Regex check 'flags' argument must be a dict or RegexFlags")

        # Convert text to string
        text_str = str(text) if text is not None else ""

        # Build regex flags
        regex_flags = 0
        if flags_obj.case_insensitive:
            regex_flags |= re.IGNORECASE
        if flags_obj.multiline:
            regex_flags |= re.MULTILINE
        if flags_obj.dot_all:
            regex_flags |= re.DOTALL

        try:
            # Compile and test the pattern
            compiled_pattern = re.compile(pattern, regex_flags)
            match = compiled_pattern.search(text_str) is not None

            # Apply negation
            passed = not match if negate else match

            return {'passed': passed}

        except re.error as e:
            raise ValidationError(f"Invalid regex pattern '{pattern}': {e!s}") from e
