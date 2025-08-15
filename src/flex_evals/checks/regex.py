"""
Combined RegexCheck implementation for FEP.

Combines schema validation with execution logic in a single class.
"""

import re
from typing import Any
from pydantic import Field, BaseModel, field_validator

from .base import BaseCheck, JSONPath, _convert_to_jsonpath
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

    # Pydantic fields with validation - can be literals or JSONPath objects
    text: str | JSONPath = Field(
        ...,
        description="Text to test against the pattern or JSONPath expression",
    )
    pattern: str | JSONPath = Field(
        ...,
        description="Regular expression pattern to match against the text",
    )
    negate: bool | JSONPath = Field(
        False,
        description="If true, passes when pattern doesn't match",
    )
    flags: RegexFlags | JSONPath = Field(
        default_factory=RegexFlags,
        description="Regex matching options",
    )

    @field_validator('text', 'pattern', 'negate', 'flags', mode='before')
    @classmethod
    def convert_jsonpath(cls, value: object) -> object | JSONPath:
        """Convert JSONPath-like strings to JSONPath objects."""
        return _convert_to_jsonpath(value)

    def __call__(self) -> dict[str, Any]:
        """
        Execute regex check using resolved Pydantic fields.

        All JSONPath objects should have been resolved by execute() before this is called.

        Returns:
            Dictionary with 'passed' key indicating check result

        Raises:
            RuntimeError: If any field contains unresolved JSONPath objects
        """
        # Validate that all fields are resolved (no JSONPath objects remain)
        if isinstance(self.text, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'text' field: {self.text}")
        if isinstance(self.pattern, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'pattern' field: {self.pattern}")
        if isinstance(self.negate, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'negate' field: {self.negate}")
        if isinstance(self.flags, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'flags' field: {self.flags}")

        # Validate pattern is a string
        if not isinstance(self.pattern, str):
            raise ValidationError("Regex check 'pattern' argument must be a string")

        # Handle flags (should already be RegexFlags from default_factory)
        flags_obj = self.flags if isinstance(self.flags, RegexFlags) else RegexFlags()

        # Convert text to string
        text_str = str(self.text) if self.text is not None else ""

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
            compiled_pattern = re.compile(self.pattern, regex_flags)
            match = compiled_pattern.search(text_str) is not None

            # Apply negation
            passed = not match if self.negate else match

            return {'passed': passed}

        except re.error as e:
            raise ValidationError(f"Invalid regex pattern '{self.pattern}': {e!s}") from e
