"""
Regex check implementation for FEP.

Tests text against regular expression patterns with configurable flags.
"""

import re
from typing import Any

from ..base import BaseCheck, EvaluationContext
from ...registry import register
from ...exceptions import ValidationError


@register("regex", version="1.0.0")
class RegexCheck(BaseCheck):
    """
    Tests text against regular expression patterns.

    Arguments Schema:
    - text: string | JSONPath - Text to test against the pattern
    - pattern: string - Regular expression pattern to match against the text
    - negate: boolean (default: false) - If true, passes when pattern doesn't match
    - flags: object - Regex matching options:
        - case_insensitive: boolean (default: false) - If true, ignores case when matching
        - multiline: boolean (default: false) - If true, ^ and $ match line boundaries
        - dot_all: boolean (default: false) - If true, . matches newline characters

    Results Schema:
    - passed: boolean - Whether the regex check passed
    """

    def __call__(self, arguments: dict[str, Any], context: EvaluationContext) -> dict[str, Any]:
        # Validate required arguments
        if "text" not in arguments:
            raise ValidationError("Regex check requires 'text' argument")

        if "pattern" not in arguments:
            raise ValidationError("Regex check requires 'pattern' argument")

        # Get argument values with defaults
        text = arguments["text"]
        pattern = arguments["pattern"]
        negate = arguments.get("negate", False)
        flags_dict = arguments.get("flags", {})

        # Validate pattern is a string
        if not isinstance(pattern, str):
            raise ValidationError("Regex check 'pattern' argument must be a string")

        # Convert text to string
        text_str = str(text) if text is not None else ""

        # Build regex flags
        regex_flags = 0
        if flags_dict.get("case_insensitive", False):
            regex_flags |= re.IGNORECASE
        if flags_dict.get("multiline", False):
            regex_flags |= re.MULTILINE
        if flags_dict.get("dot_all", False):
            regex_flags |= re.DOTALL

        try:
            # Compile and test the pattern
            compiled_pattern = re.compile(pattern, regex_flags)
            match = compiled_pattern.search(text_str) is not None

            # Apply negation
            passed = not match if negate else match

            return {"passed": passed}

        except re.error as e:
            raise ValidationError(f"Invalid regex pattern '{pattern}': {e!s}") from e
