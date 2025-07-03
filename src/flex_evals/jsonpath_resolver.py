"""
JSONPath resolution implementation for FEP.

Handles detection of JSONPath expressions, resolution against evaluation context,
and proper formatting of resolved arguments according to FEP protocol.
"""

from typing import Any
from dataclasses import asdict
from jsonpath_ng import parse as jsonpath_parse
from jsonpath_ng.exceptions import JSONPathError as JSONPathNGError

from .schemas import TestCase, Output
from .exceptions import JSONPathError


class JSONPathResolver:
    r"""
    Handles JSONPath detection and resolution for FEP evaluation context.

    Supports:
    - Detection of JSONPath expressions (strings beginning with '$.')
    - Escape syntax for literal strings (strings beginning with '\\$.')
    - Resolution against evaluation context structure
    - Proper formatting of resolved arguments for protocol compliance
    """

    def __init__(self):
        self._cache: dict[str, Any] = {}

    def is_jsonpath(self, value: str) -> bool:
        r"""
        Determine if a string value is a JSONPath expression.

        Args:
            value: String to check

        Returns:
            True if the string is a JSONPath expression, False otherwise

        Rules:
        - Strings beginning with '$.' are JSONPath expressions
        - Strings beginning with '\\$.' are literal strings starting with '$.'
        """
        if not isinstance(value, str):
            return False

        if value.startswith("\\$."):
            return False  # Escaped literal

        return value.startswith("$.")

    def resolve_argument(self, value: object, context: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve a single argument value against the evaluation context.

        Args:
            value: The argument value (literal or JSONPath string)
            context: The evaluation context to resolve against

        Returns:
            Dict with resolved argument in protocol format:
            - For JSONPath: {"jsonpath": "$.path", "value": resolved_value}
            - For literals: {"value": literal_value}

        Raises:
            JSONPathError: If JSONPath expression is invalid or cannot be resolved
        """
        if isinstance(value, str) and self.is_jsonpath(value):
            # Handle JSONPath expression
            try:
                # Try to get from cache first
                if value in self._cache:
                    jsonpath_expr = self._cache[value]
                else:
                    jsonpath_expr = jsonpath_parse(value)
                    self._cache[value] = jsonpath_expr

                matches = jsonpath_expr.find(context)
                if not matches:
                    raise JSONPathError(
                        f"JSONPath expression '{value}' did not match any data in evaluation context",  # noqa: E501
                        jsonpath_expression=value,
                    )

                # Return first match (FEP expects single values)
                resolved_value = matches[0].value

                return {
                    "jsonpath": value,
                    "value": resolved_value,
                }

            except JSONPathNGError as e:
                raise JSONPathError(
                    f"Invalid JSONPath expression '{value}': {e!s}",
                    jsonpath_expression=value,
                ) from e

        elif isinstance(value, str) and value.startswith("\\$."):
            # Handle escaped literal (remove escape prefix)
            literal_value = value[1:]  # Remove the backslash
            return {"value": literal_value}

        else:
            # Handle literal value
            return {"value": value}

    def create_evaluation_context(self, test_case: TestCase, output: Output) -> dict[str, Any]:
        """
        Create evaluation context structure from test case and output.

        Args:
            test_case: The test case being evaluated
            output: The system output for this test case

        Returns:
            Dict with evaluation context matching FEP protocol structure:
            {
                "test_case": {
                    "id": "string",
                    "input": "string | object",
                    "expected": "string | object | null",
                    "metadata": "object"
                },
                "output": {
                    "value": "string | object",
                    "metadata": "object"
                }
            }
        """
        return {
            'test_case': asdict(test_case),
            'output': asdict(output),
        }

    def resolve_arguments(
            self,
            arguments: dict[str, Any],
            context: dict[str, Any],
        ) -> dict[str, Any]:
        """
        Resolve all arguments in a check's argument dictionary.

        Args:
            arguments: Dictionary of check arguments (may contain JSONPath expressions)
            context: Evaluation context to resolve against

        Returns:
            Dictionary with all arguments resolved in protocol format

        Raises:
            JSONPathError: If any JSONPath expression is invalid or cannot be resolved
        """
        resolved = {}

        for key, value in arguments.items():
            resolved[key] = self.resolve_argument(value, context)

        return resolved
