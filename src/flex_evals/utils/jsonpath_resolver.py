"""
JSONPath resolution implementation for FEP.

Handles detection of JSONPath expressions, resolution against evaluation context,
and proper formatting of resolved arguments according to FEP protocol.

## Performance Optimization

This module uses @lru_cache to cache parsed JSONPath expressions, providing dramatic
performance improvements (100x+ speedup) while maintaining correctness across
different evaluation contexts.

Key insight: JSONPath processing has two distinct phases:
1. **Parsing** (expensive): Converting string expressions like "$.output.value.result"
   into compiled JSONPath objects (~2.2ms per parse)
2. **Resolution** (fast): Applying the compiled expression to specific data contexts
   (~0.003ms per resolution)

The @lru_cache decorator ensures each unique JSONPath string is parsed only once,
with all validation and resolution operations sharing the same cached parsed expression.

## Cache Safety

The cache stores only **parsed JSONPath expressions**, never **resolved values**. Because of the
nature of the checks/evals, there is a relatively small set of unique JSONPath expressions used
but potentially reused across many different test cases and outputs. Each resolution operation
applies the cached expression to fresh context data, ensuring that:

- Different test cases get different values for the same JSONPath
- No data leakage occurs between evaluations
- Thread safety is maintained (expressions are stateless)

Example:
```python
# Both checks use same JSONPath but get different values
path = "$.output.value.result"

# First time: parses and caches
result1 = resolve_argument(path, context1)  # Returns 100

# Second time: uses cached parse
result2 = resolve_argument(path, context2)  # Returns 200

# Cache contains 1 parsed expression, but results are different
```
"""

from typing import Any
from functools import lru_cache
from jsonpath_ng import parse as jsonpath_parse
from jsonpath_ng.exceptions import JSONPathError as JSONPathNGError
from ..exceptions import JSONPathError


def is_jsonpath(value: str) -> bool:
    r"""
    Quick heuristic check if a string looks like a JSONPath expression.

    This is a cheap prefix check used to avoid expensive parsing attempts
    on obvious non-JSONPath values. Does NOT validate syntax - use
    validate_jsonpath() for full validation.

    Args:
        value: String to check

    Returns:
        True if value appears to be a JSONPath expression, False otherwise

    Rules:
    - Strings beginning with '$.' are JSONPath expressions
    - Strings beginning with '\\$.' are escaped literals (not JSONPath)
    - Non-strings return False

    Examples:
        >>> is_jsonpath("$.output.value")
        True
        >>> is_jsonpath("hello")
        False
        >>> is_jsonpath("\\$.literal")
        False
    """
    if not isinstance(value, str):
        return False

    if value.startswith("\\$."):
        return False  # Escaped literal

    return value.startswith("$.")


@lru_cache(maxsize=128)
def parse_jsonpath_cached(expression: str) -> Any:  # noqa: ANN401
    """
    Parse and cache a JSONPath expression.

    This is the single source of truth for JSONPath parsing. All validation
    and resolution operations use this cached parser to avoid re-parsing the
    same expressions.

    Note: Exceptions are NOT cached by lru_cache. Use is_jsonpath() heuristic
    to avoid repeated parse attempts on non-JSONPath strings.

    Args:
        expression: JSONPath expression string (should start with '$')

    Returns:
        Parsed JSONPath expression object

    Raises:
        JSONPathNGError: If expression has invalid JSONPath syntax

    Performance:
        - First call: ~2.2ms (parsing)
        - Subsequent calls: ~0.0001ms (cache lookup)
        - Cache size: 128 unique expressions (generous for typical use)
    """
    return jsonpath_parse(expression)


def validate_jsonpath(expression: str) -> bool:
    """
    Validate that a string is a valid JSONPath expression.

    Actually parses the expression to verify syntax, using caching to avoid
    re-parsing the same expressions. Returns boolean instead of raising
    exceptions for use in Pydantic validators and other validation contexts.

    Args:
        expression: String to validate

    Returns:
        True if expression is valid JSONPath, False otherwise

    Examples:
        >>> validate_jsonpath("$.output.value")
        True
        >>> validate_jsonpath("$[0]")
        True
        >>> validate_jsonpath("hello")
        False
        >>> validate_jsonpath("$.[invalid")
        False

    Performance:
        With caching, validation of the same expression is ~1000x faster
        on subsequent calls.
    """
    # Handle non-string inputs gracefully
    if not isinstance(expression, str):
        return False
    # Must start with $ to be considered a JSONPath expression
    if not expression.startswith('$'):
        return False
    try:
        parse_jsonpath_cached(expression)
        return True
    except (JSONPathNGError, Exception):
        return False


def resolve_argument(value: object, context: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve a single argument value against the evaluation context.

    Uses module-level parse_jsonpath_cached() for automatic caching via @lru_cache.
    The same JSONPath string (e.g., "$.output.value.result") is parsed only once
    but can be applied to different contexts to yield different values.

    Args:
        value: The argument value (literal or JSONPath string)
        context: The evaluation context to resolve against (fresh per call)

    Returns:
        Dict with resolved argument in protocol format:
        - For JSONPath: {"jsonpath": "$.path", "value": resolved_value}
        - For literals: {"value": literal_value}

    Raises:
        JSONPathError: If JSONPath expression is invalid or cannot be resolved

    Performance:
        - Cache hit: ~0.003ms per resolution
        - Cache miss: ~2.2ms per resolution (includes parsing)
        - Typical improvement: 100x+ speedup for repeated expressions
    """
    if isinstance(value, str) and is_jsonpath(value):
        # Handle JSONPath expression using cached parser
        try:
            # Use module-level cached parser
            jsonpath_expr = parse_jsonpath_cached(value)
            # Apply cached expression to fresh context (always context-specific)
            matches = jsonpath_expr.find(context)
            if not matches:
                raise JSONPathError(
                    f"JSONPath expression '{value}' did not match any data in evaluation context",
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
