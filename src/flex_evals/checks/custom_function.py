"""Custom Function Check - executes user-provided python functions with JSONPath support."""

import asyncio
from typing import Any
from collections.abc import Callable, Mapping
from pydantic import Field, field_validator

from .base import BaseAsyncCheck, JSONPath, _convert_to_jsonpath, EvaluationContext
from ..registry import register
from ..exceptions import ValidationError, CheckExecutionError
from ..constants import CheckType
from ..utils.jsonpath_resolver import resolve_argument


@register(CheckType.CUSTOM_FUNCTION, version='1.0.0')
class CustomFunctionCheck(BaseAsyncCheck):
    """
    Executes user-provided python validation functions.

    This check supports both sync and async validation functions, which can be provided as:
    - Direct callables (functions or lambdas)
    - String function definitions (lambda or named functions)
    - JSONPath expressions that resolve to either of the above

    Special JSONPath handling:
        In addition to standard JSONPath support at the field level, this check also resolves
        JSONPath strings within the function_args dictionary values. For example:

            CustomFunctionCheck(
                validation_function=my_func,
                function_args={
                    "text": "$.output.value",      # JSONPath string resolved at runtime
                    "expected": "$.test_case.expected",
                    "threshold": 0.95,              # Regular literal value
                }
            )

    Known limitation:
        The resolved_arguments field in CheckResult shows the original function_args dict
        with JSONPath strings, not the final resolved values. The actual resolution happens
        during execution and the resolved values are used correctly, but they are not
        captured in the resolved_arguments metadata. This is a known trade-off for the
        flexibility of supporting JSONPath strings within dict values.

    Returns:
        The validation function must return a dict containing the check results.
        Common pattern: {'passed': bool, ...additional fields...}
    """

    # Pydantic fields with validation - can be literals or JSONPath objects
    validation_function: str | Callable[..., Any] | JSONPath = Field(
        ...,
        description=(
            "Function returning dict (sync or async), string function definition, or JSONPath"
        ),
    )
    function_args: dict[str, Any] | JSONPath = Field(
        default_factory=dict,
        description="Arguments to pass to validation_function or JSONPath to dict",
    )

    @field_validator('validation_function', 'function_args', mode='before')
    @classmethod
    def convert_jsonpath(cls, value: object) -> object | JSONPath:
        """Convert JSONPath-like strings to JSONPath objects."""
        return _convert_to_jsonpath(value)

    @property
    def default_results(self) -> dict[str, Any]:
        """Return default results structure for custom_function checks on error."""
        return {}

    def resolve_fields(self, context: EvaluationContext) -> dict[str, Any]:
        """
        Resolve JSONPath fields including JSONPath strings within function_args dict.

        This override handles the special case where function_args is a dict containing
        JSONPath strings as values (e.g., {"text": "$.output.value"}).
        """
        # First, resolve any top-level JSONPath fields normally
        resolved_arguments = super().resolve_fields(context)

        # Special handling: if function_args is a dict (not JSONPath), check for JSONPath strings
        if isinstance(self.function_args, dict):
            resolved_function_args = {}
            for key, value in self.function_args.items():
                if isinstance(value, str) and value.startswith("$."):
                    # This is a JSONPath string - resolve it
                    try:
                        resolved_result = resolve_argument(value, context.context_dict)
                        resolved_function_args[key] = resolved_result["value"]
                    except Exception as e:
                        # Let the error propagate naturally
                        raise ValidationError(
                            f"Failed to resolve JSONPath in function_args['{key}']: {e}",
                        ) from e
                else:
                    # Not a JSONPath string, use as-is
                    resolved_function_args[key] = value

            # Update the function_args with resolved values
            self.function_args = resolved_function_args

        return resolved_arguments

    async def __call__(self) -> dict[str, Any]:
        """
        Execute custom function check using resolved Pydantic fields.

        All JSONPath objects should have been resolved by execute() before this is called.

        Returns:
            Dictionary with results from the validation function

        Raises:
            RuntimeError: If any field contains unresolved JSONPath objects
            ValidationError: If validation_function is invalid or result is not a dict
            CheckExecutionError: If function execution fails
        """
        # Validate that all fields are resolved (no JSONPath objects remain)
        if isinstance(self.validation_function, JSONPath):
            raise RuntimeError(
                f"JSONPath not resolved for 'validation_function' field: "
                f"{self.validation_function}",
            )
        if isinstance(self.function_args, JSONPath):
            raise RuntimeError(
                f"JSONPath not resolved for 'function_args' field: {self.function_args}",
            )

        # Convert string function to callable if needed
        if isinstance(self.validation_function, str):
            validation_function = self._string_to_function(self.validation_function)
        elif not callable(self.validation_function):
            raise ValidationError(
                "validation_function must be callable or string function definition",
            )
        else:
            validation_function = self.validation_function

        # Validate function_args is a dictionary
        if not isinstance(self.function_args, Mapping):
            raise ValidationError("function_args must be a dictionary")
        fn_kwargs = dict(self.function_args)

        try:
            # Execute the validation function (handle both sync and async)
            if asyncio.iscoroutinefunction(validation_function):
                result = await validation_function(**fn_kwargs)
            else:
                result = validation_function(**fn_kwargs)

            # Validate that result is a dictionary
            if not isinstance(result, dict):
                raise ValidationError(
                    f"Validation function must return dict, got {type(result).__name__}",
                )

            return result

        except ValidationError:
            # Re-raise ValidationError as-is
            raise
        except Exception as e:
            raise CheckExecutionError(f"Error executing validation function: {e!s}") from e

    def _string_to_function(self, func_string: str) -> Callable:
        """Convert string function definition to callable."""
        try:
            # Create namespace with common imports
            namespace = {
                're': __import__('re'),
                'json': __import__('json'),
                'math': __import__('math'),
                'datetime': __import__('datetime'),
                'os': __import__('os'),
                'sys': __import__('sys'),
                'time': __import__('time'),
            }

            if func_string.strip().startswith('lambda'):
                # Handle lambda functions
                return eval(func_string, namespace, namespace)
            # Handle named function definitions
            exec(func_string, namespace, namespace)
            # Find the defined function in namespace
            functions = [v for v in namespace.values() if callable(v) and hasattr(v, '__name__')]
            if not functions:
                raise ValueError("No function found in definition")
            return functions[0]  # Return first defined function

        except Exception as e:
            raise ValidationError(f"Failed to create function from string: {e!s}") from e
