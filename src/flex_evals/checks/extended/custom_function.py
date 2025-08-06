"""
Custom Function check implementation for FEP.

Allows users to provide custom validation functions for flexible argument checking.
"""

import asyncio
from datetime import datetime, UTC
from typing import Any
from collections.abc import Callable
from collections.abc import Mapping

from ..base import BaseAsyncCheck, EvaluationContext
from ...registry import register
from ...exceptions import ValidationError, CheckExecutionError
from ...schemas import CheckResult


@register("custom_function", version="1.0.0")
class CustomFunctionCheck(BaseAsyncCheck):
    """
    Executes user-provided validation functions for flexible argument checking.

    Arguments Schema:
    - validation_function: callable|string - User-provided function or string function definition
    - function_args: dict (required) - Arguments to pass to validation_function (JSONPath OK)

    Results Schema:
    - Returns whatever the validation function returns (no processing or structure imposed)
    """

    async def execute(
        self,
        check_type: str,
        arguments: dict[str, Any],
        context: EvaluationContext,
        check_version: str | None = None,
        check_metadata: dict[str, Any] | None = None,
    ) -> CheckResult:
        """
        Execute custom function check with JSONPath resolution for function_args.

        This method overrides the base class execute() to handle JSONPath resolution
        within the function_args dictionary before delegating to the parent class.

        Args:
            check_type: The check type identifier ("custom_function")
            arguments: Raw check arguments, may contain function_args with JSONPath expressions
            context: Evaluation context with test case and output data
            check_version: Optional version string for the check definition
            check_metadata: Optional metadata from the check definition

        Returns:
            CheckResult: Complete result object with execution status and results
        """
        # Validate that function_args is provided and is a dict
        if "function_args" not in arguments:
            return self._create_error_result(
                check_type=check_type,
                error_type='validation_error',
                error_message="function_args is required for custom_function checks",
                resolved_arguments={},
                evaluated_at=datetime.now(UTC),
                check_version=check_version,
                check_metadata=check_metadata,
                recoverable=False,
            )

        if not isinstance(arguments["function_args"], dict):
            return self._create_error_result(
                check_type=check_type,
                error_type='validation_error',
                error_message="function_args must be a dictionary",
                resolved_arguments={},
                evaluated_at=datetime.now(UTC),
                check_version=check_version,
                check_metadata=check_metadata,
                recoverable=False,
            )

        # Process function_args JSONPath expressions
        try:
            # Resolve JSONPath expressions in function_args
            resolved_function_args = {}
            for key, value in arguments["function_args"].items():
                if isinstance(value, str) and value.startswith("$."):
                    # This looks like a JSONPath expression
                    resolved_result = self._resolver.resolve_argument(
                        value,
                        context.context_dict,
                    )
                    resolved_function_args[key] = resolved_result["value"]
                else:
                    # Not a JSONPath expression, use as-is
                    resolved_function_args[key] = value

            # Create modified arguments with resolved function_args
            modified_arguments = arguments.copy()
            modified_arguments["function_args"] = resolved_function_args

        except Exception as e:
            # JSONPath resolution failed - return error result
            return self._create_error_result(
                check_type=check_type,
                error_type='jsonpath_error',
                error_message=f"Error resolving function_args: {e}",
                resolved_arguments={},
                evaluated_at=datetime.now(UTC),
                check_version=check_version,
                check_metadata=check_metadata,
                recoverable=False,
            )

        # Delegate to parent class for standard execution
        return await super().execute(
            check_type,
            modified_arguments,
            context,
            check_version,
            check_metadata,
        )

    async def __call__(
        self,
        validation_function: Callable | str,
        function_args: dict,
    ) -> Any:  # noqa: ANN401
        """Execute custom function check with resolved arguments."""
        # Convert string function to callable if needed
        if isinstance(validation_function, str):
            validation_function = self._string_to_function(validation_function)
        elif not callable(validation_function):
            raise ValidationError(
                "validation_function must be callable or string function definition",
            )

        # Prepare function arguments (these are already resolved values from JSONPath)
        if not isinstance(function_args, Mapping):
            raise ValidationError("function_args must be a dictionary")
        fn_kwargs = dict(function_args)

        try:
            # Execute the validation function (handle both sync and async)
            if asyncio.iscoroutinefunction(validation_function):
                result = await validation_function(**fn_kwargs)
            else:
                result = validation_function(**fn_kwargs)

            # Return whatever the function returned - no processing
            return result

        except Exception as e:
            raise CheckExecutionError(
                f"Error executing validation function: {e!s}",
            ) from e

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
