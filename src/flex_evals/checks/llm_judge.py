"""
Combined LLMJudgeCheck implementation for FEP.

This module implements the `llm_judge` check type, which uses Large Language Models (LLMs)
to evaluate system outputs against complex, nuanced criteria that cannot be easily
expressed with traditional programmatic checks.
"""

import asyncio
import json
import re
from datetime import datetime, UTC
from typing import Any, TypeVar
from collections.abc import Callable, Awaitable
from pydantic import Field, BaseModel

from .base import BaseAsyncCheck, EvaluationContext
from ..registry import register
from ..exceptions import ValidationError, CheckExecutionError, JSONPathError
from ..constants import CheckType
from ..schemas import CheckResult

# Type variable for the response format model
T = TypeVar('T', bound=BaseModel)


@register(CheckType.LLM_JUDGE, version='1.0.0')
class LLMJudgeCheck(BaseAsyncCheck):
    """Uses an LLM to evaluate outputs against complex, nuanced criteria."""

    # Pydantic fields with validation
    prompt: str = Field(..., description='Prompt template with optional {{$.jsonpath}} placeholders')
    response_format: type[BaseModel] = Field(..., description='Pydantic model defining expected LLM response structure')
    llm_function: Any = Field(..., description='Function to call LLM with signature: (prompt, response_format) -> tuple[BaseModel, dict]')

    async def execute(
        self,
        check_type: str,
        arguments: dict[str, Any],
        context: EvaluationContext,
        check_metadata: dict[str, Any] | None = None,
    ) -> CheckResult:
        """
        Execute LLM judge check with template processing.

        This method overrides the base class execute() to implement the two-phase
        execution model required for prompt template processing:

        PHASE 1 - TEMPLATE PROCESSING:
        1. Checks if the prompt argument contains template placeholders
        2. If templates exist, processes {{$.jsonpath}} placeholders
        3. Creates modified arguments with processed prompt

        PHASE 2 - STANDARD EXECUTION:
        4. Delegates to parent class execute() with processed arguments
        """
        # Get version from registry using the class
        check_version = self._get_version()

        # PHASE 1: Template processing (if needed)
        if "prompt" in arguments and isinstance(arguments["prompt"], str):
            try:
                # Process prompt template by resolving {{$.jsonpath}} placeholders
                processed_prompt = self._process_prompt_template(
                    template=arguments["prompt"],
                    context=context,
                )

                # Create modified arguments with processed prompt
                modified_arguments = arguments.copy()
                modified_arguments["prompt"] = processed_prompt

            except JSONPathError as e:
                # Template processing failed - return error result
                return self._create_error_result(
                    check_type=check_type,
                    error_type='jsonpath_error',
                    error_message=f"Error processing prompt template: {e}",
                    resolved_arguments={},
                    evaluated_at=datetime.now(UTC),
                    check_version=check_version,
                    check_metadata=check_metadata,
                    recoverable=False,
                )
        else:
            # No template processing needed
            modified_arguments = arguments

        # PHASE 2: Standard execution with processed arguments
        # Delegate to parent class for standard JSONPath resolution and execution
        return await super().execute(
            check_type,
            modified_arguments,
            context,
            check_metadata,
        )

    async def __call__(
        self,
        prompt: str,
        response_format: type[T],
        llm_function: Callable[
            [str, type[T]],
            tuple[T, dict[str, Any]] | Awaitable[tuple[T, dict[str, Any]]],
        ],
        **kwargs: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        """
        Execute LLM evaluation with fully processed arguments.

        This method expects FULLY PROCESSED arguments with NO templates
        or JSONPath expressions. All template processing should have been
        completed by the execute() method before this is called.

        Args:
            prompt: Fully processed prompt string (no templates)
            response_format: Pydantic model class defining expected response structure
            llm_function: Callable to invoke LLM with the prompt
            **kwargs: Additional arguments (passed through but not used)

        Returns:
            Dictionary with response fields and judge_metadata
        """
        # Validate argument types
        if not isinstance(prompt, str):
            raise ValidationError("prompt must be a string")

        if not (isinstance(response_format, type) and issubclass(response_format, BaseModel)):
            raise ValidationError("response_format must be a Pydantic BaseModel class")

        if not callable(llm_function):
            raise ValidationError("llm_function must be callable")

        try:
            # Execute LLM evaluation with the fully processed prompt
            llm_response = await self._call_llm_function(llm_function, prompt, response_format)

            # Expect tuple of (model_response, metadata)
            if not isinstance(llm_response, tuple) or len(llm_response) != 2:
                raise ValidationError(
                    "llm_function must return tuple of (BaseModel, metadata_dict)",
                )

            model_response, metadata = llm_response
            validated_response = self._validate_response_format(model_response, response_format)

            # Preserve response structure and add judge_metadata field
            result = validated_response.copy()
            result["judge_metadata"] = metadata or {}
            return result

        except Exception as e:
            raise CheckExecutionError(f"Error in LLM judge evaluation: {e!s}") from e

    def _process_prompt_template(self, template: str, context: EvaluationContext) -> str:
        """
        Process prompt template by resolving {{$.jsonpath}} placeholders.

        Finds all template placeholders in the format {{$.jsonpath}} and replaces
        them with values resolved from the evaluation context.
        """
        # Find all JSONPath placeholders using regex
        placeholder_pattern = r'\{\{\$\.([^}]+)\}\}'
        placeholders = re.findall(placeholder_pattern, template)

        if not placeholders:
            # No placeholders to process
            return template

        processed_prompt = template

        for placeholder in placeholders:
            jsonpath_expr = f"$.{placeholder}"
            try:
                # Use the existing JSONPath resolver from base class
                resolved_result = self._resolver.resolve_argument(
                    jsonpath_expr,
                    context.context_dict,
                )
                resolved_value = resolved_result.get("value")

                # Convert resolved value to string for substitution
                if isinstance(resolved_value, dict | list):
                    # JSON-serialize complex objects
                    value_str = json.dumps(resolved_value, ensure_ascii=False)
                elif resolved_value is None:
                    # Convert null to empty string
                    value_str = ""
                else:
                    # Convert primitive types to string
                    value_str = str(resolved_value)

                # Replace the placeholder with the resolved value
                placeholder_full = f"{{{{$.{placeholder}}}}}"
                processed_prompt = processed_prompt.replace(placeholder_full, value_str)

            except Exception as e:
                raise JSONPathError(
                    f"Failed to resolve template placeholder '{{{{$.{placeholder}}}}}': {e}",
                    jsonpath_expression=jsonpath_expr,
                ) from e

        return processed_prompt

    async def _call_llm_function(
        self,
        llm_function: Callable[
            [str, type[T]],
            tuple[T, dict[str, Any]] | Awaitable[tuple[T, dict[str, Any]]],
        ],
        prompt: str,
        response_format: type[T],
    ) -> tuple[T, dict[str, Any]]:
        """Call the user-provided LLM function with proper error handling."""
        try:
            # Handle both sync and async LLM functions
            if asyncio.iscoroutinefunction(llm_function):
                result = await llm_function(prompt, response_format)
            else:
                result = llm_function(prompt, response_format)
            return result

        except Exception as e:
            raise CheckExecutionError(f"LLM function failed: {e!s}") from e

    def _validate_response_format(
        self,
        response: T,
        response_format: type[T],
    ) -> dict[str, Any]:
        """
        Validate LLM response and convert to standardized dict format.

        Supports BaseModel instances, JSON strings, and dictionaries.
        """
        # Handle BaseModel instances
        if isinstance(response, BaseModel):
            return response.model_dump()

        # Handle JSON string responses
        if isinstance(response, str):
            try:
                parsed_response = json.loads(response)
                model_instance = response_format(**parsed_response)
                return model_instance.model_dump()
            except json.JSONDecodeError as e:
                raise ValidationError(f"LLM response is not valid JSON: {e!s}")
            except Exception as e:
                raise ValidationError(f"Failed to create model from response: {e!s}")

        # Handle dictionary responses
        if isinstance(response, dict):
            try:
                model_instance = response_format(**response)
                return model_instance.model_dump()
            except Exception as e:
                raise ValidationError(f"Failed to create model from response dict: {e!s}")

        # Reject unsupported response types
        raise ValidationError(f"Unexpected response type: {type(response)}")
