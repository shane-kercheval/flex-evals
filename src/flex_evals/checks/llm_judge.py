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
from pydantic import Field, BaseModel, field_validator

from .base import BaseAsyncCheck, EvaluationContext, JSONPath, _convert_to_jsonpath
from ..registry import register
from ..exceptions import ValidationError, CheckExecutionError, JSONPathError
from ..constants import CheckType
from ..schemas import CheckResult

# Type variable for the response format model
T = TypeVar('T', bound=BaseModel)


@register(CheckType.LLM_JUDGE, version='1.0.0')
class LLMJudgeCheck(BaseAsyncCheck):
    """Uses an LLM to evaluate outputs against complex, nuanced criteria."""

    # Pydantic fields with validation - can be literals or JSONPath objects
    prompt: str | JSONPath = Field(
        ...,
        description="Prompt template with optional {{$.jsonpath}} placeholders",
    )
    response_format: type[BaseModel] = Field(
        ...,
        description="Pydantic model defining expected LLM response structure",
    )
    llm_function: Any = Field(
        ...,
        description="Function to call LLM with signature: (prompt, response_format) -> tuple[BaseModel, dict]",  # noqa: E501
    )

    @field_validator('prompt', mode='before')
    @classmethod
    def convert_jsonpath(cls, value: object) -> object | JSONPath:
        """Convert JSONPath-like strings to JSONPath objects."""
        return _convert_to_jsonpath(value)

    async def execute(
        self,
        context: EvaluationContext,
        check_metadata: dict[str, Any] | None = None,
    ) -> CheckResult:
        """
        Execute LLM judge check with template processing.

        This method overrides the base class execute() to implement template processing
        for prompts that contain {{$.jsonpath}} placeholders before standard field resolution.
        """
        evaluated_at = datetime.now(UTC)
        check_version = self._get_version()
        check_type = str(self.check_type)

        try:
            # PHASE 1: Handle template processing if prompt contains {{$.jsonpath}} placeholders
            prompt_to_use = self.prompt
            if isinstance(self.prompt, str) and '{{$.' in self.prompt:
                try:
                    # Process prompt template by resolving {{$.jsonpath}} placeholders
                    prompt_to_use = self._process_prompt_template(
                        template=self.prompt,
                        context=context,
                    )
                except JSONPathError as e:
                    # Template processing failed - return error result
                    return self._create_error_result(
                        check_type=check_type,
                        error_type='jsonpath_error',
                        error_message=f"Error processing prompt template: {e}",
                        resolved_arguments={},
                        evaluated_at=evaluated_at,
                        check_version=check_version,
                        check_metadata=check_metadata,
                        recoverable=False,
                    )
            elif isinstance(self.prompt, JSONPath):
                # Resolve JSONPath prompt field
                try:
                    prompt_result = self._resolver.resolve_argument(
                        self.prompt.expression,
                        context.context_dict,
                    )
                    prompt_to_use = prompt_result.get("value")
                except Exception as e:
                    return self._create_error_result(
                        check_type=check_type,
                        error_type='jsonpath_error',
                        error_message=f"Failed to resolve prompt JSONPath: {e}",
                        resolved_arguments={},
                        evaluated_at=evaluated_at,
                        check_version=check_version,
                        check_metadata=check_metadata,
                        recoverable=False,
                    )

            # Create a copy of self with the processed prompt
            # We need to temporarily set the prompt to the processed value
            original_prompt = self.prompt
            self.prompt = prompt_to_use

            try:
                # PHASE 2: Execute with processed prompt using __call__()
                results = await self()
                # Build resolved arguments for the result
                resolved_arguments = {
                    'prompt': {
                        'value': prompt_to_use,
                        'resolved_from': 'template_processed' if '{{$.' in str(original_prompt) else 'literal',  # noqa: E501
                    },
                    'response_format': {
                        'value': str(self.response_format),
                        'resolved_from': 'literal',
                    },
                    'llm_function': {'value': str(self.llm_function), 'resolved_from': 'literal'},
                }
                return CheckResult(
                    check_type=check_type,
                    check_version=check_version,
                    status='completed',
                    results=results,
                    resolved_arguments=resolved_arguments,
                    evaluated_at=evaluated_at,
                    metadata=check_metadata,
                )
            finally:
                # Restore original prompt
                self.prompt = original_prompt

        except Exception as e:
            return self._create_error_result(
                check_type=check_type,
                error_type='execution_error',
                error_message=f"LLM judge execution failed: {e}",
                resolved_arguments={},
                evaluated_at=evaluated_at,
                check_version=check_version,
                check_metadata=check_metadata,
                recoverable=False,
            )

    async def __call__(self) -> dict[str, Any]:
        """
        Execute LLM evaluation with resolved Pydantic fields.

        All JSONPath objects should have been resolved by execute() before this is called.

        Returns:
            Dictionary with response fields and judge_metadata

        Raises:
            RuntimeError: If any field contains unresolved JSONPath objects
        """
        # Validate that all fields are resolved (no JSONPath objects remain)
        if isinstance(self.prompt, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'prompt' field: {self.prompt}")

        # Validate argument types
        if not isinstance(self.prompt, str):
            raise ValidationError("prompt must be a string")

        if not (isinstance(self.response_format, type) and issubclass(self.response_format, BaseModel)):  # noqa: E501
            raise ValidationError("response_format must be a Pydantic BaseModel class")

        if not callable(self.llm_function):
            raise ValidationError("llm_function must be callable")

        try:
            # Execute LLM evaluation with the fully processed prompt
            llm_response = await self._call_llm_function(
                self.llm_function, self.prompt, self.response_format,
            )

            # Expect tuple of (model_response, metadata)
            if not isinstance(llm_response, tuple) or len(llm_response) != 2:
                raise ValidationError(
                    "llm_function must return tuple of (BaseModel, metadata_dict)",
                )

            model_response, metadata = llm_response
            validated_response = self._validate_response_format(
                model_response, self.response_format,
            )

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
