"""
LLM Judge check implementation for FEP.

This module implements the `llm_judge` check type, which uses Large Language Models (LLMs)
to evaluate system outputs against complex, nuanced criteria that cannot be easily
expressed with traditional programmatic checks.

EXECUTION FLOW AND DESIGN:
==========================

This check implementation uses a two-stage execution process to handle the complex
requirement of processing prompt templates containing JSONPath expressions:

1. TEMPLATE PROCESSING (execute() method):
   - Receives raw arguments that may contain prompt templates like:
     "User Input: {{$.test_case.input.message}}"
   - Processes {{$.jsonpath}} placeholders by resolving them against evaluation context
   - Produces a fully substituted prompt with actual values

2. STANDARD EXECUTION (__call__ method):
   - Receives fully processed arguments (no templates, no JSONPath expressions)
   - Executes the LLM evaluation with the processed prompt
   - Returns structured results according to the specified response format

This separation ensures:
- Clean separation of concerns between template processing and LLM execution
- Reuse of existing JSONPath resolution infrastructure
- Proper error handling at each stage
- Maintainable and testable code

TEMPLATE FORMAT:
===============

Prompt templates use the format {{$.jsonpath}} for dynamic substitution:

Example:
    prompt: |
        Evaluate the following interaction:

        User Input: {{$.test_case.input.user_message}}
        AI Response: {{$.output.value.response}}
        Expected Quality: {{$.test_case.expected.quality_level}}

        Rate the response quality from 1-10.

The JSONPath expressions will be resolved against the evaluation context and substituted
with actual values before the prompt is sent to the LLM.

USAGE EXAMPLE:
=============

    check = {
        "type": "llm_judge",
        "arguments": {
            "prompt": "Evaluate: User asked '{{$.test_case.input}}', AI responded '{{$.output.value}}'",
            "response_format": QualityAssessment,  # Pydantic model
            "llm_function": my_llm_function
        }
    }

Where QualityAssessment might be:
    class QualityAssessment(BaseModel):
        score: int = Field(ge=1, le=10)
        reasoning: str
        is_helpful: bool

And my_llm_function must return:
    return (
        QualityAssessment(score=8, reasoning="Good answer", is_helpful=True),
        {"cost_usd": 0.002, "tokens_used": 150, "model": "gpt-4o-mini"}
    )
"""  # noqa: E501

import asyncio
import json
import re
from datetime import datetime, UTC
from typing import Any, TypeVar
from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from ..base import BaseAsyncCheck, EvaluationContext
from ...registry import register
from ...exceptions import ValidationError, CheckExecutionError, JSONPathError
from ...constants import CheckType
from ...schemas import CheckResult

# Type variable for the response format model
T = TypeVar('T', bound=BaseModel)


@register(CheckType.LLM_JUDGE, version="1.0.0")
class LlmJudgeCheck(BaseAsyncCheck):
    """
    Uses an LLM to evaluate outputs against complex, nuanced criteria.

    This check allows for sophisticated evaluation of system outputs using Large Language
    Models, enabling assessment of qualities like helpfulness, accuracy, tone, coherence,
    and other subjective criteria that are difficult to capture with programmatic checks.

    EXECUTION ARCHITECTURE:
    =======================

    This implementation overrides the base class execution flow to handle prompt template
    processing. The execution happens in two distinct phases:

    Phase 1 - Template Processing (execute() method):
        - Receives raw check arguments including prompt templates
        - Processes {{$.jsonpath}} placeholders in the prompt
        - Resolves JSONPath expressions against evaluation context
        - Substitutes placeholders with actual values
        - Passes processed arguments to base class execution

    Phase 2 - LLM Execution (__call__ method):
        - Receives fully processed, concrete arguments
        - NO template processing occurs here
        - NO JSONPath resolution occurs here
        - Executes LLM evaluation with processed prompt
        - Returns structured results

    This design ensures clean separation between template processing logic and LLM
    execution logic, making the code more maintainable and testable.

    ARGUMENTS SCHEMA:
    ================

    Required Arguments:
    - prompt (str): Prompt template with optional {{$.jsonpath}} placeholders
    - response_format (type[BaseModel]): Pydantic model defining expected LLM response structure
    - llm_function (callable): Function to call LLM with signature:
        (prompt: str, response_format: type[BaseModel]) -> BaseModel | tuple[BaseModel, dict[str, Any]]

    Optional Arguments:
    - Any additional arguments will be passed through to __call__ after standard processing

    TEMPLATE SYNTAX:
    ===============

    Prompt templates support dynamic value substitution using {{$.jsonpath}} syntax:

    - {{$.test_case.input}} - Access the test case input
    - {{$.output.value}} - Access the system output value
    - {{$.test_case.expected}} - Access expected output
    - {{$.output.metadata.execution_time}} - Access nested metadata

    Complex values (objects/arrays) are JSON-serialized for substitution.

    RESULTS SCHEMA:
    ==============

    The results structure preserves the judge's response fields with metadata in a separate field:
    - Response fields from the BaseModel are preserved as-is in the root result object
    - Metadata from the LLM function is added as a separate "judge_metadata" field
    - This ensures response structure integrity and clear separation of concerns

    Example Response Format:
        class EvaluationResult(BaseModel):
            quality_score: int = Field(ge=1, le=5)
            is_helpful: bool
            reasoning: str
            detected_issues: list[str] = []

    Would produce results like:
        {
            "quality_score": 4,
            "is_helpful": true,
            "reasoning": "Response directly answers the question with accurate information",
            "detected_issues": [],
            "judge_metadata": {
                "cost_usd": 0.0023,
                "tokens_used": 150,
                "response_time_ms": 842,
                "model_version": "gpt-4o-mini-2024-07-02"
            }
        }

    If the response format itself contains a metadata field, it is preserved:
        {
            "passed": true,
            "score": 8,
            "metadata": {"internal_data": "value"},  # from response format
            "judge_metadata": {"cost_usd": 0.002}    # from LLM function
        }

    ERROR HANDLING:
    ==============

    Template Processing Errors:
    - Invalid JSONPath expressions in templates result in jsonpath_error
    - Missing context data for placeholders results in jsonpath_error

    LLM Execution Errors:
    - LLM function failures result in unknown_error
    - Invalid response format validation results in validation_error
    - Malformed JSON responses result in validation_error

    IMPLEMENTATION NOTES:
    ====================

    1. Template processing happens BEFORE standard JSONPath argument resolution
    2. The __call__ method expects fully processed arguments with no templates
    3. LLM functions can be either synchronous or asynchronous
    4. Response validation ensures type safety and protocol compliance
    5. Complex objects in templates are JSON-serialized for string substitution
    """  # noqa: E501

    async def execute(
        self,
        check_type: str,
        arguments: dict[str, Any],
        context: EvaluationContext,
        check_version: str | None = None,
        check_metadata: dict[str, Any] | None = None,
    ) -> CheckResult:
        """
        Execute LLM judge check with template processing.

        This method overrides the base class execute() to implement the two-phase
        execution model required for prompt template processing:

        PHASE 1 - TEMPLATE PROCESSING:
        ==============================

        1. Checks if the prompt argument contains template placeholders
        2. If templates exist, processes {{$.jsonpath}} placeholders:
           - Extracts JSONPath expressions from template syntax
           - Resolves each expression against the evaluation context
           - Substitutes placeholders with resolved values
           - Handles JSON serialization for complex objects
        3. Creates modified arguments with processed prompt

        PHASE 2 - STANDARD EXECUTION:
        =============================

        4. Delegates to parent class execute() with processed arguments
        5. Parent handles standard JSONPath resolution for other arguments
        6. Parent calls __call__ with fully resolved arguments
        7. __call__ executes LLM evaluation and returns results

        ERROR HANDLING:
        ==============

        Template processing errors (invalid JSONPath, missing data) are caught
        and returned as proper CheckResult error objects with jsonpath_error type.
        This ensures protocol compliance and proper error reporting.

        Args:
            check_type: The check type identifier ("llm_judge")
            arguments: Raw check arguments, may contain prompt templates
            context: Evaluation context with test case and output data
            check_version: Optional version string for the check definition
            check_metadata: Optional metadata from the check definition

        Returns:
            CheckResult: Complete result object with execution status, results,
                        resolved arguments, and metadata

        Raises:
            No exceptions - all errors are captured and returned as CheckResult
            objects with appropriate error status and details
        """
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
            check_version,
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
    ) -> dict[str, Any]:
        """
        Execute LLM evaluation with fully processed arguments.

        IMPORTANT: This method expects FULLY PROCESSED arguments with NO templates
        or JSONPath expressions. All template processing and JSONPath resolution
        should have been completed by the execute() method before this is called.

        EXECUTION FLOW:
        ==============

        1. Validates argument types and constraints
        2. Calls the provided LLM function with the processed prompt
        3. Validates the LLM response against the expected format
        4. Returns structured results as a dictionary

        ARGUMENT EXPECTATIONS:
        =====================

        prompt (str):
            - Must be a fully processed string with no template placeholders
            - Should contain the final prompt text to send to the LLM
            - All {{$.jsonpath}} expressions should already be resolved

        response_format (type[BaseModel]):
            - Must be a Pydantic BaseModel class (not instance)
            - Defines the expected structure of the LLM response
            - Used for response validation and type safety

        llm_function (callable):
            - Function or method to call for LLM evaluation
            - Must accept (prompt: str, response_format: type[BaseModel])
            - Can be synchronous or asynchronous
            - Must return tuple of (BaseModel instance, metadata dict)

        **kwargs:
            - Additional arguments are accepted but not used
            - Allows for extensibility without breaking the interface

        RESPONSE PROCESSING:
        ===================

        The LLM response must be a tuple of (BaseModel, metadata_dict).
        Processing steps:
        1. Validate tuple format and extract (model, metadata)
        2. Validate model instance against response_format
        3. Convert to dict via model_dump()
        4. Return structured response with response and metadata

        Args:
            prompt: Fully processed prompt string (no templates)
            response_format: Pydantic model class defining expected response structure
            llm_function: Callable to invoke LLM with the prompt
            **kwargs: Additional arguments (passed through but not used)

        Returns:
            dict[str, Any]: Structured results with:
                - Response fields preserved as-is in root
                - Metadata fields added as separate "judge_metadata" field

        Raises:
            ValidationError: If arguments are invalid or response format validation fails
            CheckExecutionError: If LLM function execution fails

        Example:
            # This is what __call__ receives after execute() processes templates:
            results = await check(
                prompt="Evaluate: User asked 'What is Python?', AI responded 'Python is a programming language'",
                response_format=QualityAssessment,
                llm_function=my_llm_function
            )
            # Returns: {
            #     "score": 5,
            #     "reasoning": "Accurate and helpful response",
            #     "judge_metadata": {
            #         "cost_usd": 0.002,
            #         "tokens_used": 150,
            #         ...
            #     }
            # }
        """  # noqa: E501
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

        This method handles the template processing phase of LLM judge execution.
        It finds all template placeholders in the format {{$.jsonpath}} and replaces
        them with values resolved from the evaluation context.

        TEMPLATE SYNTAX:
        ===============

        Placeholders use the format: {{$.jsonpath.expression}}

        Examples:
        - {{$.test_case.input}} -> Resolves to the test case input value
        - {{$.output.value.score}} -> Resolves to nested score in output
        - {{$.test_case.metadata.source}} -> Resolves to metadata field

        VALUE CONVERSION:
        ================

        Resolved values are converted to strings for template substitution:
        - Strings: Used as-is
        - Numbers/Booleans: Converted to string representation
        - Objects/Arrays: JSON-serialized with ensure_ascii=False
        - None/null: Converted to empty string

        ERROR HANDLING:
        ==============

        If a JSONPath expression cannot be resolved:
        - JSONPathError is raised with details about the failed expression
        - The error includes the original JSONPath for debugging
        - Template processing stops and error is propagated to execute()

        Args:
            template: Raw prompt template with {{$.jsonpath}} placeholders
            context: Evaluation context containing test case and output data

        Returns:
            str: Processed prompt with all placeholders substituted with actual values

        Raises:
            JSONPathError: If any JSONPath expression cannot be resolved

        Example:
            template = "User asked: {{$.test_case.input}}, AI said: {{$.output.value}}"
            context = EvaluationContext(
                test_case=TestCase(input="What is AI?", ...),
                output=Output(value="AI is artificial intelligence", ...)
            )
            result = self._process_prompt_template(template, context)
            # result = "User asked: What is AI?, AI said: AI is artificial intelligence"
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
        """
        Call the user-provided LLM function with proper error handling.

        This method handles both synchronous and asynchronous LLM functions,
        providing a consistent interface for LLM execution regardless of the
        underlying implementation.

        Args:
            llm_function: User-provided function to call for LLM evaluation
            prompt: Processed prompt string to send to LLM
            response_format: Expected response format (Pydantic model class)

        Returns:
            tuple[T, dict[str, Any]]: Tuple of (BaseModel instance, metadata dict)

        Raises:
            CheckExecutionError: If LLM function execution fails
        """
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

        This method ensures that the LLM response conforms to the expected format
        and converts it to a dictionary for consistent result handling.

        SUPPORTED RESPONSE TYPES:
        ========================

        1. BaseModel instance: Converted via model_dump()
        2. JSON string: Parsed and validated against response_format
        3. Dictionary: Validated against response_format
        4. Other types: Rejected with ValidationError

        Args:
            response: Response from LLM function
            response_format: Expected Pydantic model class

        Returns:
            dict[str, Any]: Validated response as dictionary

        Raises:
            ValidationError: If response cannot be validated or converted
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

