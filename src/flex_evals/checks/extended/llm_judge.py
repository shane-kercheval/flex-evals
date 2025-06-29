"""
LLM Judge check implementation for FEP.

Uses LLM to evaluate outputs against complex criteria with structured responses.
"""

import asyncio
import json
import re
from typing import Any
from collections.abc import Callable

from ..base import BaseAsyncCheck, EvaluationContext
from ...registry import register
from ...exceptions import ValidationError, CheckExecutionError
from ...jsonpath_resolver import JSONPathResolver


@register("llm_judge", version="1.0.0")
class LlmJudgeCheck(BaseAsyncCheck):
    """
    Uses an LLM to evaluate outputs against complex criteria.

    Arguments Schema:
    - prompt: string - Prompt template with {{$.jsonpath}} placeholders for dynamic substitution
    - response_format: object - JSON Schema defining expected LLM response structure
    - llm_function: async callable - User-provided function to call LLM

    Results Schema:
    Dynamic structure matching response_format exactly. The LLM response is validated
    against the schema and returned as-is without additional wrapper fields.
    """

    def __init__(self):
        super().__init__()
        self.jsonpath_resolver = JSONPathResolver()

    async def __call__(
        self, prompt: str, response_format: dict, llm_function: Callable,
    ) -> dict[str, Any]:
        """Execute LLM judge check with direct arguments."""
        # Validate arguments
        if not isinstance(prompt, str):
            raise ValidationError("prompt must be a string")

        if not isinstance(response_format, dict):
            raise ValidationError("response_format must be a dictionary (JSON Schema)")

        if not callable(llm_function):
            raise ValidationError("llm_function must be callable")

        try:
            # For now, use the prompt directly without template processing
            # Template processing can be added back later with enhanced JSONPath resolver
            llm_response = await self._call_llm_function(llm_function, prompt)

            # Validate response against schema
            return self._validate_response_format(llm_response, response_format)

        except Exception as e:
            raise CheckExecutionError(
                f"Error in LLM judge evaluation: {e!s}",
            ) from e

    def _process_prompt_template(self, template: str, context: EvaluationContext) -> str:
        """Replace {{$.jsonpath}} placeholders in prompt template with resolved values."""
        # Find all JSONPath placeholders in the format {{$.path}}
        placeholder_pattern = r'\{\{\$\.([^}]+)\}\}'
        placeholders = re.findall(placeholder_pattern, template)

        processed_prompt = template
        for placeholder in placeholders:
            jsonpath_expr = f"$.{placeholder}"
            try:
                # Convert EvaluationContext to dict for resolution
                context_dict = {
                    "test_case": {
                        "id": context.test_case.id,
                        "input": context.test_case.input,
                        "expected": context.test_case.expected,
                        "metadata": context.test_case.metadata,
                    },
                    "output": {
                        "value": context.output.value,
                        "metadata": context.output.metadata,
                    },
                }

                # Resolve JSONPath expression
                resolved_result = self.jsonpath_resolver.resolve_argument(
                    jsonpath_expr, context_dict,
                )
                resolved_value = resolved_result.get("value")

                # Convert to string for substitution
                if isinstance(resolved_value, dict | list):
                    value_str = json.dumps(resolved_value, ensure_ascii=False)
                else:
                    value_str = str(resolved_value) if resolved_value is not None else ""

                # Replace the placeholder
                placeholder_full = f"{{{{$.{placeholder}}}}}"
                processed_prompt = processed_prompt.replace(placeholder_full, value_str)

            except Exception:
                # If JSONPath resolution fails, leave placeholder as-is for now
                # This allows for more graceful degradation
                continue

        return processed_prompt

    async def _call_llm_function(self, llm_function: Callable, prompt: str) -> object:
        """Call the user-provided LLM function with error handling."""
        try:
            # Handle both sync and async LLM functions
            if asyncio.iscoroutinefunction(llm_function):
                result = await llm_function(prompt)
            else:
                result = llm_function(prompt)
            return result

        except Exception as e:
            raise CheckExecutionError(
                f"LLM function failed: {e!s}",
            ) from e

    def _validate_response_format(
            self,
            response: object,
            schema: dict[str, Any],
        ) -> dict[str, Any]:
        """Validate LLM response against the provided JSON schema."""
        # First, try to parse response as JSON if it's a string
        if isinstance(response, str):
            try:
                parsed_response = json.loads(response)
            except json.JSONDecodeError as e:
                raise ValidationError(f"LLM response is not valid JSON: {e!s}")
        else:
            parsed_response = response

        # Basic schema validation
        return self._validate_schema(parsed_response, schema)


    def _validate_schema(self, data: object, schema: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR0911, PLR0912
        """
        Basic JSON Schema validation.

        This is a simplified implementation that handles common schema patterns.
        For production use, consider using a full JSON Schema validation library.
        """
        schema_type = schema.get("type")

        if schema_type == "object":
            if not isinstance(data, dict):
                raise ValidationError(f"Expected object, got {type(data).__name__}")

            # Validate required properties
            required = schema.get("required", [])
            for prop in required:
                if prop not in data:
                    raise ValidationError(f"Missing required property: {prop}")

            # Validate properties
            properties = schema.get("properties", {})
            validated_data = {}
            for key, value in data.items():
                if key in properties:
                    validated_data[key] = self._validate_schema(value, properties[key])
                else:
                    # Allow additional properties by default
                    validated_data[key] = value

            return validated_data

        if schema_type == "string":
            if not isinstance(data, str):
                raise ValidationError(f"Expected string, got {type(data).__name__}")
            return data

        if schema_type == "number":
            if not isinstance(data, int | float):
                raise ValidationError(f"Expected number, got {type(data).__name__}")
            return data

        if schema_type == "integer":
            if not isinstance(data, int):
                raise ValidationError(f"Expected integer, got {type(data).__name__}")
            return data

        if schema_type == "boolean":
            if not isinstance(data, bool):
                raise ValidationError(f"Expected boolean, got {type(data).__name__}")
            return data

        if schema_type == "array":
            if not isinstance(data, list):
                raise ValidationError(f"Expected array, got {type(data).__name__}")

            items_schema = schema.get("items")
            if items_schema:
                validated_items = []
                for item in data:
                    validated_items.append(self._validate_schema(item, items_schema))
                return validated_items
            return data

        # If no type specified or unknown type, return as-is
        return data
