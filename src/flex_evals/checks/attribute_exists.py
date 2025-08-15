"""
Combined AttributeExistsCheck implementation for FEP.

Combines schema validation with execution logic in a single class.
"""

from typing import Any
from datetime import datetime, UTC
from pydantic import Field, field_validator

from .base import BaseCheck, EvaluationContext, JSONPath, _convert_to_jsonpath
from ..registry import register
from ..constants import CheckType
from ..exceptions import JSONPathError, ValidationError
from ..schemas import CheckResult


@register(CheckType.ATTRIBUTE_EXISTS, version='1.0.0')
class AttributeExistsCheck(BaseCheck):
    """Tests whether an attribute/field exists as defined by the JSONPath; useful for checking if optional fields."""  # noqa: E501

    # Pydantic fields with validation - path must be JSONPath, negate can be literal or JSONPath
    path: JSONPath = Field(
        ...,
        description="JSONPath expression to check for existence (e.g., \"$.output.error\")",
    )
    negate: bool | JSONPath = Field(
        False,
        description="If true, passes when attribute does NOT exist",
    )

    @field_validator('path', mode='before')
    @classmethod
    def convert_path_jsonpath(cls, value: object) -> object | JSONPath:
        """Convert any strings to JSONPath objects for path field."""
        if isinstance(value, str):
            return JSONPath(expression=value)
        return value

    @field_validator('negate', mode='before')
    @classmethod
    def convert_negate_jsonpath(cls, value: object) -> object | JSONPath:
        """Convert JSONPath-like strings to JSONPath objects for negate field."""
        return _convert_to_jsonpath(value)

    def __call__(self) -> dict[str, Any]:
        """
        Should never be called directly for AttributeExistsCheck.

        The check requires special JSONPath handling in the execute method
        to distinguish between path non-existence and actual errors.
        """
        raise RuntimeError(
            "AttributeExistsCheck requires special JSONPath handling - use execute() method",
        )

    def execute(
        self,
        context: EvaluationContext,
        check_metadata: dict[str, Any] | None = None,
    ) -> CheckResult:
        """
        Execute the attribute existence check.

        This overrides the base execute method to handle JSONPath resolution specially.
        Instead of treating JSONPath errors as failures, we use them to determine
        attribute existence.
        """
        evaluated_at = datetime.now(UTC)

        # Get version from registry using the class
        check_version = self._get_version()
        check_type = str(self.check_type)

        try:
            # Resolve negate field if it's JSONPath (path field stays as JSONPath object)
            if isinstance(self.negate, JSONPath):
                try:
                    negate_result = self._resolver.resolve_argument(
                        self.negate.expression,
                        context.context_dict,
                    )
                    negate = negate_result.get("value")
                except Exception as e:
                    raise ValidationError(f"Failed to resolve negate JSONPath: {e}") from e
            else:
                negate = self.negate

            if not isinstance(negate, bool):
                raise ValidationError("'negate' field must resolve to a boolean")

            # Use the path JSONPath expression directly
            path_expression = self.path.expression

            # Try to resolve the JSONPath to determine existence
            try:
                resolved_arg = self._resolver.resolve_argument(
                    path_expression, context.context_dict,
                )
                # If we get here, the path exists
                attribute_exists = True
                resolved_arguments = {'path': resolved_arg, 'negate': {'value': negate}}
            except JSONPathError:
                # Path doesn't exist
                attribute_exists = False
                resolved_arguments = {
                    'path': {'jsonpath': path_expression, 'value': None},
                    'negate': {'value': negate},
                }

            # Apply logic: pass if (exists and not negate) or (not exists and negate)
            passed = attribute_exists if not negate else not attribute_exists

            results = {'passed': passed}

            # Create successful result
            return CheckResult(
                check_type=check_type,
                check_version=check_version,
                status='completed',
                results=results,
                resolved_arguments=resolved_arguments,
                evaluated_at=evaluated_at,
                metadata=check_metadata,
            )

        except ValidationError as e:
            return self._create_error_result(
                check_type=check_type,
                error_type='validation_error',
                error_message=str(e),
                resolved_arguments={},
                evaluated_at=evaluated_at,
                check_version=check_version,
                check_metadata=check_metadata,
                recoverable=False,
            )

        except Exception as e:
            return self._create_error_result(
                check_type=check_type,
                error_type='unknown_error',
                error_message=f"Unexpected error during check execution: {e!s}",
                resolved_arguments={},
                evaluated_at=evaluated_at,
                check_version=check_version,
                check_metadata=check_metadata,
                recoverable=False,
            )
