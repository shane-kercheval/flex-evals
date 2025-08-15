"""
Combined AttributeExistsCheck implementation for FEP.

Combines schema validation with execution logic in a single class.
"""

from typing import Any
from datetime import datetime, UTC
from pydantic import Field

from .base import BaseCheck, EvaluationContext, RequiredJSONPath
from ..registry import register
from ..constants import CheckType
from ..exceptions import JSONPathError, ValidationError
from ..schemas import CheckResult


@register(CheckType.ATTRIBUTE_EXISTS, version='1.0.0')
class AttributeExistsCheck(BaseCheck):
    """Tests whether an attribute/field exists as defined by the JSONPath; useful for checking if optional fields."""  # noqa: E501

    # Pydantic fields with validation
    path: str = RequiredJSONPath('JSONPath expression to check for existence (e.g., "$.output.error")')
    negate: bool = Field(False, description='If true, passes when attribute does NOT exist')

    def __call__(
        self,
        path: str,  # noqa: ARG002
        negate: bool = False,  # noqa: ARG002
    ) -> dict[str, Any]:
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
        check_type: str,
        arguments: dict[str, Any],
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

        try:
            # Validate arguments
            if 'path' not in arguments:
                raise ValidationError("AttributeExistsCheck requires 'path' argument")

            path = arguments.get('path')
            negate = arguments.get('negate', False)

            if not isinstance(path, str):
                raise ValidationError("'path' argument must be a string")

            if not self._resolver.is_jsonpath(path):
                raise ValidationError(
                    "'path' argument must be a JSONPath expression (e.g., '$.output.error')",
                )

            if not isinstance(negate, bool):
                raise ValidationError("'negate' argument must be a boolean")

            # Try to resolve the JSONPath to determine existence
            try:
                resolved_arg = self._resolver.resolve_argument(path, context.context_dict)
                # If we get here, the path exists
                attribute_exists = True
                resolved_arguments = {'path': resolved_arg, 'negate': {'value': negate}}
            except JSONPathError:
                # Path doesn't exist
                attribute_exists = False
                resolved_arguments = {
                    'path': {'jsonpath': path, 'value': None},
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
