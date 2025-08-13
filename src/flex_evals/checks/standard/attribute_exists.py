"""
Attribute Exists check implementation for FEP.

Tests whether an attribute/field exists in the evaluation context using JSONPath,
without throwing errors if the path doesn't exist (unlike normal JSONPath resolution).
"""

from typing import Any
from datetime import datetime, UTC

from ..base import BaseCheck, EvaluationContext
from ...registry import register
from ...constants import CheckType
from ...exceptions import JSONPathError, ValidationError
from ...schemas import CheckResult


@register(CheckType.ATTRIBUTE_EXISTS, version="1.0.0")
class AttributeExistsCheck_v1_0_0(BaseCheck):  # noqa: N801
    """
    Tests whether an attribute exists in the evaluation context.

    This check is useful for validating that optional fields (like errors, metadata,
    or dynamic properties) exist without failing if they don't, unlike standard
    JSONPath resolution which throws JSONPathError for non-existent paths.

    Arguments Schema:
    - path: string - JSONPath expression to check for existence (e.g., "$.output.error")
    - negate: boolean (default: false) - If true, passes when attribute does NOT exist

    Results Schema:
    - passed: boolean - Whether the existence check passed
    """

    def __call__(  # noqa: D102
            self,
            path: str,  # noqa: ARG002
            negate: bool = False,  # noqa: ARG002
        ) -> dict[str, Any]:
        # This method should never be called directly for AttributeExistsCheck
        # because we need special JSONPath handling in the execute method
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

            results = {"passed": passed}

            # Create successful result
            metadata = {}
            if check_version:
                metadata["check_version"] = check_version
            if check_metadata:
                metadata.update(check_metadata)

            return CheckResult(
                check_type=check_type,
                check_version=check_version,
                status='completed',
                results=results,
                resolved_arguments=resolved_arguments,
                evaluated_at=evaluated_at,
                metadata=metadata if metadata else None,
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
