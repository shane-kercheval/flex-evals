"""
Base check classes for FEP check implementations.

Provides the foundation for both synchronous and asynchronous check implementations
with proper evaluation context handling, including JSONPath validation.
"""

from abc import ABC, abstractmethod
from datetime import datetime, UTC
from enum import Enum
from typing import Any, ClassVar

import jsonpath_ng
from jsonpath_ng.exceptions import JSONPathError as JSONPathNGError
from pydantic import BaseModel, Field, model_validator

from ..constants import CheckType
from ..exceptions import CheckExecutionError, JSONPathError, ValidationError
from ..jsonpath_resolver import get_shared_resolver
from ..schemas import CheckResult, CheckError, Output, TestCase


# JSONPath validation utilities
class JSONPathBehavior(Enum):
    """Enum defining JSONPath behavior for schema fields."""

    REQUIRED = 'required'  # Must be valid JSONPath
    OPTIONAL = 'optional'  # Can be literal or JSONPath (validates if starts with $)


def RequiredJSONPath(description: str) -> Any:  # noqa: ANN401, N802
    """Field that must contain a valid JSONPath expression."""
    return Field(
        ...,
        description=description,
        json_schema_extra={'jsonpath': JSONPathBehavior.REQUIRED.value},
    )


def OptionalJSONPath(description: str, default: Any = ..., **kwargs: Any) -> Any:  # noqa: ANN401, N802
    """Field that can contain literal text or JSONPath expression."""
    # Merge json_schema_extra with any existing extras
    extra = kwargs.pop('json_schema_extra', {})
    extra['jsonpath'] = JSONPathBehavior.OPTIONAL.value

    return Field(
        default,
        description=description,
        json_schema_extra=extra,
        **kwargs,
    )


def get_jsonpath_behavior(model_class: type, field_name: str) -> JSONPathBehavior | None:
    """Get JSONPath behavior for a field."""
    field_info = model_class.model_fields.get(field_name)
    if field_info and hasattr(field_info, 'json_schema_extra') and field_info.json_schema_extra:
        behavior_str = field_info.json_schema_extra.get('jsonpath')
        if behavior_str:
            return JSONPathBehavior(behavior_str)
    return None


def validate_jsonpath(expression: str) -> bool:
    """Validate that a string is a valid JSONPath expression."""
    # Must start with $ to be considered a JSONPath expression
    if not expression.startswith('$'):
        return False

    try:
        jsonpath_ng.parse(expression)
        return True
    except (JSONPathNGError, Exception):
        return False


def is_jsonpath_expression(value: str) -> bool:
    r"""
    Determine if a string appears to be a JSONPath expression.

    Uses the same logic as the existing JSONPathResolver for consistency.

    Rules:
    - Strings beginning with '$.' are JSONPath expressions
    - Strings beginning with '\\$.' are escaped literals (not JSONPath)
    """
    if not isinstance(value, str):
        return False

    if value.startswith('\\$.'):
        return False  # Escaped literal

    return value.startswith('$.')


class JSONPathValidatedModel(BaseModel):
    """Base class that automatically validates JSONPath fields based on metadata."""

    @model_validator(mode='after')
    def validate_jsonpath_fields(self) -> 'JSONPathValidatedModel':
        """Validate all fields marked as JSONPath required/optional."""
        for field_name, field_value in self.__dict__.items():
            if not isinstance(field_value, str):
                continue

            jsonpath_behavior = get_jsonpath_behavior(self.__class__, field_name)

            if jsonpath_behavior == JSONPathBehavior.REQUIRED:
                # Must be valid JSONPath
                if not validate_jsonpath(field_value):
                    raise ValueError(
                        f"Field '{field_name}' requires valid JSONPath expression, "
                        f"got: '{field_value}'",
                    )

            elif (jsonpath_behavior == JSONPathBehavior.OPTIONAL
                  and is_jsonpath_expression(field_value)
                  and not validate_jsonpath(field_value)):
                raise ValueError(
                    f"Field '{field_name}' appears to be JSONPath but is invalid: '{field_value}'",
                )

        return self


class EvaluationContext:
    """
    Evaluation context that provides access to test case and output data.

    This is a convenience wrapper around the raw context dictionary that
    provides type-safe access to test case and output data.
    """

    def __init__(self, test_case: TestCase, output: Output):
        self.test_case = test_case
        self.output = output
        self._resolver = get_shared_resolver()
        self._context_dict = self._resolver.create_evaluation_context(test_case, output)

    @property
    def context_dict(self) -> dict[str, Any]:
        """Get the raw context dictionary for JSONPath resolution."""
        return self._context_dict


class BaseCheck(JSONPathValidatedModel, ABC):
    """
    Base class for synchronous check implementations.

    Combines Pydantic field validation with check execution capabilities.
    Subclasses should define their Pydantic fields and implement __call__().
    """


    # Pydantic configuration
    model_config: ClassVar[dict[str, Any]] = {'extra': 'forbid'}

    metadata: dict[str, Any] | None = None

    def __init__(self, **data: Any) -> None:  # noqa: ANN401
        """Initialize check with field validation."""
        # Initialize Pydantic model first (validates fields)
        JSONPathValidatedModel.__init__(self, **data)
        # Setup resolver for execution
        self._resolver = get_shared_resolver()

    @property
    def check_type(self) -> CheckType:
        """Return the CheckType for this check."""
        # Import here to avoid circular import
        from ..registry import get_check_type_for_class  # noqa: PLC0415
        check_type_str = get_check_type_for_class(self.__class__)
        return CheckType(check_type_str)

    def to_arguments(self) -> dict[str, Any]:
        """Convert Pydantic fields to arguments dict for execution."""
        data = self.model_dump()
        # Remove metadata as it's handled separately
        data.pop('metadata', None)
        return data

    def _get_version(self) -> str:
        """Get the registered version for this check class."""
        # Import here to avoid circular import
        from ..registry import get_version_for_class  # noqa: PLC0415
        return get_version_for_class(self.__class__)

    @abstractmethod
    def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
        """
        Execute the check with direct arguments.

        Args:
            **kwargs: Check arguments passed directly as keyword arguments

        Returns:
            Dict containing check-specific results

        Raises:
            CheckExecutionError: If check execution fails
            ValidationError: If arguments are invalid for this check
        """
        pass

    def execute(
        self,
        check_type: str,
        arguments: dict[str, Any],
        context: EvaluationContext,
        check_metadata: dict[str, Any] | None = None,
    ) -> CheckResult:
        """
        Execute the check and return a complete CheckResult.

        This method handles argument resolution, error handling, and result formatting
        according to the FEP protocol.

        Args:
            check_type: The type identifier for this check
            arguments: Raw check arguments (may contain JSONPath expressions)
            context: Evaluation context
            check_metadata: Additional metadata from the check definition

        Returns:
            Complete CheckResult with all required fields
        """
        evaluated_at = datetime.now(UTC)

        # Get version from registry using the class
        check_version = self._get_version()

        try:
            # Resolve arguments
            resolved_arguments = self._resolver.resolve_arguments(
                arguments,
                context.context_dict,
            )

            # Extract resolved values for check execution
            resolved_values = {
                key: arg_data["value"]
                for key, arg_data in resolved_arguments.items()
            }

            # Execute the check
            try:
                results = self(**resolved_values)
            except TypeError as e:
                raise ValidationError(f"Invalid arguments for check: {e!s}") from e

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

        except JSONPathError as e:
            return self._create_error_result(
                check_type=check_type,
                error_type='jsonpath_error',
                error_message=str(e),
                resolved_arguments={},
                evaluated_at=evaluated_at,
                check_version=check_version,
                check_metadata=check_metadata,
                recoverable=False,
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

        except CheckExecutionError as e:
            return self._create_error_result(
                check_type=check_type,
                error_type='unknown_error',
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

    def _create_error_result(
        self,
        check_type: str,
        error_type: str,
        error_message: str,
        resolved_arguments: dict[str, Any],
        evaluated_at: datetime,
        check_version: str,
        check_metadata: dict[str, Any] | None = None,
        recoverable: bool = False,
    ) -> CheckResult:
        """Create a CheckResult for error cases."""
        # Create metadata that includes check_version
        return CheckResult(
            check_type=check_type,
            check_version=check_version,
            status='error',
            results={},
            resolved_arguments=resolved_arguments,
            evaluated_at=evaluated_at,
            metadata=check_metadata,
            error=CheckError(
                type=error_type,
                message=error_message,
                recoverable=recoverable,
            ),
        )


class BaseAsyncCheck(JSONPathValidatedModel, ABC):
    """
    Base class for asynchronous check implementations.

    Combines Pydantic field validation with async check execution capabilities.
    Subclasses should define their Pydantic fields and implement async __call__().
    """


    # Pydantic configuration
    model_config: ClassVar[dict[str, Any]] = {'extra': 'forbid'}

    metadata: dict[str, Any] | None = None

    def __init__(self, **data: Any) -> None:  # noqa: ANN401
        """Initialize async check with field validation."""
        # Initialize Pydantic model first (validates fields)
        JSONPathValidatedModel.__init__(self, **data)
        # Setup resolver for execution
        self._resolver = get_shared_resolver()

    @property
    def check_type(self) -> CheckType:
        """Return the CheckType for this check."""
        # Import here to avoid circular import
        from ..registry import get_check_type_for_class  # noqa: PLC0415
        check_type_str = get_check_type_for_class(self.__class__)
        return CheckType(check_type_str)

    def to_arguments(self) -> dict[str, Any]:
        """Convert Pydantic fields to arguments dict for execution."""
        data = self.model_dump()
        # Remove metadata as it's handled separately
        data.pop('metadata', None)
        return data

    def _get_version(self) -> str:
        """Get the registered version for this check class."""
        # Import here to avoid circular import
        from ..registry import get_version_for_class  # noqa: PLC0415
        return get_version_for_class(self.__class__)

    @abstractmethod
    async def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
        """
        Execute the check with direct arguments asynchronously.

        Args:
            **kwargs: Check arguments passed directly as keyword arguments

        Returns:
            Dict containing check-specific results

        Raises:
            CheckExecutionError: If check execution fails
            ValidationError: If arguments are invalid for this check
        """
        pass

    async def execute(
        self,
        check_type: str,
        arguments: dict[str, Any],
        context: EvaluationContext,
        check_metadata: dict[str, Any] | None = None,
    ) -> CheckResult:
        """
        Execute the check asynchronously and return a complete CheckResult.

        This method handles argument resolution, error handling, and result formatting
        according to the FEP protocol.

        Args:
            check_type: The type identifier for this check
            arguments: check arguments (may contain JSONPath expressions that need to be resolved)
            context: Evaluation context
            check_metadata: Additional metadata from the check definition

        Returns:
            Complete CheckResult with all required fields
        """
        evaluated_at = datetime.now(UTC)

        # Get version from registry using the class
        check_version = self._get_version()

        try:
            # Resolve arguments
            resolved_arguments = self._resolver.resolve_arguments(
                arguments,
                context.context_dict,
            )

            # Extract resolved values for check execution
            resolved_values = {
                key: arg_data["value"]
                for key, arg_data in resolved_arguments.items()
            }

            # Execute the check asynchronously
            try:
                results = await self(**resolved_values)
            except TypeError as e:
                raise ValidationError(f"Invalid arguments for check: {e!s}") from e

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

        except JSONPathError as e:
            return self._create_error_result(
                check_type=check_type,
                error_type='jsonpath_error',
                error_message=str(e),
                resolved_arguments={},
                evaluated_at=evaluated_at,
                check_version=check_version,
                check_metadata=check_metadata,
                recoverable=False,
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

        except CheckExecutionError as e:
            return self._create_error_result(
                check_type=check_type,
                error_type='unknown_error',
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
                error_message=f"Unexpected error during async check execution: {e!s}",
                resolved_arguments={},
                evaluated_at=evaluated_at,
                check_version=check_version,
                recoverable=False,
            )

    def _create_error_result(
        self,
        check_type: str,
        error_type: str,
        error_message: str,
        resolved_arguments: dict[str, Any],
        evaluated_at: datetime,
        check_version: str,
        check_metadata: dict[str, Any] | None = None,
        recoverable: bool = False,
    ) -> CheckResult:
        """Create a CheckResult for error cases."""
        # Create metadata that includes check_version
        return CheckResult(
            check_type=check_type,
            check_version=check_version,
            status='error',
            results={},
            resolved_arguments=resolved_arguments,
            evaluated_at=evaluated_at,
            metadata=check_metadata,
            error=CheckError(
                type=error_type,
                message=error_message,
                recoverable=recoverable,
            ),
        )
