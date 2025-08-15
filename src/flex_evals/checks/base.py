"""
Base check classes for FEP check implementations.

Provides the foundation for both synchronous and asynchronous check implementations
with proper evaluation context handling, including JSONPath validation.
"""

from abc import ABC, abstractmethod
from datetime import datetime, UTC
from enum import Enum
from types import UnionType
from typing import Any, ClassVar, TypeAlias, TYPE_CHECKING
import typing
import jsonpath_ng
from jsonpath_ng.exceptions import JSONPathError as JSONPathNGError
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from ..schemas.check import Check

from ..constants import CheckType
from ..exceptions import CheckExecutionError, JSONPathError, ValidationError
from ..jsonpath_resolver import get_shared_resolver
from ..schemas import CheckResult, CheckError, Output, TestCase


# JSONPath validation utilities
class JSONPathBehavior(Enum):
    """Enum defining JSONPath behavior for schema fields."""

    REQUIRED = 'required'  # Must be valid JSONPath
    OPTIONAL = 'optional'  # Can be literal or JSONPath (validates if starts with $)


def get_jsonpath_behavior(model_class: type, field_name: str) -> JSONPathBehavior | None:
    """
    Get JSONPath behavior for a field by inspecting its type annotation.

    This function determines whether a Pydantic field should be treated as:
    - REQUIRED: Field must be a JSONPath (type annotation is exactly `JSONPath`)
    - OPTIONAL: Field can be literal or JSONPath (type annotation is union like `str | JSONPath`)
    - None: Field doesn't support JSONPath (no JSONPath in type annotation)

    Used by the schema generator and validation systems to understand how to handle
    field values during check execution.

    Args:
        model_class: Pydantic model class to inspect
        field_name: Name of the field to check

    Returns:
        JSONPathBehavior.REQUIRED if field type is exactly JSONPath
        JSONPathBehavior.OPTIONAL if field type is a union containing JSONPath
        None if field doesn't exist or doesn't support JSONPath

    Examples:
        >>> class MyCheck(BaseCheck):
        ...     path: JSONPath = Field(...)  # REQUIRED
        ...     text: str | JSONPath = Field(...)  # OPTIONAL
        ...     literal: str = Field(...)  # None
        >>> get_jsonpath_behavior(MyCheck, 'path')
        JSONPathBehavior.REQUIRED
    """
    # Handle None or invalid model_class gracefully
    if not model_class or not hasattr(model_class, 'model_fields'):
        return None

    field_info = model_class.model_fields.get(field_name)
    if not field_info:
        return None

    # Get the field's type annotation
    field_type = field_info.annotation

    # Check if the field type is exactly JSONPath (required)
    if field_type is JSONPath:
        return JSONPathBehavior.REQUIRED

    # Check if the field type is a Union that includes JSONPath (optional)
    # This handles cases like str | JSONPath, bool | JSONPath, etc.
    origin = typing.get_origin(field_type)

    # Handle both old-style typing.Union and new-style types.UnionType (Python 3.10+)
    if origin is typing.Union or isinstance(field_type, UnionType):
        args = typing.get_args(field_type)
        if JSONPath in args:
            return JSONPathBehavior.OPTIONAL

    return None


def validate_jsonpath(expression: str) -> bool:
    """
    Validate that a string is a valid JSONPath expression.

    Uses the jsonpath-ng library to parse and validate JSONPath syntax.
    Only expressions starting with '$' are considered valid JSONPath.

    This function is used during:
    - JSONPath object creation (JSONPath.__init__)
    - Schema generation to verify JSONPath field definitions

    Args:
        expression: String to validate as JSONPath (gracefully handles non-strings)

    Returns:
        True if expression is a valid JSONPath starting with '$'
        False for invalid JSONPath, non-strings, or expressions not starting with '$'

    Examples:
        >>> validate_jsonpath("$.output.value")  # Valid
        True
        >>> validate_jsonpath("invalid_path")    # No $ prefix
        False
        >>> validate_jsonpath("$[invalid")       # Malformed syntax
        False
        >>> validate_jsonpath(None)              # Non-string
        False
    """
    # Handle non-string inputs gracefully
    if not isinstance(expression, str):
        return False

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


# JSONPath type for clean field definitions
class JSONPath(BaseModel):
    """Represents a JSONPath expression that needs resolution."""

    expression: str = Field(..., description="The JSONPath expression to resolve")

    @field_validator('expression')
    @classmethod
    def validate_expression(cls, v: str) -> str:
        """Validate that the expression is a valid JSONPath."""
        if not validate_jsonpath(v):
            raise ValueError(f"Invalid JSONPath expression: '{v}'")
        return v

    def __str__(self) -> str:
        return self.expression

    def __repr__(self) -> str:
        return f"JSONPath('{self.expression}')"


def _convert_to_jsonpath(value: object) -> object | JSONPath:
    """Convert JSONPath-like strings to JSONPath objects."""
    if isinstance(value, str) and value.startswith('$.'):
        return JSONPath(expression=value)
    return value


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


class BaseCheck(BaseModel, ABC):
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
        BaseModel.__init__(self, **data)
        # Setup resolver for execution
        self._resolver = get_shared_resolver()

    @property
    def check_type(self) -> CheckType | str:
        """Return the CheckType for this check."""
        # Import here to avoid circular import
        from ..registry import get_check_type_for_class  # noqa: PLC0415
        check_type_str = get_check_type_for_class(self.__class__)

        # Try to convert to CheckType enum, but allow arbitrary strings for custom/test checks
        try:
            return CheckType(check_type_str)
        except ValueError:
            return check_type_str

    def to_arguments(self) -> dict[str, Any]:
        """Convert Pydantic fields to arguments dict for engine compatibility."""
        data = self.model_dump()
        # Remove metadata as it's handled separately
        data.pop('metadata', None)
        # Convert JSONPath objects to their string expressions for engine compatibility
        for key, value in data.items():
            if isinstance(value, JSONPath):
                data[key] = value.expression
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

    def resolve_fields(self, context: EvaluationContext) -> dict[str, Any]:
        """Resolve any JSONPath fields in-place using the evaluation context."""
        resolved_arguments = {}

        for field_name in self.__class__.model_fields:
            field_value = getattr(self, field_name)

            if isinstance(field_value, JSONPath):
                try:
                    resolved_result = self._resolver.resolve_argument(
                        field_value.expression,
                        context.context_dict,
                    )
                    resolved_value = resolved_result.get("value")
                    setattr(self, field_name, resolved_value)
                    resolved_arguments[field_name] = resolved_result
                except Exception as e:
                    raise JSONPathError(
                        f"Failed to resolve JSONPath in field '{field_name}': {e}",
                        jsonpath_expression=field_value.expression,
                    ) from e
            else:
                resolved_arguments[field_name] = {
                    "value": field_value,
                    "resolved_from": "literal",
                }

        self._resolved_arguments = resolved_arguments
        return resolved_arguments

    def _restore_jsonpath_fields(self) -> None:
        """Restore JSONPath objects from their resolved values (for reuse of check instances)."""
        # Only restore if we have the resolved arguments mapping
        if not hasattr(self, '_resolved_arguments'):
            return

        for field_name, resolved_info in self._resolved_arguments.items():
            if 'jsonpath' in resolved_info:
                # This field was resolved from a JSONPath - restore the JSONPath object
                original_jsonpath = JSONPath(expression=resolved_info['jsonpath'])
                setattr(self, field_name, original_jsonpath)

    def execute(
        self,
        context: EvaluationContext,
        check_metadata: dict[str, Any] | None = None,
    ) -> CheckResult:
        """
        Execute the check and return a complete CheckResult.

        This method handles JSONPath field resolution, error handling, and result formatting
        according to the FEP protocol.

        Args:
            context: Evaluation context for JSONPath resolution
            check_metadata: Additional metadata from the check definition

        Returns:
            Complete CheckResult with all required fields
        """
        evaluated_at = datetime.now(UTC)

        # Get version from registry using the class
        check_version = self._get_version()
        check_type = str(self.check_type)

        try:
            # Resolve JSONPath fields in-place
            resolved_arguments = self.resolve_fields(context)

            # Execute the check with resolved fields
            try:
                results = self()
            except TypeError as e:
                raise ValidationError(f"Invalid arguments for check: {e!s}") from e
            finally:
                # Restore JSONPath objects to allow reuse of check instances
                self._restore_jsonpath_fields()

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


class BaseAsyncCheck(BaseModel, ABC):
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
        BaseModel.__init__(self, **data)
        # Setup resolver for execution
        self._resolver = get_shared_resolver()

    @property
    def check_type(self) -> CheckType | str:
        """Return the CheckType for this check."""
        # Import here to avoid circular import
        from ..registry import get_check_type_for_class  # noqa: PLC0415
        check_type_str = get_check_type_for_class(self.__class__)

        # Try to convert to CheckType enum, but allow arbitrary strings for custom/test checks
        try:
            return CheckType(check_type_str)
        except ValueError:
            return check_type_str

    def to_arguments(self) -> dict[str, Any]:
        """Convert Pydantic fields to arguments dict for engine compatibility."""
        data = self.model_dump()
        # Remove metadata as it's handled separately
        data.pop('metadata', None)
        # Convert JSONPath objects to their string expressions for engine compatibility
        for key, value in data.items():
            if isinstance(value, JSONPath):
                data[key] = value.expression
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

    def resolve_fields(self, context: EvaluationContext) -> dict[str, Any]:
        """Resolve any JSONPath fields in-place using the evaluation context."""
        resolved_arguments = {}

        for field_name in self.__class__.model_fields:
            field_value = getattr(self, field_name)

            if isinstance(field_value, JSONPath):
                try:
                    resolved_result = self._resolver.resolve_argument(
                        field_value.expression,
                        context.context_dict,
                    )
                    resolved_value = resolved_result.get("value")
                    setattr(self, field_name, resolved_value)
                    resolved_arguments[field_name] = resolved_result
                except Exception as e:
                    raise JSONPathError(
                        f"Failed to resolve JSONPath in field '{field_name}': {e}",
                        jsonpath_expression=field_value.expression,
                    ) from e
            else:
                resolved_arguments[field_name] = {
                    "value": field_value,
                    "resolved_from": "literal",
                }

        self._resolved_arguments = resolved_arguments
        return resolved_arguments

    def _restore_jsonpath_fields(self) -> None:
        """Restore JSONPath objects from their resolved values (for reuse of check instances)."""
        # Only restore if we have the resolved arguments mapping
        if not hasattr(self, '_resolved_arguments'):
            return

        for field_name, resolved_info in self._resolved_arguments.items():
            if 'jsonpath' in resolved_info:
                # This field was resolved from a JSONPath - restore the JSONPath object
                original_jsonpath = JSONPath(expression=resolved_info['jsonpath'])
                setattr(self, field_name, original_jsonpath)

    async def execute(
        self,
        context: EvaluationContext,
        check_metadata: dict[str, Any] | None = None,
    ) -> CheckResult:
        """
        Execute the check asynchronously and return a complete CheckResult.

        This method handles argument resolution, error handling, and result formatting
        according to the FEP protocol.

        Args:
            context: Evaluation context for JSONPath resolution
            check_metadata: Additional metadata from the check definition

        Returns:
            Complete CheckResult with all required fields
        """
        evaluated_at = datetime.now(UTC)

        # Get version from registry using the class
        check_version = self._get_version()
        check_type = str(self.check_type)

        try:
            # Resolve JSONPath fields in-place
            resolved_arguments = self.resolve_fields(context)

            # Execute the check asynchronously with resolved fields
            try:
                results = await self()
            except TypeError as e:
                raise ValidationError(f"Invalid arguments for check: {e!s}") from e
            finally:
                # Restore JSONPath objects to allow reuse of check instances
                self._restore_jsonpath_fields()

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


# Type alias for any check type
CheckTypes: TypeAlias = 'Check | BaseCheck | BaseAsyncCheck'
