"""
Base check classes for FEP check implementations.

Provides the foundation for both synchronous and asynchronous check implementations
with proper evaluation context handling.
"""

from abc import ABC, abstractmethod
from typing import Any
from datetime import datetime, UTC

from ..schemas import TestCase, Output, CheckResult, CheckError
from ..jsonpath_resolver import get_shared_resolver
from ..exceptions import CheckExecutionError, JSONPathError, ValidationError


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


class BaseCheck(ABC):
    """
    Base class for synchronous check implementations.

    All synchronous checks should inherit from this class and implement the __call__ method.
    The base class handles argument resolution, error handling, and result formatting.
    """

    def __init__(self):
        self._resolver = get_shared_resolver()

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


class BaseAsyncCheck(ABC):
    """
    Base class for asynchronous check implementations.

    All asynchronous checks (e.g., LLM-based checks) should inherit from this class
    and implement the __call__ method as an async method.
    """

    def __init__(self):
        self._resolver = get_shared_resolver()

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
