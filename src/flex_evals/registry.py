"""
Check registration system for FEP.

Provides decorator-based registration for check implementations and
handles check discovery by type string.
"""

from typing import Any
import inspect

from .checks.base import BaseCheck, BaseAsyncCheck


class CheckRegistry:
    """
    Registry for check implementations.

    Manages registration and lookup of check classes by type string.
    Supports version tracking and conflict handling.
    """

    def __init__(self):
        self._checks: dict[str, dict[str, Any]] = {}

    def register(
        self,
        check_type: str,
        check_class: type[BaseCheck | BaseAsyncCheck],
        version: str = "1.0.0",
    ) -> None:
        """
        Register a check implementation.

        Args:
            check_type: String identifier for the check type
            check_class: Check implementation class
            version: Semantic version of the check implementation

        Raises:
            ValueError: If check_type already registered with different version
        """
        if check_type in self._checks:
            existing_version = self._checks[check_type]["version"]
            if existing_version != version:
                raise ValueError(
                    f"Check type '{check_type}' already registered with version {existing_version}, "
                    f"cannot register with version {version}",
                )
            # Same version - allow re-registration (useful for testing)

        self._checks[check_type] = {
            "class": check_class,
            "version": version,
            "is_async": self._is_async_check(check_class),
        }

    def get_check_class(self, check_type: str) -> type[BaseCheck | BaseAsyncCheck]:
        """
        Get the registered check class for a check type.

        Args:
            check_type: String identifier for the check type

        Returns:
            The registered check class

        Raises:
            ValueError: If check type is not registered
        """
        if check_type not in self._checks:
            raise ValueError(f"Check type '{check_type}' is not registered")

        return self._checks[check_type]["class"]

    def get_check_info(self, check_type: str) -> dict[str, Any]:
        """
        Get complete information about a registered check.

        Args:
            check_type: String identifier for the check type

        Returns:
            Dict with check class, version, and async status

        Raises:
            ValueError: If check type is not registered
        """
        if check_type not in self._checks:
            raise ValueError(f"Check type '{check_type}' is not registered")

        return self._checks[check_type].copy()

    def is_async_check(self, check_type: str) -> bool:
        """
        Check if a registered check is asynchronous.

        Args:
            check_type: String identifier for the check type

        Returns:
            True if the check is asynchronous, False otherwise

        Raises:
            ValueError: If check type is not registered
        """
        info = self.get_check_info(check_type)
        return info["is_async"]

    def list_registered_checks(self) -> dict[str, dict[str, Any]]:
        """
        Get a list of all registered checks with their information.

        Returns:
            Dict mapping check types to their registration information
        """
        return {
            check_type: info.copy()
            for check_type, info in self._checks.items()
        }

    def clear(self) -> None:
        """Clear all registered checks (useful for testing)."""
        self._checks.clear()

    def _is_async_check(self, check_class: type[BaseCheck | BaseAsyncCheck]) -> bool:
        """Determine if a check class is asynchronous."""
        if not hasattr(check_class, '__call__'):
            return False

        # Check if the __call__ method is async
        call_method = getattr(check_class, '__call__')
        return inspect.iscoroutinefunction(call_method)


# Global registry instance
_global_registry = CheckRegistry()


def register(check_type: str, version: str = "1.0.0"):
    """
    Decorator for registering check implementations.

    Args:
        check_type: String identifier for the check type
        version: Semantic version of the check implementation

    Returns:
        Decorator function

    Example:
        @register("exact_match", version="1.0.0")
        class ExactMatchCheck(BaseCheck):
            def __call__(self, arguments, context):
                # Implementation
                return {"passed": True}
    """
    def decorator(cls: type[BaseCheck | BaseAsyncCheck]) -> type[BaseCheck | BaseAsyncCheck]:
        _global_registry.register(check_type, cls, version)
        return cls

    return decorator


def get_check_class(check_type: str) -> type[BaseCheck | BaseAsyncCheck]:
    """
    Get the registered check class for a check type.

    Args:
        check_type: String identifier for the check type

    Returns:
        The registered check class

    Raises:
        ValueError: If check type is not registered
    """
    return _global_registry.get_check_class(check_type)


def get_check_info(check_type: str) -> dict[str, Any]:
    """
    Get complete information about a registered check.

    Args:
        check_type: String identifier for the check type

    Returns:
        Dict with check class, version, and async status

    Raises:
        ValueError: If check type is not registered
    """
    return _global_registry.get_check_info(check_type)


def is_async_check(check_type: str) -> bool:
    """
    Check if a registered check is asynchronous.

    Args:
        check_type: String identifier for the check type

    Returns:
        True if the check is asynchronous, False otherwise

    Raises:
        ValueError: If check type is not registered
    """
    return _global_registry.is_async_check(check_type)


def list_registered_checks() -> dict[str, dict[str, Any]]:
    """
    Get a list of all registered checks with their information.

    Returns:
        Dict mapping check types to their registration information
    """
    return _global_registry.list_registered_checks()


def clear_registry() -> None:
    """Clear all registered checks (useful for testing)."""
    _global_registry.clear()
