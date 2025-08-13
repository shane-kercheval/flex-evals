"""
Check registration system for FEP.

Provides decorator-based registration for check implementations and
handles check discovery by type string.
"""

from typing import Any
import inspect
from packaging import version

from .checks.base import BaseCheck, BaseAsyncCheck
from .constants import CheckType


class CheckRegistry:
    """
    Registry for check implementations.

    Manages registration and lookup of check classes by type string.
    Supports version tracking and conflict handling.
    """

    def __init__(self):
        self._checks: dict[str, dict[str, dict[str, Any]]] = {}
        # Reverse mapping: class -> (check_type, version)
        self._class_to_version: dict[type, tuple[str, str]] = {}

    def register(
        self,
        check_type: str | CheckType,
        check_class: type[BaseCheck | BaseAsyncCheck],
        version: str = "1.0.0",
    ) -> None:
        """
        Register a check implementation.

        Args:
            check_type: String or CheckType enum identifier for the check type
            check_class: Check implementation class
            version: Semantic version of the check implementation

        Raises:
            ValueError: If check_type already registered with different version
        """
        # Convert enum to string for internal storage
        check_type_str = str(check_type)

        # Initialize check_type if it doesn't exist
        if check_type_str not in self._checks:
            self._checks[check_type_str] = {}

        # Allow re-registration of same version (useful for testing)
        self._checks[check_type_str][version] = {
            "class": check_class,
            "version": version,
            "is_async": self._is_async_check(check_class),
        }

        # Update reverse mapping
        self._class_to_version[check_class] = (check_type_str, version)

    def get_check_class(
        self, check_type: str, version: str | None = None,
    ) -> type[BaseCheck | BaseAsyncCheck]:
        """
        Get the registered check class for a check type and version.

        Args:
            check_type: String identifier for the check type
            version: Specific version to retrieve, or None for latest

        Returns:
            The registered check class

        Raises:
            ValueError: If check type or version is not registered
        """
        if check_type not in self._checks:
            raise ValueError(f"Check type '{check_type}' is not registered")

        if version is None:
            version = self.get_latest_version(check_type)

        if version not in self._checks[check_type]:
            available_versions = list(self._checks[check_type].keys())
            raise ValueError(
                f"Version '{version}' not found for check type '{check_type}'. "
                f"Available versions: {available_versions}",
            )

        return self._checks[check_type][version]["class"]

    def get_check_info(self, check_type: str, version: str | None = None) -> dict[str, Any]:
        """
        Get complete information about a registered check.

        Args:
            check_type: String identifier for the check type
            version: Specific version to retrieve, or None for latest

        Returns:
            Dict with check class, version, and async status

        Raises:
            ValueError: If check type or version is not registered
        """
        if check_type not in self._checks:
            raise ValueError(f"Check type '{check_type}' is not registered")

        if version is None:
            version = self.get_latest_version(check_type)

        if version not in self._checks[check_type]:
            available_versions = list(self._checks[check_type].keys())
            raise ValueError(
                f"Version '{version}' not found for check type '{check_type}'. "
                f"Available versions: {available_versions}",
            )

        return self._checks[check_type][version].copy()

    def get_latest_version(self, check_type: str) -> str:
        """
        Get the latest version for a check type using semantic versioning.

        Args:
            check_type: String identifier for the check type

        Returns:
            The latest version string

        Raises:
            ValueError: If check type is not registered
        """
        if check_type not in self._checks:
            raise ValueError(f"Check type '{check_type}' is not registered")

        versions = list(self._checks[check_type].keys())
        if not versions:
            raise ValueError(f"No versions registered for check type '{check_type}'")

        # Sort versions using semantic versioning
        return max(versions, key=version.parse)

    def list_versions(self, check_type: str) -> list[str]:
        """
        List all available versions for a check type.

        Args:
            check_type: String identifier for the check type

        Returns:
            List of version strings sorted by semantic version

        Raises:
            ValueError: If check type is not registered
        """
        if check_type not in self._checks:
            raise ValueError(f"Check type '{check_type}' is not registered")

        versions = list(self._checks[check_type].keys())
        return sorted(versions, key=version.parse)

    def is_async_check(self, check_type: str, version: str | None = None) -> bool:
        """
        Check if a registered check is asynchronous.

        Args:
            check_type: String identifier for the check type
            version: Specific version to check, or None for latest

        Returns:
            True if the check is asynchronous, False otherwise

        Raises:
            ValueError: If check type or version is not registered
        """
        info = self.get_check_info(check_type, version)
        return info["is_async"]

    def list_registered_checks(self) -> dict[str, dict[str, dict[str, Any]]]:
        """
        Get a list of all registered checks with their version information.

        Returns:
            Dict mapping check types to version dictionaries with registration information
        """
        return {
            check_type: {version: info.copy() for version, info in versions.items()}
            for check_type, versions in self._checks.items()
        }

    def get_version_for_class(self, cls: type) -> str:
        """
        Get the version for a registered check class.

        Args:
            cls: Check class to look up

        Returns:
            Version string for the class

        Raises:
            ValueError: If class is not registered
        """
        if cls not in self._class_to_version:
            raise ValueError(f"Class {cls} is not registered")
        return self._class_to_version[cls][1]  # Return version part

    def get_check_type_for_class(self, cls: type) -> str:
        """
        Get the check type for a registered check class.

        Args:
            cls: Check class to look up

        Returns:
            Check type string for the class

        Raises:
            ValueError: If class is not registered
        """
        if cls not in self._class_to_version:
            raise ValueError(f"Class {cls} is not registered")
        return self._class_to_version[cls][0]  # Return check_type part

    def clear(self) -> None:
        """Clear all registered checks (useful for testing)."""
        self._checks.clear()
        self._class_to_version.clear()

    def _is_async_check(self, check_class: type[BaseCheck | BaseAsyncCheck]) -> bool:
        """Determine if a check class is asynchronous."""
        if not hasattr(check_class, '__call__'):
            return False

        # Check if the __call__ method is async
        call_method = getattr(check_class, '__call__')
        return inspect.iscoroutinefunction(call_method)


# Global registry instance
_global_registry = CheckRegistry()


def register(check_type: str | CheckType, version: str = "1.0.0") -> callable:
    """
    Decorator for registering check implementations.

    Args:
        check_type: String or CheckType enum identifier for the check type
        version: Semantic version of the check implementation

    Returns:
        Decorator function

    Example:
        @register(CheckType.EXACT_MATCH, version="1.0.0")
        class ExactMatchCheck(BaseCheck):
            def __call__(self, arguments, context):
                # Implementation
                return {"passed": True}

        # Strings are also supported:
        @register('exact_match', version="1.0.0")
        class ExactMatchCheck(BaseCheck):
            # ...
    """
    def decorator(cls: type[BaseCheck | BaseAsyncCheck]) -> type[BaseCheck | BaseAsyncCheck]:
        _global_registry.register(check_type, cls, version)
        return cls

    return decorator


def get_check_class(
    check_type: str, version: str | None = None,
) -> type[BaseCheck | BaseAsyncCheck]:
    """
    Get the registered check class for a check type and version.

    Args:
        check_type: String identifier for the check type
        version: Specific version to retrieve, or None for latest

    Returns:
        The registered check class

    Raises:
        ValueError: If check type or version is not registered
    """
    return _global_registry.get_check_class(check_type, version)


def get_check_info(check_type: str, version: str | None = None) -> dict[str, Any]:
    """
    Get complete information about a registered check.

    Args:
        check_type: String identifier for the check type
        version: Specific version to retrieve, or None for latest

    Returns:
        Dict with check class, version, and async status

    Raises:
        ValueError: If check type or version is not registered
    """
    return _global_registry.get_check_info(check_type, version)


def is_async_check(check_type: str, version: str | None = None) -> bool:
    """
    Check if a registered check is asynchronous.

    Args:
        check_type: String identifier for the check type
        version: Specific version to check, or None for latest

    Returns:
        True if the check is asynchronous, False otherwise

    Raises:
        ValueError: If check type or version is not registered
    """
    return _global_registry.is_async_check(check_type, version)


def list_registered_checks() -> dict[str, dict[str, dict[str, Any]]]:
    """
    Get a list of all registered checks with their version information.

    Returns:
        Dict mapping check types to version dictionaries with registration information
    """
    return _global_registry.list_registered_checks()


def get_latest_version(check_type: str) -> str:
    """
    Get the latest version for a check type using semantic versioning.

    Args:
        check_type: String identifier for the check type

    Returns:
        The latest version string

    Raises:
        ValueError: If check type is not registered
    """
    return _global_registry.get_latest_version(check_type)


def list_versions(check_type: str) -> list[str]:
    """
    List all available versions for a check type.

    Args:
        check_type: String identifier for the check type

    Returns:
        List of version strings sorted by semantic version

    Raises:
        ValueError: If check type is not registered
    """
    return _global_registry.list_versions(check_type)


def get_version_for_class(cls: type) -> str:
    """
    Get the version for a registered check class.

    Args:
        cls: Check class to look up

    Returns:
        Version string for the class

    Raises:
        ValueError: If class is not registered
    """
    return _global_registry.get_version_for_class(cls)


def get_check_type_for_class(cls: type) -> str:
    """
    Get the check type for a registered check class.

    Args:
        cls: Check class to look up

    Returns:
        Check type string for the class

    Raises:
        ValueError: If class is not registered
    """
    return _global_registry.get_check_type_for_class(cls)


def clear_registry() -> None:
    """Clear all registered checks (useful for testing)."""
    _global_registry.clear()


def get_registry_state() -> dict[str, dict[str, dict[str, Any]]]:
    """
    Get the current state of the registry for serialization.

    Returns:
        Dict containing all registered checks and their version information
    """
    return _global_registry.list_registered_checks()


def restore_registry_state(registry_state: dict[str, dict[str, dict[str, Any]]]) -> None:
    """
    Restore the registry state from serialized data.

    Args:
        registry_state: Dict containing check registrations to restore
    """
    # Clear current registry
    _global_registry.clear()

    # Restore each check registration
    for check_type, versions in registry_state.items():
        for version_str, info in versions.items():
            _global_registry.register(check_type, info["class"], info["version"])
