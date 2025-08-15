"""Check and CheckResult schema implementations for FEP."""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from ..constants import CheckType, ErrorType, Status


@dataclass
class Check:
    """
    Defines evaluation criteria for test cases, specifying how to assess outputs.

    Required Fields:
    - type: Identifier for the check implementation
    - arguments: Parameters passed to the check function

    Optional Fields:
    - version: Semantic version of the check implementation
    - metadata: Additional metadata for the check
    """

    type: str | CheckType
    arguments: dict[str, Any]
    version: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate required fields and constraints."""
        if not self.type or not isinstance(self.type, str | CheckType):
            raise ValueError("Check.type must be a non-empty string or CheckType enum")

        # Convert enum to string for consistency
        if isinstance(self.type, CheckType):
            self.type = self.type.value

        if not isinstance(self.arguments, dict):
            raise ValueError("Check.arguments must be a dictionary")

        if self.version is not None and not self._is_valid_semver(self.version):
            raise ValueError(f"Check.version must be valid semver format, got: {self.version}")

    def _is_valid_semver(self, version: str) -> bool:
        """Validate semantic version format."""
        semver_pattern = r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$'  # noqa: E501
        return bool(re.match(semver_pattern, version))



@dataclass
class CheckError:
    """Error details for failed check execution."""

    type: ErrorType | Literal['jsonpath_error', 'validation_error', 'timeout_error', 'unknown_error']  # noqa: E501
    message: str
    recoverable: bool = False




@dataclass
class CheckResult:
    """
    Represents the complete results of executing a single check against a test case and system
    output.

    This format provides full auditability by capturing the execution context,
    resolved arguments, check outcome, and any errors that occurred.

    Required Fields:
    - check_type: The type of check that was executed
    - check_version: The version of the check that was executed
    - status: Execution status of the check
    - results: Check outcome data (structure defined by check type)
    - resolved_arguments: Arguments after JSONPath resolution
    - evaluated_at: UTC timestamp when check was evaluated

    Optional Fields:
    - metadata: Dictionary containing check-specific metadata
    - error: Error details (only present when status is 'error')
    """

    check_type: str
    check_version: str
    status: Status | Literal['completed', 'error', 'skip']
    results: dict[str, Any]
    resolved_arguments: dict[str, Any]
    evaluated_at: datetime
    metadata: dict | None = None
    error: CheckError | None = None

    def __post_init__(self):
        """Validate required fields and constraints."""
        if not self.check_type:
            raise ValueError("CheckResult.check_type must be non-empty")

        if not isinstance(self.results, dict):
            raise ValueError("CheckResult.results must be a dictionary")

        if not isinstance(self.resolved_arguments, dict):
            raise ValueError("CheckResult.resolved_arguments must be a dictionary")

        if self.status == 'error' and self.error is None:
            raise ValueError("CheckResult.error is required when status is 'error'")

        if self.status != 'error' and self.error is not None:
            raise ValueError("CheckResult.error should only be present when status is 'error'")
