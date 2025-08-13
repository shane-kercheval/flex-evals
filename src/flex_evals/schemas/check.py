"""Check and CheckResult schema implementations for FEP."""

from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, Literal
import re
from enum import Enum
from pydantic import BaseModel, Field, model_validator

import jsonpath_ng
from jsonpath_ng.exceptions import JSONPathError as JSONPathNGError
from ..constants import CheckType, ErrorType, Status


class JSONPathBehavior(Enum):
    """Enum defining JSONPath behavior for schema fields."""

    REQUIRED = "required"  # Must be valid JSONPath
    OPTIONAL = "optional"  # Can be literal or JSONPath (validates if starts with $)


def RequiredJSONPath(description: str) -> Any:  # noqa: ANN401, N802
    """Field that must contain a valid JSONPath expression."""
    return Field(
        ...,
        description=description,
        json_schema_extra={"jsonpath": JSONPathBehavior.REQUIRED.value},
    )


def OptionalJSONPath(description: str, **kwargs: Any) -> Any:  # noqa: ANN401, N802
    """Field that can contain literal text or JSONPath expression."""
    # Merge json_schema_extra with any existing extras
    extra = kwargs.pop("json_schema_extra", {})
    extra["jsonpath"] = JSONPathBehavior.OPTIONAL.value

    return Field(
        ...,
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

    if value.startswith("\\$."):
        return False  # Escaped literal

    return value.startswith("$.")


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


class SchemaCheck(JSONPathValidatedModel, ABC):
    """
    Abstract base class for typed check schemas.

    Provides a type-safe, strongly-typed alternative to the generic Check class
    while maintaining full backward compatibility with the existing evaluation engine.

    Benefits:
    - Eliminates untyped arguments dictionaries
    - Provides IDE autocompletion and type checking
    - Adds validation at creation time
    - Seamless integration with existing evaluate() function
    """

    metadata: dict[str, Any] | None = None

    # Class attribute that subclasses must override
    CHECK_TYPE: ClassVar[CheckType]

    @property
    def check_type(self) -> CheckType:
        """Return the CheckType for this check."""
        return self.CHECK_TYPE

    def to_check(self) -> Check:
        """Convert to generic Check object for execution."""
        data = self.model_dump()
        metadata = data.pop('metadata', None)

        return Check(
            type=self.check_type,
            arguments=data,
            version=getattr(self, 'VERSION', None),
            metadata=metadata,
        )


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
