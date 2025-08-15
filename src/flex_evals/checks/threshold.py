"""
Combined ThresholdCheck implementation for FEP.

Combines schema validation with execution logic in a single class.
"""

from typing import Any
from pydantic import field_validator, Field

from .base import BaseCheck, JSONPath, _convert_to_jsonpath
from ..registry import register
from ..exceptions import ValidationError
from ..constants import CheckType


@register(CheckType.THRESHOLD, version='1.0.0')
class ThresholdCheck(BaseCheck):
    """Checks if a numeric value meets minimum/maximum thresholds."""

    # Pydantic fields with validation
    value: float | int | JSONPath = Field(
        ...,
        description="Numeric value to check or JSONPath expression",
    )
    min_value: float | int | JSONPath | None = Field(
        None,
        description="Minimum acceptable value or JSONPath expression",
    )
    max_value: float | int | JSONPath | None = Field(
        None,
        description="Maximum acceptable value or JSONPath expression",
    )
    min_inclusive: bool | JSONPath = Field(
        True,
        description="If true, min_value is inclusive (>=), else exclusive (>)",
    )
    max_inclusive: bool | JSONPath = Field(
        True,
        description="If true, max_value is inclusive (<=), else exclusive (<)",
    )
    negate: bool | JSONPath = Field(
        False,
        description="If true, passes when value is outside the specified range",
    )


    @field_validator(
        'value', 'min_value', 'max_value', 'min_inclusive', 'max_inclusive', 'negate',
        mode='before',
    )
    @classmethod
    def convert_jsonpath(cls, value: object) -> object | JSONPath:
        """Convert JSONPath-like strings to JSONPath objects."""
        return _convert_to_jsonpath(value)

    @field_validator('min_value', 'max_value')
    @classmethod
    def validate_threshold_values(
            cls, v: float | int | JSONPath | None,
        ) -> float | int | JSONPath | None:
        """Validate threshold values are numeric or JSONPath objects."""
        if v is not None and not isinstance(v, int | float | JSONPath):
            raise ValueError(
                f"Threshold values must be numeric or JSONPath, got: {type(v).__name__}",
            )
        return v

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        """Validate that at least one threshold is specified."""
        if self.min_value is None and self.max_value is None:
            raise ValueError("Threshold check requires at least one of 'min_value' or 'max_value'")

    def __call__(self) -> dict[str, Any]:  # noqa: PLR0912
        """
        Execute threshold check using resolved Pydantic fields.

        All JSONPath objects should have been resolved by execute() before this is called.

        Returns:
            Dictionary with 'passed' key indicating check result

        Raises:
            RuntimeError: If any field contains unresolved JSONPath objects
        """
        # Validate that all fields are resolved (no JSONPath objects remain)
        if isinstance(self.value, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'value' field: {self.value}")
        if isinstance(self.min_value, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'min_value' field: {self.min_value}")
        if isinstance(self.max_value, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'max_value' field: {self.max_value}")
        if isinstance(self.min_inclusive, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'min_inclusive' field: {self.min_inclusive}")  # noqa: E501
        if isinstance(self.max_inclusive, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'max_inclusive' field: {self.max_inclusive}")  # noqa: E501
        if isinstance(self.negate, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'negate' field: {self.negate}")

        # Check that at least one threshold is specified
        if self.min_value is None and self.max_value is None:
            raise ValidationError("Threshold check requires at least one of 'min_value' or 'max_value'")  # noqa: E501

        # Validate value is numeric
        value = self.value
        if not isinstance(value, int | float):
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValidationError(f"Threshold check 'value' must be numeric, got: {type(value).__name__}")  # noqa: E501

        # Validate thresholds are numeric
        min_value = self.min_value
        max_value = self.max_value
        if min_value is not None and not isinstance(min_value, int | float):
            raise ValidationError(f"Threshold check 'min_value' must be numeric, got: {type(min_value).__name__}")  # noqa: E501

        if max_value is not None and not isinstance(max_value, int | float):
            raise ValidationError(f"Threshold check 'max_value' must be numeric, got: {type(max_value).__name__}")  # noqa: E501

        # Check minimum threshold
        min_passed = True
        if min_value is not None:
            min_passed = value >= min_value if self.min_inclusive else value > min_value

        # Check maximum threshold
        max_passed = True
        if max_value is not None:
            max_passed = value <= max_value if self.max_inclusive else value < max_value

        # Overall threshold check
        within_bounds = min_passed and max_passed

        # Apply negation
        passed = not within_bounds if self.negate else within_bounds

        return {'passed': passed}
