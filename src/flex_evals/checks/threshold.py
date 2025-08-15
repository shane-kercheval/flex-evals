"""
Combined ThresholdCheck implementation for FEP.

Combines schema validation with execution logic in a single class.
"""

from typing import Any
from pydantic import Field, field_validator

from .base import BaseCheck, OptionalJSONPath
from ..registry import register
from ..exceptions import ValidationError
from ..constants import CheckType


@register(CheckType.THRESHOLD, version='1.0.0')
class ThresholdCheck(BaseCheck):
    """Checks if a numeric value meets minimum/maximum thresholds."""

    # Pydantic fields with validation
    value: float | int = OptionalJSONPath('Numeric value to check or JSONPath expression')
    min_value: float | int | None = Field(None, description='Minimum acceptable value')
    max_value: float | int | None = Field(None, description='Maximum acceptable value')
    min_inclusive: bool = Field(True, description='If true, min_value is inclusive (>=), else exclusive (>)')
    max_inclusive: bool = Field(True, description='If true, max_value is inclusive (<=), else exclusive (<)')
    negate: bool = Field(False, description='If true, passes when value is outside the specified range')

    @field_validator('min_value', 'max_value')
    @classmethod
    def validate_threshold_values(cls, v: float | int | None) -> float | int | None:
        """Validate threshold values are numeric."""
        if v is not None and not isinstance(v, int | float):
            raise ValueError(f"Threshold values must be numeric, got: {type(v).__name__}")
        return v

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        """Validate that at least one threshold is specified."""
        if self.min_value is None and self.max_value is None:
            raise ValueError("Threshold check requires at least one of 'min_value' or 'max_value'")

    def __call__(
        self,
        value: float | int,
        min_value: float | int | None = None,
        max_value: float | int | None = None,
        min_inclusive: bool = True,
        max_inclusive: bool = True,
        negate: bool = False,
    ) -> dict[str, Any]:
        """
        Execute threshold check with resolved arguments.

        Args:
            value: Resolved numeric value to check
            min_value: Minimum acceptable value (optional)
            max_value: Maximum acceptable value (optional)
            min_inclusive: If true, min_value is inclusive (>=), else exclusive (>)
            max_inclusive: If true, max_value is inclusive (<=), else exclusive (<)
            negate: If true, passes when value is outside the specified range

        Returns:
            Dictionary with 'passed' key indicating check result
        """
        # Check that at least one threshold is specified
        if min_value is None and max_value is None:
            raise ValidationError("Threshold check requires at least one of 'min_value' or 'max_value'")

        # Validate value is numeric
        if not isinstance(value, int | float):
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValidationError(f"Threshold check 'value' must be numeric, got: {type(value).__name__}")

        # Validate thresholds are numeric
        if min_value is not None and not isinstance(min_value, int | float):
            raise ValidationError(f"Threshold check 'min_value' must be numeric, got: {type(min_value).__name__}")

        if max_value is not None and not isinstance(max_value, int | float):
            raise ValidationError(f"Threshold check 'max_value' must be numeric, got: {type(max_value).__name__}")

        # Check minimum threshold
        min_passed = True
        if min_value is not None:
            min_passed = value >= min_value if min_inclusive else value > min_value

        # Check maximum threshold
        max_passed = True
        if max_value is not None:
            max_passed = value <= max_value if max_inclusive else value < max_value

        # Overall threshold check
        within_bounds = min_passed and max_passed

        # Apply negation
        passed = not within_bounds if negate else within_bounds

        return {'passed': passed}
