"""
Threshold check implementation for FEP.

Checks if a numeric value meets minimum/maximum thresholds with configurable inclusivity.
"""

from typing import Any

from ..base import BaseCheck, EvaluationContext
from ...registry import register
from ...exceptions import ValidationError


@register("threshold", version="1.0.0")
class ThresholdCheck(BaseCheck):
    """
    Checks if a numeric value meets minimum/maximum thresholds.

    Arguments Schema:
    - value: number | JSONPath - Numeric value to check
    - min_value: number (optional) - Minimum acceptable value
    - max_value: number (optional) - Maximum acceptable value
    - min_inclusive: boolean (default: true) - If true, min_value is inclusive (>=), if false, exclusive (>)
    - max_inclusive: boolean (default: true) - If true, max_value is inclusive (<=), if false, exclusive (<)
    - negate: boolean (default: false) - If true, passes when value is outside the specified range

    At least one of min_value or max_value must be specified.

    Results Schema:
    - passed: boolean - Whether the threshold check passed
    """

    def __call__(self, arguments: dict[str, Any], context: EvaluationContext) -> dict[str, Any]:
        # Validate required arguments
        if "value" not in arguments:
            raise ValidationError("Threshold check requires 'value' argument")

        # Check that at least one threshold is specified
        min_value = arguments.get("min_value")
        max_value = arguments.get("max_value")

        if min_value is None and max_value is None:
            raise ValidationError("Threshold check requires at least one of 'min_value' or 'max_value'")

        # Get argument values with defaults
        value = arguments["value"]
        min_inclusive = arguments.get("min_inclusive", True)
        max_inclusive = arguments.get("max_inclusive", True)
        negate = arguments.get("negate", False)

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

        return {"passed": passed}
