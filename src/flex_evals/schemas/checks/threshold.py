"""ThresholdCheck schema class for type-safe threshold check definitions."""

from typing import ClassVar
from pydantic import Field, model_validator

from ...constants import CheckType
from ..check import Check, SchemaCheck, OptionalJSONPath


class ThresholdCheck(SchemaCheck):
    """
    Type-safe schema for threshold check.

    Checks if a numeric value meets minimum/maximum thresholds with configurable inclusivity.

    Fields:
    - value: numeric value to check or JSONPath expression pointing to the value
    - min_value: Minimum acceptable value (optional)
    - max_value: Maximum acceptable value (optional)
    - min_inclusive: If true, min_value is inclusive (>=), if false, exclusive (>) (default: True)
    - max_inclusive: If true, max_value is inclusive (<=), if false, exclusive (<) (default: True)
    - negate: If true, passes when value is outside the specified range (default: False)
    - version: Optional version string for the check

    At least one of min_value or max_value must be specified.
    """

    VERSION: ClassVar[str] = "1.0.0"
    CHECK_TYPE: ClassVar[CheckType] = CheckType.THRESHOLD

    value: str = OptionalJSONPath(
        "numeric value to check or JSONPath expression pointing to the value",
        min_length=1,
    )
    min_value: float | None = Field(None, description="Minimum acceptable value")
    max_value: float | None = Field(None, description="Maximum acceptable value")
    min_inclusive: bool = Field(True, description="If true, min_value is inclusive (>=), if false, exclusive (>)")  # noqa: E501
    max_inclusive: bool = Field(True, description="If true, max_value is inclusive (<=), if false, exclusive (<)")  # noqa: E501
    negate: bool = Field(False, description="If true, passes when value is outside the specified range")  # noqa: E501

    @model_validator(mode='after')
    def validate_thresholds(self) -> 'ThresholdCheck':
        """Validate that at least one threshold is specified."""
        if self.min_value is None and self.max_value is None:
            raise ValueError("At least one of 'min_value' or 'max_value' must be specified")
        return self

    def to_check(self) -> Check:
        """Convert to generic Check object for execution."""
        arguments = {
            "value": self.value,
            "min_inclusive": self.min_inclusive,
            "max_inclusive": self.max_inclusive,
            "negate": self.negate,
        }

        if self.min_value is not None:
            arguments["min_value"] = self.min_value

        if self.max_value is not None:
            arguments["max_value"] = self.max_value

        return Check(
            type=self.check_type,
            arguments=arguments,
            version=self.VERSION,
        )
