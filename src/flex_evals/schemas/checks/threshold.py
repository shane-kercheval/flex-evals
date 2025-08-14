"""ThresholdCheck schema class for type-safe threshold check definitions."""

from typing import ClassVar
from pydantic import Field, model_validator

from ...constants import CheckType
from ..check import Check, SchemaCheck, OptionalJSONPath


class ThresholdCheck(SchemaCheck):
    """Checks if a numeric value meets minimum/maximum thresholds."""

    VERSION: ClassVar[str] = "1.0.0"
    CHECK_TYPE: ClassVar[CheckType] = CheckType.THRESHOLD

    value: str | int | float = OptionalJSONPath(
        "Numeric value to check or JSONPath expression pointing to the value",
    )
    min_value: str | int | float | None = OptionalJSONPath(
        "Minimum acceptable value, or JSONPath expression pointing to min value",
        default=None,
    )
    max_value: str | int | float | None = OptionalJSONPath(
        "Maximum acceptable value, or JSONPath expression pointing to max value",
        default=None,
    )
    min_inclusive: str | bool = OptionalJSONPath(
        "If true, min_value is inclusive (>=), if false, exclusive (>), or JSONPath expression pointing to boolean",  # noqa: E501
        default=True,
    )
    max_inclusive: str | bool = OptionalJSONPath(
        "If true, max_value is inclusive (<=), if false, exclusive (<), or JSONPath expression pointing to boolean",  # noqa: E501
        default=True,
    )
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
