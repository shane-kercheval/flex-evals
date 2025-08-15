"""Check classes for test utility checks."""

from typing import Any
from pydantic import Field, field_validator

from flex_evals.checks.base import BaseCheck, BaseAsyncCheck, JSONPath
from flex_evals.registry import register


@register("test_check", version="1.0.0")
class TestCheck(BaseCheck):
    """Test utility check."""

    expected: str | JSONPath = Field(
        "Paris",
        description="Expected value or JSONPath expression pointing to the expected value",
    )
    actual: str | JSONPath | None = Field(
        None,
        description="Actual value or JSONPath expression pointing to the actual value",
    )

    @field_validator('expected', 'actual', mode='before')
    @classmethod
    def convert_jsonpath(cls, v):  # noqa: ANN001
        """Convert JSONPath-like strings to JSONPath objects."""
        if isinstance(v, str) and v.startswith('$.'):
            return JSONPath(expression=v)
        return v

    def __call__(self) -> dict[str, Any]:
        """Execute test check comparison using resolved Pydantic fields."""
        # Validate that all fields are resolved (no JSONPath objects remain)
        if isinstance(self.expected, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'expected' field: {self.expected}")
        if isinstance(self.actual, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'actual' field: {self.actual}")

        return {"passed": str(self.expected) == str(self.actual)}


@register("test_async_check", version="1.0.0")
class TestAsyncCheck(BaseAsyncCheck):
    """Test utility async check."""

    expected: str | JSONPath = Field(
        "Paris",
        description="Expected value or JSONPath expression pointing to the expected value",
    )
    actual: str | JSONPath | None = Field(
        None,
        description="Actual value or JSONPath expression pointing to the actual value",
    )

    @field_validator('expected', 'actual', mode='before')
    @classmethod
    def convert_jsonpath(cls, v):  # noqa: ANN001
        """Convert JSONPath-like strings to JSONPath objects."""
        if isinstance(v, str) and v.startswith('$.'):
            return JSONPath(expression=v)
        return v

    async def __call__(self) -> dict[str, Any]:
        """Execute async test check comparison using resolved Pydantic fields."""
        # Validate that all fields are resolved (no JSONPath objects remain)
        if isinstance(self.expected, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'expected' field: {self.expected}")
        if isinstance(self.actual, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'actual' field: {self.actual}")

        return {"passed": str(self.expected) == str(self.actual)}


@register("test_failing_check", version="1.0.0")
class TestFailingCheck(BaseCheck):
    """Combined test utility check that always fails."""

    # No fields - this check takes no parameters and always fails

    def __call__(self) -> dict[str, Any]:
        """Execute test check that always fails."""
        return {"passed": False}


# # TODO: Convert remaining test utilities to combined approach as needed
# # class SlowAsyncCheck(SchemaCheck):
#     """Schema for SlowAsyncCheck test utility."""
#
#     CHECK_TYPE: ClassVar[str] = "slow_async_check"  # Custom check type
#     VERSION: ClassVar[str] = "1.0.0"
#
#     delay: float = Field(0.1, description="Sleep duration in seconds")
#
#
# class CustomUserCheck(SchemaCheck):
#     """Schema for CustomUserCheck test utility."""
#
#     CHECK_TYPE: ClassVar[str] = "custom_user_check"  # Custom check type
#     VERSION: ClassVar[str] = "1.0.0"
#
#     test_value: str = Field("expected", description="Test value for verification")
#
#
# class AsyncSleepCheck(SchemaCheck):
#     """Schema for AsyncSleepCheck test utility."""
#
#     CHECK_TYPE: ClassVar[str] = "async_sleep"  # Custom check type
#     VERSION: ClassVar[str] = "1.0.0"
#
#     sleep_duration: float = Field(0.1, description="Sleep duration in seconds")
#
#
# # Schema classes for engine versioning tests
# class VersionTestV1(SchemaCheck):
#     """Schema for version_test check v1.0.0."""
#
#     CHECK_TYPE: ClassVar[str] = "version_test"
#     VERSION: ClassVar[str] = "1.0.0"
#
#     # No fields - test check takes no parameters
#
#
# class VersionTestV2(SchemaCheck):
#     """Schema for version_test check v2.0.0."""
#
#     CHECK_TYPE: ClassVar[str] = "version_test"
#     VERSION: ClassVar[str] = "2.0.0"
#
#     # No fields - test check takes no parameters
#
#
# class LatestTestV1(SchemaCheck):
#     """Schema for latest_test check v1.0.0."""
#
#     CHECK_TYPE: ClassVar[str] = "latest_test"
#     VERSION: ClassVar[str] = "1.0.0"
#
#     # No fields - test check takes no parameters
#
#
# class LatestTestV2_1(SchemaCheck):
#     """Schema for latest_test check v2.1.0."""
#
#     CHECK_TYPE: ClassVar[str] = "latest_test"
#     VERSION: ClassVar[str] = "2.1.0"
#
#     # No fields - test check takes no parameters
#
#
# class LatestTestV2_0(SchemaCheck):
#     """Schema for latest_test check v2.0.0."""
#
#     CHECK_TYPE: ClassVar[str] = "latest_test"
#     VERSION: ClassVar[str] = "2.0.0"
#
#     # No fields - test check takes no parameters
#
#
# class AsyncVersionTestV1(SchemaCheck):
#     """Schema for async_version_test check v1.0.0."""
#
#     CHECK_TYPE: ClassVar[str] = "async_version_test"
#     VERSION: ClassVar[str] = "1.0.0"
#
#     # No fields - test check takes no parameters
#
#
# class AsyncVersionTestV2(SchemaCheck):
#     """Schema for async_version_test check v2.0.0."""
#
#     CHECK_TYPE: ClassVar[str] = "async_version_test"
#     VERSION: ClassVar[str] = "2.0.0"
#
#     # No fields - test check takes no parameters
#
#
# class MixedTestV1Sync(SchemaCheck):
#     """Schema for mixed_test check v1.0.0."""
#
#     CHECK_TYPE: ClassVar[str] = "mixed_test"
#     VERSION: ClassVar[str] = "1.0.0"
#
#     # No fields - test check takes no parameters
#
#
# class MixedTestV2Async(SchemaCheck):
#     """Schema for mixed_test check v2.0.0."""
#
#     CHECK_TYPE: ClassVar[str] = "mixed_test"
#     VERSION: ClassVar[str] = "2.0.0"
#
#     # No fields - test check takes no parameters
#
#
# class ErrorTestV1(SchemaCheck):
#     """Schema for error_test check v1.0.0."""
#
#     CHECK_TYPE: ClassVar[str] = "error_test"
#     VERSION: ClassVar[str] = "1.0.0"
#
#     # No fields - test check takes no parameters
#
#
# class ErrorTestV2(SchemaCheck):
#     """Schema for error_test check v2.0.0."""
#
#     CHECK_TYPE: ClassVar[str] = "error_test"
#     VERSION: ClassVar[str] = "2.0.0"
#
#     # No fields - test check takes no parameters
#
#
# class MetadataTestV1_5(SchemaCheck):
#     """Schema for metadata_test check v1.5.0."""
#
#     CHECK_TYPE: ClassVar[str] = "metadata_test"
#     VERSION: ClassVar[str] = "1.5.0"
#
#     # No fields - test check takes no parameters
#
#
# class LatestCheckV1(SchemaCheck):
#     """Schema for latest_check check v1.0.0."""
#
#     CHECK_TYPE: ClassVar[str] = "latest_check"
#     VERSION: ClassVar[str] = "1.0.0"
#
#     # No fields - test check takes no parameters
#
#
# class LatestCheckV2_1(SchemaCheck):
#     """Schema for latest_check check v2.1.0."""
#
#     CHECK_TYPE: ClassVar[str] = "latest_check"
#     VERSION: ClassVar[str] = "2.1.0"
#
#     # No fields - test check takes no parameters
#
#
# class MixedCheckV1(SchemaCheck):
#     """Schema for mixed_check test v1.0.0."""
#
#     CHECK_TYPE: ClassVar[str] = "mixed_check"
#     VERSION: ClassVar[str] = "1.0.0"
#
#     # No fields - test check takes no parameters
#
#
# class TestCaseCheckV1(SchemaCheck):
#     """Schema for testcase_check test v1.0.0."""
#
#     CHECK_TYPE: ClassVar[str] = "testcase_check"
#     VERSION: ClassVar[str] = "1.0.0"
#
#     text: str = OptionalJSONPath("Text input for the check")
#
#
# class TestCaseCheckV1_5(SchemaCheck):
#     """Schema for testcase_check test v1.5.0."""
#
#     CHECK_TYPE: ClassVar[str] = "testcase_check"
#     VERSION: ClassVar[str] = "1.5.0"
#
#     text: str = OptionalJSONPath("Text input for the check")
