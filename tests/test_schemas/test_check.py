"""Tests for Check and CheckResult schema implementations."""

import dataclasses
import pytest
from datetime import datetime, UTC
from flex_evals import Check, CheckResult, CheckError, CheckType, ErrorType


class TestCheck:
    """Test Check schema implementation."""

    def test_check_required_fields(self):
        """Test Check with type and arguments required."""
        check = Check(
            type='exact_match',
            arguments={"actual": "$.output.value", "expected": "Paris"},
        )

        assert check.type == 'exact_match'
        assert check.arguments == {"actual": "$.output.value", "expected": "Paris"}
        assert check.version is None


    def test_check_version_semver(self):
        """Test valid semver versions."""
        valid_versions = ["1.0.0", "2.1.3", "0.0.1", "1.0.0-alpha", "2.1.3-beta.1", "1.0.0+build.1"]  # noqa: E501

        for version in valid_versions:
            check = Check(type='exact_match', arguments={}, version=version)
            assert check.version == version

    def test_check_version_invalid(self):
        """Test invalid versions raise ValidationError."""
        invalid_versions = ["1.0", "v1.0.0", "1.0.0.0", "1.0.0-", "1.0.0+", "invalid"]

        for version in invalid_versions:
            with pytest.raises(ValueError, match="Check.version must be valid semver format"):
                Check(type='exact_match', arguments={}, version=version)

    def test_check_empty_type_error(self):
        """Test empty type string raises error."""
        with pytest.raises(ValueError, match="Check.type must be a non-empty string"):
            Check(type="", arguments={})

        with pytest.raises(ValueError, match="Check.type must be a non-empty string"):
            Check(type=None, arguments={})


    def test_check_arguments_dict(self):
        """Test arguments must be Dict."""
        with pytest.raises(ValueError, match="Check.arguments must be a dictionary"):
            Check(type='exact_match', arguments="not a dict")

        with pytest.raises(ValueError, match="Check.arguments must be a dictionary"):
            Check(type='exact_match', arguments=["list", "not", "allowed"])


class TestCheckResult:
    """Test CheckResult schema implementation."""

    def test_check_result_all_required_fields(self):
        """Test CheckResult with all required fields present."""
        metadata = {
            "check_version": "1.0.0",
            "execution_time_ms": 245,
        }

        result = CheckResult(
            check_type='exact_match',
            status='completed',
            results={"passed": True},
            resolved_arguments={
                "actual": {"value": "Paris", "jsonpath": "$.output.value"},
                "expected": {"value": "Paris"},
            },
            evaluated_at=datetime.now(UTC),
            metadata=metadata,
        )

        assert result.check_type == 'exact_match'
        assert result.status == 'completed'
        assert result.results["passed"] is True
        assert result.metadata["check_version"] == "1.0.0"

    def test_check_result_status_enum(self):
        """Test valid status values."""
        metadata = {}
        base_args = {
            "check_type": 'exact_match',
            "results": {"passed": True},
            "resolved_arguments": {"actual": {"value": "test"}},
            "evaluated_at": datetime.now(UTC),
            "metadata": metadata,
        }

        # Test completed and skip statuses
        for status in ['completed', 'skip']:
            result = CheckResult(status=status, **base_args)
            assert result.status == status

        # Test error status with error object
        error = CheckError(type='validation_error', message="Test error")
        result = CheckResult(status='error', error=error, **base_args)
        assert result.status == 'error'

    def test_check_result_resolved_args_jsonpath(self):
        """Test resolved_arguments with JSONPath includes both value and jsonpath fields."""
        metadata = {}

        result = CheckResult(
            check_type='exact_match',
            status='completed',
            results={"passed": True},
            resolved_arguments={
                "actual": {
                    "value": "Paris",
                    "jsonpath": "$.output.value",
                },
                "expected": {
                    "value": "Paris",
                    "jsonpath": "$.test_case.expected",
                },
            },
            evaluated_at=datetime.now(UTC),
            metadata=metadata,
        )

        assert result.resolved_arguments["actual"]["value"] == "Paris"
        assert result.resolved_arguments["actual"]["jsonpath"] == "$.output.value"
        assert result.resolved_arguments["expected"]["value"] == "Paris"
        assert result.resolved_arguments["expected"]["jsonpath"] == "$.test_case.expected"

    def test_check_result_resolved_args_literal(self):
        """Test resolved_arguments with literals only includes value field."""
        metadata = {}

        result = CheckResult(
            check_type='exact_match',
            status='completed',
            results={"passed": True},
            resolved_arguments={
                "actual": {"value": "Paris", "jsonpath": "$.output.value"},
                "case_sensitive": {"value": True},  # No jsonpath field for literals
            },
            evaluated_at=datetime.now(UTC),
            metadata=metadata,
        )

        # JSONPath argument has both fields
        assert result.resolved_arguments["actual"]["value"] == "Paris"
        assert result.resolved_arguments["actual"]["jsonpath"] == "$.output.value"

        # Literal argument only has value field
        assert result.resolved_arguments["case_sensitive"]["value"] is True
        assert "jsonpath" not in result.resolved_arguments["case_sensitive"]

    def test_check_result_timestamp_format(self):
        """Test evaluated_at serializes to ISO 8601 UTC."""
        metadata = {}
        timestamp = datetime.now(UTC)

        result = CheckResult(
            check_type='exact_match',
            status='completed',
            results={"passed": True},
            resolved_arguments={"actual": {"value": "test"}},
            evaluated_at=timestamp,
            metadata=metadata,
        )

        assert result.evaluated_at == timestamp
        assert result.evaluated_at.tzinfo == UTC

        # Test ISO 8601 format
        iso_string = result.evaluated_at.isoformat()
        assert iso_string.endswith("+00:00")

    def test_check_result_metadata_test_case_id(self):
        """Test metadata check_version defaults."""
        metadata = {}

        result = CheckResult(
            check_type='exact_match',
            status='completed',
            results={"passed": True},
            resolved_arguments={"actual": {"value": "test"}},
            evaluated_at=datetime.now(UTC),
            metadata=metadata,
        )

        assert result.metadata.get("check_version") is None

        # Test with explicit version
        metadata_with_version = {"check_version": "2.0.0"}
        result_with_version = CheckResult(
            check_type='exact_match',
            status='completed',
            results={"passed": True},
            resolved_arguments={"actual": {"value": "test"}},
            evaluated_at=datetime.now(UTC),
            metadata=metadata_with_version,
        )
        assert result_with_version.metadata["check_version"] == "2.0.0"

    def test_check_result_error_types(self):
        """Test all valid error.type enum values."""
        metadata = {}
        error_types = ['jsonpath_error', 'validation_error', "timeout_error", 'unknown_error']

        for error_type in error_types:
            error = CheckError(type=error_type, message="Test error")
            result = CheckResult(
                check_type='exact_match',
                status='error',
                results={},
                resolved_arguments={"actual": {"value": "test"}},
                evaluated_at=datetime.now(UTC),
                metadata=metadata,
                error=error,
            )
            assert result.error.type == error_type

    def test_check_result_error_optional(self):
        """Test error only present when status='error'."""
        metadata = {}

        # No error when status is completed
        result1 = CheckResult(
            check_type='exact_match',
            status='completed',
            results={"passed": True},
            resolved_arguments={"actual": {"value": "test"}},
            evaluated_at=datetime.now(UTC),
            metadata=metadata,
        )
        assert result1.error is None

        # Error required when status is error
        with pytest.raises(ValueError, match="CheckResult.error is required when status is 'error'"):  # noqa: E501
            CheckResult(
                check_type='exact_match',
                status='error',
                results={},
                resolved_arguments={"actual": {"value": "test"}},
                evaluated_at=datetime.now(UTC),
                metadata=metadata,
            )

    def test_check_result_error_required_fields(self):
        """Test error.type and error.message required."""
        error = CheckError(type='validation_error', message="Invalid input")
        assert error.type == 'validation_error'
        assert error.message == "Invalid input"
        assert error.recoverable is False  # Default value

        # Test with recoverable=True
        error2 = CheckError(type="timeout_error", message="Request timed out", recoverable=True)
        assert error2.recoverable is True

    def test_check_result_serialization(self):
        """Test full CheckResult JSON serialization."""
        metadata = {
            "check_version": "2.1.0",
            "execution_time_ms": 245,
        }

        result = CheckResult(
            check_type='exact_match',
            status='completed',
            results={"passed": True},
            resolved_arguments={
                "actual": {"value": "Paris", "jsonpath": "$.output.value"},
                "expected": {"value": "Paris"},
            },
            evaluated_at=datetime(2025, 6, 26, 19, 4, 23, tzinfo=UTC),
            metadata=metadata,
        )

        # Convert to dict for serialization
        data = dataclasses.asdict(result)

        assert data["check_type"] == 'exact_match'
        assert data["status"] == 'completed'
        assert data["results"]["passed"] is True
        assert data["resolved_arguments"]["actual"]["value"] == "Paris"
        assert data["metadata"]["check_version"] == "2.1.0"
        assert data["metadata"]["execution_time_ms"] == 245
        assert data['error'] is None

    def test_check_accepts_check_type_enum(self):
        """Test that Check schema accepts CheckType enums."""
        check = Check(
            type=CheckType.EXACT_MATCH,
            arguments={"actual": "test", "expected": "test"},
        )
        # Should be converted to string internally
        assert check.type == 'exact_match'
        assert isinstance(check.type, str)

    def test_check_accepts_check_type_string(self):
        """Test that Check schema accepts string types."""
        check = Check(
            type='exact_match',
            arguments={"actual": "test", "expected": "test"},
        )
        assert check.type == 'exact_match'

    def test_check_accepts_custom_string_type(self):
        """Test that Check schema accepts custom check type strings."""
        check = Check(
            type="custom_check_type",
            arguments={"param": "value"},
        )
        assert check.type == "custom_check_type"

    def test_check_type_enum_values(self):
        """Test that CheckType enum values match their string equivalents."""
        assert CheckType.CONTAINS == 'contains'
        assert CheckType.EXACT_MATCH == 'exact_match'
        assert CheckType.REGEX == 'regex'
        assert CheckType.THRESHOLD == 'threshold'
        assert CheckType.SEMANTIC_SIMILARITY == 'semantic_similarity'
        assert CheckType.LLM_JUDGE == 'llm_judge'

    def test_check_type_enum_string_conversion(self):
        """Test that CheckType enums convert to strings properly."""
        assert str(CheckType.CONTAINS) == 'contains'
        assert str(CheckType.EXACT_MATCH) == 'exact_match'

    def test_check_type_enum_equality_with_strings(self):
        """Test that CheckType enums compare equal to their string values."""
        assert CheckType.CONTAINS == 'contains'
        assert CheckType.EXACT_MATCH == 'exact_match'

    def test_check_error_accepts_error_type_enum(self):
        """Test CheckError accepts ErrorType enum."""
        error = CheckError(
            type=ErrorType.VALIDATION_ERROR,
            message="Test validation error",
            recoverable=False,
        )
        # Should work with enum
        assert error.type in [ErrorType.VALIDATION_ERROR, 'validation_error']

    def test_check_error_accepts_error_type_string(self):
        """Test CheckError accepts error type string."""
        error = CheckError(
            type='validation_error',
            message="Test validation error",
            recoverable=False,
        )
        assert error.type == 'validation_error'

    def test_error_type_enum_values(self):
        """Test ErrorType enum values."""
        assert ErrorType.JSONPATH_ERROR == 'jsonpath_error'
        assert ErrorType.VALIDATION_ERROR == 'validation_error'
        assert ErrorType.TIMEOUT_ERROR == "timeout_error"
        assert ErrorType.UNKNOWN_ERROR == 'unknown_error'

    def test_backward_compatibility_and_mixed_usage(self):
        """Test that existing string-based code continues to work and can be mixed with enums."""
        # This represents existing user code that uses strings
        check_with_string = Check(
            type='exact_match',
            arguments={"actual": "test", "expected": "test"},
        )

        # New code using enums
        check_with_enum = Check(
            type=CheckType.CONTAINS,
            arguments={"text": "test", "phrases": ["test"]},
        )

        # Both should work
        assert check_with_string.type == 'exact_match'
        assert check_with_enum.type == 'contains'

        # Mixed error types
        error_with_string = CheckError(
            type='validation_error',
            message="Test error",
            recoverable=False,
        )

        error_with_enum = CheckError(
            type=ErrorType.UNKNOWN_ERROR,
            message="Test error",
            recoverable=False,
        )

        assert error_with_string.type == 'validation_error'
        assert error_with_enum.type in [ErrorType.UNKNOWN_ERROR, 'unknown_error']
