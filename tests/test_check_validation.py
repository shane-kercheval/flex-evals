"""Tests for Check object validation against SchemaCheck schemas."""

import pytest
from typing import Any

from flex_evals import evaluate, TestCase, Output, Check
from flex_evals.checks.exact_match import ExactMatchCheck
from flex_evals.exceptions import ValidationError
from flex_evals.checks.base import BaseCheck
from flex_evals.registry import register, clear_registry
from tests.conftest import restore_standard_checks


class TestCheckValidation:
    """Test validation of Check objects against their corresponding SchemaCheck classes."""

    def setup_method(self):
        """Set up test fixtures."""
        restore_standard_checks()
        self.test_case = TestCase(id="test_001", input="test")
        self.output = Output(value="Paris")

    def test_valid_check_passes_validation(self):
        """Test that valid Check objects pass full Pydantic validation."""
        check = Check(
            type="exact_match",
            arguments={"actual": "$.output.value", "expected": "Paris"},
        )

        result = evaluate([self.test_case], [self.output], [check])
        assert result.status == 'completed'

    def test_missing_required_field_fails_validation(self):
        """Test that missing required fields fail validation."""
        check = Check(
            type="exact_match",
            arguments={"actual": "$.output.value"},  # Missing required 'expected'
        )

        with pytest.raises(ValidationError, match="Check arguments validation failed for 'exact_match'"):  # noqa: E501
            evaluate([self.test_case], [self.output], [check])

    def test_invalid_type_fails_validation(self):
        """Test that ExactMatchCheck now accepts any types and converts them to strings."""
        # ExactMatchCheck now accepts any types and converts them to strings for comparison
        check = Check(
            type="exact_match",
            arguments={"actual": "$.output.value", "expected": 123},
        )

        # This should now succeed because ExactMatch converts everything to strings
        result = evaluate([self.test_case], [self.output], [check])
        assert result.status == 'completed'
        # The comparison will be str(output.value) == str(123)

    def test_invalid_jsonpath_fails_validation(self):
        """Test that invalid JSONPath expressions fail validation."""
        check = Check(
            type="exact_match",
            arguments={
                "actual": "$.invalid[unclosed",  # Invalid JSONPath syntax
                "expected": "Paris",
            },
        )

        with pytest.raises(ValidationError, match="Check arguments validation failed for 'exact_match'"):  # noqa: E501
            evaluate([self.test_case], [self.output], [check])

    def test_business_rule_validation_fails(self):
        """Test that business rule violations fail validation."""
        # ThresholdCheck requires at least one of min_value or max_value
        check = Check(
            type="threshold",
            arguments={"value": 0.5},  # Neither min_value nor max_value provided
        )

        with pytest.raises(ValidationError, match="Check arguments validation failed for 'threshold'"):  # noqa: E501
            evaluate([self.test_case], [self.output], [check])

    def test_extra_fields_validation(self):
        """Test that extra fields are rejected by strict validation."""
        check = Check(
            type="exact_match",
            arguments={
                "actual": "$.output.value",
                "expected": "Paris",
                "invalid_extra_field": "should_be_rejected",  # Should fail validation
            },
        )

        # Should fail - SchemaCheck classes use strict validation to catch typos
        with pytest.raises(ValidationError, match="Check arguments validation failed for 'exact_match'"):  # noqa: E501
            evaluate([self.test_case], [self.output], [check])

    def test_boolean_field_type_validation(self):
        """Test validation of boolean fields with wrong types."""
        check = Check(
            type="exact_match",
            arguments={
                "actual": "$.output.value",
                "expected": "Paris",
                "case_sensitive": "not_a_boolean",  # Should be boolean
            },
        )

        with pytest.raises(ValidationError, match="Check arguments validation failed for 'exact_match'"):  # noqa: E501
            evaluate([self.test_case], [self.output], [check])

    def test_complex_validation_rules(self):
        """Test complex validation rules specific to check types."""
        # ContainsCheck with invalid phrases - empty list should fail at execution time
        check = Check(
            type="contains",
            arguments={
                "text": "$.output.value",
                "phrases": [],  # Empty list should fail validation
            },
        )

        # The evaluation should result in an error status due to check execution failure
        result = evaluate([self.test_case], [self.output], [check])
        assert result.status == 'error'
        assert result.results[0].status == 'error'
        # Check that the error is about empty phrases
        assert "phrases" in result.results[0].check_results[0].error.message

    def test_version_specific_validation(self):
        """Test validation with specific version."""
        check = Check(
            type="exact_match",
            arguments={"actual": "$.output.value", "expected": "Paris"},
            version="1.0.0",
        )

        result = evaluate([self.test_case], [self.output], [check])
        assert result.status == 'completed'

    def test_nonexistent_version_fails(self):
        """Test that non-existent versions fail."""
        check = Check(
            type="exact_match",
            arguments={"actual": "$.output.value", "expected": "Paris"},
            version="99.0.0",  # Non-existent version
        )

        with pytest.raises(ValueError, match="Version '99.0.0' not found for check type 'exact_match'"):  # noqa: E501
            evaluate([self.test_case], [self.output], [check])

    def test_unregistered_check_type_fails(self):
        """Test that unregistered check types fail early."""
        check = Check(
            type="nonexistent_check_type",
            arguments={"some": "argument"},
        )

        with pytest.raises(ValueError, match="Check type 'nonexistent_check_type' is not registered"):  # noqa: E501
            evaluate([self.test_case], [self.output], [check])

    def test_custom_check_without_schema_fails(self):
        """Test that custom checks without proper Pydantic fields fail validation."""
        # Register a custom check without proper Pydantic fields
        @register("custom_no_schema", version="1.0.0")
        class CustomCheck(BaseCheck):
            def __call__(self) -> dict[str, Any]:
                return {"passed": True}

        try:
            check = Check(
                type="custom_no_schema",
                arguments={"some": "argument"},
            )

            # Should fail because the check doesn't accept 'some' field
            with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
                evaluate([self.test_case], [self.output], [check])
        finally:
            clear_registry()
            restore_standard_checks()

    def test_schemacheck_objects_bypass_additional_validation(self):
        """Test that SchemaCheck objects don't get re-validated."""
        # SchemaCheck objects are already validated, so they should be converted directly
        schema_check = ExactMatchCheck(actual="$.output.value", expected="Paris")

        result = evaluate([self.test_case], [self.output], [schema_check])
        assert result.status == 'completed'

    def test_validation_preserves_check_structure(self):
        """Test that validation doesn't modify the original Check object."""
        original_check = Check(
            type="exact_match",
            arguments={"actual": "$.output.value", "expected": "Paris"},
            metadata={"test": "data"},
        )

        result = evaluate([self.test_case], [self.output], [original_check])
        assert result.status == 'completed'

        # Original check should be unchanged
        assert original_check.type == "exact_match"
        assert original_check.arguments == {"actual": "$.output.value", "expected": "Paris"}
        assert original_check.metadata == {"test": "data"}

    def test_per_testcase_check_validation(self):
        """Test that per-test-case checks are also validated."""
        test_case_with_invalid_check = TestCase(
            id="test_001",
            input="test",
            checks=[
                Check(
                    type="exact_match",
                    arguments={"actual": "$.output.value"},  # Missing 'expected'
                ),
            ],
        )

        with pytest.raises(ValidationError, match="Check arguments validation failed for 'exact_match'"):  # noqa: E501
            evaluate([test_case_with_invalid_check], [self.output])

