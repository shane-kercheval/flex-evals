"""
Comprehensive tests for ThresholdCheck implementation.

This module consolidates all tests for the ThresholdCheck including:
- Pydantic validation tests
- Implementation execution tests
- Engine integration tests
- Edge cases and error handling

Tests are organized by functionality rather than implementation details.
"""

import pytest
from typing import Any

from flex_evals import (
    ThresholdCheck,
    JSONPath,
    EvaluationContext,
    CheckType,
    Status,
    evaluate,
    Check,
    ValidationError,
    TestCase,
    Output,
)
from pydantic import ValidationError as PydanticValidationError


class TestThresholdValidation:
    """Test Pydantic validation and field handling for ThresholdCheck."""

    def test_threshold_check_creation_min_only(self):
        """Test basic ThresholdCheck creation with min_value only."""
        check = ThresholdCheck(
            value="$.output.value",
            min_value=10,
        )

        assert isinstance(check.value, JSONPath)
        assert check.value.expression == "$.output.value"
        assert check.min_value == 10
        assert check.max_value is None
        assert check.min_inclusive is True
        assert check.max_inclusive is True
        assert check.negate is False

    def test_threshold_check_creation_max_only(self):
        """Test basic ThresholdCheck creation with max_value only."""
        check = ThresholdCheck(
            value="$.output.value",
            max_value=100,
        )

        assert check.max_value == 100
        assert check.min_value is None

    def test_threshold_check_creation_with_range(self):
        """Test ThresholdCheck creation with both min and max."""
        check = ThresholdCheck(
            value="$.output.value",
            min_value=10,
            max_value=100,
            min_inclusive=False,
            max_inclusive=False,
            negate=True,
        )

        assert check.min_value == 10
        assert check.max_value == 100
        assert check.min_inclusive is False
        assert check.max_inclusive is False
        assert check.negate is True

    def test_threshold_check_with_literal_value(self):
        """Test ThresholdCheck with literal numeric value."""
        check = ThresholdCheck(
            value=42,
            min_value=0,
            max_value=100,
        )

        assert check.value == 42
        assert check.min_value == 0
        assert check.max_value == 100

    def test_threshold_check_type_property(self):
        """Test ThresholdCheck check_type property returns correct type."""
        check = ThresholdCheck(value=50, min_value=0)
        assert check.check_type == CheckType.THRESHOLD

    def test_threshold_validation_no_bounds_error(self):
        """Test that at least one threshold is required during construction."""
        with pytest.raises(ValueError, match="at least one of 'min_value' or 'max_value'"):
            ThresholdCheck(value=50)

    def test_threshold_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(PydanticValidationError):
            ThresholdCheck()  # type: ignore


class TestThresholdExecution:
    """Test ThresholdCheck execution logic and __call__ method."""

    def test_threshold_min_only_pass(self):
        """Test minimum threshold passes when value meets requirement."""
        check = ThresholdCheck(value=50, min_value=30)
        result = check()
        assert result == {"passed": True}

    def test_threshold_min_only_fail(self):
        """Test minimum threshold fails when value below requirement."""
        check = ThresholdCheck(value=20, min_value=30)
        result = check()
        assert result == {"passed": False}

    def test_threshold_max_only_pass(self):
        """Test maximum threshold passes when value meets requirement."""
        check = ThresholdCheck(value=50, max_value=100)
        result = check()
        assert result == {"passed": True}

    def test_threshold_max_only_fail(self):
        """Test maximum threshold fails when value exceeds requirement."""
        check = ThresholdCheck(value=150, max_value=100)
        result = check()
        assert result == {"passed": False}

    def test_threshold_range_inside(self):
        """Test value within range passes."""
        check = ThresholdCheck(value=50, min_value=10, max_value=100)
        result = check()
        assert result == {"passed": True}

    def test_threshold_range_below_min(self):
        """Test value below minimum fails."""
        check = ThresholdCheck(value=5, min_value=10, max_value=100)
        result = check()
        assert result == {"passed": False}

    def test_threshold_range_above_max(self):
        """Test value above maximum fails."""
        check = ThresholdCheck(value=150, min_value=10, max_value=100)
        result = check()
        assert result == {"passed": False}

    def test_threshold_min_inclusive_true(self):
        """Test min_inclusive=True allows exact minimum value."""
        check = ThresholdCheck(value=10, min_value=10, min_inclusive=True)
        result = check()
        assert result == {"passed": True}

    def test_threshold_min_inclusive_false(self):
        """Test min_inclusive=False rejects exact minimum value."""
        check = ThresholdCheck(value=10, min_value=10, min_inclusive=False)
        result = check()
        assert result == {"passed": False}

    def test_threshold_max_inclusive_true(self):
        """Test max_inclusive=True allows exact maximum value."""
        check = ThresholdCheck(value=100, max_value=100, max_inclusive=True)
        result = check()
        assert result == {"passed": True}

    def test_threshold_max_inclusive_false(self):
        """Test max_inclusive=False rejects exact maximum value."""
        check = ThresholdCheck(value=100, max_value=100, max_inclusive=False)
        result = check()
        assert result == {"passed": False}

    def test_threshold_negate_outside_range(self):
        """Test negate=True passes when value is outside range."""
        check = ThresholdCheck(value=150, min_value=10, max_value=100, negate=True)
        result = check()
        assert result == {"passed": True}  # Outside range, negated

    def test_threshold_negate_inside_range(self):
        """Test negate=True fails when value is inside range."""
        check = ThresholdCheck(value=50, min_value=10, max_value=100, negate=True)
        result = check()
        assert result == {"passed": False}  # Inside range, negated

    def test_threshold_float_values(self):
        """Test threshold check with float values."""
        check = ThresholdCheck(value=3.14, min_value=3.0, max_value=4.0)
        result = check()
        assert result == {"passed": True}

    def test_threshold_string_numeric_conversion(self):
        """Test threshold check converts string numbers."""
        check = ThresholdCheck(value="42", min_value=30, max_value=50)
        result = check()
        assert result == {"passed": True}

    def test_threshold_non_numeric_string_error(self):
        """Test threshold check fails with non-numeric string at construction."""
        # Non-numeric strings are now caught at construction time
        with pytest.raises(PydanticValidationError):
            ThresholdCheck(value="not_a_number", min_value=0)


class TestThresholdEngineIntegration:
    """Test ThresholdCheck integration with the evaluation engine."""

    @pytest.mark.parametrize(("output_value", "min_val", "max_val", "expected_passed"), [
        ({"score": 0.75}, 0.8, 1.0, False),  # Below minimum
        ({"score": 0.85}, 0.8, 1.0, True),   # Within range
        ({"score": 0.95}, 0.8, 1.0, True),   # Within range
        ({"score": 1.05}, 0.8, 1.0, False),  # Above maximum
        ({"score": 42}, 0, 100, True),       # Integer within range
        ({"score": -5}, 0, 100, False),      # Below minimum
    ])
    def test_threshold_via_evaluate(
        self, output_value: dict[str, Any], min_val: float, max_val: float, expected_passed: bool,
    ):
        """Test using JSONPath for value with various threshold combinations."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test input",
                checks=[
                    Check(
                        type=CheckType.THRESHOLD,
                        arguments={
                            "value": "$.output.value.score",
                            "min_value": min_val,
                            "max_value": max_val,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value=output_value)]
        results = evaluate(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.summary.error_test_cases == 0
        assert results.summary.skipped_test_cases == 0
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": expected_passed}

    def test_threshold_check_instance_via_evaluate(self):
        """Test direct check instance usage in evaluate function."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test input",
                checks=[
                    ThresholdCheck(
                        value="$.output.value.confidence",
                        min_value=0.8,
                        max_value=1.0,
                    ),
                ],
            ),
        ]

        outputs = [Output(value={"confidence": 0.95})]
        results = evaluate(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_threshold_negate_via_evaluate(self):
        """Test negation through engine evaluation."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test input",
                checks=[
                    Check(
                        type=CheckType.THRESHOLD,
                        arguments={
                            "value": "$.output.value.score",
                            "min_value": 0.8,
                            "max_value": 1.0,
                            "negate": True,  # Should pass because score is outside range
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value={"score": 0.5})]  # Below minimum
        results = evaluate(test_cases, outputs)

        assert results.results[0].check_results[0].results == {"passed": True}

    def test_threshold_exclusivity_via_evaluate(self):
        """Test inclusive/exclusive bounds through engine evaluation."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test input",
                checks=[
                    Check(
                        type=CheckType.THRESHOLD,
                        arguments={
                            "value": "$.output.value.score",
                            "min_value": 0.8,
                            "max_value": 1.0,
                            "min_inclusive": False,  # Exclusive minimum
                            "max_inclusive": False,  # Exclusive maximum
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value={"score": 0.8})]  # Exactly at minimum
        results = evaluate(test_cases, outputs)

        # Should fail because 0.8 is not > 0.8 (exclusive minimum)
        assert results.results[0].check_results[0].results == {"passed": False}


class TestThresholdErrorHandling:
    """Test error handling and edge cases for ThresholdCheck."""

    def test_threshold_jsonpath_validation_in_engine(self):
        """Test that invalid JSONPath expressions are caught during evaluation."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.THRESHOLD,
                        arguments={
                            "value": "$..[invalid",  # Invalid JSONPath syntax
                            "min_value": 0,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="test")]

        # Should raise validation error for invalid JSONPath
        with pytest.raises(ValidationError, match="Invalid JSONPath expression"):
            evaluate(test_cases, outputs)

    def test_threshold_missing_jsonpath_data(self):
        """Test behavior when JSONPath doesn't find data."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.THRESHOLD,
                        arguments={
                            "value": "$.output.value.nonexistent",
                            "min_value": 0,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value={"response": "test"})]

        results = evaluate(test_cases, outputs)
        # Should result in error when JSONPath resolution fails
        assert results.results[0].status == Status.ERROR
        # Missing JSONPath data now causes errors rather than silent failures

    def test_threshold_non_numeric_value_error(self):
        """Test error when value cannot be converted to number."""
        # Non-numeric values are now caught at construction time by Pydantic
        with pytest.raises(PydanticValidationError):
            ThresholdCheck(value="invalid_number", min_value=0)

    def test_threshold_non_numeric_threshold_values(self):
        """Test validation of threshold values during construction."""
        # These should work fine - non-numeric values are caught during construction
        # by the field validators
        with pytest.raises((PydanticValidationError, ValueError)):
            ThresholdCheck(value=50, min_value="invalid")  # type: ignore

    def test_threshold_jsonpath_comprehensive(self):
        """Comprehensive JSONPath string conversion and execution test."""
        # 1. Create check with all JSONPath fields as strings
        check = ThresholdCheck(
            value="$.output.value.score",
            min_value="$.test_case.expected.min_threshold",
            max_value="$.test_case.expected.max_threshold",
            min_inclusive="$.test_case.expected.min_inclusive",
            max_inclusive="$.test_case.expected.max_inclusive",
            negate="$.test_case.expected.should_negate",
        )

        # 2. Verify conversion happened
        assert isinstance(check.value, JSONPath)
        assert check.value.expression == "$.output.value.score"
        assert isinstance(check.min_value, JSONPath)
        assert check.min_value.expression == "$.test_case.expected.min_threshold"
        assert isinstance(check.max_value, JSONPath)
        assert check.max_value.expression == "$.test_case.expected.max_threshold"
        assert isinstance(check.min_inclusive, JSONPath)
        assert check.min_inclusive.expression == "$.test_case.expected.min_inclusive"
        assert isinstance(check.max_inclusive, JSONPath)
        assert check.max_inclusive.expression == "$.test_case.expected.max_inclusive"
        assert isinstance(check.negate, JSONPath)
        assert check.negate.expression == "$.test_case.expected.should_negate"

        # 3. Test execution with EvaluationContext

        test_case = TestCase(
            id="test_001",
            input="test",
            expected={
                "min_threshold": 80,
                "max_threshold": 100,
                "min_inclusive": True,
                "max_inclusive": True,
                "should_negate": False,
            },
        )
        output = Output(value={"score": 85})
        context = EvaluationContext(test_case, output)

        result = check.execute(context)
        assert result.status == "completed"
        assert result.results["passed"] is True  # 85 is within [80, 100]
        assert result.resolved_arguments["value"]["value"] == 85
        assert result.resolved_arguments["min_value"]["value"] == 80
        assert result.resolved_arguments["max_value"]["value"] == 100
        assert result.resolved_arguments["min_inclusive"]["value"] is True
        assert result.resolved_arguments["max_inclusive"]["value"] is True
        assert result.resolved_arguments["negate"]["value"] is False

        # 4. Test invalid JSONPath string (should raise exception during validation)
        with pytest.raises(PydanticValidationError, match="Invalid JSONPath expression"):
            ThresholdCheck(value="$.invalid[", min_value=10)

        with pytest.raises(PydanticValidationError, match="Invalid JSONPath expression"):
            ThresholdCheck(value="$.valid.value", min_value="$.invalid[")

        # 5. Test non-JSONPath strings that are valid for their types (should remain unchanged)
        check_non_jsonpath = ThresholdCheck(
            value=42,  # Numeric literal
            min_value=10,  # Valid numeric value
            max_value=100,  # Valid numeric value
            min_inclusive=True,  # Valid boolean value
            max_inclusive=False,  # Valid boolean value
            negate=False,  # Valid boolean value
        )
        assert check_non_jsonpath.value == 42
        assert check_non_jsonpath.min_value == 10
        assert check_non_jsonpath.max_value == 100
        assert check_non_jsonpath.min_inclusive is True
        assert check_non_jsonpath.max_inclusive is False
        assert check_non_jsonpath.negate is False

        # 6. Test that invalid non-JSONPath strings are rejected for non-string fields
        # min_value: float | int | JSONPath | None - "not_a_number" is none of these
        with pytest.raises(PydanticValidationError):
            ThresholdCheck(value=42, min_value="not_a_number")

        # negate: bool | JSONPath - "not_a_boolean" is neither bool nor JSONPath
        with pytest.raises(PydanticValidationError):
            ThresholdCheck(value=42, min_value=10, negate="not_a_boolean")


class TestThresholdJSONPathIntegration:
    """Test ThresholdCheck with various JSONPath expressions and data structures."""

    def test_threshold_nested_jsonpath(self):
        """Test threshold with deeply nested JSONPath expressions."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.THRESHOLD,
                        arguments={
                            "value": "$.output.value.analysis.confidence_score",
                            "min_value": 0.7,
                            "max_value": 1.0,
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "analysis": {
                    "confidence_score": 0.85,
                    "method": "statistical",
                },
                "timestamp": "2024-01-01",
            }),
        ]

        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_threshold_array_access_jsonpath(self):
        """Test threshold with JSONPath array access."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.THRESHOLD,
                        arguments={
                            "value": "$.output.value.scores[0]",
                            "min_value": 80,
                            "max_value": 100,
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "scores": [95, 87, 92],
                "average": 91.3,
            }),
        ]

        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_threshold_dynamic_bounds_from_jsonpath(self):
        """Test threshold with bounds also coming from JSONPath."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected={
                    "min_threshold": 70,
                    "max_threshold": 100,
                },
                checks=[
                    Check(
                        type=CheckType.THRESHOLD,
                        arguments={
                            "value": "$.output.value.score",
                            "min_value": "$.test_case.expected.min_threshold",
                            "max_value": "$.test_case.expected.max_threshold",
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "score": 85,
                "category": "high",
            }),
        ]

        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}
