"""Comprehensive tests for ThresholdCheck implementation.

This module consolidates all tests for the ThresholdCheck including:
- Pydantic validation tests (from test_schema_check_classes.py)
- Implementation execution tests (from test_standard_checks.py) 
- Engine integration tests
- Edge cases and error handling

Tests are organized by functionality rather than implementation details.
"""

import pytest
from typing import Any

from flex_evals.checks.threshold import ThresholdCheck
from flex_evals import CheckType, Status, evaluate, Check, Output, TestCase
from flex_evals.exceptions import ValidationError
from pydantic import ValidationError as PydanticValidationError


class TestThresholdValidation:
    """Test Pydantic validation and field handling for ThresholdCheck."""

    def test_threshold_check_creation_min_only(self):
        """Test ThresholdCheck creation with min_value only."""
        check = ThresholdCheck(
            value="$.output.score",
            min_value=80.0,
        )

        assert check.value == "$.output.score"
        assert check.min_value == 80.0
        assert check.max_value is None
        assert check.min_inclusive is True
        assert check.max_inclusive is True
        assert check.negate is False

    def test_threshold_check_creation_max_only(self):
        """Test ThresholdCheck creation with max_value only."""
        check = ThresholdCheck(
            value="$.output.score",
            max_value=90.0,
        )

        assert check.max_value == 90.0
        assert check.min_value is None

    def test_threshold_check_creation_both_bounds(self):
        """Test ThresholdCheck creation with both bounds."""
        check = ThresholdCheck(
            value="$.output.score",
            min_value=80.0,
            max_value=90.0,
            min_inclusive=False,
            max_inclusive=False,
            negate=True,
        )

        assert check.min_value == 80.0
        assert check.max_value == 90.0
        assert check.min_inclusive is False
        assert check.max_inclusive is False
        assert check.negate is True

    def test_threshold_check_validation_empty_value(self):
        """Test ThresholdCheck allows empty value as valid literal value."""
        check = ThresholdCheck(value="", min_value=80.0)
        assert check.value == ""
        assert check.min_value == 80.0

    def test_threshold_check_validation_no_bounds(self):
        """Test ThresholdCheck validation for no bounds."""
        with pytest.raises(PydanticValidationError, match="at least one of 'min_value' or 'max_value'"):
            ThresholdCheck(value="$.value")

    def test_threshold_check_type_property(self):
        """Test ThresholdCheck check_type property returns correct type."""
        check = ThresholdCheck(value="test", min_value=0)
        assert check.check_type == CheckType.THRESHOLD

    def test_threshold_check_jsonpath_min_value(self):
        """Test ThresholdCheck with JSONPath min_value."""
        check = ThresholdCheck(
            value="$.output.score",
            min_value="$.thresholds.min",
        )

        assert check.value == "$.output.score"
        assert check.min_value == "$.thresholds.min"
        assert check.max_value is None

    def test_threshold_check_mixed_literal_and_jsonpath(self):
        """Test ThresholdCheck with mixed literal and JSONPath values."""
        check = ThresholdCheck(
            value="$.output.score",
            min_value="$.thresholds.min",
            max_value=95.0,
            min_inclusive=False,
            max_inclusive="$.flags.max_inclusive",
        )

        assert check.min_value == "$.thresholds.min"
        assert check.max_value == 95.0
        assert check.min_inclusive is False
        assert check.max_inclusive == "$.flags.max_inclusive"


class TestThresholdExecution:
    """Test ThresholdCheck execution logic and __call__ method."""

    def test_threshold_min_only_pass(self):
        """Test value >= min_value passes."""
        check = ThresholdCheck(value="test", min_value=0.8)
        result = check(value=0.85, min_value=0.8)
        assert result == {"passed": True}

    def test_threshold_min_only_fail(self):
        """Test value < min_value fails."""
        check = ThresholdCheck(value="test", min_value=0.8)
        result = check(value=0.75, min_value=0.8)
        assert result == {"passed": False}

    def test_threshold_max_only_pass(self):
        """Test value <= max_value passes."""
        check = ThresholdCheck(value="test", max_value=1.0)
        result = check(value=0.85, max_value=1.0)
        assert result == {"passed": True}

    def test_threshold_max_only_fail(self):
        """Test value > max_value fails."""
        check = ThresholdCheck(value="test", max_value=1.0)
        result = check(value=1.2, max_value=1.0)
        assert result == {"passed": False}

    def test_threshold_range_inside(self):
        """Test value within min and max passes."""
        check = ThresholdCheck(value="test", min_value=0.8, max_value=1.0)
        result = check(value=0.85, min_value=0.8, max_value=1.0)
        assert result == {"passed": True}

    def test_threshold_range_outside(self):
        """Test value outside range fails."""
        check = ThresholdCheck(value="test", min_value=0.8, max_value=1.0)
        result = check(value=1.2, min_value=0.8, max_value=1.0)
        assert result == {"passed": False}

    def test_threshold_min_exclusive(self):
        """Test min_inclusive=false excludes boundary."""
        check = ThresholdCheck(value="test", min_value=0.8)
        result = check(value=0.8, min_value=0.8, min_inclusive=False)
        assert result == {"passed": False}

        # But greater than boundary should pass
        result = check(value=0.81, min_value=0.8, min_inclusive=False)
        assert result == {"passed": True}

    def test_threshold_max_exclusive(self):
        """Test max_inclusive=false excludes boundary."""
        check = ThresholdCheck(value="test", max_value=1.0)
        result = check(value=1.0, max_value=1.0, max_inclusive=False)
        assert result == {"passed": False}

        # But less than boundary should pass
        result = check(value=0.99, max_value=1.0, max_inclusive=False)
        assert result == {"passed": True}

    def test_threshold_negate_outside(self):
        """Test negate=true passes when outside bounds."""
        check = ThresholdCheck(value="test", min_value=0.8, max_value=1.0)
        result = check(value=1.2, min_value=0.8, max_value=1.0, negate=True)
        assert result == {"passed": True}

    def test_threshold_negate_inside(self):
        """Test negate=true fails when inside bounds."""
        check = ThresholdCheck(value="test", min_value=0.8, max_value=1.0)
        result = check(value=0.9, min_value=0.8, max_value=1.0, negate=True)
        assert result == {"passed": False}

    def test_threshold_no_bounds_error(self):
        """Test error when neither min nor max specified."""
        check = ThresholdCheck(value="test", min_value=0.8)
        with pytest.raises(ValidationError, match="requires at least one of"):
            check(value=0.85)

    def test_threshold_non_numeric_error(self):
        """Test error when value is not numeric."""
        check = ThresholdCheck(value="test", min_value=0.8)
        with pytest.raises(ValidationError, match="must be numeric"):
            check(value="not a number", min_value=0.8)

    def test_threshold_string_numeric_conversion(self):
        """Test numeric string conversion."""
        check = ThresholdCheck(value="test", min_value=0.8)
        result = check(value="0.85", min_value=0.8)
        assert result == {"passed": True}


class TestThresholdEngineIntegration:
    """Test ThresholdCheck integration with the evaluation engine."""

    @pytest.mark.parametrize(("confidence_value", "min_val", "max_val", "expected_passed"), [
        (0.95, 0.8, 1.0, True),  # Value within range
        (0.75, 0.8, 1.0, False),  # Value below minimum
        (1.1, 0.8, 1.0, False),  # Value above maximum
        (0.8, 0.8, 1.0, True),  # Value equals minimum (inclusive by default)
        (1.0, 0.8, 1.0, True),  # Value equals maximum (inclusive by default)
    ])
    def test_threshold_via_evaluate(
        self, confidence_value: float, min_val: float, max_val: float, expected_passed: bool,
    ):
        """Test using JSONPath for value and thresholds with various combinations."""
        test_cases = [
            TestCase(
                id="test_001",
                input="What is the confidence score?",
                expected={"min": min_val, "max": max_val},
                checks=[
                    Check(
                        type=CheckType.THRESHOLD,
                        arguments={
                            "value": "$.output.value.confidence",
                            "min_value": "$.test_case.expected.min",
                            "max_value": "$.test_case.expected.max",
                        },
                    ),
                ],
            ),
        ]
        
        outputs = [Output(value={"message": "High confidence", "confidence": confidence_value})]
        results = evaluate(test_cases, outputs)
        
        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": expected_passed}

    def test_threshold_check_instance_via_evaluate(self):
        """Test direct check instance usage in evaluate function."""
        test_cases = [
            TestCase(
                id="test_001",
                input="What is the score?",
                checks=[
                    ThresholdCheck(
                        value="$.output.value",
                        min_value=80.0,
                        max_value=100.0,
                    ),
                ],
            ),
        ]
        
        outputs = [Output(value=85.0)]
        results = evaluate(test_cases, outputs)
        
        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": True}


class TestThresholdErrorHandling:
    """Test error handling and edge cases for ThresholdCheck."""

    def test_threshold_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(PydanticValidationError):
            ThresholdCheck()  # type: ignore
        
        with pytest.raises(PydanticValidationError):
            ThresholdCheck(value="test")  # type: ignore - missing threshold
        
    def test_threshold_invalid_numeric_values(self):
        """Test validation of numeric threshold values."""
        with pytest.raises(PydanticValidationError):
            ThresholdCheck(value="test", min_value="not_numeric")  # type: ignore

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
                            "min_value": 0.8,
                        },
                    ),
                ],
            ),
        ]
        
        outputs = [Output(value="test")]
        
        with pytest.raises(ValidationError, match="appears to be JSONPath but is invalid"):
            evaluate(test_cases, outputs)