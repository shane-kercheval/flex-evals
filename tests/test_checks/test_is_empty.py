"""Comprehensive tests for IsEmptyCheck implementation.

This module consolidates all tests for the IsEmptyCheck including:
- Pydantic validation tests 
- Implementation execution tests (from test_standard_checks.py) 
- Engine integration tests
- Edge cases and error handling

Tests are organized by functionality rather than implementation details.
"""

import pytest
from typing import Any

from flex_evals.checks.is_empty import IsEmptyCheck
from flex_evals import CheckType, Status, evaluate, Check, Output, TestCase
from flex_evals.exceptions import ValidationError
from pydantic import ValidationError as PydanticValidationError


class TestIsEmptyValidation:
    """Test Pydantic validation and field handling for IsEmptyCheck."""

    def test_is_empty_check_creation(self):
        """Test basic IsEmptyCheck creation."""
        check = IsEmptyCheck(
            value="$.output.value",
        )

        assert check.value == "$.output.value"
        assert check.negate is False
        assert check.strip_whitespace is True

    def test_is_empty_check_with_options(self):
        """Test IsEmptyCheck with all options."""
        check = IsEmptyCheck(
            value="$.output.value",
            negate=True,
            strip_whitespace=False,
        )

        assert check.negate is True
        assert check.strip_whitespace is False

    def test_is_empty_check_type_property(self):
        """Test IsEmptyCheck check_type property returns correct type."""
        check = IsEmptyCheck(value="test")
        assert check.check_type == CheckType.IS_EMPTY


class TestIsEmptyExecution:
    """Test IsEmptyCheck execution logic and __call__ method."""

    @pytest.mark.parametrize(("value", "strip_whitespace", "expected_passed", "description"), [
        # Empty values (should pass)
        ("", True, True, "empty string"),
        (None, True, True, "None value"),
        ("   \t\n  ", True, True, "whitespace-only with strip=True"),
        ([], True, True, "empty list"),
        ({}, True, True, "empty dict"),
        (set(), True, True, "empty set"),
        ((), True, True, "empty tuple"),
        (frozenset(), True, True, "empty frozenset"),

        # Non-empty values (should fail)
        ("hello", True, False, "non-empty string"),
        ("  hello  ", True, False, "string with content and whitespace"),
        ("   \t\n  ", False, False, "whitespace-only with strip=False"),
        (123, True, False, "positive integer"),
        (0, True, False, "zero"),
        (-1, True, False, "negative integer"),
        (0.0, True, False, "float zero"),
        (False, True, False, "boolean False"),
        (True, True, False, "boolean True"),
        ([1, 2, 3], True, False, "list with items"),
        ({"key": "value"}, True, False, "dict with items"),
        ({1, 2, 3}, True, False, "set with items"),
        ((1, 2, 3), True, False, "tuple with items"),
        (frozenset([1, 2, 3]), True, False, "frozenset with items"),
        ({"nested": {"list": [1, 2, {"key": None}]}}, True, False, "complex data structure"),
    ])
    def test_is_empty_values(self, value: Any, strip_whitespace: bool, expected_passed: bool, description: str):
        """Test various values for emptiness."""
        check = IsEmptyCheck(value="test")
        result = check(value=value, strip_whitespace=strip_whitespace)
        assert result == {"passed": expected_passed}, f"Failed for {description}"

    @pytest.mark.parametrize(("value", "expected_passed", "description"), [
        # Empty values (should fail when negated)
        ("", False, "empty string"),
        (None, False, "None value"),
        ("   \t\n  ", False, "whitespace-only string"),
        ([], False, "empty list"),
        ({}, False, "empty dict"),
        (set(), False, "empty set"),
        ((), False, "empty tuple"),

        # Non-empty values (should pass when negated)
        ("hello", True, "non-empty string"),
        (123, True, "numeric value"),
        (False, True, "False value"),
        ([1, 2, 3], True, "non-empty list"),
        ({"key": "value"}, True, "non-empty dict"),
        ({1, 2, 3}, True, "non-empty set"),
        ((1, 2, 3), True, "non-empty tuple"),
    ])
    def test_is_empty_negate(self, value: Any, expected_passed: bool, description: str):
        """Test negate=True behavior (not empty check)."""
        check = IsEmptyCheck(value="test")
        result = check(value=value, negate=True)
        assert result == {"passed": expected_passed}, f"Failed negate test for {description}"

    def test_is_empty_missing_value(self):
        """Test missing value argument raises TypeError."""
        check = IsEmptyCheck(value="test")
        with pytest.raises(TypeError):
            check()

    def test_is_empty_result_schema(self):
        r"""Test result matches {\"passed\": boolean} exactly."""
        check = IsEmptyCheck(value="test")
        result = check(value="")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"passed"}
        assert isinstance(result["passed"], bool)


class TestIsEmptyEngineIntegration:
    """Test IsEmptyCheck integration with the evaluation engine."""

    @pytest.mark.parametrize(("output_value", "jsonpath", "expected_passed"), [
        ("", "$.output.value", True),
        ("not empty", "$.output.value", False),
        ("   ", "$.output.value", True),
        ("  text  ", "$.output.value", False),
        ({"data": []}, "$.output.value.data", True),
        ({"data": [1, 2, 3]}, "$.output.value.data", False),
        ({"data": {}}, "$.output.value.data", True),
        ({"data": {"key": "value"}}, "$.output.value.data", False),
    ])
    def test_is_empty_via_evaluate(self, output_value: Any, jsonpath: str, expected_passed: bool):
        """Test using JSONPath for value with various empty/non-empty combinations."""
        test_cases = [
            TestCase(
                id="test_001",
                input="What is your name?",
                expected="",  # Empty expected value
                checks=[
                    Check(
                        type=CheckType.IS_EMPTY,
                        arguments={
                            "value": jsonpath,
                        },
                    ),
                ],
            ),
        ]
        
        outputs = [Output(value=output_value)]
        results = evaluate(test_cases, outputs)
        
        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": expected_passed}

    @pytest.mark.parametrize(("output_value", "expected_passed"), [
        (None, True),
        (1, False),
    ])
    def test_none_empty_via_evaluate(self, output_value: Any, expected_passed: bool):
        """Test using JSONPath for value with various empty/non-empty combinations."""
        test_cases = [
            TestCase(
                id="test_001",
                input="What is your name?",
                expected="",  # Empty expected value
                checks=[
                    Check(
                        type=CheckType.IS_EMPTY,
                        arguments={
                            "value": "$.output.value.value",
                        },
                    ),
                ],
            ),
        ]
        
        outputs = [Output(value={"value": output_value})]
        results = evaluate(test_cases, outputs)
        
        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": expected_passed}

    def test_is_empty_check_instance_via_evaluate(self):
        """Test direct check instance usage in evaluate function."""
        test_cases = [
            TestCase(
                id="test_001",
                input="What is your name?",
                checks=[
                    IsEmptyCheck(value="$.output.value"),
                ],
            ),
        ]
        
        outputs = [Output(value="")]
        results = evaluate(test_cases, outputs)
        
        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": True}


class TestIsEmptyErrorHandling:
    """Test error handling and edge cases for IsEmptyCheck."""

    def test_is_empty_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(PydanticValidationError):
            IsEmptyCheck()  # type: ignore

    def test_is_empty_jsonpath_validation_in_engine(self):
        """Test that invalid JSONPath expressions are caught during evaluation."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.IS_EMPTY,
                        arguments={
                            "value": "$..[invalid",  # Invalid JSONPath syntax
                        },
                    ),
                ],
            ),
        ]
        
        outputs = [Output(value="test")]
        
        with pytest.raises(ValidationError, match="appears to be JSONPath but is invalid"):
            evaluate(test_cases, outputs)