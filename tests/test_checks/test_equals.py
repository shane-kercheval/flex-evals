"""Comprehensive tests for EqualsCheck implementation.

This module consolidates all tests for the EqualsCheck including:
- Pydantic validation tests (from test_schema_check_classes.py)
- Implementation execution tests (from test_standard_checks.py) 
- Engine integration tests
- Edge cases and error handling

Tests are organized by functionality rather than implementation details.
"""

import pytest
from typing import Any

from flex_evals.checks.equals import EqualsCheck
from flex_evals import CheckType, Status, evaluate, Check, Output, TestCase
from flex_evals.exceptions import ValidationError
from pydantic import ValidationError as PydanticValidationError


class TestEqualsValidation:
    """Test Pydantic validation and field handling for EqualsCheck."""

    def test_equals_check_creation(self):
        """Test basic EqualsCheck creation."""
        check = EqualsCheck(
            actual="$.output.value",
            expected="$.expected",
        )

        assert check.actual == "$.output.value"
        assert check.expected == "$.expected"
        assert check.negate is False

    def test_equals_check_with_options(self):
        """Test EqualsCheck with all options."""
        check = EqualsCheck(
            actual="$.output.value",
            expected="$.expected",
            negate=True,
        )

        assert check.negate is True

    def test_equals_check_validation_empty_actual(self):
        """Test EqualsCheck allows empty actual as valid literal value."""
        # Empty strings should be allowed as literal values
        check = EqualsCheck(actual="", expected="$.expected")
        assert check.actual == ""
        assert check.expected == "$.expected"

    def test_equals_check_validation_empty_expected(self):
        """Test EqualsCheck allows empty expected as valid literal value."""
        # Empty strings should be allowed as literal values
        check = EqualsCheck(actual="$.actual", expected="")
        assert check.actual == "$.actual"
        assert check.expected == ""

    def test_equals_check_type_property(self):
        """Test EqualsCheck check_type property returns correct type."""
        check = EqualsCheck(actual="test", expected="test")
        assert check.check_type == CheckType.EQUALS

    def test_equals_check_version_property(self):
        """Test EqualsCheck version property works correctly."""
        check = EqualsCheck(actual="test", expected="test")
        assert check._get_version() == "1.0.0"


class TestEqualsExecution:
    """Test EqualsCheck execution logic and __call__ method."""

    def test_equals_identical_strings(self):
        """Test equals passes when strings are identical."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual="hello world",
            expected="hello world",
            negate=False,
        )
        assert result == {"passed": True}

    def test_equals_different_strings(self):
        """Test equals fails when strings are different."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual="hello world",
            expected="goodbye world",
            negate=False,
        )
        assert result == {"passed": False}

    def test_equals_case_sensitive_by_default(self):
        """Test equals is case sensitive by default."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual="Hello World",
            expected="hello world",
            negate=False,
        )
        assert result == {"passed": False}

    def test_equals_integers(self):
        """Test equals works with identical integers."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual=42,
            expected=42,
            negate=False,
        )
        assert result == {"passed": True}

    def test_equals_different_integers(self):
        """Test equals fails with different integers."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual=42,
            expected=43,
            negate=False,
        )
        assert result == {"passed": False}

    def test_equals_floats(self):
        """Test equals works with identical floats."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual=3.14,
            expected=3.14,
            negate=False,
        )
        assert result == {"passed": True}

    def test_equals_different_floats(self):
        """Test equals fails with different floats."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual=3.14,
            expected=2.71,
            negate=False,
        )
        assert result == {"passed": False}

    def test_equals_booleans(self):
        """Test equals works with identical booleans."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual=True,
            expected=True,
            negate=False,
        )
        assert result == {"passed": True}

    def test_equals_different_booleans(self):
        """Test equals fails with different booleans."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual=True,
            expected=False,
            negate=False,
        )
        assert result == {"passed": False}

    def test_equals_lists_identical(self):
        """Test equals works with identical lists."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual=[1, 2, 3],
            expected=[1, 2, 3],
            negate=False,
        )
        assert result == {"passed": True}

    def test_equals_lists_different_values(self):
        """Test equals fails with lists containing different values."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual=[1, 2, 3],
            expected=[1, 2, 4],
            negate=False,
        )
        assert result == {"passed": False}

    def test_equals_lists_different_order(self):
        """Test equals fails with same values in different order."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual=[1, 2, 3],
            expected=[3, 2, 1],
            negate=False,
        )
        assert result == {"passed": False}

    def test_equals_lists_different_length(self):
        """Test equals fails with different length lists."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual=[1, 2, 3],
            expected=[1, 2],
            negate=False,
        )
        assert result == {"passed": False}

    def test_equals_dicts_identical(self):
        """Test equals works with identical dictionaries."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual={"a": 1, "b": 2},
            expected={"a": 1, "b": 2},
            negate=False,
        )
        assert result == {"passed": True}

    def test_equals_dicts_different_values(self):
        """Test equals fails with dicts having different values."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual={"a": 1, "b": 2},
            expected={"a": 1, "b": 3},
            negate=False,
        )
        assert result == {"passed": False}

    def test_equals_dicts_different_keys(self):
        """Test equals fails with dicts having different keys."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual={"a": 1, "b": 2},
            expected={"a": 1, "c": 2},
            negate=False,
        )
        assert result == {"passed": False}

    def test_equals_dicts_different_order_same_content(self):
        """Test equals passes with same dict content in different order."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual={"a": 1, "b": 2},
            expected={"b": 2, "a": 1},
            negate=False,
        )
        assert result == {"passed": True}

    def test_equals_sets_identical(self):
        """Test equals works with identical sets."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual={1, 2, 3},
            expected={1, 2, 3},
            negate=False,
        )
        assert result == {"passed": True}

    def test_equals_sets_different_values(self):
        """Test equals fails with sets having different values."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual={1, 2, 3},
            expected={1, 2, 4},
            negate=False,
        )
        assert result == {"passed": False}

    def test_equals_sets_different_order_same_content(self):
        """Test equals passes with same set content in different order."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual={1, 2, 3},
            expected={3, 1, 2},
            negate=False,
        )
        assert result == {"passed": True}

    def test_equals_tuples_identical(self):
        """Test equals works with identical tuples."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual=(1, 2, 3),
            expected=(1, 2, 3),
            negate=False,
        )
        assert result == {"passed": True}

    def test_equals_tuples_different_order(self):
        """Test equals fails with tuples in different order."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual=(1, 2, 3),
            expected=(3, 2, 1),
            negate=False,
        )
        assert result == {"passed": False}

    def test_equals_none_values(self):
        """Test equals works with None values."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual=None,
            expected=None,
            negate=False,
        )
        assert result == {"passed": True}

    def test_equals_none_vs_empty_string(self):
        """Test equals fails when comparing None to empty string."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual=None,
            expected="",
            negate=False,
        )
        assert result == {"passed": False}

    def test_equals_mixed_types_different(self):
        """Test equals fails when comparing different types."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual="42",
            expected=42,
            negate=False,
        )
        assert result == {"passed": False}

    def test_equals_nested_structures(self):
        """Test equals works with nested data structures."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual={"list": [1, 2, {"nested": "value"}], "number": 42},
            expected={"list": [1, 2, {"nested": "value"}], "number": 42},
            negate=False,
        )
        assert result == {"passed": True}

    def test_equals_nested_structures_different(self):
        """Test equals fails with different nested structures."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual={"list": [1, 2, {"nested": "value"}], "number": 42},
            expected={"list": [1, 2, {"nested": "different"}], "number": 42},
            negate=False,
        )
        assert result == {"passed": False}

    def test_equals_negate_true_equal_values(self):
        """Test negate=true fails when values are equal."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual="hello",
            expected="hello",
            negate=True,
        )
        assert result == {"passed": False}

    def test_equals_negate_true_different_values(self):
        """Test negate=true passes when values are different."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual="hello",
            expected="world",
            negate=True,
        )
        assert result == {"passed": True}

    def test_equals_negate_false_equal_values(self):
        """Test negate=false passes when values are equal."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual="hello",
            expected="hello",
            negate=False,
        )
        assert result == {"passed": True}

    def test_equals_negate_false_different_values(self):
        """Test negate=false fails when values are different."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual="hello",
            expected="world",
            negate=False,
        )
        assert result == {"passed": False}

    def test_equals_missing_actual_argument(self):
        """Test equals raises error when actual argument is missing."""
        check = EqualsCheck(actual="test", expected="test")
        with pytest.raises(TypeError):
            check(expected="value", negate=False)

    def test_equals_missing_expected_argument(self):
        """Test equals raises error when expected argument is missing."""
        check = EqualsCheck(actual="test", expected="test")
        with pytest.raises(TypeError):
            check(actual="value", negate=False)

    def test_equals_result_schema(self):
        """Test equals returns correct result schema."""
        check = EqualsCheck(actual="test", expected="test")
        result = check(
            actual="test",
            expected="test",
            negate=False,
        )

        assert isinstance(result, dict)
        assert "passed" in result
        assert isinstance(result["passed"], bool)
        assert len(result) == 1


class TestEqualsEngineIntegration:
    """Test EqualsCheck integration with the evaluation engine."""

    @pytest.mark.parametrize(("actual_value", "expected_value", "expected_passed"), [
        # Same type comparisons
        ("hello", "hello", True),
        ("hello", "world", False),
        (42, 42, True),
        (42, 43, False),
        ([1, 2, 3], [1, 2, 3], True),
        ([1, 2, 3], [1, 2, 4], False),
        ({"a": 1}, {"a": 1}, True),
        ({"a": 1}, {"a": 2}, False),

        # Cross-type comparisons (should be False)
        ("42", 42, False),
        (1, 1.0, True),  # Python considers 1 == 1.0 to be True
        ([1, 2], (1, 2), False),
        ({"a": 1}, [("a", 1)], False),

        # None comparisons
        (None, None, True),
        (None, "", False),
        (None, 0, False),
        (None, [], False),
    ])
    def test_equals_via_evaluate(
        self, actual_value: Any, expected_value: Any, expected_passed: bool,
    ):
        """Test using JSONPath for values with various equal/non-equal combinations."""
        test_cases = [
            TestCase(
                id="test_001",
                input="Input data",
                expected="Expected output",
                checks=[
                    Check(
                        type=CheckType.EQUALS,
                        arguments={
                            "actual": "$.output.value.actual_value",
                            "expected": "$.output.value.expected_value",
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={"actual_value": actual_value, "expected_value": expected_value}),
        ]

        results = evaluate(test_cases, outputs)
        assert len(results.results) == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": expected_passed}

    def test_equals_check_instance_via_evaluate(self):
        """Test direct check instance usage in evaluate function."""
        test_cases = [
            TestCase(
                id="test_001",
                input="Input data",
                expected="hello",
                checks=[
                    EqualsCheck(
                        actual="$.output.value",
                        expected="$.test_case.expected",
                    ),
                ],
            ),
        ]
        
        outputs = [Output(value="hello")]
        results = evaluate(test_cases, outputs)
        
        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_equals_negate_via_evaluate(self):
        """Test negation through engine evaluation."""
        test_cases = [
            TestCase(
                id="test_001",
                input="Input data",
                expected="world",
                checks=[
                    Check(
                        type=CheckType.EQUALS,
                        arguments={
                            "actual": "$.output.value",
                            "expected": "$.test_case.expected",
                            "negate": True,
                        },
                    ),
                ],
            ),
        ]
        
        outputs = [Output(value="hello")]
        results = evaluate(test_cases, outputs)
        
        assert results.results[0].check_results[0].results == {"passed": True}


class TestEqualsErrorHandling:
    """Test error handling and edge cases for EqualsCheck."""

    def test_equals_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(PydanticValidationError):
            EqualsCheck()  # type: ignore
        
        with pytest.raises(PydanticValidationError):
            EqualsCheck(actual="test")  # type: ignore
        
        with pytest.raises(PydanticValidationError):
            EqualsCheck(expected="test")  # type: ignore

    def test_equals_jsonpath_validation_in_engine(self):
        """Test that invalid JSONPath expressions are caught during evaluation."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.EQUALS,
                        arguments={
                            "actual": "$..[invalid",  # Invalid JSONPath syntax
                            "expected": "test",
                        },
                    ),
                ],
            ),
        ]
        
        outputs = [Output(value="test")]
        
        # Should raise validation error for invalid JSONPath
        with pytest.raises(ValidationError, match="appears to be JSONPath but is invalid"):
            evaluate(test_cases, outputs)


class TestEqualsJSONPathIntegration:
    """Test EqualsCheck with various JSONPath expressions and data structures."""

    def test_equals_nested_jsonpath(self):
        """Test equals with deeply nested JSONPath expressions."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected={"location": {"city": "Paris", "country": "France"}},
                checks=[
                    Check(
                        type=CheckType.EQUALS,
                        arguments={
                            "actual": "$.output.value.user.location.city",
                            "expected": "$.test_case.expected.location.city",
                        },
                    ),
                ],
            ),
        ]
        
        outputs = [
            Output(value={
                "user": {
                    "name": "Alice",
                    "location": {"city": "Paris", "country": "France"},
                },
            }),
        ]
        
        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_equals_array_access_jsonpath(self):
        """Test equals with JSONPath array access."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected={"scores": [95, 87, 92]},
                checks=[
                    Check(
                        type=CheckType.EQUALS,
                        arguments={
                            "actual": "$.output.value.results[0]",
                            "expected": "$.test_case.expected.scores[0]",
                        },
                    ),
                ],
            ),
        ]
        
        outputs = [
            Output(value={
                "results": [95, 87, 92],
                "average": 91.33,
            }),
        ]
        
        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_equals_complex_data_structures(self):
        """Test equals with complex nested data structures."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected={
                    "response": {
                        "data": {
                            "items": [
                                {"id": 1, "name": "item1"},
                                {"id": 2, "name": "item2"},
                            ],
                        },
                    },
                },
                checks=[
                    Check(
                        type=CheckType.EQUALS,
                        arguments={
                            "actual": "$.output.value.response.data.items",
                            "expected": "$.test_case.expected.response.data.items",
                        },
                    ),
                ],
            ),
        ]
        
        outputs = [
            Output(value={
                "response": {
                    "status": "success",
                    "data": {
                        "total": 2,
                        "items": [
                            {"id": 1, "name": "item1"},
                            {"id": 2, "name": "item2"},
                        ],
                    },
                },
            }),
        ]
        
        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}