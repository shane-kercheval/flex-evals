"""
Comprehensive tests for EqualsCheck implementation.

This module consolidates all tests for the EqualsCheck including:
- Pydantic validation tests
- Implementation execution tests
- Engine integration tests
- Edge cases and error handling

Tests are organized by functionality rather than implementation details.
"""

import pytest
from typing import Any

from flex_evals import (
    EqualsCheck,
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


class TestEqualsValidation:
    """Test Pydantic validation and field handling for EqualsCheck."""

    def test_equals_check_creation(self):
        """Test basic EqualsCheck creation."""
        check = EqualsCheck(
            actual="$.output.value",
            expected="$.expected",
        )

        assert isinstance(check.actual, JSONPath)
        assert check.actual.expression == "$.output.value"
        assert isinstance(check.expected, JSONPath)
        assert check.expected.expression == "$.expected"
        assert check.negate is False

    def test_equals_check_with_options(self):
        """Test EqualsCheck with all options."""
        check = EqualsCheck(
            actual="$.output.value",
            expected="$.expected",
            negate=True,
        )

        assert check.negate is True

    def test_equals_jsonpath_comprehensive(self):
        """Comprehensive JSONPath string conversion and execution test."""
        # 1. Create check with all JSONPath fields as strings
        check = EqualsCheck(
            actual="$.output.value.result",
            expected="$.test_case.expected.target_value",
            negate="$.test_case.expected.should_negate",
        )

        # 2. Verify conversion happened
        assert isinstance(check.actual, JSONPath)
        assert check.actual.expression == "$.output.value.result"
        assert isinstance(check.expected, JSONPath)
        assert check.expected.expression == "$.test_case.expected.target_value"
        assert isinstance(check.negate, JSONPath)
        assert check.negate.expression == "$.test_case.expected.should_negate"

        # 3. Test execution with EvaluationContext

        test_case = TestCase(
            id="test_001",
            input="test",
            expected={
                "target_value": {"status": "success", "count": 42},
                "should_negate": False,
            },
        )
        output = Output(value={"result": {"status": "success", "count": 42}})
        context = EvaluationContext(test_case, output)

        result = check.execute(context)
        assert result.status == "completed"
        assert result.results["passed"] is True  # Objects are equal, no negate
        assert result.resolved_arguments["actual"]["value"] == {"status": "success", "count": 42}
        assert result.resolved_arguments["expected"]["value"] == {"status": "success", "count": 42}
        assert result.resolved_arguments["negate"]["value"] is False

        # 4. Test invalid JSONPath string (should raise exception during validation)
        with pytest.raises(PydanticValidationError, match="Invalid JSONPath expression"):
            EqualsCheck(actual="$.invalid[", expected="valid_value")

        with pytest.raises(PydanticValidationError, match="Invalid JSONPath expression"):
            EqualsCheck(actual="$.valid.actual", expected="$.invalid[")

        # 5. Test valid literal values (should work)
        check_literal = EqualsCheck(
            actual="literal_actual_value",  # Any | JSONPath - any value works
            expected="literal_expected_value",  # Any | JSONPath - any value works
            negate=False,  # bool | JSONPath - boolean literal works
        )
        assert check_literal.actual == "literal_actual_value"
        assert check_literal.expected == "literal_expected_value"
        assert check_literal.negate is False

        # 6. Test invalid non-JSONPath strings for fields that don't support string
        # negate: bool | JSONPath - "not_a_boolean" is neither bool nor JSONPath
        with pytest.raises(PydanticValidationError):
            EqualsCheck(actual="test", expected="test", negate="not_a_boolean")

    def test_equals_check_with_literals(self):
        """Test EqualsCheck with literal values."""
        check = EqualsCheck(
            actual="test_value",
            expected=42,
            negate=False,
        )

        assert check.actual == "test_value"
        assert check.expected == 42
        assert check.negate is False

    def test_equals_check_type_property(self):
        """Test EqualsCheck check_type property returns correct type."""
        check = EqualsCheck(actual="test", expected="test")
        assert check.check_type == CheckType.EQUALS

    def test_equals_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(PydanticValidationError):
            EqualsCheck()  # type: ignore

        with pytest.raises(PydanticValidationError):
            EqualsCheck(actual="test")  # type: ignore

        with pytest.raises(PydanticValidationError):
            EqualsCheck(expected="test")  # type: ignore


class TestEqualsExecution:
    """Test EqualsCheck execution logic and __call__ method."""

    def test_equals_identical_strings(self):
        """Test identical strings return passed=true."""
        check = EqualsCheck(actual="hello", expected="hello")
        result = check()
        assert result == {"passed": True}

    def test_equals_different_strings(self):
        """Test different strings return passed=false."""
        check = EqualsCheck(actual="hello", expected="world")
        result = check()
        assert result == {"passed": False}

    def test_equals_integers(self):
        """Test identical integers."""
        check = EqualsCheck(actual=42, expected=42)
        result = check()
        assert result == {"passed": True}

    def test_equals_different_integers(self):
        """Test different integers."""
        check = EqualsCheck(actual=42, expected=24)
        result = check()
        assert result == {"passed": False}

    def test_equals_floats(self):
        """Test identical floats."""
        check = EqualsCheck(actual=3.14, expected=3.14)
        result = check()
        assert result == {"passed": True}

    def test_equals_different_floats(self):
        """Test different floats."""
        check = EqualsCheck(actual=3.14, expected=2.71)
        result = check()
        assert result == {"passed": False}

    def test_equals_booleans(self):
        """Test identical booleans."""
        check = EqualsCheck(actual=True, expected=True)
        result = check()
        assert result == {"passed": True}

    def test_equals_different_booleans(self):
        """Test different booleans."""
        check = EqualsCheck(actual=True, expected=False)
        result = check()
        assert result == {"passed": False}

    def test_equals_lists_identical(self):
        """Test identical lists."""
        check = EqualsCheck(actual=[1, 2, 3], expected=[1, 2, 3])
        result = check()
        assert result == {"passed": True}

    def test_equals_lists_different_values(self):
        """Test lists with different values."""
        check = EqualsCheck(actual=[1, 2, 3], expected=[1, 2, 4])
        result = check()
        assert result == {"passed": False}

    def test_equals_lists_different_order(self):
        """Test lists with different order."""
        check = EqualsCheck(actual=[1, 2, 3], expected=[3, 2, 1])
        result = check()
        assert result == {"passed": False}

    def test_equals_dicts_identical(self):
        """Test identical dictionaries."""
        check = EqualsCheck(
            actual={"name": "Alice", "age": 30},
            expected={"name": "Alice", "age": 30},
        )
        result = check()
        assert result == {"passed": True}

    def test_equals_dicts_different_values(self):
        """Test dictionaries with different values."""
        check = EqualsCheck(
            actual={"name": "Alice", "age": 30},
            expected={"name": "Bob", "age": 30},
        )
        result = check()
        assert result == {"passed": False}

    def test_equals_sets_identical(self):
        """Test identical sets."""
        check = EqualsCheck(actual={1, 2, 3}, expected={1, 2, 3})
        result = check()
        assert result == {"passed": True}

    def test_equals_sets_different_order_same_content(self):
        """Test sets with different order but same content."""
        check = EqualsCheck(actual={1, 2, 3}, expected={3, 1, 2})
        result = check()
        assert result == {"passed": True}  # Sets ignore order

    def test_equals_none_values(self):
        """Test None values."""
        check = EqualsCheck(actual=None, expected=None)
        result = check()
        assert result == {"passed": True}

    def test_equals_none_vs_empty_string(self):
        """Test None vs empty string."""
        check = EqualsCheck(actual=None, expected="")
        result = check()
        assert result == {"passed": False}

    def test_equals_mixed_types_different(self):
        """Test different types."""
        check = EqualsCheck(actual="42", expected=42)
        result = check()
        assert result == {"passed": False}  # String vs int

    def test_equals_nested_structures(self):
        """Test nested data structures."""
        check = EqualsCheck(
            actual={"users": [{"name": "Alice"}, {"name": "Bob"}]},
            expected={"users": [{"name": "Alice"}, {"name": "Bob"}]},
        )
        result = check()
        assert result == {"passed": True}

    def test_equals_negate_true(self):
        """Test negate=true passes when values differ."""
        check = EqualsCheck(actual="hello", expected="world", negate=True)
        result = check()
        assert result == {"passed": True}

    def test_equals_negate_false_match(self):
        """Test negate=true fails when values match."""
        check = EqualsCheck(actual="hello", expected="hello", negate=True)
        result = check()
        assert result == {"passed": False}


class TestEqualsEngineIntegration:
    """Test EqualsCheck integration with the evaluation engine."""

    @pytest.mark.parametrize(("output_value", "expected_value", "expected_passed"), [
        ("Paris", "Paris", True),  # Exact match should pass
        ("paris", "Paris", False),  # Case mismatch should fail
        ({"result": 42}, 42, True),  # Identical numbers (extract from dict)
        ({"result": 42}, 24, False),  # Different numbers
        ({"result": [1, 2, 3]}, [1, 2, 3], True),  # Identical lists
        ({"result": [1, 2, 3]}, [3, 2, 1], False),  # Different order
    ])
    def test_equals_via_evaluate(
        self, output_value: Any, expected_value: Any, expected_passed: bool,
    ):
        """Test using JSONPath for actual and expected with various combinations."""
        # Use different JSONPath based on output structure
        actual_path = "$.output.value.result" if isinstance(output_value, dict) and "result" in output_value else "$.output.value"  # noqa: E501

        test_cases = [
            TestCase(
                id="test_001",
                input="What is the capital of France?",
                expected=expected_value,
                checks=[
                    Check(
                        type=CheckType.EQUALS,
                        arguments={
                            "actual": actual_path,
                            "expected": "$.test_case.expected",
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

    def test_equals_check_instance_via_evaluate(self):
        """Test direct check instance usage in evaluate function."""
        test_cases = [
            TestCase(
                id="test_001",
                input="What is the capital of France?",
                expected="Paris",
                checks=[
                    EqualsCheck(
                        actual="$.output.value",
                        expected="$.test_case.expected",
                    ),
                ],
            ),
        ]

        outputs = [Output(value="Paris")]
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
                input="What is the capital of France?",
                expected="London",  # Wrong answer
                checks=[
                    Check(
                        type=CheckType.EQUALS,
                        arguments={
                            "actual": "$.output.value",
                            "expected": "$.test_case.expected",
                            "negate": True,  # Should pass because Paris != London
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="Paris")]
        results = evaluate(test_cases, outputs)

        assert results.results[0].check_results[0].results == {"passed": True}


class TestEqualsErrorHandling:
    """Test error handling and edge cases for EqualsCheck."""

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
        with pytest.raises(ValidationError, match="Invalid JSONPath expression"):
            evaluate(test_cases, outputs)

    def test_equals_missing_jsonpath_data(self):
        """Test behavior when JSONPath doesn't find data."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.EQUALS,
                        arguments={
                            "actual": "$.output.value.nonexistent",
                            "expected": "test",
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


class TestEqualsJSONPathIntegration:
    """Test EqualsCheck with various JSONPath expressions and data structures."""

    def test_equals_nested_jsonpath(self):
        """Test equals with deeply nested JSONPath expressions."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected={"result": {"answer": "Paris", "confidence": 0.95}},
                checks=[
                    Check(
                        type=CheckType.EQUALS,
                        arguments={
                            "actual": "$.output.value.answer",
                            "expected": "$.test_case.expected.result.answer",
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "answer": "Paris",
                "confidence": 0.95,
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
                expected={"first_item": "item1"},
                checks=[
                    Check(
                        type=CheckType.EQUALS,
                        arguments={
                            "actual": "$.output.value.items[0]",
                            "expected": "$.test_case.expected.first_item",
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "items": ["item1", "item2", "item3"],
                "count": 3,
            }),
        ]

        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_equals_complex_data_structures(self):
        """Test equals with complex nested data structures from JSONPath."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected={
                    "user_data": {
                        "profile": {"name": "Alice", "age": 30},
                        "preferences": ["reading", "coding"],
                    },
                },
                checks=[
                    Check(
                        type=CheckType.EQUALS,
                        arguments={
                            "actual": "$.output.value.user",
                            "expected": "$.test_case.expected.user_data",
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "user": {
                    "profile": {"name": "Alice", "age": 30},
                    "preferences": ["reading", "coding"],
                },
                "timestamp": "2024-01-01T00:00:00Z",
            }),
        ]

        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}
