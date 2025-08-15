"""
Comprehensive tests for IsEmptyCheck implementation.

This module consolidates all tests for the IsEmptyCheck including:
- Pydantic validation tests
- Implementation execution tests
- Engine integration tests
- Edge cases and error handling

Tests are organized by functionality rather than implementation details.
"""

import pytest
from typing import Any

from flex_evals import (
    IsEmptyCheck,
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


class TestIsEmptyValidation:
    """Test Pydantic validation and field handling for IsEmptyCheck."""

    def test_is_empty_check_creation(self):
        """Test basic IsEmptyCheck creation."""
        check = IsEmptyCheck(value="$.output.value")

        assert isinstance(check.value, JSONPath)
        assert check.value.expression == "$.output.value"
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

    def test_is_empty_jsonpath_comprehensive(self):
        """Comprehensive JSONPath string conversion and execution test."""
        # 1. Create check with all JSONPath fields as strings
        check = IsEmptyCheck(
            value="$.output.value.content",
            negate="$.test_case.expected.should_negate",
            strip_whitespace="$.test_case.expected.strip_whitespace",
        )

        # 2. Verify conversion happened
        assert isinstance(check.value, JSONPath)
        assert check.value.expression == "$.output.value.content"
        assert isinstance(check.negate, JSONPath)
        assert check.negate.expression == "$.test_case.expected.should_negate"
        assert isinstance(check.strip_whitespace, JSONPath)
        assert check.strip_whitespace.expression == "$.test_case.expected.strip_whitespace"

        # 3. Test execution with EvaluationContext

        test_case = TestCase(
            id="test_001",
            input="test",
            expected={
                "should_negate": False,
                "strip_whitespace": True,
            },
        )
        output = Output(value={"content": "   "})  # Whitespace only
        context = EvaluationContext(test_case, output)

        result = check.execute(context)
        assert result.status == "completed"
        assert result.results["passed"] is True  # Empty after stripping whitespace
        assert result.resolved_arguments["value"]["value"] == "   "
        assert result.resolved_arguments["negate"]["value"] is False
        assert result.resolved_arguments["strip_whitespace"]["value"] is True

        # 4. Test invalid JSONPath string (should raise exception during validation)
        with pytest.raises(PydanticValidationError, match="Invalid JSONPath expression"):
            IsEmptyCheck(value="$.invalid[")

        with pytest.raises(PydanticValidationError, match="Invalid JSONPath expression"):
            IsEmptyCheck(value="$.valid.value", negate="$.invalid[")

        # 5. Test valid literal values (should work)
        check_literal = IsEmptyCheck(
            value=[],  # Any supported type | JSONPath - empty list works
            negate=False,  # bool | JSONPath - boolean literal works
            strip_whitespace=True,  # bool | JSONPath - boolean literal works
        )
        assert check_literal.value == []
        assert check_literal.negate is False
        assert check_literal.strip_whitespace is True

        # 6. Test invalid non-JSONPath strings for fields that don't support string
        # negate: bool | JSONPath - "not_a_boolean" is neither bool nor JSONPath
        with pytest.raises(PydanticValidationError):
            IsEmptyCheck(value="", negate="not_a_boolean")

        # strip_whitespace: bool | JSONPath - "not_a_boolean" is neither bool nor JSONPath
        with pytest.raises(PydanticValidationError):
            IsEmptyCheck(value="", strip_whitespace="not_a_boolean")

    def test_is_empty_check_with_literal_value(self):
        """Test IsEmptyCheck with literal value."""
        check = IsEmptyCheck(
            value="",
            negate=False,
            strip_whitespace=True,
        )

        assert check.value == ""
        assert check.negate is False
        assert check.strip_whitespace is True

    def test_is_empty_check_type_property(self):
        """Test IsEmptyCheck check_type property returns correct type."""
        check = IsEmptyCheck(value="")
        assert check.check_type == CheckType.IS_EMPTY

    def test_is_empty_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(PydanticValidationError):
            IsEmptyCheck()  # type: ignore


class TestIsEmptyExecution:
    """Test IsEmptyCheck execution logic and __call__ method."""

    def test_is_empty_none_value(self):
        """Test None is considered empty."""
        check = IsEmptyCheck(value=None)
        result = check()
        assert result == {"passed": True}

    def test_is_empty_empty_string(self):
        """Test empty string is considered empty."""
        check = IsEmptyCheck(value="")
        result = check()
        assert result == {"passed": True}

    def test_is_empty_whitespace_string_strip_true(self):
        """Test whitespace-only string with strip_whitespace=True."""
        check = IsEmptyCheck(value="   ", strip_whitespace=True)
        result = check()
        assert result == {"passed": True}

    def test_is_empty_whitespace_string_strip_false(self):
        """Test whitespace-only string with strip_whitespace=False."""
        check = IsEmptyCheck(value="   ", strip_whitespace=False)
        result = check()
        assert result == {"passed": False}  # Non-empty because we don't strip

    def test_is_empty_non_empty_string(self):
        """Test non-empty string."""
        check = IsEmptyCheck(value="hello")
        result = check()
        assert result == {"passed": False}

    def test_is_empty_empty_list(self):
        """Test empty list."""
        check = IsEmptyCheck(value=[])
        result = check()
        assert result == {"passed": True}

    def test_is_empty_non_empty_list(self):
        """Test non-empty list."""
        check = IsEmptyCheck(value=[1, 2, 3])
        result = check()
        assert result == {"passed": False}

    def test_is_empty_empty_dict(self):
        """Test empty dictionary."""
        check = IsEmptyCheck(value={})
        result = check()
        assert result == {"passed": True}

    def test_is_empty_non_empty_dict(self):
        """Test non-empty dictionary."""
        check = IsEmptyCheck(value={"key": "value"})
        result = check()
        assert result == {"passed": False}

    def test_is_empty_empty_set(self):
        """Test empty set."""
        check = IsEmptyCheck(value=set())
        result = check()
        assert result == {"passed": True}

    def test_is_empty_non_empty_set(self):
        """Test non-empty set."""
        check = IsEmptyCheck(value={1, 2, 3})
        result = check()
        assert result == {"passed": False}

    def test_is_empty_empty_tuple(self):
        """Test empty tuple."""
        check = IsEmptyCheck(value=())
        result = check()
        assert result == {"passed": True}

    def test_is_empty_non_empty_tuple(self):
        """Test non-empty tuple."""
        check = IsEmptyCheck(value=(1, 2, 3))
        result = check()
        assert result == {"passed": False}

    def test_is_empty_numeric_values(self):
        """Test numeric values are never considered empty."""
        check_zero = IsEmptyCheck(value=0)
        assert check_zero() == {"passed": False}

        check_int = IsEmptyCheck(value=42)
        assert check_int() == {"passed": False}

        check_float = IsEmptyCheck(value=3.14)
        assert check_float() == {"passed": False}

    def test_is_empty_boolean_values(self):
        """Test boolean values are never considered empty."""
        check_true = IsEmptyCheck(value=True)
        assert check_true() == {"passed": False}

        check_false = IsEmptyCheck(value=False)
        assert check_false() == {"passed": False}

    def test_is_empty_negate_true_empty_value(self):
        """Test negate=True with empty value."""
        check = IsEmptyCheck(value="", negate=True)
        result = check()
        assert result == {"passed": False}  # Empty value, but negated

    def test_is_empty_negate_true_non_empty_value(self):
        """Test negate=True with non-empty value."""
        check = IsEmptyCheck(value="hello", negate=True)
        result = check()
        assert result == {"passed": True}  # Non-empty value, negated

    def test_is_empty_string_with_newlines(self):
        """Test string with only newlines and whitespace."""
        check_strip = IsEmptyCheck(value="\n\t  \r\n", strip_whitespace=True)
        assert check_strip() == {"passed": True}

        check_no_strip = IsEmptyCheck(value="\n\t  \r\n", strip_whitespace=False)
        assert check_no_strip() == {"passed": False}


class TestIsEmptyEngineIntegration:
    """Test IsEmptyCheck integration with the evaluation engine."""

    @pytest.mark.parametrize(("output_value", "expected_passed"), [
        ("", True),  # Empty string
        ("hello", False),  # Non-empty string
        ("   ", True),  # Whitespace (strip_whitespace=True by default)
        ({"result": []}, True),  # Empty list (via JSONPath)
        ({"result": [1, 2, 3]}, False),  # Non-empty list (via JSONPath)
        ({"result": {}}, True),  # Empty dict (via JSONPath)
        ({"result": {"key": "value"}}, False),  # Non-empty dict (via JSONPath)
    ])
    def test_is_empty_via_evaluate(self, output_value: Any, expected_passed: bool):
        """Test using JSONPath for value with various combinations."""
        # Use different JSONPath based on output structure
        value_path = "$.output.value.result" if isinstance(output_value, dict) and "result" in output_value else "$.output.value"  # noqa: E501

        test_cases = [
            TestCase(
                id="test_001",
                input="test input",
                checks=[
                    Check(
                        type=CheckType.IS_EMPTY,
                        arguments={
                            "value": value_path,
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

    def test_is_empty_check_instance_via_evaluate(self):
        """Test direct check instance usage in evaluate function."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test input",
                checks=[
                    IsEmptyCheck(
                        value="$.output.value.content",
                    ),
                ],
            ),
        ]

        outputs = [Output(value={"content": ""})]
        results = evaluate(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_is_empty_negate_via_evaluate(self):
        """Test negation through engine evaluation."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test input",
                checks=[
                    Check(
                        type=CheckType.IS_EMPTY,
                        arguments={
                            "value": "$.output.value",
                            "negate": True,  # Should pass because value is not empty
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="not empty")]
        results = evaluate(test_cases, outputs)

        assert results.results[0].check_results[0].results == {"passed": True}

    def test_is_empty_strip_whitespace_via_evaluate(self):
        """Test strip_whitespace option through engine evaluation."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test input",
                checks=[
                    Check(
                        type=CheckType.IS_EMPTY,
                        arguments={
                            "value": "$.output.value",
                            "strip_whitespace": False,  # Don't strip whitespace
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="   ")]  # Whitespace only
        results = evaluate(test_cases, outputs)

        # With strip_whitespace=False, whitespace is not empty
        assert results.results[0].check_results[0].results == {"passed": False}


class TestIsEmptyErrorHandling:
    """Test error handling and edge cases for IsEmptyCheck."""

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

        # Should raise validation error for invalid JSONPath
        with pytest.raises(ValidationError, match="Invalid JSONPath expression"):
            evaluate(test_cases, outputs)

    def test_is_empty_missing_jsonpath_data(self):
        """Test behavior when JSONPath doesn't find data."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.IS_EMPTY,
                        arguments={
                            "value": "$.output.value.nonexistent",
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


class TestIsEmptyJSONPathIntegration:
    """Test IsEmptyCheck with various JSONPath expressions and data structures."""

    def test_is_empty_nested_jsonpath(self):
        """Test is_empty with deeply nested JSONPath expressions."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.IS_EMPTY,
                        arguments={
                            "value": "$.output.value.response.data.items",
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "response": {
                    "data": {
                        "items": [],
                        "count": 0,
                    },
                    "status": "success",
                },
            }),
        ]

        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_is_empty_array_access_jsonpath(self):
        """Test is_empty with JSONPath array access."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.IS_EMPTY,
                        arguments={
                            "value": "$.output.value.messages[0].content",
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "messages": [
                    {"content": "", "timestamp": "2024-01-01"},
                    {"content": "Hello", "timestamp": "2024-01-02"},
                ],
            }),
        ]

        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_is_empty_complex_data_structures(self):
        """Test is_empty with various data structures from JSONPath."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.IS_EMPTY,
                        arguments={
                            "value": "$.output.value.user.preferences",
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "user": {
                    "profile": {"name": "Alice", "age": 30},
                    "preferences": [],  # Empty array
                },
                "timestamp": "2024-01-01T00:00:00Z",
            }),
        ]

        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}
