"""
Comprehensive tests for ExactMatchCheck implementation.

This module consolidates all tests for the ExactMatchCheck including:
- Pydantic validation tests (from test_schema_check_classes.py)
- Implementation execution tests (from test_standard_checks.py)
- Engine integration tests
- Edge cases and error handling

Tests are organized by functionality rather than implementation details.
"""

import pytest

from flex_evals import (
    ExactMatchCheck,
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


class TestExactMatchValidation:
    """Test Pydantic validation and field handling for ExactMatchCheck."""

    def test_exact_match_check_creation(self):
        """Test basic ExactMatchCheck creation."""
        check = ExactMatchCheck(
            actual="$.output.value",
            expected="$.expected",
        )

        assert isinstance(check.actual, JSONPath)


        assert check.actual.expression == "$.output.value"
        assert isinstance(check.expected, JSONPath)

        assert check.expected.expression == "$.expected"
        assert check.case_sensitive is True
        assert check.negate is False

    def test_exact_match_check_with_options(self):
        """Test ExactMatchCheck with all options."""
        check = ExactMatchCheck(
            actual="$.output.value",
            expected="$.expected",
            case_sensitive=False,
            negate=True,
        )

        assert check.case_sensitive is False
        assert check.negate is True

    def test_exact_match_jsonpath_comprehensive(self):
        """Comprehensive JSONPath string conversion and execution test."""
        # 1. Create check with all JSONPath fields as strings
        check = ExactMatchCheck(
            actual="$.output.value.text",
            expected="$.test_case.expected.target_text",
            case_sensitive="$.test_case.expected.case_sensitive",
            negate="$.test_case.expected.should_negate",
        )

        # 2. Verify conversion happened
        assert isinstance(check.actual, JSONPath)
        assert check.actual.expression == "$.output.value.text"
        assert isinstance(check.expected, JSONPath)
        assert check.expected.expression == "$.test_case.expected.target_text"
        assert isinstance(check.case_sensitive, JSONPath)
        assert check.case_sensitive.expression == "$.test_case.expected.case_sensitive"
        assert isinstance(check.negate, JSONPath)
        assert check.negate.expression == "$.test_case.expected.should_negate"

        # 3. Test execution with EvaluationContext

        test_case = TestCase(
            id="test_001",
            input="test",
            expected={
                "target_text": "Hello World",
                "case_sensitive": False,
                "should_negate": False,
            },
        )
        output = Output(value={"text": "hello world"})
        context = EvaluationContext(test_case, output)

        result = check.execute(context)
        assert result.status == "completed"
        assert result.results["passed"] is True  # Case-insensitive match
        assert result.resolved_arguments["actual"]["value"] == "hello world"
        assert result.resolved_arguments["expected"]["value"] == "Hello World"
        assert result.resolved_arguments["case_sensitive"]["value"] is False
        assert result.resolved_arguments["negate"]["value"] is False

        # 4. Test invalid JSONPath string (should raise exception during validation)
        with pytest.raises(PydanticValidationError, match="Invalid JSONPath expression"):
            ExactMatchCheck(actual="$.invalid[", expected="valid_value")

        with pytest.raises(PydanticValidationError, match="Invalid JSONPath expression"):
            ExactMatchCheck(actual="$.valid.actual", expected="$.invalid[")

        # 5. Test valid literal values (should work)
        check_literal = ExactMatchCheck(
            actual="literal_actual",  # Any | JSONPath - any value works
            expected="literal_expected",  # Any | JSONPath - any value works
            case_sensitive=True,  # bool | JSONPath - boolean literal works
            negate=False,  # bool | JSONPath - boolean literal works
        )
        assert check_literal.actual == "literal_actual"
        assert check_literal.expected == "literal_expected"
        assert check_literal.case_sensitive is True
        assert check_literal.negate is False

        # 6. Test invalid non-JSONPath strings for fields that don't support string
        # case_sensitive: bool | JSONPath - "not_a_boolean" is neither bool nor JSONPath
        with pytest.raises(PydanticValidationError):
            ExactMatchCheck(actual="test", expected="test", case_sensitive="not_a_boolean")

        # negate: bool | JSONPath - "not_a_boolean" is neither bool nor JSONPath
        with pytest.raises(PydanticValidationError):
            ExactMatchCheck(actual="test", expected="test", negate="not_a_boolean")

    def test_exact_match_check_validation_empty_actual(self):
        """Test ExactMatchCheck validation for empty actual - now allowed."""
        check = ExactMatchCheck(actual="", expected="$.expected")
        assert check.actual == ""
        assert isinstance(check.expected, JSONPath)

        assert check.expected.expression == "$.expected"

    def test_exact_match_check_validation_empty_expected(self):
        """Test ExactMatchCheck validation for empty expected - now allowed."""
        check = ExactMatchCheck(actual="$.actual", expected="")
        assert isinstance(check.actual, JSONPath)

        assert check.actual.expression == "$.actual"
        assert check.expected == ""

    def test_exact_match_check_type_property(self):
        """Test ExactMatchCheck check_type property returns correct type."""
        check = ExactMatchCheck(actual="test", expected="test")
        assert check.check_type == CheckType.EXACT_MATCH

    def test_exact_match_check_version_property(self):
        """Test ExactMatchCheck version property works correctly."""
        check = ExactMatchCheck(actual="test", expected="test")
        assert check._get_version() == "1.0.0"


class TestExactMatchExecution:
    """Test ExactMatchCheck execution logic and __call__ method."""

    def test_exact_match_string_equal(self):
        """Test matching strings return passed=true."""
        check = ExactMatchCheck(actual="Paris", expected="Paris")
        result = check()
        assert result == {"passed": True}

    def test_exact_match_string_not_equal(self):
        """Test non-matching strings return passed=false."""
        check = ExactMatchCheck(actual="paris", expected="Paris")
        result = check()
        assert result == {"passed": False}

    def test_exact_match_case_sensitive_true(self):
        """Test 'Hello' != 'hello' when case_sensitive=true."""
        check = ExactMatchCheck(actual="Hello", expected="hello", case_sensitive=True)
        result = check()
        assert result == {"passed": False}

    def test_exact_match_case_sensitive_false(self):
        """Test 'Hello' == 'hello' when case_sensitive=false."""
        check = ExactMatchCheck(actual="Hello", expected="hello", case_sensitive=False)
        result = check()
        assert result == {"passed": True}

    def test_exact_match_negate_true(self):
        """Test negate=true passes when values differ."""
        check = ExactMatchCheck(actual="Paris", expected="London", negate=True)
        result = check()
        assert result == {"passed": True}

    def test_exact_match_negate_false(self):
        """Test negate=false passes when values match."""
        check = ExactMatchCheck(actual="Paris", expected="Paris", negate=False)
        result = check()
        assert result == {"passed": True}

    def test_exact_match_object_comparison(self):
        """Test comparing complex objects."""
        # Objects will be converted to strings for comparison
        check = ExactMatchCheck(actual={"city": "Paris"}, expected="{'city': 'Paris'}")
        result = check()
        assert result == {"passed": True}

    def test_exact_match_null_values(self):
        """Test comparison with null/None values."""
        check = ExactMatchCheck(actual=None, expected="")
        result = check()
        assert result == {"passed": True}  # None converts to empty string

    def test_exact_match_result_schema(self):
        r"""Test result matches {\"passed\": boolean} exactly."""
        check = ExactMatchCheck(actual="test", expected="test")
        result = check()
        assert isinstance(result, dict)
        assert set(result.keys()) == {"passed"}
        assert isinstance(result["passed"], bool)


class TestExactMatchEngineIntegration:
    """Test ExactMatchCheck integration with the evaluation engine."""

    @pytest.mark.parametrize(("output_value", "expected_value", "expected_passed"), [
        ("Paris", "Paris", True),  # Exact match should pass
        ("paris", "Paris", False),  # Case mismatch should fail (case_sensitive=True by default)
        ("London", "Paris", False),  # Different values should fail
        ("", "", True),  # Empty strings should match
    ])
    def test_exact_match_via_evaluate(
        self, output_value: str, expected_value: str, expected_passed: bool,
    ):
        """Test using JSONPath for actual and expected values with various combinations."""
        # Define your test cases
        test_cases = [
            TestCase(
                id="test_001",
                input="What is the capital of France?",
                expected=expected_value,
                checks=[
                    Check(
                        type=CheckType.EXACT_MATCH,
                        arguments={
                            "actual": "$.output.value",
                            "expected": "$.test_case.expected",
                        },
                    ),
                ],
            ),
        ]
        # System outputs to evaluate
        outputs = [
            Output(value=output_value),
        ]
        # Run evaluation
        results = evaluate(test_cases, outputs)
        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.summary.error_test_cases == 0
        assert results.summary.skipped_test_cases == 0
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": expected_passed}

    def test_exact_match_check_instance_via_evaluate(self):
        """Test direct check instance usage in evaluate function."""
        # Test with check instance instead of Check dataclass
        test_cases = [
            TestCase(
                id="test_001",
                input="What is the capital of France?",
                expected="Paris",
                checks=[
                    ExactMatchCheck(
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

    def test_exact_match_with_case_insensitive_via_evaluate(self):
        """Test case insensitive matching through engine evaluation."""
        test_cases = [
            TestCase(
                id="test_001",
                input="What is the capital of France?",
                expected="paris",
                checks=[
                    Check(
                        type=CheckType.EXACT_MATCH,
                        arguments={
                            "actual": "$.output.value",
                            "expected": "$.test_case.expected",
                            "case_sensitive": False,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="Paris")]
        results = evaluate(test_cases, outputs)

        assert results.results[0].check_results[0].results == {"passed": True}

    def test_exact_match_with_negate_via_evaluate(self):
        """Test negation through engine evaluation."""
        test_cases = [
            TestCase(
                id="test_001",
                input="What is the capital of France?",
                expected="London",
                checks=[
                    Check(
                        type=CheckType.EXACT_MATCH,
                        arguments={
                            "actual": "$.output.value",
                            "expected": "$.test_case.expected",
                            "negate": True,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="Paris")]
        results = evaluate(test_cases, outputs)

        assert results.results[0].check_results[0].results == {"passed": True}


class TestExactMatchErrorHandling:
    """Test error handling and edge cases for ExactMatchCheck."""

    def test_exact_match_invalid_field_type(self):
        """Test that various field types are accepted since ExactMatch can compare any types."""
        # ExactMatchCheck now accepts any types for comparison
        check1 = ExactMatchCheck(actual=123, expected="test")
        assert check1.actual == 123
        assert check1.expected == "test"

        check2 = ExactMatchCheck(actual="test", expected=123)
        assert check2.actual == "test"
        assert check2.expected == 123

    def test_exact_match_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(PydanticValidationError):
            ExactMatchCheck()  # type: ignore

        with pytest.raises(PydanticValidationError):
            ExactMatchCheck(actual="test")  # type: ignore

        with pytest.raises(PydanticValidationError):
            ExactMatchCheck(expected="test")  # type: ignore

    def test_exact_match_jsonpath_validation_in_engine(self):
        """Test that invalid JSONPath expressions are caught during evaluation."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.EXACT_MATCH,
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

    def test_exact_match_complex_data_types(self):
        """Test exact match with complex data types."""
        # Test with nested dictionaries
        actual = {"user": {"name": "Alice", "age": 30}}
        expected = {"user": {"name": "Alice", "age": 30}}
        check1 = ExactMatchCheck(actual=actual, expected=expected)
        result = check1()
        assert result == {"passed": True}

        # Test with lists
        check2 = ExactMatchCheck(actual=[1, 2, 3], expected=[1, 2, 3])
        result = check2()
        assert result == {"passed": True}

        # Test with mixed types (string representation matches)
        check3 = ExactMatchCheck(actual="123", expected=123)
        result = check3()
        assert result == {"passed": True}  # str(123) == "123"

        # Test with truly different types that don't match
        check4 = ExactMatchCheck(actual="hello", expected=123)
        result = check4()
        assert result == {"passed": False}


class TestExactMatchJSONPathIntegration:
    """Test ExactMatchCheck with various JSONPath expressions and data structures."""

    def test_exact_match_nested_jsonpath(self):
        """Test exact match with deeply nested JSONPath expressions."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected={"location": {"city": "Paris", "country": "France"}},
                checks=[
                    Check(
                        type=CheckType.EXACT_MATCH,
                        arguments={
                            "actual": "$.output.value.response.location.city",
                            "expected": "$.test_case.expected.location.city",
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "response": {
                    "location": {"city": "Paris", "country": "France"},
                    "confidence": 0.95,
                },
            }),
        ]

        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_exact_match_array_access_jsonpath(self):
        """Test exact match with JSONPath array access."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected=["Paris", "London", "Berlin"],
                checks=[
                    Check(
                        type=CheckType.EXACT_MATCH,
                        arguments={
                            "actual": "$.output.value.cities[0]",
                            "expected": "$.test_case.expected[0]",
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "cities": ["Paris", "London", "Berlin"],
                "total": 3,
            }),
        ]

        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_exact_match_missing_jsonpath_data(self):
        """Test behavior when JSONPath doesn't find data."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.EXACT_MATCH,
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
