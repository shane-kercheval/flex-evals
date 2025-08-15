"""
Comprehensive tests for ContainsCheck implementation.

This module consolidates all tests for the ContainsCheck including:
- Pydantic validation tests (from test_schema_check_classes.py)
- Implementation execution tests (from test_standard_checks.py)
- Engine integration tests
- Edge cases and error handling

Tests are organized by functionality rather than implementation details.
"""

import pytest

from flex_evals import (
    ContainsCheck,
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


class TestContainsValidation:
    """Test Pydantic validation and field handling for ContainsCheck."""

    def test_contains_check_creation(self):
        """Test basic ContainsCheck creation."""
        check = ContainsCheck(
            text="$.output.value",
            phrases=["hello", "world"],
        )

        assert isinstance(check.text, JSONPath)
        assert check.text.expression == "$.output.value"
        assert check.phrases == ["hello", "world"]
        assert check.case_sensitive is True
        assert check.negate is False

    def test_contains_check_with_options(self):
        """Test ContainsCheck with all options."""
        check = ContainsCheck(
            text="$.output.value",
            phrases=["hello"],
            case_sensitive=False,
            negate=True,
        )

        assert check.case_sensitive is False
        assert check.negate is True

    def test_contains_jsonpath_comprehensive(self):
        """Comprehensive JSONPath string conversion and execution test."""
        # 1. Create check with all JSONPath fields as strings
        check = ContainsCheck(
            text="$.output.value.message",
            phrases="$.test_case.expected.required_phrases",
            case_sensitive="$.test_case.expected.case_sensitive",
            negate="$.test_case.expected.should_negate",
        )

        # 2. Verify conversion happened
        assert isinstance(check.text, JSONPath)
        assert check.text.expression == "$.output.value.message"
        assert isinstance(check.phrases, JSONPath)
        assert check.phrases.expression == "$.test_case.expected.required_phrases"
        assert isinstance(check.case_sensitive, JSONPath)
        assert check.case_sensitive.expression == "$.test_case.expected.case_sensitive"
        assert isinstance(check.negate, JSONPath)
        assert check.negate.expression == "$.test_case.expected.should_negate"

        # 3. Test execution with EvaluationContext

        test_case = TestCase(
            id="test_001",
            input="test",
            expected={
                "required_phrases": ["success", "completed"],
                "case_sensitive": True,
                "should_negate": False,
            },
        )
        output = Output(value={"message": "Task completed successfully!"})
        context = EvaluationContext(test_case, output)

        result = check.execute(context)
        assert result.status == "completed"
        assert result.results["passed"] is True  # Contains both "success" and "completed"
        assert result.resolved_arguments["text"]["value"] == "Task completed successfully!"
        assert result.resolved_arguments["phrases"]["value"] == ["success", "completed"]
        assert result.resolved_arguments["case_sensitive"]["value"] is True
        assert result.resolved_arguments["negate"]["value"] is False

        # 4. Test invalid JSONPath string (should raise exception during validation)
        with pytest.raises(PydanticValidationError, match="Invalid JSONPath expression"):
            ContainsCheck(text="$.invalid[", phrases=["test"])

        with pytest.raises(PydanticValidationError, match="Invalid JSONPath expression"):
            ContainsCheck(text="$.valid.text", phrases="$.invalid[")

        # 5. Test valid literal values (should work)
        check_literal = ContainsCheck(
            text="literal text content",  # str | JSONPath - string literal works
            phrases=["phrase1", "phrase2"],  # str | list[str] | JSONPath - list works
            case_sensitive=True,  # bool | JSONPath - boolean literal works
            negate=False,  # bool | JSONPath - boolean literal works
        )
        assert check_literal.text == "literal text content"
        assert check_literal.phrases == ["phrase1", "phrase2"]
        assert check_literal.case_sensitive is True
        assert check_literal.negate is False

        # 6. Test invalid non-JSONPath strings for fields that don't support string
        # case_sensitive: bool | JSONPath - "not_a_boolean" is neither bool nor JSONPath
        with pytest.raises(PydanticValidationError):
            ContainsCheck(text="test", phrases=["test"], case_sensitive="not_a_boolean")

        # negate: bool | JSONPath - "not_a_boolean" is neither bool nor JSONPath
        with pytest.raises(PydanticValidationError):
            ContainsCheck(text="test", phrases=["test"], negate="not_a_boolean")

        # But text and phrases fields DO support strings, so non-JSONPath strings should work
        check_string_fields = ContainsCheck(
            text="not_a_jsonpath_but_valid_string",  # Valid string
            phrases="also_not_jsonpath_but_valid_string",  # Valid string
        )
        assert check_string_fields.text == "not_a_jsonpath_but_valid_string"
        assert check_string_fields.phrases == "also_not_jsonpath_but_valid_string"

    def test_contains_check_validation_empty_text(self):
        """Test ContainsCheck validation for empty text - now allowed."""
        check = ContainsCheck(text="", phrases=["hello"])
        assert check.text == ""
        assert check.phrases == ["hello"]

    def test_contains_check_validation_empty_phrases(self):
        """Test ContainsCheck validation for empty phrases."""
        # Empty phrases are now allowed at construction but fail at execution
        check = ContainsCheck(text="$.value", phrases=[])
        assert isinstance(check.text, JSONPath)
        assert check.phrases == []

    def test_contains_check_validation_invalid_phrases(self):
        """Test ContainsCheck validation for invalid phrases - empty strings now allowed."""
        check = ContainsCheck(text="$.value", phrases=["valid", ""])
        assert isinstance(check.text, JSONPath)

        assert check.text.expression == "$.value"
        assert check.phrases == ["valid", ""]

    def test_contains_check_type_property(self):
        """Test ContainsCheck check_type property returns correct type."""
        check = ContainsCheck(text="test", phrases=["test"])
        assert check.check_type == CheckType.CONTAINS

    def test_contains_check_phrases_string(self):
        """Test ContainsCheck with string phrases."""
        check = ContainsCheck(
            text="$.output.value",
            phrases="hello",
        )

        assert isinstance(check.text, JSONPath)


        assert check.text.expression == "$.output.value"
        assert check.phrases == "hello"
        assert isinstance(check.phrases, str)

    def test_contains_check_phrases_jsonpath(self):
        """Test ContainsCheck with JSONPath phrases."""
        check = ContainsCheck(
            text="$.output.value",
            phrases="$.expected.phrases",
        )

        assert isinstance(check.phrases, JSONPath)
        assert check.phrases.expression == "$.expected.phrases"

    def test_contains_check_validation_empty_phrases_string(self):
        """Test ContainsCheck validation for empty phrases string - now allowed."""
        check = ContainsCheck(text="$.value", phrases="")
        assert isinstance(check.text, JSONPath)

        assert check.text.expression == "$.value"
        assert check.phrases == ""

    def test_contains_check_validation_invalid_phrases_type(self):
        """Test ContainsCheck validation for invalid phrases type."""
        with pytest.raises(PydanticValidationError):
            ContainsCheck(text="$.value", phrases=123)  # type: ignore


class TestContainsExecution:
    """Test ContainsCheck execution logic and __call__ method."""

    def test_contains_all_phrases_found(self):
        """Test negate=false passes when all phrases present."""
        check = ContainsCheck(
            text="Paris is the capital of France",
            phrases=["Paris", "France"],
            negate=False,
        )
        result = check()
        assert result == {"passed": True}

    def test_contains_some_phrases_missing(self):
        """Test negate=false fails when any phrase missing."""
        check = ContainsCheck(
            text="Paris is the capital of France",
            phrases=["Paris", "Spain"],  # Spain is missing
            negate=False,
        )
        result = check()
        assert result == {"passed": False}

    def test_contains_negate_none_found(self):
        """Test negate=true passes when no phrases found."""
        check = ContainsCheck(
            text="Paris is the capital of France",
            phrases=["London", "Spain"],
            negate=True,
        )
        result = check()
        assert result == {"passed": True}

    def test_contains_negate_some_found(self):
        """Test negate=true fails when any phrase found."""
        check = ContainsCheck(
            text="Paris is the capital of France",
            phrases=["Paris", "Spain"],  # Paris is found
            negate=True,
        )
        result = check()
        assert result == {"passed": False}

    def test_contains_case_sensitive(self):
        """Test case sensitivity in phrase matching."""
        check = ContainsCheck(
            text="Paris is the capital of France",
            phrases=["paris"],  # Lowercase
            case_sensitive=True,
        )
        result = check()
        assert result == {"passed": False}

    def test_contains_case_insensitive(self):
        """Test case insensitive matching."""
        check = ContainsCheck(
            text="Paris is the capital of France",
            phrases=["paris"],  # Lowercase
            case_sensitive=False,
        )
        result = check()
        assert result == {"passed": True}

    def test_contains_single_phrase(self):
        """Test with single phrase in array."""
        check = ContainsCheck(
            text="Paris is the capital of France",
            phrases=["capital"],
        )
        result = check()
        assert result == {"passed": True}

    def test_contains_overlapping_phrases(self):
        """Test with overlapping/duplicate phrases."""
        check = ContainsCheck(
            text="Paris Paris is great",
            phrases=["Paris", "Paris"],  # Duplicate phrase
        )
        result = check()
        assert result == {"passed": True}

    def test_contains_single_string_phrase_found(self):
        """Test single string phrase that is found."""
        check = ContainsCheck(
            text="Paris is the capital of France",
            phrases="Paris",  # Single string instead of list
        )
        result = check()
        assert result == {"passed": True}

    def test_contains_single_string_phrase_not_found(self):
        """Test single string phrase that is not found."""
        check = ContainsCheck(
            text="Paris is the capital of France",
            phrases="Spain",  # Single string not found
        )
        result = check()
        assert result == {"passed": False}

    def test_contains_single_string_case_sensitive(self):
        """Test single string phrase with case sensitivity."""
        check = ContainsCheck(
            text="Paris is the capital of France",
            phrases="paris",  # Lowercase
            case_sensitive=True,
        )
        result = check()
        assert result == {"passed": False}

    def test_contains_single_string_case_insensitive(self):
        """Test single string phrase case insensitive."""
        check = ContainsCheck(
            text="Paris is the capital of France",
            phrases="paris",  # Lowercase
            case_sensitive=False,
        )
        result = check()
        assert result == {"passed": True}

    def test_contains_single_string_with_negate(self):
        """Test single string phrase with negate=True."""
        check = ContainsCheck(
            text="Paris is the capital of France",
            phrases="Spain",  # Not found
            negate=True,
        )
        result = check()
        assert result == {"passed": True}


class TestContainsEngineIntegration:
    """Test ContainsCheck integration with the evaluation engine."""

    @pytest.mark.parametrize(("output_value", "phrases", "expected_passed"), [
        ("The capital of France is Paris.", ["Paris", "France"], True),  # All phrases found
        ("The capital of France is Paris.", ["Paris", "Spain"], False),  # Not all phrases found
        ("Hello world", ["hello"], False),  # Case sensitive by default
        ("", ["anything"], False),  # Empty text should not contain phrases
        ("Contains everything", ["Contains", "everything"], True),  # Multiple phrases found
    ])
    def test_contains_via_evaluate(
        self, output_value: str, phrases: list[str], expected_passed: bool,
    ):
        """Test using JSONPath for text and phrases with various combinations."""
        # Define your test cases
        test_cases = [
            TestCase(
                id="test_001",
                input="What is the capital of France?",
                expected=phrases,
                checks=[
                    Check(
                        type=CheckType.CONTAINS,
                        arguments={
                            "text": "$.output.value",
                            "phrases": "$.test_case.expected",
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

    def test_contains_check_instance_via_evaluate(self):
        """Test direct check instance usage in evaluate function."""
        test_cases = [
            TestCase(
                id="test_001",
                input="What is the capital of France?",
                expected=["Paris", "France"],
                checks=[
                    ContainsCheck(
                        text="$.output.value",
                        phrases="$.test_case.expected",
                    ),
                ],
            ),
        ]

        outputs = [Output(value="The capital of France is Paris.")]
        results = evaluate(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_contains_case_insensitive_via_evaluate(self):
        """Test case insensitive matching through engine evaluation."""
        test_cases = [
            TestCase(
                id="test_001",
                input="What is the capital of France?",
                expected=["paris", "france"],
                checks=[
                    Check(
                        type=CheckType.CONTAINS,
                        arguments={
                            "text": "$.output.value",
                            "phrases": "$.test_case.expected",
                            "case_sensitive": False,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="The capital of France is Paris.")]
        results = evaluate(test_cases, outputs)

        assert results.results[0].check_results[0].results == {"passed": True}

    def test_contains_negate_via_evaluate(self):
        """Test negation through engine evaluation."""
        test_cases = [
            TestCase(
                id="test_001",
                input="What is the capital of France?",
                expected=["London", "Spain"],
                checks=[
                    Check(
                        type=CheckType.CONTAINS,
                        arguments={
                            "text": "$.output.value",
                            "phrases": "$.test_case.expected",
                            "negate": True,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="The capital of France is Paris.")]
        results = evaluate(test_cases, outputs)

        assert results.results[0].check_results[0].results == {"passed": True}


class TestContainsErrorHandling:
    """Test error handling and edge cases for ContainsCheck."""

    def test_contains_empty_phrases_execution_error(self):
        """Test behavior with empty phrases array during execution."""
        check = ContainsCheck(text="test text", phrases=[])
        with pytest.raises(ValidationError, match="must not be empty"):
            check()

    def test_contains_empty_string_phrase_execution_error(self):
        """Test empty string phrase raises ValidationError during execution."""
        check = ContainsCheck(text="test text", phrases="")
        with pytest.raises(ValidationError, match="must not be empty"):
            check()

    def test_contains_invalid_phrases_type_execution_error(self):
        """Test invalid phrases type raises ValidationError during execution."""
        # With new architecture, invalid types are caught during construction
        with pytest.raises(PydanticValidationError):
            ContainsCheck(text="test", phrases=123)  # type: ignore

    def test_contains_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(PydanticValidationError):
            ContainsCheck()  # type: ignore

        with pytest.raises(PydanticValidationError):
            ContainsCheck(text="test")  # type: ignore

        with pytest.raises(PydanticValidationError):
            ContainsCheck(phrases=["test"])  # type: ignore

    def test_contains_jsonpath_validation_in_engine(self):
        """Test that invalid JSONPath expressions are caught during evaluation."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.CONTAINS,
                        arguments={
                            "text": "$..[invalid",  # Invalid JSONPath syntax
                            "phrases": ["test"],
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="test")]

        # Should raise validation error for invalid JSONPath
        with pytest.raises(ValidationError, match="Invalid JSONPath expression"):
            evaluate(test_cases, outputs)


class TestContainsJSONPathIntegration:
    """Test ContainsCheck with various JSONPath expressions and data structures."""

    def test_contains_nested_jsonpath(self):
        """Test contains with deeply nested JSONPath expressions."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected={"phrases": ["success", "completed"]},
                checks=[
                    Check(
                        type=CheckType.CONTAINS,
                        arguments={
                            "text": "$.output.value.response.message",
                            "phrases": "$.test_case.expected.phrases",
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "response": {
                    "message": "Operation completed successfully",
                    "status": "success",
                },
            }),
        ]

        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_contains_array_access_jsonpath(self):
        """Test contains with JSONPath array access."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected={"search_terms": ["error", "failed"]},
                checks=[
                    Check(
                        type=CheckType.CONTAINS,
                        arguments={
                            "text": "$.output.value.logs[0]",
                            "phrases": "$.test_case.expected.search_terms",
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "logs": [
                    "System error occurred during processing",
                    "Retry mechanism activated",
                    "Process completed",
                ],
            }),
        ]

        results = evaluate(test_cases, outputs)
        # Should fail because text contains "error" but not "failed" (ContainsCheck requires ALL phrases)  # noqa: E501
        assert results.results[0].check_results[0].results == {"passed": False}

    def test_contains_complex_phrases_structure(self):
        """Test contains with complex phrase structures from JSONPath."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected={
                    "keywords": ["machine learning", "artificial intelligence", "neural networks"],
                },
                checks=[
                    Check(
                        type=CheckType.CONTAINS,
                        arguments={
                            "text": "$.output.value.content",
                            "phrases": "$.test_case.expected.keywords",
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "content": "This article discusses machine learning and artificial intelligence, particularly focusing on neural networks and deep learning architectures.",  # noqa: E501
                "category": "tech",
            }),
        ]

        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}
