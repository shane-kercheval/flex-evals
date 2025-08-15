"""
Comprehensive tests for RegexCheck implementation.

This module consolidates all tests for the RegexCheck including:
- Pydantic validation tests
- Implementation execution tests
- Engine integration tests
- Edge cases and error handling

Tests are organized by functionality rather than implementation details.
"""

import pytest

from flex_evals import (
    RegexCheck,
    RegexFlags,
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


class TestRegexValidation:
    """Test Pydantic validation and field handling for RegexCheck."""

    def test_regex_check_creation(self):
        """Test basic RegexCheck creation."""
        check = RegexCheck(
            text="$.output.value",
            pattern="$.expected_pattern",
        )

        assert isinstance(check.text, JSONPath)
        assert check.text.expression == "$.output.value"
        assert isinstance(check.pattern, JSONPath)
        assert check.pattern.expression == "$.expected_pattern"
        assert check.negate is False
        assert isinstance(check.flags, RegexFlags)

    def test_regex_check_with_literal_values(self):
        """Test RegexCheck with literal text and pattern."""
        check = RegexCheck(
            text="test123",
            pattern="\\d+",
            negate=True,
        )

        assert check.text == "test123"
        assert check.pattern == "\\d+"
        assert check.negate is True

    def test_regex_jsonpath_comprehensive(self):
        """Comprehensive JSONPath string conversion and execution test."""
        # 1. Create check with all JSONPath fields as strings
        check = RegexCheck(
            text="$.output.value.message",
            pattern="$.test_case.expected.regex_pattern",
            negate="$.test_case.expected.should_negate",
            flags="$.test_case.expected.regex_flags",
        )

        # 2. Verify conversion happened
        assert isinstance(check.text, JSONPath)
        assert check.text.expression == "$.output.value.message"
        assert isinstance(check.pattern, JSONPath)
        assert check.pattern.expression == "$.test_case.expected.regex_pattern"
        assert isinstance(check.negate, JSONPath)
        assert check.negate.expression == "$.test_case.expected.should_negate"
        assert isinstance(check.flags, JSONPath)
        assert check.flags.expression == "$.test_case.expected.regex_flags"

        # 3. Test execution with EvaluationContext

        test_case = TestCase(
            id="test_001",
            input="test",
            expected={
                "regex_pattern": r"\d+",  # Match digits
                "should_negate": False,
                "regex_flags": RegexFlags(),  # Default RegexFlags instance
            },
        )
        output = Output(value={"message": "Score: 95 points"})
        context = EvaluationContext(test_case, output)

        result = check.execute(context)
        assert result.status == "completed"
        assert result.results["passed"] is True  # Contains digits
        assert result.resolved_arguments["text"]["value"] == "Score: 95 points"
        assert result.resolved_arguments["pattern"]["value"] == r"\d+"
        assert result.resolved_arguments["negate"]["value"] is False

        # 4. Test invalid JSONPath string (should raise exception during validation)
        with pytest.raises(PydanticValidationError, match="Invalid JSONPath expression"):
            RegexCheck(text="$.invalid[", pattern="test")

        # 5. Test valid literal values (should work)
        check_literal = RegexCheck(
            text="literal text",  # str | JSONPath - string literal works
            pattern="literal_pattern",  # str | JSONPath - string literal works
            negate=False,  # bool | JSONPath - boolean literal works
            flags=RegexFlags(),  # RegexFlags | JSONPath - RegexFlags instance works
        )
        assert check_literal.text == "literal text"
        assert check_literal.pattern == "literal_pattern"
        assert check_literal.negate is False
        assert isinstance(check_literal.flags, RegexFlags)

        # 6. Test invalid non-JSONPath strings for fields that don't support string
        # negate: bool | JSONPath - "not_a_boolean" is neither bool nor JSONPath
        with pytest.raises(PydanticValidationError):
            RegexCheck(text="test", pattern="test", negate="not_a_boolean")

        # flags: RegexFlags | JSONPath - "not_flags" is neither RegexFlags nor JSONPath
        with pytest.raises(PydanticValidationError):
            RegexCheck(text="test", pattern="test", flags="not_flags")

    def test_regex_check_with_flags(self):
        """Test RegexCheck with custom flags."""
        flags = RegexFlags(
            case_insensitive=True,
            multiline=True,
            dot_all=False,
        )
        check = RegexCheck(
            text="$.output.value",
            pattern="[A-Z]+",
            flags=flags,
        )

        assert check.flags.case_insensitive is True
        assert check.flags.multiline is True
        assert check.flags.dot_all is False

    def test_regex_check_type_property(self):
        """Test RegexCheck check_type property returns correct type."""
        check = RegexCheck(text="test", pattern="\\w+")
        assert check.check_type == CheckType.REGEX

    def test_regex_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(PydanticValidationError):
            RegexCheck()  # type: ignore

        with pytest.raises(PydanticValidationError):
            RegexCheck(text="test")  # type: ignore

        with pytest.raises(PydanticValidationError):
            RegexCheck(pattern="\\d+")  # type: ignore


class TestRegexExecution:
    """Test RegexCheck execution logic and __call__ method."""

    def test_regex_basic_match(self):
        """Test basic regex pattern matching."""
        check = RegexCheck(text="test123", pattern="\\d+")
        result = check()
        assert result == {"passed": True}

    def test_regex_no_match(self):
        """Test regex pattern with no match."""
        check = RegexCheck(text="testABC", pattern="\\d+")
        result = check()
        assert result == {"passed": False}

    def test_regex_case_sensitive_default(self):
        """Test case sensitive matching (default)."""
        check = RegexCheck(text="Hello World", pattern="hello")
        result = check()
        assert result == {"passed": False}  # Case sensitive by default

    def test_regex_case_insensitive(self):
        """Test case insensitive matching."""
        flags = RegexFlags(case_insensitive=True)
        check = RegexCheck(text="Hello World", pattern="hello", flags=flags)
        result = check()
        assert result == {"passed": True}

    def test_regex_multiline(self):
        """Test multiline matching."""
        flags = RegexFlags(multiline=True)
        text = "line1\nline2\nline3"
        check = RegexCheck(text=text, pattern="^line2$", flags=flags)
        result = check()
        assert result == {"passed": True}

    def test_regex_dot_all(self):
        """Test dot_all flag (. matches newlines)."""
        flags = RegexFlags(dot_all=True)
        text = "line1\nline2"
        check = RegexCheck(text=text, pattern="line1.line2", flags=flags)
        result = check()
        assert result == {"passed": True}

    def test_regex_negate_true(self):
        """Test negate=true passes when pattern doesn't match."""
        check = RegexCheck(text="hello world", pattern="\\d+", negate=True)
        result = check()
        assert result == {"passed": True}  # No digits, negated

    def test_regex_negate_false_with_match(self):
        """Test negate=true fails when pattern matches."""
        check = RegexCheck(text="hello123", pattern="\\d+", negate=True)
        result = check()
        assert result == {"passed": False}  # Has digits, negated

    def test_regex_complex_pattern(self):
        """Test complex regex pattern."""
        check = RegexCheck(
            text="user@example.com",
            pattern="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
        )
        result = check()
        assert result == {"passed": True}

    def test_regex_empty_text(self):
        """Test regex with empty text."""
        check = RegexCheck(text="", pattern=".*")
        result = check()
        assert result == {"passed": True}  # .* matches empty string

    def test_regex_invalid_pattern_error(self):
        """Test invalid regex pattern raises error during execution."""
        check = RegexCheck(text="test", pattern="[invalid")
        with pytest.raises(ValidationError, match="Invalid regex pattern"):
            check()

    def test_regex_anchors_and_boundaries(self):
        """Test regex anchors and word boundaries."""
        # Start anchor
        check_start = RegexCheck(text="hello world", pattern="^hello")
        assert check_start() == {"passed": True}

        # End anchor
        check_end = RegexCheck(text="hello world", pattern="world$")
        assert check_end() == {"passed": True}

        # Word boundary
        check_boundary = RegexCheck(text="hello world", pattern="\\bworld\\b")
        assert check_boundary() == {"passed": True}

    def test_regex_special_characters(self):
        """Test regex with special characters."""
        check = RegexCheck(text="Price: $25.99", pattern="\\$\\d+\\.\\d{2}")
        result = check()
        assert result == {"passed": True}


class TestRegexEngineIntegration:
    """Test RegexCheck integration with the evaluation engine."""

    @pytest.mark.parametrize(("output_value", "pattern", "expected_passed"), [
        ("test123", "\\d+", True),        # Contains digits
        ("testABC", "\\d+", False),       # No digits
        ("Hello", "^[A-Z]", True),        # Starts with capital
        ("hello", "^[A-Z]", False),       # Doesn't start with capital
        ("user@test.com", "@", True),     # Contains @
        ("user.test.com", "@", False),    # No @
    ])
    def test_regex_via_evaluate(
        self, output_value: str, pattern: str, expected_passed: bool,
    ):
        """Test using JSONPath for text and pattern with various combinations."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test input",
                expected=pattern,
                checks=[
                    Check(
                        type=CheckType.REGEX,
                        arguments={
                            "text": "$.output.value",
                            "pattern": "$.test_case.expected",
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

    def test_regex_check_instance_via_evaluate(self):
        """Test direct check instance usage in evaluate function."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test input",
                checks=[
                    RegexCheck(
                        text="$.output.value.message",
                        pattern="\\berror\\b",  # Word "error"
                    ),
                ],
            ),
        ]

        outputs = [Output(value={"message": "An error occurred during processing"})]
        results = evaluate(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_regex_flags_via_evaluate(self):
        """Test regex flags through engine evaluation."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test input",
                checks=[
                    Check(
                        type=CheckType.REGEX,
                        arguments={
                            "text": "$.output.value",
                            "pattern": "HELLO",
                            "flags": {
                                "case_insensitive": True,
                                "multiline": False,
                                "dot_all": False,
                            },
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="hello world")]  # Lowercase
        results = evaluate(test_cases, outputs)

        # Should pass due to case_insensitive=True
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_regex_negate_via_evaluate(self):
        """Test negation through engine evaluation."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test input",
                checks=[
                    Check(
                        type=CheckType.REGEX,
                        arguments={
                            "text": "$.output.value",
                            "pattern": "\\d+",  # Digits
                            "negate": True,     # Should pass if no digits
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="hello world")]  # No digits
        results = evaluate(test_cases, outputs)

        assert results.results[0].check_results[0].results == {"passed": True}


class TestRegexErrorHandling:
    """Test error handling and edge cases for RegexCheck."""

    def test_regex_jsonpath_validation_in_engine(self):
        """Test that invalid JSONPath expressions are caught during evaluation."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.REGEX,
                        arguments={
                            "text": "$..[invalid",  # Invalid JSONPath syntax
                            "pattern": "\\d+",
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="test")]

        # Should raise validation error for invalid JSONPath
        with pytest.raises(ValidationError, match="Invalid JSONPath expression"):
            evaluate(test_cases, outputs)

    def test_regex_missing_jsonpath_data(self):
        """Test behavior when JSONPath doesn't find data."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.REGEX,
                        arguments={
                            "text": "$.output.value.nonexistent",
                            "pattern": "\\d+",
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

    def test_regex_invalid_pattern_in_execution(self):
        """Test invalid regex pattern during execution."""
        check = RegexCheck(text="test text", pattern="[unclosed")
        with pytest.raises(ValidationError, match="Invalid regex pattern"):
            check()

    def test_regex_pattern_compilation_errors(self):
        """Test various regex compilation errors."""
        patterns_and_errors = [
            ("[unclosed", "Invalid regex pattern"),
            ("(?P<incomplete", "Invalid regex pattern"),
            ("*invalid", "Invalid regex pattern"),
            ("+invalid", "Invalid regex pattern"),
        ]

        for pattern, error_match in patterns_and_errors:
            check = RegexCheck(text="test", pattern=pattern)
            with pytest.raises(ValidationError, match=error_match):
                check()


class TestRegexJSONPathIntegration:
    """Test RegexCheck with various JSONPath expressions and data structures."""

    def test_regex_nested_jsonpath(self):
        """Test regex with deeply nested JSONPath expressions."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected={"patterns": {"email": "@\\w+\\.(com|org|net)"}},
                checks=[
                    Check(
                        type=CheckType.REGEX,
                        arguments={
                            "text": "$.output.value.user.email",
                            "pattern": "$.test_case.expected.patterns.email",
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "user": {
                    "email": "alice@example.com",
                    "name": "Alice Smith",
                },
                "timestamp": "2024-01-01",
            }),
        ]

        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_regex_array_access_jsonpath(self):
        """Test regex with JSONPath array access."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.REGEX,
                        arguments={
                            "text": "$.output.value.messages[0].content",
                            "pattern": "\\b(success|completed)\\b",
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "messages": [
                    {"content": "Operation completed successfully", "level": "info"},
                    {"content": "Processing started", "level": "debug"},
                ],
            }),
        ]

        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_regex_complex_text_processing(self):
        """Test regex with complex text processing scenarios."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.REGEX,
                        arguments={
                            "text": "$.output.value.log_entry",
                            "pattern": "\\[(\\d{4}-\\d{2}-\\d{2})\\s(\\d{2}:\\d{2}:\\d{2})\\]\\s(ERROR|WARN|INFO)",  # noqa: E501
                            "flags": {
                                "case_insensitive": False,
                                "multiline": True,
                                "dot_all": False,
                            },
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "log_entry": "[2024-01-15 14:30:22] ERROR Failed to process request",
                "source": "application.log",
            }),
        ]

        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_regex_unicode_and_special_characters(self):
        """Test regex with Unicode and special characters."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.REGEX,
                        arguments={
                            "text": "$.output.value.message",
                            "pattern": "[\\u00C0-\\u017F]+",  # Latin extended characters
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "message": "Café résumé naïve",
                "language": "french",
            }),
        ]

        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}
