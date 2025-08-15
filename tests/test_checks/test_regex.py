"""Comprehensive tests for RegexCheck implementation.

This module consolidates all tests for the RegexCheck including:
- Pydantic validation tests (from test_schema_check_classes.py)
- Implementation execution tests (from test_standard_checks.py) 
- Engine integration tests
- Edge cases and error handling

Tests are organized by functionality rather than implementation details.
"""

import pytest
from typing import Any

from flex_evals.checks.regex import RegexCheck, RegexFlags
from flex_evals import CheckType, Status, evaluate, Check, Output, TestCase
from flex_evals.exceptions import ValidationError
from pydantic import ValidationError as PydanticValidationError


class TestRegexValidation:
    """Test Pydantic validation and field handling for RegexCheck."""

    def test_regex_check_creation(self):
        """Test basic RegexCheck creation."""
        check = RegexCheck(
            text="$.output.value",
            pattern=r"\d+",
        )

        assert check.text == "$.output.value"
        assert check.pattern == r"\d+"
        assert check.negate is False
        assert isinstance(check.flags, RegexFlags)

    def test_regex_check_with_flags(self):
        """Test RegexCheck with flags."""
        flags = RegexFlags(case_insensitive=True, multiline=True)
        check = RegexCheck(
            text="$.output.value",
            pattern="hello",
            negate=True,
            flags=flags,
        )

        assert check.flags.case_insensitive is True
        assert check.flags.multiline is True
        assert check.flags.dot_all is False
        assert check.negate is True

    def test_regex_check_validation_empty_text(self):
        """Test RegexCheck validation for empty text - now allowed."""
        check = RegexCheck(text="", pattern="test")
        assert check.text == ""
        assert check.pattern == "test"

    def test_regex_check_validation_empty_pattern(self):
        """Test RegexCheck validation for empty pattern - now allowed."""
        check = RegexCheck(text="$.value", pattern="")
        assert check.text == "$.value"
        assert check.pattern == ""

    def test_regex_check_type_property(self):
        """Test RegexCheck check_type property returns correct type."""
        check = RegexCheck(text="test", pattern="test")
        assert check.check_type == CheckType.REGEX

    def test_regex_check_pattern_jsonpath(self):
        """Test RegexCheck with JSONPath pattern."""
        check = RegexCheck(
            text="$.output.value",
            pattern="$.expected.pattern",
        )

        assert check.text == "$.output.value"
        assert check.pattern == "$.expected.pattern"
        assert check.negate is False


class TestRegexExecution:
    """Test RegexCheck execution logic and __call__ method."""

    def test_regex_basic_match(self):
        """Test simple pattern matching."""
        check = RegexCheck(text="test", pattern="test")
        result = check(
            text="user@example.com",
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        )
        assert result == {"passed": True}

    def test_regex_no_match(self):
        """Test pattern that doesn't match."""
        check = RegexCheck(text="test", pattern="test")
        result = check(
            text="not an email",
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        )
        assert result == {"passed": False}

    def test_regex_case_insensitive(self):
        """Test case_insensitive flag."""
        check = RegexCheck(text="test", pattern="test")
        result = check(
            text="Hello World",
            pattern="hello",
            flags={"case_insensitive": True},
        )
        assert result == {"passed": True}

    def test_regex_multiline(self):
        """Test multiline flag with ^ and $ anchors."""
        text = "First line\nSecond line\nThird line"
        check = RegexCheck(text="test", pattern="test")
        result = check(
            text=text,
            pattern="^Second",
            flags={"multiline": True},
        )

        assert result == {"passed": True}

    def test_regex_dot_all(self):
        """Test dot_all flag with . matching newlines."""
        text = "First line\nSecond line"
        check = RegexCheck(text="test", pattern="test")
        result = check(
            text=text,
            pattern="First.*Second",
            flags={"dot_all": True},
        )
        assert result == {"passed": True}

    def test_regex_negate_true(self):
        """Test negate=true passes when pattern doesn't match."""
        check = RegexCheck(text="test", pattern="test")
        result = check(
            text="not an email",
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            negate=True,
        )
        assert result == {"passed": True}

    def test_regex_complex_pattern(self):
        """Test complex regex with groups, quantifiers."""
        check = RegexCheck(text="test", pattern="test")
        result = check(
            text="Phone: (555) 123-4567",
            pattern=r"Phone: \((\d{3})\) (\d{3})-(\d{4})",
        )
        assert result == {"passed": True}

    def test_regex_invalid_pattern(self):
        """Test invalid regex pattern raises appropriate error."""
        check = RegexCheck(text="test", pattern="test")
        with pytest.raises(ValidationError, match="Invalid regex pattern"):
            check(
                text="test text",
                pattern="[invalid",  # Unclosed bracket
            )

    def test_regex_empty_text(self):
        """Test pattern matching against empty string."""
        check = RegexCheck(text="test", pattern="test")
        result = check(
            text="",
            pattern="^$",  # Match empty string
        )
        assert result == {"passed": True}

    def test_regex_invalid_pattern_type(self):
        """Test non-string pattern raises ValueError."""
        check = RegexCheck(text="test", pattern="test")
        with pytest.raises(ValidationError, match="must be a string"):
            check(text="test", pattern=123)


class TestRegexEngineIntegration:
    """Test RegexCheck integration with the evaluation engine."""

    @pytest.mark.parametrize(("output_value", "pattern", "expected_passed"), [
        ("user@example.com", r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", True),  # Valid
        ("not-an-email", r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", False),  # Invalid
        ("Hello World", r"hello", False),  # Case sensitive by default
        ("Phone: (555) 123-4567", r"Phone: \((\d{3})\) (\d{3})-(\d{4})", True),  # Phone match
        ("", r"^$", True),  # Empty string matches empty pattern
    ])
    def test_regex_via_evaluate(
        self, output_value: str, pattern: str, expected_passed: bool,
    ):
        """Test using JSONPath for text and pattern with various combinations."""
        test_cases = [
            TestCase(
                id="test_001",
                input="What is your email?",
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
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": expected_passed}

    def test_regex_check_instance_via_evaluate(self):
        """Test direct check instance usage in evaluate function."""
        test_cases = [
            TestCase(
                id="test_001",
                input="What is your email?",
                expected=r"\w+@\w+\.\w+",
                checks=[
                    RegexCheck(
                        text="$.output.value",
                        pattern="$.test_case.expected",
                    ),
                ],
            ),
        ]
        
        outputs = [Output(value="user@example.com")]
        results = evaluate(test_cases, outputs)
        
        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": True}


class TestRegexErrorHandling:
    """Test error handling and edge cases for RegexCheck."""

    def test_regex_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(PydanticValidationError):
            RegexCheck()  # type: ignore
        
        with pytest.raises(PydanticValidationError):
            RegexCheck(text="test")  # type: ignore
        
        with pytest.raises(PydanticValidationError):
            RegexCheck(pattern="test")  # type: ignore

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
                            "pattern": "test",
                        },
                    ),
                ],
            ),
        ]
        
        outputs = [Output(value="test")]
        
        with pytest.raises(ValidationError, match="appears to be JSONPath but is invalid"):
            evaluate(test_cases, outputs)