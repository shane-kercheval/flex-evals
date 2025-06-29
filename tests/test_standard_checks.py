"""Tests for standard check implementations."""

import pytest

from flex_evals.checks.standard.exact_match import ExactMatchCheck
from flex_evals.checks.standard.contains import ContainsCheck
from flex_evals.checks.standard.regex import RegexCheck
from flex_evals.checks.standard.threshold import ThresholdCheck
from flex_evals.constants import CheckType, Status
from flex_evals.engine import evaluate
from flex_evals.exceptions import ValidationError
from flex_evals.schemas.check import Check
from flex_evals.schemas.output import Output
from flex_evals.schemas.test_case import TestCase


class TestExactMatchCheck:
    """Test ExactMatch check implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.check = ExactMatchCheck()

    def test_exact_match_string_equal(self):
        """Test matching strings return passed=true."""
        result = self.check(actual="Paris", expected="Paris")

        assert result == {"passed": True}

    def test_exact_match_string_not_equal(self):
        """Test non-matching strings return passed=false."""
        result = self.check(actual="paris", expected="Paris")

        assert result == {"passed": False}

    def test_exact_match_case_sensitive_true(self):
        """Test 'Hello' != 'hello' when case_sensitive=true."""
        result = self.check(actual="Hello", expected="hello", case_sensitive=True)

        assert result == {"passed": False}

    def test_exact_match_case_sensitive_false(self):
        """Test 'Hello' == 'hello' when case_sensitive=false."""
        result = self.check(actual="Hello", expected="hello", case_sensitive=False)

        assert result == {"passed": True}

    def test_exact_match_negate_true(self):
        """Test negate=true passes when values differ."""
        result = self.check(actual="Paris", expected="London", negate=True)

        assert result == {"passed": True}

    def test_exact_match_negate_false(self):
        """Test negate=false passes when values match."""
        result = self.check(actual="Paris", expected="Paris", negate=False)

        assert result == {"passed": True}

    def test_exact_match_object_comparison(self):
        """Test comparing complex objects."""
        # Objects will be converted to strings for comparison
        result = self.check(actual={"city": "Paris"}, expected="{'city': 'Paris'}")

        assert result == {"passed": True}

    def test_exact_match_null_values(self):
        """Test comparison with null/None values."""
        result = self.check(actual=None, expected="")

        assert result == {"passed": True}  # None converts to empty string

    def test_exact_match_missing_actual(self):
        """Test missing actual argument raises TypeError."""
        with pytest.raises(TypeError):
            self.check(expected="Paris")

    def test_exact_match_missing_expected(self):
        """Test missing expected argument raises TypeError."""
        with pytest.raises(TypeError):
            self.check(actual="Paris")

    def test_exact_match_result_schema(self):
        r"""Test result matches {\"passed\": boolean} exactly."""
        result = self.check(actual="test", expected="test")

        assert isinstance(result, dict)
        assert set(result.keys()) == {"passed"}
        assert isinstance(result["passed"], bool)

    def test_exact_match_via_evaluate(self):
        """Test using JSONPath for actual and expected values."""
        # Define your test cases
        test_cases = [
            TestCase(
                id="test_001",
                input="What is the capital of France?",
                expected="Paris",
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
            Output(value="Paris"),
        ]
        # Run evaluation
        results = evaluate(test_cases, outputs)
        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.summary.error_test_cases == 0
        assert results.summary.skipped_test_cases == 0
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": True}


class TestContainsCheck:
    """Test Contains check implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.check = ContainsCheck()

    def test_contains_all_phrases_found(self):
        """Test negate=false passes when all phrases present."""
        result = self.check(
            text="Paris is the capital of France",
            phrases=["Paris", "France"],
            negate=False,
        )

        assert result == {"passed": True}

    def test_contains_some_phrases_missing(self):
        """Test negate=false fails when any phrase missing."""
        result = self.check(
            text="Paris is the capital of France",
            phrases=["Paris", "Spain"],  # Spain is missing
            negate=False,
        )

        assert result == {"passed": False}

    def test_contains_negate_none_found(self):
        """Test negate=true passes when no phrases found."""
        result = self.check(
            text="Paris is the capital of France",
            phrases=["London", "Spain"],
            negate=True,
        )

        assert result == {"passed": True}

    def test_contains_negate_some_found(self):
        """Test negate=true fails when any phrase found."""
        result = self.check(
            text="Paris is the capital of France",
            phrases=["Paris", "Spain"],  # Paris is found
            negate=True,
        )

        assert result == {"passed": False}

    def test_contains_case_sensitive(self):
        """Test case sensitivity in phrase matching."""
        result = self.check(
            text="Paris is the capital of France",
            phrases=["paris"],  # Lowercase
            case_sensitive=True,
        )

        assert result == {"passed": False}

    def test_contains_case_insensitive(self):
        """Test case insensitive matching."""
        result = self.check(
            text="Paris is the capital of France",
            phrases=["paris"],  # Lowercase
            case_sensitive=False,
        )

        assert result == {"passed": True}

    def test_contains_empty_phrases(self):
        """Test behavior with empty phrases array."""
        with pytest.raises(ValidationError, match="must not be empty"):
            self.check(text="test text", phrases=[])

    def test_contains_single_phrase(self):
        """Test with single phrase in array."""
        result = self.check(
            text="Paris is the capital of France",
            phrases=["capital"],
        )

        assert result == {"passed": True}

    def test_contains_overlapping_phrases(self):
        """Test with overlapping/duplicate phrases."""
        result = self.check(
            text="Paris Paris is great",
            phrases=["Paris", "Paris"],  # Duplicate phrase
        )

        assert result == {"passed": True}

    def test_contains_missing_text(self):
        """Test missing text argument raises TypeError."""
        with pytest.raises(TypeError):
            self.check(phrases=["test"])

    def test_contains_missing_phrases(self):
        """Test missing phrases argument raises TypeError."""
        with pytest.raises(TypeError):
            self.check(text="test text")

    def test_contains_invalid_phrases_type(self):
        """Test non-list phrases argument raises ValueError."""
        with pytest.raises(ValidationError, match="must be a list"):
            self.check(text="test", phrases="not a list")

    def test_contains_via_evaluate(self):
        """Test using JSONPath for text and phrases."""
        # Define your test cases
        test_cases = [
            TestCase(
                id="test_001",
                input="What is the capital of France?",
                expected=["Paris", "France"],
                checks=[
                    Check(
                        type=CheckType.CONTAINS,  # Can also use 'exact_match' string
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
            Output(value="The capital of France is Paris."),
        ]
        # Run evaluation
        results = evaluate(test_cases, outputs)
        print(results)
        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.summary.error_test_cases == 0
        assert results.summary.skipped_test_cases == 0
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": True}


class TestRegexCheck:
    """Test Regex check implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.check = RegexCheck()

    def test_regex_basic_match(self):
        """Test simple pattern matching."""
        result = self.check(
            text="user@example.com",
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        )

        assert result == {"passed": True}

    def test_regex_no_match(self):
        """Test pattern that doesn't match."""
        result = self.check(
            text="not an email",
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        )

        assert result == {"passed": False}

    def test_regex_case_insensitive(self):
        """Test case_insensitive flag."""
        result = self.check(
            text="Hello World",
            pattern="hello",
            flags={"case_insensitive": True},
        )

        assert result == {"passed": True}

    def test_regex_multiline(self):
        """Test multiline flag with ^ and $ anchors."""
        text = "First line\nSecond line\nThird line"
        result = self.check(
            text=text,
            pattern="^Second",
            flags={"multiline": True},
        )

        assert result == {"passed": True}

    def test_regex_dot_all(self):
        """Test dot_all flag with . matching newlines."""
        text = "First line\nSecond line"
        result = self.check(
            text=text,
            pattern="First.*Second",
            flags={"dot_all": True},
        )

        assert result == {"passed": True}

    def test_regex_negate_true(self):
        """Test negate=true passes when pattern doesn't match."""
        result = self.check(
            text="not an email",
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            negate=True,
        )

        assert result == {"passed": True}

    def test_regex_complex_pattern(self):
        """Test complex regex with groups, quantifiers."""
        result = self.check(
            text="Phone: (555) 123-4567",
            pattern=r"Phone: \((\d{3})\) (\d{3})-(\d{4})",
        )

        assert result == {"passed": True}

    def test_regex_invalid_pattern(self):
        """Test invalid regex pattern raises appropriate error."""
        with pytest.raises(ValidationError, match="Invalid regex pattern"):
            self.check(
                text="test text",
                pattern="[invalid",  # Unclosed bracket
            )

    def test_regex_empty_text(self):
        """Test pattern matching against empty string."""
        result = self.check(
            text="",
            pattern="^$",  # Match empty string
        )

        assert result == {"passed": True}

    def test_regex_missing_text(self):
        """Test missing text argument raises TypeError."""
        with pytest.raises(TypeError):
            self.check(pattern="test")

    def test_regex_missing_pattern(self):
        """Test missing pattern argument raises TypeError."""
        with pytest.raises(TypeError):
            self.check(text="test")

    def test_regex_invalid_pattern_type(self):
        """Test non-string pattern raises ValueError."""
        with pytest.raises(ValidationError, match="must be a string"):
            self.check(text="test", pattern=123)

    def test_regex_via_evaluate(self):
        """Test using JSONPath for text and pattern."""
        # Define your test cases
        test_cases = [
            TestCase(
                id="test_001",
                input="What is your email?",
                expected=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
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
        # System outputs to evaluate
        outputs = [
            Output(value="user@example.com"),
        ]
        # Run evaluation
        results = evaluate(test_cases, outputs)
        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.summary.error_test_cases == 0
        assert results.summary.skipped_test_cases == 0
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": True}


class TestThresholdCheck:
    """Test Threshold check implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.check = ThresholdCheck()

    def test_threshold_min_only_pass(self):
        """Test value >= min_value passes."""
        result = self.check(value=0.85, min_value=0.8)

        assert result == {"passed": True}

    def test_threshold_min_only_fail(self):
        """Test value < min_value fails."""
        result = self.check(value=0.75, min_value=0.8)

        assert result == {"passed": False}

    def test_threshold_max_only_pass(self):
        """Test value <= max_value passes."""
        result = self.check(value=0.85, max_value=1.0)

        assert result == {"passed": True}

    def test_threshold_max_only_fail(self):
        """Test value > max_value fails."""
        result = self.check(value=1.2, max_value=1.0)

        assert result == {"passed": False}

    def test_threshold_range_inside(self):
        """Test value within min and max passes."""
        result = self.check(value=0.85, min_value=0.8, max_value=1.0)

        assert result == {"passed": True}

    def test_threshold_range_outside(self):
        """Test value outside range fails."""
        result = self.check(value=1.2, min_value=0.8, max_value=1.0)

        assert result == {"passed": False}

    def test_threshold_min_exclusive(self):
        """Test min_inclusive=false excludes boundary."""
        result = self.check(value=0.8, min_value=0.8, min_inclusive=False)

        assert result == {"passed": False}

        # But greater than boundary should pass
        result = self.check(value=0.81, min_value=0.8, min_inclusive=False)

        assert result == {"passed": True}

    def test_threshold_max_exclusive(self):
        """Test max_inclusive=false excludes boundary."""
        result = self.check(value=1.0, max_value=1.0, max_inclusive=False)

        assert result == {"passed": False}

        # But less than boundary should pass
        result = self.check(value=0.99, max_value=1.0, max_inclusive=False)

        assert result == {"passed": True}

    def test_threshold_negate_outside(self):
        """Test negate=true passes when outside bounds."""
        result = self.check(value=1.2, min_value=0.8, max_value=1.0, negate=True)

        assert result == {"passed": True}

    def test_threshold_negate_inside(self):
        """Test negate=true fails when inside bounds."""
        result = self.check(value=0.9, min_value=0.8, max_value=1.0, negate=True)

        assert result == {"passed": False}

    def test_threshold_no_bounds_error(self):
        """Test error when neither min nor max specified."""
        with pytest.raises(ValidationError, match="requires at least one of"):
            self.check(value=0.85)

    def test_threshold_non_numeric_error(self):
        """Test error when value is not numeric."""
        with pytest.raises(ValidationError, match="must be numeric"):
            self.check(value="not a number", min_value=0.8)

    def test_threshold_string_numeric_conversion(self):
        """Test numeric string conversion."""
        result = self.check(value="0.85", min_value=0.8)

        assert result == {"passed": True}

    def test_threshold_missing_value(self):
        """Test missing value argument raises TypeError."""
        with pytest.raises(TypeError):
            self.check(min_value=0.8)

    def test_threshold_invalid_min_value_type(self):
        """Test non-numeric min_value raises ValidationError."""
        with pytest.raises(ValidationError, match="'min_value' must be numeric"):
            self.check(value=0.85, min_value="not numeric")

    def test_threshold_invalid_max_value_type(self):
        """Test non-numeric max_value raises ValidationError."""
        with pytest.raises(ValidationError, match="'max_value' must be numeric"):
            self.check(value=0.85, max_value="not numeric")

    def test_threshold_via_evaluate(self):
        """Test using JSONPath for value and thresholds."""
        # Define your test cases
        test_cases = [
            TestCase(
                id="test_001",
                input="What is the confidence score?",
                expected={"min": 0.8, "max": 1.0},
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
        # System outputs to evaluate
        outputs = [
            Output(value={"message": "High confidence", "confidence": 0.95}),
        ]
        # Run evaluation
        results = evaluate(test_cases, outputs)
        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.summary.error_test_cases == 0
        assert results.summary.skipped_test_cases == 0
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": True}
