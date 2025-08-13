"""Tests for standard check implementations."""

import pytest

from flex_evals.checks.standard.exact_match import ExactMatchCheck_v1_0_0
from flex_evals.checks.standard.contains import ContainsCheck_v1_0_0
from flex_evals.checks.standard.is_empty import IsEmptyCheck_v1_0_0
from flex_evals.checks.standard.regex import RegexCheck_v1_0_0
from flex_evals.checks.standard.threshold import ThresholdCheck_v1_0_0
from flex_evals import CheckType, Status, evaluate, Check, Output, TestCase
from flex_evals.exceptions import ValidationError
from flex_evals.registry import list_registered_checks


class TestExactMatchCheck:
    """Test ExactMatch check implementation."""

    def test_exact_match_string_equal(self):
        """Test matching strings return passed=true."""
        result = ExactMatchCheck_v1_0_0()(actual="Paris", expected="Paris")
        assert result == {"passed": True}

    def test_exact_match_string_not_equal(self):
        """Test non-matching strings return passed=false."""
        result = ExactMatchCheck_v1_0_0()(actual="paris", expected="Paris")
        assert result == {"passed": False}

    def test_exact_match_case_sensitive_true(self):
        """Test 'Hello' != 'hello' when case_sensitive=true."""
        result = ExactMatchCheck_v1_0_0()(actual="Hello", expected="hello", case_sensitive=True)
        assert result == {"passed": False}

    def test_exact_match_case_sensitive_false(self):
        """Test 'Hello' == 'hello' when case_sensitive=false."""
        result = ExactMatchCheck_v1_0_0()(actual="Hello", expected="hello", case_sensitive=False)
        assert result == {"passed": True}

    def test_exact_match_negate_true(self):
        """Test negate=true passes when values differ."""
        result = ExactMatchCheck_v1_0_0()(actual="Paris", expected="London", negate=True)
        assert result == {"passed": True}

    def test_exact_match_negate_false(self):
        """Test negate=false passes when values match."""
        result = ExactMatchCheck_v1_0_0()(actual="Paris", expected="Paris", negate=False)
        assert result == {"passed": True}

    def test_exact_match_object_comparison(self):
        """Test comparing complex objects."""
        # Objects will be converted to strings for comparison
        result = ExactMatchCheck_v1_0_0()(actual={"city": "Paris"}, expected="{'city': 'Paris'}")
        assert result == {"passed": True}

    def test_exact_match_null_values(self):
        """Test comparison with null/None values."""
        result = ExactMatchCheck_v1_0_0()(actual=None, expected="")
        assert result == {"passed": True}  # None converts to empty string

    def test_exact_match_missing_actual(self):
        """Test missing actual argument raises TypeError."""
        with pytest.raises(TypeError):
            ExactMatchCheck_v1_0_0()(expected="Paris")

    def test_exact_match_missing_expected(self):
        """Test missing expected argument raises TypeError."""
        with pytest.raises(TypeError):
            ExactMatchCheck_v1_0_0()(actual="Paris")

    def test_exact_match_result_schema(self):
        r"""Test result matches {\"passed\": boolean} exactly."""
        result = ExactMatchCheck_v1_0_0()(actual="test", expected="test")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"passed"}
        assert isinstance(result["passed"], bool)

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


class TestContainsCheck:
    """Test Contains check implementation."""

    def test_contains_all_phrases_found(self):
        """Test negate=false passes when all phrases present."""
        result = ContainsCheck_v1_0_0()(
            text="Paris is the capital of France",
            phrases=["Paris", "France"],
            negate=False,
        )
        assert result == {"passed": True}

    def test_contains_some_phrases_missing(self):
        """Test negate=false fails when any phrase missing."""
        result = ContainsCheck_v1_0_0()(
            text="Paris is the capital of France",
            phrases=["Paris", "Spain"],  # Spain is missing
            negate=False,
        )
        assert result == {"passed": False}

    def test_contains_negate_none_found(self):
        """Test negate=true passes when no phrases found."""
        result = ContainsCheck_v1_0_0()(
            text="Paris is the capital of France",
            phrases=["London", "Spain"],
            negate=True,
        )
        assert result == {"passed": True}

    def test_contains_negate_some_found(self):
        """Test negate=true fails when any phrase found."""
        result = ContainsCheck_v1_0_0()(
            text="Paris is the capital of France",
            phrases=["Paris", "Spain"],  # Paris is found
            negate=True,
        )
        assert result == {"passed": False}

    def test_contains_case_sensitive(self):
        """Test case sensitivity in phrase matching."""
        result = ContainsCheck_v1_0_0()(
            text="Paris is the capital of France",
            phrases=["paris"],  # Lowercase
            case_sensitive=True,
        )
        assert result == {"passed": False}

    def test_contains_case_insensitive(self):
        """Test case insensitive matching."""
        result = ContainsCheck_v1_0_0()(
            text="Paris is the capital of France",
            phrases=["paris"],  # Lowercase
            case_sensitive=False,
        )
        assert result == {"passed": True}

    def test_contains_empty_phrases(self):
        """Test behavior with empty phrases array."""
        with pytest.raises(ValidationError, match="must not be empty"):
            ContainsCheck_v1_0_0()(text="test text", phrases=[])

    def test_contains_single_phrase(self):
        """Test with single phrase in array."""
        result = ContainsCheck_v1_0_0()(
            text="Paris is the capital of France",
            phrases=["capital"],
        )
        assert result == {"passed": True}

    def test_contains_overlapping_phrases(self):
        """Test with overlapping/duplicate phrases."""
        result = ContainsCheck_v1_0_0()(
            text="Paris Paris is great",
            phrases=["Paris", "Paris"],  # Duplicate phrase
        )
        assert result == {"passed": True}

    def test_contains_missing_text(self):
        """Test missing text argument raises TypeError."""
        with pytest.raises(TypeError):
            ContainsCheck_v1_0_0()(phrases=["test"])

    def test_contains_missing_phrases(self):
        """Test missing phrases argument raises TypeError."""
        with pytest.raises(TypeError):
            ContainsCheck_v1_0_0()(text="test text")

    def test_contains_invalid_phrases_type(self):
        """Test non-list phrases argument raises ValueError."""
        with pytest.raises(ValidationError, match="must be a list"):
            ContainsCheck_v1_0_0()(text="test", phrases="not a list")

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


class TestRegexCheck:
    """Test Regex check implementation."""

    def test_regex_basic_match(self):
        """Test simple pattern matching."""
        result = RegexCheck_v1_0_0()(
            text="user@example.com",
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        )
        assert result == {"passed": True}

    def test_regex_no_match(self):
        """Test pattern that doesn't match."""
        result = RegexCheck_v1_0_0()(
            text="not an email",
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        )
        assert result == {"passed": False}

    def test_regex_case_insensitive(self):
        """Test case_insensitive flag."""
        result = RegexCheck_v1_0_0()(
            text="Hello World",
            pattern="hello",
            flags={"case_insensitive": True},
        )
        assert result == {"passed": True}

    def test_regex_multiline(self):
        """Test multiline flag with ^ and $ anchors."""
        text = "First line\nSecond line\nThird line"
        result = RegexCheck_v1_0_0()(
            text=text,
            pattern="^Second",
            flags={"multiline": True},
        )

        assert result == {"passed": True}

    def test_regex_dot_all(self):
        """Test dot_all flag with . matching newlines."""
        text = "First line\nSecond line"
        result = RegexCheck_v1_0_0()(
            text=text,
            pattern="First.*Second",
            flags={"dot_all": True},
        )
        assert result == {"passed": True}

    def test_regex_negate_true(self):
        """Test negate=true passes when pattern doesn't match."""
        result = RegexCheck_v1_0_0()(
            text="not an email",
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            negate=True,
        )
        assert result == {"passed": True}

    def test_regex_complex_pattern(self):
        """Test complex regex with groups, quantifiers."""
        result = RegexCheck_v1_0_0()(
            text="Phone: (555) 123-4567",
            pattern=r"Phone: \((\d{3})\) (\d{3})-(\d{4})",
        )
        assert result == {"passed": True}

    def test_regex_invalid_pattern(self):
        """Test invalid regex pattern raises appropriate error."""
        with pytest.raises(ValidationError, match="Invalid regex pattern"):
            RegexCheck_v1_0_0()(
                text="test text",
                pattern="[invalid",  # Unclosed bracket
            )

    def test_regex_empty_text(self):
        """Test pattern matching against empty string."""
        result = RegexCheck_v1_0_0()(
            text="",
            pattern="^$",  # Match empty string
        )
        assert result == {"passed": True}

    def test_regex_missing_text(self):
        """Test missing text argument raises TypeError."""
        with pytest.raises(TypeError):
            RegexCheck_v1_0_0()(pattern="test")

    def test_regex_missing_pattern(self):
        """Test missing pattern argument raises TypeError."""
        with pytest.raises(TypeError):
            RegexCheck_v1_0_0()(text="test")

    def test_regex_invalid_pattern_type(self):
        """Test non-string pattern raises ValueError."""
        with pytest.raises(ValidationError, match="must be a string"):
            RegexCheck_v1_0_0()(text="test", pattern=123)

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
        # Define your test cases
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


class TestIsEmptyCheck:
    """Test IsEmpty check implementation."""

    def test_is_empty_empty_string(self):
        """Test empty string passes is_empty check."""
        result = IsEmptyCheck_v1_0_0()(value="")
        assert result == {"passed": True}

    def test_is_empty_none_value(self):
        """Test None value passes is_empty check."""
        result = IsEmptyCheck_v1_0_0()(value=None)
        assert result == {"passed": True}

    def test_is_empty_whitespace_only_with_strip(self):
        """Test whitespace-only string passes when strip_whitespace=True."""
        result = IsEmptyCheck_v1_0_0()(value="   \t\n  ", strip_whitespace=True)
        assert result == {"passed": True}

    def test_is_empty_whitespace_only_without_strip(self):
        """Test whitespace-only string fails when strip_whitespace=False."""
        result = IsEmptyCheck_v1_0_0()(value="   \t\n  ", strip_whitespace=False)
        assert result == {"passed": False}

    def test_is_empty_non_empty_string(self):
        """Test non-empty string fails is_empty check."""
        result = IsEmptyCheck_v1_0_0()(value="hello")
        assert result == {"passed": False}

    def test_is_empty_string_with_content_and_whitespace(self):
        """Test string with content and whitespace fails is_empty check."""
        result = IsEmptyCheck_v1_0_0()(value="  hello  ")
        assert result == {"passed": False}

    def test_is_empty_negate_empty_string(self):
        """Test negate=True fails for empty string (not empty check)."""
        result = IsEmptyCheck_v1_0_0()(value="", negate=True)
        assert result == {"passed": False}

    def test_is_empty_negate_non_empty_string(self):
        """Test negate=True passes for non-empty string (not empty check)."""
        result = IsEmptyCheck_v1_0_0()(value="hello", negate=True)
        assert result == {"passed": True}

    def test_is_empty_negate_none_value(self):
        """Test negate=True fails for None value (not empty check)."""
        result = IsEmptyCheck_v1_0_0()(value=None, negate=True)
        assert result == {"passed": False}

    def test_is_empty_numeric_value(self):
        """Test numeric value is considered non-empty."""
        result = IsEmptyCheck_v1_0_0()(value=123)
        assert result == {"passed": False}

    def test_is_empty_zero_value(self):
        """Test zero value is considered non-empty."""
        result = IsEmptyCheck_v1_0_0()(value=0)
        assert result == {"passed": False}

    def test_is_empty_negative_value(self):
        """Test negative value is considered non-empty."""
        result = IsEmptyCheck_v1_0_0()(value=-1)
        assert result == {"passed": False}

    def test_is_empty_float_value(self):
        """Test float value is considered non-empty."""
        result = IsEmptyCheck_v1_0_0()(value=0.0)
        assert result == {"passed": False}

    def test_is_empty_boolean_false(self):
        """Test False value is considered non-empty."""
        result = IsEmptyCheck_v1_0_0()(value=False)
        assert result == {"passed": False}

    def test_is_empty_boolean_true(self):
        """Test True value is considered non-empty."""
        result = IsEmptyCheck_v1_0_0()(value=True)
        assert result == {"passed": False}

    def test_is_empty_list_empty(self):
        """Test empty list is considered non-empty."""
        result = IsEmptyCheck_v1_0_0()(value=[])
        assert result == {"passed": False}

    def test_is_empty_list_with_items(self):
        """Test list with items is considered non-empty."""
        result = IsEmptyCheck_v1_0_0()(value=[1, 2, 3])
        assert result == {"passed": False}

    def test_is_empty_dict_empty(self):
        """Test empty dict is considered non-empty."""
        result = IsEmptyCheck_v1_0_0()(value={})
        assert result == {"passed": False}

    def test_is_empty_dict_with_items(self):
        """Test dict with items is considered non-empty."""
        result = IsEmptyCheck_v1_0_0()(value={"key": "value"})
        assert result == {"passed": False}

    def test_is_empty_complex_type(self):
        """Test complex data structure is considered non-empty."""
        result = IsEmptyCheck_v1_0_0()(value={"nested": {"list": [1, 2, {"key": None}]}})
        assert result == {"passed": False}

    def test_is_empty_negate_with_numeric(self):
        """Test negate=True with numeric value passes (not empty check)."""
        result = IsEmptyCheck_v1_0_0()(value=123, negate=True)
        assert result == {"passed": True}

    def test_is_empty_negate_with_false(self):
        """Test negate=True with False value passes (not empty check)."""
        result = IsEmptyCheck_v1_0_0()(value=False, negate=True)
        assert result == {"passed": True}

    def test_is_empty_negate_with_empty_list(self):
        """Test negate=True with empty list passes (not empty check)."""
        result = IsEmptyCheck_v1_0_0()(value=[], negate=True)
        assert result == {"passed": True}

    def test_is_empty_missing_value(self):
        """Test missing value argument raises TypeError."""
        with pytest.raises(TypeError):
            IsEmptyCheck_v1_0_0()()

    def test_is_empty_result_schema(self):
        r"""Test result matches {\"passed\": boolean} exactly."""
        result = IsEmptyCheck_v1_0_0()(value="")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"passed"}
        assert isinstance(result["passed"], bool)

    @pytest.mark.parametrize(("output_value", "expected_passed"), [
        ("", True),
        ("not empty", False),
        ("   ", True),
        ("  text  ", False),
    ])
    def test_is_empty_via_evaluate(self, output_value: str, expected_passed: bool):
        """Test using JSONPath for value with various empty/non-empty combinations."""
        # Define your test cases
        test_cases = [
            TestCase(
                id="test_001",
                input="What is your name?",
                expected="",  # Empty expected value
                checks=[
                    Check(
                        type=CheckType.IS_EMPTY,
                        arguments={
                            "value": "$.output.value",
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

    @pytest.mark.parametrize(("output_value", "expected_passed"), [
        (None, True),
        (1, False),
    ])
    def test_none_empty_via_evaluate(self, output_value: str, expected_passed: bool):
        """Test using JSONPath for value with various empty/non-empty combinations."""
        # Define your test cases
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
        # System outputs to evaluate
        outputs = [
            Output(value={"value": output_value}),
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



class TestThresholdCheck:
    """Test Threshold check implementation."""

    def test_threshold_min_only_pass(self):
        """Test value >= min_value passes."""
        result = ThresholdCheck_v1_0_0()(value=0.85, min_value=0.8)
        assert result == {"passed": True}

    def test_threshold_min_only_fail(self):
        """Test value < min_value fails."""
        result = ThresholdCheck_v1_0_0()(value=0.75, min_value=0.8)
        assert result == {"passed": False}

    def test_threshold_max_only_pass(self):
        """Test value <= max_value passes."""
        result = ThresholdCheck_v1_0_0()(value=0.85, max_value=1.0)
        assert result == {"passed": True}

    def test_threshold_max_only_fail(self):
        """Test value > max_value fails."""
        result = ThresholdCheck_v1_0_0()(value=1.2, max_value=1.0)
        assert result == {"passed": False}

    def test_threshold_range_inside(self):
        """Test value within min and max passes."""
        result = ThresholdCheck_v1_0_0()(value=0.85, min_value=0.8, max_value=1.0)
        assert result == {"passed": True}

    def test_threshold_range_outside(self):
        """Test value outside range fails."""
        result = ThresholdCheck_v1_0_0()(value=1.2, min_value=0.8, max_value=1.0)
        assert result == {"passed": False}

    def test_threshold_min_exclusive(self):
        """Test min_inclusive=false excludes boundary."""
        result = ThresholdCheck_v1_0_0()(value=0.8, min_value=0.8, min_inclusive=False)
        assert result == {"passed": False}

        # But greater than boundary should pass
        result = ThresholdCheck_v1_0_0()(value=0.81, min_value=0.8, min_inclusive=False)
        assert result == {"passed": True}

    def test_threshold_max_exclusive(self):
        """Test max_inclusive=false excludes boundary."""
        result = ThresholdCheck_v1_0_0()(value=1.0, max_value=1.0, max_inclusive=False)
        assert result == {"passed": False}

        # But less than boundary should pass
        result = ThresholdCheck_v1_0_0()(value=0.99, max_value=1.0, max_inclusive=False)
        assert result == {"passed": True}

    def test_threshold_negate_outside(self):
        """Test negate=true passes when outside bounds."""
        result = ThresholdCheck_v1_0_0()(value=1.2, min_value=0.8, max_value=1.0, negate=True)
        assert result == {"passed": True}

    def test_threshold_negate_inside(self):
        """Test negate=true fails when inside bounds."""
        result = ThresholdCheck_v1_0_0()(value=0.9, min_value=0.8, max_value=1.0, negate=True)
        assert result == {"passed": False}

    def test_threshold_no_bounds_error(self):
        """Test error when neither min nor max specified."""
        with pytest.raises(ValidationError, match="requires at least one of"):
            ThresholdCheck_v1_0_0()(value=0.85)

    def test_threshold_non_numeric_error(self):
        """Test error when value is not numeric."""
        with pytest.raises(ValidationError, match="must be numeric"):
            ThresholdCheck_v1_0_0()(value="not a number", min_value=0.8)

    def test_threshold_string_numeric_conversion(self):
        """Test numeric string conversion."""
        result = ThresholdCheck_v1_0_0()(value="0.85", min_value=0.8)

        assert result == {"passed": True}

    def test_threshold_missing_value(self):
        """Test missing value argument raises TypeError."""
        with pytest.raises(TypeError):
            ThresholdCheck_v1_0_0()(min_value=0.8)

    def test_threshold_invalid_min_value_type(self):
        """Test non-numeric min_value raises ValidationError."""
        with pytest.raises(ValidationError, match="'min_value' must be numeric"):
            ThresholdCheck_v1_0_0()(value=0.85, min_value="not numeric")

    def test_threshold_invalid_max_value_type(self):
        """Test non-numeric max_value raises ValidationError."""
        with pytest.raises(ValidationError, match="'max_value' must be numeric"):
            ThresholdCheck_v1_0_0()(value=0.85, max_value="not numeric")

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
        # Define your test cases
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
        # System outputs to evaluate
        outputs = [
            Output(value={"message": "High confidence", "confidence": confidence_value}),
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


class TestAttributeExistsCheck:
    """Test AttributeExists check implementation."""

    def test_attribute_exists_via_evaluate_exists(self):
        """Test attribute exists check when attribute is present."""
        # Define test case with error present
        from flex_evals import AttributeExistsCheck  # noqa: PLC0415
        test_cases = [
            TestCase(
                id="test_001",
                input="Test with error",
                checks=[
                    AttributeExistsCheck(path="$.output.value.error"),
                ],
            ),
        ]

        # System output with error field
        outputs = [
            Output(value={"result": "failed", "error": "Something went wrong"}),
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

    def test_attribute_exists_via_evaluate_does_not_exist(self):
        """Test attribute exists check when attribute is not present."""
        # Define test case checking for error field
        test_cases = [
            TestCase(
                id="test_001",
                input="Test without error",
                checks=[
                    Check(
                        type=CheckType.ATTRIBUTE_EXISTS,
                        arguments={
                            "path": "$.output.value.error",
                        },
                    ),
                ],
            ),
        ]

        # System output without error field
        outputs = [
            Output(value={"result": "success", "message": "All good"}),
        ]

        # Run evaluation
        results = evaluate(test_cases, outputs)
        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.summary.error_test_cases == 0
        assert results.summary.skipped_test_cases == 0
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": False}

    def test_attribute_exists_via_evaluate_negate_exists(self):
        """Test attribute exists check with negate when attribute is present."""
        # Define test case checking that error does NOT exist
        test_cases = [
            TestCase(
                id="test_001",
                input="Test with error but negate",
                checks=[
                    Check(
                        type=CheckType.ATTRIBUTE_EXISTS,
                        arguments={
                            "path": "$.output.value.error",
                            "negate": True,
                        },
                    ),
                ],
            ),
        ]

        # System output with error field (should fail because negate=True)
        outputs = [
            Output(value={"result": "failed", "error": "Something went wrong"}),
        ]

        # Run evaluation
        results = evaluate(test_cases, outputs)
        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.summary.error_test_cases == 0
        assert results.summary.skipped_test_cases == 0
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": False}

    def test_attribute_exists_via_evaluate_negate_does_not_exist(self):
        """Test attribute exists check with negate when attribute is not present."""
        # Define test case checking that error does NOT exist
        test_cases = [
            TestCase(
                id="test_001",
                input="Test without error and negate",
                checks=[
                    Check(
                        type=CheckType.ATTRIBUTE_EXISTS,
                        arguments={
                            "path": "$.output.value.error",
                            "negate": True,
                        },
                    ),
                ],
            ),
        ]

        # System output without error field (should pass because negate=True)
        outputs = [
            Output(value={"result": "success", "message": "All good"}),
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

    def test_attribute_exists_nested_path_exists(self):
        """Test deeply nested path that exists."""
        test_cases = [
            TestCase(
                id="test_001",
                input="Test nested path",
                checks=[
                    Check(
                        type=CheckType.ATTRIBUTE_EXISTS,
                        arguments={
                            "path": "$.output.value.metadata.processing.duration",
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "result": "success",
                "metadata": {
                    "processing": {
                        "duration": 1.25,
                        "steps": ["parse", "validate", "execute"],
                    },
                    "version": "1.0.0",
                },
            }),
        ]

        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_attribute_exists_nested_path_missing(self):
        """Test deeply nested path where intermediate path is missing."""
        test_cases = [
            TestCase(
                id="test_001",
                input="Test nested path missing",
                checks=[
                    Check(
                        type=CheckType.ATTRIBUTE_EXISTS,
                        arguments={
                            "path": "$.output.value.metadata.processing.duration",
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "result": "success",
                "metadata": {
                    # missing "processing" object
                    "version": "1.0.0",
                },
            }),
        ]

        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": False}

    def test_attribute_exists_invalid_path_error(self):
        """Test with invalid JSONPath expression raises validation error."""
        test_cases = [
            TestCase(
                id="test_001",
                input="Test invalid path",
                checks=[
                    Check(
                        type=CheckType.ATTRIBUTE_EXISTS,
                        arguments={
                            "path": "not_a_jsonpath",  # Missing $.
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={"result": "success"}),
        ]

        results = evaluate(test_cases, outputs)
        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 0
        assert results.summary.error_test_cases == 1
        assert results.summary.skipped_test_cases == 0
        assert results.results[0].status == Status.ERROR
        assert results.results[0].check_results[0].status == Status.ERROR
        assert "JSONPath expression" in results.results[0].check_results[0].error.message

    def test_attribute_exists_missing_path_argument(self):
        """Test missing path argument raises validation error."""
        test_cases = [
            TestCase(
                id="test_001",
                input="Test missing path",
                checks=[
                    Check(
                        type=CheckType.ATTRIBUTE_EXISTS,
                        arguments={
                            # Missing "path" argument
                            "negate": False,
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={"result": "success"}),
        ]

        results = evaluate(test_cases, outputs)
        assert results.summary.error_test_cases == 1
        assert results.results[0].status == Status.ERROR
        assert "requires 'path' argument" in results.results[0].check_results[0].error.message

    def test_attribute_exists_various_data_types(self):
        """Test attribute existence with various data types."""
        test_cases = [
            TestCase(
                id="test_001",
                input="Test various types",
                checks=[
                    Check(type=CheckType.ATTRIBUTE_EXISTS, arguments={"path": "$.output.value.number"}),  # noqa: E501
                    Check(type=CheckType.ATTRIBUTE_EXISTS, arguments={"path": "$.output.value.boolean"}),  # noqa: E501
                    Check(type=CheckType.ATTRIBUTE_EXISTS, arguments={"path": "$.output.value.null_value"}),  # noqa: E501
                    Check(type=CheckType.ATTRIBUTE_EXISTS, arguments={"path": "$.output.value.array"}),  # noqa: E501
                    Check(type=CheckType.ATTRIBUTE_EXISTS, arguments={"path": "$.output.value.missing", "negate": True}),  # noqa: E501
                ],
            ),
        ]

        outputs = [
            Output(value={
                "number": 42,
                "boolean": False,
                "null_value": None,
                "array": [],
                # "missing" key intentionally absent
            }),
        ]

        results = evaluate(test_cases, outputs)

        # All checks should pass
        for i in range(5):
            assert results.results[0].check_results[i].status == Status.COMPLETED
            assert results.results[0].check_results[i].results == {"passed": True}

    @pytest.mark.parametrize(("output_value", "path", "negate", "expected_passed"), [
        ({"error": "failed"}, "$.output.value.error", False, True),  # Attribute exists, no negate
        ({"success": True}, "$.output.value.error", False, False),  # Attribute missing, no negate
        ({"error": "failed"}, "$.output.value.error", True, False),  # Attribute exists, with negate  # noqa: E501
        ({"success": True}, "$.output.value.error", True, True),  # Attribute missing, with negate
        ({"nested": {"field": "value"}}, "$.output.value.nested.field", False, True),  # Nested exists  # noqa: E501
        ({"nested": {}}, "$.output.value.nested.field", False, False),  # Nested missing
    ])
    def test_attribute_exists_parametrized(
        self, output_value: dict, path: str, negate: bool, expected_passed: bool,
    ):
        """Test attribute exists check with various combinations."""
        test_cases = [
            TestCase(
                id="test_001",
                input="Parametrized test",
                checks=[
                    Check(
                        type=CheckType.ATTRIBUTE_EXISTS,
                        arguments={
                            "path": path,
                            "negate": negate,
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


def test_schema_implementation_consistency():
    """Test that each schema class has a properly registered implementation."""
    # Map schema classes to their check types
    from flex_evals import (  # noqa: PLC0415
        ExactMatchCheck,
        ContainsCheck,
        RegexCheck,
        IsEmptyCheck,
        ThresholdCheck,
        AttributeExistsCheck,
    )
    schema_to_check_type = {
        ExactMatchCheck: CheckType.EXACT_MATCH,
        ContainsCheck: CheckType.CONTAINS,
        RegexCheck: CheckType.REGEX,
        IsEmptyCheck: CheckType.IS_EMPTY,
        ThresholdCheck: CheckType.THRESHOLD,
        AttributeExistsCheck: CheckType.ATTRIBUTE_EXISTS,
    }

    registered_checks = list_registered_checks()

    for schema_class, check_type in schema_to_check_type.items():
        # Check schema has VERSION
        assert hasattr(schema_class, 'VERSION'), f"{schema_class.__name__} missing VERSION"

        # Check corresponding implementation is registered
        check_type_str = check_type.value
        assert check_type_str in registered_checks, f"No implementation registered for {check_type_str}"  # noqa: E501

        # Check version matches
        registered_version = registered_checks[check_type_str]["1.0.0"]["version"]
        assert registered_version == schema_class.VERSION, f"Version mismatch for {check_type_str}"
