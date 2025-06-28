"""
Tests for standard check implementations.
"""

import pytest
from typing import Dict, Any

from flex_evals.checks.base import EvaluationContext
from flex_evals.checks.standard.exact_match import ExactMatchCheck
from flex_evals.checks.standard.contains import ContainsCheck
from flex_evals.checks.standard.regex import RegexCheck
from flex_evals.checks.standard.threshold import ThresholdCheck
from flex_evals.schemas import TestCase, Output
from flex_evals.exceptions import ValidationError


class TestExactMatchCheck:
    """Test ExactMatch check implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.check = ExactMatchCheck()
        self.test_case = TestCase(id="test_001", input="test", expected="Paris")
        self.output = Output(value="Paris")
        self.context = EvaluationContext(self.test_case, self.output)
    
    def test_exact_match_string_equal(self):
        """Test matching strings return passed=true."""
        arguments = {"actual": "Paris", "expected": "Paris"}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_exact_match_string_not_equal(self):
        """Test non-matching strings return passed=false."""
        arguments = {"actual": "paris", "expected": "Paris"}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": False}
    
    def test_exact_match_case_sensitive_true(self):
        """Test 'Hello' != 'hello' when case_sensitive=true."""
        arguments = {"actual": "Hello", "expected": "hello", "case_sensitive": True}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": False}
    
    def test_exact_match_case_sensitive_false(self):
        """Test 'Hello' == 'hello' when case_sensitive=false."""
        arguments = {"actual": "Hello", "expected": "hello", "case_sensitive": False}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_exact_match_negate_true(self):
        """Test negate=true passes when values differ."""
        arguments = {"actual": "Paris", "expected": "London", "negate": True}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_exact_match_negate_false(self):
        """Test negate=false passes when values match."""
        arguments = {"actual": "Paris", "expected": "Paris", "negate": False}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_exact_match_object_comparison(self):
        """Test comparing complex objects."""
        # Objects will be converted to strings for comparison
        arguments = {"actual": {"city": "Paris"}, "expected": "{'city': 'Paris'}"}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_exact_match_null_values(self):
        """Test comparison with null/None values."""
        arguments = {"actual": None, "expected": ""}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}  # None converts to empty string
    
    def test_exact_match_missing_actual(self):
        """Test missing actual argument raises ValidationError."""
        arguments = {"expected": "Paris"}
        
        with pytest.raises(ValidationError, match="requires 'actual' argument"):
            self.check(arguments, self.context)
    
    def test_exact_match_missing_expected(self):
        """Test missing expected argument raises ValidationError."""
        arguments = {"actual": "Paris"}
        
        with pytest.raises(ValidationError, match="requires 'expected' argument"):
            self.check(arguments, self.context)
    
    def test_exact_match_result_schema(self):
        """Test result matches {\"passed\": boolean} exactly."""
        arguments = {"actual": "test", "expected": "test"}
        result = self.check(arguments, self.context)
        
        assert isinstance(result, dict)
        assert set(result.keys()) == {"passed"}
        assert isinstance(result["passed"], bool)


class TestContainsCheck:
    """Test Contains check implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.check = ContainsCheck()
        self.test_case = TestCase(id="test_001", input="test")
        self.output = Output(value="Paris is the capital of France")
        self.context = EvaluationContext(self.test_case, self.output)
    
    def test_contains_all_phrases_found(self):
        """Test negate=false passes when all phrases present."""
        arguments = {
            "text": "Paris is the capital of France",
            "phrases": ["Paris", "France"],
            "negate": False
        }
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_contains_some_phrases_missing(self):
        """Test negate=false fails when any phrase missing."""
        arguments = {
            "text": "Paris is the capital of France",
            "phrases": ["Paris", "Spain"],  # Spain is missing
            "negate": False
        }
        result = self.check(arguments, self.context)
        
        assert result == {"passed": False}
    
    def test_contains_negate_none_found(self):
        """Test negate=true passes when no phrases found."""
        arguments = {
            "text": "Paris is the capital of France",
            "phrases": ["London", "Spain"],
            "negate": True
        }
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_contains_negate_some_found(self):
        """Test negate=true fails when any phrase found."""
        arguments = {
            "text": "Paris is the capital of France", 
            "phrases": ["Paris", "Spain"],  # Paris is found
            "negate": True
        }
        result = self.check(arguments, self.context)
        
        assert result == {"passed": False}
    
    def test_contains_case_sensitive(self):
        """Test case sensitivity in phrase matching."""
        arguments = {
            "text": "Paris is the capital of France",
            "phrases": ["paris"],  # Lowercase
            "case_sensitive": True
        }
        result = self.check(arguments, self.context)
        
        assert result == {"passed": False}
    
    def test_contains_case_insensitive(self):
        """Test case insensitive matching."""
        arguments = {
            "text": "Paris is the capital of France",
            "phrases": ["paris"],  # Lowercase
            "case_sensitive": False
        }
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_contains_empty_phrases(self):
        """Test behavior with empty phrases array."""
        arguments = {"text": "test text", "phrases": []}
        
        with pytest.raises(ValidationError, match="must not be empty"):
            self.check(arguments, self.context)
    
    def test_contains_single_phrase(self):
        """Test with single phrase in array."""
        arguments = {
            "text": "Paris is the capital of France",
            "phrases": ["capital"]
        }
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_contains_overlapping_phrases(self):
        """Test with overlapping/duplicate phrases."""
        arguments = {
            "text": "Paris Paris is great",
            "phrases": ["Paris", "Paris"]  # Duplicate phrase
        }
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_contains_missing_text(self):
        """Test missing text argument raises ValidationError."""
        arguments = {"phrases": ["test"]}
        
        with pytest.raises(ValidationError, match="requires 'text' argument"):
            self.check(arguments, self.context)
    
    def test_contains_missing_phrases(self):
        """Test missing phrases argument raises ValidationError."""
        arguments = {"text": "test text"}
        
        with pytest.raises(ValidationError, match="requires 'phrases' argument"):
            self.check(arguments, self.context)
    
    def test_contains_invalid_phrases_type(self):
        """Test non-list phrases argument raises ValidationError."""
        arguments = {"text": "test", "phrases": "not a list"}
        
        with pytest.raises(ValidationError, match="must be a list"):
            self.check(arguments, self.context)


class TestRegexCheck:
    """Test Regex check implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.check = RegexCheck()
        self.test_case = TestCase(id="test_001", input="test")
        self.output = Output(value="user@example.com")
        self.context = EvaluationContext(self.test_case, self.output)
    
    def test_regex_basic_match(self):
        """Test simple pattern matching."""
        arguments = {
            "text": "user@example.com",
            "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        }
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_regex_no_match(self):
        """Test pattern that doesn't match."""
        arguments = {
            "text": "not an email",
            "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        }
        result = self.check(arguments, self.context)
        
        assert result == {"passed": False}
    
    def test_regex_case_insensitive(self):
        """Test case_insensitive flag."""
        arguments = {
            "text": "Hello World",
            "pattern": "hello",
            "flags": {"case_insensitive": True}
        }
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_regex_multiline(self):
        """Test multiline flag with ^ and $ anchors."""
        text = "First line\nSecond line\nThird line"
        arguments = {
            "text": text,
            "pattern": "^Second",
            "flags": {"multiline": True}
        }
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_regex_dot_all(self):
        """Test dot_all flag with . matching newlines."""
        text = "First line\nSecond line"
        arguments = {
            "text": text,
            "pattern": "First.*Second",
            "flags": {"dot_all": True}
        }
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_regex_negate_true(self):
        """Test negate=true passes when pattern doesn't match."""
        arguments = {
            "text": "not an email",
            "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "negate": True
        }
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_regex_complex_pattern(self):
        """Test complex regex with groups, quantifiers."""
        arguments = {
            "text": "Phone: (555) 123-4567",
            "pattern": r"Phone: \((\d{3})\) (\d{3})-(\d{4})"
        }
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_regex_invalid_pattern(self):
        """Test invalid regex pattern raises appropriate error."""
        arguments = {
            "text": "test text",
            "pattern": "[invalid"  # Unclosed bracket
        }
        
        with pytest.raises(ValidationError, match="Invalid regex pattern"):
            self.check(arguments, self.context)
    
    def test_regex_empty_text(self):
        """Test pattern matching against empty string."""
        arguments = {
            "text": "",
            "pattern": "^$"  # Match empty string
        }
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_regex_missing_text(self):
        """Test missing text argument raises ValidationError."""
        arguments = {"pattern": "test"}
        
        with pytest.raises(ValidationError, match="requires 'text' argument"):
            self.check(arguments, self.context)
    
    def test_regex_missing_pattern(self):
        """Test missing pattern argument raises ValidationError."""
        arguments = {"text": "test"}
        
        with pytest.raises(ValidationError, match="requires 'pattern' argument"):
            self.check(arguments, self.context)
    
    def test_regex_invalid_pattern_type(self):
        """Test non-string pattern raises ValidationError."""
        arguments = {"text": "test", "pattern": 123}
        
        with pytest.raises(ValidationError, match="must be a string"):
            self.check(arguments, self.context)


class TestThresholdCheck:
    """Test Threshold check implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.check = ThresholdCheck()
        self.test_case = TestCase(id="test_001", input="test")
        self.output = Output(value={"score": 0.85})
        self.context = EvaluationContext(self.test_case, self.output)
    
    def test_threshold_min_only_pass(self):
        """Test value >= min_value passes."""
        arguments = {"value": 0.85, "min_value": 0.8}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_threshold_min_only_fail(self):
        """Test value < min_value fails."""
        arguments = {"value": 0.75, "min_value": 0.8}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": False}
    
    def test_threshold_max_only_pass(self):
        """Test value <= max_value passes."""
        arguments = {"value": 0.85, "max_value": 1.0}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_threshold_max_only_fail(self):
        """Test value > max_value fails."""
        arguments = {"value": 1.2, "max_value": 1.0}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": False}
    
    def test_threshold_range_inside(self):
        """Test value within min and max passes."""
        arguments = {"value": 0.85, "min_value": 0.8, "max_value": 1.0}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_threshold_range_outside(self):
        """Test value outside range fails."""
        arguments = {"value": 1.2, "min_value": 0.8, "max_value": 1.0}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": False}
    
    def test_threshold_min_exclusive(self):
        """Test min_inclusive=false excludes boundary."""
        arguments = {"value": 0.8, "min_value": 0.8, "min_inclusive": False}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": False}
        
        # But greater than boundary should pass
        arguments = {"value": 0.81, "min_value": 0.8, "min_inclusive": False}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_threshold_max_exclusive(self):
        """Test max_inclusive=false excludes boundary."""
        arguments = {"value": 1.0, "max_value": 1.0, "max_inclusive": False}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": False}
        
        # But less than boundary should pass
        arguments = {"value": 0.99, "max_value": 1.0, "max_inclusive": False}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_threshold_negate_outside(self):
        """Test negate=true passes when outside bounds."""
        arguments = {"value": 1.2, "min_value": 0.8, "max_value": 1.0, "negate": True}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_threshold_negate_inside(self):
        """Test negate=true fails when inside bounds."""
        arguments = {"value": 0.9, "min_value": 0.8, "max_value": 1.0, "negate": True}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": False}
    
    def test_threshold_no_bounds_error(self):
        """Test error when neither min nor max specified."""
        arguments = {"value": 0.85}
        
        with pytest.raises(ValidationError, match="requires at least one of"):
            self.check(arguments, self.context)
    
    def test_threshold_non_numeric_error(self):
        """Test error when value is not numeric."""
        arguments = {"value": "not a number", "min_value": 0.8}
        
        with pytest.raises(ValidationError, match="must be numeric"):
            self.check(arguments, self.context)
    
    def test_threshold_string_numeric_conversion(self):
        """Test numeric string conversion."""
        arguments = {"value": "0.85", "min_value": 0.8}
        result = self.check(arguments, self.context)
        
        assert result == {"passed": True}
    
    def test_threshold_missing_value(self):
        """Test missing value argument raises ValidationError."""
        arguments = {"min_value": 0.8}
        
        with pytest.raises(ValidationError, match="requires 'value' argument"):
            self.check(arguments, self.context)
    
    def test_threshold_invalid_min_value_type(self):
        """Test non-numeric min_value raises ValidationError."""
        arguments = {"value": 0.85, "min_value": "not numeric"}
        
        with pytest.raises(ValidationError, match="'min_value' must be numeric"):
            self.check(arguments, self.context)
    
    def test_threshold_invalid_max_value_type(self):
        """Test non-numeric max_value raises ValidationError."""
        arguments = {"value": 0.85, "max_value": "not numeric"}
        
        with pytest.raises(ValidationError, match="'max_value' must be numeric"):
            self.check(arguments, self.context)