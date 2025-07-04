"""Tests for pytest decorator implementation using real checks (no mocks)."""

import pytest
import _pytest.outcomes
from pydantic import BaseModel, Field

from flex_evals.pytest_decorator import evaluate
from flex_evals.schemas import TestCase, Check
from flex_evals.constants import CheckType
from typing import Never


# Test response models for LLM judge tests
class MockQualityResult(BaseModel):
    """Mock quality assessment result."""

    passed: bool = Field(description="Whether the check passed")
    score: int = Field(ge=1, le=5, description="Quality score")
    reasoning: str = Field(description="Reasoning for the assessment")


class MockBooleanResult(BaseModel):
    """Mock boolean result."""

    passed: bool = Field(description="Whether the check passed")


class TestEvaluateDecoratorBasicFunctionality:
    """Test basic decorator functionality with real checks."""

    def test_basic_success_scenario(self):
        """Test basic successful evaluation with deterministic function."""

        @evaluate(
            test_cases=[TestCase(id="basic", input="test input")],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["Python"]},
            )],
            samples=3,
            success_threshold=1.0,
        )
        def deterministic_success() -> str:
            return "Python is awesome"

        # Should pass silently (pytest convention)
        result = deterministic_success()
        assert result is None

    def test_basic_failure_scenario(self):
        """Test basic failure scenario with deterministic function."""

        @evaluate(
            test_cases=[TestCase(id="fail", input="test input")],
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "never_matches", "actual": "$.output.value"},
            )],
            samples=3,
            success_threshold=0.5,
        )
        def deterministic_failure() -> str:
            return "different_value"

        # Should fail with pytest.fail
        with pytest.raises(_pytest.outcomes.Failed) as exc_info:
            deterministic_failure()

        error_message = str(exc_info.value)
        assert "Statistical evaluation failed" in error_message
        assert "Success rate: 0.00%" in error_message
        assert "Required threshold: 50.00%" in error_message

    def test_exact_threshold_success(self):
        """Test success when success rate exactly meets threshold."""
        call_count = 0

        @evaluate(
            test_cases=[TestCase(id="exact", input="test")],
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "pass", "actual": "$.output.value"},
            )],
            samples=4,
            success_threshold=0.75,  # Exactly 75%
        )
        def exact_threshold_function() -> str:
            nonlocal call_count
            call_count += 1
            # Pass exactly 3 out of 4 times (75%)
            return "pass" if call_count <= 3 else "fail"

        # Should pass since 75% meets 75% threshold
        exact_threshold_function()


class TestEvaluateDecoratorStatistical:
    """Test statistical evaluation scenarios."""

    def test_statistical_evaluation_success(self):
        """Test statistical threshold evaluation with passing rate."""
        call_count = 0

        @evaluate(
            test_cases=[TestCase(id="stats", input="test")],
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "pass", "actual": "$.output.value"},
            )],
            samples=10,
            success_threshold=0.7,  # 70% threshold
        )
        def statistical_function() -> str:
            nonlocal call_count
            call_count += 1
            # 80% success rate (8 out of 10)
            return "pass" if call_count % 10 != 1 and call_count % 10 != 2 else "fail"

        # Should pass (80% > 70%)
        statistical_function()

    def test_statistical_evaluation_failure(self):
        """Test statistical threshold evaluation with failing rate."""
        call_count = 0

        @evaluate(
            test_cases=[TestCase(id="stats_fail", input="test")],
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "pass", "actual": "$.output.value"},
            )],
            samples=10,
            success_threshold=0.8,  # 80% threshold
        )
        def statistical_failure_function() -> str:
            nonlocal call_count
            call_count += 1
            # 60% success rate (6 out of 10)
            return "pass" if call_count % 10 < 6 else "fail"

        # Should fail (60% < 80%)
        with pytest.raises(_pytest.outcomes.Failed) as exc_info:
            statistical_failure_function()

        error_message = str(exc_info.value)
        assert "Success rate: 60.00%" in error_message
        assert "Required threshold: 80.00%" in error_message

    def test_high_variance_statistical_evaluation(self):
        """Test with high variance in results."""
        call_count = 0

        @evaluate(
            test_cases=[TestCase(id="variance", input="test")],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["success"]},
            )],
            samples=20,
            success_threshold=0.65,  # 65% threshold
        )
        def high_variance_function() -> str:
            nonlocal call_count
            call_count += 1
            # ~70% success rate with some variance
            return "success result" if call_count % 10 < 7 else "failure result"

        # Should pass most of the time (70% > 65%)
        high_variance_function()


class TestEvaluateDecoratorErrorHandling:
    """Test error handling scenarios."""

    def test_function_exceptions_as_failures(self):
        """Test that function exceptions are counted as failures."""
        call_count = 0

        @evaluate(
            test_cases=[TestCase(id="exceptions", input="test")],
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "success", "actual": "$.output.value"},
            )],
            samples=6,
            success_threshold=0.4,  # 40% threshold
        )
        def exception_prone_function() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return "success"  # First 3 succeed
            raise RuntimeError(f"Error on call {call_count}")

        # 3/6 = 50% success rate > 40% threshold, should pass
        exception_prone_function()

    def test_all_exceptions_failure(self):
        """Test scenario where all function calls raise exceptions."""

        @evaluate(
            test_cases=[TestCase(id="all_exceptions", input="test")],
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "success", "actual": "$.output.value"},
            )],
            samples=3,
            success_threshold=0.1,  # Even 10% threshold
        )
        def always_throws() -> Never:
            raise ValueError("Always fails")

        # Should fail with 0% success rate
        with pytest.raises(_pytest.outcomes.Failed) as exc_info:
            always_throws()

        error_message = str(exc_info.value)
        assert "Success rate: 0.00%" in error_message
        assert "Exception: ValueError: Always fails" in error_message

    def test_mixed_exceptions_and_failures(self):
        """Test mix of exceptions and check failures."""
        call_count = 0

        @evaluate(
            test_cases=[TestCase(id="mixed", input="test")],
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "success", "actual": "$.output.value"},
            )],
            samples=9,
            success_threshold=0.25,  # 25% threshold
        )
        def mixed_failure_function() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return "success"  # First 3 succeed
            if call_count <= 6:
                return "fail"  # Next 3 fail checks
            raise RuntimeError(f"Exception on call {call_count}")  # Last 3 throw

        # 3/9 = 33.33% success rate > 25% threshold, should pass
        mixed_failure_function()


class TestEvaluateDecoratorCheckTypes:
    """Test with different check types."""

    def test_multiple_checks_all_pass(self):
        """Test with multiple different check types that all pass."""

        @evaluate(
            test_cases=[TestCase(
                id="multi",
                input="test input",
                expected="expected_output",
            )],
            checks=[
                Check(
                    type=CheckType.CONTAINS,
                    arguments={"text": "$.output.value.message", "phrases": ["key"]},
                ),
                Check(
                    type=CheckType.EXACT_MATCH,
                    arguments={"expected": "$.test_case.expected", "actual": "$.output.value.result"},  # noqa: E501
                ),
            ],
            samples=2,
            success_threshold=1.0,
        )
        def multi_check_function():  # noqa: ANN202
            return {
                "result": "expected_output",  # Matches exact check
                "message": "This contains the key phrase",  # Matches contains check
            }

        multi_check_function()

    def test_multiple_checks_partial_pass(self):
        """Test with multiple checks where some pass and some fail."""

        @evaluate(
            test_cases=[TestCase(
                id="partial",
                input="test",
                expected="exact_match",
            )],
            checks=[
                Check(
                    type=CheckType.CONTAINS,
                    arguments={"text": "$.output.value", "phrases": ["success"]},
                ),
                Check(
                    type=CheckType.EXACT_MATCH,
                    arguments={"expected": "$.test_case.expected", "actual": "$.output.value"},
                ),
            ],
            samples=4,
            success_threshold=0.25,  # 25% threshold
        )
        def partial_check_function() -> str:
            # Only passes contains check, fails exact match
            return "success message but not exact match"

        # Should fail since BOTH checks must pass for a sample to pass
        with pytest.raises(_pytest.outcomes.Failed):
            partial_check_function()

    def test_regex_check_integration(self):
        """Test with regex check."""

        @evaluate(
            test_cases=[TestCase(id="regex_test", input="test")],
            checks=[Check(
                type=CheckType.REGEX,
                arguments={
                    "pattern": r"Result: \d+",
                    "text": "$.output.value",
                },
            )],
            samples=3,
            success_threshold=1.0,
        )
        def regex_function() -> str:
            return "Result: 42"

        regex_function()

    def test_threshold_check_integration(self):
        """Test with threshold check."""

        @evaluate(
            test_cases=[TestCase(id="threshold_test", input="test")],
            checks=[Check(
                type=CheckType.THRESHOLD,
                arguments={
                    "value": "$.output.value.score",
                    "min_value": 80,
                    "max_value": 100,
                },
            )],
            samples=2,
            success_threshold=1.0,
        )
        def threshold_function():  # noqa: ANN202
            return {"score": 95, "message": "High quality"}

        threshold_function()


class TestEvaluateDecoratorLLMJudge:
    """Test LLM judge integration."""

    def test_llm_judge_with_mock_function(self):
        """Test LLM judge with deterministic mock function."""

        def deterministic_llm_function(prompt: str, response_format: type) -> tuple:
            """Deterministic mock LLM for testing."""
            if "good response" in prompt:
                return response_format(
                    passed=True,
                    score=5,
                    reasoning="Excellent response",
                ), {"cost_usd": 0.01}
            return response_format(
                passed=False,
                score=2,
                reasoning="Poor response",
            ), {"cost_usd": 0.01}

        @evaluate(
            test_cases=[TestCase(id="llm_test", input="test query")],
            checks=[Check(
                type=CheckType.LLM_JUDGE,
                arguments={
                    "prompt": "Evaluate this response: {{$.output.value}}",
                    "response_format": MockQualityResult,
                    "llm_function": deterministic_llm_function,
                },
            )],
            samples=3,
            success_threshold=1.0,
        )
        def good_response_function() -> str:
            return "This is a good response"

        good_response_function()

    def test_llm_judge_statistical_evaluation(self):
        """Test LLM judge with variable quality responses."""

        def variable_llm_function(prompt: str, response_format: type) -> tuple:
            """Mock LLM that varies assessment based on content."""
            if "excellent" in prompt:
                return response_format(passed=True, score=5, reasoning="Great"), {}
            if "good" in prompt:
                return response_format(passed=True, score=4, reasoning="Good"), {}
            if "okay" in prompt:
                return response_format(passed=True, score=3, reasoning="Okay"), {}
            return response_format(passed=False, score=2, reasoning="Poor"), {}

        call_count = 0

        @evaluate(
            test_cases=[TestCase(id="variable_llm", input="test")],
            checks=[Check(
                type=CheckType.LLM_JUDGE,
                arguments={
                    "prompt": "Rate this: {{$.output.value}}",
                    "response_format": MockQualityResult,
                    "llm_function": variable_llm_function,
                },
            )],
            samples=8,
            success_threshold=0.6,  # 60% threshold
        )
        def variable_quality_function() -> str:
            nonlocal call_count
            call_count += 1
            responses = ["excellent", "good", "okay", "poor"] * 2
            return f"This is {responses[call_count - 1]} content"

        # Should pass: excellent(pass) + good(pass) + okay(pass) + poor(fail) = 6/8 = 75% > 60%
        variable_quality_function()

    def test_llm_judge_with_flattened_response(self):
        """Test that LLM judge returns flattened response structure."""

        def simple_llm_function(prompt: str, response_format: type) -> tuple:  # noqa: ARG001
            return response_format(passed=True), {"cost": 0.01}

        @evaluate(
            test_cases=[TestCase(id="flattened", input="test")],
            checks=[Check(
                type=CheckType.LLM_JUDGE,
                arguments={
                    "prompt": "Simple evaluation: {{$.output.value}}",
                    "response_format": MockBooleanResult,
                    "llm_function": simple_llm_function,
                },
            )],
            samples=2,
            success_threshold=1.0,
        )
        def simple_function() -> str:
            return "test response"

        # This test verifies the LLM judge integration works with flattened response
        simple_function()


class TestEvaluateDecoratorTestCaseChecks:
    """Test TestCase-defined checks functionality."""

    def test_test_case_defined_checks(self):
        """Test with checks defined in TestCase objects."""
        test_case_with_checks = TestCase(
            id="with_checks",
            input="test input",
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["success"]},
            )],
        )

        @evaluate(
            test_cases=[test_case_with_checks],
            checks=None,  # Use TestCase checks
            samples=3,
            success_threshold=1.0,
        )
        def test_case_checks_function() -> str:
            return "This is a success message"

        test_case_checks_function()

    def test_multiple_test_cases_with_different_checks(self):
        """Test multiple test cases with different checks."""
        test_cases = [
            TestCase(
                id="contains_check",
                input="input1",
                checks=[Check(
                    type=CheckType.CONTAINS,
                    arguments={"text": "$.output.value", "phrases": ["alpha"]},
                )],
            ),
            TestCase(
                id="exact_check",
                input="input2",
                checks=[Check(
                    type=CheckType.EXACT_MATCH,
                    arguments={"expected": "beta", "actual": "$.output.value"},
                )],
            ),
        ]

        call_count = 0

        @evaluate(
            test_cases=test_cases,
            checks=None,
            samples=4,  # Will cycle: contains, exact, contains, exact
            success_threshold=1.0,
        )
        def cycling_function() -> str:
            nonlocal call_count
            call_count += 1
            # Return appropriate response based on which test case we're cycling to
            if call_count % 2 == 1:
                return "response with alpha"  # For contains check
            return "beta"  # For exact match check

        cycling_function()

    def test_cycling_through_test_cases(self):
        """Test cycling through multiple test cases preserves original structure."""
        test_cases = [
            TestCase(id="case1", input="input1", expected="result1"),
            TestCase(id="case2", input="input2", expected="result2"),
        ]

        call_count = 0

        @evaluate(
            test_cases=test_cases,
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "$.test_case.expected", "actual": "$.output.value"},
            )],
            samples=4,  # Will cycle: case1, case2, case1, case2
            success_threshold=1.0,
        )
        def cycling_function() -> str:
            nonlocal call_count
            call_count += 1
            # Return expected value based on which test case we're on
            if call_count % 2 == 1:
                return "result1"  # For case1
            return "result2"  # For case2

        cycling_function()


class TestEvaluateDecoratorEdgeCases:
    """Test edge cases and complex scenarios."""

    def test_complex_return_value_structures(self):
        """Test with complex return value types."""

        @evaluate(
            test_cases=[TestCase(id="complex", input="test")],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value.message", "phrases": ["success"]},
            )],
            samples=2,
            success_threshold=1.0,
        )
        def complex_return_function():  # noqa: ANN202
            return {
                "message": "Operation was a success",
                "data": {"items": [1, 2, 3]},
                "metadata": {"timestamp": "2024-01-01"},
            }

        complex_return_function()

    def test_none_return_value(self):
        """Test function that returns None (wrapped in dict)."""

        @evaluate(
            test_cases=[TestCase(id="none_test", input="test")],
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": None, "actual": "$.output.value.result"},
            )],
            samples=2,
            success_threshold=1.0,
        )
        def none_return_function():  # noqa: ANN202
            # Output.value can't be None, so wrap it in a dict
            return {"result": None}

        none_return_function()

    def test_string_return_value(self):
        """Test function that returns simple string."""

        @evaluate(
            test_cases=[TestCase(id="string_test", input="test")],
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "simple string", "actual": "$.output.value"},
            )],
            samples=2,
            success_threshold=1.0,
        )
        def string_return_function() -> str:
            return "simple string"

        string_return_function()

    def test_list_return_value(self):
        """Test function that returns list (wrapped in dict)."""

        @evaluate(
            test_cases=[TestCase(id="list_test", input="test")],
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": [1, 2, 3], "actual": "$.output.value.items"},
            )],
            samples=2,
            success_threshold=1.0,
        )
        def list_return_function():  # noqa: ANN202
            # Output.value can't be a list, so wrap it in a dict
            return {"items": [1, 2, 3]}

        list_return_function()

    def test_single_sample_evaluation(self):
        """Test evaluation with only one sample."""

        @evaluate(
            test_cases=[TestCase(id="single", input="test")],
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "success", "actual": "$.output.value"},
            )],
            samples=1,
            success_threshold=1.0,
        )
        def single_sample_function() -> str:
            return "success"

        single_sample_function()

    def test_large_sample_count(self):
        """Test with larger sample count for performance validation."""

        @evaluate(
            test_cases=[TestCase(id="large", input="test")],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["pass"]},
            )],
            samples=50,
            success_threshold=0.9,
        )
        def large_sample_function() -> str:
            return "This should pass every time"

        large_sample_function()


class TestEvaluateDecoratorParameterValidation:
    """Test parameter validation."""

    def test_empty_test_cases_error(self):
        """Test error when test_cases is empty."""
        with pytest.raises(ValueError, match="test_cases list cannot be empty"):
            @evaluate(
                test_cases=[],
                checks=[Check(type=CheckType.EXACT_MATCH, arguments={})],
                samples=1,
                success_threshold=1.0,
            )
            def empty_test_cases() -> str:
                return "test"

    def test_zero_samples_error(self):
        """Test error when samples is zero."""
        with pytest.raises(ValueError, match="samples must be positive"):
            @evaluate(
                test_cases=[TestCase(id="test", input="test")],
                checks=[Check(type=CheckType.EXACT_MATCH, arguments={})],
                samples=0,
                success_threshold=1.0,
            )
            def zero_samples() -> str:
                return "test"

    def test_negative_samples_error(self):
        """Test error when samples is negative."""
        with pytest.raises(ValueError, match="samples must be positive"):
            @evaluate(
                test_cases=[TestCase(id="test", input="test")],
                checks=[Check(type=CheckType.EXACT_MATCH, arguments={})],
                samples=-1,
                success_threshold=1.0,
            )
            def negative_samples() -> str:
                return "test"

    def test_invalid_threshold_high_error(self):
        """Test error when success_threshold is > 1.0."""
        with pytest.raises(ValueError, match="success_threshold must be between 0.0 and 1.0"):
            @evaluate(
                test_cases=[TestCase(id="test", input="test")],
                checks=[Check(type=CheckType.EXACT_MATCH, arguments={})],
                samples=1,
                success_threshold=1.5,
            )
            def invalid_high_threshold() -> str:
                return "test"

    def test_invalid_threshold_low_error(self):
        """Test error when success_threshold is < 0.0."""
        with pytest.raises(ValueError, match="success_threshold must be between 0.0 and 1.0"):
            @evaluate(
                test_cases=[TestCase(id="test", input="test")],
                checks=[Check(type=CheckType.EXACT_MATCH, arguments={})],
                samples=1,
                success_threshold=-0.1,
            )
            def invalid_low_threshold() -> str:
                return "test"

    def test_no_checks_anywhere_error(self):
        """Test error when no checks are defined anywhere."""
        test_case_no_checks = TestCase(id="no_checks", input="test input")

        with pytest.raises(ValueError, match="each TestCase must define its own checks"):
            @evaluate(
                test_cases=[test_case_no_checks],
                checks=None,
                samples=1,
                success_threshold=1.0,
            )
            def no_checks_function() -> str:
                return "test"


class TestEvaluateDecoratorFailureReporting:
    """Test failure reporting functionality."""

    def test_detailed_failure_report(self):
        """Test that failure reports contain detailed information."""

        @evaluate(
            test_cases=[TestCase(id="detailed_fail", input="test")],
            checks=[
                Check(
                    type=CheckType.EXACT_MATCH,
                    arguments={"expected": "never", "actual": "$.output.value"},
                ),
                Check(
                    type=CheckType.CONTAINS,
                    arguments={"text": "$.output.value", "phrases": ["missing"]},
                ),
            ],
            samples=2,
            success_threshold=0.5,
        )
        def detailed_failure_function() -> str:
            return "always fails both checks"

        with pytest.raises(_pytest.outcomes.Failed) as exc_info:
            detailed_failure_function()

        error_message = str(exc_info.value)

        # Check that detailed information is present
        assert "Statistical evaluation failed" in error_message
        assert "Total samples: 2" in error_message
        assert "Passed: 0" in error_message
        assert "Failed: 2" in error_message
        assert "Failed samples:" in error_message
        assert "Sample 0:" in error_message
        assert "Sample 1:" in error_message

    def test_failure_report_with_exceptions(self):
        """Test failure report includes exception details."""
        call_count = 0

        @evaluate(
            test_cases=[TestCase(id="exception_report", input="test")],
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "success", "actual": "$.output.value"},
            )],
            samples=3,
            success_threshold=0.8,  # 80% threshold
        )
        def exception_reporting_function() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "success"  # First call succeeds
            raise ValueError(f"Test exception {call_count}")

        with pytest.raises(_pytest.outcomes.Failed) as exc_info:
            exception_reporting_function()

        error_message = str(exc_info.value)
        assert "Exception: ValueError: Test exception" in error_message
