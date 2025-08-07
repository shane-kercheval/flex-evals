"""Tests for pytest decorator implementation using real checks (no mocks)."""

import inspect
import pytest
import time
import _pytest.outcomes
import asyncio
from pydantic import BaseModel, Field
import threading

from flex_evals.pytest_decorator import evaluate
from flex_evals import TestCase, Check, CheckType, ContainsCheck

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
            checks=[ContainsCheck(
                text="$.output.value",
                phrases=["Python"],
            )],
            samples=3,
            success_threshold=1.0,
        )
        def deterministic_success(test_case) -> str:  # noqa
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
        def deterministic_failure(test_case) -> str:  # noqa
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
        def exact_threshold_function(test_case) -> str:  # noqa
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
        def statistical_function(test_case) -> str:  # noqa
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
        def statistical_failure_function(test_case) -> str:  # noqa
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
        def high_variance_function(test_case) -> str:  # noqa
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
        def exception_prone_function(test_case) -> str:  # noqa
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
        def always_throws(test_case) -> Never:  # noqa
            raise ValueError("Always fails")

        # Should fail with 0% success rate
        with pytest.raises(_pytest.outcomes.Failed) as exc_info:
            always_throws()

        error_message = str(exc_info.value)
        assert "Success rate: 0.00%" in error_message
        assert "Test case 0 (id: all_exceptions) exception: ValueError: Always fails" in error_message  # noqa: E501

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
        def mixed_failure_function(test_case) -> str:  # noqa
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
        def multi_check_function(test_case):  # noqa
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
        def partial_check_function(test_case) -> str:  # noqa
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
        def regex_function(test_case) -> str:  # noqa
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
        def threshold_function(test_case):  # noqa
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
        def good_response_function(test_case) -> str:  # noqa
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
        def variable_quality_function(test_case) -> str:  # noqa
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
        def simple_function(test_case) -> str:  # noqa
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
        def test_case_checks_function(test_case) -> str:  # noqa
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
        def cycling_function(test_case) -> str:  # noqa
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
        def cycling_function(test_case) -> str:  # noqa
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
        def complex_return_function(test_case):  # noqa
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
        def none_return_function(test_case):  # noqa
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
        def string_return_function(test_case) -> str:  # noqa
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
        def list_return_function(test_case):  # noqa
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
        def single_sample_function(test_case) -> str:  # noqa
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
        def large_sample_function(test_case) -> str:  # noqa
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
            def empty_test_cases(test_case) -> str:  # noqa
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
            def zero_samples(test_case) -> str:  # noqa
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
            def negative_samples(test_case) -> str:  # noqa
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
            def invalid_high_threshold(test_case) -> str:  # noqa
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
            def invalid_low_threshold(test_case) -> str:  # noqa
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
            def no_checks_function(test_case) -> str:  # noqa
                return "test"


class TestEvaluateDecoratorTestCaseParameter:
    """Test that test_case parameter is passed correctly to test functions."""

    def test_single_test_case_parameter_passed(self):
        """Test that test_case parameter is passed with single test case."""
        received_test_cases = []

        @evaluate(
            test_cases=[TestCase(id="param_test", input="test_input", expected="expected_value")],
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "success", "actual": "$.output.value"},
            )],
            samples=3,
            success_threshold=1.0,
        )
        def test_function_with_test_case(test_case) -> str:  # noqa
            received_test_cases.append(test_case)
            # Verify test_case has expected attributes
            assert test_case.id == "param_test"
            assert test_case.input == "test_input"
            assert test_case.expected == "expected_value"
            return "success"

        test_function_with_test_case()

        # Verify test_case was passed to all 3 samples
        assert len(received_test_cases) == 3
        for test_case in received_test_cases:
            assert test_case.id == "param_test"
            assert test_case.input == "test_input"
            assert test_case.expected == "expected_value"

    def test_multiple_test_cases_parameter_cycling(self):
        """Test that test_case parameter cycles through multiple test cases correctly."""
        received_test_cases = []

        test_cases = [
            TestCase(id="case_1", input="input_1", expected="result_1"),
            TestCase(id="case_2", input="input_2", expected="result_2"),
            TestCase(id="case_3", input="input_3", expected="result_3"),
        ]

        @evaluate(
            test_cases=test_cases,
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "$.test_case.expected", "actual": "$.output.value"},
            )],
            samples=7,  # Will cycle: case_1, case_2, case_3, case_1, case_2, case_3, case_1
            success_threshold=1.0,
        )
        def cycling_test_function(test_case) -> str:  # noqa
            received_test_cases.append(test_case)
            # Return the expected value from the test case
            return test_case.expected

        cycling_test_function()

        # Verify correct expansion pattern: 3 test cases * 7 samples = 21 total calls
        assert len(received_test_cases) == 21

        # Pattern should be: [case_1, case_2, case_3] repeated 7 times
        expected_pattern = ["case_1", "case_2", "case_3"] * 7

        for i, test_case in enumerate(received_test_cases):
            expected_id = expected_pattern[i]
            assert test_case.id == expected_id

            # Verify the corresponding input and expected values
            case_number = expected_id.split("_")[1]
            assert test_case.input == f"input_{case_number}"
            assert test_case.expected == f"result_{case_number}"

    def test_test_case_parameter_with_llm_function(self):
        """Test that test_case parameter can be used with LLM functions."""
        def mock_llm_function(prompt: str) -> str:
            # Simple mock that processes the prompt
            return f"Processed: {prompt}"

        @evaluate(
            test_cases=[TestCase(id="llm_test", input="What is Python?")],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["Processed", "Python"]},
            )],
            samples=2,
            success_threshold=1.0,
        )
        def test_llm_integration(test_case) -> str:  # noqa
            # Use test_case.input to call the LLM function
            return mock_llm_function(test_case.input)

        test_llm_integration()


class TestEvaluateDecoratorSampling:
    """Test sample-based evaluation with multiple test cases."""

    def test_multiple_test_cases_samples_counting(self):
        """Test correct sample counting with multiple test cases."""
        execution_log = []

        test_cases = [
            TestCase(id="case_A", input="A", expected="result_A"),
            TestCase(id="case_B", input="B", expected="result_B"),
            TestCase(id="case_C", input="C", expected="result_C"),
        ]

        @evaluate(
            test_cases=test_cases,
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "$.test_case.expected", "actual": "$.output.value"},
            )],
            samples=4,  # 4 samples * 3 test cases = 12 total function calls
            success_threshold=1.0,
        )
        def test_sampling_function(test_case) -> str:  # noqa
            execution_log.append((test_case.id, test_case.input))
            return test_case.expected  # Always return expected value (should pass)

        test_sampling_function()

        # Verify execution pattern: should be called 12 times total (4 samples * 3 test cases)
        assert len(execution_log) == 12

        # Verify cycling pattern: A,B,C,A,B,C,A,B,C,A,B,C
        expected_pattern = [("case_A", "A"), ("case_B", "B"), ("case_C", "C")] * 4
        assert execution_log == expected_pattern

    def test_sample_failure_when_any_test_case_fails(self):
        """Test that a sample fails if ANY test case in the sample fails."""
        test_cases = [
            TestCase(id="case_pass", input="pass", expected="pass"),
            TestCase(id="case_fail", input="fail", expected="pass"),  # This will fail
        ]

        @evaluate(
            test_cases=test_cases,
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "$.test_case.expected", "actual": "$.output.value"},
            )],
            samples=3,  # 3 samples * 2 test cases = 6 total calls
            success_threshold=0.1,  # Very low threshold (should still fail)
        )
        def test_partial_failure(test_case) -> str:  # noqa
            return test_case.input  # case_pass returns "pass", case_fail returns "fail"

        # Should fail because every sample has one failing test case
        with pytest.raises(_pytest.outcomes.Failed) as exc_info:
            test_partial_failure()

        error_message = str(exc_info.value)
        assert "Success rate: 0.00%" in error_message
        assert "Required threshold: 10.00%" in error_message

    def test_sample_success_when_all_test_cases_pass(self):
        """Test that a sample passes only when ALL test cases pass."""
        test_cases = [
            TestCase(id="case_1", input="input1", expected="output1"),
            TestCase(id="case_2", input="input2", expected="output2"),
            TestCase(id="case_3", input="input3", expected="output3"),
        ]

        @evaluate(
            test_cases=test_cases,
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "$.test_case.expected", "actual": "$.output.value"},
            )],
            samples=5,
            success_threshold=1.0,
        )
        def test_all_pass(test_case) -> str:  # noqa
            # Map input to expected output
            mapping = {"input1": "output1", "input2": "output2", "input3": "output3"}
            return mapping[test_case.input]

        # Should pass (all samples pass because all test cases pass)
        test_all_pass()

    def test_mixed_sample_results(self):
        """Test scenario with some samples passing and some failing."""
        call_count = 0
        test_cases = [
            TestCase(id="case_1", input="1", expected="pass"),
            TestCase(id="case_2", input="2", expected="pass"),
        ]

        @evaluate(
            test_cases=test_cases,
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "$.test_case.expected", "actual": "$.output.value"},
            )],
            samples=4,  # 4 samples * 2 test cases = 8 calls
            success_threshold=0.6,  # 60% threshold
        )
        def test_mixed_results(test_case) -> str:  # noqa
            nonlocal call_count
            call_count += 1

            # Pattern: make samples 0 and 2 pass, samples 1 and 3 fail
            # Sample 0: calls 1,2 -> pass,pass -> sample passes
            # Sample 1: calls 3,4 -> fail,pass -> sample fails (one failed)
            # Sample 2: calls 5,6 -> pass,pass -> sample passes
            # Sample 3: calls 7,8 -> fail,pass -> sample fails (one failed)
            # Result: 2/4 = 50% < 60% -> should fail

            if call_count in [1, 2, 5, 6]:  # Samples 0 and 2 pass completely
                return "pass"
            # Samples 1 and 3 have one failure each
            return "fail"

        # Should fail (50% success rate < 60% threshold)
        with pytest.raises(_pytest.outcomes.Failed) as exc_info:
            test_mixed_results()

        error_message = str(exc_info.value)
        assert "Success rate: 50.00%" in error_message
        assert "Required threshold: 60.00%" in error_message

    def test_single_test_case_multiple_samples(self):
        """Test edge case with single test case and multiple samples."""
        call_count = 0

        @evaluate(
            test_cases=[TestCase(id="single", input="test", expected="pass")],
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "$.test_case.expected", "actual": "$.output.value"},
            )],
            samples=10,
            success_threshold=0.7,  # 70% threshold
        )
        def test_single_case_sampling(test_case) -> str:  # noqa
            nonlocal call_count
            call_count += 1
            # Pass 8 out of 10 times (80% > 70%)
            return "pass" if call_count <= 8 else "fail"

        # Should pass (80% success rate > 70% threshold)
        test_single_case_sampling()

        # Verify function was called 10 times
        assert call_count == 10


class TestEvaluateDecoratorAsync:
    """Test async function support with timing validation."""

    def test_async_function_basic(self):
        """Test basic async function support."""
        execution_log = []

        @evaluate(
            test_cases=[TestCase(id="async_test", input="test_input")],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["async"]},
            )],
            samples=3,
            success_threshold=1.0,
        )
        async def test_async_function(test_case) -> str:  # noqa
            execution_log.append(test_case.input)
            await asyncio.sleep(0.01)  # Small delay to verify async execution
            return f"async result for {test_case.input}"

        test_async_function()

        # Verify function was called correct number of times
        assert len(execution_log) == 3
        assert all(inp == "test_input" for inp in execution_log)

    def test_async_function_concurrency_timing(self):
        """Test that async functions run concurrently, not sequentially."""
        num_samples = 50  # each sample has 0.1s delay
        delay = 0.1  # 100ms delay per sample
        num_func_calls = 0
        # Allow some buffer for overhead; still much lower than sequential execution (5s)
        max_concurrent_duration = 1  # 1 second max for all calls

        @evaluate(
            test_cases=[TestCase(id="timing_test", input="test")],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["result"]},
            )],
            samples=num_samples,
            success_threshold=1.0,
        )
        async def test_concurrent_async(test_case) -> str:  # noqa
            nonlocal num_func_calls
            num_func_calls += 1
            await asyncio.sleep(delay)
            return "async result"

        overall_start = time.time()
        test_concurrent_async()
        total_duration = time.time() - overall_start

        assert total_duration < max_concurrent_duration, f"Async functions may not be concurrent (took {total_duration:.3f}s)"  # noqa: E501
        assert num_func_calls == num_samples, "Function should be called exactly num_samples times"

    def test_async_with_multiple_test_cases_timing(self):
        """Test async concurrency with multiple test cases per sample."""
        num_samples = 10
        num_test_cases = 20
        call_count = 0
        test_cases = [
            TestCase(id=f"case_{i}", input=f"input_{i}")
            for i in range(0, num_test_cases)
        ]

        @evaluate(
            test_cases=test_cases,
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["processed"]},
            )],
            samples=num_samples,
            success_threshold=1.0,
        )
        async def test_multi_case_async(test_case) -> str:  # noqa
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # 100ms delay per test case
            print(f"Processing {test_case.input}")
            return f"processed {test_case.input}"

        start_time = time.time()
        test_multi_case_async()
        duration = time.time() - start_time

        assert call_count == num_samples * num_test_cases
        # Should be much less than 20 seconds if concurrent
        assert duration < 0.5, f"Multiple test case async not concurrent (took {duration:.3f}s)"

    def test_async_with_exceptions(self):
        """Test async function exception handling."""
        # Use a thread-safe counter for concurrent execution
        execution_count = threading.Lock()
        call_numbers = []

        @evaluate(
            test_cases=[TestCase(id="exception_test", input="test")],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["success"]},
            )],
            samples=4,
            success_threshold=0.6,  # 60% threshold
        )
        async def test_async_exceptions(test_case) -> str:  # noqa
            await asyncio.sleep(0.01)

            # Use thread-safe counter to assign unique execution numbers
            with execution_count:
                current_call = len(call_numbers) + 1
                call_numbers.append(current_call)

            if current_call <= 3:
                return "success result"
            raise ValueError("Async exception")

        # Should pass (75% success rate > 60% threshold)
        test_async_exceptions()

    def test_async_function_with_simple_fixtures(self):
        """Test async function without complex pytest fixtures."""

        @evaluate(
            test_cases=[TestCase(id="fixture_test", input="test_data")],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["processed", "test_data"]},
            )],
            samples=2,
            success_threshold=1.0,
        )
        async def test_async_simple(test_case: TestCase) -> str:
            # Use test_case parameter
            await asyncio.sleep(0.01)
            return f"processed {test_case.input}"

        test_async_simple()


class TestEvaluateDecoratorPytestFixtures:
    """Test integration with pytest fixtures."""

    def test_fixture_signature_handling(self):
        """Test that decorator correctly handles function signatures with fixtures."""
        @evaluate(
            test_cases=[TestCase(id="test", input="data")],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["result"]},
            )],
            samples=1,
            success_threshold=1.0,
        )
        def test_with_fixture(test_case: TestCase, tmp_path) -> str:  # noqa
            return f"result for {test_case.input}"
        # Verify the decorator created the correct signature
        sig = inspect.signature(test_with_fixture)
        param_names = list(sig.parameters.keys())
        # Should have tmp_path but not test_case (test_case is handled internally)
        assert param_names == ["tmp_path"]
        # Verify parameter has correct annotation
        tmp_path_param = sig.parameters["tmp_path"]
        assert tmp_path_param.annotation == inspect.Parameter.empty  # No annotation in original

    def test_multiple_fixtures_signature_handling(self):
        """Test signature handling with multiple fixtures."""
        @evaluate(
            test_cases=[TestCase(id="test", input="data")],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["result"]},
            )],
            samples=1,
            success_threshold=1.0,
        )
        def test_with_multiple_fixtures(test_case: TestCase, tmp_path, monkeypatch) -> str:  # noqa
            return f"result for {test_case.input}"
        # Verify the decorator created the correct signature
        sig = inspect.signature(test_with_multiple_fixtures)
        param_names = list(sig.parameters.keys())
        # Should have both fixtures but not test_case
        assert param_names == ["tmp_path", "monkeypatch"]

    def test_function_without_test_case_param_with_fixtures(self):
        """Test functions without test_case parameter still work with fixtures."""
        @evaluate(
            test_cases=[TestCase(id="test", input="data")],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["result"]},
            )],
            samples=1,
            success_threshold=1.0,
        )
        def test_no_test_case_with_fixture(tmp_path) -> str:  # noqa
            return "static result"

        # Verify the decorator preserved the original signature
        sig = inspect.signature(test_no_test_case_with_fixture)
        param_names = list(sig.parameters.keys())

        # Should have tmp_path exactly as original
        assert param_names == ["tmp_path"]


class TestEvaluateDecoratorEdgeCasesAdvanced:
    """Test advanced edge cases and error conditions."""

    def test_function_without_test_case_parameter(self):
        """Test functions that don't expect test_case parameter."""

        @evaluate(
            test_cases=[TestCase(id="no_param", input="ignored")],
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "static_result", "actual": "$.output.value"},
            )],
            samples=3,
            success_threshold=1.0,
        )
        def test_no_test_case_param() -> str:
            # Function doesn't take test_case parameter
            return "static_result"

        test_no_test_case_param()  # Should work without issues

    def test_function_with_mixed_parameters(self, tmp_path):  # noqa
        """Test function with test_case in different positions."""
        @evaluate(
            test_cases=[TestCase(id="mixed_params", input="data")],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["data", "tmp"]},
            )],
            samples=2,
            success_threshold=1.0,
        )
        def test_mixed_param_order(tmp_path, test_case: TestCase) -> str:  # noqa
            # test_case comes after fixture parameter
            return f"processed {test_case.input} in {tmp_path}"

        # Verify that parameter order is preserved (except test_case is removed)
        sig = inspect.signature(test_mixed_param_order)
        param_names = list(sig.parameters.keys())
        assert param_names == ["tmp_path"]  # test_case should be removed, tmp_path preserved

    def test_very_large_test_case_expansion(self):
        """Test performance with large test case expansion."""
        execution_count = 0

        @evaluate(
            test_cases=[
                TestCase(id=f"large_{i}", input=f"input_{i}")
                for i in range(5)  # 5 test cases
            ],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["processed"]},
            )],
            samples=10,  # 5 test cases * 10 samples = 50 total calls
            success_threshold=0.9,
        )
        def test_large_expansion(test_case: TestCase) -> str:
            nonlocal execution_count
            execution_count += 1
            return f"processed {test_case.input}"

        test_large_expansion()

        # Verify all 50 calls were made
        assert execution_count == 50

    def test_concurrent_async_with_shared_state(self):
        """Test async functions with shared state don't interfere."""
        results = []

        @evaluate(
            test_cases=[
                TestCase(id="concurrent_1", input="data_1"),
                TestCase(id="concurrent_2", input="data_2"),
            ],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["result"]},
            )],
            samples=5,  # 2 test cases * 5 samples = 10 concurrent calls
            success_threshold=1.0,
        )
        async def test_concurrent_state(test_case: TestCase) -> str:
            # Simulate some async work with shared data structure
            await asyncio.sleep(0.01)
            results.append(test_case.input)
            return f"result for {test_case.input}"

        test_concurrent_state()

        # Verify all 10 calls completed and no data was lost
        assert len(results) == 10
        assert results.count("data_1") == 5
        assert results.count("data_2") == 5


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
        def detailed_failure_function(test_case) -> str:  # noqa
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
        def exception_reporting_function(test_case) -> str:  # noqa
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "success"  # First call succeeds
            raise ValueError(f"Test exception {call_count}")

        with pytest.raises(_pytest.outcomes.Failed) as exc_info:
            exception_reporting_function()

        error_message = str(exc_info.value)
        assert "Test case 0 (id: exception_report) exception: ValueError: Test exception" in error_message  # noqa: E501


class TestEvaluateDecoratorDurationMetadata:
    """Test that duration_seconds metadata is populated in Output objects."""

    def test_sync_function_duration_populated(self):
        """Test that sync functions populate duration_seconds."""

        @evaluate(
            test_cases=[TestCase(id="sync_duration", input="test")],
            checks=[
                Check(
                    type=CheckType.CONTAINS,
                    arguments={"text": "$.output.value", "phrases": ["result"]},
                ),
                Check(
                    type=CheckType.IS_EMPTY,
                    arguments={"value": "$.output.metadata.duration_seconds", "negate": True},
                ),
            ],
            samples=2,
            success_threshold=1.0,
        )
        def sync_test(test_case) -> str:  # noqa
            return "sync result"

        sync_test()

    def test_async_function_duration_populated(self):
        """Test that async functions populate duration_seconds."""

        @evaluate(
            test_cases=[TestCase(id="async_duration", input="test")],
            checks=[
                Check(
                    type=CheckType.CONTAINS,
                    arguments={"text": "$.output.value", "phrases": ["result"]},
                ),
                Check(
                    type=CheckType.IS_EMPTY,
                    arguments={"value": "$.output.metadata.duration_seconds", "negate": True},
                ),
            ],
            samples=2,
            success_threshold=1.0,
        )
        async def async_test(test_case) -> str:  # noqa
            return "async result"

        async_test()

    def test_exception_duration_populated(self):
        """Test that exceptions also populate duration_seconds."""

        @evaluate(
            test_cases=[TestCase(id="exception_duration", input="test")],
            checks=[
                Check(
                    type=CheckType.IS_EMPTY,
                    arguments={"value": "$.output.metadata.duration_seconds", "negate": True},
                ),
            ],
            samples=1,
            success_threshold=1.0,  # Exception outputs should still have duration
        )
        def exception_test(test_case) -> str:  # noqa
            raise ValueError("Test exception")

        exception_test()
