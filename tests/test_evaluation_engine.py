"""Tests for core evaluation engine."""

import pytest
from datetime import datetime, UTC
from typing import Any

from flex_evals.engine import evaluate
from flex_evals.schemas import (
    TestCase, Output, Check, EvaluationRunResult,
    ExperimentMetadata,
)
from flex_evals.checks.base import BaseCheck, BaseAsyncCheck
from flex_evals.registry import register, clear_registry
from flex_evals.exceptions import ValidationError


class TestExampleCheck(BaseCheck):
    """Test check for evaluation engine testing."""

    def __call__(self, expected: str = "Paris", actual: str | None = None) -> dict[str, Any]:
        # For test purposes, if actual is not provided, we'll handle it in the integration
        return {"passed": str(actual) == str(expected)}


class TestExampleAsyncCheck(BaseAsyncCheck):
    """Test async check for evaluation engine testing."""

    async def __call__(self, expected: str = "Paris", actual: str | None = None) -> dict[str, Any]:
        # For test purposes, if actual is not provided, we'll handle it in the integration
        return {"passed": str(actual) == str(expected)}


class TestFailingCheck(BaseCheck):
    """Test check that always fails for error testing."""

    def __call__(self, **kwargs) -> dict[str, Any]:  # noqa
        raise RuntimeError("This check always fails")


class TestEvaluationEngine:
    """Test evaluation engine functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear registry and register test checks
        clear_registry()
        register("test_check", version="1.0.0")(TestExampleCheck)
        register("test_async_check", version="1.0.0")(TestExampleAsyncCheck)
        register("test_failing_check", version="1.0.0")(TestFailingCheck)

        # Create test data
        self.test_cases = [
            TestCase(id="test_001", input="What is the capital of France?", expected="Paris"),
            TestCase(id="test_002", input="What is 2 + 2?", expected="4"),
        ]

        self.outputs = [
            Output(value="Paris"),
            Output(value="4"),
        ]

        self.shared_checks = [
            Check(type="test_check", arguments={"expected": "$.test_case.expected", "actual": "$.output.value"}),  # noqa: E501
        ]

    def teardown_method(self):
        """Clean up after tests."""
        clear_registry()

    def test_evaluate_function_signature(self):
        """Test function accepts correct parameters."""
        result = evaluate(self.test_cases, self.outputs, self.shared_checks)

        assert isinstance(result, EvaluationRunResult)
        assert result.evaluation_id is not None
        assert isinstance(result.started_at, datetime)
        assert isinstance(result.completed_at, datetime)

    def test_evaluate_length_constraint(self):
        """Test test_cases and outputs same length validation."""
        # Mismatched lengths should raise ValidationError
        with pytest.raises(ValidationError, match="test_cases and outputs must have same length"):
            evaluate(self.test_cases, [Output(value="Paris")], self.shared_checks)  # Only 1 output

    def test_evaluate_empty_inputs(self):
        """Test behavior with empty test_cases and outputs."""
        with pytest.raises(ValidationError, match="At least one test case is required"):
            evaluate([], [], [])

    def test_evaluate_association(self):
        """Test test_cases[i] paired with outputs[i]."""
        checks = [
            Check(type="test_check", arguments={"expected": "Paris", "actual": "$.output.value"}),
        ]

        result = evaluate(self.test_cases, self.outputs, checks)

        # Verify each test case is paired with correct output
        assert len(result.results) == 2
        assert result.results[0].test_case_id == "test_001"
        assert result.results[1].test_case_id == "test_002"

        # First test case (Paris) should pass, second (4) should fail
        assert result.results[0].check_results[0].results["passed"] is True
        assert result.results[1].check_results[0].results["passed"] is False

    def test_evaluate_shared_checks(self):
        """Test same checks applied to all test cases."""
        result = evaluate(self.test_cases, self.outputs, self.shared_checks)

        assert len(result.results) == 2

        # Both test cases should have the same check applied
        for test_result in result.results:
            assert len(test_result.check_results) == 1
            assert test_result.check_results[0].check_type == "test_check"

    def test_evaluate_per_test_case_checks(self):
        """Test different checks per test case."""
        per_test_case_checks = [
            [Check(type="test_check", arguments={"expected": "Paris", "actual": "$.output.value"})],  # noqa: E501
            [Check(type="test_check", arguments={"expected": "4", "actual": "$.output.value"})],
        ]

        result = evaluate(self.test_cases, self.outputs, per_test_case_checks)

        assert len(result.results) == 2

        # Both should pass with their specific expected values
        assert result.results[0].check_results[0].results["passed"] is True
        assert result.results[1].check_results[0].results["passed"] is True

    def test_evaluate_per_test_case_checks_length_mismatch(self):
        """Test per-test-case checks length validation."""
        per_test_case_checks = [
            [Check(type="test_check", arguments={"expected": "Paris"})],
            # Missing second check list
        ]

        with pytest.raises(ValidationError, match="checks list must have same length as test_cases"):  # noqa: E501
            evaluate(self.test_cases, self.outputs, per_test_case_checks)

    def test_evaluate_checks_none_fallback(self):
        """Test using TestCase.checks when checks=None."""
        test_cases_with_checks = [
            TestCase(
                id="test_001",
                input="test",
                checks=[Check(type="test_check", arguments={"expected": "Paris", "actual": "$.output.value"})],  # noqa: E501
            ),
            TestCase(
                id="test_002",
                input="test",
                checks=[Check(type="test_check", arguments={"expected": "4", "actual": "$.output.value"})],  # noqa: E501
            ),
        ]

        result = evaluate(test_cases_with_checks, self.outputs, checks=None)

        assert len(result.results) == 2
        assert result.results[0].check_results[0].results["passed"] is True
        assert result.results[1].check_results[0].results["passed"] is True

    def test_evaluate_mixed_checks_none(self):
        """Test some test cases have checks, others don't."""
        test_cases_mixed = [
            TestCase(
                id="test_001",
                input="test",
                checks=[Check(type="test_check", arguments={"expected": "Paris"})],
            ),
            TestCase(id="test_002", input="test"),  # No checks
        ]

        result = evaluate(test_cases_mixed, self.outputs, checks=None)

        assert len(result.results) == 2
        assert len(result.results[0].check_results) == 1  # Has check
        assert len(result.results[1].check_results) == 0  # No checks

    def test_evaluate_sync_only(self):
        """Test pure sync execution path."""
        result = evaluate(self.test_cases, self.outputs, self.shared_checks)

        assert result.status == 'completed'
        assert len(result.results) == 2

        # Verify all checks executed successfully
        for test_result in result.results:
            assert test_result.status == 'completed'
            for check_result in test_result.check_results:
                assert check_result.status == 'completed'

    def test_evaluate_async_detection(self):
        """Test async checks trigger async execution."""
        async_checks = [
            Check(type="test_async_check", arguments={"expected": "$.test_case.expected", "actual": "$.output.value"}),  # noqa: E501
        ]

        result = evaluate(self.test_cases, self.outputs, async_checks)

        assert result.status == 'completed'
        assert len(result.results) == 2

        # Verify async checks executed successfully
        for test_result in result.results:
            assert test_result.status == 'completed'
            assert test_result.check_results[0].check_type == "test_async_check"

    def test_evaluate_mixed_async_sync(self):
        """Test evaluation with both async and sync checks."""
        mixed_checks = [
            Check(type="test_check", arguments={"expected": "$.test_case.expected"}),
            Check(type="test_async_check", arguments={"expected": "$.test_case.expected"}),
        ]

        result = evaluate(self.test_cases, self.outputs, mixed_checks)

        assert result.status == 'completed'

        # Both check types should execute in async mode
        for test_result in result.results:
            assert len(test_result.check_results) == 2
            check_types = [cr.check_type for cr in test_result.check_results]
            assert "test_check" in check_types
            assert "test_async_check" in check_types

    def test_evaluate_experiment_metadata(self):
        """Test experiment metadata included in result."""
        experiment = ExperimentMetadata(
            name="test_experiment",
            metadata={"version": "1.0", "purpose": "testing"},
        )

        result = evaluate(self.test_cases, self.outputs, self.shared_checks, experiment)

        assert result.experiment == experiment
        assert result.experiment.name == "test_experiment"
        assert result.experiment.metadata["version"] == "1.0"

    def test_evaluate_unique_evaluation_id(self):
        """Test each evaluation gets unique ID."""
        result1 = evaluate(self.test_cases, self.outputs, self.shared_checks)
        result2 = evaluate(self.test_cases, self.outputs, self.shared_checks)

        assert result1.evaluation_id != result2.evaluation_id

    def test_evaluate_timestamps(self):
        """Test started_at and completed_at are set correctly."""
        result = evaluate(self.test_cases, self.outputs, self.shared_checks)

        assert isinstance(result.started_at, datetime)
        assert isinstance(result.completed_at, datetime)
        assert result.completed_at >= result.started_at
        assert result.started_at.tzinfo == UTC
        assert result.completed_at.tzinfo == UTC

    def test_evaluate_check_version_preservation(self):
        """Test check version is preserved in results."""
        checks_with_version = [
            Check(
                type="test_check",
                arguments={"expected": "Paris"},
                version="2.1.0",
            ),
        ]

        result = evaluate(self.test_cases, self.outputs, checks_with_version)

        check_result = result.results[0].check_results[0]
        assert check_result.metadata.check_version == "2.1.0"

    def test_evaluate_jsonpath_resolution(self):
        """Test JSONPath expressions are resolved correctly."""
        checks_with_jsonpath = [
            Check(
                type="test_check",
                arguments={
                    "expected": "$.test_case.expected",
                    "actual": "$.output.value",
                },
            ),
        ]

        result = evaluate(self.test_cases, self.outputs, checks_with_jsonpath)

        check_result = result.results[0].check_results[0]

        # Verify JSONPath was resolved
        assert check_result.resolved_arguments["expected"]["jsonpath"] == "$.test_case.expected"
        assert check_result.resolved_arguments["expected"]["value"] == "Paris"
        assert check_result.resolved_arguments["actual"]["jsonpath"] == "$.output.value"
        assert check_result.resolved_arguments["actual"]["value"] == "Paris"

    def test_evaluate_error_handling(self):
        """Test error handling for failing checks."""
        failing_checks = [
            Check(type="test_failing_check", arguments={}),
        ]

        result = evaluate(self.test_cases, self.outputs, failing_checks)

        assert result.status == 'error'  # Overall status should be error

        # All test case results should have errors
        for test_result in result.results:
            assert test_result.status == 'error'
            check_result = test_result.check_results[0]
            assert check_result.status == 'error'
            assert check_result.error is not None
            assert "This check always fails" in check_result.error.message

    def test_evaluate_unregistered_check_type(self):
        """Test handling of unregistered check types."""
        invalid_checks = [
            Check(type="nonexistent_check", arguments={}),
        ]

        result = evaluate(self.test_cases, self.outputs, invalid_checks)

        assert result.status == 'error'

        # Should have error results for unregistered check
        for test_result in result.results:
            assert test_result.status == 'error'
            check_result = test_result.check_results[0]
            assert check_result.status == 'error'
            assert "not registered" in check_result.error.message

    def test_evaluate_summary_statistics(self):
        """Test summary statistics are computed correctly."""
        # Mix of passing and failing checks
        mixed_checks = [
            Check(type="test_check", arguments={"expected": "Paris"}),  # Will pass for first, fail for second  # noqa: E501
            Check(type="test_failing_check", arguments={}),  # Will fail for both
        ]

        result = evaluate(self.test_cases, self.outputs, mixed_checks)

        assert result.summary.total_test_cases == 2
        assert result.summary.error_test_cases == 2  # Both have failing checks
        assert result.summary.completed_test_cases == 0
        assert result.summary.skipped_test_cases == 0

    def test_evaluate_status_computation(self):
        """Test overall status computation from test case results."""
        # All passing
        passing_checks = [
            Check(type="test_check", arguments={"expected": "$.test_case.expected"}),
        ]
        result = evaluate(self.test_cases, self.outputs, passing_checks)
        assert result.status == 'completed'

        # Some failing
        failing_checks = [
            Check(type="test_failing_check", arguments={}),
        ]
        result = evaluate(self.test_cases, self.outputs, failing_checks)
        assert result.status == 'error'
