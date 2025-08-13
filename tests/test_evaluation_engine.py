"""Tests for core evaluation engine."""

import pytest
import asyncio
import time
from datetime import datetime, UTC
from typing import Any

from flex_evals import (
    evaluate, TestCase, Output, Check, EvaluationRunResult,
)
from flex_evals.schemas import ExperimentMetadata
from flex_evals.checks.base import BaseCheck, BaseAsyncCheck
from flex_evals.registry import register, clear_registry
from flex_evals.exceptions import ValidationError
from tests.conftest import restore_standard_checks


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


class SlowAsyncCheck(BaseAsyncCheck):
    """Test async check with configurable delay for concurrency testing."""

    async def __call__(self, delay: float = 0.1, **kwargs) -> dict[str, Any]:  # noqa
        await asyncio.sleep(delay)
        return {"passed": True, "delay_used": delay}


class CustomUserCheck(BaseCheck):
    """Custom check to verify parallel worker registry transfer."""

    def __call__(self, test_value: str = "expected", **kwargs) -> dict[str, Any]:  # noqa
        # Return a unique identifier to prove this exact check was executed
        return {
            "passed": True,
            "check_identifier": "custom_user_check_v2",
            "test_value": test_value,
        }


class TestEvaluationEngine:
    """Test evaluation engine functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear registry and register test checks
        clear_registry()
        register("test_check", version="1.0.0")(TestExampleCheck)
        register("test_async_check", version="1.0.0")(TestExampleAsyncCheck)
        register("test_failing_check", version="1.0.0")(TestFailingCheck)
        register("slow_async_check", version="1.0.0")(SlowAsyncCheck)

        # Create test data
        self.test_cases = [
            TestCase(id="test_001", input="What is the capital of France?", expected="Paris"),
            TestCase(id="test_002", input="What is 2 + 2?", expected="4"),
        ]

        self.outputs = [
            Output(value="Paris", id="output_001"),
            Output(value="4", id="output_002"),
        ]

        self.shared_checks = [
            Check(type="test_check", arguments={"expected": "$.test_case.expected", "actual": "$.output.value"}),  # noqa: E501
        ]

    def teardown_method(self):
        """Clean up after tests."""
        clear_registry()
        # Restore standard checks for other tests
        restore_standard_checks()

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
            evaluate(
                self.test_cases,
                [Output(value="Paris", id="output_001")],
                self.shared_checks,
            )  # Only 1 output

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
        assert result.results[0].execution_context.test_case.id == "test_001"
        assert result.results[0].execution_context.output.id == "output_001"
        assert result.results[1].execution_context.test_case.id == "test_002"
        assert result.results[1].execution_context.output.id == "output_002"

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
            TestCase(
                id="test_002",
                input="test",
                checks=[
                    Check(type="test_check", arguments={"expected": "Richland"}),
                    Check(type="test_check", arguments={"expected": "Seattle"}),
                ],
            ),
        ]

        result = evaluate(test_cases_mixed, self.outputs, checks=None)

        assert len(result.results) == 2
        assert len(result.results[0].check_results) == 1
        assert result.results[0].check_results[0].resolved_arguments["expected"]["value"] == "Paris"  # noqa: E501
        assert len(result.results[1].check_results) == 2
        assert result.results[1].check_results[0].resolved_arguments["expected"]["value"] == "Richland"  # noqa: E501
        assert result.results[1].check_results[1].resolved_arguments["expected"]["value"] == "Seattle"  # noqa: E501

    def test_evaluate_mixed_checks_none__missing_check_raises_error(self):
        """Test some test cases have checks, others don't."""
        test_cases_mixed = [
            TestCase(
                id="test_001",
                input="test",
                checks=[Check(type="test_check", arguments={"expected": "Paris"})],
            ),
            TestCase(id="test_002", input="test"),  # No checks
        ]

        with pytest.raises(ValidationError):
            _ = evaluate(test_cases_mixed, self.outputs, checks=None)

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
        assert check_result.check_version == "2.1.0"

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

    def test_async_checks_run_concurrently(self):
        """Test that multiple async checks run in parallel, not sequentially."""
        # Create 20 async checks, each with 0.1s delay
        # If sequential: ~2.0s total (20 * 0.1s)
        # If parallel: ~0.1s total (max delay)
        num_checks = 20
        delay_per_check = 0.1

        concurrent_checks = [
            Check(
                type="slow_async_check",
                arguments={"delay": delay_per_check, "expected": "test"},
            )
            for _ in range(num_checks)
        ]

        start_time = time.time()
        result = evaluate(
            test_cases=[TestCase(id="test_1", input="test input", expected="test")],
            outputs=[Output(value="test", id="output_test")],
            checks=[concurrent_checks],
        )  # Single check per test case
        duration = time.time() - start_time

        # Verify the evaluation completed successfully
        assert result.status == 'completed'
        assert len(result.results) == 1  # Single test case

        # Verify all checks passed
        test_result = result.results[0]
        assert test_result.status == 'completed'
        assert len(test_result.check_results) == num_checks

        for check_result in test_result.check_results:
            assert check_result.results["passed"] is True
            assert check_result.results["delay_used"] == delay_per_check

        # Key assertion: total time should be much closer to single delay than sum of all delays
        # Allow generous buffer for test environment overhead
        max_allowed_time = delay_per_check + 0.2  # buffer for overhead

        # This test will fail if async checks run sequentially instead of concurrently
        assert duration < max_allowed_time, (
            f"Async checks appear to run sequentially "
            f"(took {duration:.3f}s, expected < {max_allowed_time:.1f}s)"
        )

    def test_mixed_sync_async_performance(self):
        """Test that mixed sync/async checks perform optimally."""
        # Create mix: 20 fast sync checks + 20 slow async checks
        # Total time should be dominated by async (concurrent) not sync (sequential)
        num_sync = 20
        num_async = 20
        async_delay = 0.1

        mixed_checks = []

        # Add sync checks (fast)
        for i in range(num_sync):
            mixed_checks.append(
                Check(type="test_check", arguments={"expected": "test", "actual": "test"}),
            )

        # Add async checks (slow)
        for i in range(num_async):
            mixed_checks.append(
                Check(
                    type="slow_async_check",
                    arguments={"delay": async_delay, "expected": "test"},
                ),
            )

        start_time = time.time()
        result = evaluate(
            test_cases=[TestCase(id="test_1", input="test", expected="test")],
            outputs=[Output(value="test", id="output_test")],
            checks=mixed_checks,
        )
        duration = time.time() - start_time

        # Verify success
        assert result.status == 'completed'
        test_result = result.results[0]
        assert len(test_result.check_results) == num_sync + num_async

        # Performance assertion: should complete in ~async_delay time
        # not sync_time + async_delay time
        max_allowed_time = async_delay + 0.3  # Buffer for overhead

        assert duration < max_allowed_time, (
            f"Mixed sync/async execution too slow "
            f"(took {duration:.3f}s, expected < {max_allowed_time:.1f}s)"
        )

    def test_check_order_preservation_mixed(self):
        """Test that check results maintain original order in mixed sync/async scenarios."""
        # Create alternating pattern: sync, async, sync, async, sync
        checks = [
            Check(type="test_check", arguments={"expected": "sync1"}),          # sync
            Check(type="slow_async_check", arguments={"delay": 0.05}),         # async
            Check(type="test_check", arguments={"expected": "sync2"}),          # sync
            Check(type="slow_async_check", arguments={"delay": 0.05}),         # async
            Check(type="test_check", arguments={"expected": "sync3"}),          # sync
        ]

        result = evaluate(
            test_cases=[TestCase(id="test_1", input="test", expected="test")],
            outputs=[Output(value="test", id="output_test")],
            checks=checks,
        )

        # Verify order is preserved
        assert result.status == 'completed'
        test_result = result.results[0]
        assert len(test_result.check_results) == 5

        check_results = test_result.check_results

        # Check that results are in the same order as input checks
        assert check_results[0].check_type == "test_check"      # sync1
        assert check_results[1].check_type == "slow_async_check" # async1
        assert check_results[2].check_type == "test_check"      # sync2
        assert check_results[3].check_type == "slow_async_check" # async2
        assert check_results[4].check_type == "test_check"      # sync3

        # Verify sync checks got their expected values (order-dependent)
        sync_results = [check_results[0], check_results[2], check_results[4]]
        expected_values = ["sync1", "sync2", "sync3"]

        for i, (sync_result, expected) in enumerate(zip(sync_results, expected_values)):
            resolved_expected = sync_result.resolved_arguments["expected"]["value"]
            assert resolved_expected == expected, f"Sync check {i+1} order wrong"

    def test_pure_sync_no_async_overhead(self):
        """Test that pure sync checks don't use async machinery."""
        # This test verifies that sync-only evaluations don't call asyncio.run()
        # We test this indirectly by ensuring very fast execution
        num_checks = 50
        sync_checks = [
            Check(type="test_check", arguments={"expected": "test", "actual": "test"})
            for _ in range(num_checks)
        ]

        start_time = time.time()
        result = evaluate(
            test_cases=[TestCase(id="test_1", input="test", expected="test")],
            outputs=[Output(value="test", id="output_test")],
            checks=sync_checks,
        )
        duration = time.time() - start_time

        # Verify success
        assert result.status == 'completed'
        assert len(result.results[0].check_results) == num_checks

        # Should complete very quickly (no async overhead)
        max_allowed_time = 0.05  # 50ms should be plenty for 50 simple sync checks

        assert duration < max_allowed_time, (
            f"Pure sync execution too slow, may have async overhead "
            f"(took {duration:.4f}s, expected < {max_allowed_time:.2f}s)"
        )

    def test_evaluate_with_async_concurrency_limit(self):
        """Test async concurrency control with max_async_concurrent parameter."""
        # Create 10 async checks with delay
        num_checks = 10
        delay_per_check = 0.05
        max_concurrent = 3

        concurrent_checks = [
            Check(
                type="slow_async_check",
                arguments={"delay": delay_per_check, "expected": "test"},
            )
            for _ in range(num_checks)
        ]

        start_time = time.time()
        result = evaluate(
            test_cases=[TestCase(id="test_1", input="test input", expected="test")],
            outputs=[Output(value="test", id="output_test")],
            checks=concurrent_checks,
            max_async_concurrent=max_concurrent,
        )
        duration = time.time() - start_time

        # Verify the evaluation completed successfully
        assert result.status == 'completed'
        assert len(result.results) == 1
        test_result = result.results[0]
        assert len(test_result.check_results) == num_checks

        # All checks should pass
        for check_result in test_result.check_results:
            assert check_result.results["passed"] is True

        # Duration should be roughly (num_checks / max_concurrent) * delay_per_check
        # With max_concurrent=3 and 10 checks, we expect ~4 batches: 10/3 = 3.33 -> 4 batches
        # So duration should be around 4 * 0.05 = 0.2s
        expected_duration = (num_checks / max_concurrent) * delay_per_check
        max_allowed_time = expected_duration + 0.15  # Buffer for overhead

        assert duration < max_allowed_time, (
            f"Async concurrency limit not working properly "
            f"(took {duration:.3f}s, expected < {max_allowed_time:.3f}s)"
        )

    def test_evaluate_with_parallel_workers(self):
        """Test parallel test case processing with max_parallel_workers parameter."""
        # Create multiple test cases to distribute across workers
        num_test_cases = 8
        max_workers = 4

        test_cases = [
            TestCase(id=f"test_{i}", input=f"input_{i}", expected=f"output_{i}")
            for i in range(num_test_cases)
        ]

        outputs = [
            Output(value=f"output_{i}", id=f"output_{i}")
            for i in range(num_test_cases)
        ]

        # Use sync checks to make timing more predictable
        checks = [
            Check(
                type="test_check",
                arguments={"expected": "$.test_case.expected", "actual": "$.output.value"},
            ),
        ]

        start_time = time.time()
        result = evaluate(
            test_cases=test_cases,
            outputs=outputs,
            checks=checks,
            max_parallel_workers=max_workers,
        )
        time.time() - start_time

        # Debug: print result details if there are errors
        if result.status == 'error':
            for test_result in result.results:
                if test_result.status == 'error':
                    for check_result in test_result.check_results:
                        if check_result.status == 'error' and check_result.error:
                            print(f"Error in {test_result.execution_context.test_case.id}: {check_result.error.message}")  # noqa: E501

        # Verify the evaluation completed successfully
        assert result.status == 'completed'
        assert len(result.results) == num_test_cases

        # All test cases should pass (matching expected values)
        for i, test_result in enumerate(result.results):
            assert test_result.execution_context.test_case.id == f"test_{i}"
            assert test_result.execution_context.output.id == f"output_{i}"
            assert test_result.status == 'completed'
            assert len(test_result.check_results) == 1
            assert test_result.check_results[0].results["passed"] is True

        # Results should be in the same order as input
        for i, test_result in enumerate(result.results):
            assert test_result.execution_context.test_case.id == f"test_{i}"

    def test_evaluate_mixed_parallelization_and_async_concurrency(self):
        """Test combination of parallel workers and async concurrency control."""
        # Create test cases with async checks
        num_test_cases = 6
        num_async_checks_per_case = 4
        max_workers = 3
        max_async_concurrent = 2
        delay_per_check = 0.03

        test_cases = [
            TestCase(id=f"test_{i}", input=f"input_{i}", expected="test")
            for i in range(num_test_cases)
        ]

        outputs = [Output(value="test", id=f"output_{i}") for i in range(num_test_cases)]

        # Each test case gets multiple async checks
        checks = [
            Check(
                type="slow_async_check",
                arguments={"delay": delay_per_check, "expected": "test"},
            )
            for _ in range(num_async_checks_per_case)
        ]

        start_time = time.time()
        result = evaluate(
            test_cases=test_cases,
            outputs=outputs,
            checks=checks,
            max_async_concurrent=max_async_concurrent,
            max_parallel_workers=max_workers,
        )
        duration = time.time() - start_time

        # Verify the evaluation completed successfully
        assert result.status == 'completed'
        assert len(result.results) == num_test_cases

        # All checks should pass
        for test_result in result.results:
            assert test_result.status == 'completed'
            assert len(test_result.check_results) == num_async_checks_per_case
            for check_result in test_result.check_results:
                assert check_result.results["passed"] is True

        # Verify reasonable performance with both parallelization features
        # This should be significantly faster than serial execution
        max_allowed_time = 0.5  # Should be much faster than serial
        assert duration < max_allowed_time, (
            f"Combined parallelization too slow "
            f"(took {duration:.3f}s, expected < {max_allowed_time:.1f}s)"
        )

    def test_evaluate_default_parameters_unchanged(self):
        """Test that default behavior is unchanged when new parameters not specified."""
        # This test ensures backward compatibility
        result1 = evaluate(self.test_cases, self.outputs, self.shared_checks)

        # Same call with explicit defaults
        result2 = evaluate(
            self.test_cases,
            self.outputs,
            self.shared_checks,
            max_async_concurrent=None,
            max_parallel_workers=1,
        )

        # Results should be identical (except for evaluation_id and timestamps)
        assert result1.status == result2.status
        assert len(result1.results) == len(result2.results)

        for r1, r2 in zip(result1.results, result2.results):
            assert r1.execution_context.test_case.id == r2.execution_context.test_case.id
            assert r1.status == r2.status
            assert len(r1.check_results) == len(r2.check_results)

    def test_evaluate_single_worker_same_as_serial(self):
        """Test that max_parallel_workers=1 produces same results as serial execution."""
        # Test with both sync and async checks
        mixed_checks = [
            Check(type="test_check", arguments={"expected": "$.test_case.expected", "actual": "$.output.value"}),  # noqa: E501
            Check(type="test_async_check", arguments={"expected": "$.test_case.expected", "actual": "$.output.value"}),  # noqa: E501
        ]

        # Serial execution (default)
        result_serial = evaluate(self.test_cases, self.outputs, mixed_checks)

        # Parallel with 1 worker (should be equivalent)
        result_parallel = evaluate(
            self.test_cases,
            self.outputs,
            mixed_checks,
            max_parallel_workers=1,
        )

        # Results should be functionally identical
        assert result_serial.status == result_parallel.status
        assert len(result_serial.results) == len(result_parallel.results)

        for r1, r2 in zip(result_serial.results, result_parallel.results):
            assert r1.execution_context.test_case.id == r2.execution_context.test_case.id
            assert r1.status == r2.status
            assert len(r1.check_results) == len(r2.check_results)

            # Check results should have same pass/fail status
            for c1, c2 in zip(r1.check_results, r2.check_results):
                assert c1.check_type == c2.check_type
                assert c1.status == c2.status
                assert c1.results.get("passed") == c2.results.get("passed")

    def test_evaluate_async_concurrency_no_limit(self):
        """Test that max_async_concurrent=None allows unlimited concurrency."""
        # Create many async checks that should all run concurrently
        num_checks = 20
        delay_per_check = 0.05

        concurrent_checks = [
            Check(
                type="slow_async_check",
                arguments={"delay": delay_per_check, "expected": "test"},
            )
            for _ in range(num_checks)
        ]

        start_time = time.time()
        result = evaluate(
            test_cases=[TestCase(id="test_1", input="test", expected="test")],
            outputs=[Output(value="test", id="output_test")],
            checks=concurrent_checks,
            max_async_concurrent=None,  # No limit
        )
        duration = time.time() - start_time

        # Verify success
        assert result.status == 'completed'

        # Should complete in roughly delay_per_check time (all concurrent)
        max_allowed_time = delay_per_check + 0.2  # Buffer
        assert duration < max_allowed_time, (
            f"Unlimited async concurrency not working "
            f"(took {duration:.3f}s, expected < {max_allowed_time:.1f}s)"
        )

    def test_evaluate_custom_checks_available_in_parallel_workers(self):
        """
        Test that custom user-registered checks work correctly in parallel workers.

        Note: Custom checks must be defined at module level (not locally) to be
        serializable for multiprocessing. This is a limitation of Python's pickle module.
        """
        # Register the module-level custom check
        register("custom_user_check", version="2.0.0")(CustomUserCheck)

        # Create test cases that use the custom check
        test_cases = [
            TestCase(id=f"test_{i}", input=f"input_{i}", expected="test")
            for i in range(4)
        ]

        outputs = [Output(value="test", id=f"output_{i}") for i in range(4)]

        # Use per-test-case checks with unique arguments per test case
        per_test_case_checks = [
            [Check(
                type="custom_user_check",
                arguments={"test_value": f"custom_value_{i}"},
            )]
            for i in range(4)
        ]

        # Execute with parallel workers
        result = evaluate(
            test_cases=test_cases,
            outputs=outputs,
            checks=per_test_case_checks,
            max_parallel_workers=2,  # Force parallel execution
        )

        # Verify the evaluation completed successfully
        assert result.status == 'completed'
        assert len(result.results) == 4

        # Verify each test case used the custom check correctly
        for i, test_result in enumerate(result.results):
            assert test_result.status == 'completed'
            assert len(test_result.check_results) == 1

            check_result = test_result.check_results[0]
            assert check_result.check_type == "custom_user_check"
            assert check_result.status == 'completed'

            # Verify the custom check actually executed (not an error fallback)
            assert check_result.results["passed"] is True
            assert check_result.results["check_identifier"] == "custom_user_check_v2"
            assert check_result.results["test_value"] == f"custom_value_{i}"

    def test_output_id_preservation(self):
        """Test that output IDs are preserved through evaluation process."""
        # Create test cases with specific output IDs
        test_cases = [
            TestCase(id="test_unique", input="test", expected="test"),
        ]

        outputs = [
            Output(value="test", id="output_unique_123"),
        ]

        checks = [
            Check(type="test_check", arguments={"expected": "test", "actual": "$.output.value"}),
        ]

        result = evaluate(test_cases, outputs, checks)

        # Verify output ID is preserved in execution context
        assert result.status == 'completed'
        assert len(result.results) == 1

        test_result = result.results[0]
        assert test_result.execution_context.test_case.id == "test_unique"
        assert test_result.execution_context.output.id == "output_unique_123"
        assert test_result.execution_context.output.value == "test"

    def test_jsonpath_resolution_per_test_case_checks(self):
        """
        Test JSONPath correctly resolves different values for each test case with per-test-case
        checks.

        This test validates that:
        1. Same JSONPath expressions extract different values from different contexts
        2. Shared resolver cache doesn't cause value leakage between test cases
        3. Each test case's checks operate on their own independent execution context
        4. Complex nested JSONPath expressions work correctly across different data structures
        """
        # Create test cases with different data structures and per-test-case checks
        test_cases = [
            TestCase(
                id="geography_test",
                input={
                    "question": "What is the capital of France?",
                    "user_id": 1001,
                    "difficulty": "easy",
                    "metadata": {"category": "geography", "points": 10},
                },
                expected="Paris",
                checks=[
                    # Check 1: Basic output validation
                    Check(
                        type="test_check",
                        arguments={
                            "expected": "$.test_case.expected",
                            "actual": "$.output.value.answer",
                        },
                    ),
                    # Check 2: User context validation
                    Check(
                        type="test_check",
                        arguments={
                            "expected": "$.test_case.input.user_id",
                            "actual": "$.output.metadata.user_id",
                        },
                    ),
                    # Check 3: Category validation
                    Check(
                        type="test_check",
                        arguments={
                            "expected": "geography",
                            "actual": "$.test_case.input.metadata.category",
                        },
                    ),
                ],
            ),
            TestCase(
                id="math_test",
                input={
                    "question": "What is 15 * 7?",
                    "user_id": 2002,
                    "difficulty": "medium",
                    "metadata": {"category": "mathematics", "points": 20},
                },
                expected="105",
                checks=[
                    # Check 1: Answer validation
                    Check(
                        type="test_check",
                        arguments={
                            "expected": "$.test_case.expected",
                            "actual": "$.output.value.result",
                        },
                    ),
                    # Check 2: Difficulty validation
                    Check(
                        type="test_check",
                        arguments={
                            "expected": "medium",
                            "actual": "$.test_case.input.difficulty",
                        },
                    ),
                    # Check 3: Points validation
                    Check(
                        type="test_check",
                        arguments={
                            "expected": "$.test_case.input.metadata.points",
                            "actual": "$.output.metadata.points_awarded",
                        },
                    ),
                ],
            ),
            TestCase(
                id="science_test",
                input={
                    "question": "What is the chemical symbol for gold?",
                    "user_id": 3003,
                    "difficulty": "hard",
                    "metadata": {"category": "chemistry", "points": 30},
                },
                expected="Au",
                checks=[
                    # Check 1: Symbol validation
                    Check(
                        type="test_check",
                        arguments={
                            "expected": "Au",
                            "actual": "$.output.value.symbol",
                        },
                    ),
                    # Check 2: Question type validation
                    Check(
                        type="test_check",
                        arguments={
                            "expected": "chemistry",
                            "actual": "$.output.metadata.subject",
                        },
                    ),
                ],
            ),
        ]

        # Create corresponding outputs with different structures
        outputs = [
            Output(
                value={
                    "answer": "Paris",
                    "confidence": 0.95,
                    "reasoning": "Paris is the capital city of France",
                },
                metadata={
                    "user_id": 1001,
                    "processing_time": 150,
                    "model": "geography-expert",
                },
            ),
            Output(
                value={
                    "result": "105",
                    "calculation": "15 * 7 = 105",
                    "steps": ["15", "*", "7", "=", "105"],
                },
                metadata={
                    "user_id": 2002,
                    "points_awarded": 20,
                    "processing_time": 200,
                },
            ),
            Output(
                value={
                    "symbol": "Au",
                    "element_name": "Gold",
                    "atomic_number": 79,
                },
                metadata={
                    "user_id": 3003,
                    "subject": "chemistry",
                    "processing_time": 180,
                },
            ),
        ]

        # Execute evaluation using per-test-case checks
        result = evaluate(test_cases, outputs, checks=None)

        # Verify evaluation succeeded
        assert result.status == "completed"
        assert len(result.results) == 3

        # Verify test case 1 (Geography) - resolved arguments should contain correct values
        geo_result = result.results[0]
        assert geo_result.status == "completed"
        assert len(geo_result.check_results) == 3

        # Check 1: expected="Paris", actual="Paris"
        geo_check1 = geo_result.check_results[0]
        assert geo_check1.status == "completed"
        assert geo_check1.resolved_arguments["expected"]["jsonpath"] == "$.test_case.expected"
        assert geo_check1.resolved_arguments["expected"]["value"] == "Paris"
        assert geo_check1.resolved_arguments["actual"]["jsonpath"] == "$.output.value.answer"
        assert geo_check1.resolved_arguments["actual"]["value"] == "Paris"
        assert geo_check1.results["passed"] is True

        # Check 2: expected=1001, actual=1001
        geo_check2 = geo_result.check_results[1]
        assert geo_check2.resolved_arguments["expected"]["jsonpath"] == "$.test_case.input.user_id"
        assert geo_check2.resolved_arguments["expected"]["value"] == 1001
        assert geo_check2.resolved_arguments["actual"]["jsonpath"] == "$.output.metadata.user_id"
        assert geo_check2.resolved_arguments["actual"]["value"] == 1001
        assert geo_check2.results["passed"] is True

        # Check 3: expected="geography", actual="geography"
        geo_check3 = geo_result.check_results[2]
        assert geo_check3.resolved_arguments["expected"]["value"] == "geography"
        assert geo_check3.resolved_arguments["actual"]["jsonpath"] == "$.test_case.input.metadata.category"  # noqa: E501
        assert geo_check3.resolved_arguments["actual"]["value"] == "geography"
        assert geo_check3.results["passed"] is True

        # Verify test case 2 (Math) - different values extracted from different context
        math_result = result.results[1]
        assert math_result.status == "completed"
        assert len(math_result.check_results) == 3

        # Check 1: expected="105", actual="105"
        math_check1 = math_result.check_results[0]
        assert math_check1.resolved_arguments["expected"]["jsonpath"] == "$.test_case.expected"
        assert math_check1.resolved_arguments["expected"]["value"] == "105"
        assert math_check1.resolved_arguments["actual"]["jsonpath"] == "$.output.value.result"
        assert math_check1.resolved_arguments["actual"]["value"] == "105"
        assert math_check1.results["passed"] is True

        # Check 2: expected="medium", actual="medium"
        math_check2 = math_result.check_results[1]
        assert math_check2.resolved_arguments["expected"]["value"] == "medium"
        assert math_check2.resolved_arguments["actual"]["jsonpath"] == "$.test_case.input.difficulty"  # noqa: E501
        assert math_check2.resolved_arguments["actual"]["value"] == "medium"
        assert math_check2.results["passed"] is True

        # Check 3: expected=20, actual=20
        math_check3 = math_result.check_results[2]
        assert math_check3.resolved_arguments["expected"]["jsonpath"] == "$.test_case.input.metadata.points"  # noqa: E501
        assert math_check3.resolved_arguments["expected"]["value"] == 20
        assert math_check3.resolved_arguments["actual"]["jsonpath"] == "$.output.metadata.points_awarded"  # noqa: E501
        assert math_check3.resolved_arguments["actual"]["value"] == 20
        assert math_check3.results["passed"] is True

        # Verify test case 3 (Science) - yet different values from different context
        science_result = result.results[2]
        assert science_result.status == "completed"
        assert len(science_result.check_results) == 2

        # Check 1: expected="Au", actual="Au"
        science_check1 = science_result.check_results[0]
        assert science_check1.resolved_arguments["expected"]["value"] == "Au"
        assert science_check1.resolved_arguments["actual"]["jsonpath"] == "$.output.value.symbol"
        assert science_check1.resolved_arguments["actual"]["value"] == "Au"
        assert science_check1.results["passed"] is True

        # Check 2: expected="chemistry", actual="chemistry"
        science_check2 = science_result.check_results[1]
        assert science_check2.resolved_arguments["expected"]["value"] == "chemistry"
        assert science_check2.resolved_arguments["actual"]["jsonpath"] == "$.output.metadata.subject"  # noqa: E501
        assert science_check2.resolved_arguments["actual"]["value"] == "chemistry"
        assert science_check2.results["passed"] is True

        # Verify that the same JSONPath expressions extracted different values
        # For example, "$.test_case.expected" should have extracted different values:
        # - "Paris" for geography test
        # - "105" for math test
        # - No usage in science test (different check structure)

        # Verify "$.test_case.input.user_id" extracted different user IDs:
        assert geo_check2.resolved_arguments["expected"]["value"] == 1001  # Geography user
        # Math test doesn't use user_id in checks, but we verified it has 2002 in input
