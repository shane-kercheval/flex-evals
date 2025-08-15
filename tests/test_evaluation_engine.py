"""Tests for core evaluation engine."""

import pytest
import asyncio
import time
from datetime import datetime, UTC
from typing import Any

from flex_evals import (
    evaluate,
    TestCase,
    Output,
    Check,
    EvaluationRunResult,
    ExperimentMetadata,
    AttributeExistsCheck,
    ContainsCheck,
    EqualsCheck,
    ExactMatchCheck,
    IsEmptyCheck,
    RegexCheck,
    ThresholdCheck,
    BaseCheck,
    BaseAsyncCheck,
    JSONPath,
    register,
    ValidationError,
)
from pydantic import field_validator
from flex_evals.registry import clear_registry
from tests.conftest import restore_standard_checks


class TestExampleCheck(BaseCheck):
    """Test check for evaluation engine testing."""

    # Pydantic fields with validation - can be literals or JSONPath objects
    expected: str | JSONPath = "Paris"
    actual: str | JSONPath | None = None

    @field_validator('expected', 'actual', mode='before')
    @classmethod
    def convert_jsonpath(cls, v):  # noqa: ANN001
        """Convert JSONPath-like strings to JSONPath objects."""
        if isinstance(v, str) and v.startswith('$.'):
            return JSONPath(expression=v)
        return v

    def __call__(self) -> dict[str, Any]:
        """Execute test check using resolved Pydantic fields."""
        # Validate that all fields are resolved (no JSONPath objects remain)
        if isinstance(self.expected, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'expected' field: {self.expected}")
        if isinstance(self.actual, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'actual' field: {self.actual}")

        # For test purposes, compare string representations
        return {"passed": str(self.actual) == str(self.expected)}


class TestExampleAsyncCheck(BaseAsyncCheck):
    """Test async check for evaluation engine testing."""

    # Pydantic fields with validation - can be literals or JSONPath objects
    expected: str | JSONPath = "Paris"
    actual: str | JSONPath | None = None

    @field_validator('expected', 'actual', mode='before')
    @classmethod
    def convert_jsonpath(cls, v):  # noqa: ANN001
        """Convert JSONPath-like strings to JSONPath objects."""
        if isinstance(v, str) and v.startswith('$.'):
            return JSONPath(expression=v)
        return v

    async def __call__(self) -> dict[str, Any]:
        """Execute async test check using resolved Pydantic fields."""
        # Validate that all fields are resolved (no JSONPath objects remain)
        if isinstance(self.expected, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'expected' field: {self.expected}")
        if isinstance(self.actual, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'actual' field: {self.actual}")

        # For test purposes, compare string representations
        return {"passed": str(self.actual) == str(self.expected)}


class TestFailingCheck(BaseCheck):
    """Test check that always fails for error testing."""

    def __call__(self) -> dict[str, Any]:
        raise RuntimeError("This check always fails")


class SlowAsyncCheck(BaseAsyncCheck):
    """Test async check with configurable delay for concurrency testing."""

    # Pydantic fields with validation
    delay: float = 0.1

    async def __call__(self) -> dict[str, Any]:
        await asyncio.sleep(self.delay)
        return {"passed": True, "delay_used": self.delay}


class CustomUserCheck(BaseCheck):
    """Custom check to verify parallel worker registry transfer."""

    # Pydantic fields with validation
    test_value: str = "expected"

    def __call__(self) -> dict[str, Any]:
        # Return a unique identifier to prove this exact check was executed
        return {
            "passed": True,
            "check_identifier": "custom_user_check_v2",
            "test_value": self.test_value,
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
                version="1.0.0",
            ),
        ]

        result = evaluate(self.test_cases, self.outputs, checks_with_version)

        check_result = result.results[0].check_results[0]
        assert check_result.check_version == "1.0.0"

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
        """Test that unregistered check types fail early with clear error."""
        invalid_checks = [
            Check(type="nonexistent_check", arguments={}),
        ]

        with pytest.raises(ValueError, match="Check type 'nonexistent_check' is not registered"):
            evaluate(self.test_cases, self.outputs, invalid_checks)

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
                arguments={"delay": delay_per_check},
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
                    arguments={"delay": async_delay},
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
                arguments={"delay": delay_per_check},
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

        # Assert all components completed successfully
        assert result.status == 'completed', (
            f"Expected result status 'completed', got '{result.status}'"
        )

        # Verify all test case results completed successfully
        for i, test_result in enumerate(result.results):
            assert test_result.status == 'completed', (
                f"Test case {i} ({test_result.execution_context.test_case.id}) status "
                f"expected 'completed', got '{test_result.status}'"
            )

            # Verify all check results within each test case completed successfully
            for j, check_result in enumerate(test_result.check_results):
                assert check_result.status == 'completed', (
                    f"Test case {i} ({test_result.execution_context.test_case.id}), check {j} "
                    f"status expected 'completed', got '{check_result.status}', "
                    f"error: {check_result.error.message if check_result.error else 'None'}"
                )
                assert check_result.error is None, (
                    f"Test case {i} ({test_result.execution_context.test_case.id}), check {j} "
                    f"should not have error when completed, "
                    f"got error: {check_result.error.message}"
                )

        # Verify basic evaluation structure
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
                arguments={"delay": delay_per_check},
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
                arguments={"delay": delay_per_check},
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
        register("custom_user_check", version="1.0.0")(CustomUserCheck)

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

    def test_evaluate_combined_testcase_and_global_checks_basic(self):
        """Test that both TestCase.checks and global checks parameter are executed."""
        # Create test cases with their own checks
        test_cases_with_checks = [
            TestCase(
                id="test_001",
                input="What is the capital of France?",
                expected="Paris",
                checks=[
                    Check(
                        type="test_check",
                        arguments={"expected": "Paris", "actual": "$.output.value"},
                    ),
                ],
            ),
            TestCase(
                id="test_002",
                input="What is 2 + 2?",
                expected="4",
                checks=[
                    Check(
                        type="test_check",
                        arguments={"expected": "4", "actual": "$.output.value"},
                    ),
                ],
            ),
        ]

        # Define global checks (applied to all test cases)
        global_checks = [
            Check(
                type="test_check",
                arguments={"expected": "global_check", "actual": "global_check"},
            ),
        ]

        result = evaluate(test_cases_with_checks, self.outputs, global_checks)

        assert result.status == 'completed'
        assert len(result.results) == 2

        # Each test case should have BOTH TestCase-specific checks AND global checks
        for test_result in result.results:
            # Should have 2 checks: 1 from TestCase + 1 from global checks
            assert len(test_result.check_results) == 2

            # Verify check types
            check_types = [cr.check_type for cr in test_result.check_results]
            assert "test_check" in check_types

            # Verify that we have both the TestCase-specific check and global check
            testcase_check = None
            global_check = None
            for check_result in test_result.check_results:
                expected_value = check_result.resolved_arguments.get("expected", {}).get("value")
                if expected_value == "global_check":
                    global_check = check_result
                else:
                    testcase_check = check_result

            assert testcase_check is not None, "TestCase-specific check should be present"
            assert global_check is not None, "Global check should be present"

    def test_evaluate_combined_testcase_and_global_checks_multiple_each(self):
        """Test multiple TestCase checks and multiple global checks are all executed."""
        # Create test cases with multiple checks each
        test_cases_with_checks = [
            TestCase(
                id="test_001",
                input="test",
                expected="Paris",
                checks=[
                    Check(type="test_check", arguments={"expected": "testcase1_check1", "actual": "testcase1_check1"}),  # noqa: E501
                    Check(type="test_check", arguments={"expected": "testcase1_check2", "actual": "testcase1_check2"}),  # noqa: E501
                ],
            ),
            TestCase(
                id="test_002",
                input="test",
                expected="4",
                checks=[
                    Check(type="test_check", arguments={"expected": "testcase2_check1", "actual": "testcase2_check1"}),  # noqa: E501
                ],
            ),
        ]

        # Multiple global checks
        global_checks = [
            Check(type="test_check", arguments={"expected": "global_check_1", "actual": "global_check_1"}),  # noqa: E501
            Check(type="test_check", arguments={"expected": "global_check_2", "actual": "global_check_2"}),  # noqa: E501
        ]

        result = evaluate(test_cases_with_checks, self.outputs, global_checks)

        assert result.status == 'completed'
        assert len(result.results) == 2

        # Test case 1 should have: 2 TestCase checks + 2 global checks = 4 total
        test1_result = result.results[0]
        assert len(test1_result.check_results) == 4

        # Verify all expected checks are present
        expected_values = {"testcase1_check1", "testcase1_check2", "global_check_1", "global_check_2"}  # noqa: E501
        actual_values = {
            cr.resolved_arguments.get("expected", {}).get("value")
            for cr in test1_result.check_results
        }
        assert expected_values == actual_values

        # Test case 2 should have: 1 TestCase check + 2 global checks = 3 total
        test2_result = result.results[1]
        assert len(test2_result.check_results) == 3

        expected_values = {"testcase2_check1", "global_check_1", "global_check_2"}
        actual_values = {
            cr.resolved_arguments.get("expected", {}).get("value")
            for cr in test2_result.check_results
        }
        assert expected_values == actual_values

    def test_evaluate_combined_checks_with_per_testcase_checks_list(self):
        """Test that both per-test-case checks list and TestCase.checks are executed."""
        # Test cases with their own checks
        test_cases_with_checks = [
            TestCase(
                id="test_001",
                input="test",
                expected="Paris",
                checks=[
                    Check(type="test_check", arguments={"expected": "embedded_check", "actual": "embedded_check"}),  # noqa: E501
                ],
            ),
            TestCase(
                id="test_002",
                input="test",
                expected="4",
                checks=[
                    Check(type="test_check", arguments={"expected": "embedded_check", "actual": "embedded_check"}),  # noqa: E501
                ],
            ),
        ]

        # Per-test-case checks (different for each test case)
        per_testcase_checks = [
            [Check(type="test_check", arguments={"expected": "per_test_1", "actual": "per_test_1"})],  # For test case 1  # noqa: E501
            [Check(type="test_check", arguments={"expected": "per_test_2", "actual": "per_test_2"})],  # For test case 2  # noqa: E501
        ]

        result = evaluate(test_cases_with_checks, self.outputs, per_testcase_checks)

        assert result.status == 'completed'
        assert len(result.results) == 2

        # Test case 1 should have: embedded check + per-test-case check
        test1_result = result.results[0]
        assert len(test1_result.check_results) == 2
        expected_values = {"embedded_check", "per_test_1"}
        actual_values = {
            cr.resolved_arguments.get("expected", {}).get("value")
            for cr in test1_result.check_results
        }
        assert expected_values == actual_values

        # Test case 2 should have: embedded check + per-test-case check
        test2_result = result.results[1]
        assert len(test2_result.check_results) == 2
        expected_values = {"embedded_check", "per_test_2"}
        actual_values = {
            cr.resolved_arguments.get("expected", {}).get("value")
            for cr in test2_result.check_results
        }
        assert expected_values == actual_values


class TestSchemaCheckTypes:
    """Test evaluation engine with schema-based check types for type-safe evaluations."""

    def setup_method(self):
        """Set up test fixtures with schema-based checks."""
        # Restore standard checks to ensure SchemaCheck types work
        restore_standard_checks()

        # Create test data for schema check demonstrations
        self.test_cases = [
            TestCase(id="test_001", input="What is the capital of France?", expected="Paris"),
            TestCase(id="test_002", input="What is 2 + 2?", expected="4"),
        ]

        # Outputs with complex structures to test various schema checks
        self.outputs = [
            Output(
                value={
                    "answer": "Paris",
                    "confidence": 0.95,
                    "metadata": {"source": "geography_db", "timestamp": "2023-01-01"},
                },
                id="output_001",
            ),
            Output(
                value={
                    "result": "4",
                    "calculation": "2 + 2 = 4",
                    "metadata": {"operation": "addition"},
                },
                id="output_002",
            ),
        ]

    def test_attribute_exists_check_schema(self):
        """Test AttributeExistsCheck schema type via evaluate()."""
        # Create schema checks using AttributeExistsCheck
        schema_checks = [
            AttributeExistsCheck(path="$.output.value.answer"),
            AttributeExistsCheck(path="$.output.value.confidence"),
            AttributeExistsCheck(path="$.output.value.nonexistent", negate=True),
        ]

        result = evaluate(self.test_cases, self.outputs, schema_checks)

        assert result.status == 'completed'
        assert len(result.results) == 2

        # First test case should have all attribute checks pass
        test1_result = result.results[0]
        assert len(test1_result.check_results) == 3

        # answer exists
        assert test1_result.check_results[0].check_type == "attribute_exists"
        assert test1_result.check_results[0].results["passed"] is True

        # confidence exists
        assert test1_result.check_results[1].results["passed"] is True

        # nonexistent doesn't exist (negate=True means this should pass)
        assert test1_result.check_results[2].results["passed"] is True

    def test_contains_check_schema(self):
        """Test ContainsCheck schema type via evaluate()."""
        # Use per-test-case checks since data structures are different
        per_case_checks = [
            # Test case 1: Check answer field
            [ContainsCheck(
                text="$.output.value.answer",
                phrases=["Paris"],
                case_sensitive=True,
            )],
            # Test case 2: Check calculation field
            [ContainsCheck(
                text="$.output.value.calculation",
                phrases=["2", "+", "="],
                case_sensitive=False,
            )],
        ]

        result = evaluate(self.test_cases, self.outputs, per_case_checks)

        assert result.status == 'completed'

        # First test case: "Paris" should contain "Paris"
        test1_result = result.results[0]
        assert test1_result.check_results[0].results["passed"] is True

        # Second test case: "2 + 2 = 4" should contain "2", "+", "="
        test2_result = result.results[1]
        assert test2_result.check_results[0].results["passed"] is True

    def test_equals_check_schema(self):
        """Test EqualsCheck schema type via evaluate()."""
        # Use per-test-case checks for different field structures
        per_case_checks = [
            # Test case 1: Check answer field matches expected
            [EqualsCheck(
                actual="$.output.value.answer",
                expected="$.test_case.expected",
            )],
            # Test case 2: Check result field matches literal "4"
            [EqualsCheck(
                actual="$.output.value.result",
                expected="4",
            )],
        ]

        result = evaluate(self.test_cases, self.outputs, per_case_checks)

        assert result.status == 'completed'

        # Both test cases should pass their equality checks
        for test_result in result.results:
            assert len(test_result.check_results) == 1
            assert test_result.check_results[0].check_type == "equals"
            assert test_result.check_results[0].results["passed"] is True

    def test_exact_match_check_schema(self):
        """Test ExactMatchCheck schema type via evaluate()."""
        # Use per-test-case checks for different field structures
        per_case_checks = [
            # Test case 1: Check answer field matches "Paris" exactly
            [ExactMatchCheck(
                actual="$.output.value.answer",
                expected="Paris",
                case_sensitive=True,
            )],
            # Test case 2: Check result field matches "4" exactly
            [ExactMatchCheck(
                actual="$.output.value.result",
                expected="4",
                case_sensitive=False,
            )],
        ]

        result = evaluate(self.test_cases, self.outputs, per_case_checks)

        assert result.status == 'completed'

        # Both test cases should match exactly
        for test_result in result.results:
            assert len(test_result.check_results) == 1
            assert test_result.check_results[0].results["passed"] is True

    def test_is_empty_check_schema(self):
        """Test IsEmptyCheck schema type via evaluate()."""
        # Create output with empty values to test
        empty_outputs = [
            Output(value={"answer": "", "list": [], "dict": {}}, id="empty_001"),
            Output(value={"result": "4", "empty_field": None}, id="empty_002"),
        ]

        # Use per-test-case checks for different field structures
        per_case_checks = [
            # Test case 1: Check empty values
            [
                IsEmptyCheck(value="$.output.value.answer"),
                IsEmptyCheck(value="$.output.value.list"),
                IsEmptyCheck(value="$.output.value.dict"),
            ],
            # Test case 2: Check non-empty result (negate=True should pass)
            [
                IsEmptyCheck(value="$.output.value.result", negate=True),
            ],
        ]

        result = evaluate(self.test_cases, empty_outputs, per_case_checks)

        assert result.status == 'completed'

        # First test case: empty string, list, dict should be empty
        test1_result = result.results[0]
        assert len(test1_result.check_results) == 3
        assert test1_result.check_results[0].results["passed"] is True  # empty string
        assert test1_result.check_results[1].results["passed"] is True  # empty list
        assert test1_result.check_results[2].results["passed"] is True  # empty dict

        # Second test case: "4" is not empty (negate=True should pass)
        test2_result = result.results[1]
        assert len(test2_result.check_results) == 1
        assert test2_result.check_results[0].results["passed"] is True

    def test_regex_check_schema(self):
        """Test RegexCheck schema type via evaluate()."""
        # Use per-test-case checks for different field structures
        per_case_checks = [
            # Test case 1: Check answer field matches capital + lowercase pattern
            [RegexCheck(
                text="$.output.value.answer",
                pattern=r"^[A-Z][a-z]+$",  # Capital letter followed by lowercase
            )],
            # Test case 2: Check calculation field matches math expression pattern
            [RegexCheck(
                text="$.output.value.calculation",
                pattern=r"\d+\s*\+\s*\d+\s*=\s*\d+",  # Math expression pattern
            )],
        ]

        result = evaluate(self.test_cases, self.outputs, per_case_checks)

        assert result.status == 'completed'

        # "Paris" should match capital + lowercase pattern
        test1_result = result.results[0]
        assert test1_result.check_results[0].results["passed"] is True

        # "2 + 2 = 4" should match math expression pattern
        test2_result = result.results[1]
        assert test2_result.check_results[0].results["passed"] is True

    def test_threshold_check_schema(self):
        """Test ThresholdCheck schema type via evaluate()."""
        # Only test first case since only it has confidence field
        schema_checks = [
            ThresholdCheck(
                value="$.output.value.confidence",
                min_value=0.8,
                max_value=1.0,
            ),
            ThresholdCheck(
                value="$.output.value.confidence",
                min_value=0.9,  # No max_value
            ),
        ]

        # Test just the first test case and output that has confidence
        result = evaluate(self.test_cases[:1], self.outputs[:1], schema_checks)

        assert result.status == 'completed'

        # Confidence of 0.95 should pass both threshold checks
        test1_result = result.results[0]
        assert len(test1_result.check_results) == 2
        assert test1_result.check_results[0].results["passed"] is True  # 0.8 <= 0.95 <= 1.0
        assert test1_result.check_results[1].results["passed"] is True  # 0.95 >= 0.9

    def test_mixed_schema_and_generic_checks(self):
        """Test mixing schema checks with generic Check objects."""
        # Test that schema checks work alongside standard checks
        mixed_checks = [
            EqualsCheck(actual="$.output.value.answer", expected="Paris"),
            AttributeExistsCheck(path="$.output.value.metadata"),
            # Use a standard check that should be available
            Check(
                type="exact_match",
                arguments={"expected": "Paris", "actual": "$.output.value.answer"},
            ),
        ]

        result = evaluate(self.test_cases[:1], self.outputs[:1], mixed_checks)

        assert result.status == 'completed'

        # Single test case should have 3 checks (2 schema + 1 standard)
        assert len(result.results) == 1
        test_result = result.results[0]
        assert len(test_result.check_results) == 3

        # Verify we have the expected check types
        check_types = [cr.check_type for cr in test_result.check_results]
        assert "equals" in check_types
        assert "attribute_exists" in check_types
        assert "exact_match" in check_types

    def test_schema_check_jsonpath_validation(self):
        """Test that schema checks properly validate JSONPath expressions."""
        # Test with invalid JSONPath (should work but give unexpected results)
        try:
            schema_check = EqualsCheck(
                actual="invalid_jsonpath",  # Not a JSONPath, treated as literal
                expected="Paris",
            )

            result = evaluate(self.test_cases, self.outputs, [schema_check])

            # Should complete but fail the equality check (literal "invalid_jsonpath" != "Paris")
            assert result.status == 'completed'
            assert result.results[0].check_results[0].results["passed"] is False

        except Exception as e:
            # If validation catches this, that's also acceptable
            assert "jsonpath" in str(e).lower() or "path" in str(e).lower()  # noqa: PT017

    def test_schema_check_type_safety(self):
        """Test that schema checks provide type safety and validation."""
        # Test that schema checks validate their arguments
        with pytest.raises((ValueError, TypeError)):
            # phrases must be a string or list, not a number
            ContainsCheck(text="$.output.value.answer", phrases=123)  # type: ignore

        with pytest.raises((ValueError, TypeError)):
            # min_value should be numeric, string, or None (but "not_a_number" is valid as a
            # JSONPath)
            # So let's test with an invalid type instead
            ThresholdCheck(value="$.output.value.confidence", min_value=["invalid"])  # type: ignore

    def test_schema_check_with_per_testcase_checks(self):
        """Test schema checks used as per-test-case checks."""
        # Create different schema checks for each test case
        per_testcase_schema_checks = [
            # Test case 1: Check for geography-specific attributes
            [
                AttributeExistsCheck(path="$.output.value.answer"),
                ContainsCheck(text="$.output.value.answer", phrases=["Paris"]),
            ],
            # Test case 2: Check for math-specific attributes
            [
                AttributeExistsCheck(path="$.output.value.result"),
                RegexCheck(text="$.output.value.calculation", pattern=r"\d+.*=.*\d+"),
            ],
        ]

        result = evaluate(self.test_cases, self.outputs, per_testcase_schema_checks)

        assert result.status == 'completed'
        assert len(result.results) == 2

        # Each test case should have 2 checks
        for test_result in result.results:
            assert len(test_result.check_results) == 2
            assert all(cr.status == 'completed' for cr in test_result.check_results)
            assert all(cr.results["passed"] is True for cr in test_result.check_results)

    def test_schema_check_metadata_preservation(self):
        """Test that schema check metadata is preserved through evaluation."""
        schema_check = EqualsCheck(
            actual="$.output.value.answer",
            expected="Paris",
            metadata={"test_category": "geography", "difficulty": "easy"},
        )

        result = evaluate(self.test_cases[:1], self.outputs[:1], [schema_check])

        assert result.status == 'completed'
        check_result = result.results[0].check_results[0]

        # Metadata should be preserved in the check result
        assert check_result.metadata is not None
        assert check_result.metadata.get("test_category") == "geography"
        assert check_result.metadata.get("difficulty") == "easy"


class TestSchemaCheckTypesWithLiteralValues:
    """Test schema check types using literal values instead of JSONPath expressions."""

    def setup_method(self):
        """Set up test fixtures for literal value schema checks."""
        restore_standard_checks()

        # Simple test cases and outputs for literal value testing
        self.test_cases = [
            TestCase(id="test_001", input="What is the capital of France?", expected="Paris"),
            TestCase(id="test_002", input="What is 2 + 2?", expected="4"),
        ]

        # Simple outputs that we'll extract values from manually
        self.outputs = [
            Output(value="Paris", id="output_001"),
            Output(value="4", id="output_002"),
        ]

    def test_contains_check_with_literal_values(self):
        """Test ContainsCheck using literal string values instead of JSONPath."""
        # Extract values manually and use literal strings
        schema_checks = [
            ContainsCheck(text="Paris", phrases=["Par", "is"]),
            ContainsCheck(text="The answer is 4", phrases=["answer", "4"], case_sensitive=False),
        ]

        result = evaluate(self.test_cases, self.outputs, schema_checks)

        assert result.status == 'completed'

        # Both test cases should pass their contains checks
        for test_result in result.results:
            for check_result in test_result.check_results:
                assert check_result.results["passed"] is True

    def test_equals_check_with_literal_values(self):
        """Test EqualsCheck using literal values instead of JSONPath."""
        schema_checks = [
            EqualsCheck(actual="Paris", expected="Paris"),
            EqualsCheck(actual=4, expected=4),  # Numeric literals
            EqualsCheck(actual=["a", "b"], expected=["a", "b"]),  # List literals work
            EqualsCheck(actual="", expected=""),  # Empty strings are valid literal values
        ]

        result = evaluate(self.test_cases[:1], self.outputs[:1], schema_checks)

        assert result.status == 'completed'

        # All equality checks should pass
        test_result = result.results[0]
        assert len(test_result.check_results) == 4
        for check_result in test_result.check_results:
            assert check_result.results["passed"] is True

    def test_exact_match_check_with_literal_values(self):
        """Test ExactMatchCheck using literal string values instead of JSONPath."""
        schema_checks = [
            ExactMatchCheck(actual="Paris", expected="Paris", case_sensitive=True),
            ExactMatchCheck(actual="HELLO", expected="hello", case_sensitive=False),
            ExactMatchCheck(
                actual="Test", expected="test", case_sensitive=True, negate=True,
            ),  # Should pass
        ]

        result = evaluate(self.test_cases[:1], self.outputs[:1], schema_checks)

        assert result.status == 'completed'

        # All exact match checks should pass
        test_result = result.results[0]
        assert len(test_result.check_results) == 3
        for check_result in test_result.check_results:
            assert check_result.results["passed"] is True

    def test_is_empty_check_with_literal_values(self):
        """Test IsEmptyCheck using literal values instead of JSONPath."""
        schema_checks = [
            IsEmptyCheck(value=""),           # Empty string
            IsEmptyCheck(value=[]),           # Empty list
            IsEmptyCheck(value={}),           # Empty dict
            IsEmptyCheck(value=None),         # None value
            IsEmptyCheck(value="not empty", negate=True),  # Non-empty (negated)
            IsEmptyCheck(value=[1, 2, 3], negate=True),    # Non-empty list (negated)
        ]

        result = evaluate(self.test_cases[:1], self.outputs[:1], schema_checks)

        assert result.status == 'completed'

        # All empty/non-empty checks should pass
        test_result = result.results[0]
        assert len(test_result.check_results) == 6
        for check_result in test_result.check_results:
            assert check_result.results["passed"] is True

    def test_regex_check_with_literal_values(self):
        """Test RegexCheck using literal string values instead of JSONPath."""
        schema_checks = [
            RegexCheck(text="Paris", pattern=r"^[A-Z][a-z]+$"),           # Capital + lowercase
            RegexCheck(
                text="user@example.com", pattern=r"^[\w.-]+@[\w.-]+\.\w+$",
            ),  # Email pattern
            RegexCheck(text="123-45-6789", pattern=r"^\d{3}-\d{2}-\d{4}$"),        # SSN pattern
            RegexCheck(text="Hello123", pattern=r"\d+"),                   # Contains digits
        ]

        result = evaluate(self.test_cases[:1], self.outputs[:1], schema_checks)

        assert result.status == 'completed'

        # All regex checks should pass
        test_result = result.results[0]
        assert len(test_result.check_results) == 4
        for check_result in test_result.check_results:
            assert check_result.results["passed"] is True

    def test_threshold_check_with_literal_values(self):
        """Test ThresholdCheck using literal numeric values instead of JSONPath."""
        schema_checks = [
            ThresholdCheck(value=0.95, min_value=0.8, max_value=1.0),     # Within range
            ThresholdCheck(value=100, min_value=50),                       # Above minimum only
            ThresholdCheck(value=25, max_value=50),                        # Below maximum only
            ThresholdCheck(value=3.14159, min_value=3.0, max_value=4.0),  # Float within range
        ]

        result = evaluate(self.test_cases[:1], self.outputs[:1], schema_checks)

        assert result.status == 'completed'

        # All threshold checks should pass
        test_result = result.results[0]
        assert len(test_result.check_results) == 4
        for check_result in test_result.check_results:
            assert check_result.results["passed"] is True

    def test_mixed_literal_and_jsonpath_values(self):
        """Test mixing literal values with JSONPath expressions in same evaluation."""
        # Create more complex output structure
        complex_outputs = [
            Output(value={"text": "Hello World", "score": 0.95}, id="complex_001"),
        ]

        schema_checks = [
            # Literal value checks
            ContainsCheck(text="Hello World", phrases=["Hello"]),
            EqualsCheck(actual=42, expected=42),
            # JSONPath value checks
            ThresholdCheck(value="$.output.value.score", min_value=0.8),
            ContainsCheck(text="$.output.value.text", phrases=["World"]),
        ]

        result = evaluate(self.test_cases[:1], complex_outputs, schema_checks)

        assert result.status == 'completed'

        # All mixed checks should pass
        test_result = result.results[0]
        assert len(test_result.check_results) == 4
        for check_result in test_result.check_results:
            assert check_result.results["passed"] is True

    def test_literal_values_with_complex_data_types(self):
        """Test schema checks with complex literal data types."""
        schema_checks = [
            # Complex data structure comparisons
            EqualsCheck(
                actual={"name": "John", "age": 30, "skills": ["Python", "JavaScript"]},
                expected={"name": "John", "age": 30, "skills": ["Python", "JavaScript"]},
            ),
            EqualsCheck(
                actual=[1, 2, {"nested": True}],
                expected=[1, 2, {"nested": True}],
            ),
            # Empty collection checks
            IsEmptyCheck(value=set()),  # Empty set
            IsEmptyCheck(value=(), negate=False),  # Empty tuple
            # String operations on complex converted values
            ContainsCheck(
                text=str({"key": "value"}),
                phrases=["key", "value"],
            ),
        ]

        result = evaluate(self.test_cases[:1], self.outputs[:1], schema_checks)

        assert result.status == 'completed'

        # All complex data type checks should pass
        test_result = result.results[0]
        assert len(test_result.check_results) == 5
        for check_result in test_result.check_results:
            assert check_result.results["passed"] is True

    def test_user_resolved_values_pattern(self):
        """Test a realistic pattern where users resolve values themselves."""
        # Simulate user extracting and processing values from outputs
        raw_output = {"model_response": "The capital of France is Paris", "confidence": 0.92}

        # User extracts and processes the values
        extracted_text = raw_output["model_response"]
        confidence_score = raw_output["confidence"]
        word_count = len(extracted_text.split())
        contains_capital = "capital" in extracted_text.lower()

        # User creates checks with their resolved values
        user_resolved_checks = [
            ContainsCheck(text=extracted_text, phrases=["France", "Paris"]),
            ThresholdCheck(value=confidence_score, min_value=0.8),
            ThresholdCheck(value=word_count, min_value=5, max_value=20),
            EqualsCheck(actual=contains_capital, expected=True),
            RegexCheck(text=extracted_text, pattern=r"The .+ of .+ is .+"),  # Template pattern
        ]

        result = evaluate(self.test_cases[:1], self.outputs[:1], user_resolved_checks)

        assert result.status == 'completed'

        # All user-resolved checks should pass
        test_result = result.results[0]
        assert len(test_result.check_results) == 5
        for i, check_result in enumerate(test_result.check_results):
            assert check_result.results["passed"] is True, (
                f"Check {i} failed: {check_result.check_type}"
            )

    def test_literal_values_error_cases(self):
        """Test that literal values still produce meaningful errors when checks fail."""
        schema_checks = [
            ContainsCheck(text="Hello", phrases=["Goodbye"]),  # Should fail
            EqualsCheck(actual="Apple", expected="Orange"),     # Should fail
            ThresholdCheck(value=5, min_value=10),              # Should fail
        ]

        result = evaluate(self.test_cases[:1], self.outputs[:1], schema_checks)

        assert result.status == 'completed'

        # All checks should fail but complete successfully
        test_result = result.results[0]
        assert len(test_result.check_results) == 3
        for check_result in test_result.check_results:
            assert check_result.status == 'completed'
            assert check_result.results["passed"] is False
