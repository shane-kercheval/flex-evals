"""Tests for core evaluation engine."""

import pytest
import asyncio
import time
from datetime import datetime, UTC
from typing import Any

from flex_evals.engine import _reconstruct_check_order, _separate_checks_by_type, evaluate
from flex_evals.schemas import (
    TestCase, Output, Check, EvaluationRunResult,
    ExperimentMetadata,
)
from flex_evals.checks.base import BaseCheck, BaseAsyncCheck
from flex_evals.registry import register, clear_registry
from flex_evals.exceptions import ValidationError
from flex_evals.schemas.check import CheckResult, CheckResultMetadata
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
            Output(value="Paris"),
            Output(value="4"),
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
            outputs=[Output(value="test")],
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
            outputs=[Output(value="test")],
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
            outputs=[Output(value="test")],
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
            outputs=[Output(value="test")],
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

    def test_separate_checks_by_type_all_sync(self):
        """Test _separate_checks_by_type with only sync checks."""
        checks = [
            Check(type="test_check", arguments={"expected": "test1"}),
            Check(type="test_check", arguments={"expected": "test2"}),
            Check(type="test_check", arguments={"expected": "test3"}),
        ]

        sync_checks, async_checks, order_map = _separate_checks_by_type(checks)

        # All should be sync
        assert len(sync_checks) == 3
        assert len(async_checks) == 0
        assert len(order_map) == 3

        # Order map should be all sync
        expected_order = [('sync', 0), ('sync', 1), ('sync', 2)]
        assert order_map == expected_order

        # Sync checks should be in same order as input
        assert sync_checks[0].arguments["expected"] == "test1"
        assert sync_checks[1].arguments["expected"] == "test2"
        assert sync_checks[2].arguments["expected"] == "test3"

    def test_separate_checks_by_type_all_async(self):
        """Test _separate_checks_by_type with only async checks."""
        checks = [
            Check(type="slow_async_check", arguments={"delay": 0.1}),
            Check(type="slow_async_check", arguments={"delay": 0.2}),
            Check(type="slow_async_check", arguments={"delay": 0.3}),
        ]

        sync_checks, async_checks, order_map = _separate_checks_by_type(checks)

        # All should be async
        assert len(sync_checks) == 0
        assert len(async_checks) == 3
        assert len(order_map) == 3

        # Order map should be all async
        expected_order = [('async', 0), ('async', 1), ('async', 2)]
        assert order_map == expected_order

        # Async checks should be in same order as input
        assert async_checks[0].arguments["delay"] == 0.1
        assert async_checks[1].arguments["delay"] == 0.2
        assert async_checks[2].arguments["delay"] == 0.3

    def test_separate_checks_by_type_mixed(self):
        """Test _separate_checks_by_type with mixed sync/async checks."""
        checks = [
            Check(type="test_check", arguments={"expected": "sync1"}),       # sync
            Check(type="slow_async_check", arguments={"delay": 0.1}),       # async
            Check(type="test_check", arguments={"expected": "sync2"}),       # sync
            Check(type="slow_async_check", arguments={"delay": 0.2}),       # async
            Check(type="test_check", arguments={"expected": "sync3"}),       # sync
        ]

        sync_checks, async_checks, order_map = _separate_checks_by_type(checks)

        # Should separate correctly
        assert len(sync_checks) == 3
        assert len(async_checks) == 2
        assert len(order_map) == 5

        # Order map should reflect alternating pattern
        expected_order = [
            ('sync', 0), ('async', 0), ('sync', 1), ('async', 1), ('sync', 2),
        ]
        assert order_map == expected_order

        # Sync checks should be grouped but maintain relative order
        assert sync_checks[0].arguments["expected"] == "sync1"
        assert sync_checks[1].arguments["expected"] == "sync2"
        assert sync_checks[2].arguments["expected"] == "sync3"

        # Async checks should be grouped but maintain relative order
        assert async_checks[0].arguments["delay"] == 0.1
        assert async_checks[1].arguments["delay"] == 0.2

    def test_separate_checks_by_type_empty(self):
        """Test _separate_checks_by_type with empty input."""
        checks = []
        sync_checks, async_checks, order_map = _separate_checks_by_type(checks)

        assert len(sync_checks) == 0
        assert len(async_checks) == 0
        assert len(order_map) == 0

    def test_separate_checks_by_type_unregistered_check(self):
        """Test _separate_checks_by_type with unregistered check types."""
        checks = [
            Check(type="nonexistent_check", arguments={}),  # Should be treated as sync
            Check(type="test_check", arguments={"expected": "test"}),
        ]

        sync_checks, async_checks, order_map = _separate_checks_by_type(checks)

        # Unregistered check should be treated as sync
        assert len(sync_checks) == 2
        assert len(async_checks) == 0
        assert len(order_map) == 2

        expected_order = [('sync', 0), ('sync', 1)]
        assert order_map == expected_order

    def test_reconstruct_check_order_all_sync(self):
        """Test _reconstruct_check_order with only sync results."""
        # Mock sync results
        sync_results = [
            CheckResult(
                check_type="test_check",
                status="completed",
                results={"id": i},
                resolved_arguments={},
                evaluated_at=datetime.now(UTC),
                metadata=CheckResultMetadata(
                    test_case_id="test",
                    test_case_metadata=None,
                    output_metadata=None,
                    check_version=None,
                ),
            )
            for i in range(3)
        ]
        async_results = []
        order_map = [('sync', 0), ('sync', 1), ('sync', 2)]

        final_results = _reconstruct_check_order(sync_results, async_results, order_map)

        assert len(final_results) == 3
        assert final_results[0].results["id"] == 0
        assert final_results[1].results["id"] == 1
        assert final_results[2].results["id"] == 2

    def test_reconstruct_check_order_all_async(self):
        """Test _reconstruct_check_order with only async results."""
        # Mock async results
        sync_results = []
        async_results = [
            CheckResult(
                check_type="slow_async_check",
                status="completed",
                results={"id": i + 10},
                resolved_arguments={},
                evaluated_at=datetime.now(UTC),
                metadata=CheckResultMetadata(
                    test_case_id="test",
                    test_case_metadata=None,
                    output_metadata=None,
                    check_version=None,
                ),
            )
            for i in range(2)
        ]
        order_map = [('async', 0), ('async', 1)]

        final_results = _reconstruct_check_order(sync_results, async_results, order_map)

        assert len(final_results) == 2
        assert final_results[0].results["id"] == 10
        assert final_results[1].results["id"] == 11

    def test_reconstruct_check_order_mixed(self):
        """Test _reconstruct_check_order with mixed sync/async results."""
        # Mock mixed results
        sync_results = [
            CheckResult(
                check_type="test_check",
                status="completed",
                results={"type": "sync", "id": i},
                resolved_arguments={},
                evaluated_at=datetime.now(UTC),
                metadata=CheckResultMetadata(
                    test_case_id="test",
                    test_case_metadata=None,
                    output_metadata=None,
                    check_version=None,
                ),
            )
            for i in range(3)
        ]

        async_results = [
            CheckResult(
                check_type="slow_async_check",
                status="completed",
                results={"type": "async", "id": i + 100},
                resolved_arguments={},
                evaluated_at=datetime.now(UTC),
                metadata=CheckResultMetadata(
                    test_case_id="test",
                    test_case_metadata=None,
                    output_metadata=None,
                    check_version=None,
                ),
            )
            for i in range(2)
        ]

        # Pattern: sync, async, sync, async, sync
        order_map = [
            ('sync', 0), ('async', 0), ('sync', 1), ('async', 1), ('sync', 2),
        ]

        final_results = _reconstruct_check_order(sync_results, async_results, order_map)

        assert len(final_results) == 5

        # Verify alternating pattern is preserved
        assert final_results[0].results["type"] == "sync"
        assert final_results[0].results["id"] == 0
        assert final_results[1].results["type"] == "async"
        assert final_results[1].results["id"] == 100
        assert final_results[2].results["type"] == "sync"
        assert final_results[2].results["id"] == 1
        assert final_results[3].results["type"] == "async"
        assert final_results[3].results["id"] == 101
        assert final_results[4].results["type"] == "sync"
        assert final_results[4].results["id"] == 2

    def test_reconstruct_check_order_empty(self):
        """Test _reconstruct_check_order with empty inputs."""
        sync_results = []
        async_results = []
        order_map = []

        final_results = _reconstruct_check_order(sync_results, async_results, order_map)

        assert len(final_results) == 0
