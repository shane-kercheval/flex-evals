"""
Unit tests for evaluation engine optimization functions.

Tests the flattening/unflattening logic and performance improvements.
"""

import asyncio
import time
from datetime import datetime, UTC
from typing import Any

from flex_evals.schemas import TestCase, Output, Check, CheckResult
from flex_evals.checks.base import BaseAsyncCheck, EvaluationContext
from flex_evals.engine import (
    _flatten_checks_for_execution,
    _unflatten_check_results,
    evaluate,
)
from flex_evals.registry import register


class AsyncSleepCheck(BaseAsyncCheck):
    """Async check that sleeps for a specified duration."""

    async def __call__(self, sleep_duration: float = 0.1) -> dict[str, Any]:
        """Sleep for the specified duration and return timing info."""
        start_time = time.time()
        await asyncio.sleep(sleep_duration)
        end_time = time.time()
        return {
            "slept_for": end_time - start_time,
            "sleep_duration": sleep_duration,
        }


class TestFlattenUnflatten:
    """Test the flattening and unflattening logic."""

    def test_flatten_checks_basic(self):
        """Test basic flattening of checks across test cases."""
        # Create test data
        test_case1 = TestCase(id="1", input={"value": 1})
        test_case2 = TestCase(id="2", input={"value": 2})
        output1 = Output(value={"result": 10})
        output2 = Output(value={"result": 20})

        check1 = Check(type="exact_match", arguments={"expected": 10})
        check2 = Check(type="exact_match", arguments={"expected": 20})

        work_items = [
            (test_case1, output1, [check1]),
            (test_case2, output2, [check2]),
        ]

        # Flatten checks
        sync_checks, async_checks, tracking = _flatten_checks_for_execution(work_items)

        # Verify results
        assert len(sync_checks) == 2
        assert len(async_checks) == 0
        assert len(tracking) == 2

        # Check first flattened check
        assert sync_checks[0][0] == check1
        assert isinstance(sync_checks[0][1], EvaluationContext)
        assert sync_checks[0][1].test_case == test_case1

        # Check tracking
        assert tracking[0] == (0, 0, False, 0)  # test_idx=0, check_idx=0, is_async=False, flattened_idx=0  # noqa: E501
        assert tracking[1] == (1, 0, False, 1)  # test_idx=1, check_idx=0, is_async=False, flattened_idx=1  # noqa: E501

    def test_flatten_checks_mixed_sync_async(self):
        """Test flattening with mixed sync and async checks."""
        # Register async check
        register("async_sleep")(AsyncSleepCheck)

        test_case = TestCase(id="1", input={"value": 1})
        output = Output(value={"result": 10})

        sync_check = Check(type="exact_match", arguments={"expected": 10})
        async_check = Check(type="async_sleep", arguments={"sleep_duration": 0.1})

        work_items = [(test_case, output, [sync_check, async_check])]

        # Flatten checks
        sync_checks, async_checks, tracking = _flatten_checks_for_execution(work_items)

        # Verify results
        assert len(sync_checks) == 1
        assert len(async_checks) == 1
        assert len(tracking) == 2

        # Check tracking maintains order
        assert tracking[0] == (0, 0, False, 0)  # sync check
        assert tracking[1] == (0, 1, True, 0)   # async check

    def test_unflatten_check_results(self):
        """Test unflattening check results back to test cases."""
        # Create test data
        test_case1 = TestCase(id="1", input={"value": 1})
        test_case2 = TestCase(id="2", input={"value": 2})
        output1 = Output(value={"result": 10})
        output2 = Output(value={"result": 20})

        work_items = [
            (test_case1, output1, []),
            (test_case2, output2, []),
        ]

        # Create mock results
        sync_result1 = CheckResult(
            check_type="exact_match",
            check_version="1.0.0",
            status="completed",
            results={"matched": True},
            resolved_arguments={},
            evaluated_at=datetime.now(UTC),
        )
        sync_result2 = CheckResult(
            check_type="exact_match",
            check_version="1.0.0",
            status="completed",
            results={"matched": False},
            resolved_arguments={},
            evaluated_at=datetime.now(UTC),
        )

        # Create tracking that maps results to test cases
        tracking = [
            (0, 0, False, 0),  # test_case 0, check 0, sync, result 0
            (1, 0, False, 1),  # test_case 1, check 0, sync, result 1
        ]

        # Unflatten results
        results = _unflatten_check_results(
            work_items,
            [sync_result1, sync_result2],
            [],
            tracking,
        )

        # Verify results
        assert len(results) == 2
        assert results[0].check_results[0] == sync_result1
        assert results[1].check_results[0] == sync_result2

    def test_unflatten_preserves_check_order(self):
        """Test that unflattening preserves the original check order."""
        # Create test data with multiple checks in different order
        test_case = TestCase(id="1", input={"value": 1})
        output = Output(value={"result": 10})

        work_items = [(test_case, output, [])]

        # Create results in scrambled order
        result1 = CheckResult(check_type="check1", check_version="1.0.0", status="completed", results={}, resolved_arguments={}, evaluated_at=datetime.now(UTC))  # noqa: E501
        result2 = CheckResult(check_type="check2", check_version="1.0.0", status="completed", results={}, resolved_arguments={}, evaluated_at=datetime.now(UTC))  # noqa: E501
        result3 = CheckResult(check_type="check3", check_version="1.0.0", status="completed", results={}, resolved_arguments={}, evaluated_at=datetime.now(UTC))  # noqa: E501

        # Tracking with checks out of order in flattened lists
        tracking = [
            (0, 2, False, 0),  # check index 2 -> sync result 0
            (0, 0, True, 0),   # check index 0 -> async result 0
            (0, 1, False, 1),  # check index 1 -> sync result 1
        ]

        # Unflatten results
        results = _unflatten_check_results(
            work_items,
            [result1, result3],  # sync results
            [result2],           # async results
            tracking,
        )

        # Verify check order is preserved (0, 1, 2)
        assert len(results) == 1
        assert results[0].check_results[0] == result2  # check index 0
        assert results[0].check_results[1] == result3  # check index 1
        assert results[0].check_results[2] == result1  # check index 2

    def test_flatten_unflatten_complex_scenario(self):
        """Test complex scenario with multiple test cases having different check counts."""
        # Register async check
        register("async_sleep")(AsyncSleepCheck)

        # Create test cases with varying numbers of checks
        test_case1 = TestCase(
            id="tc1",
            input={"value": 100},
            checks=[
                Check(type="exact_match", arguments={"expected": 100}),
                Check(type="async_sleep", arguments={"sleep_duration": 0.01}),
            ],
        )
        test_case2 = TestCase(
            id="tc2",
            input={"value": 200},
            checks=[
                Check(type="exact_match", arguments={"expected": 200}),
                Check(type="exact_match", arguments={"expected": 201}),
                Check(type="async_sleep", arguments={"sleep_duration": 0.02}),
                Check(type="exact_match", arguments={"expected": 202}),
            ],
        )
        test_case3 = TestCase(
            id="tc3",
            input={"value": 300},
            checks=[  # Only async checks
                Check(type="async_sleep", arguments={"sleep_duration": 0.03}),
                Check(type="async_sleep", arguments={"sleep_duration": 0.04}),
            ],
        )
        test_case4 = TestCase(
            id="tc4",
            input={"value": 400},
            checks=[  # Only sync checks
                Check(type="exact_match", arguments={"expected": 400}),
            ],
        )
        test_case5 = TestCase(
            id="tc5",
            input={"value": 500},
            checks=[],  # No checks
        )

        output1 = Output(value={"result": 100})
        output2 = Output(value={"result": 200})
        output3 = Output(value={"result": 300})
        output4 = Output(value={"result": 400})
        output5 = Output(value={"result": 500})

        work_items = [
            (test_case1, output1, test_case1.checks),
            (test_case2, output2, test_case2.checks),
            (test_case3, output3, test_case3.checks),
            (test_case4, output4, test_case4.checks),
            (test_case5, output5, test_case5.checks),
        ]

        # Flatten checks
        sync_checks, async_checks, tracking = _flatten_checks_for_execution(work_items)

        # Verify flattening counts
        # TC1: 1 sync, 1 async
        # TC2: 3 sync, 1 async
        # TC3: 0 sync, 2 async
        # TC4: 1 sync, 0 async
        # TC5: 0 sync, 0 async
        # Total: 5 sync, 4 async
        assert len(sync_checks) == 5
        assert len(async_checks) == 4
        assert len(tracking) == 9  # Total checks across all test cases

        # Verify tracking is correct for first test case
        tc1_tracking = [t for t in tracking if t[0] == 0]
        assert len(tc1_tracking) == 2
        assert tc1_tracking[0] == (0, 0, False, 0)  # First sync check
        assert tc1_tracking[1] == (0, 1, True, 0)   # First async check

        # Verify tracking for test case with most checks (TC2)
        tc2_tracking = [t for t in tracking if t[0] == 1]
        assert len(tc2_tracking) == 4
        assert not tc2_tracking[0][2]  # sync
        assert not tc2_tracking[1][2]  # sync
        assert tc2_tracking[2][2]   # async
        assert not tc2_tracking[3][2]  # sync

        # Create mock results for all checks
        sync_results = [
            CheckResult(
                check_type="exact_match",
                check_version="1.0.0",
                status="completed",
                results={"matched": True, "check_id": f"sync_{i}"},
                resolved_arguments={},
                evaluated_at=datetime.now(UTC),
            )
            for i in range(5)
        ]

        async_results = [
            CheckResult(
                check_type="async_sleep",
                check_version="1.0.0",
                status="completed",
                results={"slept_for": 0.01 * (i + 1), "check_id": f"async_{i}"},
                resolved_arguments={},
                evaluated_at=datetime.now(UTC),
            )
            for i in range(4)
        ]

        # Unflatten results
        results = _unflatten_check_results(work_items, sync_results, async_results, tracking)

        # Verify unflattening
        assert len(results) == 5

        # TC1: 2 checks (1 sync, 1 async)
        assert len(results[0].check_results) == 2
        assert results[0].check_results[0].check_type == "exact_match"
        assert results[0].check_results[1].check_type == "async_sleep"

        # TC2: 4 checks (3 sync, 1 async, in specific order)
        assert len(results[1].check_results) == 4
        assert results[1].check_results[0].check_type == "exact_match"
        assert results[1].check_results[1].check_type == "exact_match"
        assert results[1].check_results[2].check_type == "async_sleep"
        assert results[1].check_results[3].check_type == "exact_match"

        # TC3: 2 checks (both async)
        assert len(results[2].check_results) == 2
        assert all(r.check_type == "async_sleep" for r in results[2].check_results)

        # TC4: 1 check (sync)
        assert len(results[3].check_results) == 1
        assert results[3].check_results[0].check_type == "exact_match"

        # TC5: 0 checks
        assert len(results[4].check_results) == 0

    def test_flatten_unflatten_end_to_end(self):
        """Test end-to-end flattening and unflattening with actual evaluation."""
        # Register async check
        register("async_sleep")(AsyncSleepCheck)

        # Create test cases with per-test-case checks
        test_cases = [
            TestCase(
                id="1",
                input={"value": 10},
                checks=[
                    Check(type="exact_match", arguments={
                        "actual": "$.output.value.result",
                        "expected": 10,
                    }),
                    Check(type="async_sleep", arguments={"sleep_duration": 0.01}),
                ],
            ),
            TestCase(
                id="2",
                input={"value": 20},
                checks=[
                    Check(type="async_sleep", arguments={"sleep_duration": 0.01}),
                    Check(type="exact_match", arguments={
                        "actual": "$.output.value.result",
                        "expected": 20,
                    }),
                    Check(type="async_sleep", arguments={"sleep_duration": 0.01}),
                ],
            ),
            TestCase(
                id="3",
                input={"value": 30},
                checks=[  # No checks
                ],
            ),
        ]

        outputs = [
            Output(value={"result": 10}),
            Output(value={"result": 20}),
            Output(value={"result": 30}),
        ]

        # Run evaluation (uses flatten/unflatten internally)
        result = evaluate(test_cases, outputs)

        # Verify results
        assert result.status == "completed"
        assert len(result.results) == 3

        # TC1: 2 checks
        assert len(result.results[0].check_results) == 2
        assert result.results[0].check_results[0].check_type == "exact_match"
        assert result.results[0].check_results[0].status == "completed"
        assert result.results[0].check_results[1].check_type == "async_sleep"
        assert result.results[0].check_results[1].status == "completed"

        # TC2: 3 checks in correct order
        assert len(result.results[1].check_results) == 3
        assert result.results[1].check_results[0].check_type == "async_sleep"
        assert result.results[1].check_results[1].check_type == "exact_match"
        assert result.results[1].check_results[2].check_type == "async_sleep"

        # TC3: 0 checks
        assert len(result.results[2].check_results) == 0
        assert result.results[2].status == "completed"


class TestPerformanceOptimization:
    """Test performance improvements from the optimization."""

    def test_async_checks_run_concurrently_across_test_cases(self):
        """Test that async checks run concurrently across all test cases."""
        # Register async check
        register("async_sleep")(AsyncSleepCheck)

        # Create 10 test cases with sleep checks
        num_test_cases = 10
        sleep_duration = 0.1

        test_cases = [
            TestCase(id=str(i), input={"value": i})
            for i in range(num_test_cases)
        ]
        outputs = [
            Output(value={"result": i * 10})
            for i in range(num_test_cases)
        ]

        # Each test case has one async check
        checks = [
            Check(type="async_sleep", arguments={"sleep_duration": sleep_duration}),
        ]

        # Measure execution time
        start_time = time.time()
        result = evaluate(test_cases, outputs, checks)
        end_time = time.time()

        total_time = end_time - start_time

        # If run sequentially, would take ~1 second (10 * 0.1)
        # If run concurrently, should take ~0.1 seconds
        # Allow some overhead, but should be much less than sequential time
        assert total_time < num_test_cases * sleep_duration * 0.5

        # Verify all checks completed successfully
        assert result.status == "completed"
        assert result.summary.total_test_cases == num_test_cases
        assert result.summary.completed_test_cases == num_test_cases

        # Verify actual sleep times
        for test_result in result.results:
            check_result = test_result.check_results[0]
            assert check_result.status == "completed"
            assert abs(check_result.results["slept_for"] - sleep_duration) < 0.05

    def test_sync_checks_have_no_async_overhead(self):
        """Test that sync-only evaluations don't create event loops."""
        # Create many test cases with sync checks
        num_test_cases = 1000

        test_cases = [
            TestCase(id=str(i), input={"value": i})
            for i in range(num_test_cases)
        ]
        outputs = [
            Output(value={"result": i})
            for i in range(num_test_cases)
        ]

        # Sync check only
        checks = [
            Check(type="exact_match", arguments={
                "actual": "$.output.value.result",
                "expected": "$.test_case.input.value",
            }),
        ]

        # This should be fast as no event loop is created
        start_time = time.time()
        result = evaluate(test_cases, outputs, checks)
        end_time = time.time()

        total_time = end_time - start_time

        # Should complete very quickly (no async overhead)
        assert total_time < 0.5  # generous upper bound

        # Verify all checks completed
        if result.status != "completed":
            print(f"Result status: {result.status}")
            print(f"Summary: {result.summary}")
            for i, test_result in enumerate(result.results[:5]):  # Print first 5
                print(f"Test {i}: {test_result.status}")
                for j, check_result in enumerate(test_result.check_results):
                    print(f"  Check {j}: {check_result.status}, error: {check_result.error}")
        assert result.status == "completed"
        assert result.summary.total_test_cases == num_test_cases
        assert result.summary.completed_test_cases == num_test_cases

    def test_max_async_concurrent_applies_globally(self):
        """Test that max_async_concurrent limits concurrency across all test cases."""
        # Register async check
        register("async_sleep")(AsyncSleepCheck)

        # Create test cases
        num_test_cases = 10
        sleep_duration = 0.1
        max_concurrent = 2

        test_cases = [
            TestCase(id=str(i), input={"value": i})
            for i in range(num_test_cases)
        ]
        outputs = [
            Output(value={"result": i * 10})
            for i in range(num_test_cases)
        ]

        checks = [
            Check(type="async_sleep", arguments={"sleep_duration": sleep_duration}),
        ]

        # Measure execution time with concurrency limit
        start_time = time.time()
        result = evaluate(test_cases, outputs, checks, max_async_concurrent=max_concurrent)
        end_time = time.time()

        total_time = end_time - start_time

        # With max_concurrent=2, should take ~0.5 seconds (10 checks / 2 concurrent * 0.1s)
        # Allow some overhead
        expected_time = (num_test_cases / max_concurrent) * sleep_duration
        assert total_time >= expected_time * 0.8
        assert total_time < expected_time * 1.5

        # Verify all checks completed
        assert result.status == "completed"
        assert result.summary.completed_test_cases == num_test_cases

    def test_complex_per_test_case_performance(self):
        """Test performance with complex per-test-case check scenarios."""
        # Register async check
        register("async_sleep")(AsyncSleepCheck)

        # Create test cases with varying check patterns
        num_test_cases = 1000
        test_cases = []
        for i in range(num_test_cases):
            # Each test case has different numbers of sync/async checks
            checks = []

            # Add 1-3 sync checks per test case
            for _ in range((i % 3) + 1):
                checks.append(Check(
                    type="exact_match",
                    arguments={
                        "actual": "$.output.value.result",
                        "expected": f"{i * 10}",
                    },
                ))

            # Add 0-2 async checks per test case
            for _ in range(i % 3):
                checks.append(Check(
                    type="async_sleep",
                    arguments={"sleep_duration": 0.02},  # 20ms each
                ))

            test_cases.append(TestCase(
                id=f"tc_{i}",
                input={"value": i * 10},
                checks=checks,
            ))

        outputs = [
            Output(value={"result": str(i * 10)})
            for i in range(num_test_cases)
        ]

        # Count total async checks
        total_async_checks = sum(
            len([c for c in tc.checks if c.type == "async_sleep"])
            for tc in test_cases
        )

        # Measure execution time
        start_time = time.time()
        result = evaluate(test_cases, outputs)
        end_time = time.time()

        total_time = end_time - start_time

        # With our optimization, all async checks should run concurrently
        # Sequential would take total_async_checks * 0.02 seconds
        # Concurrent should take ~0.02 seconds (plus overhead)
        sequential_time = total_async_checks * 0.02
        assert total_time < sequential_time * 0.5  # Much faster than sequential

        # Verify all test cases completed
        assert result.status == "completed"
        assert result.summary.total_test_cases == num_test_cases
        assert result.summary.completed_test_cases == num_test_cases

        # Verify each test case has the correct number of checks
        for i, test_result in enumerate(result.results):
            expected_sync_checks = (i % 3) + 1
            expected_async_checks = i % 3
            expected_total = expected_sync_checks + expected_async_checks

            assert len(test_result.check_results) == expected_total

            # Verify all checks completed successfully
            for check_result in test_result.check_results:
                assert check_result.status == "completed"

        print("\nPerformance test results:")
        print(f"Total async checks: {total_async_checks}")
        print(f"Sequential time would be: {sequential_time:.3f}s")
        print(f"Actual time: {total_time:.3f}s")
        print(f"Speedup: {sequential_time / total_time:.1f}x")
