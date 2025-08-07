"""
Tests for async fixture support in flex-evals pytest decorator.

PROBLEM SOLVED:
The @evaluate decorator in flex-evals failed when test functions used async fixtures
from pytest-asyncio. This caused AttributeError: 'coroutine' object has no attribute 'X'
because:

1. pytest-asyncio passes async fixtures as coroutine objects (not awaited values)
2. flex-evals created a new event loop with asyncio.run(), breaking the pytest-asyncio
   event loop context
3. The coroutine objects couldn't be awaited in the new event loop

ROOT CAUSE:
The original async_wrapper used asyncio.run() unconditionally:

    def async_wrapper(*args, **kwargs):
        return asyncio.run(_run_async_evaluation(args, kwargs))  # BROKE PYTEST-ASYNCIO

This created a new event loop that was disconnected from pytest's loop, making it
impossible to await the coroutine objects from async fixtures.

SOLUTION IMPLEMENTED:
1. **Context Detection**: Check if we're in pytest-asyncio vs standalone usage
2. **Fixture Resolution**: Add _resolve_async_fixtures() to await coroutine objects
3. **Event Loop Handling**: Use appropriate execution strategy based on context:
   - pytest-asyncio: Use existing loop or thread-based execution
   - Standalone: Use asyncio.run() as before
4. **Error Handling**: Provide clear error messages for configuration issues

KEY FUNCTIONS ADDED:
- _resolve_async_fixtures(): Awaits coroutine objects from async fixtures
- _handle_pytest_asyncio_context(): Manages execution in pytest-asyncio contexts
- Enhanced async_wrapper: Detects context and chooses appropriate execution strategy

BACKWARD COMPATIBILITY:
All existing functionality is preserved. The fix only adds support for async fixtures
without changing behavior for existing use cases.
"""

import asyncio
import time
import pytest
import _pytest.outcomes
from flex_evals import TestCase, Check, CheckType
from flex_evals.pytest_decorator import evaluate
from typing import Never


@pytest.fixture
async def simple_async_fixture():
    """Simple async fixture that returns a string."""
    await asyncio.sleep(0.001)  # Simulate async work
    return "async_fixture_value"


@pytest.fixture
async def complex_async_fixture():
    """Complex async fixture that returns an object."""
    class AsyncResult:
        def __init__(self, value: str):
            self.value = value

        def get_data(self) -> str:
            return f"data_{self.value}"

    await asyncio.sleep(0.001)  # Simulate async work
    return AsyncResult("test")


@pytest.fixture
async def async_list_fixture():
    """Async fixture that returns a list."""
    await asyncio.sleep(0.001)
    return ["item1", "item2", "item3"]


@pytest.fixture
def sync_fixture():
    """Regular sync fixture for mixed fixture tests."""
    return "sync_value"


class TestAsyncFixtureBasic:
    """
    Test that async fixtures are properly resolved - before the fix they would be coroutine
    objects.
    """

    @evaluate(
        test_cases=[TestCase(id="simple", input="test")],
        checks=[Check(
            type=CheckType.CONTAINS,
            arguments={"text": "$.output.value", "phrases": ["async_fixture_value"]},
        )],
        samples=2,
        success_threshold=1.0,
    )
    async def test_simple_async_fixture(
            self,
            test_case: TestCase,  # noqa: ARG002
            simple_async_fixture: str,
        ) -> str:
        """Test async fixture returns actual string value, not coroutine object."""
        return f"Got: {simple_async_fixture}"

    @evaluate(
        test_cases=[TestCase(id="complex", input="test")],
        checks=[Check(
            type=CheckType.CONTAINS,
            arguments={"text": "$.output.value", "phrases": ["data_test"]},
        )],
        samples=2,
        success_threshold=1.0,
    )
    async def test_complex_async_fixture(
            self,
            test_case: TestCase,  # noqa: ARG002
            complex_async_fixture: object,
        ) -> str:
        """Test async fixture returns actual object with callable methods."""
        return complex_async_fixture.get_data()

    @evaluate(
        test_cases=[TestCase(id="list", input="test")],
        checks=[Check(
            type=CheckType.CONTAINS,
            arguments={"text": "$.output.value", "phrases": ["item1", "item2"]},
        )],
        samples=1,
        success_threshold=1.0,
    )
    async def test_async_list_fixture(
            self,
            test_case: TestCase,  # noqa: ARG002
            async_list_fixture: list,
        ) -> str:
        """Test async fixture returns actual list that can be iterated/joined."""
        return f"Items: {', '.join(async_list_fixture)}"


class TestMixedFixtures:
    """Test that _resolve_async_fixtures() handles both sync and async fixtures correctly."""

    @evaluate(
        test_cases=[TestCase(id="mixed", input="test")],
        checks=[Check(
            type=CheckType.CONTAINS,
            arguments={"text": "$.output.value", "phrases": ["sync_value", "async_fixture_value"]},
        )],
        samples=2,
        success_threshold=1.0,
    )
    async def test_mixed_sync_and_async_fixtures(
        self,
        test_case: TestCase,  # noqa: ARG002
        sync_fixture: str,
        simple_async_fixture: str,
    ) -> str:
        """Test both sync and async fixtures are resolved correctly in same test."""
        return f"Sync: {sync_fixture}, Async: {simple_async_fixture}"


class TestMultipleAsyncFixtures:
    """Test that multiple async fixtures can be resolved simultaneously."""

    @evaluate(
        test_cases=[TestCase(id="multiple", input="test")],
        checks=[Check(
            type=CheckType.CONTAINS,
            arguments={"text": "$.output.value", "phrases": ["async_fixture_value", "data_test"]},
        )],
        samples=1,
        success_threshold=1.0,
    )
    async def test_multiple_async_fixtures(
        self,
        test_case: TestCase,  # noqa: ARG002
        simple_async_fixture: str,
        complex_async_fixture: object,
    ) -> str:
        """Test multiple async fixtures are all resolved correctly."""
        return f"Simple: {simple_async_fixture}, Complex: {complex_async_fixture.get_data()}"


class TestAsyncFixtureStatisticalEvaluation:
    """Test that async fixtures work with statistical sampling (multiple executions)."""

    @pytest.fixture
    async def variable_async_fixture(self):
        """Async fixture that provides data for statistical testing."""
        await asyncio.sleep(0.001)
        return {"success_rate": 0.8}

    call_count = 0

    @evaluate(
        test_cases=[TestCase(id="stats", input="test")],
        checks=[Check(
            type=CheckType.CONTAINS,
            arguments={"text": "$.output.value", "phrases": ["success"]},
        )],
        samples=10,
        success_threshold=0.7,  # 70% threshold
    )
    async def test_statistical_with_async_fixture(
        self,
        test_case: TestCase,  # noqa: ARG002
        variable_async_fixture: dict,
    ) -> str:
        """Test async fixtures work with multiple sample executions."""
        # Use class variable to track calls across samples
        TestAsyncFixtureStatisticalEvaluation.call_count += 1
        variable_async_fixture["success_rate"]

        # Simulate 80% success rate
        if TestAsyncFixtureStatisticalEvaluation.call_count % 10 < 8:
            return "success result"
        return "failure result"


class TestAsyncFixturePerformance:
    """Test that async fixtures don't break concurrent execution of samples."""

    @pytest.fixture
    async def timed_async_fixture(self):
        """Async fixture with artificial delay."""
        await asyncio.sleep(0.05)  # 50ms delay
        return "timed_value"

    @evaluate(
        test_cases=[TestCase(id="timing", input="test")],
        checks=[Check(
            type=CheckType.CONTAINS,
            arguments={"text": "$.output.value", "phrases": ["timed_value"]},
        )],
        samples=5,  # 5 samples with 50ms fixture delay each
        success_threshold=1.0,
    )
    async def test_async_fixture_concurrency(
        self,
        test_case: TestCase,  # noqa: ARG002
        timed_async_fixture: str,
    ) -> str:
        """Test async fixtures don't prevent concurrent sample execution."""
        await asyncio.sleep(0.01)  # Additional 10ms test delay
        return f"Result: {timed_async_fixture}"


class TestAsyncFixtureContextValidation:
    """Test that async fixtures work correctly in pytest-asyncio contexts."""

    @evaluate(
        test_cases=[TestCase(id="context", input="test")],
        checks=[Check(
            type=CheckType.CONTAINS,
            arguments={"text": "$.output.value", "phrases": ["async_fixture_value"]},
        )],
        samples=1,
        success_threshold=1.0,
    )
    async def test_async_fixture_in_pytest_asyncio_context(
        self,
        test_case: TestCase,  # noqa: ARG002
        simple_async_fixture: str,
    ) -> str:
        """Test async fixtures work in pytest-asyncio event loop context."""
        return f"Context test: {simple_async_fixture}"


class TestAsyncFixtureFailureScenarios:
    """Test that @evaluate with async functions correctly fails when it should."""

    def test_async_function_check_failure(self):
        """Test that @evaluate fails when async test returns wrong value."""
        @evaluate(
            test_cases=[TestCase(id="should_fail", input="test")],
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "expected_value", "actual": "$.output.value"},
            )],
            samples=2,
            success_threshold=1.0,
        )
        async def failing_async_test(test_case: TestCase) -> str:  # noqa: ARG001
            # Return wrong value - should fail the exact match check
            return "wrong_value"

        # This meta-test should PASS by confirming the evaluation FAILS
        with pytest.raises(_pytest.outcomes.Failed) as exc_info:
            failing_async_test()

        error_message = str(exc_info.value)
        assert "Statistical evaluation failed" in error_message
        assert "Success rate: 0.00%" in error_message

    def test_async_function_exception_failure(self):
        """Test that @evaluate fails when async test throws exceptions."""
        @evaluate(
            test_cases=[TestCase(id="exception_test", input="test")],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["success"]},
            )],
            samples=2,
            success_threshold=1.0,
        )
        async def exception_async_test(test_case: TestCase) -> str:  # noqa: ARG001
            # Always throw exception - should fail
            raise ValueError("Test exception")

        # This meta-test should PASS by confirming the evaluation FAILS
        with pytest.raises(_pytest.outcomes.Failed) as exc_info:
            exception_async_test()

        error_message = str(exc_info.value)
        assert "Statistical evaluation failed" in error_message
        assert "Test case 0 exception: ValueError" in error_message


class TestNoEventLoopContext:
    """
    Test async fixtures when no event loop is running (simulates VS Code click-to-run).

    ISSUE TESTED: Before the fix, running @evaluate decorated functions with async fixtures
    outside of pytest-asyncio context (e.g., VS Code "Run Test" button) would fail with:
    RuntimeError: "Detected async fixtures but no running event loop"

    FIX VALIDATED: These tests confirm that async fixtures are now properly resolved
    by creating a new event loop context, enabling IDE integration.
    """

    def test_direct_call_with_async_fixture(self):
        """Simulate IDE direct test execution with async fixtures."""
        # Ensure no event loop is running
        try:
            asyncio.get_running_loop()
            pytest.skip("Event loop already running - can't test no-loop scenario")
        except RuntimeError:
            pass  # Good - no loop running

        # Create a test function with decorator
        @evaluate(
            test_cases=[TestCase(id="direct_call", input="test_input")],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["fixture_value"]},
            )],
            samples=2,
            success_threshold=1.0,
        )
        async def test_func(test_case: TestCase, simple_fixture) -> str:  # noqa: ANN001, ARG001
            """Test function that uses an async fixture."""
            return f"Got: {simple_fixture}"

        # Get the wrapped function directly
        wrapped_func = test_func

        # Create a coroutine object to simulate pytest-asyncio fixture
        async def mock_async_fixture() -> str:
            await asyncio.sleep(0.001)
            return "fixture_value"

        # Simulate direct call with coroutine object (as pytest-asyncio would pass it)
        # This should NOT raise RuntimeError anymore
        result = wrapped_func(simple_fixture=mock_async_fixture())

        # Should complete successfully
        assert result is None  # pytest convention

    def test_multiple_async_fixtures_no_loop(self):
        """Test multiple async fixtures in no-event-loop context."""
        # Ensure no event loop is running
        try:
            asyncio.get_running_loop()
            pytest.skip("Event loop already running - can't test no-loop scenario")
        except RuntimeError:
            pass

        @evaluate(
            test_cases=[TestCase(id="multi_fixture", input="test")],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["value1", "value2"]},
            )],
            samples=1,
            success_threshold=1.0,
        )
        async def test_multi_fixtures(test_case: TestCase, fixture1, fixture2) -> str:  # noqa: ANN001, ARG001
            """Test with multiple async fixtures."""
            return f"Results: {fixture1}, {fixture2}"

        # Mock async fixtures as coroutine objects
        async def async_fixture1() -> str:
            await asyncio.sleep(0.001)
            return "value1"

        async def async_fixture2() -> str:
            await asyncio.sleep(0.001)
            return "value2"

        # Direct call with coroutine objects
        result = test_multi_fixtures(
            fixture1=async_fixture1(),
            fixture2=async_fixture2(),
        )

        assert result is None

    def test_mixed_sync_async_fixtures_no_loop(self):
        """Test mix of sync and async fixtures in no-event-loop context."""
        try:
            asyncio.get_running_loop()
            pytest.skip("Event loop already running - can't test no-loop scenario")
        except RuntimeError:
            pass

        @evaluate(
            test_cases=[TestCase(id="mixed", input="test")],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["sync_val", "async_val"]},
            )],
            samples=2,
            success_threshold=1.0,
        )
        async def test_mixed(test_case: TestCase, sync_fix, async_fix) -> str:  # noqa: ANN001, ARG001
            """Test with mixed fixture types."""
            return f"Sync: {sync_fix}, Async: {async_fix}"

        async def async_fixture() -> str:
            await asyncio.sleep(0.001)
            return "async_val"

        # Direct call with mixed fixtures
        result = test_mixed(
            sync_fix="sync_val",  # Regular value
            async_fix=async_fixture(),  # Coroutine object
        )

        assert result is None

    def test_fixture_error_handling_no_loop(self):
        """
        Test error handling when fixture fails in no-event-loop context.

        PURPOSE: Validates that when an async fixture throws an exception during setup,
        the error message is informative and includes both the original fixture error
        AND context about the no-event-loop scenario. This helps developers debug
        fixture issues when running tests outside pytest-asyncio context.

        NOTE: This throws RuntimeError (not pytest.Failed) because fixture resolution
        fails before the evaluation can even run.
        """
        try:
            asyncio.get_running_loop()
            pytest.skip("Event loop already running - can't test no-loop scenario")
        except RuntimeError:
            pass

        @evaluate(
            test_cases=[TestCase(id="error_test", input="test")],
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "test", "actual": "$.output.value"},
            )],
            samples=2,
            success_threshold=1.0,
        )
        async def test_with_failing_fixture(test_case: TestCase, failing_fixture):  # noqa: ANN001, ANN202, ARG001
            """Test function with fixture that will fail."""
            return failing_fixture

        async def failing_async_fixture() -> Never:
            await asyncio.sleep(0.001)
            raise ValueError("Fixture setup failed")

        # This should fail with informative error about the fixture failure
        # Note: This fails during fixture resolution, not during test evaluation,
        # so we get RuntimeError directly, not wrapped in pytest.Failed
        with pytest.raises(RuntimeError) as exc_info:
            test_with_failing_fixture(failing_fixture=failing_async_fixture())

        error_message = str(exc_info.value)
        # The error should contain both our no-event-loop context AND the underlying fixture error
        assert "Failed in no-event-loop context" in error_message
        assert "Fixture setup failed" in error_message

    def test_performance_no_loop_vs_normal(self):
        """Compare performance between no-loop and normal execution."""
        try:
            asyncio.get_running_loop()
            pytest.skip("Event loop already running - can't test no-loop scenario")
        except RuntimeError:
            pass

        @evaluate(
            test_cases=[TestCase(id="perf", input="test")],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["result"]},
            )],
            samples=5,
            success_threshold=1.0,
        )
        async def test_performance(test_case: TestCase, perf_fixture) -> str:  # noqa: ANN001, ARG001
            """Test function for performance measurement."""
            return f"result: {perf_fixture}"

        async def perf_async_fixture() -> str:
            await asyncio.sleep(0.001)
            return "perf_value"

        # Measure time for no-loop execution
        start_time = time.time()
        test_performance(perf_fixture=perf_async_fixture())
        duration = time.time() - start_time

        # Should complete reasonably quickly (less than 2 seconds for 5 samples)
        assert duration < 2.0, f"Took too long: {duration}s"


class TestMixedContextCompatibility:
    """
    Ensure the fix doesn't break existing pytest-asyncio contexts.

    BACKWARD COMPATIBILITY: These tests validate that our no-event-loop fix doesn't
    interfere with existing pytest-asyncio functionality. Users should be able to
    use async fixtures in both contexts without any changes to their code.
    """

    def test_pytest_asyncio_compatibility(self, simple_async_fixture):  # noqa: ANN001
        """
        Ensure pytest-asyncio context compatibility is maintained.

        PURPOSE: Validates that existing pytest-asyncio usage patterns continue to work
        after our fix. The fixture is already resolved to its value in pytest context,
        so we pass it as a regular parameter to avoid event loop conflicts.
        """
        # This test runs in pytest context but doesn't use @pytest.mark.asyncio
        # to avoid the event loop conflict issue

        @evaluate(
            test_cases=[TestCase(id="compat", input="test")],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["async_fixture_value"]},
            )],
            samples=1,
            success_threshold=1.0,
        )
        async def test_func(
                test_case: TestCase,  # noqa: ARG001
                fixture_value: str,
            ) -> str:
            """Test that uses async fixture value in pytest context."""
            return f"Result: {fixture_value}"

        # The simple_async_fixture is already resolved to its string value in pytest context
        # So we pass it directly as a regular parameter
        result = test_func(fixture_value=simple_async_fixture)
        assert result is None

    def test_no_async_fixtures_still_works(self):
        """Test that functions without async fixtures still work normally."""
        @evaluate(
            test_cases=[TestCase(id="no_async", input="test")],
            checks=[Check(
                type=CheckType.EXACT_MATCH,
                arguments={"expected": "plain_result", "actual": "$.output.value"},
            )],
            samples=3,
            success_threshold=1.0,
        )
        async def test_no_async(test_case: TestCase) -> str:  # noqa: ARG001
            """Test without any async fixtures."""
            await asyncio.sleep(0.001)
            return "plain_result"

        # Should work as before
        result = test_no_async()
        assert result is None

    @evaluate(
        test_cases=[TestCase(id="async_fixture_duration", input="test")],
        checks=[
            Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["async_fixture_value"]},
            ),
            Check(
                type=CheckType.IS_EMPTY,
                arguments={"value": "$.output.metadata.duration_seconds", "negate": True},
            ),
        ],
        samples=1,
        success_threshold=1.0,
    )
    async def test_async_with_fixtures_duration_populated(
            self,
            test_case: TestCase,  # noqa: ARG002
            simple_async_fixture: str,
        ) -> str:
        """Test that async functions with fixtures populate duration_seconds."""
        return f"Got: {simple_async_fixture}"
