"""
Simplified pytest decorator for statistical evaluation using flex-evals.

This module provides the @evaluate decorator that allows running test functions
multiple times and validating success rates using existing flex-evals checks.
"""

import asyncio
import concurrent.futures
import inspect
import time
import traceback
from typing import Any, TypeVar
from collections.abc import Callable
from functools import wraps

import pytest

from .engine import evaluate as engine_evaluate
from .schemas import TestCase, Output, Check, CheckResult

F = TypeVar('F', bound=Callable[..., Any])


def evaluate(  # noqa: PLR0915
    test_cases: list[TestCase],
    checks: list[Check] | None = None,
    samples: int = 1,
    success_threshold: float = 1.0,
) -> Callable[[F], F]:
    """
    Pytest decorator for statistical evaluation of test functions.

    This decorator executes the wrapped function multiple times and validates
    the success rate using existing flex-evals checks. It integrates with
    pytest's testing framework and provides rich failure reporting.

    Args:
        test_cases: List of TestCase objects to cycle through for evaluation
        checks: Optional list of checks to apply to all samples
        samples: Number of times to execute the function (default: 1)
        success_threshold: Minimum success rate required (0.0 to 1.0, default: 1.0)

    Returns:
        Decorated function that performs statistical evaluation

    Example:
        @evaluate(
            test_cases=[TestCase(id="basic", input="sample input")],
            checks=[Check(
                type=CheckType.CONTAINS,
                arguments={"text": "$.output.value", "phrases": ["expected"]},
            )],
            samples=10,
            success_threshold=0.8
        )
        def test_my_function(test_case):
            response = example_llm_function(test_case.input)
            return {"result": response}

    Error Handling:
        - Function exceptions are caught and counted as failures
        - Invalid check configurations raise ValueError at decoration time
        - Missing 'passed' field in CheckResult raises clear error
        - Incompatible checks (no 'passed' field) are validated early
    """
    def decorator(func: F) -> F:  # noqa: PLR0915
        # Validate decorator parameters
        if not test_cases:
            raise ValueError("test_cases list cannot be empty")
        if samples <= 0:
            raise ValueError("samples must be positive")
        if not 0.0 <= success_threshold <= 1.0:
            raise ValueError("success_threshold must be between 0.0 and 1.0")

        # Validate check configuration
        _validate_check_configuration(test_cases, checks)

        # Check if function expects test_case parameter
        sig = inspect.signature(func)
        expects_test_case = 'test_case' in sig.parameters

        # Create new signature without test_case parameter for pytest
        if expects_test_case:
            new_params = [p for name, p in sig.parameters.items() if name != 'test_case']
            new_sig = sig.replace(parameters=new_params)
        else:
            new_sig = sig

        def _create_error_output(exception, duration_seconds=None):  # noqa: ANN001, ANN202
            """Create an error output for an exception."""
            metadata = {}
            if duration_seconds is not None:
                metadata["duration_seconds"] = duration_seconds

            return Output(
                value={
                    "error": True,
                    "exception_type": type(exception).__name__,
                    "exception_message": str(exception),
                    "traceback": traceback.format_exc() if not isinstance(exception, type) else
                                traceback.format_exception(type(exception), exception, exception.__traceback__),  # noqa: E501
                },
                metadata=metadata if metadata else None,
            )


        def _evaluate_results(expanded_test_cases, outputs, exceptions) -> None:  # noqa: ANN001
            """Evaluate results and check success threshold."""
            try:
                evaluation_result = engine_evaluate(
                    test_cases=expanded_test_cases,
                    outputs=outputs,
                    checks=checks,
                )
            except Exception as e:
                pytest.fail(f"Evaluation failed: {e}")

            # Calculate sample-based success rate
            num_test_cases_per_sample = len(test_cases)
            passed_samples = 0
            failed_samples = []

            for sample_idx in range(samples):
                start_idx = sample_idx * num_test_cases_per_sample
                end_idx = start_idx + num_test_cases_per_sample

                sample_test_case_results = evaluation_result.results[start_idx:end_idx]
                sample_exceptions = exceptions[start_idx:end_idx]

                # Sample passes if ALL test cases pass
                sample_passed = all(
                    _check_sample_passed(tcr.check_results) for tcr in sample_test_case_results
                )

                if sample_passed:
                    passed_samples += 1
                else:
                    failed_samples.append({
                        "sample_index": sample_idx,
                        "exceptions": sample_exceptions,
                        "test_case_results": sample_test_case_results,
                    })

            # Check success threshold
            success_rate = passed_samples / samples
            if success_rate < success_threshold:
                _generate_failure_report(
                    func_name=func.__name__,
                    samples=samples,
                    passed_count=passed_samples,
                    success_rate=success_rate,
                    success_threshold=success_threshold,
                    failed_samples=failed_samples,
                )

        async def _execute_async_calls(expanded_test_cases, args, kwargs) -> None:  # noqa: ANN001
            """Execute async function calls concurrently."""
            # Create tasks for all calls with timing
            async def _timed_call(test_case):  # noqa: ANN001, ANN202
                start_time = time.time()
                try:
                    if expects_test_case:
                        result = await func(test_case, *args, **kwargs)
                    else:
                        result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    return result, duration, None
                except Exception as e:
                    duration = time.time() - start_time
                    return None, duration, e

            tasks = []
            for test_case in expanded_test_cases:
                tasks.append(_timed_call(test_case))

            # Execute concurrently and collect results
            try:
                timed_results = await asyncio.gather(*tasks)
            except Exception as e:
                pytest.fail(f"Function execution failed: {e}")

            # Process timed results
            outputs = []
            exceptions = []
            for result, duration, exception in timed_results:
                if exception:
                    outputs.append(_create_error_output(exception, duration))
                    exceptions.append(exception)
                else:
                    metadata = {"duration_seconds": duration}
                    outputs.append(Output(value=result, metadata=metadata))
                    exceptions.append(None)

            _evaluate_results(expanded_test_cases, outputs, exceptions)

        def _execute_sync_calls(expanded_test_cases, args, kwargs) -> None:  # noqa: ANN001
            """Execute sync function calls sequentially."""
            outputs = []
            exceptions = []

            for test_case in expanded_test_cases:
                start_time = time.time()
                try:
                    if expects_test_case:
                        result = func(test_case, *args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    metadata = {"duration_seconds": duration}
                    outputs.append(Output(value=result, metadata=metadata))
                    exceptions.append(None)
                except Exception as e:
                    duration = time.time() - start_time
                    outputs.append(_create_error_output(e, duration))
                    exceptions.append(e)

            _evaluate_results(expanded_test_cases, outputs, exceptions)

        async def _resolve_async_fixtures(kwargs: dict) -> dict:
            """
            Resolve async fixtures with detailed error context.

            When pytest-asyncio passes async fixtures to test functions, they come
            as coroutine objects that need to be awaited before use.

            Args:
                kwargs: Dictionary that may contain coroutine objects from async fixtures

            Returns:
                Dictionary with all coroutines awaited and resolved to actual values

            Raises:
                RuntimeError: If any async fixture fails to resolve, with detailed context
            """
            resolved_kwargs = {}
            async_fixture_names = []

            for key, value in kwargs.items():
                if inspect.iscoroutine(value):
                    async_fixture_names.append(key)
                    try:
                        resolved_kwargs[key] = await value
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to resolve async fixture '{key}' in flex-evals evaluation.\n"
                            f"Error: {e!s}\n"
                            f"Context: {type(e).__name__}\n"
                            f"All async fixtures detected: {async_fixture_names}\n"
                            f"This usually indicates:\n"
                            f"  1. Event loop context mismatch\n"
                            f"  2. pytest-asyncio configuration issue\n"
                            f"  3. Async fixture implementation problem\n"
                            f"Please check your pytest-asyncio setup and fixture implementation.",
                        ) from e
                else:
                    resolved_kwargs[key] = value

            return resolved_kwargs

        async def _run_with_async_fixtures_resolved(args, kwargs):  # noqa: ANN001, ANN202
            """
            Run evaluation with async fixtures resolved in a new event loop context.

            PROBLEM SOLVED: Before this fix, when users tried to run individual tests with
            async fixtures outside of pytest-asyncio (e.g., VS Code "Run Test" button),
            they would get RuntimeError: "Detected async fixtures but no running event loop".

            SOLUTION: Instead of throwing that error, we create a new event loop and resolve
            the async fixtures before passing them to the evaluation logic. This enables
            IDE integration while maintaining backward compatibility.

            Args:
                args: Function arguments
                kwargs: Function keyword arguments (may contain coroutine objects from async
                fixtures)

            Returns:
                The result of the async evaluation

            Raises:
                RuntimeError: If fixture resolution fails with contextual information
            """
            try:
                # Resolve async fixtures first
                resolved_kwargs = await _resolve_async_fixtures(kwargs)

                # Pass resolved fixtures to existing evaluation logic
                return await _run_async_evaluation(args, resolved_kwargs)

            except Exception as e:
                # Only add context about no-event-loop scenario
                raise RuntimeError(
                    f"Failed in no-event-loop context: {e}",
                ) from e

        def _handle_pytest_asyncio_context(args, kwargs, loop) -> None:  # noqa: ANN001
            """
            Handle the pytest-asyncio context explicitly.

            Args:
                args: Function arguments
                kwargs: Function keyword arguments (may contain async fixtures)
                loop: The running event loop

            Returns:
                The result of the async evaluation

            Raises:
                RuntimeError: If execution fails with detailed context
            """
            async def resolve_and_run():  # noqa: ANN202
                return await _run_async_evaluation(args, kwargs)

            try:
                # Method 1: Try to use loop.run_until_complete if possible
                return loop.run_until_complete(resolve_and_run())

            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    # This is expected in some pytest-asyncio setups
                    # Use threading approach as the designed solution, not a fallback
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, resolve_and_run())
                        try:
                            return future.result(timeout=300)  # 5 minute timeout
                        except concurrent.futures.TimeoutError:
                            raise RuntimeError(
                                "Async evaluation timed out after 5 minutes. "
                                "This may indicate a problem with async fixture resolution.",
                            ) from None
                else:
                    # Unexpected error - don't mask it
                    raise RuntimeError(
                        f"Failed to execute in pytest-asyncio context: {e!s}. "
                        f"This indicates a compatibility issue that needs investigation.",
                    ) from e

        async def _run_async_evaluation(args, kwargs) -> None:  # noqa: ANN001
            """
            Execute async evaluation with proper fixture resolution.

            This function resolves any async fixtures first, then proceeds with
            the standard evaluation logic using the resolved values.
            """
            # IMPORTANT: Resolve async fixtures FIRST before any other processing
            resolved_kwargs = await _resolve_async_fixtures(kwargs)

            # Continue with existing logic but use resolved_kwargs instead of kwargs
            expanded_test_cases = test_cases * samples
            await _execute_async_calls(expanded_test_cases, args, resolved_kwargs)

        def _run_sync_evaluation(args, kwargs) -> None:  # noqa: ANN001
            """Run evaluation for sync functions."""
            expanded_test_cases = test_cases * samples
            _execute_sync_calls(expanded_test_cases, args, kwargs)

        # Create wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            def async_wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
                """Handle async functions with explicit context detection."""
                # Detect if we have any async fixtures
                has_async_fixtures = any(inspect.iscoroutine(v) for v in kwargs.values())

                try:
                    # Check current event loop context
                    loop = asyncio.get_running_loop()

                    # We're in an event loop - this should be pytest-asyncio
                    if has_async_fixtures:
                        # This is the expected case for pytest-asyncio with async fixtures
                        return _handle_pytest_asyncio_context(args, kwargs, loop)
                    # Event loop running but no async fixtures - could be normal pytest-asyncio
                    # usage
                    return _handle_pytest_asyncio_context(args, kwargs, loop)

                except RuntimeError:
                    # No event loop running - standalone usage (e.g., VS Code "Run Test")
                    if has_async_fixtures:
                        # FIX: Instead of throwing "Detected async fixtures but no running event
                        # loop", create a new event loop and resolve the async fixtures properly.
                        # This enables IDE integration for tests with async fixtures.
                        return asyncio.run(_run_with_async_fixtures_resolved(args, kwargs))
                    # Normal standalone case (no async fixtures)
                    return asyncio.run(_run_async_evaluation(args, kwargs))

            async_wrapper.__signature__ = new_sig
            return async_wrapper
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
            _run_sync_evaluation(args, kwargs)

        sync_wrapper.__signature__ = new_sig
        return sync_wrapper

    return decorator


def _validate_check_configuration(test_cases: list[TestCase], checks: list[Check] | None) -> None:
    """Validate that check configuration is valid for evaluation."""
    if checks is None:
        for test_case in test_cases:
            if not test_case.checks:
                raise ValueError(
                    f"When checks=None, each TestCase must define its own checks. "
                    f"TestCase '{test_case.id}' has no checks defined.",
                )


def _check_sample_passed(check_results: list[CheckResult]) -> bool:
    """Check if a sample passed by examining all check results."""
    if not check_results:
        return False

    for check_result in check_results:
        if check_result.status != 'completed':
            return False

        if 'passed' not in check_result.results:
            raise ValueError(
                f"Check result for '{check_result.check_type}' is missing 'passed' field. "
                f"This check type is not compatible with the @evaluate decorator.",
            )

        if not check_result.results['passed']:
            return False

    return True


def _generate_failure_report(
    func_name: str,
    samples: int,
    passed_count: int,
    success_rate: float,
    success_threshold: float,
    failed_samples: list[dict],
) -> None:
    """Generate detailed failure report and call pytest.fail()."""
    failed_count = samples - passed_count

    report_lines = [
        f"Statistical evaluation failed for {func_name}:",
        f"  Total samples: {samples}",
        f"  Passed: {passed_count}",
        f"  Failed: {failed_count}",
        f"  Success rate: {success_rate:.2%}",
        f"  Required threshold: {success_threshold:.2%}",
        "",
        "Failed samples:",
    ]

    for failure in failed_samples[:5]:  # Show first 5 failures
        sample_idx = failure["sample_index"]
        exceptions = failure["exceptions"]
        test_case_results = failure["test_case_results"]

        report_lines.append(f"  Sample {sample_idx}:")

        # Report exceptions
        for i, exception in enumerate(exceptions):
            if exception:
                test_case_id = test_case_results[i].execution_context.test_case.id
                report_lines.append(f"    Test case {i} (id: {test_case_id}) exception: {type(exception).__name__}: {exception}")  # noqa: E501

        # Report check failures
        for i, test_case_result in enumerate(test_case_results):
            test_case_id = test_case_result.execution_context.test_case.id
            for check_result in test_case_result.check_results:
                if check_result.status != 'completed':
                    report_lines.append(f"    Test case {i} (id: {test_case_id}) check '{check_result.check_type}': {check_result.status}")  # noqa: E501
                    if check_result.error:
                        report_lines.append(f"      Error: {check_result.error.message}")
                    if check_result.resolved_arguments:
                        report_lines.append(f"      Resolved arguments: {check_result.resolved_arguments}")  # noqa: E501
                elif not check_result.results.get('passed', False):
                    report_lines.append(f"    Test case {i} (id: {test_case_id}) check '{check_result.check_type}': failed")  # noqa: E501
                    if 'reasoning' in check_result.results:
                        report_lines.append(f"      Reasoning: {check_result.results['reasoning']}")  # noqa: E501
                    if check_result.resolved_arguments:
                        report_lines.append(f"      Resolved arguments: {check_result.resolved_arguments}")  # noqa: E501

    if len(failed_samples) > 5:
        report_lines.append(f"  ... and {len(failed_samples) - 5} more failures")

    pytest.fail("\n".join(report_lines))
