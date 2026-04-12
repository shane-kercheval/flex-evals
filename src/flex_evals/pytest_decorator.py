"""
Simplified pytest decorator for statistical evaluation using flex-evals.

This module provides the @evaluate decorator that allows running test functions
multiple times and validating success rates using existing flex-evals checks.
"""

import asyncio
import concurrent.futures
import gc
import inspect
import json
import logging
import os
import sys
import time
import traceback
import uuid
import warnings
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, TypeVar
from collections.abc import Callable, Coroutine
from functools import wraps

import pytest

from .constants import PassMode
from .engine import evaluate as evaluate_async, evaluate_sync
from .schemas import TestCase, Output, Check, CheckResult, FlexEvalsJSONEncoder

F = TypeVar('F', bound=Callable[..., Any])


T = TypeVar('T')


def _run_async_suppressing_cleanup_errors(coro: Coroutine[object, object, T]) -> T:
    """Run async coroutine via asyncio.run(), suppressing event-loop-closed cleanup noise."""
    original_hook = sys.unraisablehook

    def _suppressing_hook(unraisable: object) -> None:
        if (
            isinstance(unraisable.exc_value, RuntimeError)
            and "Event loop is closed" in str(unraisable.exc_value)
        ):
            return
        original_hook(unraisable)

    class _EventLoopClosedFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "Event loop is closed" not in record.getMessage()

    asyncio_logger = logging.getLogger("asyncio")
    log_filter = _EventLoopClosedFilter()

    sys.unraisablehook = _suppressing_hook
    asyncio_logger.addFilter(log_filter)
    try:
        return asyncio.run(coro)
    finally:
        gc.collect()  # Force __del__ methods while suppression is active
        sys.unraisablehook = original_hook
        asyncio_logger.removeFilter(log_filter)


def evaluate(  # noqa: PLR0915
    test_cases: list[TestCase],
    checks: list[Check] | None = None,
    samples: int = 1,
    pass_threshold: float = 1.0,
    pass_mode: PassMode | str = PassMode.PER_TEST_CASE,
    max_concurrency: int | None = None,
    output_dir: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """
    Pytest decorator for statistical evaluation of test functions.

    This decorator executes the wrapped function multiple times and validates
    the success rate using existing flex-evals checks. It integrates with
    pytest's testing framework and provides rich failure reporting.

    Note:
        All checks used with this decorator must return a 'passed' field in their results.
        Standard checks (exact_match, contains, regex, threshold) include this field by
        default. Custom checks must include 'passed': bool in their return dictionary.

    Args:
        test_cases: List of TestCase objects to cycle through for evaluation
        checks: Optional list of checks to apply to all samples
        samples: Number of times to execute the function (default: 1)
        pass_threshold: Minimum pass rate required (0.0 to 1.0, default: 1.0).
            How this threshold is applied depends on ``pass_mode``.
        pass_mode: How pass/fail is aggregated (default: PassMode.PER_TEST_CASE).
            Accepts PassMode enum or string ('sample', 'per_test_case').

            - ``PassMode.PER_TEST_CASE``: Each test case is evaluated independently
              across samples. A test case passes if it succeeds in at least
              ``pass_threshold`` proportion of samples. The overall eval passes if ALL
              test cases meet their individual threshold. Use this mode when test cases
              are independent scenarios (e.g., different inputs, tag suggestions,
              classification evals). This is the default and the right choice for most
              LLM evaluations.

            - ``PassMode.SAMPLE``: A sample passes only if ALL test cases pass within
              that sample. The overall eval passes if at least ``pass_threshold``
              proportion of samples fully pass. Use this mode when test cases are
              interdependent or sequential (e.g., end-to-end pipeline tests where
              "the whole pipeline works" is the meaningful question).

        max_concurrency: Maximum number of async test cases to execute concurrently
            (default: None, meaning unlimited). When set, a semaphore limits how many
            test cases run at once. The semaphore is acquired before timing starts, so
            queuing time is excluded from duration_seconds. Only affects async functions;
            sync functions always run sequentially. Must be a positive integer if set.
        output_dir: Directory to save evaluation results as JSON. If None, falls back to
            the FLEX_EVALS_OUTPUT_DIR environment variable. If neither is set, results are
            not saved.
        metadata: Optional dictionary of metadata to include in the saved result. Keys
            prefixed with ``_test_`` are reserved for auto-captured data and will be
            overwritten if provided.

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
            pass_threshold=0.8,
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
    # Normalize pass_mode from str to PassMode enum
    try:
        pass_mode = PassMode(pass_mode)
    except ValueError:
        raise ValueError(
            f"Invalid pass_mode: {pass_mode!r}. "
            f"Must be one of: {', '.join(repr(m.value) for m in PassMode)}",
        ) from None

    def decorator(func: F) -> F:  # noqa: PLR0915
        # Validate decorator parameters
        if not test_cases:
            raise ValueError("test_cases list cannot be empty")
        if samples <= 0:
            raise ValueError("samples must be positive")
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be between 0.0 and 1.0")
        if max_concurrency is not None and max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer")

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


        def _save_result_if_configured(evaluation_result) -> None:  # noqa: ANN001
            """Save evaluation result to JSON if output_dir is configured."""
            try:
                effective_dir = output_dir if output_dir is not None else os.environ.get('FLEX_EVALS_OUTPUT_DIR')  # noqa: E501
                if effective_dir is None:
                    return

                num_tc = len(test_cases)
                sample_passed, sample_failed, sample_pass_rate = _compute_sample_results(
                    evaluation_result.results, num_tc, samples,
                )
                per_tc = _compute_per_test_case_results(
                    evaluation_result.results, num_tc, samples,
                )

                if pass_mode == PassMode.PER_TEST_CASE:
                    overall_passed = all(
                        entry["rate"] >= pass_threshold for entry in per_tc
                    )
                else:
                    overall_passed = sample_pass_rate >= pass_threshold

                evaluation_result.metadata = {
                    **(evaluation_result.metadata or {}),
                    **(metadata or {}),
                    '_test_config': {
                        'test_function': func.__name__,
                        'test_module': func.__module__,
                        'samples': samples,
                        'pass_threshold': pass_threshold,
                        'pass_mode': str(pass_mode),
                        'max_concurrency': max_concurrency,
                        'num_test_cases': num_tc,
                    },
                    '_test_results': {
                        'pass_mode': str(pass_mode),
                        'pass_threshold': pass_threshold,
                        'passed': overall_passed,
                        'passed_samples': sample_passed,
                        'failed_samples': sample_failed,
                        'total_samples': samples,
                        'sample_pass_rate': sample_pass_rate,
                        'per_test_case': per_tc,
                        'failed_test_cases': [
                            {"index": e["index"], "id": e["id"]}
                            for e in per_tc if e["rate"] < pass_threshold
                        ],
                    },
                }

                serialized = evaluation_result.serialize()
                timestamp = datetime.now(UTC).isoformat().replace(':', '-')
                filename = f"{timestamp}_{uuid.uuid4().hex[:8]}.json"
                out_path = Path(effective_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                (out_path / filename).write_text(
                    json.dumps(serialized, indent=2, cls=FlexEvalsJSONEncoder),
                )
            except Exception as e:
                warnings.warn(
                    f"Failed to save evaluation result: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        def _evaluate_results_sync(expanded_test_cases, outputs, exceptions, eval_started_at) -> None:  # noqa: ANN001, E501
            """Evaluate results and check success threshold (synchronous)."""
            try:
                evaluation_result = evaluate_sync(
                    test_cases=expanded_test_cases,
                    outputs=outputs,
                    checks=checks,
                )
            except Exception as e:
                pytest.fail(f"Evaluation failed: {e}")

            evaluation_result.started_at = eval_started_at
            _save_result_if_configured(evaluation_result)
            _process_evaluation_result(evaluation_result, exceptions)

        async def _evaluate_results_async(expanded_test_cases, outputs, exceptions, eval_started_at) -> None:  # noqa: ANN001, E501
            """Evaluate results and check success threshold (asynchronous)."""
            try:
                evaluation_result = await evaluate_async(
                    test_cases=expanded_test_cases,
                    outputs=outputs,
                    checks=checks,
                )
            except Exception as e:
                pytest.fail(f"Evaluation failed: {e}")

            evaluation_result.started_at = eval_started_at
            _save_result_if_configured(evaluation_result)
            _process_evaluation_result(evaluation_result, exceptions)

        def _process_evaluation_result(evaluation_result, exceptions) -> None:  # noqa: ANN001
            """Process evaluation result and check pass threshold."""
            num_tc = len(test_cases)

            if pass_mode == PassMode.PER_TEST_CASE:
                per_tc = _compute_per_test_case_results(
                    evaluation_result.results, num_tc, samples,
                )
                failed_entries = [e for e in per_tc if e["rate"] < pass_threshold]
                if failed_entries:
                    _generate_per_test_case_failure_report(
                        func_name=func.__name__,
                        pass_threshold=pass_threshold,
                        per_test_case_results=per_tc,
                        all_results=evaluation_result.results,
                        all_exceptions=exceptions,
                        num_test_cases=num_tc,
                        samples=samples,
                    )
            else:
                # Sample mode: existing logic
                passed_samples = 0
                failed_samples = []

                for sample_idx in range(samples):
                    start_idx = sample_idx * num_tc
                    end_idx = start_idx + num_tc

                    sample_test_case_results = evaluation_result.results[start_idx:end_idx]
                    sample_exceptions = exceptions[start_idx:end_idx]

                    sample_passed = all(
                        _check_sample_passed(tcr.check_results)
                        for tcr in sample_test_case_results
                    )

                    if sample_passed:
                        passed_samples += 1
                    else:
                        failed_samples.append({
                            "sample_index": sample_idx,
                            "exceptions": sample_exceptions,
                            "test_case_results": sample_test_case_results,
                        })

                sample_pass_rate = passed_samples / samples
                if sample_pass_rate < pass_threshold:
                    _generate_sample_failure_report(
                        func_name=func.__name__,
                        samples=samples,
                        passed_count=passed_samples,
                        sample_pass_rate=sample_pass_rate,
                        pass_threshold=pass_threshold,
                        failed_samples=failed_samples,
                    )

        async def _execute_async_calls(expanded_test_cases, args, kwargs) -> None:  # noqa: ANN001
            """Execute async function calls concurrently."""
            eval_started_at = datetime.now(UTC)
            semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None

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

            async def _throttled_call(test_case):  # noqa: ANN001, ANN202
                if semaphore is not None:
                    async with semaphore:
                        return await _timed_call(test_case)
                return await _timed_call(test_case)

            tasks = [_throttled_call(tc) for tc in expanded_test_cases]

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

            await _evaluate_results_async(
                expanded_test_cases, outputs, exceptions, eval_started_at,
            )

        def _execute_sync_calls(expanded_test_cases, args, kwargs) -> None:  # noqa: ANN001
            """Execute sync function calls sequentially."""
            eval_started_at = datetime.now(UTC)
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

            _evaluate_results_sync(expanded_test_cases, outputs, exceptions, eval_started_at)

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
                        future = executor.submit(
                            _run_async_suppressing_cleanup_errors,
                            resolve_and_run(),
                        )
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
                        return _run_async_suppressing_cleanup_errors(
                            _run_with_async_fixtures_resolved(args, kwargs),
                        )
                    # Normal standalone case (no async fixtures)
                    return _run_async_suppressing_cleanup_errors(
                        _run_async_evaluation(args, kwargs),
                    )

            async_wrapper.__signature__ = new_sig
            return async_wrapper
        if max_concurrency is not None:
            warnings.warn(
                "max_concurrency has no effect on synchronous test functions",
                UserWarning,
                stacklevel=2,
            )
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


def _compute_sample_results(
    results: list,
    num_test_cases: int,
    samples: int,
) -> tuple[int, int, float]:
    """
    Compute sample-level pass/fail stats.

    Returns (passed_samples, failed_samples, sample_pass_rate).

    Results list is assumed to be in test_cases * samples order:
    all test cases for sample 0, then sample 1, etc.
    """
    passed = sum(
        1 for s_idx in range(samples)
        if all(
            _check_sample_passed(results[s_idx * num_test_cases + tc_idx].check_results)
            for tc_idx in range(num_test_cases)
        )
    )
    failed = samples - passed
    rate = passed / samples if samples > 0 else 0.0
    return passed, failed, rate


def _compute_per_test_case_results(
    results: list,
    num_test_cases: int,
    samples: int,
) -> list[dict[str, Any]]:
    """
    Compute per-test-case pass rates across samples.

    Returns list preserving original test case order, each entry:
        {"index": int, "id": str | None, "passed": int, "failed": int,
         "total": int, "rate": float}

    Grouping is by positional index, not by ID, to handle duplicate or None IDs.

    Results list is assumed to be in test_cases * samples order:
    all test cases for sample 0, then sample 1, etc.
    """
    per_tc: list[dict[str, Any]] = []
    for tc_idx in range(num_test_cases):
        passed = 0
        for s_idx in range(samples):
            tcr = results[s_idx * num_test_cases + tc_idx]
            if _check_sample_passed(tcr.check_results):
                passed += 1
        failed = samples - passed
        tc_id = results[tc_idx].execution_context.test_case.id
        per_tc.append({
            "index": tc_idx,
            "id": tc_id,
            "passed": passed,
            "failed": failed,
            "total": samples,
            "rate": passed / samples if samples > 0 else 0.0,
        })
    return per_tc


def _format_test_case_label(entry: dict[str, Any]) -> str:
    """Format a test case label with index and optional ID."""
    if entry["id"] is not None:
        return f"test_case {entry['index']} (id: {entry['id']})"
    return f"test_case {entry['index']}"


def _generate_per_test_case_failure_report(  # noqa: PLR0912
    func_name: str,
    pass_threshold: float,
    per_test_case_results: list[dict[str, Any]],
    all_results: list,
    all_exceptions: list,
    num_test_cases: int,
    samples: int,
) -> None:
    """Generate per-test-case failure report and call pytest.fail()."""
    failed_count = sum(1 for e in per_test_case_results if e["rate"] < pass_threshold)
    total_count = len(per_test_case_results)

    report_lines = [
        f"Statistical evaluation failed for {func_name}:",
        "  Pass mode: per_test_case",
        f"  Pass threshold: {pass_threshold:.2%}",
        "",
        "  Per-test-case results:",
    ]

    # Show all test cases (up to 20), mark failures
    display_limit = 20
    for entry in per_test_case_results[:display_limit]:
        label = _format_test_case_label(entry)
        rate_str = f"{entry['passed']}/{entry['total']} ({entry['rate']:.2%})"
        if entry["rate"] < pass_threshold:
            report_lines.append(f"    FAIL {label}: {rate_str}")
        else:
            report_lines.append(f"    PASS {label}: {rate_str}")

    if len(per_test_case_results) > display_limit:
        remaining = len(per_test_case_results) - display_limit
        report_lines.append(f"    ... and {remaining} more test cases")

    report_lines.append("")
    report_lines.append(f"  Failed test cases: {failed_count} of {total_count}")

    # Show first failure detail for each failing test case (up to 5)
    failed_entries = [e for e in per_test_case_results if e["rate"] < pass_threshold]
    if failed_entries:
        report_lines.append("")
        report_lines.append("  Failure details:")
        for entry in failed_entries[:5]:
            tc_idx = entry["index"]
            label = _format_test_case_label(entry)
            # Find the first failing sample for this test case
            for s_idx in range(samples):
                result_idx = s_idx * num_test_cases + tc_idx
                tcr = all_results[result_idx]
                exc = all_exceptions[result_idx] if result_idx < len(all_exceptions) else None
                if not _check_sample_passed(tcr.check_results):
                    report_lines.append(f"    {label} (first failure in sample {s_idx}):")
                    if exc:
                        report_lines.append(f"      Exception: {type(exc).__name__}: {exc}")
                    for check_result in tcr.check_results:
                        if check_result.status != 'completed':
                            report_lines.append(f"      Check '{check_result.check_type}': {check_result.status}")  # noqa: E501
                            if check_result.error:
                                report_lines.append(f"        Error: {check_result.error.message}")
                            if check_result.resolved_arguments:
                                report_lines.append(f"        Resolved arguments: {check_result.resolved_arguments}")  # noqa: E501
                        elif not check_result.results.get('passed', False):
                            report_lines.append(f"      Check '{check_result.check_type}': failed")
                            if 'reasoning' in check_result.results:
                                report_lines.append(f"        Reasoning: {check_result.results['reasoning']}")  # noqa: E501
                            if check_result.resolved_arguments:
                                report_lines.append(f"        Resolved arguments: {check_result.resolved_arguments}")  # noqa: E501
                    break  # Only show first failure per test case

        if len(failed_entries) > 5:
            report_lines.append(f"    ... and {len(failed_entries) - 5} more failed test cases")

    pytest.fail("\n".join(report_lines), pytrace=False)


def _generate_sample_failure_report(
    func_name: str,
    samples: int,
    passed_count: int,
    sample_pass_rate: float,
    pass_threshold: float,
    failed_samples: list[dict],
) -> None:
    """Generate sample-mode failure report and call pytest.fail()."""
    failed_count = samples - passed_count

    report_lines = [
        f"Statistical evaluation failed for {func_name}:",
        "  Pass mode: sample",
        f"  Total samples: {samples}",
        f"  Passed: {passed_count}",
        f"  Failed: {failed_count}",
        f"  Sample pass rate: {sample_pass_rate:.2%}",
        f"  Pass threshold: {pass_threshold:.2%}",
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

    pytest.fail("\n".join(report_lines), pytrace=False)
