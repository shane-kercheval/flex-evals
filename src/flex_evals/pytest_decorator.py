"""
Simplified pytest decorator for statistical evaluation using flex-evals.

This module provides the @evaluate decorator that allows running test functions
multiple times and validating success rates using existing flex-evals checks.
"""

import asyncio
import inspect
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

        def _create_error_output(exception):  # noqa: ANN001, ANN202
            """Create an error output for an exception."""
            return Output(
                value={
                    "error": True,
                    "exception_type": type(exception).__name__,
                    "exception_message": str(exception),
                    "traceback": traceback.format_exc() if not isinstance(exception, type) else
                                traceback.format_exception(type(exception), exception, exception.__traceback__),  # noqa: E501
                },
            )

        def _process_task_results(results):  # noqa: ANN001, ANN202
            """Process results from asyncio.gather."""
            outputs = []
            exceptions = []

            for result in results:
                if isinstance(result, Exception):
                    outputs.append(_create_error_output(result))
                    exceptions.append(result)
                else:
                    outputs.append(Output(value=result))
                    exceptions.append(None)

            return outputs, exceptions

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
            # Create tasks for all calls
            tasks = []
            for test_case in expanded_test_cases:
                if expects_test_case:
                    task = func(test_case, *args, **kwargs)
                else:
                    task = func(*args, **kwargs)
                tasks.append(task)

            # Execute concurrently and collect results
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                pytest.fail(f"Function execution failed: {e}")

            outputs, exceptions = _process_task_results(results)
            _evaluate_results(expanded_test_cases, outputs, exceptions)

        def _execute_sync_calls(expanded_test_cases, args, kwargs) -> None:  # noqa: ANN001
            """Execute sync function calls sequentially."""
            outputs = []
            exceptions = []

            for test_case in expanded_test_cases:
                try:
                    if expects_test_case:
                        result = func(test_case, *args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    outputs.append(Output(value=result))
                    exceptions.append(None)
                except Exception as e:
                    outputs.append(_create_error_output(e))
                    exceptions.append(e)

            _evaluate_results(expanded_test_cases, outputs, exceptions)

        async def _run_async_evaluation(args, kwargs) -> None:  # noqa: ANN001
            """Run evaluation for async functions."""
            expanded_test_cases = test_cases * samples
            await _execute_async_calls(expanded_test_cases, args, kwargs)

        def _run_sync_evaluation(args, kwargs) -> None:  # noqa: ANN001
            """Run evaluation for sync functions."""
            expanded_test_cases = test_cases * samples
            _execute_sync_calls(expanded_test_cases, args, kwargs)

        # Create wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            def async_wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
                # Run async function using asyncio.run
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
                report_lines.append(f"    Test case {i} exception: {type(exception).__name__}: {exception}")  # noqa: E501

        # Report check failures
        for i, test_case_result in enumerate(test_case_results):
            for check_result in test_case_result.check_results:
                if check_result.status != 'completed':
                    report_lines.append(f"    Test case {i} check '{check_result.check_type}': {check_result.status}")  # noqa: E501
                    if check_result.error:
                        report_lines.append(f"      Error: {check_result.error.message}")
                    if check_result.resolved_arguments:
                        report_lines.append(f"      Resolved arguments: {check_result.resolved_arguments}")  # noqa: E501
                elif not check_result.results.get('passed', False):
                    report_lines.append(f"    Test case {i} check '{check_result.check_type}': failed")  # noqa: E501
                    if 'reasoning' in check_result.results:
                        report_lines.append(f"      Reasoning: {check_result.results['reasoning']}")  # noqa: E501
                    if check_result.resolved_arguments:
                        report_lines.append(f"      Resolved arguments: {check_result.resolved_arguments}")  # noqa: E501

    if len(failed_samples) > 5:
        report_lines.append(f"  ... and {len(failed_samples) - 5} more failures")

    pytest.fail("\n".join(report_lines))
