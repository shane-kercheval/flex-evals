"""
Pytest decorator for statistical evaluation using flex-evals.

This module provides the @evaluate decorator that allows running test functions
multiple times and validating success rates using existing flex-evals checks.
"""

import asyncio
import inspect
import traceback
from typing import Any, TypeVar
from collections.abc import Callable

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

    EXECUTION FLOW:
    ==============

    1. Execute the wrapped function `samples` number of times, passing test_case as first parameter
    2. Collect all return values and any exceptions
    3. Reuse original TestCase objects (cycling through if multiple)
    4. Wrap function outputs in Output objects
    5. Call engine.evaluate() with test cases, outputs, and checks
    6. Extract 'passed' field from CheckResult objects
    7. Calculate success rate and compare against threshold
    8. Use pytest.fail() with detailed reporting if threshold not met

    CHECK SOURCES:
    =============

    Checks can be provided in two ways:
    1. Via the checks parameter (shared across all test cases)
    2. Via TestCase.checks field (per-test-case checks)

    If checks parameter is provided, it overrides any checks defined in TestCase objects.
    If checks=None, each TestCase must define its own checks.

    TEST CASE REUSE:
    ===============

    The decorator reuses the original TestCase objects for each sample:
    - Preserves original input structure (JSONPath expressions work correctly)
    - Preserves expected values for comparison checks
    - Preserves check definitions if using TestCase.checks pattern
    - No side effects since engine.evaluate() only reads TestCase data

    If the original test_cases list has multiple items, the decorator cycles
    through them to provide variety in the evaluation.

    OUTPUT WRAPPING:
    ===============

    Function return values are wrapped in Output objects:
    - Value: The actual function return value
    - Simple structure preserving the original data

    For exceptions, error outputs are created with exception details including
    error type, message, and traceback information.

    SUCCESS RATE CALCULATION:
    ========================

    The decorator extracts the 'passed' field from each CheckResult:
    - Counts total samples executed
    - Counts samples where all checks passed
    - Calculates success_rate = passed_samples / total_samples
    - Compares against success_threshold

    PYTEST INTEGRATION:
    ==================

    - Uses pytest.fail() for threshold failures with detailed message
    - Provides rich reporting showing:
      * Total samples executed
      * Number passed/failed
      * Success rate achieved vs required
      * Details of failed samples
      * Exception details if any occurred

    Args:
        test_cases: List of TestCase objects to cycle through for evaluation
        checks: Optional list of checks to apply to all samples
        samples: Number of times to execute the function (default: 1)
        success_threshold: Minimum success rate required (0.0 to 1.0, default: 1.0)

    Returns:
        Decorated function that performs statistical evaluation

    Raises:
        ValueError: If checks configuration is invalid
        pytest.fail: If success threshold is not met

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

        # Get original function signature
        sig = inspect.signature(func)
        expects_test_case = 'test_case' in sig.parameters

        # Create wrapper signature without test_case parameter for pytest
        if expects_test_case:
            wrapper_params = [p for name, p in sig.parameters.items() if name != 'test_case']
            wrapper_sig = sig.replace(parameters=wrapper_params)
        else:
            wrapper_sig = sig

        # Create execution functions
        def _execute_sync_expansion():  # noqa: ANN202
            # Expand test cases: [A, B] * samples = [A,B,A,B,...]
            expanded_test_cases = []
            for _ in range(samples):
                expanded_test_cases.extend(test_cases)
            return expanded_test_cases

        async def _execute_async_calls(expanded_test_cases, kwargs):  # noqa
            # Generate outputs concurrently
            tasks = []
            for test_case in expanded_test_cases:
                if expects_test_case:  # noqa: SIM108
                    # Call function with test_case as first argument
                    task = func(test_case, **kwargs)
                else:
                    task = func(**kwargs)
                tasks.append(task)

            # Execute all tasks concurrently
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                pytest.fail(f"Function execution failed: {e}")

            # Process results and exceptions
            outputs = []
            exceptions = []
            for result in results:
                if isinstance(result, Exception):
                    # Create error output for exception
                    error_output = Output(
                        value={
                            "error": True,
                            "exception_type": type(result).__name__,
                            "exception_message": str(result),
                            "traceback": traceback.format_exception(type(result), result, result.__traceback__),  # noqa: E501
                        },
                    )
                    outputs.append(error_output)
                    exceptions.append(result)
                else:
                    outputs.append(Output(value=result))
                    exceptions.append(None)

            return outputs, exceptions

        def _execute_sync_calls(expanded_test_cases, kwargs):  # noqa
            # Generate outputs synchronously
            outputs = []
            exceptions = []
            for test_case in expanded_test_cases:
                try:
                    if expects_test_case:  # noqa: SIM108
                        # Call function with test_case as first argument
                        result = func(test_case, **kwargs)
                    else:
                        result = func(**kwargs)
                    outputs.append(Output(value=result))
                    exceptions.append(None)
                except Exception as e:
                    # Create error output for exception
                    error_output = Output(
                        value={
                            "error": True,
                            "exception_type": type(e).__name__,
                            "exception_message": str(e),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    outputs.append(error_output)
                    exceptions.append(e)

            return outputs, exceptions

        # Create wrapper function dynamically with correct signature
        # We need to create a function that actually has the signature pytest expects

        # Get parameter names for the wrapper (excluding test_case)
        param_names = list(wrapper_sig.parameters.keys())

        if asyncio.iscoroutinefunction(func):
            # Create async wrapper with proper signature
            async def async_wrapper_impl(kwargs_dict) -> None:  # noqa: ANN001
                expanded_test_cases = _execute_sync_expansion()
                outputs, exceptions = await _execute_async_calls(expanded_test_cases, kwargs_dict)

                # Process evaluation results
                _process_evaluation_results(
                    expanded_test_cases, outputs, exceptions, checks, samples, success_threshold, func.__name__,  # noqa: E501
                )

            # Create dynamic wrapper function with correct parameter names
            if param_names:
                # Create function string with actual parameter names
                param_str = ', '.join(param_names)
                kwargs_str = ', '.join(f"'{name}': {name}" for name in param_names)
                func_code = f"""
def wrapper({param_str}) -> None:
    import asyncio
    kwargs_dict = {{{kwargs_str}}}
    asyncio.run(async_wrapper_impl(kwargs_dict))
"""
            else:
                func_code = """
def wrapper() -> None:
    import asyncio
    asyncio.run(async_wrapper_impl({}))
"""

            # Execute the dynamic function code
            namespace = {'async_wrapper_impl': async_wrapper_impl, 'asyncio': asyncio}
            exec(func_code, namespace, namespace)
            wrapper = namespace['wrapper']
        else:
            # Sync wrapper
            def sync_wrapper_impl(kwargs_dict) -> None:  # noqa
                expanded_test_cases = _execute_sync_expansion()
                outputs, exceptions = _execute_sync_calls(expanded_test_cases, kwargs_dict)

                # Process evaluation results
                _process_evaluation_results(
                    expanded_test_cases, outputs, exceptions, checks, samples, success_threshold, func.__name__,  # noqa: E501
                )

            # Create dynamic wrapper function with correct parameter names
            if param_names:
                # Create function string with actual parameter names
                param_str = ', '.join(param_names)
                kwargs_str = ', '.join(f"'{name}': {name}" for name in param_names)
                func_code = f"""
def wrapper({param_str}) -> None:
    kwargs_dict = {{{kwargs_str}}}
    sync_wrapper_impl(kwargs_dict)
"""
            else:
                func_code = """
def wrapper() -> None:
    sync_wrapper_impl({})
"""

            # Execute the dynamic function code
            namespace = {'sync_wrapper_impl': sync_wrapper_impl}
            exec(func_code, namespace, namespace)
            wrapper = namespace['wrapper']

        # The wrapper function now has the correct signature naturally

        return wrapper
    return decorator


def _process_evaluation_results(
    expanded_test_cases: list[TestCase],
    outputs: list[Output],
    exceptions: list[Exception | None],
    checks: list[Check] | None,
    samples: int,
    success_threshold: float,
    func_name: str,
) -> None:
    """Process evaluation results and handle sample-based success calculation."""
    # Call evaluate once with all expanded test cases
    try:
        evaluation_result = engine_evaluate(
            test_cases=expanded_test_cases,
            outputs=outputs,
            checks=checks,
        )
    except Exception as e:
        pytest.fail(f"Evaluation failed: {e}")

    # Group results back into samples
    # Each sample contains len(original_test_cases) consecutive results
    num_test_cases_per_sample = len(expanded_test_cases) // samples
    sample_results = []

    for sample_idx in range(samples):
        start_idx = sample_idx * num_test_cases_per_sample
        end_idx = start_idx + num_test_cases_per_sample
        sample_test_case_results = evaluation_result.results[start_idx:end_idx]
        sample_exceptions = exceptions[start_idx:end_idx]

        # A sample passes if ALL its test cases pass
        sample_passed = all(
            _check_sample_passed(tcr.check_results) for tcr in sample_test_case_results
        )

        if not sample_passed:
            sample_results.append({
                "sample_index": sample_idx,
                "exceptions": sample_exceptions,
                "test_case_results": sample_test_case_results,
            })
        else:
            sample_results.append(None)  # Sample passed

    # Calculate success rate based on samples, not individual test cases
    passed_samples = sum(1 for result in sample_results if result is None)
    success_rate = passed_samples / samples

    # Check if threshold is met
    if success_rate < success_threshold:
        failed_samples = [result for result in sample_results if result is not None]
        _generate_failure_report(
            func_name=func_name,
            samples=samples,
            passed_count=passed_samples,
            success_rate=success_rate,
            success_threshold=success_threshold,
            failed_samples=failed_samples,
        )


def _validate_check_configuration(test_cases: list[TestCase], checks: list[Check] | None) -> None:
    """
    Validate that check configuration is valid for evaluation.

    Args:
        test_cases: List of test cases
        checks: Optional shared checks

    Raises:
        ValueError: If configuration is invalid
    """
    if checks is None:
        # Each test case must have checks defined
        for test_case in test_cases:
            if not test_case.checks:
                raise ValueError(
                    f"When checks=None, each TestCase must define its own checks. "
                    f"TestCase '{test_case.id}' has no checks defined.",
                )


def _check_sample_passed(check_results: list[CheckResult]) -> bool:
    """
    Check if a sample passed by examining all check results.

    A sample passes if all checks have a 'passed' field that is True.

    Args:
        check_results: List of check results for a sample

    Returns:
        bool: True if sample passed, False otherwise

    Raises:
        ValueError: If any check result doesn't have a 'passed' field
    """
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
    """
    Generate detailed failure report and call pytest.fail().

    Args:
        func_name: Name of the test function
        samples: Total number of samples
        passed_count: Number of samples that passed
        success_rate: Calculated success rate
        success_threshold: Required success threshold
        failed_samples: List of failed sample details
    """
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

        # Report exceptions for this sample
        for i, exception in enumerate(exceptions):
            if exception:
                report_lines.append(f"    Test case {i} exception: {type(exception).__name__}: {exception}")  # noqa: E501

        # Report check failures for this sample
        for i, test_case_result in enumerate(test_case_results):
            for check_result in test_case_result.check_results:
                if check_result.status != 'completed':
                    report_lines.append(f"    Test case {i} check '{check_result.check_type}': {check_result.status}")  # noqa: E501
                    if check_result.error:
                        report_lines.append(f"      Error: {check_result.error.message}")
                elif not check_result.results.get('passed', False):
                    report_lines.append(f"    Test case {i} check '{check_result.check_type}': failed")  # noqa: E501
                    # Add additional details if available
                    if 'reasoning' in check_result.results:
                        report_lines.append(f"      Reasoning: {check_result.results['reasoning']}")  # noqa: E501

    if len(failed_samples) > 5:
        report_lines.append(f"  ... and {len(failed_samples) - 5} more failures")

    pytest.fail("\n".join(report_lines))
