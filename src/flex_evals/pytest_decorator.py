"""
Pytest decorator for statistical evaluation using flex-evals.

This module provides the @evaluate decorator that allows running test functions
multiple times and validating success rates using existing flex-evals checks.
"""

import functools
import traceback
from typing import Any, TypeVar
from collections.abc import Callable

import pytest

from .engine import evaluate as engine_evaluate
from .schemas import TestCase, Output, Check, CheckResult

F = TypeVar('F', bound=Callable[..., Any])


def evaluate(
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

    1. Execute the wrapped function `samples` number of times
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
        def test_my_function():
            return {"result": "expected output"}

    Error Handling:
        - Function exceptions are caught and counted as failures
        - Invalid check configurations raise ValueError at decoration time
        - Missing 'passed' field in CheckResult raises clear error
        - Incompatible checks (no 'passed' field) are validated early
    """
    def decorator(func: F) -> F:
        # Validate decorator parameters
        if not test_cases:
            raise ValueError("test_cases list cannot be empty")

        if samples <= 0:
            raise ValueError("samples must be positive")

        if not 0.0 <= success_threshold <= 1.0:
            raise ValueError("success_threshold must be between 0.0 and 1.0")

        # Validate check configuration
        _validate_check_configuration(test_cases, checks)

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
            # Execute function multiple times and collect results
            outputs = []
            test_cases_for_evaluation = []
            exceptions = []

            for sample_idx in range(samples):
                try:
                    # Execute the function
                    result = func(*args, **kwargs)

                    # Create Output object
                    output = Output(value=result)
                    outputs.append(output)
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

                # Just reuse the original test case - no copying needed!
                base_test_case = test_cases[sample_idx % len(test_cases)]
                test_cases_for_evaluation.append(base_test_case)

            # Evaluate using flex-evals engine
            try:
                evaluation_result = engine_evaluate(
                    test_cases=test_cases_for_evaluation,
                    outputs=outputs,
                    checks=checks,
                )
            except Exception as e:
                pytest.fail(f"Evaluation failed: {e}")

            # Extract passed status from check results
            passed_count = 0
            failed_samples = []

            for i, test_case_result in enumerate(evaluation_result.results):
                sample_passed = _check_sample_passed(test_case_result.check_results)
                if sample_passed:
                    passed_count += 1
                else:
                    failed_samples.append({
                        "sample_index": i,
                        "exception": exceptions[i],
                        "check_results": test_case_result.check_results,
                    })

            # Calculate success rate
            success_rate = passed_count / samples

            # Check if threshold is met
            if success_rate < success_threshold:
                _generate_failure_report(
                    func_name=func.__name__,
                    samples=samples,
                    passed_count=passed_count,
                    success_rate=success_rate,
                    success_threshold=success_threshold,
                    failed_samples=failed_samples,
                )
            # Test passed - return None (pytest convention)
        return wrapper
    return decorator


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
        exception = failure["exception"]
        check_results = failure["check_results"]

        report_lines.append(f"  Sample {sample_idx}:")

        if exception:
            report_lines.append(f"    Exception: {type(exception).__name__}: {exception}")

        for check_result in check_results:
            if check_result.status != 'completed':
                report_lines.append(f"    Check '{check_result.check_type}': {check_result.status}")  # noqa: E501
                if check_result.error:
                    report_lines.append(f"      Error: {check_result.error.message}")
            elif not check_result.results.get('passed', False):
                report_lines.append(f"    Check '{check_result.check_type}': failed")
                # Add additional details if available
                if 'reasoning' in check_result.results:
                    report_lines.append(f"      Reasoning: {check_result.results['reasoning']}")

    if len(failed_samples) > 5:
        report_lines.append(f"  ... and {len(failed_samples) - 5} more failures")

    pytest.fail("\n".join(report_lines))
