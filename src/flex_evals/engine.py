"""
Core evaluation engine for FEP.

This module implements the main evaluate() function that processes test cases,
outputs, and checks to produce evaluation results.
"""

import asyncio
import uuid
from datetime import datetime, UTC
from concurrent.futures import ProcessPoolExecutor

from .schemas import (
    TestCase, Output, Check, CheckResult, TestCaseResult, TestCaseSummary,
    EvaluationRunResult, EvaluationSummary, ExperimentMetadata,
)
from .schemas.results import ExecutionContext
from .schemas.check import CheckError, SchemaCheck
from .checks.base import BaseCheck, BaseAsyncCheck, EvaluationContext
from .registry import (
    get_check_class,
    is_async_check,
    get_registry_state,
    restore_registry_state,
    get_latest_version,
)
from .exceptions import ValidationError


def evaluate(
        test_cases: list[TestCase],
        outputs: list[Output],
        checks: list[Check | SchemaCheck] | list[list[Check | SchemaCheck]] | None = None,
        experiment_metadata: ExperimentMetadata | None = None,
        max_async_concurrent: int | None = None,
        max_parallel_workers: int = 1,
    ) -> EvaluationRunResult:
    """
    Execute checks against test cases and their corresponding outputs.

    This is the main evaluation function that processes test cases, outputs, and checks
    to produce comprehensive evaluation results according to the FEP protocol.

    Asynchronous checks are automatically detected and executed concurrently, while synchronous
    checks are executed directly without event loop overhead. The `max_async_concurrent`
    parameter allows limiting the number of concurrent async checks to prevent overwhelming any
    external systems or APIs.

    Args:
        test_cases: List of test cases to evaluate
        outputs: List of system outputs corresponding to test cases
        checks: Either:
            - List[Check | SchemaCheck]: Same checks applied to all test cases (shared pattern)
            - List[List[Check | SchemaCheck]]: checks[i] applies to test_cases[i] (per-test-case pattern)
            - None: Extract checks from TestCase.checks field (convenience pattern)
            Check | SchemaCheck can be either Check or SchemaCheck objects
        experiment_metadata: Optional experiment context information
        max_async_concurrent: Maximum number of concurrent async "check" executions (default: no limit)
        max_parallel_workers: Number of parallel worker processes (default: 1, no parallelization)

    Returns:
        Complete evaluation results with all test case results and summary statistics

    Raises:
        ValidationError: If inputs don't meet FEP protocol requirements

    Protocol Requirements:
        - test_cases.length == outputs.length
        - test_cases[i] is associated with outputs[i]
        - Automatic async/sync detection based on check types
    """  # noqa: E501
    started_at = datetime.now(UTC)
    evaluation_id = str(uuid.uuid4())

    # Validate input constraints according to FEP protocol
    _validate_inputs(test_cases, outputs, checks)

    # checks are either shared across all test cases or per-test-case
    resolved_checks = _resolve_checks(test_cases, checks)

    # Execute evaluation with optimal sync/async separation and parallelization
    if max_parallel_workers > 1:
        test_case_results = _evaluate_parallel(
            test_cases, outputs, resolved_checks, max_async_concurrent, max_parallel_workers,
        )
    else:
        work_items = list(zip(test_cases, outputs, resolved_checks))
        test_case_results = _evaluate(work_items, max_async_concurrent)

    completed_at = datetime.now(UTC)

    # Compute overall summary and status
    summary = _compute_evaluation_summary(test_case_results)
    status = _compute_evaluation_status(test_case_results)

    return EvaluationRunResult(
        evaluation_id=evaluation_id,
        started_at=started_at,
        completed_at=completed_at,
        status=status,
        summary=summary,
        results=test_case_results,
        experiment=experiment_metadata,
    )


def _convert_check_input(check_input: Check | SchemaCheck) -> Check:
    """Convert a Check | SchemaCheck (Check or SchemaCheck) to a Check object."""
    if isinstance(check_input, SchemaCheck):
        return check_input.to_check()
    return check_input


def _validate_inputs(
        test_cases: list[TestCase],
        outputs: list[Output],
        checks: list[Check | SchemaCheck] | list[list[Check | SchemaCheck]] | None,
    ) -> None:
    """Validate evaluation inputs according to FEP protocol."""
    if len(test_cases) != len(outputs):
        raise ValidationError(
            f"test_cases and outputs must have same length. "
            f"Got {len(test_cases)} test cases and {len(outputs)} outputs",
        )

    if len(test_cases) == 0:
        raise ValidationError("At least one test case is required for evaluation")

    # if checks is None, checks are expected to be defined in TestCase.checks
    if not checks:
        for test_case in test_cases:
            if test_case.checks is None:
                raise ValidationError(
                    "When checks is None, each TestCase must define its own checks",
                )
    # if checks is a list of lists, then the outer list must match test_cases length
    elif isinstance(checks, list) and len(checks) > 0 and isinstance(checks[0], list):
        if len(checks) != len(test_cases):
            raise ValidationError(
                f"When using per-test-case checks pattern, checks list must have same length as test_cases. "  # noqa: E501
                f"Got {len(checks)} check lists and {len(test_cases)} test cases",
            )
    # otherwise, checks must be a single list of Check | SchemaCheck objects
    else:  # noqa: PLR5501
        if not all(isinstance(check, Check | SchemaCheck) for check in checks):
            raise ValidationError(
                "When using shared checks pattern, checks must be a list of Check or SchemaCheck objects",  # noqa: E501
            )


def _resolve_checks(
        test_cases: list[TestCase],
        checks: list[Check | SchemaCheck] | list[list[Check | SchemaCheck]] | None,
    ) -> list[list[Check]]:
    """
    Resolve checks for each test case according to FEP patterns.

    Converts SchemaCheck objects to Check objects and returns List[List[Check]]
    where result[i] contains checks for test_cases[i].
    """
    if checks is None:
        # Extract from TestCase.checks (convenience pattern)
        resolved = []
        for test_case in test_cases:
            if test_case.checks is not None:
                # Convert any SchemaCheck objects to Check objects
                converted_checks = [_convert_check_input(check) for check in test_case.checks]
                resolved.append(converted_checks)
            else:
                resolved.append([])  # No checks for this test case
        return resolved

    if len(checks) > 0 and isinstance(checks[0], Check | SchemaCheck):
        # Shared checks pattern: same checks for all test cases
        shared_checks = [_convert_check_input(check) for check in checks]  # type: ignore
        return [shared_checks for _ in test_cases]

    if len(checks) > 0 and isinstance(checks[0], list):
        # Per-test-case checks pattern
        resolved = []
        for check_list in checks:  # type: ignore
            converted_checks = [_convert_check_input(check) for check in check_list]
            resolved.append(converted_checks)
        return resolved

    # Empty checks list
    return [[] for _ in test_cases]


def _flatten_checks_for_execution(
        work_items: list[tuple[TestCase, Output, list[Check]]],
    ) -> tuple[
        list[tuple[Check, EvaluationContext]],
        list[tuple[Check, EvaluationContext]],
        list[tuple[int, int, bool, int]],
    ]:
    """
    Flatten checks across all test cases and separate by type.

    Returns:
        - flattened_sync_checks: List of (check, context) for sync checks
        - flattened_async_checks: List of (check, context) for async checks
        - check_tracking: List of (test_idx, check_idx, is_async, flattened_idx)
    """
    flattened_sync_checks = []
    flattened_async_checks = []
    check_tracking = []  # Maps: (test_case_idx, check_idx, is_async, flattened_idx)

    for test_idx, (test_case, output, checks) in enumerate(work_items):
        context = EvaluationContext(test_case, output)

        for check_idx, check in enumerate(checks):
            try:
                if is_async_check(check.type, check.version):
                    flattened_async_checks.append((check, context))
                    check_tracking.append((test_idx, check_idx, True,
                                         len(flattened_async_checks) - 1))
                else:
                    flattened_sync_checks.append((check, context))
                    check_tracking.append((test_idx, check_idx, False,
                                         len(flattened_sync_checks) - 1))
            except ValueError:
                # Check type not registered - treat as sync
                flattened_sync_checks.append((check, context))
                check_tracking.append((test_idx, check_idx, False,
                                     len(flattened_sync_checks) - 1))

    return flattened_sync_checks, flattened_async_checks, check_tracking


def _unflatten_check_results(
        work_items: list[tuple[TestCase, Output, list[Check]]],
        sync_results: list[CheckResult],
        async_results: list[CheckResult],
        check_tracking: list[tuple[int, int, bool, int]],
    ) -> list[TestCaseResult]:
    """
    Reconstruct test case results from flattened check results.

    Args:
        work_items: Original work items with test cases and outputs
        sync_results: Results from sync check execution
        async_results: Results from async check execution
        check_tracking: Tracking info to map results back to test cases

    Returns:
        List of TestCaseResult objects in original order
    """
    results = []
    for test_idx, (test_case, output, _) in enumerate(work_items):
        # Collect check results for this test case in original order
        test_check_results = []
        for t_idx, check_idx, is_async, flattened_idx in check_tracking:
            if t_idx == test_idx:
                if is_async:
                    test_check_results.append((check_idx, async_results[flattened_idx]))
                else:
                    test_check_results.append((check_idx, sync_results[flattened_idx]))

        # Sort by original check index to maintain order
        test_check_results.sort(key=lambda x: x[0])
        check_results = [result for _, result in test_check_results]

        # Create test case result
        context = EvaluationContext(test_case, output)
        test_case_result = _create_test_case_result(check_results, context)
        results.append(test_case_result)

    return results


def _evaluate(
        work_items: list[tuple[TestCase, Output, list[Check]]],
        max_async_concurrent: int | None = None,
    ) -> list[TestCaseResult]:
    """Execute evaluation with optimal sync/async separation across all test cases."""
    # Flatten all checks across all test cases
    flattened_sync_checks, flattened_async_checks, check_tracking = (
        _flatten_checks_for_execution(work_items)
    )

    # Execute ALL sync checks at once (no event loop overhead)
    sync_results = [_execute_sync_check(check, context)
                    for check, context in flattened_sync_checks]

    # Execute ALL async checks concurrently in a single event loop
    if flattened_async_checks:
        async_results = asyncio.run(
            _execute_all_async_checks(flattened_async_checks, max_async_concurrent),
        )
    else:
        async_results = []

    # Reconstruct results by test case
    return _unflatten_check_results(work_items, sync_results, async_results, check_tracking)


def _evaluate_parallel(
        test_cases: list[TestCase],
        outputs: list[Output],
        resolved_checks: list[list[Check]],
        max_async_concurrent: int | None = None,
        max_parallel_workers: int = 2,
    ) -> list[TestCaseResult]:
    """Execute evaluation with parallel processing across test cases."""
    # Get current registry state to pass to workers; registered checks are not available in
    # worker processes, so we need to restore the registry state in each worker.
    registry_state = get_registry_state()

    # Prepare data for multiprocessing - group test cases with their data
    work_items = list(zip(test_cases, outputs, resolved_checks))

    # Split work items across workers
    chunk_size = max(1, len(work_items) // max_parallel_workers)
    work_chunks = [
        work_items[i:i + chunk_size]
        for i in range(0, len(work_items), chunk_size)
    ]

    # Execute in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_parallel_workers) as executor:
        futures = []
        for chunk in work_chunks:
            future = executor.submit(
                _evaluate_with_registry, chunk, max_async_concurrent, registry_state,
            )
            futures.append(future)

        # Collect results maintaining order
        all_results = []
        for future in futures:
            chunk_results = future.result()
            all_results.extend(chunk_results)

    return all_results


def _evaluate_with_registry(
        work_items: list[tuple[TestCase, Output, list[Check]]],
        max_async_concurrent: int | None = None,
        registry_state: dict | None = None,
    ) -> list[TestCaseResult]:
    """Evaluate work items in a separate process with registry restoration."""
    # Restore registry state in the worker process
    if registry_state:
        restore_registry_state(registry_state)

    return _evaluate(work_items, max_async_concurrent)




def _execute_sync_check(check: Check, context: EvaluationContext) -> CheckResult:
    """Execute a single synchronous check and return the result."""
    try:
        # Resolve version - if None, get latest version
        resolved_version = check.version
        if resolved_version is None:
            resolved_version = get_latest_version(check.type)

        check_class = get_check_class(check.type, resolved_version)
        check_instance = check_class()

        # Ensure this is a sync check
        if not isinstance(check_instance, BaseCheck):
            raise ValidationError(
                f"Check type '{check.type}' is async but was categorized as sync",
            )

        # Execute the check (version is determined automatically from registry)
        return check_instance.execute(
            check_type=check.type,
            arguments=check.arguments,
            context=context,
            check_metadata=check.metadata,
        )

    except Exception as e:
        # Create error result for unhandled exceptions
        return _create_error_check_result(check, str(e))




async def _execute_all_async_checks(
        async_check_contexts: list[tuple[Check, EvaluationContext]],
        max_async_concurrent: int | None = None,
    ) -> list[CheckResult]:
    """Execute ALL async checks across all test cases concurrently."""
    # Create semaphore if concurrency limit is specified
    semaphore = asyncio.Semaphore(max_async_concurrent) if max_async_concurrent else None

    tasks = []
    for check, context in async_check_contexts:
        task = asyncio.create_task(_execute_async_check(check, context, semaphore))
        tasks.append(task)

    return await asyncio.gather(*tasks)




async def _execute_async_check(
        check: Check,
        context: EvaluationContext,
        semaphore: asyncio.Semaphore | None = None,
    ) -> CheckResult:
    """Execute a single asynchronous check and return the result."""
    async def _run_check() -> CheckResult:
        try:
            # Resolve version - if None, get latest version
            resolved_version = check.version
            if resolved_version is None:
                resolved_version = get_latest_version(check.type)

            check_class = get_check_class(check.type, resolved_version)
            check_instance = check_class()

            # Ensure this is an async check
            if not isinstance(check_instance, BaseAsyncCheck):
                raise ValidationError(
                    f"Check type '{check.type}' is sync but was categorized as async",
                )

            # Execute the check (version is determined automatically from registry)
            return await check_instance.execute(
                check_type=check.type,
                arguments=check.arguments,
                context=context,
                check_metadata=check.metadata,
            )

        except Exception as e:
            # Create error result for unhandled exceptions
            return _create_error_check_result(check, str(e))

    # Use semaphore if provided, otherwise run directly
    if semaphore:
        async with semaphore:
            return await _run_check()
    else:
        return await _run_check()




def _create_error_check_result(
        check: Check,
        error_message: str,
    ) -> CheckResult:
    """Create a CheckResult for unhandled errors during check execution."""
    # Resolve version for error result
    resolved_version = check.version
    if resolved_version is None:
        try:
            resolved_version = get_latest_version(check.type)
        except ValueError:
            # If we can't resolve version, use "unknown"
            resolved_version = "unknown"

    return CheckResult(
        check_type=check.type,
        check_version=resolved_version,
        status='error',
        results={},
        resolved_arguments={},
        evaluated_at=datetime.now(UTC),
        metadata=check.metadata,
        error=CheckError(
            type='unknown_error',
            message=f"Unhandled error during check execution: {error_message}",
            recoverable=False,
        ),
    )


def _create_test_case_result(
        check_results: list[CheckResult],
        execution_context: EvaluationContext,
    ) -> TestCaseResult:
    """Create a TestCaseResult from check results."""
    # Create ExecutionContext for the result
    exec_context = ExecutionContext(
        test_case=execution_context.test_case,
        output=execution_context.output,
    )

    # Compute summary statistics
    total_checks = len(check_results)
    completed_checks = sum(1 for r in check_results if r.status == 'completed')
    error_checks = sum(1 for r in check_results if r.status == 'error')
    skipped_checks = sum(1 for r in check_results if r.status == 'skip')

    summary = TestCaseSummary(
        total_checks=total_checks,
        completed_checks=completed_checks,
        error_checks=error_checks,
        skipped_checks=skipped_checks,
    )

    # Compute overall status
    if error_checks > 0:
        status = 'error'
    elif skipped_checks > 0:
        status = 'skip'
    else:
        status = 'completed'

    return TestCaseResult(
        status=status,
        execution_context=exec_context,
        check_results=check_results,
        summary=summary,
    )


def _compute_evaluation_summary(test_case_results: list[TestCaseResult]) -> EvaluationSummary:
    """Compute aggregate evaluation summary from test case results."""
    total_test_cases = len(test_case_results)
    completed_test_cases = sum(1 for r in test_case_results if r.status == 'completed')
    error_test_cases = sum(1 for r in test_case_results if r.status == 'error')
    skipped_test_cases = sum(1 for r in test_case_results if r.status == 'skip')

    return EvaluationSummary(
        total_test_cases=total_test_cases,
        completed_test_cases=completed_test_cases,
        error_test_cases=error_test_cases,
        skipped_test_cases=skipped_test_cases,
    )


def _compute_evaluation_status(test_case_results: list[TestCaseResult]) -> str:
    """Compute overall evaluation status from test case results."""
    if any(r.status == 'error' for r in test_case_results):
        return 'error'
    if any(r.status == 'skip' for r in test_case_results):
        return 'skip'
    return 'completed'
