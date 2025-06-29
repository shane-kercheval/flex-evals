"""
Core evaluation engine for FEP.

This module implements the main evaluate() function that processes test cases,
outputs, and checks to produce evaluation results.
"""

import asyncio
import uuid
from datetime import datetime, UTC

from .schemas import (
    TestCase, Output, Check, CheckResult, TestCaseResult, TestCaseSummary,
    EvaluationRunResult, EvaluationSummary, ExperimentMetadata,
)
from .schemas.check import CheckError, CheckResultMetadata
from .checks.base import BaseCheck, BaseAsyncCheck, EvaluationContext
from .registry import get_check_class, is_async_check
from .exceptions import ValidationError


def evaluate(
    test_cases: list[TestCase],
    outputs: list[Output],
    checks: list[Check] | list[list[Check]] | None = None,
    experiment_metadata: ExperimentMetadata | None = None,
) -> EvaluationRunResult:
    """
    Execute checks against test cases and their corresponding outputs.

    This is the main evaluation function that processes test cases, outputs, and checks
    to produce comprehensive evaluation results according to the FEP protocol.

    Args:
        test_cases: List of test cases to evaluate
        outputs: List of system outputs corresponding to test cases
        checks: Either:
            - List[Check]: Same checks applied to all test cases (shared pattern)
            - List[List[Check]]: checks[i] applies to test_cases[i] (per-test-case pattern)
            - None: Extract checks from TestCase.checks field (convenience pattern)
        experiment_metadata: Optional experiment context information

    Returns:
        Complete evaluation results with all test case results and summary statistics

    Raises:
        ValidationError: If inputs don't meet FEP protocol requirements

    Protocol Requirements:
        - test_cases.length == outputs.length
        - test_cases[i] is associated with outputs[i]
        - Automatic async/sync detection based on check types
    """
    started_at = datetime.now(UTC)
    evaluation_id = str(uuid.uuid4())

    # Validate input constraints
    _validate_inputs(test_cases, outputs, checks)

    # Resolve checks for each test case
    resolved_checks = _resolve_checks(test_cases, checks)

    # Execute evaluation with optimal sync/async separation
    test_case_results = _evaluate(test_cases, outputs, resolved_checks)

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


def _validate_inputs(
    test_cases: list[TestCase],
    outputs: list[Output],
    checks: list[Check] | list[list[Check]] | None,
) -> None:
    """Validate evaluation inputs according to FEP protocol."""
    if len(test_cases) != len(outputs):
        raise ValidationError(
            f"test_cases and outputs must have same length. "
            f"Got {len(test_cases)} test cases and {len(outputs)} outputs",
        )

    if len(test_cases) == 0:
        raise ValidationError("At least one test case is required for evaluation")

    # Validate per-test-case checks pattern
    if isinstance(checks, list) and len(checks) > 0 and isinstance(checks[0], list):  # noqa: SIM102
        if len(checks) != len(test_cases):
            raise ValidationError(
                f"When using per-test-case checks pattern, checks list must have same length as test_cases. "  # noqa: E501
                f"Got {len(checks)} check lists and {len(test_cases)} test cases",
            )


def _resolve_checks(
    test_cases: list[TestCase],
    checks: list[Check] | list[list[Check]] | None,
) -> list[list[Check]]:
    """
    Resolve checks for each test case according to FEP patterns.

    Returns List[List[Check]] where result[i] contains checks for test_cases[i].
    """
    if checks is None:
        # Extract from TestCase.checks (convenience pattern)
        resolved = []
        for test_case in test_cases:
            if test_case.checks is not None:
                resolved.append(test_case.checks)
            else:
                resolved.append([])  # No checks for this test case
        return resolved

    if len(checks) > 0 and isinstance(checks[0], Check):
        # Shared checks pattern: same checks for all test cases
        shared_checks = checks  # type: ignore
        return [shared_checks for _ in test_cases]

    if len(checks) > 0 and isinstance(checks[0], list):
        # Per-test-case checks pattern
        return checks  # type: ignore

    # Empty checks list
    return [[] for _ in test_cases]


def _evaluate(
    test_cases: list[TestCase],
    outputs: list[Output],
    resolved_checks: list[list[Check]],
) -> list[TestCaseResult]:
    """Execute evaluation with optimal sync/async separation."""
    results = []

    for (test_case, output, checks) in zip(test_cases, outputs, resolved_checks):
        context = EvaluationContext(test_case, output)

        # Separate checks by type and track original order
        sync_checks, async_checks, order_map = _separate_checks_by_type(checks)

        # Execute sync checks directly (no event loop overhead)
        sync_results = [_execute_sync_check(check, context) for check in sync_checks]

        # Execute async checks concurrently (only if needed)
        if async_checks:
            async_results = asyncio.run(_execute_async_checks_concurrent(async_checks, context))
        else:
            async_results = []

        # Reconstruct results in original order
        check_results = _reconstruct_check_order(sync_results, async_results, order_map)

        # Create test case result
        test_case_result = _create_test_case_result(test_case.id, check_results)
        results.append(test_case_result)

    return results


def _separate_checks_by_type(
    checks: list[Check],
) -> tuple[list[Check], list[Check], list[tuple[str, int]]]:
    """
    Separate checks into sync and async lists, tracking original order.

    Returns:
        sync_checks: List of synchronous checks
        async_checks: List of asynchronous checks
        order_map: List of (type, index) tuples to reconstruct original order
    """
    sync_checks = []
    async_checks = []
    order_map = []

    for check in checks:
        try:
            if is_async_check(check.type):
                async_checks.append(check)
                order_map.append(('async', len(async_checks) - 1))
            else:
                sync_checks.append(check)
                order_map.append(('sync', len(sync_checks) - 1))
        except ValueError:
            # Check type not registered - treat as sync and handle error during execution
            sync_checks.append(check)
            order_map.append(('sync', len(sync_checks) - 1))

    return sync_checks, async_checks, order_map


def _execute_sync_check(check: Check, context: EvaluationContext) -> CheckResult:
    """Execute a single synchronous check and return the result."""
    try:
        check_class = get_check_class(check.type)
        check_instance = check_class()

        # Ensure this is a sync check
        if not isinstance(check_instance, BaseCheck):
            raise ValidationError(
                f"Check type '{check.type}' is async but was categorized as sync",
            )

        # Execute the check
        return check_instance.execute(
            check_type=check.type,
            arguments=check.arguments,
            context=context,
            check_version=check.version,
        )

    except Exception as e:
        # Create error result for unhandled exceptions
        return _create_error_check_result(check, context, str(e))


async def _execute_async_checks_concurrent(
    async_checks: list[Check], context: EvaluationContext,
) -> list[CheckResult]:
    """Execute multiple async checks concurrently."""
    tasks = []
    for check in async_checks:
        task = asyncio.create_task(_execute_async_check(check, context))
        tasks.append(task)

    return await asyncio.gather(*tasks)


async def _execute_async_check(check: Check, context: EvaluationContext) -> CheckResult:
    """Execute a single asynchronous check and return the result."""
    try:
        check_class = get_check_class(check.type)
        check_instance = check_class()

        # Ensure this is an async check
        if not isinstance(check_instance, BaseAsyncCheck):
            raise ValidationError(
                f"Check type '{check.type}' is sync but was categorized as async",
            )

        # Execute the check
        return await check_instance.execute(
            check_type=check.type,
            arguments=check.arguments,
            context=context,
            check_version=check.version,
        )

    except Exception as e:
        # Create error result for unhandled exceptions
        return _create_error_check_result(check, context, str(e))


def _reconstruct_check_order(
    sync_results: list[CheckResult],
    async_results: list[CheckResult],
    order_map: list[tuple[str, int]],
) -> list[CheckResult]:
    """Reconstruct check results in their original order."""
    final_results = []
    sync_idx = 0
    async_idx = 0

    for check_type, _ in order_map:
        if check_type == 'sync':
            final_results.append(sync_results[sync_idx])
            sync_idx += 1
        else:  # async
            final_results.append(async_results[async_idx])
            async_idx += 1

    return final_results


def _create_error_check_result(
        check: Check,
        context: EvaluationContext,
        error_message: str) -> CheckResult:
    """Create a CheckResult for unhandled errors during check execution."""
    return CheckResult(
        check_type=check.type,
        status='error',
        results={},
        resolved_arguments={},
        evaluated_at=datetime.now(UTC),
        metadata=CheckResultMetadata(
            test_case_id=context.test_case.id,
            test_case_metadata=context.test_case.metadata,
            output_metadata=context.output.metadata,
            check_version=check.version,
        ),
        error=CheckError(
            type='unknown_error',
            message=f"Unhandled error during check execution: {error_message}",
            recoverable=False,
        ),
    )


def _create_test_case_result(
        test_case_id: str,
        check_results: list[CheckResult],
    ) -> TestCaseResult:
    """Create a TestCaseResult from check results."""
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
        test_case_id=test_case_id,
        status=status,
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
