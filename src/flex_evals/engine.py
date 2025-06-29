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

    # Detect if any checks are async
    has_async_checks = _detect_async_checks(resolved_checks)

    # Execute evaluation (async or sync based on check types)
    if has_async_checks:
        test_case_results = asyncio.run(
            _evaluate_async(test_cases, outputs, resolved_checks),
        )
    else:
        test_case_results = _evaluate_sync(test_cases, outputs, resolved_checks)

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


def _detect_async_checks(resolved_checks: list[list[Check]]) -> bool:
    """Detect if any checks require async execution."""
    for check_list in resolved_checks:
        for check in check_list:
            try:
                if is_async_check(check.type):
                    return True
            except ValueError:
                # Check type not registered - will be handled during execution
                continue
    return False


def _evaluate_sync(
    test_cases: list[TestCase],
    outputs: list[Output],
    resolved_checks: list[list[Check]],
) -> list[TestCaseResult]:
    """Execute evaluation synchronously."""
    results = []

    for i, (test_case, output, checks) in enumerate(zip(test_cases, outputs, resolved_checks)):
        context = EvaluationContext(test_case, output)
        check_results = []

        for check in checks:
            try:
                check_class = get_check_class(check.type)
                check_instance = check_class()

                # Ensure this is a sync check
                if not isinstance(check_instance, BaseCheck):
                    raise ValidationError(f"Check type '{check.type}' is async but sync execution was selected")  # noqa: E501

                # Execute the check
                result = check_instance.execute(
                    check_type=check.type,
                    arguments=check.arguments,
                    context=context,
                    check_version=check.version,
                )
                check_results.append(result)

            except Exception as e:
                # Create error result for unhandled exceptions
                error_result = _create_error_check_result(check, context, str(e))
                check_results.append(error_result)

        # Create test case result
        test_case_result = _create_test_case_result(test_case.id, check_results)
        results.append(test_case_result)

    return results


async def _evaluate_async(
    test_cases: list[TestCase],
    outputs: list[Output],
    resolved_checks: list[list[Check]],
) -> list[TestCaseResult]:
    """Execute evaluation asynchronously."""
    results = []

    for i, (test_case, output, checks) in enumerate(zip(test_cases, outputs, resolved_checks)):
        context = EvaluationContext(test_case, output)
        check_results = []

        for check in checks:
            try:
                check_class = get_check_class(check.type)
                check_instance = check_class()

                # Execute the check (async or sync)
                if isinstance(check_instance, BaseAsyncCheck):
                    result = await check_instance.execute(
                        check_type=check.type,
                        arguments=check.arguments,
                        context=context,
                        check_version=check.version,
                    )
                elif isinstance(check_instance, BaseCheck):
                    result = check_instance.execute(
                        check_type=check.type,
                        arguments=check.arguments,
                        context=context,
                        check_version=check.version,
                    )
                else:
                    raise ValidationError(f"Invalid check class type for '{check.type}'")

                check_results.append(result)

            except Exception as e:
                # Create error result for unhandled exceptions
                error_result = _create_error_check_result(check, context, str(e))
                check_results.append(error_result)

        # Create test case result
        test_case_result = _create_test_case_result(test_case.id, check_results)
        results.append(test_case_result)

    return results


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
