"""Result schema implementations for FEP."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from flex_evals.schemas.check import CheckResult
from flex_evals.schemas.experiments import ExperimentMetadata
from flex_evals.schemas.test_case import TestCase
from flex_evals.schemas.output import Output

from ..constants import Status


@dataclass
class ExecutionContext:
    """Complete execution context - the test case and output that were evaluated together."""

    test_case: TestCase
    output: Output

    def __post_init__(self):
        """Validate execution context."""
        if not isinstance(self.test_case, TestCase):
            raise ValueError("ExecutionContext.test_case must be a TestCase instance")
        if not isinstance(self.output, Output):
            raise ValueError("ExecutionContext.output must be an Output instance")


@dataclass
class TestCaseSummary:
    """Aggregate statistics for test case check results."""

    total_checks: int
    completed_checks: int
    error_checks: int
    skipped_checks: int

    def __post_init__(self):
        """Validate summary statistics."""
        if self.total_checks < 0:
            raise ValueError("TestCaseSummary.total_checks must be non-negative")

        if (self.completed_checks + self.error_checks + self.skipped_checks) != self.total_checks:
            raise ValueError("TestCaseSummary check counts must sum to total_checks")


@dataclass
class TestCaseResult:
    """
    Aggregates the results of all checks run against a single test case.

    Provides both individual check results and summary statistics.

    Required Fields:
    - status: Computed overall status based on individual check statuses
    - execution_context: Complete execution context including test case and output
    - check_results: Array of individual check execution results
    - summary: Aggregate statistics for all check results

    Optional Fields:
    - metadata: Implementation-specific metadata

    Status Logic:
    - 'completed': All checks have status 'completed'
    - 'error': At least one check has status 'error'
    - 'skip': No errors, but at least one check has status 'skip'
    """

    status: Status | Literal['completed', 'error', 'skip']
    execution_context: ExecutionContext
    check_results: list[CheckResult]
    summary: TestCaseSummary
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate status matches check results and summary is accurate."""
        if not isinstance(self.execution_context, ExecutionContext):
            raise ValueError(
                "TestCaseResult.execution_context must be an ExecutionContext instance",
            )

        # Validate summary matches check_results
        actual_total = len(self.check_results)
        actual_completed = sum(1 for r in self.check_results if r.status == 'completed')
        actual_error = sum(1 for r in self.check_results if r.status == 'error')
        actual_skipped = sum(1 for r in self.check_results if r.status == 'skip')

        if (self.summary.total_checks != actual_total or
            self.summary.completed_checks != actual_completed or
            self.summary.error_checks != actual_error or
            self.summary.skipped_checks != actual_skipped):
            raise ValueError("TestCaseResult.summary does not match check_results")

        # Validate status matches check results
        expected_status = self._compute_status(self.check_results)
        if self.status != expected_status:
            raise ValueError(f"TestCaseResult.status should be '{expected_status}' based on check results")  # noqa: E501

    def _compute_status(self, check_results: list[CheckResult]) -> Status | Literal['completed', 'error', 'skip']:  # noqa: E501
        """Compute status based on check result statuses."""
        if any(r.status == 'error' for r in check_results):
            return 'error'
        if any(r.status == 'skip' for r in check_results):
            return 'skip'
        return 'completed'


@dataclass
class EvaluationSummary:
    """Aggregate execution statistics across all test cases in an evaluation run."""

    total_test_cases: int
    completed_test_cases: int
    error_test_cases: int
    skipped_test_cases: int

    def __post_init__(self):
        """Validate summary statistics."""
        if self.total_test_cases < 0:
            raise ValueError("EvaluationSummary.total_test_cases must be non-negative")

        if (self.completed_test_cases + self.error_test_cases + self.skipped_test_cases) != self.total_test_cases:  # noqa: E501
            raise ValueError("EvaluationSummary test case counts must sum to total_test_cases")


@dataclass
class EvaluationRunResult:
    """
    Provides comprehensive results for an entire evaluation run.

    Includes experiment context, summary statistics, and all individual test case results.

    Required Fields:
    - evaluation_id: Unique identifier for this evaluation run
    - started_at: When the evaluation started (ISO 8601 UTC)
    - completed_at: When the evaluation completed (ISO 8601 UTC)
    - status: Overall evaluation execution status
    - summary: Aggregate execution statistics
    - results: Individual test case results

    Optional Fields:
    - experiment: Experiment metadata
    - metadata: Implementation-specific metadata
    """

    evaluation_id: str
    started_at: datetime
    completed_at: datetime
    status: Status | Literal['completed', 'error', 'skip']
    summary: EvaluationSummary
    results: list[TestCaseResult]
    experiment: ExperimentMetadata | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate timestamps and summary accuracy."""
        if not self.evaluation_id:
            raise ValueError("EvaluationRunResult.evaluation_id must be non-empty")

        if self.completed_at < self.started_at:
            raise ValueError("EvaluationRunResult.completed_at must be >= started_at")

        # Validate summary matches results
        actual_total = len(self.results)
        actual_completed = sum(1 for r in self.results if r.status == 'completed')
        actual_error = sum(1 for r in self.results if r.status == 'error')
        actual_skipped = sum(1 for r in self.results if r.status == 'skip')

        if (self.summary.total_test_cases != actual_total or
            self.summary.completed_test_cases != actual_completed or
            self.summary.error_test_cases != actual_error or
            self.summary.skipped_test_cases != actual_skipped):
            raise ValueError("EvaluationRunResult.summary does not match results")

        # Validate status matches results
        expected_status = self._compute_status(self.results)
        if self.status != expected_status:
            raise ValueError(f"EvaluationRunResult.status should be '{expected_status}' based on test case results")  # noqa: E501

    def _compute_status(self, results: list[TestCaseResult]) -> Status | Literal['completed', 'error', 'skip']:  # noqa: E501
        """Compute status based on test case result statuses."""
        if any(r.status == 'error' for r in results):
            return 'error'
        if any(r.status == 'skip' for r in results):
            return 'skip'
        return 'completed'
