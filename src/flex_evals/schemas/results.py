"""Result schema implementations for FEP."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Literal

from flex_evals.schemas.check import CheckResult
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

    def __post_init__(self):
        """Validate summary statistics."""
        if self.total_checks < 0:
            raise ValueError("TestCaseSummary.total_checks must be non-negative")

        if (self.completed_checks + self.error_checks) != self.total_checks:
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
    """

    status: Status | Literal['completed', 'error']
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

        if (self.summary.total_checks != actual_total or
            self.summary.completed_checks != actual_completed or
            self.summary.error_checks != actual_error):
            raise ValueError("TestCaseResult.summary does not match check_results")

        # Validate status matches check results
        expected_status = self._compute_status(self.check_results)
        if self.status != expected_status:
            raise ValueError(f"TestCaseResult.status should be '{expected_status}' based on check results")  # noqa: E501

    def _compute_status(
        self, check_results: list[CheckResult],
    ) -> Status | Literal['completed', 'error']:
        """Compute status based on check result statuses."""
        if any(r.status == 'error' for r in check_results):
            return 'error'
        return 'completed'


@dataclass
class EvaluationSummary:
    """Aggregate execution statistics across all test cases in an evaluation run."""

    total_test_cases: int
    completed_test_cases: int
    error_test_cases: int

    def __post_init__(self):
        """Validate summary statistics."""
        if self.total_test_cases < 0:
            raise ValueError("EvaluationSummary.total_test_cases must be non-negative")

        if (self.completed_test_cases + self.error_test_cases) != self.total_test_cases:
            raise ValueError("EvaluationSummary test case counts must sum to total_test_cases")


@dataclass
class EvaluationRunResult:
    """
    Provides comprehensive results for an entire evaluation run.

    Includes summary statistics and all individual test case results.

    Required Fields:
    - evaluation_id: Unique identifier for this evaluation run
    - started_at: When the evaluation started (ISO 8601 UTC)
    - completed_at: When the evaluation completed (ISO 8601 UTC)
    - status: Overall evaluation execution status
    - summary: Aggregate execution statistics
    - results: Individual test case results

    Optional Fields:
    - metadata: Implementation-specific metadata
    """

    evaluation_id: str
    started_at: datetime
    completed_at: datetime
    status: Status | Literal['completed', 'error']
    summary: EvaluationSummary
    results: list[TestCaseResult]
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

        if (self.summary.total_test_cases != actual_total or
            self.summary.completed_test_cases != actual_completed or
            self.summary.error_test_cases != actual_error):
            raise ValueError("EvaluationRunResult.summary does not match results")

        # Validate status matches results
        expected_status = self._compute_status(self.results)
        if self.status != expected_status:
            raise ValueError(f"EvaluationRunResult.status should be '{expected_status}' based on test case results")  # noqa: E501

    def _compute_status(
        self, results: list[TestCaseResult],
    ) -> Status | Literal['completed', 'error']:
        """Compute status based on test case result statuses."""
        if any(r.status == 'error' for r in results):
            return 'error'
        return 'completed'

    def to_dict_list(self) -> list[dict[str, Any]]:
        """
        Flatten evaluation results into a list of dictionaries for tabular analysis.

        Each dictionary represents one check result with context from the evaluation run,
        test case, and check execution. This format is suitable for conversion to
        pandas DataFrame or other tabular analysis tools.

        Returns:
            List of dictionaries, one per check result, containing flattened data
            from the evaluation run, test case, and check execution.
        """
        flattened_rows = []

        for test_case_result in self.results:
            # Extract common data that applies to all checks in this test case
            test_case_data = {
                # Evaluation run context
                'evaluation_id': self.evaluation_id,
                'started_at': self.started_at,
                'completed_at': self.completed_at,
                'evaluation_status': self.status,

                # Test case context
                'test_case_id': test_case_result.execution_context.test_case.id,
                'test_case_status': test_case_result.status,

                # Test case input data
                'input_data': test_case_result.execution_context.test_case.input,
                'expected_output': test_case_result.execution_context.test_case.expected,

                # System output
                'actual_output': test_case_result.execution_context.output.value,

                # Test case summary stats
                'total_checks': test_case_result.summary.total_checks,
                'completed_checks': test_case_result.summary.completed_checks,
                'error_checks': test_case_result.summary.error_checks,
            }

            # Add test case metadata if present
            if test_case_result.execution_context.test_case.metadata:
                test_case_data['test_case_metadata'] = (
                    test_case_result.execution_context.test_case.metadata
                )

            # Add output metadata if present
            if test_case_result.execution_context.output.metadata:
                test_case_data['output_metadata'] = (
                    test_case_result.execution_context.output.metadata
                )

            # Add test case result metadata if present
            if test_case_result.metadata:
                test_case_data['test_case_result_metadata'] = test_case_result.metadata

            # Add evaluation metadata if present
            if self.metadata:
                test_case_data['evaluation_metadata'] = self.metadata

            # Create one row per check result
            for check_result in test_case_result.check_results:
                row = test_case_data.copy()

                # Add check-specific data
                row.update({
                    'check_type': check_result.check_type,
                    'check_version': check_result.check_version,
                    'check_status': check_result.status,
                    'check_results': check_result.results,
                    'resolved_arguments': check_result.resolved_arguments,
                    'evaluated_at': check_result.evaluated_at,
                })

                # Extract 'passed' field from check_results if it exists
                if (isinstance(check_result.results, dict) and
                    'passed' in check_result.results):
                    row['check_results_passed'] = check_result.results['passed']
                else:
                    row['check_results_passed'] = None

                # Add check metadata if present
                if check_result.metadata:
                    row['check_metadata'] = check_result.metadata

                # Add error details if present
                if check_result.error:
                    row['error_type'] = check_result.error.type
                    row['error_message'] = check_result.error.message

                flattened_rows.append(row)

        return flattened_rows

    def serialize(self) -> dict[str, Any]:
        """
        Serialize EvaluationRunResult to JSON-compatible dictionary.

        Converts datetime objects to ISO 8601 format strings and handles
        other non-JSON-serializable objects (functions, classes, etc.).

        This method ensures consistent serialization format when transmitting
        evaluation results over HTTP or storing them in external systems.

        Returns:
            JSON-compatible dictionary representation suitable for HTTP
            transmission or JSON serialization.

        Example:
            >>> result = evaluate(test_cases=test_cases, outputs=outputs)
            >>> serialized = result.serialize()
            >>> # Can now safely use json.dumps(serialized) or send via HTTP
        """

        class CustomEncoder(json.JSONEncoder):
            """Custom JSON encoder that handles datetime and non-serializable objects."""

            def default(self, obj: Any) -> Any:  # noqa: ANN401, PLR0911
                """Convert non-serializable objects to JSON-compatible format."""
                if isinstance(obj, datetime):
                    return obj.isoformat()

                # Handle classes/types FIRST (before checking for methods)
                # Classes have methods like model_dump as unbound methods, which would
                # pass hasattr/callable checks but fail when called without an instance
                if isinstance(obj, type):
                    return f"<class {obj.__name__}>"

                # Handle Pydantic models (v2 has model_dump, v1 has dict)
                if hasattr(obj, 'model_dump') and callable(getattr(obj, 'model_dump')):
                    return obj.model_dump()
                if hasattr(obj, 'dict') and callable(getattr(obj, 'dict')):
                    # Pydantic v1 compatibility
                    return obj.dict()

                # Handle objects with common dict conversion methods
                if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
                    return obj.to_dict()
                if hasattr(obj, 'todict') and callable(getattr(obj, 'todict')):
                    return obj.todict()

                # Handle functions/lambdas/callables
                if callable(obj):
                    return f"<function {getattr(obj, '__name__', 'anonymous')}>"

                # Try to convert to string as last resort
                try:
                    return str(obj)
                except Exception:
                    return "<non-serializable object>"

        # Convert dataclass to dict, then serialize and deserialize to handle special objects
        result_dict = asdict(self)
        json_str = json.dumps(result_dict, cls=CustomEncoder)
        return json.loads(json_str)
