"""Tests for result schema implementations."""

import pytest
from datetime import datetime, UTC

from flex_evals.schemas.results import (
    TestCaseSummary,
    TestCaseResult,
    EvaluationSummary,
    EvaluationRunResult,
    ExecutionContext,
)
from flex_evals import TestCase, Output, Status, CheckResult, CheckError
from flex_evals.schemas.experiments import ExperimentMetadata


class TestTestCaseSummary:
    """Test TestCaseSummary schema validation and behavior."""

    def test_valid_summary(self):
        """Test creating valid TestCaseSummary."""
        summary = TestCaseSummary(
            total_checks=5,
            completed_checks=3,
            error_checks=1,
            skipped_checks=1,
        )

        assert summary.total_checks == 5
        assert summary.completed_checks == 3
        assert summary.error_checks == 1
        assert summary.skipped_checks == 1

    def test_zero_checks(self):
        """Test summary with zero checks."""
        summary = TestCaseSummary(
            total_checks=0,
            completed_checks=0,
            error_checks=0,
            skipped_checks=0,
        )

        assert summary.total_checks == 0

    def test_all_completed(self):
        """Test summary with all checks completed."""
        summary = TestCaseSummary(
            total_checks=3,
            completed_checks=3,
            error_checks=0,
            skipped_checks=0,
        )

        assert summary.total_checks == 3
        assert summary.completed_checks == 3

    def test_negative_total_checks(self):
        """Test that negative total_checks raises ValueError."""
        with pytest.raises(ValueError, match="total_checks must be non-negative"):
            TestCaseSummary(
                total_checks=-1,
                completed_checks=0,
                error_checks=0,
                skipped_checks=0,
            )

    def test_counts_dont_sum_to_total(self):
        """Test that mismatched counts raise ValueError."""
        with pytest.raises(ValueError, match="check counts must sum to total_checks"):
            TestCaseSummary(
                total_checks=5,
                completed_checks=2,
                error_checks=1,
                skipped_checks=1,  # Should be 2 to sum to 5
            )

    def test_counts_exceed_total(self):
        """Test that counts exceeding total raise ValueError."""
        with pytest.raises(ValueError, match="check counts must sum to total_checks"):
            TestCaseSummary(
                total_checks=3,
                completed_checks=2,
                error_checks=2,
                skipped_checks=2,  # 2+2+2=6 > 3
            )


class TestTestCaseResult:
    """Test TestCaseResult schema validation and behavior."""

    def create_check_result(self, status: str, check_type: str = "test_check") -> CheckResult:
        """Helper to create CheckResult for testing."""
        return CheckResult(
            check_type=check_type,
            status=status,
            results={},
            resolved_arguments={},
            evaluated_at=datetime.now(UTC),
            metadata=None,
        )

    def create_test_case(self, test_id: str = "test-1") -> TestCase:
        """Helper to create TestCase for testing."""
        return TestCase(id=test_id, input="test input", expected="test expected")

    def create_output(self, value: str = "test output", output_id: str = "output-1") -> Output:
        """Helper to create Output for testing."""
        return Output(value=value, id=output_id)

    def create_execution_context(self, test_id: str = "test-1") -> ExecutionContext:
        """Helper to create ExecutionContext for testing."""
        return ExecutionContext(
            test_case=self.create_test_case(test_id),
            output=self.create_output(),
        )

    def test_valid_test_case_result(self):
        """Test creating valid TestCaseResult."""
        check_results = [
            self.create_check_result('completed'),
            self.create_check_result('completed'),
        ]

        summary = TestCaseSummary(
            total_checks=2,
            completed_checks=2,
            error_checks=0,
            skipped_checks=0,
        )

        result = TestCaseResult(
            status='completed',
            execution_context=self.create_execution_context("test-1"),
            check_results=check_results,
            summary=summary,
        )

        assert result.execution_context.test_case.id == "test-1"
        assert result.execution_context.output.id == "output-1"
        assert result.status == 'completed'
        assert len(result.check_results) == 2
        assert result.metadata is None

    def test_with_metadata(self):
        """Test TestCaseResult with metadata."""
        check_results = [self.create_check_result('completed')]
        summary = TestCaseSummary(total_checks=1, completed_checks=1, error_checks=0, skipped_checks=0)  # noqa: E501

        result = TestCaseResult(
            status='completed',
            execution_context=self.create_execution_context("test-1"),
            check_results=check_results,
            summary=summary,
            metadata={"custom": "value"},
        )

        assert result.metadata == {"custom": "value"}
        # Verify output ID is accessible through execution context
        assert result.execution_context.output.id == "output-1"

    def test_empty_test_case_id(self):
        """Test that empty test_case_id raises ValueError."""
        # Test that empty test case id in TestCase raises error immediately
        with pytest.raises(ValueError, match="TestCase.id must be a non-empty string"):
            TestCase(id="", input="test", expected="test")

    def test_summary_mismatch_total(self):
        """Test that summary total mismatch raises ValueError."""
        check_results = [self.create_check_result('completed')]

        with pytest.raises(ValueError, match="summary does not match check_results"):
            TestCaseResult(
                status='completed',
                execution_context=self.create_execution_context("test-1"),
                check_results=check_results,
                summary=TestCaseSummary(2, 2, 0, 0),  # Wrong total
            )

    def test_summary_mismatch_completed(self):
        """Test that summary completed count mismatch raises ValueError."""
        error_check = CheckResult(
            check_type="test_check",
            status='error',
            results={},
            resolved_arguments={},
            evaluated_at=datetime.now(UTC),
            metadata={"check_version": "1.0.0"},
            error=CheckError("test_error", "Test error", True),
        )
        check_results = [
            self.create_check_result('completed'),
            error_check,
        ]

        with pytest.raises(ValueError, match="summary does not match check_results"):
            TestCaseResult(
                status='error',
                execution_context=self.create_execution_context("test-1"),
                check_results=check_results,
                summary=TestCaseSummary(2, 2, 0, 0),  # Wrong completed count
            )

    def test_status_mismatch_should_be_error(self):
        """Test that incorrect status raises ValueError when should be error."""
        error_check = CheckResult(
            check_type="test_check",
            status='error',
            results={},
            resolved_arguments={},
            evaluated_at=datetime.now(UTC),
            metadata={"check_version": "1.0.0"},
            error=CheckError("test_error", "Test error", True),
        )
        check_results = [
            self.create_check_result('completed'),
            error_check,
        ]
        summary = TestCaseSummary(2, 1, 1, 0)

        with pytest.raises(ValueError, match="status should be 'error'"):
            TestCaseResult(
                status='completed',  # Wrong - should be error
                execution_context=self.create_execution_context("test-1"),
                check_results=check_results,
                summary=summary,
            )

    def test_status_mismatch_should_be_skip(self):
        """Test that incorrect status raises ValueError when should be skip."""
        check_results = [
            self.create_check_result('completed'),
            self.create_check_result('skip'),
        ]
        summary = TestCaseSummary(2, 1, 0, 1)

        with pytest.raises(ValueError, match="status should be 'skip'"):
            TestCaseResult(
                status='completed',  # Wrong - should be skip
                execution_context=self.create_execution_context("test-1"),
                check_results=check_results,
                summary=summary,
            )

    def test_status_priority_error_over_skip(self):
        """Test that error status takes priority over skip."""
        error_check = CheckResult(
            check_type="test_check",
            status='error',
            results={},
            resolved_arguments={},
            evaluated_at=datetime.now(UTC),
            metadata={"check_version": "1.0.0"},
            error=CheckError("test_error", "Test error", True),
        )
        check_results = [
            error_check,
            self.create_check_result('skip'),
        ]
        summary = TestCaseSummary(2, 0, 1, 1)

        result = TestCaseResult(
            status='error',
            execution_context=self.create_execution_context("test-1"),
            check_results=check_results,
            summary=summary,
        )

        assert result.status == 'error'

    def test_empty_check_results(self):
        """Test TestCaseResult with empty check results."""
        result = TestCaseResult(
            status='completed',
            execution_context=self.create_execution_context("test-1"),
            check_results=[],
            summary=TestCaseSummary(0, 0, 0, 0),
        )

        assert len(result.check_results) == 0
        assert result.status == 'completed'

    def test_output_id_in_execution_context(self):
        """Test that output ID is properly stored and accessible."""
        # Create execution context with specific IDs
        test_case = self.create_test_case("test-specific")
        output = self.create_output("test output", "output-specific")
        execution_context = ExecutionContext(test_case=test_case, output=output)

        result = TestCaseResult(
            status='completed',
            execution_context=execution_context,
            check_results=[],
            summary=TestCaseSummary(0, 0, 0, 0),
        )

        # Verify both IDs are accessible
        assert result.execution_context.test_case.id == "test-specific"
        assert result.execution_context.output.id == "output-specific"


class TestEvaluationSummary:
    """Test EvaluationSummary schema validation and behavior."""

    def test_valid_summary(self):
        """Test creating valid EvaluationSummary."""
        summary = EvaluationSummary(
            total_test_cases=10,
            completed_test_cases=7,
            error_test_cases=2,
            skipped_test_cases=1,
        )

        assert summary.total_test_cases == 10
        assert summary.completed_test_cases == 7
        assert summary.error_test_cases == 2
        assert summary.skipped_test_cases == 1

    def test_zero_test_cases(self):
        """Test summary with zero test cases."""
        summary = EvaluationSummary(
            total_test_cases=0,
            completed_test_cases=0,
            error_test_cases=0,
            skipped_test_cases=0,
        )

        assert summary.total_test_cases == 0

    def test_negative_total_test_cases(self):
        """Test that negative total_test_cases raises ValueError."""
        with pytest.raises(ValueError, match="total_test_cases must be non-negative"):
            EvaluationSummary(
                total_test_cases=-1,
                completed_test_cases=0,
                error_test_cases=0,
                skipped_test_cases=0,
            )

    def test_counts_dont_sum_to_total(self):
        """Test that mismatched counts raise ValueError."""
        with pytest.raises(ValueError, match="test case counts must sum to total_test_cases"):
            EvaluationSummary(
                total_test_cases=5,
                completed_test_cases=2,
                error_test_cases=1,
                skipped_test_cases=1,  # Should be 2 to sum to 5
            )


class TestEvaluationRunResult:
    """Test EvaluationRunResult schema validation and behavior."""

    def create_test_case(self, test_id: str = "test-1") -> TestCase:
        """Helper to create TestCase for testing."""
        return TestCase(id=test_id, input="test input", expected="test expected")

    def create_output(self, value: str = "test output", output_id: str = "output-1") -> Output:
        """Helper to create Output for testing."""
        return Output(value=value, id=output_id)

    def create_execution_context(self, test_id: str = "test-1") -> ExecutionContext:
        """Helper to create ExecutionContext for testing."""
        return ExecutionContext(
            test_case=self.create_test_case(test_id),
            output=self.create_output(),
        )

    def create_test_case_result(self, status: str, test_case_id: str = "test-1") -> TestCaseResult:
        """Helper to create TestCaseResult for testing."""
        if status == 'completed':
            check_results = [
                CheckResult(
                    check_type="test_check",
                    status='completed',
                    results={},
                    resolved_arguments={},
                    evaluated_at=datetime.now(UTC),
                    metadata={"check_version": "1.0.0"},
                ),
            ]
            summary = TestCaseSummary(1, 1, 0, 0)
        elif status == 'error':
            check_results = [
                CheckResult(
                    check_type="test_check",
                    status='error',
                    results={},
                    resolved_arguments={},
                    evaluated_at=datetime.now(UTC),
                    metadata={"check_version": "1.0.0"},
                    error=CheckError('validation_error', "Test error", True),
                ),
            ]
            summary = TestCaseSummary(1, 0, 1, 0)
        else:  # skip
            check_results = [
                CheckResult(
                    check_type="test_check",
                    status='skip',
                    results={},
                    resolved_arguments={},
                    evaluated_at=datetime.now(UTC),
                    metadata={"check_version": "1.0.0"},
                ),
            ]
            summary = TestCaseSummary(1, 0, 0, 1)

        return TestCaseResult(
            status=status,
            execution_context=self.create_execution_context(test_case_id),
            check_results=check_results,
            summary=summary,
        )

    def test_valid_evaluation_result(self):
        """Test creating valid EvaluationRunResult."""
        started_at = datetime.now(UTC)
        completed_at = datetime.now(UTC)

        results = [
            self.create_test_case_result('completed', "test-1"),
            self.create_test_case_result('completed', "test-2"),
        ]

        summary = EvaluationSummary(
            total_test_cases=2,
            completed_test_cases=2,
            error_test_cases=0,
            skipped_test_cases=0,
        )

        evaluation = EvaluationRunResult(
            evaluation_id="eval-123",
            started_at=started_at,
            completed_at=completed_at,
            status='completed',
            summary=summary,
            results=results,
        )

        assert evaluation.evaluation_id == "eval-123"
        assert evaluation.status == 'completed'
        assert len(evaluation.results) == 2
        assert evaluation.experiment is None
        assert evaluation.metadata is None

    def test_with_experiment_metadata(self):
        """Test EvaluationRunResult with experiment metadata."""
        started_at = datetime.now(UTC)
        completed_at = datetime.now(UTC)

        experiment = ExperimentMetadata(
            name="test-experiment",
            metadata={"version": "1.0.0"},
        )

        evaluation = EvaluationRunResult(
            evaluation_id="eval-123",
            started_at=started_at,
            completed_at=completed_at,
            status='completed',
            summary=EvaluationSummary(0, 0, 0, 0),
            results=[],
            experiment=experiment,
            metadata={"custom": "value"},
        )

        assert evaluation.experiment.name == "test-experiment"
        assert evaluation.experiment.metadata["version"] == "1.0.0"
        assert evaluation.metadata == {"custom": "value"}

    def test_empty_evaluation_id(self):
        """Test that empty evaluation_id raises ValueError."""
        with pytest.raises(ValueError, match="evaluation_id must be non-empty"):
            EvaluationRunResult(
                evaluation_id="",
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                status='completed',
                summary=EvaluationSummary(0, 0, 0, 0),
                results=[],
            )

    def test_completed_before_started(self):
        """Test that completed_at before started_at raises ValueError."""
        started_at = datetime.now(UTC)
        completed_at = datetime(2020, 1, 1, tzinfo=UTC)  # Before started_at

        with pytest.raises(ValueError, match="completed_at must be >= started_at"):
            EvaluationRunResult(
                evaluation_id="eval-123",
                started_at=started_at,
                completed_at=completed_at,
                status='completed',
                summary=EvaluationSummary(0, 0, 0, 0),
                results=[],
            )

    def test_summary_mismatch_total(self):
        """Test that summary total mismatch raises ValueError."""
        started_at = datetime.now(UTC)
        completed_at = datetime.now(UTC)

        results = [self.create_test_case_result('completed')]

        with pytest.raises(ValueError, match="summary does not match results"):
            EvaluationRunResult(
                evaluation_id="eval-123",
                started_at=started_at,
                completed_at=completed_at,
                status='completed',
                summary=EvaluationSummary(2, 2, 0, 0),  # Wrong total
                results=results,
            )

    def test_status_mismatch_should_be_error(self):
        """Test that incorrect status raises ValueError when should be error."""
        started_at = datetime.now(UTC)
        completed_at = datetime.now(UTC)

        results = [
            self.create_test_case_result('completed'),
            self.create_test_case_result('error'),
        ]
        summary = EvaluationSummary(2, 1, 1, 0)

        with pytest.raises(ValueError, match="status should be 'error'"):
            EvaluationRunResult(
                evaluation_id="eval-123",
                started_at=started_at,
                completed_at=completed_at,
                status='completed',  # Wrong - should be error
                summary=summary,
                results=results,
            )

    def test_status_priority_error_over_skip(self):
        """Test that error status takes priority over skip."""
        started_at = datetime.now(UTC)
        completed_at = datetime.now(UTC)

        results = [
            self.create_test_case_result('error'),
            self.create_test_case_result('skip'),
        ]
        summary = EvaluationSummary(2, 0, 1, 1)

        evaluation = EvaluationRunResult(
            evaluation_id="eval-123",
            started_at=started_at,
            completed_at=completed_at,
            status='error',
            summary=summary,
            results=results,
        )

        assert evaluation.status == 'error'

    def test_empty_results(self):
        """Test EvaluationRunResult with empty results."""
        started_at = datetime.now(UTC)
        completed_at = datetime.now(UTC)

        evaluation = EvaluationRunResult(
            evaluation_id="eval-123",
            started_at=started_at,
            completed_at=completed_at,
            status='completed',
            summary=EvaluationSummary(0, 0, 0, 0),
            results=[],
        )

        assert len(evaluation.results) == 0
        assert evaluation.status == 'completed'

    def test_status_enum_values(self):
        """Test Status enum values."""
        assert Status.COMPLETED == 'completed'
        assert Status.ERROR == 'error'
        assert Status.SKIP == 'skip'

    def test_check_result_accepts_status_enum(self):
        """Test CheckResult accepts Status enum."""
        metadata = {"check_version": "1.0.0"}
        result = CheckResult(
            check_type='exact_match',
            status=Status.COMPLETED,
            results={"passed": True},
            resolved_arguments={"actual": "test", "expected": "test"},
            evaluated_at=datetime.now(UTC),
            metadata=metadata,
        )
        assert result.status == 'completed'

    def test_test_case_result_accepts_status_enum(self):
        """Test TestCaseResult accepts Status enum."""
        summary = TestCaseSummary(
            total_checks=1,
            completed_checks=1,
            error_checks=0,
            skipped_checks=0,
        )
        metadata = {"check_version": "1.0.0"}
        check_result = CheckResult(
            check_type='exact_match',
            status='completed',
            results={"passed": True},
            resolved_arguments={"actual": "test", "expected": "test"},
            evaluated_at=datetime.now(UTC),
            metadata=metadata,
        )

        result = TestCaseResult(
            status=Status.COMPLETED,
            execution_context=self.create_execution_context("test_001"),
            check_results=[check_result],
            summary=summary,
        )
        assert result.status == 'completed'

    def test_evaluation_run_result_accepts_status_enum(self):
        """Test EvaluationRunResult accepts Status enum."""
        summary = EvaluationSummary(
            total_test_cases=1,
            completed_test_cases=1,
            error_test_cases=0,
            skipped_test_cases=0,
        )
        test_case_summary = TestCaseSummary(
            total_checks=1,
            completed_checks=1,
            error_checks=0,
            skipped_checks=0,
        )
        metadata = {"check_version": "1.0.0"}
        check_result = CheckResult(
            check_type='exact_match',
            status='completed',
            results={"passed": True},
            resolved_arguments={"actual": "test", "expected": "test"},
            evaluated_at=datetime.now(UTC),
            metadata=metadata,
        )
        test_case_result = TestCaseResult(
            status='completed',
            execution_context=self.create_execution_context("test_001"),
            check_results=[check_result],
            summary=test_case_summary,
        )

        now = datetime.now(UTC)
        result = EvaluationRunResult(
            evaluation_id="eval_001",
            started_at=now,
            completed_at=now,
            status=Status.COMPLETED,
            summary=summary,
            results=[test_case_result],
        )
        assert result.status == 'completed'
