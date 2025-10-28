"""Tests for result schema implementations."""

import pandas as pd
import json
import pytest
from pydantic import BaseModel
from datetime import datetime, UTC
from flex_evals import (
    TestCaseSummary,
    TestCaseResult,
    EvaluationSummary,
    EvaluationRunResult,
    TestCase,
    Output,
    Status,
    CheckResult,
    CheckError,
    Check,
    evaluate,
)
from flex_evals.schemas.results import ExecutionContext


class TestTestCaseSummary:
    """Test TestCaseSummary schema validation and behavior."""

    def test_valid_summary(self):
        """Test creating valid TestCaseSummary."""
        summary = TestCaseSummary(
            total_checks=4,
            completed_checks=3,
            error_checks=1,
        )

        assert summary.total_checks == 4
        assert summary.completed_checks == 3
        assert summary.error_checks == 1

    def test_zero_checks(self):
        """Test summary with zero checks."""
        summary = TestCaseSummary(
            total_checks=0,
            completed_checks=0,
            error_checks=0,
        )

        assert summary.total_checks == 0

    def test_all_completed(self):
        """Test summary with all checks completed."""
        summary = TestCaseSummary(
            total_checks=3,
            completed_checks=3,
            error_checks=0,
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
            )

    def test_counts_dont_sum_to_total(self):
        """Test that mismatched counts raise ValueError."""
        with pytest.raises(ValueError, match="check counts must sum to total_checks"):
            TestCaseSummary(
                total_checks=5,
                completed_checks=2,
                error_checks=1,  # Should be 3 to sum to 5
            )

    def test_counts_exceed_total(self):
        """Test that counts exceeding total raise ValueError."""
        with pytest.raises(ValueError, match="check counts must sum to total_checks"):
            TestCaseSummary(
                total_checks=3,
                completed_checks=2,
                error_checks=2,  # 2+2=4 > 3
            )


class TestTestCaseResult:
    """Test TestCaseResult schema validation and behavior."""

    def create_check_result(self, status: str, check_type: str = "test_check") -> CheckResult:
        """Helper to create CheckResult for testing."""
        return CheckResult(
            check_type=check_type,
            check_version='1.0.0',
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
        summary = TestCaseSummary(total_checks=1, completed_checks=1, error_checks=0)

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
                summary=TestCaseSummary(2, 2, 0),  # Wrong total
            )

    def test_summary_mismatch_completed(self):
        """Test that summary completed count mismatch raises ValueError."""
        error_check = CheckResult(
            check_type="test_check",
            check_version='1.0.0',
            status='error',
            results={},
            resolved_arguments={},
            evaluated_at=datetime.now(UTC),
            metadata={"check_version": "1.0.0"},
            error=CheckError("test_error", "Test error"),
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
                summary=TestCaseSummary(2, 2, 0),  # Wrong completed count
            )

    def test_status_mismatch_should_be_error(self):
        """Test that incorrect status raises ValueError when should be error."""
        error_check = CheckResult(
            check_type="test_check",
            check_version='1.0.0',
            status='error',
            results={},
            resolved_arguments={},
            evaluated_at=datetime.now(UTC),
            metadata={"check_version": "1.0.0"},
            error=CheckError("test_error", "Test error"),
        )
        check_results = [
            self.create_check_result('completed'),
            error_check,
        ]
        summary = TestCaseSummary(2, 1, 1)

        with pytest.raises(ValueError, match="status should be 'error'"):
            TestCaseResult(
                status='completed',  # Wrong - should be error
                execution_context=self.create_execution_context("test-1"),
                check_results=check_results,
                summary=summary,
            )



    def test_empty_check_results(self):
        """Test TestCaseResult with empty check results."""
        result = TestCaseResult(
            status='completed',
            execution_context=self.create_execution_context("test-1"),
            check_results=[],
            summary=TestCaseSummary(0, 0, 0),
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
            summary=TestCaseSummary(0, 0, 0),
        )

        # Verify both IDs are accessible
        assert result.execution_context.test_case.id == "test-specific"
        assert result.execution_context.output.id == "output-specific"


class TestEvaluationSummary:
    """Test EvaluationSummary schema validation and behavior."""

    def test_valid_summary(self):
        """Test creating valid EvaluationSummary."""
        summary = EvaluationSummary(
            total_test_cases=9,
            completed_test_cases=7,
            error_test_cases=2,
        )

        assert summary.total_test_cases == 9
        assert summary.completed_test_cases == 7
        assert summary.error_test_cases == 2

    def test_zero_test_cases(self):
        """Test summary with zero test cases."""
        summary = EvaluationSummary(
            total_test_cases=0,
            completed_test_cases=0,
            error_test_cases=0,
        )

        assert summary.total_test_cases == 0

    def test_negative_total_test_cases(self):
        """Test that negative total_test_cases raises ValueError."""
        with pytest.raises(ValueError, match="total_test_cases must be non-negative"):
            EvaluationSummary(
                total_test_cases=-1,
                completed_test_cases=0,
                error_test_cases=0,
            )

    def test_counts_dont_sum_to_total(self):
        """Test that mismatched counts raise ValueError."""
        with pytest.raises(ValueError, match="test case counts must sum to total_test_cases"):
            EvaluationSummary(
                total_test_cases=5,
                completed_test_cases=2,
                error_test_cases=1,  # Should be 3 to sum to 5
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
                    check_version='1.0.0',
                    status='completed',
                    results={},
                    resolved_arguments={},
                    evaluated_at=datetime.now(UTC),
                    metadata={"check_version": "1.0.0"},
                ),
            ]
            summary = TestCaseSummary(1, 1, 0)
        elif status == 'error':
            check_results = [
                CheckResult(
                    check_type="test_check",
                    check_version='1.0.0',
                    status='error',
                    results={},
                    resolved_arguments={},
                    evaluated_at=datetime.now(UTC),
                    metadata={"check_version": "1.0.0"},
                    error=CheckError('validation_error', "Test error"),
                ),
            ]
            summary = TestCaseSummary(1, 0, 1)
        else:
            raise ValueError(f"Invalid status: {status}")

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
        assert evaluation.metadata is None

    def test_with_metadata(self):
        """Test EvaluationRunResult with metadata."""
        started_at = datetime.now(UTC)
        completed_at = datetime.now(UTC)
        metadata = {
            "experiment_name": "test-experiment",
            "version": "1.0.0",
            "custom": "value",
        }
        evaluation = EvaluationRunResult(
            evaluation_id="eval-123",
            started_at=started_at,
            completed_at=completed_at,
            status='completed',
            summary=EvaluationSummary(0, 0, 0),
            results=[],
            metadata=metadata,
        )
        assert evaluation.metadata == metadata

    def test_empty_evaluation_id(self):
        """Test that empty evaluation_id raises ValueError."""
        with pytest.raises(ValueError, match="evaluation_id must be non-empty"):
            EvaluationRunResult(
                evaluation_id="",
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                status='completed',
                summary=EvaluationSummary(0, 0, 0),
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
                summary=EvaluationSummary(0, 0, 0),
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
                summary=EvaluationSummary(2, 2, 0),  # Wrong total
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
        summary = EvaluationSummary(2, 1, 1)

        with pytest.raises(ValueError, match="status should be 'error'"):
            EvaluationRunResult(
                evaluation_id="eval-123",
                started_at=started_at,
                completed_at=completed_at,
                status='completed',  # Wrong - should be error
                summary=summary,
                results=results,
            )

    def test_empty_results(self):
        """Test EvaluationRunResult with empty results."""
        started_at = datetime.now(UTC)
        completed_at = datetime.now(UTC)

        evaluation = EvaluationRunResult(
            evaluation_id="eval-123",
            started_at=started_at,
            completed_at=completed_at,
            status='completed',
            summary=EvaluationSummary(0, 0, 0),
            results=[],
        )

        assert len(evaluation.results) == 0
        assert evaluation.status == 'completed'

    def test_status_enum_values(self):
        """Test Status enum values."""
        assert Status.COMPLETED == 'completed'
        assert Status.ERROR == 'error'

    def test_check_result_accepts_status_enum(self):
        """Test CheckResult accepts Status enum."""
        metadata = {"check_version": "1.0.0"}
        result = CheckResult(
            check_type='exact_match',
            check_version='1.0.0',
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
        )
        metadata = {"check_version": "1.0.0"}
        check_result = CheckResult(
            check_type='exact_match',
            check_version='1.0.0',
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
        )
        test_case_summary = TestCaseSummary(
            total_checks=1,
            completed_checks=1,
            error_checks=0,
        )
        metadata = {"check_version": "1.0.0"}
        check_result = CheckResult(
            check_type='exact_match',
            check_version='1.0.0',
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

    def test_to_dict_list_empty_results(self):
        """Test to_dict_list with empty results."""
        now = datetime.now(UTC)
        evaluation = EvaluationRunResult(
            evaluation_id="eval-empty",
            started_at=now,
            completed_at=now,
            status='completed',
            summary=EvaluationSummary(0, 0, 0),
            results=[],
        )

        dict_list = evaluation.to_dict_list()
        assert dict_list == []

    def test_to_dict_list_single_test_case_single_check(self):
        """Test to_dict_list with a single test case and single check."""
        now = datetime.now(UTC)
        check_time = datetime.now(UTC)

        # Create a comprehensive test case with metadata
        test_case = TestCase(
            id="test-001",
            input={"query": "test input"},
            expected={"answer": "expected result"},
            metadata={"test_type": "unit"},
        )

        output = Output(
            value={"answer": "actual result"},
            id="output-001",
            metadata={"model": "test-model"},
        )

        check_result = CheckResult(
            check_type="exact_match",
            check_version='1.0.0',
            status='completed',
            results={"passed": True, "score": 1.0},
            resolved_arguments={"actual": "actual result", "expected": "expected result"},
            evaluated_at=check_time,
            metadata={"check_version": "1.0.0", "execution_time_ms": 50},
        )

        test_case_result = TestCaseResult(
            status='completed',
            execution_context=ExecutionContext(test_case=test_case, output=output),
            check_results=[check_result],
            summary=TestCaseSummary(1, 1, 0),
            metadata={"test_duration": "100ms"},
        )

        metadata = {
                "experiment_name": "test-experiment",
                "version": "1.0.0",
                "description": "Test experiment",
                "experiment_type": "evaluation",
                "evaluation_type": "test",
            }

        evaluation = EvaluationRunResult(
            evaluation_id="eval-001",
            started_at=now,
            completed_at=now,
            status='completed',
            summary=EvaluationSummary(1, 1, 0),
            results=[test_case_result],
            metadata=metadata,
        )

        dict_list = evaluation.to_dict_list()

        assert len(dict_list) == 1
        row = dict_list[0]

        # Verify evaluation context
        assert row['evaluation_id'] == "eval-001"
        assert row['started_at'] == now
        assert row['completed_at'] == now
        assert row['evaluation_status'] == 'completed'

        # Verify test case context
        assert row['test_case_id'] == "test-001"
        assert row['test_case_status'] == 'completed'
        assert row['input_data'] == {"query": "test input"}
        assert row['expected_output'] == {"answer": "expected result"}
        assert row['actual_output'] == {"answer": "actual result"}

        # Verify summary stats
        assert row['total_checks'] == 1
        assert row['completed_checks'] == 1
        assert row['error_checks'] == 0

        # Verify check-specific data
        assert row['check_type'] == "exact_match"
        assert row['check_status'] == 'completed'
        assert row['check_results'] == {"passed": True, "score": 1.0}
        assert row['check_results_passed'] is True
        assert row['resolved_arguments'] == {
            "actual": "actual result",
            "expected": "expected result",
        }
        assert row['evaluated_at'] == check_time

        # Verify metadata
        assert row['test_case_metadata'] == {"test_type": "unit"}
        assert row['output_metadata'] == {"model": "test-model"}
        assert row['test_case_result_metadata'] == {"test_duration": "100ms"}
        assert row['check_metadata'] == {"check_version": "1.0.0", "execution_time_ms": 50}
        assert row['evaluation_metadata'] == metadata

        # Verify no error fields are present for successful check
        assert 'error_type' not in row
        assert 'error_message' not in row
        assert 'error_recoverable' not in row

    def test_to_dict_list_multiple_checks_with_error(self):
        """Test to_dict_list with multiple checks including an error."""
        now = datetime.now(UTC)

        test_case = TestCase(
            id="test-002",
            input={"query": "test input"},
            expected={"answer": "expected result"},
        )

        output = Output(value={"answer": "actual result"}, id="output-002")

        # Successful check
        success_check = CheckResult(
            check_type="exact_match",
            check_version='1.0.0',
            status='completed',
            results={"passed": True},
            resolved_arguments={"actual": "actual result", "expected": "expected result"},
            evaluated_at=now,
        )

        # Failed check with error
        error_check = CheckResult(
            check_type="semantic_similarity",
            check_version='1.0.0',
            status='error',
            results={},
            resolved_arguments={"actual": "actual result", "expected": "expected result"},
            evaluated_at=now,
            error=CheckError(
                type='timeout_error',
                message="Check timed out",
            ),
        )

        test_case_result = TestCaseResult(
            status='error',
            execution_context=ExecutionContext(test_case=test_case, output=output),
            check_results=[success_check, error_check],
            summary=TestCaseSummary(2, 1, 1),
        )

        evaluation = EvaluationRunResult(
            evaluation_id="eval-002",
            started_at=now,
            completed_at=now,
            status='error',
            summary=EvaluationSummary(1, 0, 1),
            results=[test_case_result],
        )

        dict_list = evaluation.to_dict_list()
        assert len(dict_list) == 2

        # First row - successful check
        row1 = dict_list[0]
        assert row1['check_type'] == "exact_match"
        assert row1['check_status'] == 'completed'
        assert row1['check_results'] == {"passed": True}
        assert row1['check_results_passed'] is True
        assert 'error_type' not in row1

        # Second row - error check
        row2 = dict_list[1]
        assert row2['check_type'] == "semantic_similarity"
        assert row2['check_status'] == 'error'
        assert row2['check_results'] == {}
        assert row2['check_results_passed'] is None
        assert row2['error_type'] == 'timeout_error'
        assert row2['error_message'] == "Check timed out"

        # Both rows should have same test case context
        for row in dict_list:
            assert row['evaluation_id'] == "eval-002"
            assert row['test_case_id'] == "test-002"
            assert row['test_case_status'] == 'error'
            assert row['evaluation_status'] == 'error'

    def test_to_dict_list_multiple_test_cases(self):
        """Test to_dict_list with multiple test cases."""
        now = datetime.now(UTC)

        # Test case 1
        test_case1 = TestCase(id="test-001", input="input1", expected="expected1")
        output1 = Output(value="output1", id="output-001")
        check1 = CheckResult(
            check_type="exact_match",
            check_version='1.0.0',
            status='completed',
            results={"passed": True},
            resolved_arguments={"actual": "output1", "expected": "expected1"},
            evaluated_at=now,
        )
        tc_result1 = TestCaseResult(
            status='completed',
            execution_context=ExecutionContext(test_case1, output1),
            check_results=[check1],
            summary=TestCaseSummary(1, 1, 0),
        )

        # Test case 2
        test_case2 = TestCase(id="test-002", input="input2", expected="expected2")
        output2 = Output(value="output2", id="output-002")
        check2 = CheckResult(
            check_type="contains",
            check_version='1.0.0',
            status='error',
            results={},
            resolved_arguments={"actual": "output2", "expected": "expected2"},
            evaluated_at=now,
            error=CheckError('validation_error', "Test error"),
        )
        tc_result2 = TestCaseResult(
            status='error',
            execution_context=ExecutionContext(test_case2, output2),
            check_results=[check2],
            summary=TestCaseSummary(1, 0, 1),
        )

        evaluation = EvaluationRunResult(
            evaluation_id="eval-multi",
            started_at=now,
            completed_at=now,
            status='error',
            summary=EvaluationSummary(2, 1, 1),
            results=[tc_result1, tc_result2],
        )

        dict_list = evaluation.to_dict_list()
        assert len(dict_list) == 2

        # Verify each row has correct test case context
        row1 = dict_list[0]
        assert row1['test_case_id'] == "test-001"
        assert row1['check_type'] == "exact_match"
        assert row1['check_status'] == 'completed'
        assert row1['check_results_passed'] is True

        row2 = dict_list[1]
        assert row2['test_case_id'] == "test-002"
        assert row2['check_type'] == "contains"
        assert row2['check_status'] == 'error'
        assert row2['check_results_passed'] is None

        # Both should have same evaluation context
        for row in dict_list:
            assert row['evaluation_id'] == "eval-multi"
            assert row['evaluation_status'] == 'error'
        assert pd.DataFrame(dict_list).shape[0] == 2
        pd.DataFrame(dict_list).iloc[0].transpose()

    def test_to_dict_list_pandas_compatibility(self):
        """Test that to_dict_list output can be converted to pandas DataFrame."""
        now = datetime.now(UTC)

        # Create a simple evaluation result
        test_case = TestCase(id="test-pandas", input="test", expected="expected")
        output = Output(value="actual", id="output-pandas")
        check_result = CheckResult(
            check_type="exact_match",
            check_version='1.0.0',
            status='completed',
            results={"passed": False, "similarity": 0.8},
            resolved_arguments={"actual": "actual", "expected": "expected"},
            evaluated_at=now,
        )
        test_case_result = TestCaseResult(
            status='completed',
            execution_context=ExecutionContext(test_case, output),
            check_results=[check_result],
            summary=TestCaseSummary(1, 1, 0),
        )

        evaluation = EvaluationRunResult(
            evaluation_id="eval-pandas",
            started_at=now,
            completed_at=now,
            status='completed',
            summary=EvaluationSummary(1, 1, 0),
            results=[test_case_result],
        )

        dict_list = evaluation.to_dict_list()

        # Test that pandas can create a DataFrame from this data
        # First verify the structure is valid regardless of pandas availability
        assert isinstance(dict_list, list)
        assert len(dict_list) == 1
        assert isinstance(dict_list[0], dict)

        # Verify all expected keys are present
        expected_keys = {
            'evaluation_id', 'started_at', 'completed_at', 'evaluation_status',
            'test_case_id', 'test_case_status', 'input_data', 'expected_output',
            'actual_output', 'total_checks', 'completed_checks', 'error_checks',
            'check_type', 'check_status', 'check_results',
            'check_results_passed', 'resolved_arguments', 'evaluated_at',
        }
        assert expected_keys.issubset(dict_list[0].keys())

        # Verify the check_results_passed extraction
        assert dict_list[0]['check_results_passed'] is False

        # Convert to DataFrame
        dataframe = pd.DataFrame(dict_list)
        # Verify DataFrame properties
        assert len(dataframe) == 1
        assert 'evaluation_id' in dataframe.columns
        assert 'test_case_id' in dataframe.columns
        assert 'check_type' in dataframe.columns
        assert 'check_status' in dataframe.columns
        assert 'check_results' in dataframe.columns

        # Verify data types and values
        assert dataframe.loc[0, 'evaluation_id'] == "eval-pandas"
        assert dataframe.loc[0, 'test_case_id'] == "test-pandas"
        assert dataframe.loc[0, 'check_type'] == "exact_match"
        assert dataframe.loc[0, 'check_status'] == 'completed'

    def test_to_dict_list_with_check_metadata(self):
        """Test to_dict_list with check metadata flowing through to results."""
        now = datetime.now(UTC)

        # Create a CheckResult with custom metadata merged from Check
        test_case = TestCase(id="test-metadata", input="test", expected="expected")
        output = Output(value="actual", id="output-metadata")

        # Simulate what happens when a Check with metadata is executed
        # (metadata from Check gets merged with check_version)
        check_result = CheckResult(
            check_type="exact_match",
            check_version='1.0.0',
            status='completed',
            results={"passed": True},
            resolved_arguments={"actual": "actual", "expected": "expected"},
            evaluated_at=now,
            metadata={
                "check_version": "1.0.0",  # This comes from Check.version
                "custom_field": "custom_value",  # This comes from Check.metadata
                "priority": "high",  # This also comes from Check.metadata
            },
        )

        test_case_result = TestCaseResult(
            status='completed',
            execution_context=ExecutionContext(test_case, output),
            check_results=[check_result],
            summary=TestCaseSummary(1, 1, 0),
        )

        evaluation = EvaluationRunResult(
            evaluation_id="eval-metadata",
            started_at=now,
            completed_at=now,
            status='completed',
            summary=EvaluationSummary(1, 1, 0),
            results=[test_case_result],
        )

        dict_list = evaluation.to_dict_list()
        assert len(dict_list) == 1

        row = dict_list[0]

        # Verify the check metadata is properly extracted
        assert row['check_metadata'] == {
            "check_version": "1.0.0",
            "custom_field": "custom_value",
            "priority": "high",
        }

        # Verify other fields work as expected
        assert row['check_results_passed'] is True
        assert row['check_type'] == "exact_match"
        assert row['check_status'] == 'completed'

    def test_metadata_propagation_through_evaluate(self):
        """Test Check metadata and version propagation through evaluate to to_dict_list."""
        # Create test cases with different types of checks
        test_cases = [
            TestCase(id="test-001", input="hello world", expected="hello world"),
            TestCase(id="test-002", input="foo bar", expected="foo baz"),
        ]

        outputs = [
            Output(value="hello world", id="output-001"),
            Output(value="foo bar", id="output-002"),
        ]

        # Create checks with metadata and version
        checks = [
            Check(
                type="exact_match",
                arguments={
                    "actual": "$.output.value",
                    "expected": "$.test_case.expected",
                },
                version="1.0.0",
                metadata={"priority": "high", "category": "strict"},
            ),
            Check(
                type="contains",
                arguments={
                    "text": "$.output.value",
                    "phrases": ["foo"],
                },
                version="1.0.0",
                metadata={"priority": "medium", "timeout_ms": 1000},
            ),
        ]

        # Execute evaluation
        result = evaluate(test_cases, outputs, checks)

        # Convert to dict list for analysis
        dict_list = result.to_dict_list()

        # Should have 4 rows: 2 test cases x 2 checks
        assert len(dict_list) == 4

        # Check first test case, first check (exact_match on test-001)
        row1 = dict_list[0]
        assert row1['test_case_id'] == "test-001"
        assert row1['check_type'] == "exact_match"
        assert row1['check_metadata'] == {
            "priority": "high",
            "category": "strict",
        }
        assert row1['check_results_passed'] is True

        # Check first test case, second check (contains on test-001)
        row2 = dict_list[1]
        assert row2['test_case_id'] == "test-001"
        assert row2['check_type'] == "contains"
        assert row2['check_metadata'] == {
            "priority": "medium",
            "timeout_ms": 1000,
        }
        assert row2['check_results_passed'] is False  # "hello world" doesn't contain "foo"

        # Check second test case, first check (exact_match on test-002)
        row3 = dict_list[2]
        assert row3['test_case_id'] == "test-002"
        assert row3['check_type'] == "exact_match"
        assert row3['check_metadata'] == {
            "priority": "high",
            "category": "strict",
        }
        assert row3['check_results_passed'] is False  # "foo bar" != "foo baz"

        # Check second test case, second check (contains on test-002)
        row4 = dict_list[3]
        assert row4['test_case_id'] == "test-002"
        assert row4['check_type'] == "contains"
        assert row4['check_metadata'] == {
            "priority": "medium",
            "timeout_ms": 1000,
        }
        assert row4['check_results_passed'] is True  # "foo bar" contains "foo"

        # Verify that all rows have the expected structure
        for row in dict_list:
            assert 'check_version' in row  # Should be inside check_metadata
            assert 'check_metadata' in row
            assert isinstance(row['check_metadata'], dict)

    def test_serialize_method_basic(self):
        """Test serialize method converts dataclass to JSON-compatible dict."""
        test_case = TestCase(id='tc-1', input='test', expected='expected')
        output = Output(value='result')
        check_result = CheckResult(
            check_type='exact_match',
            check_version='1.0.0',
            status='completed',
            results={'passed': True},
            resolved_arguments={},
            evaluated_at=datetime.now(UTC),
            metadata=None,
            error=None,
        )
        execution_context = ExecutionContext(test_case=test_case, output=output)
        test_case_result = TestCaseResult(
            status='completed',
            execution_context=execution_context,
            check_results=[check_result],
            summary=TestCaseSummary(total_checks=1, completed_checks=1, error_checks=0),
            metadata=None,
        )
        result = EvaluationRunResult(
            evaluation_id='eval-123',
            started_at=datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC),
            completed_at=datetime(2024, 1, 15, 10, 35, 20, tzinfo=UTC),
            status='completed',
            summary=EvaluationSummary(
                total_test_cases=1,
                completed_test_cases=1,
                error_test_cases=0,
            ),
            results=[test_case_result],
            metadata={'source': 'test'},
        )

        serialized = result.serialize()

        # Verify it's a dictionary
        assert isinstance(serialized, dict)

        # Verify datetimes are converted to ISO strings
        assert isinstance(serialized['started_at'], str)
        assert serialized['started_at'] == '2024-01-15T10:30:45+00:00'
        assert isinstance(serialized['completed_at'], str)
        assert serialized['completed_at'] == '2024-01-15T10:35:20+00:00'

        # Verify evaluated_at in check results is also converted
        check_evaluated_at = serialized['results'][0]['check_results'][0]['evaluated_at']
        assert isinstance(check_evaluated_at, str)
        assert 'T' in check_evaluated_at  # ISO 8601 format

        # Verify other data is preserved
        assert serialized['evaluation_id'] == 'eval-123'
        assert serialized['status'] == 'completed'
        assert serialized['metadata'] == {'source': 'test'}

    def test_serialize_method_json_compatibility(self):
        """Test that serialized result can be JSON serialized."""
        test_case = TestCase(id='tc-1', input='test')
        output = Output(value='result')
        check_result = CheckResult(
            check_type='exact_match',
            check_version='1.0.0',
            status='completed',
            results={'passed': True},
            resolved_arguments={},
            evaluated_at=datetime.now(UTC),
            metadata=None,
            error=None,
        )
        execution_context = ExecutionContext(test_case=test_case, output=output)
        test_case_result = TestCaseResult(
            status='completed',
            execution_context=execution_context,
            check_results=[check_result],
            summary=TestCaseSummary(total_checks=1, completed_checks=1, error_checks=0),
        )
        result = EvaluationRunResult(
            evaluation_id='eval-123',
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            status='completed',
            summary=EvaluationSummary(
                total_test_cases=1,
                completed_test_cases=1,
                error_test_cases=0,
            ),
            results=[test_case_result],
        )

        serialized = result.serialize()

        # Should not raise an error
        json_str = json.dumps(serialized)
        assert len(json_str) > 0

        # Should be able to load it back
        loaded = json.loads(json_str)
        assert loaded['evaluation_id'] == 'eval-123'
        assert loaded['status'] == 'completed'

    def test_serialize_method_with_null_values(self):
        """Test serialize handles None/null values correctly."""
        test_case = TestCase(id='tc-1', input='test')
        output = Output(value='result', metadata=None)
        check_result = CheckResult(
            check_type='exact_match',
            check_version='1.0.0',
            status='completed',
            results={'passed': True},
            resolved_arguments={},
            evaluated_at=datetime.now(UTC),
            metadata=None,  # Explicitly None
            error=None,
        )
        execution_context = ExecutionContext(test_case=test_case, output=output)
        test_case_result = TestCaseResult(
            status='completed',
            execution_context=execution_context,
            check_results=[check_result],
            summary=TestCaseSummary(total_checks=1, completed_checks=1, error_checks=0),
            metadata=None,  # Explicitly None
        )
        result = EvaluationRunResult(
            evaluation_id='eval-123',
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            status='completed',
            summary=EvaluationSummary(
                total_test_cases=1,
                completed_test_cases=1,
                error_test_cases=0,
            ),
            results=[test_case_result],
            metadata=None,  # Explicitly None
        )

        serialized = result.serialize()

        # Verify None values are preserved
        assert serialized['metadata'] is None
        assert serialized['results'][0]['metadata'] is None
        assert serialized['results'][0]['check_results'][0]['metadata'] is None

    def test_serialize_method_with_nested_structures(self):
        """Test serialize handles nested complex structures."""
        test_case = TestCase(
            id='tc-1',
            input={'nested': {'data': 'value'}},
            expected={'result': [1, 2, 3]},
            metadata={'key': 'value', 'nested': {'meta': 'data'}},
        )
        output = Output(
            value={'output': {'nested': 'result'}},
            metadata={'tokens': 100, 'cost': 0.01},
        )
        check_result = CheckResult(
            check_type='exact_match',
            check_version='1.0.0',
            status='completed',
            results={'passed': True, 'details': {'match': 'exact'}},
            resolved_arguments={'actual': '$.output', 'expected': 'test'},
            evaluated_at=datetime.now(UTC),
            metadata={'custom': 'metadata'},
            error=None,
        )
        execution_context = ExecutionContext(test_case=test_case, output=output)
        test_case_result = TestCaseResult(
            status='completed',
            execution_context=execution_context,
            check_results=[check_result],
            summary=TestCaseSummary(total_checks=1, completed_checks=1, error_checks=0),
            metadata={'test_meta': 'value'},
        )
        result = EvaluationRunResult(
            evaluation_id='eval-123',
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            status='completed',
            summary=EvaluationSummary(
                total_test_cases=1,
                completed_test_cases=1,
                error_test_cases=0,
            ),
            results=[test_case_result],
            metadata={'run_meta': {'nested': 'value'}},
        )

        serialized = result.serialize()

        # Verify nested structures are preserved
        assert serialized['results'][0]['execution_context']['test_case']['input'] == {
            'nested': {'data': 'value'},
        }
        assert serialized['results'][0]['execution_context']['test_case']['metadata'] == {
            'key': 'value',
            'nested': {'meta': 'data'},
        }
        assert serialized['results'][0]['execution_context']['output']['value'] == {
            'output': {'nested': 'result'},
        }
        assert serialized['metadata'] == {'run_meta': {'nested': 'value'}}

    def test_serialize_method_with_non_serializable_objects(self):
        """Test serialize handles all non-serializable objects in metadata."""
        # Define test objects
        def custom_function() -> str:
            return "test"

        class CustomClass:
            pass

        class CustomPydanticModel(BaseModel):
            name: str
            value: int
            nested: dict[str, str]

        class CustomObjectWithToDict:
            """Custom class with to_dict method (common pattern)."""

            def __init__(self) -> None:
                self.name = 'custom_obj'
                self.value = 123

            def to_dict(self) -> dict[str, str | int]:
                return {'name': self.name, 'value': self.value}

        class CustomObjectWithToDictNoUnderscore:
            """Custom class with todict method (less common but exists)."""

            def __init__(self) -> None:
                self.field = 'test_field'

            def todict(self) -> dict[str, str]:
                return {'field': self.field}

        pydantic_instance = CustomPydanticModel(
            name='test_model',
            value=42,
            nested={'key': 'value'},
        )

        # Create test case with all edge cases in metadata
        test_case = TestCase(
            id='tc-1',
            input='test',
            metadata={
                'function': custom_function,
                'class': CustomClass,
                'lambda': lambda x: x * 2,
                # Pydantic CLASS (not instance) - source of a fixed bug
                'pydantic_class': CustomPydanticModel,
                'pydantic_model': pydantic_instance,
                'custom_to_dict': CustomObjectWithToDict(),
                'custom_todict': CustomObjectWithToDictNoUnderscore(),
                'datetime': datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC),
                'normal_data': {'key': 'value'},
            },
        )
        output = Output(value='result')
        check_result = CheckResult(
            check_type='exact_match',
            check_version='1.0.0',
            status='completed',
            results={'passed': True},
            resolved_arguments={},
            evaluated_at=datetime.now(UTC),
            metadata=None,
            error=None,
        )
        execution_context = ExecutionContext(test_case=test_case, output=output)
        test_case_result = TestCaseResult(
            status='completed',
            execution_context=execution_context,
            check_results=[check_result],
            summary=TestCaseSummary(total_checks=1, completed_checks=1, error_checks=0),
        )
        result = EvaluationRunResult(
            evaluation_id='eval-123',
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            status='completed',
            summary=EvaluationSummary(
                total_test_cases=1,
                completed_test_cases=1,
                error_test_cases=0,
            ),
            results=[test_case_result],
        )

        serialized = result.serialize()

        # Get metadata for assertions
        metadata = serialized['results'][0]['execution_context']['test_case']['metadata']

        # Verify function is converted to string
        assert isinstance(metadata['function'], str)
        assert metadata['function'] == '<function custom_function>'

        # Verify class is converted to string
        assert isinstance(metadata['class'], str)
        assert metadata['class'] == '<class CustomClass>'

        # Verify lambda is converted to string
        assert isinstance(metadata['lambda'], str)
        assert '<function' in metadata['lambda']

        # Verify Pydantic class (not instance) is converted to string
        assert isinstance(metadata['pydantic_class'], str)
        assert metadata['pydantic_class'] == '<class CustomPydanticModel>'

        # Verify Pydantic model is converted to dict (preserves data!)
        assert isinstance(metadata['pydantic_model'], dict)
        assert metadata['pydantic_model']['name'] == 'test_model'
        assert metadata['pydantic_model']['value'] == 42
        assert metadata['pydantic_model']['nested'] == {'key': 'value'}

        # Verify custom object with to_dict() method is converted to dict
        assert isinstance(metadata['custom_to_dict'], dict)
        assert metadata['custom_to_dict']['name'] == 'custom_obj'
        assert metadata['custom_to_dict']['value'] == 123

        # Verify custom object with todict() method is converted to dict
        assert isinstance(metadata['custom_todict'], dict)
        assert metadata['custom_todict']['field'] == 'test_field'

        # Verify datetime is converted to ISO string
        assert isinstance(metadata['datetime'], str)
        assert metadata['datetime'] == '2024-01-15T10:30:45+00:00'

        # Verify normal data is preserved
        assert metadata['normal_data'] == {'key': 'value'}

        # Verify the entire result can be JSON serialized
        json_str = json.dumps(serialized)
        assert len(json_str) > 0
