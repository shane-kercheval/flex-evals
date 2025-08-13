"""Tests for base check classes and evaluation context."""

from datetime import datetime, UTC
from typing import Any

from flex_evals.checks.base import BaseCheck, BaseAsyncCheck, EvaluationContext
from flex_evals.registry import register
from flex_evals.schemas import TestCase, Output, CheckResult


class TestExampleCheck(BaseCheck):
    """Test implementation of BaseCheck for testing."""

    def __call__(self, value: str, literal: str = "expected") -> dict[str, Any]:
        # Simple test check that validates required arguments
        return {
            "passed": value == literal,
            "actual_value": value,
        }


class TestExampleAsyncCheck(BaseAsyncCheck):
    """Test implementation of BaseAsyncCheck for testing."""

    async def __call__(self, value: str, literal: str = "expected") -> dict[str, Any]:
        # Simple async test check
        return {
            "passed": value == literal,
            "actual_value": value,
        }


class TestEvaluationContext:
    """Test EvaluationContext functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_case = TestCase(
            id="test_001",
            input={"question": "What is the capital of France?"},
            expected="Paris",
            metadata={"version": "1.0"},
        )

        self.output = Output(
            value={"answer": "Paris", "confidence": 0.95},
            metadata={"execution_time_ms": 245},
        )

        self.context = EvaluationContext(self.test_case, self.output)

    def test_evaluation_context_creation(self):
        """Test EvaluationContext creation and data access."""
        assert self.context.test_case == self.test_case
        assert self.context.output == self.output

        # Test context dict structure
        context_dict = self.context.context_dict
        assert context_dict["test_case"]["id"] == "test_001"
        assert context_dict["test_case"]["expected"] == "Paris"
        assert context_dict["output"]["value"]["answer"] == "Paris"
        assert context_dict["output"]["metadata"]["execution_time_ms"] == 245


class TestBaseCheck:
    """Test BaseCheck functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_case = TestCase(
            id="test_001",
            input={"question": "What is the capital of France?"},
            expected="Paris",
        )

        self.output = Output(
            value={"answer": "Paris", "confidence": 0.95},
        )

        self.context = EvaluationContext(self.test_case, self.output)
        self.check = TestExampleCheck()

    def test_successful_check_execution(self):
        """Test successful check execution."""
        # First register the test check so it can be looked up
        register("test_check", version="1.0.0")(TestExampleCheck)

        arguments = {"value": "expected"}
        result = self.check.execute("test_check", arguments, self.context)

        assert isinstance(result, CheckResult)
        assert result.check_type == "test_check"
        assert result.check_version == "1.0.0"
        assert result.status == 'completed'
        assert result.results["passed"] is True
        assert result.results["actual_value"] == "expected"
        assert result.error is None

    def test_check_with_jsonpath_arguments(self):
        """Test check execution with JSONPath argument resolution."""
        # Register the test check
        register("test_check", version="1.0.0")(TestExampleCheck)

        arguments = {
            "value": "$.output.value.answer",  # Should resolve to "Paris"
            "literal": "expected",
        }

        result = self.check.execute("test_check", arguments, self.context)

        assert result.status == 'completed'
        assert result.results["actual_value"] == "Paris"

        # Test resolved arguments format
        assert result.resolved_arguments["value"]["jsonpath"] == "$.output.value.answer"
        assert result.resolved_arguments["value"]["value"] == "Paris"
        assert result.resolved_arguments["literal"]["value"] == "expected"
        assert "jsonpath" not in result.resolved_arguments["literal"]

    def test_check_validation_error(self):
        """Test check execution with validation error."""
        # Register the test check
        register("test_check", version="1.0.0")(TestExampleCheck)

        arguments = {}  # Missing required "value" argument
        result = self.check.execute("test_check", arguments, self.context)

        assert result.status == 'error'
        assert result.error is not None
        assert result.error.type == 'validation_error'
        assert "Invalid arguments for check" in result.error.message
        assert result.error.recoverable is False
        assert result.results == {}

    def test_check_jsonpath_error(self):
        """Test check execution with JSONPath resolution error."""
        # Register the test check
        register("test_check", version="1.0.0")(TestExampleCheck)

        arguments = {"value": "$.nonexistent.path"}
        result = self.check.execute("test_check", arguments, self.context)

        assert result.status == 'error'
        assert result.error is not None
        assert result.error.type == 'jsonpath_error'
        assert "did not match any data" in result.error.message
        assert result.error.recoverable is False

    def test_check_execution_error(self):
        """Test check execution with unexpected error."""
        class FailingCheck(BaseCheck):
            def __call__(self, **kwargs):  # noqa
                raise RuntimeError("Unexpected error")

        # Register the failing check
        register("failing_check", version="1.0.0")(FailingCheck)

        check = FailingCheck()
        arguments = {"value": "test"}
        result = check.execute("failing_check", arguments, self.context)

        assert result.status == 'error'
        assert result.error is not None
        assert result.error.type == 'unknown_error'
        assert "Unexpected error during check execution" in result.error.message

    def test_check_version_preservation(self):
        """Test that check version is preserved in results."""
        # Register the test check with version 2.5.0
        register("test_check", version="2.5.0")(TestExampleCheck)

        arguments = {"value": "expected"}
        result = self.check.execute("test_check", arguments, self.context)

        assert result.check_version == "2.5.0"

    def test_timestamp_format(self):
        """Test that evaluated_at timestamp is in correct format."""
        # Register the test check
        register("test_check", version="1.0.0")(TestExampleCheck)

        arguments = {"value": "expected"}
        result = self.check.execute("test_check", arguments, self.context)

        assert isinstance(result.evaluated_at, datetime)
        assert result.evaluated_at.tzinfo == UTC


class TestBaseAsyncCheck:
    """Test BaseAsyncCheck functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_case = TestCase(
            id="test_001",
            input={"question": "What is the capital of France?"},
            expected="Paris",
        )

        self.output = Output(
            value={"answer": "Paris", "confidence": 0.95},
        )

        self.context = EvaluationContext(self.test_case, self.output)
        self.check = TestExampleAsyncCheck()

    async def test_successful_async_check_execution(self):
        """Test successful async check execution."""
        # Register the async test check
        register("test_async_check", version="1.0.0")(TestExampleAsyncCheck)

        arguments = {"value": "expected"}
        result = await self.check.execute("test_async_check", arguments, self.context)

        assert isinstance(result, CheckResult)
        assert result.check_type == "test_async_check"
        assert result.status == 'completed'
        assert result.results["passed"] is True
        assert result.results["actual_value"] == "expected"
        assert result.error is None

    async def test_async_check_with_jsonpath_arguments(self):
        """Test async check execution with JSONPath argument resolution."""
        # Register the async test check
        register("test_async_check", version="1.0.0")(TestExampleAsyncCheck)

        arguments = {
            "value": "$.output.value.answer",  # Should resolve to "Paris"
            "literal": "expected",
        }

        result = await self.check.execute("test_async_check", arguments, self.context)

        assert result.status == 'completed'
        assert result.results["actual_value"] == "Paris"

        # Test resolved arguments format
        assert result.resolved_arguments["value"]["jsonpath"] == "$.output.value.answer"
        assert result.resolved_arguments["value"]["value"] == "Paris"
        assert result.resolved_arguments["literal"]["value"] == "expected"

    async def test_async_check_validation_error(self):
        """Test async check execution with validation error."""
        # Register the async test check
        register("test_async_check", version="1.0.0")(TestExampleAsyncCheck)

        arguments = {}  # Missing required "value" argument
        result = await self.check.execute("test_async_check", arguments, self.context)

        assert result.status == 'error'
        assert result.error is not None
        assert result.error.type == 'validation_error'
        assert "Invalid arguments for check" in result.error.message

    async def test_async_check_execution_error(self):
        """Test async check execution with unexpected error."""
        class FailingAsyncCheck(BaseAsyncCheck):
            async def __call__(self, **kwargs):  # noqa
                raise RuntimeError("Async error")

        # Register the failing async check
        register("failing_async_check", version="1.0.0")(FailingAsyncCheck)

        check = FailingAsyncCheck()
        arguments = {"value": "test"}
        result = await check.execute("failing_async_check", arguments, self.context)

        assert result.status == 'error'
        assert result.error is not None
        assert result.error.type == 'unknown_error'
        assert "Unexpected error during async check execution" in result.error.message
