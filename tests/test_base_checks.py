"""Tests for base check classes and evaluation context."""

from datetime import datetime, UTC
from typing import Any

import pytest

from flex_evals.checks.base import BaseCheck, BaseAsyncCheck, EvaluationContext, JSONPath
from flex_evals.registry import register
from flex_evals.schemas import TestCase, Output, CheckResult
from pydantic import Field, field_validator


class TestExampleCheck(BaseCheck):
    """Test implementation of BaseCheck for testing."""

    # Pydantic fields with validation - can be literals or JSONPath objects
    value: str | JSONPath = Field(..., description='Test value to compare')
    literal: str | JSONPath = Field('expected', description='Expected value to match against')

    @field_validator('value', 'literal', mode='before')
    @classmethod
    def convert_jsonpath(cls, v):  # noqa: ANN001
        """Convert JSONPath-like strings to JSONPath objects."""
        if isinstance(v, str) and v.startswith('$.'):
            return JSONPath(expression=v)
        return v

    def __call__(self) -> dict[str, Any]:
        """Execute check using resolved Pydantic fields."""
        # Validate that all fields are resolved (no JSONPath objects remain)
        if isinstance(self.value, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'value' field: {self.value}")
        if isinstance(self.literal, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'literal' field: {self.literal}")

        # Simple test check that validates required arguments
        return {
            "passed": self.value == self.literal,
            "actual_value": self.value,
        }


class TestExampleAsyncCheck(BaseAsyncCheck):
    """Test implementation of BaseAsyncCheck for testing."""

    # Pydantic fields with validation - can be literals or JSONPath objects
    value: str | JSONPath = Field(..., description='Test value to compare')
    literal: str | JSONPath = Field('expected', description='Expected value to match against')

    @field_validator('value', 'literal', mode='before')
    @classmethod
    def convert_jsonpath(cls, v):  # noqa: ANN001
        """Convert JSONPath-like strings to JSONPath objects."""
        if isinstance(v, str) and v.startswith('$.'):
            return JSONPath(expression=v)
        return v

    async def __call__(self) -> dict[str, Any]:
        """Execute async check using resolved Pydantic fields."""
        # Validate that all fields are resolved (no JSONPath objects remain)
        if isinstance(self.value, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'value' field: {self.value}")
        if isinstance(self.literal, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'literal' field: {self.literal}")

        # Simple async test check
        return {
            "passed": self.value == self.literal,
            "actual_value": self.value,
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

    def test_successful_check_execution(self):
        """Test successful check execution."""
        # First register the test check so it can be looked up
        register("test_check", version="1.0.0")(TestExampleCheck)

        # Create check with field values
        check = TestExampleCheck(value="expected")
        result = check.execute(self.context)

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

        # Create check with JSONPath and literal values
        check = TestExampleCheck(
            value=JSONPath(expression="$.output.value.answer"),  # Should resolve to "Paris"
            literal="Paris",
        )
        result = check.execute(self.context)

        assert result.status == 'completed'
        assert result.results["actual_value"] == "Paris"
        assert result.results["passed"] is True

        # Test resolved arguments format
        assert result.resolved_arguments["value"]["jsonpath"] == "$.output.value.answer"
        assert result.resolved_arguments["value"]["value"] == "Paris"
        assert result.resolved_arguments["literal"]["value"] == "Paris"
        assert "jsonpath" not in result.resolved_arguments["literal"]

    def test_check_validation_error(self):
        """Test check execution with validation error."""
        # Register the test check
        register("test_check", version="1.0.0")(TestExampleCheck)

        # Missing required "value" field should raise validation error during instantiation
        try:
            check = TestExampleCheck()  # Missing required value field
            check.execute(self.context)
            pytest.fail("Should have raised validation error")
        except Exception as e:
            # Either during instantiation or execution, we should get validation error
            assert "value" in str(e).lower() or "required" in str(e).lower()  # noqa: PT017

    def test_check_jsonpath_error(self):
        """Test check execution with JSONPath resolution error."""
        # Register the test check
        register("test_check", version="1.0.0")(TestExampleCheck)

        # Create check with invalid JSONPath
        check = TestExampleCheck(value=JSONPath(expression="$.nonexistent.path"))
        result = check.execute(self.context)

        assert result.status == 'error'
        assert result.error is not None
        assert result.error.type == 'jsonpath_error'
        assert "did not match any data" in result.error.message
        assert result.error.recoverable is False

    def test_check_execution_error(self):
        """Test check execution with unexpected error."""
        class FailingCheck(BaseCheck):
            # Add required Pydantic field
            value: str = Field(default="test")

            def __call__(self) -> dict[str, Any]:
                raise RuntimeError("Unexpected error")

        # Register the failing check
        register("failing_check", version="1.0.0")(FailingCheck)

        check = FailingCheck()
        result = check.execute(self.context)

        assert result.status == 'error'
        assert result.error is not None
        assert result.error.type == 'unknown_error'
        assert "Unexpected error during check execution" in result.error.message

    def test_check_version_preservation(self):
        """Test that check version is preserved in results."""
        # Register the test check with version 2.5.0
        register("test_check", version="2.5.0")(TestExampleCheck)

        check = TestExampleCheck(value="expected")
        result = check.execute(self.context)

        assert result.check_version == "2.5.0"

    def test_timestamp_format(self):
        """Test that evaluated_at timestamp is in correct format."""
        # Register the test check
        register("test_check", version="1.0.0")(TestExampleCheck)

        check = TestExampleCheck(value="expected")
        result = check.execute(self.context)

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

    async def test_successful_async_check_execution(self):
        """Test successful async check execution."""
        # Register the async test check
        register("test_async_check", version="1.0.0")(TestExampleAsyncCheck)

        check = TestExampleAsyncCheck(value="expected")
        result = await check.execute(self.context)

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

        # Create check with JSONPath and literal values
        check = TestExampleAsyncCheck(
            value=JSONPath(expression="$.output.value.answer"),  # Should resolve to "Paris"
            literal="Paris",
        )
        result = await check.execute(self.context)

        assert result.status == 'completed'
        assert result.results["actual_value"] == "Paris"
        assert result.results["passed"] is True

        # Test resolved arguments format
        assert result.resolved_arguments["value"]["jsonpath"] == "$.output.value.answer"
        assert result.resolved_arguments["value"]["value"] == "Paris"
        assert result.resolved_arguments["literal"]["value"] == "Paris"

    async def test_async_check_validation_error(self):
        """Test async check execution with validation error."""
        # Register the async test check
        register("test_async_check", version="1.0.0")(TestExampleAsyncCheck)

        # Missing required "value" field should raise validation error during instantiation
        try:
            check = TestExampleAsyncCheck()  # Missing required value field
            await check.execute(self.context)
            pytest.fail("Should have raised validation error")
        except Exception as e:
            # Either during instantiation or execution, we should get validation error
            assert "value" in str(e).lower() or "required" in str(e).lower()  # noqa: PT017

    async def test_async_check_execution_error(self):
        """Test async check execution with unexpected error."""
        class FailingAsyncCheck(BaseAsyncCheck):
            # Add required Pydantic field
            value: str = Field(default="test")

            async def __call__(self) -> dict[str, Any]:
                raise RuntimeError("Async error")

        # Register the failing async check
        register("failing_async_check", version="1.0.0")(FailingAsyncCheck)

        check = FailingAsyncCheck()
        result = await check.execute(self.context)

        assert result.status == 'error'
        assert result.error is not None
        assert result.error.type == 'unknown_error'
        assert "Unexpected error during async check execution" in result.error.message
