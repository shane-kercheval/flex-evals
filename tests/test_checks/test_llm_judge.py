"""Comprehensive tests for LLMJudgeCheck implementation.

This module consolidates all tests for the LLMJudgeCheck including:
- Pydantic validation tests (from test_schema_check_classes.py)
- Implementation execution tests (from test_extended_checks.py) 
- Engine integration tests
- Template processing tests
- Edge cases and error handling

Tests are organized by functionality rather than implementation details.
"""

import pytest
import json
from datetime import datetime, UTC
from typing import Any

from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from flex_evals.checks.llm_judge import LLMJudgeCheck
from flex_evals.checks.base import EvaluationContext
from flex_evals import CheckType, Status, evaluate, Check, Output, TestCase
from flex_evals.exceptions import ValidationError, CheckExecutionError


# Test response format models for LLM judge
class QualityAssessment(BaseModel):
    """Simple quality assessment response format."""

    score: int = Field(ge=1, le=5, description="Quality score from 1-5")
    is_helpful: bool = Field(description="Whether the response is helpful")
    reasoning: str = Field(description="Explanation of the assessment")


class DetailedEvaluation(BaseModel):
    """Complex response format with nested structures."""

    overall_score: float = Field(ge=0.0, le=1.0, description="Overall quality score")
    categories: dict[str, int] = Field(description="Scores by category")
    issues: list[str] = Field(default_factory=list, description="Identified issues")
    recommendations: list[str] = Field(default_factory=list, description="Improvement suggestions")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SimpleBooleanResult(BaseModel):
    """Minimal boolean response format."""

    passed: bool = Field(description="Whether the check passed")


class TestLLMJudgeValidation:
    """Test Pydantic validation and field handling for LLMJudgeCheck."""

    def test_llm_judge_check_creation(self):
        """Test basic LLMJudgeCheck creation."""
        class TestResponse(BaseModel):
            score: int = Field(ge=1, le=10)

        def mock_llm_function(prompt: str, response_format: type) -> tuple[BaseModel, dict]:  # noqa: ARG001
            return TestResponse(score=8), {"cost": 0.01}

        check = LLMJudgeCheck(
            prompt="Rate this: {{$.output.value}}",
            response_format=TestResponse,
            llm_function=mock_llm_function,
        )

        assert check.prompt == "Rate this: {{$.output.value}}"
        assert check.response_format == TestResponse
        assert check.llm_function == mock_llm_function

    def test_llm_judge_check_validation_empty_prompt(self):
        """Test LLMJudgeCheck validation for empty prompt - now allowed."""
        class TestResponse(BaseModel):
            score: int

        def mock_llm_function(prompt: str, response_format: type) -> tuple[BaseModel, dict]:  # noqa: ARG001
            return TestResponse(score=8), {}

        check = LLMJudgeCheck(
            prompt="",
            response_format=TestResponse,
            llm_function=mock_llm_function,
        )
        assert check.prompt == ""
        assert check.response_format == TestResponse

    def test_llm_judge_check_type_property(self):
        """Test LLMJudgeCheck check_type property returns correct type."""
        def mock_llm_function(prompt: str, response_format: type) -> tuple[BaseModel, dict]:  # noqa: ARG001
            return BaseModel(), {}

        check = LLMJudgeCheck(
            prompt="test",
            response_format=BaseModel,
            llm_function=mock_llm_function,
        )
        assert check.check_type == CheckType.LLM_JUDGE

    def test_llm_judge_check_jsonpath_prompt(self):
        """Test LLMJudgeCheck with JSONPath prompt."""
        class TestResponse(BaseModel):
            score: int = Field(ge=1, le=10)

        def mock_llm_function(prompt: str, response_format: type) -> tuple[BaseModel, dict]:  # noqa: ARG001
            return TestResponse(score=8), {"cost": 0.01}

        check = LLMJudgeCheck(
            prompt="$.prompt_template",
            response_format=TestResponse,
            llm_function=mock_llm_function,
        )

        assert check.prompt == "$.prompt_template"
        assert check.response_format == TestResponse
        assert check.llm_function == mock_llm_function

    def test_llm_judge_check_jsonpath_prompt(self):
        """Test LLMJudgeCheck with JSONPath prompt (response_format must be actual class)."""
        class TestResponse(BaseModel):
            score: int = Field(ge=1, le=10)

        def mock_llm_function(prompt: str, response_format: type) -> tuple[BaseModel, dict]:  # noqa: ARG001
            return TestResponse(score=8), {"cost": 0.01}

        check = LLMJudgeCheck(
            prompt="$.templates.evaluation_prompt",
            response_format=TestResponse,  # Must be actual class, not JSONPath
            llm_function=mock_llm_function,
        )

        assert check.prompt == "$.templates.evaluation_prompt"
        assert check.response_format == TestResponse
        assert check.llm_function == mock_llm_function

    def test_llm_judge_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(PydanticValidationError):
            LLMJudgeCheck()  # type: ignore
        
        with pytest.raises(PydanticValidationError):
            LLMJudgeCheck(prompt="test")  # type: ignore


class TestLLMJudgeExecution:
    """Test LLMJudgeCheck execution logic and __call__ method."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test evaluation context
        self.test_case = TestCase(
            id="test_001",
            input={"user_message": "What is the capital of France?", "context": "geography"},
            expected={"answer": "Paris", "quality_level": "high"},
            metadata={"source": "test_suite", "difficulty": "easy"},
        )

        self.output = Output(
            value={"response": "The capital of France is Paris.", "confidence": 0.95},
            metadata={"execution_time_ms": 250, "model": "gpt-4"},
        )

        self.context = EvaluationContext(self.test_case, self.output)

    def create_sync_llm_function(self, response: Any, metadata: dict[str, Any] | None = None):
        """Create a synchronous mock LLM function that returns the given response."""
        def mock_llm(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict[str, Any]]:  # noqa
            if isinstance(response, BaseModel):
                parsed = response
            elif isinstance(response, dict):
                parsed = response_format(**response)
            elif isinstance(response, str):
                parsed = response_format(**json.loads(response))
            else:
                parsed = response_format(response)

            # Return tuple with metadata
            mock_metadata = metadata or {
                "cost_usd": 0.0023,
                "tokens_used": 150,
                "response_time_ms": 842,
                "model_version": "gpt-4o-mini-2024-07-02",
            }
            return parsed, mock_metadata
        return mock_llm

    def create_async_llm_function(self, response: Any, metadata: dict[str, Any] | None = None):
        """Create an asynchronous mock LLM function that returns the given response."""
        async def mock_llm(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict[str, Any]]:  # noqa
            if isinstance(response, BaseModel):
                parsed = response
            elif isinstance(response, dict):
                parsed = response_format(**response)
            elif isinstance(response, str):
                parsed = response_format(**json.loads(response))
            else:
                parsed = response_format(response)

            # Return tuple with metadata
            mock_metadata = metadata or {
                "cost_usd": 0.0015,
                "tokens_used": 120,
                "response_time_ms": 650,
                "model_version": "gpt-4o-mini-async",
            }
            return parsed, mock_metadata
        return mock_llm

    def create_failing_llm_function(self, error_message: str = "LLM API failed"):
        """Create an LLM function that always raises an exception."""
        def failing_llm(prompt: str, response_format: type[BaseModel]) -> BaseModel:  # noqa
            raise RuntimeError(error_message)
        return failing_llm

    @pytest.mark.asyncio
    async def test_call_basic_functionality(self):
        """Test basic __call__ functionality with processed prompt."""
        response_data = {
            "score": 4,
            "is_helpful": True,
            "reasoning": "Clear and accurate response",
        }
        llm_function = self.create_sync_llm_function(response_data)
        check = LLMJudgeCheck(prompt="test", response_format=QualityAssessment, llm_function=llm_function)

        result = await check(
            prompt="Evaluate: User asked 'What is the capital of France?', AI responded 'Paris'",
            response_format=QualityAssessment,
            llm_function=llm_function,
        )

        # Check new structure with separate judge_metadata
        expected_structure = {
            "score": 4,
            "is_helpful": True,
            "reasoning": "Clear and accurate response",
            "judge_metadata": {
                "cost_usd": 0.0023,
                "tokens_used": 150,
                "response_time_ms": 842,
                "model_version": "gpt-4o-mini-2024-07-02",
            },
        }
        assert result == expected_structure

    @pytest.mark.asyncio
    async def test_call_with_async_llm_function(self):
        """Test __call__ with asynchronous LLM function."""
        response_data = {"passed": True}
        llm_function = self.create_async_llm_function(response_data)
        check = LLMJudgeCheck(prompt="test", response_format=SimpleBooleanResult, llm_function=llm_function)

        result = await check(
            prompt="Is this response helpful?",
            response_format=SimpleBooleanResult,
            llm_function=llm_function,
        )

        assert result["passed"] is True
        assert result["judge_metadata"]["model_version"] == "gpt-4o-mini-async"

    @pytest.mark.asyncio
    async def test_call_with_complex_response_format(self):
        """Test __call__ with complex nested response format."""
        response_data = {
            "overall_score": 0.85,
            "categories": {"accuracy": 5, "clarity": 4, "completeness": 4},
            "issues": ["Could provide more examples"],
            "recommendations": ["Add specific examples", "Include sources"],
            "metadata": {"confidence": 0.9, "processing_time": 1.2},
        }
        llm_function = self.create_sync_llm_function(response_data)
        check = LLMJudgeCheck(prompt="test", response_format=DetailedEvaluation, llm_function=llm_function)

        result = await check(
            prompt="Provide detailed evaluation",
            response_format=DetailedEvaluation,
            llm_function=llm_function,
        )

        # Check structure includes all response fields and separate judge_metadata
        assert result["overall_score"] == 0.85
        assert result["categories"] == {"accuracy": 5, "clarity": 4, "completeness": 4}
        assert result["issues"] == ["Could provide more examples"]
        assert result["recommendations"] == ["Add specific examples", "Include sources"]
        assert result["metadata"] == {"confidence": 0.9, "processing_time": 1.2}
        # Should also have LLM metadata in separate field
        assert "cost_usd" in result["judge_metadata"]
        assert "tokens_used" in result["judge_metadata"]

    @pytest.mark.asyncio
    async def test_call_argument_validation(self):
        """Test __call__ argument validation."""
        llm_function = self.create_sync_llm_function({"passed": True})
        check = LLMJudgeCheck(prompt="test", response_format=SimpleBooleanResult, llm_function=llm_function)

        # Test invalid prompt type
        with pytest.raises(ValidationError, match="prompt must be a string"):
            await check(
                prompt=123,  # Invalid type
                response_format=SimpleBooleanResult,
                llm_function=llm_function,
            )

        # Test invalid response_format type
        with pytest.raises(ValidationError, match="response_format must be a Pydantic BaseModel class"):
            await check(
                prompt="test prompt",
                response_format=dict,  # Invalid type
                llm_function=llm_function,
            )

        # Test invalid llm_function type
        with pytest.raises(ValidationError, match="llm_function must be callable"):
            await check(
                prompt="test prompt",
                response_format=SimpleBooleanResult,
                llm_function="not callable",  # Invalid type
            )

    @pytest.mark.asyncio
    async def test_call_llm_function_failure(self):
        """Test __call__ handling of LLM function failures."""
        failing_llm = self.create_failing_llm_function("API quota exceeded")
        check = LLMJudgeCheck(prompt="test", response_format=SimpleBooleanResult, llm_function=failing_llm)

        with pytest.raises(CheckExecutionError, match="Error in LLM judge evaluation.*API quota exceeded"):
            await check(
                prompt="test prompt",
                response_format=SimpleBooleanResult,
                llm_function=failing_llm,
            )

    @pytest.mark.asyncio
    async def test_call_response_validation_errors(self):
        """Test __call__ response format validation errors."""
        # Test invalid return type (not tuple)
        def invalid_json_llm(prompt: str, response_format: type[BaseModel]) -> str:  # noqa
            return "invalid json{{"

        check = LLMJudgeCheck(prompt="test", response_format=SimpleBooleanResult, llm_function=invalid_json_llm)

        with pytest.raises(ValidationError, match="llm_function must return tuple"):
            await check(
                prompt="test",
                response_format=SimpleBooleanResult,
                llm_function=invalid_json_llm,
            )

    @pytest.mark.asyncio
    async def test_call_response_format_handling(self):
        """Test __call__ handling of different response types."""
        # Test with tuple response (new required format)
        response_instance = QualityAssessment(score=5, is_helpful=True, reasoning="Perfect")
        custom_metadata = {"cost_usd": 0.001, "tokens_used": 75, "model": "test-model"}
        
        def llm_function(p, rf):  # noqa
            return response_instance, custom_metadata

        check = LLMJudgeCheck(prompt="test", response_format=QualityAssessment, llm_function=llm_function)

        result = await check(
            prompt="test",
            response_format=QualityAssessment,
            llm_function=llm_function,
        )
        
        # Check structure with separate judge_metadata
        assert result["score"] == 5
        assert result["is_helpful"] is True
        assert result["reasoning"] == "Perfect"
        assert result["judge_metadata"]["cost_usd"] == 0.001
        assert result["judge_metadata"]["tokens_used"] == 75
        assert result["judge_metadata"]["model"] == "test-model"


class TestLLMJudgeTemplateProcessing:
    """Test LLMJudgeCheck template processing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_case = TestCase(
            id="test_001",
            input={"user_message": "What is the capital of France?"},
            expected={"answer": "Paris"},
        )

        self.output = Output(
            value={"response": "The capital of France is Paris.", "confidence": 0.95},
        )

        self.context = EvaluationContext(self.test_case, self.output)

    def test_process_prompt_template_basic(self):
        """Test basic template processing."""
        check = LLMJudgeCheck(
            prompt="test",
            response_format=BaseModel,
            llm_function=lambda x, y: (BaseModel(), {}),
        )

        template = "User asked: {{$.test_case.input.user_message}}, AI responded: {{$.output.value.response}}"
        processed = check._process_prompt_template(template, self.context)

        expected = "User asked: What is the capital of France?, AI responded: The capital of France is Paris."
        assert processed == expected

    def test_process_prompt_template_no_placeholders(self):
        """Test template processing with no placeholders."""
        check = LLMJudgeCheck(
            prompt="test",
            response_format=BaseModel,
            llm_function=lambda x, y: (BaseModel(), {}),
        )

        template = "This is a simple prompt with no templates"
        processed = check._process_prompt_template(template, self.context)

        assert processed == template

    def test_process_prompt_template_complex_data(self):
        """Test template processing with complex data structures."""
        context_with_complex_data = EvaluationContext(
            TestCase(
                id="test",
                input={"data": {"items": [{"name": "item1"}, {"name": "item2"}]}},
                expected={"result": {"success": True, "count": 2}},
            ),
            Output(value={"response": {"status": "completed", "items_processed": 2}}),
        )

        check = LLMJudgeCheck(
            prompt="test",
            response_format=BaseModel,
            llm_function=lambda x, y: (BaseModel(), {}),
        )

        template = "Input had {{$.test_case.input.data.items}} and output was {{$.output.value.response}}"
        processed = check._process_prompt_template(template, context_with_complex_data)

        # JSON serialization of complex objects
        assert '[{"name": "item1"}, {"name": "item2"}]' in processed
        assert '{"status": "completed", "items_processed": 2}' in processed

    def test_process_prompt_template_null_values(self):
        """Test template processing with null/None values."""
        context_with_nulls = EvaluationContext(
            TestCase(id="test", input={"value": None}, expected={"result": None}),
            Output(value={"response": None}),
        )

        check = LLMJudgeCheck(
            prompt="test",
            response_format=BaseModel,
            llm_function=lambda x, y: (BaseModel(), {}),
        )

        template = "Input: {{$.test_case.input.value}}, Output: {{$.output.value.response}}"
        processed = check._process_prompt_template(template, context_with_nulls)

        # None values should be converted to empty strings
        assert processed == "Input: , Output: "

    def test_process_prompt_template_missing_data(self):
        """Test template processing with missing JSONPath data."""
        check = LLMJudgeCheck(
            prompt="test",
            response_format=BaseModel,
            llm_function=lambda x, y: (BaseModel(), {}),
        )

        # Template references non-existent path
        template = "User asked: {{$.test_case.input.nonexistent_field}}"

        # Should raise JSONPathError
        from flex_evals.exceptions import JSONPathError
        with pytest.raises(JSONPathError):
            check._process_prompt_template(template, self.context)


class TestLLMJudgeEngineIntegration:
    """Test LLMJudgeCheck integration with the evaluation engine."""

    @pytest.mark.asyncio
    async def test_llm_judge_via_evaluate(self):
        """Test LLM judge through engine evaluation."""
        def mock_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:
            return response_format(passed=True), {"cost": 0.01, "model": "gpt-4"}

        test_cases = [
            TestCase(
                id="test_001",
                input={"question": "What is the capital of France?"},
                expected={"answer": "Paris"},
                checks=[
                    Check(
                        type=CheckType.LLM_JUDGE,
                        arguments={
                            "prompt": "Evaluate if the answer is correct: {{$.output.value}}",
                            "response_format": SimpleBooleanResult,
                            "llm_function": mock_llm_function,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="Paris")]
        results = evaluate(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results["passed"] is True
        assert "judge_metadata" in results.results[0].check_results[0].results

    @pytest.mark.asyncio
    async def test_llm_judge_check_instance_via_evaluate(self):
        """Test direct LLM judge check instance usage in evaluate function."""
        def mock_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:
            return response_format(score=4, is_helpful=True, reasoning="Good response"), {"cost": 0.01}

        test_cases = [
            TestCase(
                id="test_001",
                input={"question": "What is the capital of France?"},
                checks=[
                    LLMJudgeCheck(
                        prompt="Rate this response: {{$.output.value}}",
                        response_format=QualityAssessment,
                        llm_function=mock_llm_function,
                    ),
                ],
            ),
        ]

        outputs = [Output(value="Paris")]
        results = evaluate(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results["score"] == 4
        assert results.results[0].check_results[0].results["is_helpful"] is True

    @pytest.mark.asyncio
    async def test_llm_judge_template_via_evaluate(self):
        """Test LLM judge template processing through engine evaluation."""
        def mock_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:
            # Verify template was processed correctly
            assert "Question: What is the capital of France?" in prompt
            assert "Answer: Paris" in prompt
            return response_format(passed=True), {"cost": 0.01}

        test_cases = [
            TestCase(
                id="test_001",
                input={"question": "What is the capital of France?"},
                expected={"answer": "Paris"},
                checks=[
                    Check(
                        type=CheckType.LLM_JUDGE,
                        arguments={
                            "prompt": "Question: {{$.test_case.input.question}}, Answer: {{$.output.value}}",
                            "response_format": SimpleBooleanResult,
                            "llm_function": mock_llm_function,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="Paris")]
        results = evaluate(test_cases, outputs)

        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results["passed"] is True


class TestLLMJudgeErrorHandling:
    """Test error handling and edge cases for LLMJudgeCheck."""

    def test_llm_judge_template_processing_error_in_engine(self):
        """Test that template processing errors are handled correctly in engine."""
        def mock_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:
            return response_format(passed=True), {}

        test_cases = [
            TestCase(
                id="test_001",
                input={"question": "test"},
                checks=[
                    Check(
                        type=CheckType.LLM_JUDGE,
                        arguments={
                            "prompt": "Invalid template: {{$.nonexistent.path}}",
                            "response_format": SimpleBooleanResult,
                            "llm_function": mock_llm_function,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="test")]
        results = evaluate(test_cases, outputs)

        # Should complete but with error status due to template processing failure
        assert results.summary.total_test_cases == 1
        assert results.summary.error_test_cases == 1
        assert results.results[0].status == Status.ERROR

    @pytest.mark.asyncio
    async def test_llm_judge_llm_function_error_in_engine(self):
        """Test LLM function errors are handled correctly in engine."""
        def failing_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:
            raise RuntimeError("API Error")

        test_cases = [
            TestCase(
                id="test_001",
                input={"question": "test"},
                checks=[
                    Check(
                        type=CheckType.LLM_JUDGE,
                        arguments={
                            "prompt": "Simple prompt",
                            "response_format": SimpleBooleanResult,
                            "llm_function": failing_llm_function,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="test")]
        results = evaluate(test_cases, outputs)

        # Should complete but with error status due to LLM function failure
        assert results.summary.total_test_cases == 1
        assert results.summary.error_test_cases == 1
        assert results.results[0].status == Status.ERROR