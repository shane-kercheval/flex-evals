"""Tests for extended check implementations."""

import pytest
import json
from datetime import datetime, UTC
from typing import Any, Never

from pydantic import BaseModel, Field

from flex_evals.checks.extended.llm_judge import LlmJudgeCheck_v1_0_0
from flex_evals.checks.extended.custom_function import CustomFunctionCheck_v1_0_0
from flex_evals.checks.base import EvaluationContext
from flex_evals import CheckType, Status, evaluate, Check, Output, TestCase
from flex_evals.exceptions import ValidationError, CheckExecutionError
from flex_evals.registry import list_registered_checks


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


class TestLlmJudgeCheck:
    """Test LLM Judge check implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.check = LlmJudgeCheck_v1_0_0()

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

    def create_sync_llm_function(self, response: Any, metadata: dict[str, Any] | None = None):  # noqa
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

    def create_async_llm_function(self, response: Any, metadata: dict[str, Any] | None = None):  # noqa
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

        result = await self.check(
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

        result = await self.check(
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

        result = await self.check(
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

        # Test invalid prompt type
        with pytest.raises(ValidationError, match="prompt must be a string"):
            await self.check(
                prompt=123,  # Invalid type
                response_format=SimpleBooleanResult,
                llm_function=llm_function,
            )

        # Test invalid response_format type
        with pytest.raises(ValidationError, match="response_format must be a Pydantic BaseModel class"):  # noqa: E501
            await self.check(
                prompt="test prompt",
                response_format=dict,  # Invalid type
                llm_function=llm_function,
            )

        # Test invalid llm_function type
        with pytest.raises(ValidationError, match="llm_function must be callable"):
            await self.check(
                prompt="test prompt",
                response_format=SimpleBooleanResult,
                llm_function="not callable",  # Invalid type
            )

    @pytest.mark.asyncio
    async def test_call_llm_function_failure(self):
        """Test __call__ handling of LLM function failures."""
        failing_llm = self.create_failing_llm_function("API quota exceeded")

        with pytest.raises(CheckExecutionError, match="Error in LLM judge evaluation.*API quota exceeded"):  # noqa: E501
            await self.check(
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

        with pytest.raises(CheckExecutionError, match="llm_function must return tuple"):
            await self.check(
                prompt="test",
                response_format=SimpleBooleanResult,
                llm_function=invalid_json_llm,
            )

        # Test invalid return type (dict instead of tuple)
        def invalid_schema_llm(prompt: str, response_format: type[BaseModel]) -> dict:  # noqa
            return {"invalid_field": "value"}  # Not a tuple

        with pytest.raises(CheckExecutionError, match="llm_function must return tuple"):
            await self.check(
                prompt="test",
                response_format=SimpleBooleanResult,
                llm_function=invalid_schema_llm,
            )

    @pytest.mark.asyncio
    async def test_call_response_format_handling(self):
        """Test __call__ handling of different response types."""
        # Test with tuple response (new required format)
        response_instance = QualityAssessment(score=5, is_helpful=True, reasoning="Perfect")
        custom_metadata = {"cost_usd": 0.001, "tokens_used": 75, "model": "test-model"}
        def llm_function(p, rf):  # noqa
            return response_instance, custom_metadata

        result = await self.check(
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

        # Test invalid response format (not tuple)
        def invalid_llm_function(p, rf):  # noqa
            # Invalid - not tuple
            return QualityAssessment(score=3, is_helpful=False, reasoning="Not tuple")

        with pytest.raises(CheckExecutionError, match="llm_function must return tuple"):
            await self.check(
                prompt="test",
                response_format=QualityAssessment,
                llm_function=invalid_llm_function,
            )

    # Test execute method (handles template processing)

    @pytest.mark.asyncio
    async def test_execute_template_processing(self):
        """Test execute method processes templates correctly."""
        raw_arguments = {
            "prompt": "User asked: {{$.test_case.input.user_message}}, AI responded: {{$.output.value.response}}",  # noqa: E501
            "response_format": QualityAssessment,
            "llm_function": self.create_sync_llm_function({
                "score": 4,
                "is_helpful": True,
                "reasoning": "Good",
            }),
        }

        result = await self.check.execute(
            check_type="llm_judge",
            arguments=raw_arguments,
            context=self.context,
        )

        assert result.status == Status.COMPLETED
        assert result.check_type == "llm_judge"
        # Check structure with separate judge_metadata in results
        assert result.results["score"] == 4
        assert result.results["is_helpful"] is True
        assert result.results["reasoning"] == "Good"
        # Should also have metadata fields in separate judge_metadata
        assert "cost_usd" in result.results["judge_metadata"]
        assert "tokens_used" in result.results["judge_metadata"]

        # Verify template was processed
        resolved_prompt = result.resolved_arguments["prompt"]["value"]
        expected_prompt = "User asked: What is the capital of France?, AI responded: The capital of France is Paris."  # noqa: E501
        assert resolved_prompt == expected_prompt

    @pytest.mark.asyncio
    async def test_execute_complex_template_processing(self):
        """Test execute with complex JSONPath expressions in templates."""
        raw_arguments = {
            "prompt": """
            Evaluation Context:
            User Input: {{$.test_case.input}}
            Expected Quality: {{$.test_case.expected.quality_level}}
            AI Response: {{$.output.value.response}}
            Confidence: {{$.output.value.confidence}}
            Model: {{$.output.metadata.model}}
            Execution Time: {{$.output.metadata.execution_time_ms}}ms
            """,
            "response_format": DetailedEvaluation,
            "llm_function": self.create_sync_llm_function({
                "overall_score": 0.9,
                "categories": {"accuracy": 5},
                "issues": [],
                "recommendations": [],
                "metadata": {},
            }),
        }

        result = await self.check.execute(
            check_type="llm_judge",
            arguments=raw_arguments,
            context=self.context,
        )

        assert result.status == Status.COMPLETED
        resolved_prompt = result.resolved_arguments["prompt"]["value"]

        # Verify all placeholders were resolved
        assert "What is the capital of France?" in resolved_prompt
        assert "high" in resolved_prompt
        assert "The capital of France is Paris." in resolved_prompt
        assert "0.95" in resolved_prompt
        assert "gpt-4" in resolved_prompt
        assert "250ms" in resolved_prompt

    @pytest.mark.asyncio
    async def test_execute_json_serialization_in_templates(self):
        """Test template processing with complex objects that need JSON serialization."""
        # Test case with complex nested input
        complex_test_case = TestCase(
            id="complex_001",
            input={
                "conversation": [
                    {"role": "user", "message": "Hello"},
                    {"role": "assistant", "message": "Hi there!"},
                ],
                "metadata": {"priority": "high", "tags": ["greeting", "casual"]},
            },
            expected=["greeting", "polite", "concise"],
        )

        complex_output = Output(
            value={
                "response": "Hello! How can I help you today?",
                "analysis": {
                    "sentiment": "positive",
                    "topics": ["greeting", "assistance"],
                    "confidence_scores": {"sentiment": 0.95, "topics": 0.88},
                },
            },
        )

        complex_context = EvaluationContext(complex_test_case, complex_output)

        raw_arguments = {
            "prompt": """
            Conversation: {{$.test_case.input.conversation}}
            Expected Topics: {{$.test_case.expected}}
            AI Analysis: {{$.output.value.analysis}}
            """,
            "response_format": SimpleBooleanResult,
            "llm_function": self.create_sync_llm_function({"passed": True}),
        }

        result = await self.check.execute(
            check_type="llm_judge",
            arguments=raw_arguments,
            context=complex_context,
        )

        assert result.status == Status.COMPLETED
        resolved_prompt = result.resolved_arguments["prompt"]["value"]

        # Verify JSON serialization occurred
        assert '"role": "user"' in resolved_prompt
        assert '"message": "Hello"' in resolved_prompt
        assert '["greeting", "polite", "concise"]' in resolved_prompt
        assert '"sentiment": "positive"' in resolved_prompt

    @pytest.mark.asyncio
    async def test_execute_template_error_handling(self):
        """Test execute method error handling for invalid templates."""
        # Test invalid JSONPath in template
        raw_arguments = {
            "prompt": "Invalid path: {{$.nonexistent.path.here}}",
            "response_format": SimpleBooleanResult,
            "llm_function": self.create_sync_llm_function({"passed": True}),
        }

        result = await self.check.execute(
            check_type="llm_judge",
            arguments=raw_arguments,
            context=self.context,
        )

        assert result.status == Status.ERROR
        assert result.error is not None
        assert result.error.type == "jsonpath_error"
        assert "Error processing prompt template" in result.error.message
        assert "nonexistent.path.here" in result.error.message

    @pytest.mark.asyncio
    async def test_execute_no_template_processing(self):
        """Test execute method when no template processing is needed."""
        raw_arguments = {
            "prompt": "Simple prompt with no templates",
            "response_format": SimpleBooleanResult,
            "llm_function": self.create_sync_llm_function({"passed": True}),
        }

        result = await self.check.execute(
            check_type="llm_judge",
            arguments=raw_arguments,
            context=self.context,
        )

        assert result.status == Status.COMPLETED
        assert result.resolved_arguments["prompt"]["value"] == "Simple prompt with no templates"

    @pytest.mark.asyncio
    async def test_execute_empty_template_placeholders(self):
        """Test execute with template that has no placeholders to process."""
        raw_arguments = {
            "prompt": "Evaluate this response for quality and accuracy.",
            "response_format": QualityAssessment,
            "llm_function": self.create_sync_llm_function({
                "score": 3,
                "is_helpful": True,
                "reasoning": "OK",
            }),
        }

        result = await self.check.execute(
            check_type="llm_judge",
            arguments=raw_arguments,
            context=self.context,
        )

        assert result.status == Status.COMPLETED
        assert result.resolved_arguments["prompt"]["value"] == "Evaluate this response for quality and accuracy."  # noqa: E501

    @pytest.mark.asyncio
    async def test_execute_metadata_preservation(self):
        """Test execute method preserves all required metadata."""
        raw_arguments = {
            "prompt": "Test prompt",
            "response_format": SimpleBooleanResult,
            "llm_function": self.create_sync_llm_function({"passed": True}),
        }

        result = await self.check.execute(
            check_type="llm_judge",
            arguments=raw_arguments,
            context=self.context,
        )

        assert result.check_type == "llm_judge"
        assert result.check_version == "1.0.0"
        assert isinstance(result.evaluated_at, datetime)
        assert result.evaluated_at.tzinfo == UTC

    # Test end-to-end via evaluate function

    def test_llm_judge_via_evaluate_basic(self):
        """Test LLM judge via evaluate function with basic template."""
        def mock_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict[str, Any]]:  # noqa
            # Simple mock that returns based on prompt content
            if "Paris" in prompt:
                response = QualityAssessment(
                    score=5,
                    is_helpful=True,
                    reasoning="Accurate geographical answer",
                )
            else:
                response = QualityAssessment(score=1, is_helpful=False, reasoning="Incorrect answer")  # noqa: E501

            metadata = {"cost_usd": 0.002, "tokens_used": 100, "model": "test-model"}
            return response, metadata

        test_cases = [
            TestCase(
                id="test_001",
                input="What is the capital of France?",
                expected="Paris",
                checks=[
                    Check(
                        type=CheckType.LLM_JUDGE,
                        arguments={
                            "prompt": "Evaluate: User asked '{{$.test_case.input}}', AI answered '{{$.output.value}}'",  # noqa: E501
                            "response_format": QualityAssessment,
                            "llm_function": mock_llm_function,
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value="The capital of France is Paris."),
        ]

        results = evaluate(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.summary.error_test_cases == 0
        assert results.summary.skipped_test_cases == 0
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].status == Status.COMPLETED

        check_result = results.results[0].check_results[0]
        # Check structure with separate judge_metadata in check results
        assert check_result.results["score"] == 5
        assert check_result.results["is_helpful"] is True
        assert "Accurate geographical answer" in check_result.results["reasoning"]
        assert "cost_usd" in check_result.results["judge_metadata"]
        assert "tokens_used" in check_result.results["judge_metadata"]

    def test_llm_judge_via_evaluate_complex_jsonpath(self):
        """Test LLM judge via evaluate with complex JSONPath expressions."""
        def mock_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict[str, Any]]:  # noqa
            # Mock that evaluates based on confidence scores
            if "0.9" in prompt:  # High confidence
                response = SimpleBooleanResult(passed=True)
            else:
                response = SimpleBooleanResult(passed=False)
            metadata = {"cost_usd": 0.001, "tokens_used": 50, "model": "test-eval"}
            return response, metadata

        test_cases = [
            TestCase(
                id="test_001",
                input={"question": "Complex question", "domain": "science"},
                expected={"min_confidence": 0.8, "required_topics": ["science", "accuracy"]},
                metadata={"difficulty": "hard", "source": "expert_review"},
                checks=[
                    Check(
                        type=CheckType.LLM_JUDGE,
                        arguments={
                            "prompt": """
                            Question Domain: {{$.test_case.input.domain}}
                            Difficulty: {{$.test_case.metadata.difficulty}}
                            AI Confidence: {{$.output.value.confidence}}
                            Required Topics: {{$.test_case.expected.required_topics}}
                            Response Quality: {{$.output.metadata.quality_rating}}
                            """,
                            "response_format": SimpleBooleanResult,
                            "llm_function": mock_llm_function,
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(
                value={"answer": "Detailed scientific answer", "confidence": 0.92},
                metadata={"model": "expert-model", "quality_rating": "high"},
            ),
        ]

        results = evaluate(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].check_results[0].results["passed"] is True

        # Verify complex JSONPath resolution
        resolved_prompt = results.results[0].check_results[0].resolved_arguments["prompt"]["value"]
        assert "science" in resolved_prompt
        assert "hard" in resolved_prompt
        assert "0.92" in resolved_prompt
        assert "high" in resolved_prompt

    def test_llm_judge_via_evaluate_multiple_checks(self):
        """Test multiple LLM judge checks on same test case."""
        def quality_evaluator(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict[str, Any]]:  # noqa
            response = QualityAssessment(score=4, is_helpful=True, reasoning="Good quality")
            metadata = {"cost_usd": 0.003, "tokens_used": 200, "model": "quality-judge"}
            return response, metadata

        def boolean_evaluator(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict[str, Any]]:  # noqa
            response = SimpleBooleanResult(passed=True)
            metadata = {"cost_usd": 0.001, "tokens_used": 30, "model": "bool-judge"}
            return response, metadata

        test_cases = [
            TestCase(
                id="test_001",
                input="Test question",
                checks=[
                    Check(
                        type=CheckType.LLM_JUDGE,
                        arguments={
                            "prompt": "Rate quality: {{$.output.value}}",
                            "response_format": QualityAssessment,
                            "llm_function": quality_evaluator,
                        },
                    ),
                    Check(
                        type=CheckType.LLM_JUDGE,
                        arguments={
                            "prompt": "Is helpful: {{$.output.value}}",
                            "response_format": SimpleBooleanResult,
                            "llm_function": boolean_evaluator,
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value="Test answer"),
        ]

        results = evaluate(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert len(results.results[0].check_results) == 2

        # First check (quality assessment) - structure with separate judge_metadata
        quality_result = results.results[0].check_results[0]
        assert quality_result.results["score"] == 4
        assert quality_result.results["is_helpful"] is True

        # Second check (boolean result) - structure with separate judge_metadata
        boolean_result = results.results[0].check_results[1]
        assert boolean_result.results["passed"] is True

    def test_llm_judge_via_evaluate_error_handling(self):
        """Test error handling via evaluate function."""
        def failing_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict[str, Any]]:  # noqa
            raise RuntimeError("LLM service unavailable")

        test_cases = [
            TestCase(
                id="test_001",
                input="Test question",
                checks=[
                    Check(
                        type=CheckType.LLM_JUDGE,
                        arguments={
                            "prompt": "Evaluate: {{$.output.value}}",
                            "response_format": SimpleBooleanResult,
                            "llm_function": failing_llm_function,
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value="Test answer"),
        ]

        results = evaluate(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.error_test_cases == 1
        assert results.summary.completed_test_cases == 0
        assert results.results[0].status == Status.ERROR

        check_result = results.results[0].check_results[0]
        assert check_result.status == Status.ERROR
        assert check_result.error is not None
        assert "LLM service unavailable" in check_result.error.message

    # Test edge cases and special scenarios

    @pytest.mark.asyncio
    async def test_template_with_null_values(self):
        """Test template processing when JSONPath resolves to null values."""
        test_case_with_nulls = TestCase(
            id="test_null",
            input="question",
            expected=None,  # Null expected value
            metadata=None,   # Null metadata
        )

        output_with_nulls = Output(
            value={"response": "answer", "optional_field": None},
            metadata=None,
        )

        context = EvaluationContext(test_case_with_nulls, output_with_nulls)

        raw_arguments = {
            "prompt": "Expected: {{$.test_case.expected}}, Optional: {{$.output.value.optional_field}}, Meta: {{$.test_case.metadata}}",  # noqa: E501
            "response_format": SimpleBooleanResult,
            "llm_function": self.create_sync_llm_function({"passed": True}),
        }

        result = await self.check.execute(
            check_type="llm_judge",
            arguments=raw_arguments,
            context=context,
        )

        assert result.status == Status.COMPLETED
        resolved_prompt = result.resolved_arguments["prompt"]["value"]

        # Null values should be converted to empty strings
        assert "Expected: , Optional: , Meta: " in resolved_prompt

    @pytest.mark.asyncio
    async def test_template_with_special_characters(self):
        """Test template processing with special characters and unicode."""
        special_test_case = TestCase(
            id="test_special",
            input="Quelle est la capitale de la France? 🇫🇷",
            expected="Paris avec des accents: café, naïve, résumé",
        )

        special_output = Output(
            value="La capitale est Paris! 🏰 Très bien!",
        )

        context = EvaluationContext(special_test_case, special_output)

        raw_arguments = {
            "prompt": "Question: {{$.test_case.input}}\nAnswer: {{$.output.value}}\nExpected: {{$.test_case.expected}}",  # noqa: E501
            "response_format": SimpleBooleanResult,
            "llm_function": self.create_sync_llm_function({"passed": True}),
        }

        result = await self.check.execute(
            check_type="llm_judge",
            arguments=raw_arguments,
            context=context,
        )

        assert result.status == Status.COMPLETED
        resolved_prompt = result.resolved_arguments["prompt"]["value"]

        # Verify special characters are preserved
        assert "🇫🇷" in resolved_prompt
        assert "🏰" in resolved_prompt
        assert "café" in resolved_prompt
        assert "naïve" in resolved_prompt

    @pytest.mark.asyncio
    async def test_template_with_very_long_content(self):
        """Test template processing with very long content."""
        # Create very long input and output
        long_input = "This is a very long question. " * 1000  # ~30KB
        long_output = "This is a very long answer. " * 1000   # ~30KB

        long_test_case = TestCase(
            id="test_long",
            input=long_input,
            expected="short expected",
        )

        long_output_obj = Output(value=long_output)

        context = EvaluationContext(long_test_case, long_output_obj)

        raw_arguments = {
            "prompt": "Input length: {{$.test_case.input}}\nOutput length: {{$.output.value}}",
            "response_format": SimpleBooleanResult,
            "llm_function": self.create_sync_llm_function({"passed": True}),
        }

        result = await self.check.execute(
            check_type="llm_judge",
            arguments=raw_arguments,
            context=context,
        )

        assert result.status == Status.COMPLETED
        resolved_prompt = result.resolved_arguments["prompt"]["value"]

        # Verify content was substituted (check for presence of repeated text)
        assert "This is a very long question." in resolved_prompt
        assert "This is a very long answer." in resolved_prompt

    @pytest.mark.asyncio
    async def test_template_escaping_edge_cases(self):
        """Test template processing with malformed or edge case templates."""
        edge_cases = [
            "{{$.test_case.input}}",  # Single placeholder
            "{{$.test_case.input}} and {{$.output.value}}",  # Multiple placeholders
            "No placeholders here",  # No placeholders
            "{{$.output.value.nested.deeply.nested.field}}",  # Deep nesting (may fail)
            "Partial {{$.test_case.input incomplete",  # Malformed (should be ignored)
            "{{$.test_case.input}} {{$.test_case.input}}",  # Duplicate placeholders
        ]

        for template in edge_cases:
            raw_arguments = {
                "prompt": template,
                "response_format": SimpleBooleanResult,
                "llm_function": self.create_sync_llm_function({"passed": True}),
            }

            # These should either succeed or fail gracefully
            result = await self.check.execute(
                check_type="llm_judge",
                arguments=raw_arguments,
                context=self.context,
                )

            # Should either complete successfully or error gracefully
            assert result.status in [Status.COMPLETED, Status.ERROR]

            if result.status == Status.COMPLETED:
                # If successful, prompt should be processed
                assert "prompt" in result.resolved_arguments
                assert "value" in result.resolved_arguments["prompt"]

    @pytest.mark.asyncio
    async def test_new_tuple_format_with_metadata(self):
        """Test that the new tuple format properly returns response and metadata."""
        custom_metadata = {
            "cost_usd": 0.0045,
            "tokens_used": 250,
            "input_tokens": 180,
            "output_tokens": 70,
            "response_time_ms": 1200,
            "model_version": "gpt-4o-mini-test",
            "temperature": 0.1,
        }

        response_data = {"passed": True}
        llm_function = self.create_sync_llm_function(response_data, custom_metadata)

        result = await self.check(
            prompt="test prompt",
            response_format=SimpleBooleanResult,
            llm_function=llm_function,
        )

        # Verify structure includes response fields and separate judge_metadata
        assert result["passed"] is True
        assert result["judge_metadata"]["cost_usd"] == 0.0045
        assert result["judge_metadata"]["tokens_used"] == 250
        assert result["judge_metadata"]["input_tokens"] == 180
        assert result["judge_metadata"]["output_tokens"] == 70
        assert result["judge_metadata"]["response_time_ms"] == 1200
        assert result["judge_metadata"]["model_version"] == "gpt-4o-mini-test"
        assert result["judge_metadata"]["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_response_format_validation_edge_cases(self):
        """Test response format validation with edge cases."""
        # Test with minimal valid response
        minimal_response = {"passed": True}
        llm_function = self.create_sync_llm_function(minimal_response)

        result = await self.check(
            prompt="test",
            response_format=SimpleBooleanResult,
            llm_function=llm_function,
        )
        assert result["passed"] is True
        # Should have metadata fields in separate judge_metadata
        assert "cost_usd" in result["judge_metadata"]
        assert "tokens_used" in result["judge_metadata"]

        # Test with response containing extra fields (should be preserved)
        extra_fields_response = {"passed": True, "extra_field": "value", "another": 123}

        class ExtendedResult(BaseModel):
            passed: bool
            extra_field: str
            another: int

        llm_function = self.create_sync_llm_function(extra_fields_response)

        result = await self.check(
            prompt="test",
            response_format=ExtendedResult,
            llm_function=llm_function,
        )
        # Check structure includes all fields with separate judge_metadata
        assert result["passed"] is True
        assert result["extra_field"] == "value"
        assert result["another"] == 123
        # Should also have metadata fields in separate judge_metadata
        assert "cost_usd" in result["judge_metadata"]
        assert "tokens_used" in result["judge_metadata"]

    def test_concurrent_evaluations(self):
        """Test multiple concurrent LLM judge evaluations."""
        def mock_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict[str, Any]]:  # noqa
            # Return different scores based on test case ID
            if "test_001" in prompt:
                response = QualityAssessment(score=5, is_helpful=True, reasoning="Excellent")
            elif "test_002" in prompt:
                response = QualityAssessment(score=3, is_helpful=True, reasoning="Good")
            else:
                response = QualityAssessment(score=1, is_helpful=False, reasoning="Poor")

            metadata = {"cost_usd": 0.002, "tokens_used": 100, "model": "batch-judge"}
            return response, metadata

        test_cases = [
            TestCase(
                id="test_001",
                input="Easy question",
                checks=[
                    Check(
                        type=CheckType.LLM_JUDGE,
                        arguments={
                            "prompt": "Evaluate test case {{$.test_case.id}}: {{$.output.value}}",
                            "response_format": QualityAssessment,
                            "llm_function": mock_llm_function,
                        },
                    ),
                ],
            ),
            TestCase(
                id="test_002",
                input="Medium question",
                checks=[
                    Check(
                        type=CheckType.LLM_JUDGE,
                        arguments={
                            "prompt": "Evaluate test case {{$.test_case.id}}: {{$.output.value}}",
                            "response_format": QualityAssessment,
                            "llm_function": mock_llm_function,
                        },
                    ),
                ],
            ),
            TestCase(
                id="test_003",
                input="Hard question",
                checks=[
                    Check(
                        type=CheckType.LLM_JUDGE,
                        arguments={
                            "prompt": "Evaluate test case {{$.test_case.id}}: {{$.output.value}}",
                            "response_format": QualityAssessment,
                            "llm_function": mock_llm_function,
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value="Perfect answer"),
            Output(value="Okay answer"),
            Output(value="Weak answer"),
        ]

        results = evaluate(test_cases, outputs)

        assert results.summary.total_test_cases == 3
        assert results.summary.completed_test_cases == 3
        assert results.summary.error_test_cases == 0

        # Verify each test case got the expected score (structure with separate judge_metadata)
        scores = [result.check_results[0].results["score"] for result in results.results]
        assert scores == [5, 3, 1]
        is_helpful = [result.check_results[0].results["is_helpful"] for result in results.results]
        assert is_helpful == [True, True, False]
        reasoning = [result.check_results[0].results["reasoning"] for result in results.results]
        assert reasoning == [
            "Excellent",
            "Good",
            "Poor",
        ]


class TestCustomFunctionCheck:
    """Test Custom Function check implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.check = CustomFunctionCheck_v1_0_0()
        self.test_case = TestCase(
            id="test_001",
            input="What is the capital of France?",
            expected="Paris",
            metadata={"category": "geography"},
        )
        self.output = Output(
            value="The capital of France is Paris.",
            metadata={"model": "gpt-4"},
        )
        self.context = EvaluationContext(self.test_case, self.output)

    @pytest.mark.asyncio
    async def test_call_with_direct_callable(self):
        """Test with direct Python function."""
        def validator(text):  # noqa: ANN001, ANN202
            return "Paris" in text

        result = await self.check(
            validation_function=validator,
            function_args={"text": "The capital of France is Paris."},
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_call_with_lambda_string(self):
        """Test with lambda as string."""
        result = await self.check(
            validation_function="lambda text: 'Paris' in text",
            function_args={"text": "The capital of France is Paris."},
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_call_with_function_string(self):
        """Test with function definition as string."""
        func_str = """def validate(text, city):
    return city.lower() in text.lower()"""

        result = await self.check(
            validation_function=func_str,
            function_args={"text": "The capital of France is Paris.", "city": "Paris"},
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_call_with_async_function(self):
        """Test with async function."""
        async def async_validator(text):  # noqa: ANN001, ANN202
            import asyncio  # noqa: PLC0415
            await asyncio.sleep(0.001)
            return {"passed": "capital" in text.lower(), "async": True}

        result = await self.check(
            validation_function=async_validator,
            function_args={"text": "The capital of France is Paris."},
        )
        assert result["passed"] is True
        assert result["async"] is True

    @pytest.mark.asyncio
    async def test_call_with_async_function_string(self):
        """Test with async function definition as string."""
        func_str = """async def validate_async(text):
    import asyncio
    await asyncio.sleep(0.001)
    return {"passed": "Paris" in text, "async_string": True}"""

        result = await self.check(
            validation_function=func_str,
            function_args={"text": "The capital of France is Paris."},
        )
        assert result["passed"] is True
        assert result["async_string"] is True

    @pytest.mark.asyncio
    async def test_call_various_return_types(self):
        """Test functions returning different types."""
        # Boolean
        result = await self.check(
            validation_function=lambda _: True,
            function_args={"_": "dummy"},
        )
        assert result is True

        # Number
        result = await self.check(
            validation_function=lambda _: 0.85,
            function_args={"_": "dummy"},
        )
        assert result == 0.85

        # Dict
        result = await self.check(
            validation_function=lambda _: {"score": 42},
            function_args={"_": "dummy"},
        )
        assert result == {"score": 42}

        # None
        result = await self.check(
            validation_function=lambda _: None,
            function_args={"_": "dummy"},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_call_with_imports(self):
        """Test string function using imports."""
        func_str = """def validate(text):
    import re
    return bool(re.search(r'[Pp]aris', text))"""

        result = await self.check(
            validation_function=func_str,
            function_args={"text": "The capital of France is Paris."},
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_call_with_provided_imports(self):
        """Test string function using pre-provided imports."""
        func_str = """def validate(data):
    return json.loads(data)['valid']"""

        result = await self.check(
            validation_function=func_str,
            function_args={"data": '{"valid": true}'},
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_call_error_handling(self):
        """Test error handling."""
        # Invalid function type
        with pytest.raises(ValidationError, match="validation_function must be callable or string"):  # noqa: E501
            await self.check(validation_function=123, function_args={"x": "test"})

        # Invalid function_args type
        with pytest.raises(ValidationError, match="function_args must be a dictionary"):
            await self.check(validation_function=lambda _: True, function_args="not a dict")

        # Function execution error
        def failing_func(x) -> Never:  # noqa: ANN001, ARG001
            raise ValueError("Test error")

        with pytest.raises(CheckExecutionError, match="Error executing validation function.*Test error"):  # noqa: E501
            await self.check(validation_function=failing_func, function_args={"x": "test"})

        # Invalid string function
        with pytest.raises(ValidationError, match="Failed to create function from string"):
            await self.check(validation_function="invalid syntax (", function_args={"x": "test"})

    @pytest.mark.asyncio
    async def test_execute_missing_function_args(self):
        """Test execute method with missing function_args."""
        result = await self.check.execute(
            check_type="custom_function",
            arguments={
                "validation_function": lambda x: {"passed": True},  # noqa: ARG005
                # Missing function_args
            },
            context=self.context,
        )

        assert result.status == Status.ERROR
        assert result.error.type == "validation_error"
        assert "function_args is required" in result.error.message

    @pytest.mark.asyncio
    async def test_execute_invalid_function_args_type(self):
        """Test execute method with invalid function_args type."""
        result = await self.check.execute(
            check_type="custom_function",
            arguments={
                "validation_function": lambda x: {"passed": True},  # noqa: ARG005
                "function_args": "not a dict",  # Invalid type
            },
            context=self.context,
        )

        assert result.status == Status.ERROR
        assert result.error.type == "validation_error"
        assert "function_args must be a dictionary" in result.error.message

    @pytest.mark.asyncio
    async def test_execute_with_jsonpath(self):
        """Test execute method with JSONPath resolution."""
        def validator(actual, expected, test_id):  # noqa: ANN001, ANN202
            return {
                "passed": expected.lower() in actual.lower(),
                "test_id": test_id,
            }

        result = await self.check.execute(
            check_type="custom_function",
            arguments={
                "validation_function": validator,
                "function_args": {
                    "actual": "$.output.value",
                    "expected": "$.test_case.expected",
                    "test_id": "$.test_case.id",
                },
            },
            context=self.context,
        )

        assert result.status == Status.COMPLETED
        assert result.results["passed"] is True
        assert result.results["test_id"] == "test_001"

    def test_custom_function_via_evaluate(self):
        """Test end-to-end via evaluate function."""
        def keyword_checker(text, keywords):  # noqa: ANN001, ANN202
            found = [kw for kw in keywords if kw.lower() in text.lower()]
            return {
                "passed": len(found) >= len(keywords) // 2,
                "found": found,
                "coverage": len(found) / len(keywords),
            }

        test_cases = [
            TestCase(
                id="test_keywords",
                input="Explain machine learning",
                expected={"keywords": ["machine", "learning", "data"]},
                checks=[
                    Check(
                        type="custom_function",
                        arguments={
                            "validation_function": keyword_checker,
                            "function_args": {
                                "text": "$.output.value",
                                "keywords": "$.test_case.expected.keywords",
                            },
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value="Machine learning uses data and algorithms to learn patterns."),
        ]

        results = evaluate(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].check_results[0].results["passed"] is True
        assert results.results[0].check_results[0].results["coverage"] == 1.0

    def test_custom_function_parallel_execution_with_lambdas(self):
        """Test parallel execution with lambda functions to ensure no pickling issues."""
        # Test both lambda strings and direct lambdas in parallel execution
        test_cases = [
            TestCase(
                id=f"test_{i}",
                input=f"Test input {i}",
                expected=f"expected_{i}",
                checks=[
                    Check(
                        type="custom_function",
                        arguments={
                            # Lambda as string - should work in parallel
                            "validation_function": f"lambda x, expected: {{'passed': expected == 'expected_{i}' and '{i}' in x}}",  # noqa: E501
                            "function_args": {
                                "x": "$.test_case.input",
                                "expected": "$.test_case.expected",
                            },
                        },
                    ),
                ],
            )
            for i in range(5)
        ]

        outputs = [
            Output(value=f"Output for test {i}")
            for i in range(5)
        ]

        # This will use parallel execution for multiple test cases
        results = evaluate(test_cases, outputs)

        assert results.summary.total_test_cases == 5
        assert results.summary.completed_test_cases == 5
        assert results.summary.error_test_cases == 0

        # All should pass since the lambda checks that the index matches
        for i, result in enumerate(results.results):
            assert result.status == Status.COMPLETED
            assert result.check_results[0].status == Status.COMPLETED
            assert result.check_results[0].results["passed"] is True

    def test_custom_function_parallel_with_direct_lambdas(self):
        """Test parallel execution with direct Python lambdas (potential pickling issues)."""
        # Direct Python lambdas can have pickling issues in multiprocessing
        # Let's test if they work or if we need to document this limitation

        # Create lambdas that capture variables from scope
        validators = [
            lambda text: {"passed": "test 0" in text.lower()},
            lambda text: {"passed": "test 1" in text.lower()},
            lambda text: {"passed": "test 2" in text.lower()},
        ]

        test_cases = [
            TestCase(
                id=f"test_{i}",
                input=f"This is test {i}",
                expected=True,
                checks=[
                    Check(
                        type="custom_function",
                        arguments={
                            "validation_function": validators[i],
                            "function_args": {
                                "text": "$.test_case.input",
                            },
                        },
                    ),
                ],
            )
            for i in range(3)
        ]

        outputs = [
            Output(value=f"Output {i}")
            for i in range(3)
        ]

        # Direct lambdas work fine in parallel execution (no pickling issues)
        results = evaluate(test_cases, outputs)

        assert results.summary.total_test_cases == 3
        assert results.summary.completed_test_cases == 3
        assert results.summary.error_test_cases == 0

        for result in results.results:
            assert result.status == Status.COMPLETED
            assert result.check_results[0].status == Status.COMPLETED
            assert result.check_results[0].results["passed"] is True


def test_schema_implementation_consistency():
    """Test that each schema class has a properly registered implementation."""
    # Map schema classes to their check types
    from flex_evals import (  # noqa: PLC0415
        LLMJudgeCheck,
        SemanticSimilarityCheck,
        CustomFunctionCheck,
    )
    schema_to_check_type = {
        LLMJudgeCheck: CheckType.LLM_JUDGE,
        SemanticSimilarityCheck: CheckType.SEMANTIC_SIMILARITY,
        CustomFunctionCheck: CheckType.CUSTOM_FUNCTION,
    }

    registered_checks = list_registered_checks()

    for schema_class, check_type in schema_to_check_type.items():
        # Check schema has VERSION
        assert hasattr(schema_class, 'VERSION'), f"{schema_class.__name__} missing VERSION"

        # Check corresponding implementation is registered
        check_type_str = check_type.value
        assert check_type_str in registered_checks, f"No implementation registered for {check_type_str}"  # noqa: E501

        # Check version matches
        registered_version = registered_checks[check_type_str]["1.0.0"]["version"]
        assert registered_version == schema_class.VERSION, f"Version mismatch for {check_type_str}"
