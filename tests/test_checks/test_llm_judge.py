"""
Comprehensive tests for LLMJudgeCheck implementation.

This module consolidates all tests for the LLMJudgeCheck including:
- Pydantic validation tests
- Implementation execution tests
- Engine integration tests
- Edge cases and error handling
- Template processing functionality

Tests are organized by functionality rather than implementation details.
"""

import pytest
from pydantic import BaseModel

from flex_evals import (
    LLMJudgeCheck,
    JSONPath,
    CheckType,
    Status,
    evaluate,
    Check,
    Output,
    TestCase,
    CheckExecutionError,
)
from pydantic import ValidationError as PydanticValidationError


# Test response format models
class JudgeResponse(BaseModel):
    """Basic response format for LLM judge tests."""

    passed: bool
    confidence: float
    reasoning: str


class ComplexJudgeResponse(BaseModel):
    """Complex response format with nested fields."""

    passed: bool
    confidence: float
    reasoning: str
    categories: dict[str, bool]
    score: int


class TestLLMJudgeValidation:
    """Test Pydantic validation and field handling for LLMJudgeCheck."""

    def test_llm_judge_check_creation(self):
        """Test basic LLMJudgeCheck creation."""
        def mock_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:  # noqa: ARG001, E501
            return JudgeResponse(passed=True, confidence=0.9, reasoning="Good"), {}

        check = LLMJudgeCheck(
            prompt="Evaluate this: {{$.output.value}}",
            response_format=JudgeResponse,
            llm_function=mock_llm_function,
        )

        assert check.prompt == "Evaluate this: {{$.output.value}}"
        assert check.response_format == JudgeResponse
        assert check.llm_function == mock_llm_function

    def test_llm_judge_check_with_jsonpath_prompt(self):
        """Test LLMJudgeCheck with JSONPath prompt."""
        def mock_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:  # noqa: ARG001, E501
            return JudgeResponse(passed=True, confidence=0.9, reasoning="Good"), {}

        check = LLMJudgeCheck(
            prompt="$.test_case.input",
            response_format=JudgeResponse,
            llm_function=mock_llm_function,
        )

        assert isinstance(check.prompt, JSONPath)
        assert check.prompt.expression == "$.test_case.input"

    def test_llm_judge_check_type_property(self):
        """Test LLMJudgeCheck check_type property returns correct type."""
        def mock_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:  # noqa: ARG001, E501
            return JudgeResponse(passed=True, confidence=0.9, reasoning="Good"), {}

        check = LLMJudgeCheck(
            prompt="Test prompt",
            response_format=JudgeResponse,
            llm_function=mock_llm_function,
        )
        assert check.check_type == CheckType.LLM_JUDGE

    def test_llm_judge_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(PydanticValidationError):
            LLMJudgeCheck()  # type: ignore

        with pytest.raises(PydanticValidationError):
            LLMJudgeCheck(prompt="test")  # type: ignore

        with pytest.raises(PydanticValidationError):
            LLMJudgeCheck(response_format=JudgeResponse)  # type: ignore


class TestLLMJudgeExecution:
    """Test LLMJudgeCheck execution logic and __call__ method."""

    @pytest.mark.asyncio
    async def test_call_basic_functionality(self):
        """Test basic LLM judge evaluation."""
        def mock_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:  # noqa: ARG001, E501
            return JudgeResponse(passed=True, confidence=0.9, reasoning="Good response"), {"model": "test"}  # noqa: E501

        check = LLMJudgeCheck(
            prompt="Evaluate this response",
            response_format=JudgeResponse,
            llm_function=mock_llm_function,
        )

        result = await check()
        assert result == {
            "passed": True,
            "confidence": 0.9,
            "reasoning": "Good response",
            "judge_metadata": {"model": "test"},
        }

    @pytest.mark.asyncio
    async def test_call_with_async_llm_function(self):
        """Test LLM judge with async LLM function."""
        async def mock_async_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:  # noqa: ARG001, E501
            return JudgeResponse(passed=False, confidence=0.7, reasoning="Needs improvement"), {"async": True}  # noqa: E501

        check = LLMJudgeCheck(
            prompt="Evaluate this response",
            response_format=JudgeResponse,
            llm_function=mock_async_llm_function,
        )

        result = await check()
        assert result == {
            "passed": False,
            "confidence": 0.7,
            "reasoning": "Needs improvement",
            "judge_metadata": {"async": True},
        }

    @pytest.mark.asyncio
    async def test_call_with_complex_response_format(self):
        """Test LLM judge with complex response format."""
        def mock_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:  # noqa: ARG001, E501
            return ComplexJudgeResponse(
                passed=True,
                confidence=0.8,
                reasoning="Detailed evaluation",
                categories={"grammar": True, "relevance": False},
                score=85,
            ), {"tokens": 150}

        check = LLMJudgeCheck(
            prompt="Detailed evaluation",
            response_format=ComplexJudgeResponse,
            llm_function=mock_llm_function,
        )

        result = await check()
        assert result == {
            "passed": True,
            "confidence": 0.8,
            "reasoning": "Detailed evaluation",
            "categories": {"grammar": True, "relevance": False},
            "score": 85,
            "judge_metadata": {"tokens": 150},
        }

    @pytest.mark.asyncio
    async def test_call_argument_validation(self):
        """Test validation of arguments in __call__ method."""
        check = LLMJudgeCheck(
            prompt=JSONPath(expression="$.unresolved"),  # Should cause RuntimeError
            response_format=JudgeResponse,
            llm_function=lambda p, r: (JudgeResponse(passed=True, confidence=0.9, reasoning="test"), {}),  # noqa: ARG005, E501
        )

        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'prompt' field"):
            await check()

    @pytest.mark.asyncio
    async def test_call_llm_function_failure(self):
        """Test handling of LLM function failures."""
        def failing_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:  # noqa: ARG001, E501
            raise Exception("LLM API error")

        check = LLMJudgeCheck(
            prompt="Test prompt",
            response_format=JudgeResponse,
            llm_function=failing_llm_function,
        )

        with pytest.raises(CheckExecutionError, match="Error in LLM judge evaluation"):
            await check()

    @pytest.mark.asyncio
    async def test_call_response_validation_errors(self):
        """Test validation of LLM responses."""
        def invalid_response_llm_function(prompt: str, response_format: type[BaseModel]) -> str:  # noqa: ARG001
            # Return wrong type instead of tuple
            return "invalid response"

        check = LLMJudgeCheck(
            prompt="Test prompt",
            response_format=JudgeResponse,
            llm_function=invalid_response_llm_function,
        )

        with pytest.raises(CheckExecutionError, match="llm_function must return tuple"):
            await check()

    @pytest.mark.asyncio
    async def test_call_response_format_handling(self):
        """Test different response format handling."""
        def dict_response_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[dict, dict]:  # noqa: ARG001, E501
            return {"passed": True, "confidence": 0.8, "reasoning": "Dict response"}, {"format": "dict"}  # noqa: E501

        check = LLMJudgeCheck(
            prompt="Test prompt",
            response_format=JudgeResponse,
            llm_function=dict_response_llm_function,
        )

        result = await check()
        assert result == {
            "passed": True,
            "confidence": 0.8,
            "reasoning": "Dict response",
            "judge_metadata": {"format": "dict"},
        }


class TestLLMJudgeEngineIntegration:
    """Test LLMJudgeCheck integration with the evaluation engine."""

    def test_llm_judge_via_evaluate(self):
        """Test LLM judge through engine evaluation."""
        def mock_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:  # noqa: ARG001, E501
            # Simple evaluation based on content
            return JudgeResponse(
                passed=True,
                confidence=0.9,
                reasoning="Response contains good content",
            ), {"model": "test-judge"}

        test_cases = [
            TestCase(
                id="test_001",
                input="test input",
                checks=[
                    Check(
                        type=CheckType.LLM_JUDGE,
                        arguments={
                            "prompt": "Evaluate: {{$.output.value}}",
                            "response_format": JudgeResponse,
                            "llm_function": mock_llm_function,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="Good quality response")]
        results = evaluate(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.summary.error_test_cases == 0
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results["passed"] is True

    def test_llm_judge_check_instance_via_evaluate(self):
        """Test direct check instance usage in evaluate function."""
        def mock_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:  # noqa: ARG001, E501
            return JudgeResponse(
                passed=False,
                confidence=0.8,
                reasoning="Response needs improvement",
            ), {"tokens_used": 50}

        test_cases = [
            TestCase(
                id="test_001",
                input="test input",
                checks=[
                    LLMJudgeCheck(
                        prompt="Judge this output: {{$.output.value}}",
                        response_format=JudgeResponse,
                        llm_function=mock_llm_function,
                    ),
                ],
            ),
        ]

        outputs = [Output(value="Poor quality response")]
        results = evaluate(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results["passed"] is False
        assert results.results[0].check_results[0].results["confidence"] == 0.8

    def test_llm_judge_template_via_evaluate(self):
        """Test template processing through engine evaluation."""
        def mock_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:  # noqa: ARG001, E501
            # Check that template was processed correctly
            assert "test input" in prompt
            assert "test response" in prompt
            return JudgeResponse(
                passed=True,
                confidence=0.95,
                reasoning="Template processed correctly",
            ), {"template": "processed"}

        test_cases = [
            TestCase(
                id="test_001",
                input="test input",
                checks=[
                    Check(
                        type=CheckType.LLM_JUDGE,
                        arguments={
                            "prompt": "Compare input '{{$.test_case.input}}' with output '{{$.output.value}}'",  # noqa: E501
                            "response_format": JudgeResponse,
                            "llm_function": mock_llm_function,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="test response")]
        results = evaluate(test_cases, outputs)

        assert results.results[0].check_results[0].results["passed"] is True


class TestLLMJudgeErrorHandling:
    """Test error handling and edge cases for LLMJudgeCheck."""

    def test_llm_judge_llm_function_error_in_engine(self):
        """Test LLM function errors through engine."""
        def failing_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:  # noqa: ARG001, E501
            raise Exception("API rate limit exceeded")

        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.LLM_JUDGE,
                        arguments={
                            "prompt": "Test prompt",
                            "response_format": JudgeResponse,
                            "llm_function": failing_llm_function,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="test")]
        results = evaluate(test_cases, outputs)

        # Should result in error when LLM function fails
        assert results.results[0].status == Status.ERROR
        assert results.results[0].check_results[0].status == Status.ERROR

    def test_llm_judge_template_processing_error_in_engine(self):
        """Test template processing errors through engine."""
        def mock_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:  # noqa: ARG001, E501
            return JudgeResponse(passed=True, confidence=0.9, reasoning="test"), {}

        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.LLM_JUDGE,
                        arguments={
                            "prompt": "Invalid template: {{$.invalid..path}}",  # Invalid JSONPath
                            "response_format": JudgeResponse,
                            "llm_function": mock_llm_function,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="test")]
        results = evaluate(test_cases, outputs)

        # Should result in error when template processing fails
        assert results.results[0].status == Status.ERROR

    @pytest.mark.asyncio
    async def test_llm_judge_invalid_response_format_error(self):
        """Test invalid response format handling."""
        def invalid_format_llm_function(prompt: str, response_format: type[BaseModel]) -> str:  # noqa: ARG001
            # Return invalid format - should be tuple[BaseModel, dict]
            return "not a BaseModel instance"

        check = LLMJudgeCheck(
            prompt="Test prompt",
            response_format=JudgeResponse,
            llm_function=invalid_format_llm_function,
        )

        with pytest.raises(CheckExecutionError):
            await check()

    def test_llm_judge_missing_jsonpath_data_in_template(self):
        """Test behavior when template JSONPath doesn't find data."""
        def mock_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:  # noqa: ARG001, E501
            return JudgeResponse(passed=True, confidence=0.9, reasoning="test"), {}

        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.LLM_JUDGE,
                        arguments={
                            "prompt": "Evaluate: {{$.output.value.nonexistent}}",
                            "response_format": JudgeResponse,
                            "llm_function": mock_llm_function,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value={"response": "test"})]
        results = evaluate(test_cases, outputs)

        # Should result in error when JSONPath resolution fails in template
        assert results.results[0].status == Status.ERROR


class TestLLMJudgeTemplateProcessing:
    """Test template processing functionality in LLMJudgeCheck."""

    def test_llm_judge_complex_template(self):
        """Test complex template with multiple JSONPath expressions."""
        def mock_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:  # noqa: ARG001, E501
            # Verify complex template was processed correctly
            expected_fragments = ["test input", "test response", "metadata_value"]
            for fragment in expected_fragments:
                assert fragment in prompt
            return JudgeResponse(passed=True, confidence=0.9, reasoning="Complex template"), {}

        test_cases = [
            TestCase(
                id="test_001",
                input="test input",
                expected={"metadata": "metadata_value"},
                checks=[
                    Check(
                        type=CheckType.LLM_JUDGE,
                        arguments={
                            "prompt": "Input: {{$.test_case.input}}, Output: {{$.output.value}}, Meta: {{$.test_case.expected.metadata}}",  # noqa: E501
                            "response_format": JudgeResponse,
                            "llm_function": mock_llm_function,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="test response")]
        results = evaluate(test_cases, outputs)

        assert results.results[0].check_results[0].results["passed"] is True

    def test_llm_judge_template_with_complex_data(self):
        """Test template with complex nested data structures."""
        def mock_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:  # noqa: ARG001, E501
            # Template should contain JSON-serialized complex data
            assert '{"name": "Alice", "role": "admin"}' in prompt  # JSON serialized dict
            assert '[95, 87, 92]' in prompt  # JSON serialized list
            return JudgeResponse(passed=True, confidence=0.9, reasoning="Complex data"), {}

        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.LLM_JUDGE,
                        arguments={
                            "prompt": "Evaluate user: {{$.output.value.user}} with scores: {{$.output.value.scores}}",  # noqa: E501
                            "response_format": JudgeResponse,
                            "llm_function": mock_llm_function,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value={
            "user": {"name": "Alice", "role": "admin"},
            "scores": [95, 87, 92],
        })]
        results = evaluate(test_cases, outputs)

        assert results.results[0].check_results[0].results["passed"] is True

    def test_llm_judge_no_template_placeholders(self):
        """Test LLM judge with no template placeholders."""
        def mock_llm_function(prompt: str, response_format: type[BaseModel]) -> tuple[BaseModel, dict]:  # noqa: ARG001, E501
            assert prompt == "Static prompt with no placeholders"
            return JudgeResponse(passed=True, confidence=0.9, reasoning="No templates"), {}

        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.LLM_JUDGE,
                        arguments={
                            "prompt": "Static prompt with no placeholders",
                            "response_format": JudgeResponse,
                            "llm_function": mock_llm_function,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="test")]
        results = evaluate(test_cases, outputs)

        assert results.results[0].check_results[0].results["passed"] is True
