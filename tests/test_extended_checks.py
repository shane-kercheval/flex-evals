"""Tests for extended check implementations."""

import asyncio
import pytest
from datetime import datetime, UTC

from flex_evals.schemas.test_case import TestCase
from flex_evals.schemas.output import Output
from flex_evals.checks.base import EvaluationContext
from flex_evals.checks.extended.semantic_similarity import SemanticSimilarityCheck
from flex_evals.checks.extended.llm_judge import LlmJudgeCheck
from flex_evals.exceptions import ValidationError, CheckExecutionError
from typing import Never


class TestSemanticSimilarityCheck:
    """Test SemanticSimilarity check implementation."""

    def create_context(self, test_case_data: dict | None = None, output_data: dict | None = None):
        """Helper to create evaluation context."""
        test_case = TestCase(
            id="test-1",
            input="test input",
            expected="expected output",
            **(test_case_data or {}),
        )

        # Handle output_data properly
        if output_data:
            value = output_data.get("value", "actual output")
            metadata = output_data.get("metadata", {"timestamp": datetime.now(UTC)})
        else:
            value = "actual output"
            metadata = {"timestamp": datetime.now(UTC)}

        output = Output(value=value, metadata=metadata)
        return EvaluationContext(test_case, output)

    async def simple_embedding_function(self, text: str) -> list[float]:
        """Simple mock embedding function that returns consistent vectors."""
        # Create simple embeddings based on text length and content
        if text == "hello":
            return [1.0, 0.0, 0.0]
        if text == "hello world":
            return [0.8, 0.6, 0.0]  # Similar to "hello" but not identical
        if text == "goodbye":
            return [0.0, 0.0, 1.0]  # Very different from "hello"
        if text == "":
            return [0.0, 0.0, 0.0]
        # Generate based on text properties
        length = len(text)
        vowels = sum(1 for c in text.lower() if c in "aeiou")
        consonants = length - vowels
        return [length / 10.0, vowels / 5.0, consonants / 10.0]

    def test_semantic_similarity_missing_text(self):
        """Test that missing text argument raises TypeError."""
        check = SemanticSimilarityCheck()

        with pytest.raises(TypeError):
            asyncio.run(check(
                reference="hello",
                embedding_function=self.simple_embedding_function,
            ))

    def test_semantic_similarity_missing_reference(self):
        """Test that missing reference argument raises TypeError."""
        check = SemanticSimilarityCheck()

        with pytest.raises(TypeError):
            asyncio.run(check(
                text="hello",
                embedding_function=self.simple_embedding_function,
            ))

    def test_semantic_similarity_missing_embedding_function(self):
        """Test that missing embedding_function raises TypeError."""
        check = SemanticSimilarityCheck()

        with pytest.raises(TypeError):
            asyncio.run(check(
                text="hello",
                reference="hello",
            ))

    def test_semantic_similarity_non_callable_embedding_function(self):
        """Test that non-callable embedding_function raises ValidationError."""
        check = SemanticSimilarityCheck()

        with pytest.raises(ValidationError, match="embedding_function must be callable"):
            asyncio.run(check(
                text="hello",
                reference="hello",
                embedding_function="not_callable",
            ))

    def test_semantic_similarity_invalid_metric(self):
        """Test that invalid similarity metric raises ValidationError."""
        check = SemanticSimilarityCheck()

        with pytest.raises(ValidationError, match="Unsupported similarity metric"):
            asyncio.run(check(
                text="hello",
                reference="hello",
                embedding_function=self.simple_embedding_function,
                similarity_metric="invalid_metric",
            ))

    @pytest.mark.asyncio
    async def test_semantic_similarity_basic_score(self):
        """Test that similarity check returns score between 0 and 1."""
        check = SemanticSimilarityCheck()
        context = self.create_context()

        result = await check(
            text="hello",
            reference="hello world",
            embedding_function=self.simple_embedding_function,
        )

        assert "score" in result
        assert 0 <= result["score"] <= 1
        assert "passed" not in result  # No threshold provided

    @pytest.mark.asyncio
    async def test_semantic_similarity_identical_texts(self):
        """Test that identical texts return high similarity score."""
        check = SemanticSimilarityCheck()
        context = self.create_context()

        result = await check({
            "text": "hello",
            "reference": "hello",
            "embedding_function": self.simple_embedding_function,
        }, context)

        assert result["score"] >= 0.99  # Should be very close to 1.0

    @pytest.mark.asyncio
    async def test_semantic_similarity_different_texts(self):
        """Test that very different texts return low similarity score."""
        check = SemanticSimilarityCheck()
        context = self.create_context()

        result = await check({
            "text": "hello",
            "reference": "goodbye",
            "embedding_function": self.simple_embedding_function,
        }, context)

        assert result["score"] <= 0.3  # Should be quite low

    @pytest.mark.asyncio
    async def test_semantic_similarity_with_threshold(self):
        """Test that threshold creates passed field."""
        check = SemanticSimilarityCheck()
        context = self.create_context()

        result = await check({
            "text": "hello",
            "reference": "hello",
            "embedding_function": self.simple_embedding_function,
            "threshold": {"min_value": 0.8},
        }, context)

        assert "score" in result
        assert "passed" in result
        assert isinstance(result["passed"], bool)

    @pytest.mark.asyncio
    async def test_semantic_similarity_threshold_pass(self):
        """Test that score above threshold passes."""
        check = SemanticSimilarityCheck()
        context = self.create_context()

        result = await check({
            "text": "hello",
            "reference": "hello",
            "embedding_function": self.simple_embedding_function,
            "threshold": {"min_value": 0.8},
        }, context)

        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_semantic_similarity_threshold_fail(self):
        """Test that score below threshold fails."""
        check = SemanticSimilarityCheck()
        context = self.create_context()

        result = await check({
            "text": "hello",
            "reference": "goodbye",
            "embedding_function": self.simple_embedding_function,
            "threshold": {"min_value": 0.8},
        }, context)

        assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_semantic_similarity_threshold_negate(self):
        """Test threshold negation logic."""
        check = SemanticSimilarityCheck()
        context = self.create_context()

        result = await check({
            "text": "hello",
            "reference": "hello",
            "embedding_function": self.simple_embedding_function,
            "threshold": {"min_value": 0.8, "negate": True},
        }, context)

        # High similarity with negation should fail
        assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_semantic_similarity_jsonpath_resolution(self):
        """Test JSONPath resolution for text and reference."""
        check = SemanticSimilarityCheck()
        context = self.create_context(
            output_data={"value": {"text1": "hello", "text2": "hello world"}},
        )

        # Arguments will be resolved by base class before reaching check
        result = await check({
            "text": "hello",  # Simulating resolved JSONPath
            "reference": "hello world",  # Simulating resolved JSONPath
            "embedding_function": self.simple_embedding_function,
        }, context)

        assert "score" in result
        assert 0 <= result["score"] <= 1

    @pytest.mark.asyncio
    async def test_semantic_similarity_embedding_error(self):
        """Test handling of embedding function errors."""
        async def failing_embedding_function(text: str) -> Never:
            raise RuntimeError("Embedding service unavailable")

        check = SemanticSimilarityCheck()
        context = self.create_context()

        with pytest.raises(CheckExecutionError, match="Error in semantic similarity computation"):
            await check({
                "text": "hello",
                "reference": "world",
                "embedding_function": failing_embedding_function,
            }, context)

    @pytest.mark.asyncio
    async def test_semantic_similarity_custom_metrics(self):
        """Test different similarity metrics."""
        check = SemanticSimilarityCheck()
        context = self.create_context()

        # Test dot product metric
        result_dot = await check({
            "text": "hello",
            "reference": "hello world",
            "embedding_function": self.simple_embedding_function,
            "similarity_metric": "dot",
        }, context)

        # Test euclidean metric
        result_euclidean = await check({
            "text": "hello",
            "reference": "hello world",
            "embedding_function": self.simple_embedding_function,
            "similarity_metric": "euclidean",
        }, context)

        assert "score" in result_dot
        assert "score" in result_euclidean
        assert 0 <= result_dot["score"] <= 1
        assert 0 <= result_euclidean["score"] <= 1

    def test_sync_embedding_function(self):
        """Test that sync embedding functions work correctly."""
        def sync_embedding_function(text: str) -> list[float]:
            return [1.0, 2.0, 3.0]

        check = SemanticSimilarityCheck()
        context = self.create_context()

        result = asyncio.run(check({
            "text": "hello",
            "reference": "world",
            "embedding_function": sync_embedding_function,
        }, context))

        assert "score" in result
        assert result["score"] == 1.0  # Identical embeddings should give score 1.0


class TestLlmJudgeCheck:
    """Test LlmJudge check implementation."""

    def create_context(self, test_case_data: dict | None = None, output_data: dict | None = None):
        """Helper to create evaluation context."""
        # Handle test_case_data properly to avoid conflicts
        test_case_kwargs = {
            "id": "test-1",
            "input": "test input",
            "expected": "expected output",
        }
        if test_case_data:
            test_case_kwargs.update(test_case_data)

        test_case = TestCase(**test_case_kwargs)

        # Handle output_data properly
        if output_data:
            value = output_data.get("value", "The answer is 42.")
            metadata = output_data.get("metadata", {"timestamp": datetime.now(UTC)})
        else:
            value = "The answer is 42."
            metadata = {"timestamp": datetime.now(UTC)}

        output = Output(value=value, metadata=metadata)
        return EvaluationContext(test_case, output)

    async def simple_llm_function(self, prompt: str) -> str:
        """Simple mock LLM function for testing."""
        if "rate the quality" in prompt.lower():
            return '{"quality": "high", "score": 8, "reasoning": "Well structured response"}'
        if "boolean" in prompt.lower():
            return '{"is_correct": true}'
        if "extract" in prompt.lower():
            return '{"entities": ["answer", "42"], "sentiment": "neutral"}'
        return '{"result": "processed"}'

    def test_llm_judge_missing_prompt(self):
        """Test that missing prompt argument raises ValidationError."""
        check = LlmJudgeCheck()
        context = self.create_context()

        with pytest.raises(ValidationError, match="requires 'prompt' argument"):
            asyncio.run(check({
                "response_format": {"type": "object"},
                "llm_function": self.simple_llm_function,
            }, context))

    def test_llm_judge_missing_response_format(self):
        """Test that missing response_format raises ValidationError."""
        check = LlmJudgeCheck()
        context = self.create_context()

        with pytest.raises(ValidationError, match="requires 'response_format' argument"):
            asyncio.run(check({
                "prompt": "Rate this response",
                "llm_function": self.simple_llm_function,
            }, context))

    def test_llm_judge_missing_llm_function(self):
        """Test that missing llm_function raises ValidationError."""
        check = LlmJudgeCheck()
        context = self.create_context()

        with pytest.raises(ValidationError, match="requires 'llm_function' argument"):
            asyncio.run(check({
                "prompt": "Rate this response",
                "response_format": {"type": "object"},
            }, context))

    def test_llm_judge_invalid_prompt_type(self):
        """Test that non-string prompt raises ValidationError."""
        check = LlmJudgeCheck()
        context = self.create_context()

        with pytest.raises(ValidationError, match="prompt must be a string"):
            asyncio.run(check({
                "prompt": 123,
                "response_format": {"type": "object"},
                "llm_function": self.simple_llm_function,
            }, context))

    def test_llm_judge_invalid_response_format_type(self):
        """Test that non-dict response_format raises ValidationError."""
        check = LlmJudgeCheck()
        context = self.create_context()

        with pytest.raises(ValidationError, match="response_format must be a dictionary"):
            asyncio.run(check({
                "prompt": "Rate this",
                "response_format": "invalid",
                "llm_function": self.simple_llm_function,
            }, context))

    def test_llm_judge_non_callable_llm_function(self):
        """Test that non-callable llm_function raises ValidationError."""
        check = LlmJudgeCheck()
        context = self.create_context()

        with pytest.raises(ValidationError, match="llm_function must be callable"):
            asyncio.run(check({
                "prompt": "Rate this",
                "response_format": {"type": "object"},
                "llm_function": "not_callable",
            }, context))

    @pytest.mark.asyncio
    async def test_llm_judge_prompt_templating(self):
        """Test JSONPath placeholder replacement in prompts."""
        check = LlmJudgeCheck()
        context = self.create_context(output_data={"value": "The answer is 42."})

        # Mock LLM function that returns the processed prompt for verification
        async def echo_llm_function(prompt: str) -> str:
            return f'{{"processed_prompt": "{prompt}"}}'

        result = await check({
            "prompt": "Please evaluate this output: {{$.output.value}}",
            "response_format": {"type": "object"},
            "llm_function": echo_llm_function,
        }, context)

        assert "processed_prompt" in result
        assert "The answer is 42." in result["processed_prompt"]

    @pytest.mark.asyncio
    async def test_llm_judge_multiple_placeholders(self):
        """Test multiple JSONPath placeholders in prompt."""
        check = LlmJudgeCheck()
        context = self.create_context(
            test_case_data={"expected": "42"},
            output_data={"value": "The answer is 42."},
        )

        async def echo_llm_function(prompt: str) -> str:
            return f'{{"processed_prompt": "{prompt}"}}'

        result = await check({
            "prompt": "Compare {{$.output.value}} with expected {{$.test_case.expected}}",
            "response_format": {"type": "object"},
            "llm_function": echo_llm_function,
        }, context)

        assert "The answer is 42." in result["processed_prompt"]
        assert "42" in result["processed_prompt"]

    @pytest.mark.asyncio
    async def test_llm_judge_boolean_response(self):
        """Test simple boolean response format."""
        check = LlmJudgeCheck()
        context = self.create_context()

        result = await check({
            "prompt": "Is this boolean correct?",
            "response_format": {
                "type": "object",
                "properties": {
                    "is_correct": {"type": "boolean"},
                },
                "required": ["is_correct"],
            },
            "llm_function": self.simple_llm_function,
        }, context)

        assert "is_correct" in result
        assert isinstance(result["is_correct"], bool)

    @pytest.mark.asyncio
    async def test_llm_judge_complex_response(self):
        """Test complex object response format."""
        check = LlmJudgeCheck()
        context = self.create_context()

        result = await check({
            "prompt": "Rate the quality of this response",
            "response_format": {
                "type": "object",
                "properties": {
                    "quality": {"type": "string"},
                    "score": {"type": "number"},
                    "reasoning": {"type": "string"},
                },
                "required": ["quality", "score"],
            },
            "llm_function": self.simple_llm_function,
        }, context)

        assert "quality" in result
        assert "score" in result
        assert "reasoning" in result
        assert isinstance(result["quality"], str)
        assert isinstance(result["score"], int | float)

    @pytest.mark.asyncio
    async def test_llm_judge_llm_function_error(self):
        """Test handling of LLM function errors."""
        async def failing_llm_function(prompt: str) -> Never:
            raise RuntimeError("LLM service unavailable")

        check = LlmJudgeCheck()
        context = self.create_context()

        with pytest.raises(CheckExecutionError, match="Error in LLM judge evaluation"):
            await check({
                "prompt": "Rate this response",
                "response_format": {"type": "object"},
                "llm_function": failing_llm_function,
            }, context)

    @pytest.mark.asyncio
    async def test_llm_judge_invalid_json_response(self):
        """Test handling of invalid JSON response from LLM."""
        async def invalid_json_llm_function(prompt: str) -> str:
            return "This is not valid JSON"

        check = LlmJudgeCheck()
        context = self.create_context()

        with pytest.raises(CheckExecutionError, match="Error in LLM judge evaluation"):
            await check({
                "prompt": "Rate this response",
                "response_format": {"type": "object"},
                "llm_function": invalid_json_llm_function,
            }, context)

    @pytest.mark.asyncio
    async def test_llm_judge_schema_validation_missing_required(self):
        """Test schema validation with missing required fields."""
        async def incomplete_llm_function(prompt: str) -> str:
            return '{"quality": "high"}'  # Missing required "score" field

        check = LlmJudgeCheck()
        context = self.create_context()

        with pytest.raises(CheckExecutionError, match="Error in LLM judge evaluation"):
            await check({
                "prompt": "Rate this response",
                "response_format": {
                    "type": "object",
                    "properties": {
                        "quality": {"type": "string"},
                        "score": {"type": "number"},
                    },
                    "required": ["quality", "score"],
                },
                "llm_function": incomplete_llm_function,
            }, context)

    def test_sync_llm_function(self):
        """Test that sync LLM functions work correctly."""
        def sync_llm_function(prompt: str) -> str:
            return '{"result": "sync_response"}'

        check = LlmJudgeCheck()
        context = self.create_context()

        result = asyncio.run(check({
            "prompt": "Process this synchronously",
            "response_format": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            },
            "llm_function": sync_llm_function,
        }, context))

        assert result["result"] == "sync_response"
