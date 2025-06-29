"""Tests for Output schema implementation."""

import dataclasses
import pytest
from flex_evals.schemas import Output


class TestOutput:
    """Test Output schema implementation."""

    def test_output_required_value(self):
        """Test Output with required value field."""
        output = Output(value="The capital of France is Paris.")

        assert output.value == "The capital of France is Paris."
        assert output.metadata is None

    def test_output_string_value(self):
        """Test Output with simple string value."""
        output = Output(value="Simple string response")

        assert isinstance(output.value, str)
        assert output.value == "Simple string response"

    def test_output_object_value(self):
        """Test Output with complex Dict value."""
        value_data = {
            "response": "Paris",
            "confidence": 0.95,
            "reasoning": "Based on geographical knowledge...",
            "trace": {
                "steps": ["knowledge_lookup", "confidence_calculation"],
                "execution_time": "245ms",
            },
        }
        output = Output(value=value_data)

        assert isinstance(output.value, dict)
        assert output.value == value_data
        assert output.value["confidence"] == 0.95
        assert len(output.value["trace"]["steps"]) == 2

    def test_output_metadata_optional(self):
        """Test Output with and without metadata."""
        # Without metadata
        output1 = Output(value="test response")
        assert output1.metadata is None

        # With metadata
        metadata = {
            "execution_time_ms": 245,
            "model_version": "gpt-4-turbo",
            "cost_usd": 0.0023,
            "tokens_used": 150,
        }
        output2 = Output(value="test response", metadata=metadata)
        assert output2.metadata == metadata
        assert output2.metadata["execution_time_ms"] == 245

    def test_output_validation_error(self):
        """Test None value raises ValidationError."""
        with pytest.raises(ValueError, match="Output.value is required and cannot be None"):
            Output(value=None)

    def test_output_nested_value(self):
        """Test deeply nested object values."""
        nested_value = {
            "result": {
                "primary": {
                    "answer": "Paris",
                    "details": {
                        "country": "France",
                        "continent": "Europe",
                        "population": 2165000,
                        "coordinates": {
                            "lat": 48.8566,
                            "lng": 2.3522,
                        },
                    },
                },
                "alternatives": ["Lyon", "Marseille"],
                "metadata": {
                    "sources": ["wikipedia", "geonames"],
                    "confidence_scores": {
                        "paris": 0.95,
                        "lyon": 0.02,
                        "marseille": 0.01,
                    },
                },
            },
        }

        output = Output(value=nested_value)

        assert output.value["result"]["primary"]["answer"] == "Paris"
        assert output.value["result"]["primary"]["details"]["coordinates"]["lat"] == 48.8566
        assert len(output.value["result"]["alternatives"]) == 2
        assert output.value["result"]["metadata"]["confidence_scores"]["paris"] == 0.95

    def test_output_invalid_value_type(self):
        """Test invalid value types raise error."""
        with pytest.raises(ValueError, match="Output.value must be a string or dictionary"):
            Output(value=123)

        with pytest.raises(ValueError, match="Output.value must be a string or dictionary"):
            Output(value=["list", "not", "allowed"])

        with pytest.raises(ValueError, match="Output.value must be a string or dictionary"):
            Output(value=True)

    def test_output_serialization(self):
        """Test Output can be converted to dict for JSON serialization."""
        output = Output(
            value={
                "answer": "Paris",
                "confidence": 0.95,
            },
            metadata={
                "execution_time_ms": 245,
                "model_version": "gpt-4-turbo",
            },
        )

        # Convert to dict for serialization
        data = dataclasses.asdict(output)

        assert data["value"]["answer"] == "Paris"
        assert data["value"]["confidence"] == 0.95
        assert data["metadata"]["execution_time_ms"] == 245
        assert data["metadata"]["model_version"] == "gpt-4-turbo"
