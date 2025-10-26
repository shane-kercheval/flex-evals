"""Tests for Output schema implementation."""

import dataclasses
from typing import Any

from pydantic import BaseModel
from flex_evals import Output


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

    def test_output_allows_none_value(self):
        """Test None value is now allowed."""
        output = Output(value=None)
        assert output.value is None
        assert output.id is None
        assert output.metadata is None

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

        # Convert to dict for serialization using dataclasses.asdict
        data = dataclasses.asdict(output)

        assert data["value"]["answer"] == "Paris"
        assert data["value"]["confidence"] == 0.95
        assert data["metadata"]["execution_time_ms"] == 245
        assert data["metadata"]["model_version"] == "gpt-4-turbo"

        # Also test using to_dict() method
        data2 = output.to_dict()
        assert data2["value"]["answer"] == "Paris"
        assert data2["value"]["confidence"] == 0.95
        assert data2["metadata"]["execution_time_ms"] == 245
        assert data2["metadata"]["model_version"] == "gpt-4-turbo"

    def test_output_serialization__pydantic_value(self):
        """Test Output can be converted to dict for JSON serialization."""
        class MyPydanticModel(BaseModel):
            field_a: str
            field_b: int

        output = Output(
            value=MyPydanticModel(field_a="test", field_b=123),
        )

        # Convert to dict for serialization
        data = dataclasses.asdict(output)
        # this won't convert to dict, but we still test that asdict works with pydantic models
        assert isinstance(data["value"], MyPydanticModel)

        # now test that we can convert the pydantic model to dict via to_dict()
        data = output.to_dict()
        assert isinstance(data["value"], dict)
        assert data["value"]["field_a"] == "test"
        assert data["value"]["field_b"] == 123

    def test_output_serialization__dict_method(self):
        """Test Output with object that has .dict() method."""
        class MyClassWithDict:
            def __init__(self, field_a: str, field_b: int):
                self.field_a = field_a
                self.field_b = field_b

            def dict(self) -> dict[str, Any]:
                return {"field_a": self.field_a, "field_b": self.field_b}

        output = Output(
            value=MyClassWithDict(field_a="test", field_b=456),
        )

        # Convert to dict for serialization
        data = dataclasses.asdict(output)
        # this won't convert to dict, but we still test that asdict works
        assert isinstance(data["value"], MyClassWithDict)

        # now test that we can convert the object to dict via to_dict()
        data = output.to_dict()
        assert isinstance(data["value"], dict)
        assert data["value"]["field_a"] == "test"
        assert data["value"]["field_b"] == 456

    def test_output_serialization__to_dict_method(self):
        """Test Output with object that has .to_dict() method."""
        class MyClassWithToDict:
            def __init__(self, field_a: str, field_b: int):
                self.field_a = field_a
                self.field_b = field_b

            def to_dict(self) -> dict[str, Any]:
                return {"field_a": self.field_a, "field_b": self.field_b}

        output = Output(
            value=MyClassWithToDict(field_a="test", field_b=789),
        )

        # Convert to dict for serialization
        data = dataclasses.asdict(output)
        # this won't convert to dict, but we still test that asdict works
        assert isinstance(data["value"], MyClassWithToDict)

        # now test that we can convert the object to dict via to_dict()
        data = output.to_dict()
        assert isinstance(data["value"], dict)
        assert data["value"]["field_a"] == "test"
        assert data["value"]["field_b"] == 789

    def test_output_serialization__dataclass_value(self):
        """Test Output with dataclass instance as value."""
        @dataclasses.dataclass
        class MyDataclass:
            field_a: str
            field_b: int
            field_c: float = 3.14

        output = Output(
            value=MyDataclass(field_a="test", field_b=999, field_c=2.71),
        )

        # Convert to dict for serialization
        data = dataclasses.asdict(output)
        # dataclasses.asdict should handle nested dataclasses
        assert isinstance(data["value"], dict)
        assert data["value"]["field_a"] == "test"
        assert data["value"]["field_b"] == 999
        assert data["value"]["field_c"] == 2.71

        # now test that we can convert the dataclass to dict via to_dict()
        data = output.to_dict()
        assert isinstance(data["value"], dict)
        assert data["value"]["field_a"] == "test"
        assert data["value"]["field_b"] == 999
        assert data["value"]["field_c"] == 2.71
