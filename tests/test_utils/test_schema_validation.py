"""
Comprehensive tests for schema validation utilities.

This module tests the validate_json_schema function with all combinations of:
- Schema types: dict, str (JSON), BaseModel class
- Data types: dict, str (JSON), BaseModel instance

Tests cover validation success/failure cases, error handling, and edge cases.
"""

import json
from typing import Any
import pytest
from pydantic import BaseModel, Field

from flex_evals.utils.schema_validation import validate_json_schema, _coerce


class PersonModel(BaseModel):
    """Test Pydantic model for schema validation tests."""

    model_config = {"extra": "forbid"}  # Equivalent to additionalProperties: false

    name: str
    age: int = Field(ge=0, le=120)  # Equivalent to minimum: 0, maximum: 120


class PersonData(BaseModel):
    """Test Pydantic model for data validation tests (no constraints for flexibility)."""

    name: str
    age: int


# Test data fixtures
VALID_SCHEMA_DICT = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0, "maximum": 120},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}

VALID_DATA_DICT = {"name": "Alice", "age": 30}
INVALID_DATA_DICT = {"name": "Bob", "age": -1}  # Violates minimum age
MISSING_REQUIRED_DATA_DICT = {"name": "Charlie"}  # Missing age

VALID_SCHEMA_JSON = json.dumps(VALID_SCHEMA_DICT)
VALID_DATA_JSON = json.dumps(VALID_DATA_DICT)
INVALID_DATA_JSON = json.dumps(INVALID_DATA_DICT)

VALID_DATA_MODEL = PersonData(name="Alice", age=30)
INVALID_DATA_MODEL = PersonData(name="Bob", age=-1)  # This is allowed in PersonData


class TestCoerceFunction:
    """Test the _coerce helper function with various input types."""

    def test_coerce_dict_as_schema(self) -> None:
        """Test coercing dictionary as schema."""
        result = _coerce(VALID_SCHEMA_DICT, as_schema=True)
        assert result == VALID_SCHEMA_DICT

    def test_coerce_dict_as_data(self) -> None:
        """Test coercing dictionary as data."""
        result = _coerce(VALID_DATA_DICT, as_schema=False)
        assert result == VALID_DATA_DICT

    def test_coerce_json_string_as_schema(self) -> None:
        """Test coercing JSON string as schema."""
        result = _coerce(VALID_SCHEMA_JSON, as_schema=True)
        assert result == VALID_SCHEMA_DICT

    def test_coerce_json_string_as_data(self) -> None:
        """Test coercing JSON string as data."""
        result = _coerce(VALID_DATA_JSON, as_schema=False)
        assert result == VALID_DATA_DICT

    def test_coerce_basemodel_as_schema(self) -> None:
        """Test coercing BaseModel instance as schema (extracts model schema)."""
        result = _coerce(VALID_DATA_MODEL, as_schema=True)
        assert isinstance(result, dict)
        assert "properties" in result
        assert "name" in result["properties"]
        assert "age" in result["properties"]

    def test_coerce_basemodel_as_data(self) -> None:
        """Test coercing BaseModel instance as data (extracts model data)."""
        result = _coerce(VALID_DATA_MODEL, as_schema=False)
        assert result == VALID_DATA_DICT

    def test_coerce_invalid_json_string(self) -> None:
        """Test coercing invalid JSON string raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            _coerce("{invalid json", as_schema=True)

    def test_coerce_non_mapping_as_schema_raises_error(self) -> None:
        """Test that non-mapping values as schema raise TypeError."""
        with pytest.raises(TypeError, match="Schema must be a JSON object"):
            _coerce("[1, 2, 3]", as_schema=True)

        with pytest.raises(TypeError, match="Schema must be a JSON object"):
            _coerce("42", as_schema=True)

    def test_coerce_non_mapping_as_data_allowed(self) -> None:
        """Test that non-mapping values as data are allowed."""
        result = _coerce("[1, 2, 3]", as_schema=False)
        assert result == [1, 2, 3]

        result = _coerce("42", as_schema=False)
        assert result == 42


class TestValidateJsonSchemaAllCombinations:
    """Test validate_json_schema with all combinations of schema and data types."""

    @pytest.mark.parametrize(("schema", "data", "expected_valid"), [
        # Valid combinations - all schema types with valid data
        (VALID_SCHEMA_DICT, VALID_DATA_DICT, True),
        (VALID_SCHEMA_DICT, VALID_DATA_JSON, True),
        (VALID_SCHEMA_DICT, VALID_DATA_MODEL, True),
        (VALID_SCHEMA_JSON, VALID_DATA_DICT, True),
        (VALID_SCHEMA_JSON, VALID_DATA_JSON, True),
        (VALID_SCHEMA_JSON, VALID_DATA_MODEL, True),
        (PersonModel(name="dummy", age=25), VALID_DATA_DICT, True),
        (PersonModel(name="dummy", age=25), VALID_DATA_JSON, True),
        (PersonModel(name="dummy", age=25), VALID_DATA_MODEL, True),
        # Invalid combinations - all schema types with invalid data
        (VALID_SCHEMA_DICT, INVALID_DATA_DICT, False),
        (VALID_SCHEMA_DICT, INVALID_DATA_JSON, False),
        (VALID_SCHEMA_DICT, INVALID_DATA_MODEL, False),
        (VALID_SCHEMA_JSON, INVALID_DATA_DICT, False),
        (VALID_SCHEMA_JSON, INVALID_DATA_JSON, False),
        (VALID_SCHEMA_JSON, INVALID_DATA_MODEL, False),
        (PersonModel(name="dummy", age=25), INVALID_DATA_DICT, False),
        (PersonModel(name="dummy", age=25), INVALID_DATA_JSON, False),
        (PersonModel(name="dummy", age=25), INVALID_DATA_MODEL, False),
    ])
    def test_all_type_combinations(self, schema: Any, data: Any, expected_valid: bool) -> None:
        """Test all combinations of schema types, data types, and data validity."""
        # Execute validation
        is_valid, errors = validate_json_schema(data, schema)
        # Assert results based on expected validity
        if expected_valid:
            assert is_valid is True
            assert errors is None
        else:
            assert is_valid is False
            assert errors is not None
            assert isinstance(errors, list)
            assert len(errors) > 0
            # Check that error mentions the age constraint
            error_str = " ".join(errors)
            assert "age" in error_str.lower()


class TestValidateJsonSchemaValidCases:
    """Test validate_json_schema with various valid data scenarios."""

    def test_valid_simple_object(self) -> None:
        """Test validation of simple valid object."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        data = {"name": "Alice"}

        is_valid, errors = validate_json_schema(data, schema)
        assert is_valid is True
        assert errors is None

    def test_valid_nested_object(self) -> None:
        """Test validation of nested object structure."""
        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
            },
            "required": ["person"],
        }
        data = {"person": {"name": "Alice", "age": 30}}

        is_valid, errors = validate_json_schema(data, schema)
        assert is_valid is True
        assert errors is None

    def test_valid_array_schema(self) -> None:
        """Test validation with array schema."""
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        }
        data = ["apple", "banana", "cherry"]

        is_valid, errors = validate_json_schema(data, schema)
        assert is_valid is True
        assert errors is None


class TestValidateJsonSchemaInvalidCases:
    """Test validate_json_schema with various invalid data scenarios."""

    def test_missing_required_field(self) -> None:
        """Test validation failure when required field is missing."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        data = {"name": "Alice"}  # Missing age

        is_valid, errors = validate_json_schema(data, schema)
        assert is_valid is False
        assert errors is not None
        assert len(errors) == 1
        assert "'age' is a required property" in errors[0]
        assert "(root)" in errors[0]

    def test_type_mismatch(self) -> None:
        """Test validation failure when data type doesn't match schema."""
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer"}},
        }
        data = {"age": "not a number"}

        is_valid, errors = validate_json_schema(data, schema)
        assert is_valid is False
        assert errors is not None
        assert len(errors) == 1
        assert "/age" in errors[0]
        assert "not of type" in errors[0]

    def test_constraint_violation(self) -> None:
        """Test validation failure when numeric constraints are violated."""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 0, "maximum": 120},
            },
        }
        data = {"age": -5}

        is_valid, errors = validate_json_schema(data, schema)
        assert is_valid is False
        assert errors is not None
        assert len(errors) == 1
        assert "/age" in errors[0]
        assert "less than the minimum" in errors[0]

    def test_multiple_validation_errors(self) -> None:
        """Test that multiple validation errors are captured."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
            },
            "required": ["name", "age"],
        }
        data = {"age": -1}  # Missing name, invalid age

        is_valid, errors = validate_json_schema(data, schema)
        assert is_valid is False
        assert errors is not None
        assert len(errors) == 2

        # Check that both errors are present
        error_str = " ".join(errors)
        assert "'name' is a required property" in error_str
        assert "less than the minimum" in error_str

    def test_additional_properties_not_allowed(self) -> None:
        """Test validation failure when additional properties are not allowed."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        data = {"name": "Alice", "extra_field": "not allowed"}

        is_valid, errors = validate_json_schema(data, schema)
        assert is_valid is False
        assert errors is not None
        assert len(errors) == 1
        assert "Additional properties are not allowed" in errors[0]


class TestValidateJsonSchemaErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_json_in_schema_string(self) -> None:
        """Test that invalid JSON in schema string raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            validate_json_schema(VALID_DATA_DICT, "{invalid json}")

    def test_invalid_json_in_data_string(self) -> None:
        """Test that invalid JSON in data string raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            validate_json_schema("{invalid json}", VALID_SCHEMA_DICT)

    def test_non_object_schema_raises_error(self) -> None:
        """Test that non-object schema raises TypeError."""
        with pytest.raises(TypeError, match="Schema must be a JSON object"):
            validate_json_schema(VALID_DATA_DICT, "42")

        with pytest.raises(TypeError, match="Schema must be a JSON object"):
            validate_json_schema(VALID_DATA_DICT, "[1, 2, 3]")

    def test_empty_schema_and_data(self) -> None:
        """Test validation with empty schema and data."""
        empty_schema = {"type": "object"}
        empty_data = {}

        is_valid, errors = validate_json_schema(empty_data, empty_schema)
        assert is_valid is True
        assert errors is None

    def test_complex_basemodel_schema_extraction(self) -> None:
        """Test that BaseModel schema extraction works with complex models."""
        class ComplexModel(BaseModel):
            """Complex model for testing schema extraction."""

            name: str
            tags: list[str]
            metadata: dict[str, int]

        # Use instance to extract schema
        model_instance = ComplexModel(name="test", tags=["a", "b"], metadata={"x": 1})
        valid_data = {"name": "Alice", "tags": ["python"], "metadata": {"score": 100}}
        invalid_data = {"name": "Bob", "tags": "not a list", "metadata": {"score": 100}}

        # Valid case
        is_valid, errors = validate_json_schema(valid_data, model_instance)
        assert is_valid is True
        assert errors is None

        # Invalid case
        is_valid, errors = validate_json_schema(invalid_data, model_instance)
        assert is_valid is False
        assert errors is not None
        assert len(errors) > 0
