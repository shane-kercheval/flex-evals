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
from pydantic import BaseModel, ConfigDict, Field

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


class TestValidateJsonSchemaAdvanced:
    """Test advanced JSON Schema validation scenarios."""

    def test_optional_fields(self) -> None:
        """Test validation with optional fields."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string", "format": "email"},  # Optional
            },
            "required": ["name", "age"],  # email is optional
            "additionalProperties": False,
        }

        # Valid with all fields
        data_all = {"name": "Alice", "age": 30, "email": "alice@example.com"}
        is_valid, errors = validate_json_schema(data_all, schema)
        assert is_valid is True
        assert errors is None

        # Valid without optional field
        data_minimal = {"name": "Bob", "age": 25}
        is_valid, errors = validate_json_schema(data_minimal, schema)
        assert is_valid is True
        assert errors is None

        # Invalid email format
        data_bad_email = {"name": "Charlie", "age": 35, "email": "not-an-email"}
        is_valid, errors = validate_json_schema(data_bad_email, schema)
        assert is_valid is False
        assert errors is not None
        assert any("email" in error.lower() for error in errors)

    def test_nested_objects_pydantic_schema(self) -> None:
        """Test validation with nested objects using Pydantic model as schema."""
        # Define nested Pydantic models that mirror the dict schema structure
        class UserPreferences(BaseModel):
            """User preferences nested model."""

            model_config = {"extra": "forbid"}

            theme: str = Field(..., pattern="^(light|dark)$")  # Enum constraint via pattern
            notifications: bool

        class UserProfile(BaseModel):
            """User profile nested model."""

            model_config = {"extra": "forbid"}

            name: str
            preferences: UserPreferences

        class UserContainer(BaseModel):
            """Top-level user container."""

            model_config = {"extra": "forbid"}

            user: UserProfile

        # Create an instance to extract schema from
        schema_model = UserContainer(
            user=UserProfile(
                name="schema_template",
                preferences=UserPreferences(theme="dark", notifications=True),
            ),
        )

        # Valid nested data - should match the Pydantic schema structure
        valid_data = {
            "user": {
                "name": "Alice",
                "preferences": {"theme": "dark", "notifications": True},
            },
        }
        is_valid, errors = validate_json_schema(valid_data, schema_model)
        assert is_valid is True
        assert errors is None

        # Missing nested required field - should fail
        invalid_data_missing = {
            "user": {
                "name": "Bob",
                "preferences": {},  # Missing required fields
            },
        }
        is_valid, errors = validate_json_schema(invalid_data_missing, schema_model)
        assert is_valid is False
        assert errors is not None
        # Should have errors for missing theme and notifications
        assert any("theme" in error or "notifications" in error for error in errors)

        # Invalid enum-like value (pattern violation)
        invalid_data_enum = {
            "user": {
                "name": "Charlie",
                "preferences": {"theme": "purple", "notifications": True},  # Invalid theme
            },
        }
        is_valid, errors = validate_json_schema(invalid_data_enum, schema_model)
        assert is_valid is False
        assert errors is not None
        # Should have error about theme pattern
        assert any("pattern" in error.lower() or "does not match" in error.lower() for error in errors)  # noqa: E501

        # Extra properties not allowed
        invalid_data_extra = {
            "user": {
                "name": "David",
                "preferences": {
                    "theme": "light",
                    "notifications": False,
                    "extra_setting": "not_allowed",  # Extra property
                },
            },
        }
        is_valid, errors = validate_json_schema(invalid_data_extra, schema_model)
        assert is_valid is False
        assert errors is not None
        assert any("additional properties" in error.lower() for error in errors)

    def test_nested_objects_pydantic_class_schema(self) -> None:
        """Test validation using Pydantic model CLASS as schema (not instance)."""

        # Define nested Pydantic models
        class UserPreferences(BaseModel):
            theme: str = Field(..., pattern="^(light|dark)$")
            notifications: bool

        class UserProfile(BaseModel):
            name: str
            preferences: UserPreferences

        class UserContainer(BaseModel):
            model_config = ConfigDict(extra='forbid')  # No additional properties allowed

            user: UserProfile

        # Valid data
        valid_data = {
            "user": {
                "name": "Alice",
                "preferences": {"theme": "dark", "notifications": True},
            },
        }

        # Test passing the Pydantic CLASS as schema (not an instance)
        is_valid, errors = validate_json_schema(valid_data, UserContainer)
        assert is_valid is True
        assert errors is None

        # Invalid data - missing required field
        invalid_data_missing = {
            "user": {
                "name": "Bob",
                "preferences": {"notifications": True},  # Missing required 'theme'
            },
        }
        is_valid, errors = validate_json_schema(invalid_data_missing, UserContainer)
        assert is_valid is False
        assert errors is not None
        assert any("theme" in error for error in errors)

        # Invalid data - pattern violation
        invalid_data_pattern = {
            "user": {
                "name": "Charlie",
                "preferences": {"theme": "purple", "notifications": True},  # Invalid theme
            },
        }
        is_valid, errors = validate_json_schema(invalid_data_pattern, UserContainer)
        assert is_valid is False
        assert errors is not None
        assert any("pattern" in error.lower() or "does not match" in error.lower() for error in errors)  # noqa: E501

    def test_pydantic_class_as_data_raises_error(self) -> None:
        """Test that using a Pydantic model class as data (not schema) raises TypeError."""

        class TestModel(BaseModel):
            name: str
            age: int

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        # Should raise TypeError when trying to use model class as data
        with pytest.raises(TypeError, match="Pydantic model class can only be used as schema"):
            validate_json_schema(TestModel, schema)

    def test_nested_objects(self) -> None:
        """Test validation with nested object structures."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "profile": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "preferences": {
                                    "type": "object",
                                    "properties": {
                                        "theme": {"type": "string", "enum": ["light", "dark"]},
                                        "notifications": {"type": "boolean"},
                                    },
                                    "required": ["theme"],
                                },
                            },
                            "required": ["name", "preferences"],
                        },
                    },
                    "required": ["profile"],
                },
            },
            "required": ["user"],
        }

        # Valid nested data
        valid_data = {
            "user": {
                "profile": {
                    "name": "Alice",
                    "preferences": {"theme": "dark", "notifications": True},
                },
            },
        }
        is_valid, errors = validate_json_schema(valid_data, schema)
        assert is_valid is True
        assert errors is None

        # Missing nested required field
        invalid_data = {
            "user": {
                "profile": {
                    "name": "Bob",
                    "preferences": {},  # Missing required 'theme'
                },
            },
        }
        is_valid, errors = validate_json_schema(invalid_data, schema)
        assert is_valid is False
        assert errors is not None
        assert any("theme" in error for error in errors)

        # Invalid enum value
        invalid_enum = {
            "user": {
                "profile": {
                    "name": "Charlie",
                    "preferences": {"theme": "purple"},  # Invalid enum value
                },
            },
        }
        is_valid, errors = validate_json_schema(invalid_enum, schema)
        assert is_valid is False
        assert errors is not None
        assert any("not one of" in error.lower() for error in errors)

    def test_array_validation(self) -> None:
        """Test validation with arrays and array constraints."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                    "uniqueItems": True,
                },
                "scores": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0, "maximum": 100},
                },
            },
            "required": ["tags"],
        }

        # Valid arrays
        valid_data = {
            "tags": ["python", "testing", "json"],
            "scores": [85.5, 92.0, 78.5],
        }
        is_valid, errors = validate_json_schema(valid_data, schema)
        assert is_valid is True
        assert errors is None

        # Empty array (violates minItems)
        empty_array = {"tags": []}
        is_valid, errors = validate_json_schema(empty_array, schema)
        assert is_valid is False
        assert errors is not None
        assert any("non-empty" in error.lower() or "too few" in error.lower() for error in errors)

        # Too many items
        too_many = {"tags": ["a", "b", "c", "d", "e", "f"]}  # 6 items > maxItems 5
        is_valid, errors = validate_json_schema(too_many, schema)
        assert is_valid is False
        assert errors is not None
        assert any("too long" in error.lower() for error in errors)

        # Non-unique items
        duplicate_items = {"tags": ["python", "testing", "python"]}  # Duplicates
        is_valid, errors = validate_json_schema(duplicate_items, schema)
        assert is_valid is False
        assert errors is not None
        assert any("non-unique" in error.lower() for error in errors)

        # Invalid item type
        wrong_type = {"tags": ["valid", 123, "string"]}  # 123 is not a string
        is_valid, errors = validate_json_schema(wrong_type, schema)
        assert is_valid is False
        assert errors is not None
        assert any("not of type" in error.lower() for error in errors)

    def test_numeric_constraints(self) -> None:
        """Test numeric validation with various constraints."""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "score": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 100},
                "rating": {"type": "number", "multipleOf": 0.5},
                "count": {"type": "integer", "minimum": 1},
            },
            "required": ["age", "score"],
        }

        # Valid numeric data
        valid_data = {"age": 25, "score": 85.5, "rating": 4.5, "count": 10}
        is_valid, errors = validate_json_schema(valid_data, schema)
        assert is_valid is True
        assert errors is None

        # Multiple constraint violations
        invalid_data = {
            "age": -5,        # Below minimum
            "score": 100,     # Not exclusive of maximum
            "rating": 4.7,    # Not multiple of 0.5
            "count": 0,       # Below minimum
        }
        is_valid, errors = validate_json_schema(invalid_data, schema)
        assert is_valid is False
        assert errors is not None
        assert len(errors) == 4  # Should have 4 separate validation errors

        # Check specific error types
        error_str = " ".join(errors).lower()
        assert "minimum" in error_str  # Age minimum violation
        assert "greater than or equal" in error_str  # Score exclusive max (jsonschema message)
        assert "multiple" in error_str  # Rating multiple of constraint

    def test_string_constraints(self) -> None:
        """Test string validation with length and pattern constraints."""
        schema = {
            "type": "object",
            "properties": {
                "username": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 20,
                    "pattern": "^[a-zA-Z0-9_]+$",
                },
                "email": {"type": "string", "format": "email"},
                "password": {"type": "string", "minLength": 8},
                "description": {"type": "string", "maxLength": 100},
            },
            "required": ["username", "email"],
        }

        # Valid string data
        valid_data = {
            "username": "user_123",
            "email": "user@example.com",
            "password": "secret123",
            "description": "A short description",
        }
        is_valid, errors = validate_json_schema(valid_data, schema)
        assert is_valid is True
        assert errors is None

        # Multiple string constraint violations
        invalid_data = {
            "username": "a",                    # Too short (< 3 chars)
            "email": "not-an-email",           # Invalid format
            "password": "short",               # Too short (< 8 chars)
            "description": "x" * 101,          # Too long (> 100 chars)
        }
        is_valid, errors = validate_json_schema(invalid_data, schema)
        assert is_valid is False
        assert errors is not None
        assert len(errors) == 4  # Should have 4 validation errors

        # Pattern violation
        pattern_invalid = {
            "username": "user-with-dashes!",   # Contains invalid characters
            "email": "valid@email.com",
        }
        is_valid, errors = validate_json_schema(pattern_invalid, schema)
        assert is_valid is False
        assert errors is not None
        assert any("does not match" in error.lower() for error in errors)

    def test_type_validation(self) -> None:
        """Test validation with different data types."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "height": {"type": "number"},
                "is_active": {"type": "boolean"},
                "metadata": {"type": "object"},
                "tags": {"type": "array"},
                "notes": {"type": ["string", "null"]},  # Union type
            },
            "required": ["name", "age", "is_active"],
        }

        # Valid with all correct types
        valid_data = {
            "name": "Alice",
            "age": 30,
            "height": 5.6,
            "is_active": True,
            "metadata": {"role": "admin"},
            "tags": ["python", "testing"],
            "notes": "Some notes",
        }
        is_valid, errors = validate_json_schema(valid_data, schema)
        assert is_valid is True
        assert errors is None

        # Valid with null union type
        valid_null = {
            "name": "Bob",
            "age": 25,
            "is_active": False,
            "notes": None,  # Allowed by union type
        }
        is_valid, errors = validate_json_schema(valid_null, schema)
        assert is_valid is True
        assert errors is None

        # Multiple type violations
        invalid_types = {
            "name": 123,           # Should be string
            "age": "thirty",       # Should be integer
            "height": "tall",      # Should be number
            "is_active": "yes",    # Should be boolean
            "metadata": "text",    # Should be object
            "tags": "tag1,tag2",   # Should be array
        }
        is_valid, errors = validate_json_schema(invalid_types, schema)
        assert is_valid is False
        assert errors is not None
        assert len(errors) == 6  # One error per type violation

    def test_conditional_schemas(self) -> None:
        """Test conditional validation using if/then/else."""
        schema = {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["student", "teacher"]},
                "grade": {"type": "string"},
                "subject": {"type": "string"},
                "student_id": {"type": "string"},
                "employee_id": {"type": "string"},
            },
            "required": ["type"],
            "if": {"properties": {"type": {"const": "student"}}},
            "then": {"required": ["student_id", "grade"]},
            "else": {"required": ["employee_id", "subject"]},
        }

        # Valid student data
        student_data = {
            "type": "student",
            "student_id": "S12345",
            "grade": "A",
        }
        is_valid, errors = validate_json_schema(student_data, schema)
        assert is_valid is True
        assert errors is None

        # Valid teacher data
        teacher_data = {
            "type": "teacher",
            "employee_id": "E67890",
            "subject": "Mathematics",
        }
        is_valid, errors = validate_json_schema(teacher_data, schema)
        assert is_valid is True
        assert errors is None

        # Invalid student (missing required fields)
        invalid_student = {"type": "student"}  # Missing student_id and grade
        is_valid, errors = validate_json_schema(invalid_student, schema)
        assert is_valid is False
        assert errors is not None
        assert len(errors) == 2  # Missing student_id and grade

    def test_complex_multiple_errors(self) -> None:
        """Test scenarios that should produce multiple validation errors."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 2, "maxLength": 50},
                        "age": {"type": "integer", "minimum": 0, "maximum": 120},
                        "email": {"type": "string", "format": "email"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 3,
                        },
                    },
                    "required": ["name", "age", "email"],
                    "additionalProperties": False,
                },
                "metadata": {
                    "type": "object",
                    "properties": {"version": {"type": "string"}},
                    "required": ["version"],
                },
            },
            "required": ["user", "metadata"],
            "additionalProperties": False,
        }

        # Data with multiple errors at different levels
        invalid_data = {
            "user": {
                "name": "A",                    # Too short (minLength: 2)
                "age": -5,                     # Below minimum (minimum: 0)
                "email": "invalid-email",      # Invalid format
                "tags": [],                    # Too few items (minItems: 1)
                "extra": "not allowed",        # Additional property not allowed
            },
            "metadata": {},                    # Missing required 'version'
            "extra_root": "not allowed",       # Additional property at root
        }

        is_valid, errors = validate_json_schema(invalid_data, schema)
        assert is_valid is False
        assert errors is not None
        assert len(errors) >= 6  # Should have at least 6 errors

        # Verify specific error types are present
        error_str = " ".join(errors).lower()
        assert "too short" in error_str or "minlength" in error_str
        assert "minimum" in error_str
        assert "email" in error_str
        assert "non-empty" in error_str or "too few" in error_str
        assert "additional properties" in error_str
        assert "required" in error_str

    def test_anyof_oneof_allof(self) -> None:
        """Test validation with anyOf, oneOf, and allOf combinators."""
        # anyOf schema - data must match at least one
        anyof_schema = {
            "anyOf": [
                {"type": "string", "minLength": 5},
                {"type": "integer", "minimum": 10},
            ],
        }

        # Valid for anyOf - use JSON strings to avoid _coerce issues
        is_valid, _ = validate_json_schema('"hello"', anyof_schema)  # Matches first
        assert is_valid is True

        is_valid, _ = validate_json_schema('15', anyof_schema)  # Matches second
        assert is_valid is True

        # Invalid for anyOf
        is_valid, errors = validate_json_schema('"hi"', anyof_schema)  # Too short string
        assert is_valid is False
        assert errors is not None

        # oneOf schema - data must match exactly one
        oneof_schema = {
            "oneOf": [
                {"type": "string", "maxLength": 5},
                {"type": "string", "minLength": 10},
            ],
        }

        # Valid for oneOf
        # Matches first only
        is_valid, _ = validate_json_schema('"hi"', oneof_schema)
        assert is_valid is True
        # Matches second only
        is_valid, _ = validate_json_schema('"long string here"', oneof_schema)
        assert is_valid is True

        # Invalid for oneOf (matches both conditions - should fail)
        # Note: A string of exactly length 5 would match first schema (maxLength: 5)
        # but not second (minLength: 10), so it should be valid
        # Let's test with length that violates both
        is_valid, errors = validate_json_schema('123', oneof_schema)  # Number, not string
        assert is_valid is False
        assert errors is not None

        # allOf schema - data must match all
        allof_schema = {
            "allOf": [
                {"type": "string"},
                {"minLength": 3},
                {"maxLength": 10},
                {"pattern": "^[a-z]+$"},
            ],
        }

        # Valid for allOf
        is_valid, _ = validate_json_schema('"hello"', allof_schema)  # Matches all conditions
        assert is_valid is True

        # Invalid for allOf (fails pattern)
        is_valid, errors = validate_json_schema('"Hello"', allof_schema)  # Capital letter
        assert is_valid is False
        assert errors is not None

    def test_ref_and_definitions(self) -> None:
        """Test validation with $ref and definitions."""
        schema = {
            "type": "object",
            "properties": {
                "person": {"$ref": "#/$defs/Person"},
                "friends": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Person"},
                },
            },
            "$defs": {
                "Person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer", "minimum": 0},
                    },
                    "required": ["name", "age"],
                },
            },
        }

        # Valid with references
        valid_data = {
            "person": {"name": "Alice", "age": 30},
            "friends": [
                {"name": "Bob", "age": 25},
                {"name": "Charlie", "age": 35},
            ],
        }
        is_valid, errors = validate_json_schema(valid_data, schema)
        assert is_valid is True
        assert errors is None

        # Invalid reference data
        invalid_data = {
            "person": {"name": "Alice"},  # Missing required 'age'
            "friends": [
                {"name": "Bob", "age": -5},  # Invalid age
            ],
        }
        is_valid, errors = validate_json_schema(invalid_data, schema)
        assert is_valid is False
        assert errors is not None
        assert len(errors) == 2  # One for missing age, one for invalid age

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
