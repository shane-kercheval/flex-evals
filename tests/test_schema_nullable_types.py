"""Tests for nullable type handling in schema generation."""

from typing import Optional
from pydantic import BaseModel, Field

from flex_evals.schema_generator import (
    _get_python_type_string,
    _extract_field_schema,
    generate_check_schema,
)


class TestNullableTypeConversion:
    """Test nullable type string conversion."""

    def test_optional_syntax_types(self):
        """Test Optional[T] syntax converts to clean type + nullable."""
        test_cases = [
            (Optional[str], "string"),  # noqa: UP045
            (Optional[int], "integer"),  # noqa: UP045
            (Optional[bool], "boolean"),  # noqa: UP045
            (Optional[dict], "object"),  # noqa: UP045
            (Optional[list], "array"),  # noqa: UP045
            (Optional[float], "number"),  # noqa: UP045
        ]

        for annotation, expected_type in test_cases:
            result = _get_python_type_string(annotation)
            # Should return just the base type, nullable handled separately
            assert result == expected_type, f"Optional[{expected_type}] should return '{expected_type}', got '{result}'"  # noqa: E501

    def test_union_syntax_types(self):
        """Test T | None syntax converts to clean type + nullable."""
        test_cases = [
            (str | None, "string"),
            (int | None, "integer"),
            (bool | None, "boolean"),
            (dict | None, "object"),
            (list | None, "array"),
            (float | None, "number"),
        ]

        for annotation, expected_type in test_cases:
            result = _get_python_type_string(annotation)
            assert result == expected_type, f"{expected_type} | None should return '{expected_type}', got '{result}'"  # noqa: E501

    def test_non_nullable_types(self):
        """Test non-nullable types return clean type."""
        test_cases = [
            (str, "string"),
            (int, "integer"),
            (bool, "boolean"),
            (dict, "object"),
            (list, "array"),
            (float, "number"),
        ]

        for annotation, expected_type in test_cases:
            result = _get_python_type_string(annotation)
            assert result == expected_type


class TestFieldSchemaExtraction:
    """Test field schema extraction with new format."""

    def test_required_non_nullable_field(self):
        """Test required, non-nullable field schema."""
        class TestModel(BaseModel):
            test_field: str = Field(..., description="Required string field")

        field_info = TestModel.model_fields["test_field"]
        schema = _extract_field_schema("test_field", field_info, TestModel)

        expected = {
            "type": "string",
            "nullable": False,
            "description": "Required string field",
        }
        assert schema == expected

    def test_optional_non_nullable_field(self):
        """Test optional, non-nullable field with default."""
        class TestModel(BaseModel):
            test_field: bool = Field(True, description="Optional boolean field")

        field_info = TestModel.model_fields["test_field"]
        schema = _extract_field_schema("test_field", field_info, TestModel)

        expected = {
            "type": "boolean",
            "nullable": False,
            "default": True,
            "description": "Optional boolean field",
        }
        assert schema == expected

    def test_optional_nullable_field_with_none_default(self):
        """Test optional, nullable field with None default."""
        class TestModel(BaseModel):
            test_field: str | None = Field(None, description="Optional nullable string")

        field_info = TestModel.model_fields["test_field"]
        schema = _extract_field_schema("test_field", field_info, TestModel)

        expected = {
            "type": "string",
            "nullable": True,
            "default": None,
            "description": "Optional nullable string",
        }
        assert schema == expected

    def test_optional_nullable_field_with_value_default(self):
        """Test optional, nullable field with non-None default."""
        class TestModel(BaseModel):
            test_field: str | None = Field("hello", description="Optional nullable with default")

        field_info = TestModel.model_fields["test_field"]
        schema = _extract_field_schema("test_field", field_info, TestModel)

        expected = {
            "type": "string",
            "nullable": True,
            "default": "hello",
            "description": "Optional nullable with default",
        }
        assert schema == expected

    def test_required_nullable_field(self):
        """Test required, nullable field (must provide, can be None)."""
        class TestModel(BaseModel):
            test_field: str | None = Field(..., description="Required but can be None")

        field_info = TestModel.model_fields["test_field"]
        schema = _extract_field_schema("test_field", field_info, TestModel)

        expected = {
            "type": "string",
            "nullable": True,
            "description": "Required but can be None",
        }
        assert schema == expected

    def test_union_syntax_nullable_field(self):
        """Test T | None syntax creates nullable field."""
        class TestModel(BaseModel):
            test_field: str | None = Field(None, description="Union syntax nullable")

        field_info = TestModel.model_fields["test_field"]
        schema = _extract_field_schema("test_field", field_info, TestModel)

        expected = {
            "type": "string",
            "nullable": True,
            "default": None,
            "description": "Union syntax nullable",
        }
        assert schema == expected

    def test_field_without_description(self):
        """Test field without description gets empty string."""
        class TestModel(BaseModel):
            test_field: str = Field(...)

        field_info = TestModel.model_fields["test_field"]
        schema = _extract_field_schema("test_field", field_info, TestModel)

        expected = {
            "type": "string",
            "nullable": False,
            "description": "",
        }
        assert schema == expected


class TestIntegrationWithExistingChecks:
    """Test integration with existing check schemas."""

    def test_contains_check_fields(self):
        """Test ContainsCheck field schema generation."""
        schema = generate_check_schema("contains", "1.0.0")
        fields = schema["fields"]

        # text field - required, non-nullable
        assert fields["text"]["type"] == "string"
        assert fields["text"]["nullable"] is False
        assert "default" not in fields["text"]

        # phrases field - required, non-nullable
        assert fields["phrases"]["type"] == "array<string>"
        assert fields["phrases"]["nullable"] is False
        assert "default" not in fields["phrases"]

        # case_sensitive field - optional, non-nullable with default
        assert fields["case_sensitive"]["type"] == "boolean"
        assert fields["case_sensitive"]["nullable"] is False
        assert fields["case_sensitive"]["default"] is True

        # negate field - optional, non-nullable with default
        assert fields["negate"]["type"] == "boolean"
        assert fields["negate"]["nullable"] is False
        assert fields["negate"]["default"] is False

    def test_regex_check_nullable_flags_field(self):
        """Test RegexCheck flags field (nullable complex type)."""
        schema = generate_check_schema("regex", "1.0.0")
        flags_field = schema["fields"]["flags"]

        # Should be clean type name, nullable, with None default
        assert flags_field["type"] == "RegexFlags"  # Clean type name
        assert flags_field["nullable"] is True
        assert flags_field["default"] is None
