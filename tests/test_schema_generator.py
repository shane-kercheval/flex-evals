"""Tests for dynamic schema generation."""

import json
import pytest
from typing import Optional, ClassVar
from flex_evals.schema_generator import (
    generate_checks_schema,
    generate_check_schema,
    _get_schema_class_for_check_type,
    _extract_field_schema,
    _get_python_type_string,
)
from flex_evals.schemas.checks.contains import ContainsCheck
from flex_evals.schemas.checks.exact_match import ExactMatchCheck
from flex_evals.schemas.checks.attribute_exists import AttributeExistsCheck
from flex_evals.schemas.check import SchemaCheck


@pytest.fixture
def mock_schema_classes():
    """Create test schema classes for testing version handling."""
    class TestSchemaV1(SchemaCheck):
        CHECK_TYPE: ClassVar[str] = "test_check_v1"
        VERSION: ClassVar[str] = "1.0.0"
        test_field: str = "v1"

    class TestSchemaV2(SchemaCheck):
        CHECK_TYPE: ClassVar[str] = "test_check_v2"
        VERSION: ClassVar[str] = "2.0.0"
        test_field: str = "v2"
        new_field: str = "only in v2"

    class TestSchemaVersioned(SchemaCheck):
        CHECK_TYPE: ClassVar[str] = "test_versioned"
        VERSION: ClassVar[str] = "1.5.0"
        versioned_field: str = "versioned"

    return {
        "v1": TestSchemaV1,
        "v2": TestSchemaV2,
        "versioned": TestSchemaVersioned,
    }


class TestSchemaGeneration:
    """Test dynamic schema generation functionality."""

    def test_generate_checks_schema_structure(self):
        """Test that generate_checks_schema returns expected structure."""
        schema = generate_checks_schema()

        # Should be a dictionary
        assert isinstance(schema, dict)

        # Should have check types we know exist
        assert "contains" in schema
        assert "exact_match" in schema

        # Each check type should have version structure
        contains_schema = schema["contains"]
        assert isinstance(contains_schema, dict)
        assert "1.0.0" in contains_schema

        # Each version should have required fields
        version_schema = contains_schema["1.0.0"]
        assert "version" in version_schema
        assert "is_async" in version_schema
        assert "fields" in version_schema

        assert version_schema["version"] == "1.0.0"
        assert isinstance(version_schema["is_async"], bool)
        assert isinstance(version_schema["fields"], dict)

    def test_generate_checks_schema_latest_only(self):
        """Test generate_checks_schema with latest_only flag."""
        all_schema = generate_checks_schema(include_latest_only=False)
        latest_schema = generate_checks_schema(include_latest_only=True)

        # Latest should be subset of all
        for check_type, versions in latest_schema.items():
            assert check_type in all_schema
            assert len(versions) == 1  # Should only have one version per check type

            version = next(iter(versions.keys()))
            assert version in all_schema[check_type]

    def test_generate_check_schema_specific(self):
        """Test generating schema for specific check type and version."""
        schema = generate_check_schema("contains", "1.0.0")

        assert schema is not None
        assert schema["version"] == "1.0.0"
        assert "is_async" in schema
        assert "fields" in schema

        # Contains check should have expected fields
        fields = schema["fields"]
        assert "text" in fields
        assert "phrases" in fields
        assert "case_sensitive" in fields
        assert "negate" in fields

    def test_generate_check_schema_latest_version(self):
        """Test generating schema for latest version of check type."""
        schema = generate_check_schema("contains", version=None)

        assert schema is not None
        assert schema["version"] == "1.0.0"  # Current latest version

        # Should be same as explicitly requesting 1.0.0
        explicit_schema = generate_check_schema("contains", "1.0.0")
        assert schema == explicit_schema

    def test_generate_check_schema_nonexistent(self):
        """Test generating schema for non-existent check type or version."""
        # Non-existent check type
        schema = generate_check_schema("nonexistent_check")
        assert schema is None

        # Non-existent version
        schema = generate_check_schema("contains", "99.0.0")
        assert schema is None

    def test_field_schema_structure_contains(self):
        """Test detailed field schema structure for ContainsCheck."""
        schema = generate_check_schema("contains", "1.0.0")
        fields = schema["fields"]

        # Test text field - required, non-nullable
        text_field = fields["text"]
        assert text_field["type"] == "string"
        assert text_field["nullable"] is False
        assert "default" not in text_field  # Required field has no default
        assert "description" in text_field
        assert text_field["jsonpath"] == "optional"

        # Test phrases field - required, non-nullable
        phrases_field = fields["phrases"]
        assert phrases_field["type"] == "array<string>"
        assert phrases_field["nullable"] is False
        assert "default" not in phrases_field  # Required field has no default

        # Test case_sensitive field - optional, non-nullable with default
        case_sensitive_field = fields["case_sensitive"]
        assert case_sensitive_field["type"] == "boolean"
        assert case_sensitive_field["nullable"] is False
        assert case_sensitive_field["default"] is True

        # Test negate field - optional, non-nullable with default
        negate_field = fields["negate"]
        assert negate_field["type"] == "boolean"
        assert negate_field["nullable"] is False
        assert negate_field["default"] is False

    def test_field_schema_structure_exact_match(self):
        """Test detailed field schema structure for ExactMatchCheck."""
        schema = generate_check_schema("exact_match", "1.0.0")
        fields = schema["fields"]

        # Test required fields exist
        assert "actual" in fields
        assert "expected" in fields

        # Test actual field - required, non-nullable
        actual_field = fields["actual"]
        assert actual_field["type"] == "string"
        assert actual_field["nullable"] is False
        assert "default" not in actual_field  # Required field has no default
        assert actual_field["jsonpath"] == "optional"

        # Test expected field - required, non-nullable
        expected_field = fields["expected"]
        assert expected_field["type"] == "string"
        assert expected_field["nullable"] is False
        assert "default" not in expected_field  # Required field has no default
        assert expected_field["jsonpath"] == "optional"

    def test_async_status_detection(self):
        """Test that async status is correctly detected from registry."""
        # Standard checks should be sync
        contains_schema = generate_check_schema("contains", "1.0.0")
        assert contains_schema["is_async"] is False

        exact_match_schema = generate_check_schema("exact_match", "1.0.0")
        assert exact_match_schema["is_async"] is False

        # Extended checks should be async (if available)
        # Note: This test may need adjustment based on what checks are actually registered
        all_schemas = generate_checks_schema()
        for check_type, versions in all_schemas.items():
            for version, schema in versions.items():
                assert "is_async" in schema
                assert isinstance(schema["is_async"], bool)


class TestSchemaClassDiscovery:
    """Test schema class discovery functionality."""

    def test_get_schema_class_for_check_type_with_version(self):
        """Test finding schema classes for check types with correct version."""
        # Should find ContainsCheck for 'contains' version '1.0.0'
        schema_class = _get_schema_class_for_check_type("contains", "1.0.0")
        assert schema_class is ContainsCheck

        # Should find ExactMatchCheck for 'exact_match' version '1.0.0'
        schema_class = _get_schema_class_for_check_type("exact_match", "1.0.0")
        assert schema_class is ExactMatchCheck

        # Should find AttributeExistsCheck for 'attribute_exists' version '1.0.0'
        schema_class = _get_schema_class_for_check_type("attribute_exists", "1.0.0")
        assert schema_class is AttributeExistsCheck

    def test_get_schema_class_nonexistent_check_type(self):
        """Test finding schema class for non-existent check type raises ValueError."""
        with pytest.raises(
            ValueError, match="No schema class found for check type 'nonexistent_check'",
        ):
            _get_schema_class_for_check_type("nonexistent_check", "1.0.0")

    def test_get_schema_class_nonexistent_version(self):
        """Test finding schema class for non-existent version raises ValueError."""
        with pytest.raises(
            ValueError, match="No schema class found for check type 'contains' version '2.0.0'",
        ):
            _get_schema_class_for_check_type("contains", "2.0.0")

    def test_get_schema_class_version_mismatch_shows_available(self):
        """Test that version mismatch error shows available versions."""
        with pytest.raises(
            ValueError, match="No schema class found for check type 'contains' version '99.0.0'",
        ) as exc_info:
            _get_schema_class_for_check_type("contains", "99.0.0")

        error_message = str(exc_info.value)
        assert "Available versions: ['1.0.0']" in error_message

    def test_multiple_schema_versions(self, mock_schema_classes):  # noqa: ANN001
        """Test handling multiple schema classes with different versions."""
        # Should find v1 when requesting v1
        schema_class = _get_schema_class_for_check_type("test_check_v1", "1.0.0")
        assert schema_class is mock_schema_classes["v1"]

        # Should find v2 when requesting v2
        schema_class = _get_schema_class_for_check_type("test_check_v2", "2.0.0")
        assert schema_class is mock_schema_classes["v2"]

        # Should find versioned when requesting versioned
        schema_class = _get_schema_class_for_check_type("test_versioned", "1.5.0")
        assert schema_class is mock_schema_classes["versioned"]

        # Should raise error for non-existent version
        with pytest.raises(ValueError, match="Available versions:"):
            _get_schema_class_for_check_type("test_check_v1", "3.0.0")


class TestFieldSchemaExtraction:
    """Test field schema extraction functionality."""

    def test_extract_field_schema_basic(self):
        """Test extracting schema for basic field types."""
        # Get a field from ContainsCheck
        field_info = ContainsCheck.model_fields["text"]
        field_schema = _extract_field_schema("text", field_info, ContainsCheck)

        assert field_schema["type"] == "string"
        assert field_schema["nullable"] is False
        assert "default" not in field_schema  # Required field has no default
        assert field_schema["jsonpath"] == "optional"
        assert "description" in field_schema

    def test_extract_field_schema_with_default(self):
        """Test extracting schema for field with default value."""
        field_info = ContainsCheck.model_fields["case_sensitive"]
        field_schema = _extract_field_schema("case_sensitive", field_info, ContainsCheck)

        assert field_schema["type"] == "boolean"
        assert field_schema["nullable"] is False
        assert field_schema["default"] is True

    def test_extract_field_schema_list_type(self):
        """Test extracting schema for list field type."""
        field_info = ContainsCheck.model_fields["phrases"]
        field_schema = _extract_field_schema("phrases", field_info, ContainsCheck)

        assert field_schema["type"] == "array<string>"
        assert field_schema["nullable"] is False
        assert "default" not in field_schema  # Required field has no default


class TestTypeStringGeneration:
    """Test Python type to string conversion."""

    def test_get_python_type_string_basic_types(self):
        """Test conversion of basic Python types."""
        assert _get_python_type_string(str) == "string"
        assert _get_python_type_string(int) == "integer"
        assert _get_python_type_string(float) == "number"
        assert _get_python_type_string(bool) == "boolean"
        assert _get_python_type_string(dict) == "object"
        assert _get_python_type_string(list) == "array"

    def test_get_python_type_string_generic_types(self):
        """Test conversion of generic types."""
        assert _get_python_type_string(list[str]) == "array<string>"
        assert _get_python_type_string(dict[str, int]) == "object<string,integer>"

        # Optional types should return clean base type (nullable handled separately)
        optional_str = Optional[str]  # This is Union[str, None]  # noqa: UP045
        result = _get_python_type_string(optional_str)
        assert result == "string"  # Clean base type, nullable info handled by _is_nullable_type

    def test_get_python_type_string_none(self):
        """Test conversion of None type."""
        result = _get_python_type_string(None)
        assert result == "any"


class TestEndToEndIntegration:
    """Test end-to-end integration with real check schemas."""

    def test_all_registered_checks_have_schemas(self):
        """Test that all registered checks can generate schemas."""
        schema = generate_checks_schema()

        # Should have entries for all check types we expect
        expected_check_types = [
            "contains", "exact_match", "attribute_exists", "is_empty",
            "threshold", "regex",
        ]

        for check_type in expected_check_types:
            assert check_type in schema, f"Missing schema for check type: {check_type}"

            # Each should have at least one version
            assert len(schema[check_type]) >= 1, f"No versions for check type: {check_type}"

            # Each version should have valid structure
            for version, version_schema in schema[check_type].items():
                assert "version" in version_schema
                assert "is_async" in version_schema
                assert "fields" in version_schema
                assert isinstance(version_schema["fields"], dict)

    def test_schema_json_serializable(self):
        """Test that generated schemas are JSON serializable."""
        schema = generate_checks_schema()

        # Should be able to serialize to JSON without errors
        json_str = json.dumps(schema, indent=2)

        # Should be able to deserialize back
        deserialized = json.loads(json_str)

        # Should maintain structure
        assert isinstance(deserialized, dict)
        assert "contains" in deserialized
        assert "exact_match" in deserialized
