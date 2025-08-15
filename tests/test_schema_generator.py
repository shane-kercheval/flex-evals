"""Tests for dynamic schema generation."""

import json
import pytest
from typing import Any, Optional
from flex_evals import (
    generate_checks_schema,
    generate_check_schema,
    ContainsCheck,
    BaseCheck,
)
from flex_evals.schema_generator import (
    _extract_field_schema,
    _get_python_type_string,
    _extract_class_description,
)
from pydantic import Field


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
        assert "description" in version_schema
        assert "fields" in version_schema

        assert version_schema["version"] == "1.0.0"
        assert isinstance(version_schema["is_async"], bool)
        assert isinstance(version_schema["description"], str)
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
        assert "description" in schema
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

        # Test text field - required, non-nullable, supports JSONPath
        text_field = fields["text"]
        assert text_field["type"] == "union<string,JSONPath>"
        assert text_field["nullable"] is False
        assert "default" not in text_field  # Required field has no default
        assert "description" in text_field
        assert text_field["jsonpath"] == "optional"

        # Test phrases field - required, non-nullable, supports JSONPath
        phrases_field = fields["phrases"]
        assert phrases_field["type"] == "union<string,array<string>,JSONPath>"
        assert phrases_field["nullable"] is False
        assert "default" not in phrases_field  # Required field has no default

        # Test case_sensitive field - optional, non-nullable with default, supports JSONPath
        case_sensitive_field = fields["case_sensitive"]
        assert case_sensitive_field["type"] == "union<boolean,JSONPath>"
        assert case_sensitive_field["nullable"] is False
        assert case_sensitive_field["default"] is True

        # Test negate field - optional, non-nullable with default, supports JSONPath
        negate_field = fields["negate"]
        assert negate_field["type"] == "union<boolean,JSONPath>"
        assert negate_field["nullable"] is False
        assert negate_field["default"] is False

    def test_field_schema_structure_exact_match(self):
        """Test detailed field schema structure for ExactMatchCheck."""
        schema = generate_check_schema("exact_match", "1.0.0")
        fields = schema["fields"]

        # Test required fields exist
        assert "actual" in fields
        assert "expected" in fields

        # Test actual field - required, non-nullable, supports JSONPath
        actual_field = fields["actual"]
        assert actual_field["type"] == "union<Any,JSONPath>"
        assert actual_field["nullable"] is False
        assert "default" not in actual_field  # Required field has no default
        assert actual_field["jsonpath"] == "optional"

        # Test expected field - required, non-nullable, supports JSONPath
        expected_field = fields["expected"]
        assert expected_field["type"] == "union<Any,JSONPath>"
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


class TestCheckClassDiscovery:
    """Test schema class discovery functionality."""


    # Removed test_multiple_schema_versions - not applicable in unified architecture


class TestFieldSchemaExtraction:
    """Test field schema extraction functionality."""

    def test_extract_field_schema_basic(self):
        """Test extracting schema for basic field types."""
        # Get a field from ContainsCheck
        field_info = ContainsCheck.model_fields["text"]
        field_schema = _extract_field_schema("text", field_info, ContainsCheck)

        assert field_schema["type"] == "union<string,JSONPath>"
        assert field_schema["nullable"] is False
        assert "default" not in field_schema  # Required field has no default
        assert field_schema["jsonpath"] == "optional"
        assert "description" in field_schema

    def test_extract_field_schema_with_default(self):
        """Test extracting schema for field with default value."""
        field_info = ContainsCheck.model_fields["case_sensitive"]
        field_schema = _extract_field_schema("case_sensitive", field_info, ContainsCheck)

        assert field_schema["type"] == "union<boolean,JSONPath>"
        assert field_schema["nullable"] is False
        assert field_schema["default"] is True

    def test_extract_field_schema_list_type(self):
        """Test extracting schema for list field type."""
        field_info = ContainsCheck.model_fields["phrases"]
        field_schema = _extract_field_schema("phrases", field_info, ContainsCheck)

        assert field_schema["type"] == "union<string,array<string>,JSONPath>"
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


class TestClassDescriptionExtraction:
    """Test class description extraction functionality."""

    def test_extract_class_description_with_docstring(self):
        """Test extracting description from class with docstring."""
        class TestSchema(BaseCheck):
            """
            This is a test schema class.

            It has multiple lines in its docstring
            to test proper extraction.
            """

            test_field: str = Field("test", description="Test field")

            def __call__(self) -> dict[str, Any]:
                return {'passed': True}

        description = _extract_class_description(TestSchema)

        # Test that inspect.getdoc() properly dedents the docstring
        expected = (
            "This is a test schema class.\n"
            "\n"
            "It has multiple lines in its docstring\n"
            "to test proper extraction."
        )
        assert description == expected

        # Test key content is preserved
        assert "This is a test schema class." in description
        assert "It has multiple lines in its docstring" in description
        assert "to test proper extraction." in description
        assert description.startswith("This is a test schema class.")
        assert description.endswith("to test proper extraction.")

        # Test that it's properly dedented (no leading whitespace)
        lines = description.split('\n')
        assert len(lines) == 4  # Should have 4 lines total
        assert not lines[0].startswith(" ")  # First line should not have leading spaces
        assert not lines[2].startswith(" ")  # Third line should not have leading spaces

    def test_extract_class_description_single_line(self):
        """Test extracting description from class with single-line docstring."""
        class TestSchema(BaseCheck):
            """Single line description."""

            test_field: str = Field("test", description="Test field")

            def __call__(self) -> dict[str, Any]:
                return {'passed': True}

        description = _extract_class_description(TestSchema)
        assert description == "Single line description."

    def test_extract_class_description_no_docstring(self):
        """Test extracting description from class without docstring."""
        class TestSchema(BaseCheck):
            test_field: str = Field("test", description="Test field")

            def __call__(self) -> dict[str, Any]:
                return {'passed': True}

        description = _extract_class_description(TestSchema)
        assert description == ""

    def test_extract_class_description_empty_docstring(self):
        """Test extracting description from class with empty docstring."""
        class TestSchema(BaseCheck):
            """"""  # noqa: D419
            test_field: str = Field("test", description="Test field")

            def __call__(self) -> dict[str, Any]:
                return {'passed': True}

        description = _extract_class_description(TestSchema)
        assert description == ""

    def test_extract_class_description_whitespace_only(self):
        """Test extracting description from class with whitespace-only docstring."""
        class TestSchema(BaseCheck):
            """

            """  # noqa: D419
            test_field: str = Field("test", description="Test field")

            def __call__(self) -> dict[str, Any]:
                return {'passed': True}

        description = _extract_class_description(TestSchema)
        assert description == ""

    def test_extract_class_description_real_schema_class(self):
        """Test extracting description from real schema class."""
        description = _extract_class_description(ContainsCheck)

        # Should contain the actual docstring
        assert description != ""

    def test_schema_generation_includes_description(self):
        """Test that generated schemas include class descriptions."""
        schema = generate_check_schema("contains", "1.0.0")

        assert "description" in schema

    def test_schema_generation_includes_description_all_checks(self):
        """Test that all generated schemas include class descriptions."""
        schemas = generate_checks_schema(include_latest_only=True)

        for check_type, versions in schemas.items():
            for version, schema in versions.items():
                assert "description" in schema, f"Missing description for {check_type} v{version}"
                # Description should be a string (may be empty for checks without schema classes)
                assert isinstance(schema["description"], str), f"Invalid description type for {check_type} v{version}"  # noqa: E501


class TestDescriptionIntegrationWithMockClasses:
    """Test description extraction with mock schema classes."""

    @pytest.fixture
    def mock_schema_classes_with_descriptions(self):
        """Create test schema classes with various docstring formats."""
        class DetailedDocstringSchema(BaseCheck):
            """
            Comprehensive test schema class.

            This schema class demonstrates the following features:
            - Field validation
            - Type checking
            - Error handling

            Fields:
            - field1: First test field
            - field2: Second test field with complex validation

            Usage:
                check = DetailedDocstringSchema(field1="value", field2=123)
                result = check.validate()

            Returns:
                ValidationResult containing success/failure information
            """

            field1: str = Field("test", description="First test field")
            field2: int = Field(42, description="Second test field with complex validation")

            def __call__(self) -> dict[str, Any]:
                return {'passed': True}

        class MinimalDocstringSchema(BaseCheck):
            """Minimal test schema for basic validation."""

            simple_field: bool = Field(True, description="Simple field")

            def __call__(self) -> dict[str, Any]:
                return {'passed': True}

        class NoDocstringSchema(BaseCheck):
            basic_field: str = Field("default", description="Basic field")

            def __call__(self) -> dict[str, Any]:
                return {'passed': True}

        return {
            "detailed": DetailedDocstringSchema,
            "minimal": MinimalDocstringSchema,
            "none": NoDocstringSchema,
        }

    def test_detailed_docstring_extraction(self, mock_schema_classes_with_descriptions):  # noqa: ANN001
        """Test extraction of detailed multi-line docstring."""
        detailed_class = mock_schema_classes_with_descriptions["detailed"]
        description = _extract_class_description(detailed_class)

        # Should contain key parts of the docstring
        assert "Comprehensive test schema class" in description
        assert "This schema class demonstrates the following features" in description
        assert "Field validation" in description
        assert "Usage:" in description
        assert "Returns:" in description

    def test_minimal_docstring_extraction(self, mock_schema_classes_with_descriptions):  # noqa: ANN001
        """Test extraction of minimal single-line docstring."""
        minimal_class = mock_schema_classes_with_descriptions["minimal"]
        description = _extract_class_description(minimal_class)

        assert description == "Minimal test schema for basic validation."

    def test_no_docstring_extraction(self, mock_schema_classes_with_descriptions: dict[str, Any]):
        """Test extraction from class without docstring."""
        no_docstring_class = mock_schema_classes_with_descriptions["none"]
        description = _extract_class_description(no_docstring_class)

        assert description == ""

    def test_docstring_preserves_formatting(self):
        """Test that docstring formatting (indentation, newlines) is preserved."""
        class FormattedSchema(BaseCheck):
            """
            Test schema with specific formatting.

                - Indented list item 1
                - Indented list item 2

            Code example:
                result = FormattedSchema().validate()
                assert result.passed
            """

            test_field: str = Field("test", description="Test field")

            def __call__(self) -> dict[str, Any]:
                return {'passed': True}

        description = _extract_class_description(FormattedSchema)

        # Should preserve the indentation and structure relative to base indentation
        assert "- Indented list item 1" in description
        assert "- Indented list item 2" in description
        assert "Code example:" in description
        assert "result = FormattedSchema().validate()" in description

        # Test that the base text is properly dedented but relative indentation is preserved
        lines = description.split('\n')
        assert lines[0] == "Test schema with specific formatting."  # No leading spaces
        assert "    - Indented list item 1" in description  # 4 spaces relative indent preserved
        # 4 spaces relative indent preserved
        assert "    result = FormattedSchema().validate()" in description

    def test_docstring_dedenting_behavior(self):
        """Test that inspect.getdoc() properly handles docstring dedenting."""
        class IndentedSchema(BaseCheck):
            """
            Base description starts here.

                This line is indented 4 spaces.
                This line is also indented 4 spaces.

            Back to base level.
                And indented again.
            """

            test_field: str = Field("test", description="Test field")

            def __call__(self) -> dict[str, Any]:
                return {'passed': True}

        description = _extract_class_description(IndentedSchema)
        lines = description.split('\n')

        # Base lines should have no leading whitespace
        assert lines[0] == "Base description starts here."
        assert lines[5] == "Back to base level."

        # Relatively indented lines should preserve their relative indentation
        assert lines[2] == "    This line is indented 4 spaces."
        assert lines[3] == "    This line is also indented 4 spaces."
        assert lines[6] == "    And indented again."

        # Empty lines should be preserved
        assert lines[1] == ""
        assert lines[4] == ""


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
                assert "description" in version_schema
                assert "fields" in version_schema
                assert isinstance(version_schema["fields"], dict)
                assert isinstance(version_schema["description"], str)

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
