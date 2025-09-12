"""
Comprehensive tests for SchemaCheck implementation.

This module consolidates all tests for the SchemaCheck including:
- Pydantic validation tests
- Implementation execution tests
- Engine integration tests
- Edge cases and error handling

Tests are organized by functionality rather than implementation details.
"""

import json
from typing import Any

import pytest
from pydantic import BaseModel, Field

from flex_evals import (
    SchemaValidationCheck,
    JSONPath,
    EvaluationContext,
    CheckType,
    Status,
    evaluate,
    Check,
    ValidationError,
    TestCase,
    Output,
)
from pydantic import ValidationError as PydanticValidationError


class PersonSchema(BaseModel):
    """Test Pydantic model for schema validation tests with constraints."""

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

VALID_SCHEMA_MODEL = PersonSchema(name="dummy", age=25)
VALID_DATA_MODEL = PersonData(name="Alice", age=30)
INVALID_DATA_MODEL = PersonData(name="Bob", age=-1)


class TestSchemaCheckValidation:
    """Test Pydantic validation and field handling for SchemaCheck."""

    def test_schema_check_creation(self) -> None:
        """Test basic SchemaCheck creation."""
        check = SchemaValidationCheck(
            schema=VALID_SCHEMA_DICT,
            data=VALID_DATA_DICT,
        )

        assert check.json_schema == VALID_SCHEMA_DICT
        assert check.data == VALID_DATA_DICT

    def test_schema_check_with_jsonpath(self) -> None:
        """Test SchemaCheck creation with JSONPath expressions."""
        check = SchemaValidationCheck(
            schema="$.test_case.expected.schema",
            data="$.output.value",
        )

        assert isinstance(check.json_schema, JSONPath)
        assert check.json_schema.expression == "$.test_case.expected.schema"
        assert isinstance(check.data, JSONPath)
        assert check.data.expression == "$.output.value"

    def test_schema_check_with_json_strings(self) -> None:
        """Test SchemaCheck with JSON string inputs."""
        check = SchemaValidationCheck(
            schema=VALID_SCHEMA_JSON,
            data=VALID_DATA_JSON,
        )

        assert check.json_schema == VALID_SCHEMA_JSON
        assert check.data == VALID_DATA_JSON

    def test_schema_check_with_pydantic_models(self) -> None:
        """Test SchemaCheck with Pydantic model inputs."""
        check = SchemaValidationCheck(
            schema=VALID_SCHEMA_MODEL,
            data=VALID_DATA_MODEL,
        )

        assert check.json_schema == VALID_SCHEMA_MODEL
        assert check.data == VALID_DATA_MODEL

    def test_schema_check_comprehensive_jsonpath(self) -> None:
        """Comprehensive JSONPath string conversion and execution test."""
        # Create check with JSONPath fields as strings
        check = SchemaValidationCheck(
            schema="$.test_case.expected.validation_schema",
            data="$.output.value.user_data",
        )

        # Verify conversion happened
        assert isinstance(check.json_schema, JSONPath)
        assert check.json_schema.expression == "$.test_case.expected.validation_schema"
        assert isinstance(check.data, JSONPath)
        assert check.data.expression == "$.output.value.user_data"

        # Test execution with EvaluationContext
        test_case = TestCase(
            id="test_001",
            input="test",
            expected={
                "validation_schema": VALID_SCHEMA_DICT,
            },
        )
        output = Output(value={"user_data": VALID_DATA_DICT})
        context = EvaluationContext(test_case, output)

        result = check.execute(context)
        assert result.status == Status.COMPLETED
        assert result.results["passed"] is True
        assert result.results["validation_errors"] is None
        assert result.resolved_arguments["json_schema"]["value"] == VALID_SCHEMA_DICT
        assert result.resolved_arguments["data"]["value"] == VALID_DATA_DICT

    def test_schema_check_required_fields(self) -> None:
        """Test that required fields are enforced."""
        with pytest.raises(PydanticValidationError):
            SchemaValidationCheck()  # type: ignore

        with pytest.raises(PydanticValidationError):
            SchemaValidationCheck(schema=VALID_SCHEMA_DICT)  # type: ignore

        with pytest.raises(PydanticValidationError):
            SchemaValidationCheck(data=VALID_DATA_DICT)  # type: ignore

    def test_schema_check_type_property(self) -> None:
        """Test SchemaCheck check_type property returns correct type."""
        check = SchemaValidationCheck(schema=VALID_SCHEMA_DICT, data=VALID_DATA_DICT)
        assert check.check_type == CheckType.SCHEMA_VALIDATION

    def test_schema_check_invalid_jsonpath(self) -> None:
        """Test that invalid JSONPath expressions are caught during validation."""
        with pytest.raises(PydanticValidationError, match="Invalid JSONPath expression"):
            SchemaValidationCheck(schema="$.invalid[", data=VALID_DATA_DICT)

        with pytest.raises(PydanticValidationError, match="Invalid JSONPath expression"):
            SchemaValidationCheck(schema=VALID_SCHEMA_DICT, data="$.invalid[")


class TestSchemaCheckExecution:
    """Test SchemaCheck execution logic and __call__ method."""

    def test_schema_check_valid_validation(self) -> None:
        """Test schema validation with valid data."""
        check = SchemaValidationCheck(
            schema=VALID_SCHEMA_DICT,
            data=VALID_DATA_DICT,
        )
        result = check()
        assert result == {
            "passed": True,
            "validation_errors": None,
        }

    def test_schema_check_invalid_validation(self) -> None:
        """Test schema validation with invalid data."""
        check = SchemaValidationCheck(
            schema=VALID_SCHEMA_DICT,
            data=INVALID_DATA_DICT,
        )
        result = check()
        assert result["passed"] is False
        assert result["validation_errors"] is not None
        assert isinstance(result["validation_errors"], list)
        assert len(result["validation_errors"]) > 0
        # Check that error mentions age constraint
        error_str = " ".join(result["validation_errors"])
        assert "age" in error_str.lower()
        assert "minimum" in error_str.lower()

    def test_schema_check_missing_required_field(self) -> None:
        """Test schema validation with missing required field."""
        check = SchemaValidationCheck(
            schema=VALID_SCHEMA_DICT,
            data=MISSING_REQUIRED_DATA_DICT,
        )
        result = check()
        assert result["passed"] is False
        assert result["validation_errors"] is not None
        assert len(result["validation_errors"]) == 1
        assert "'age' is a required property" in result["validation_errors"][0]

    def test_schema_check_json_string_inputs(self) -> None:
        """Test schema validation with JSON string inputs."""
        check = SchemaValidationCheck(
            schema=VALID_SCHEMA_JSON,
            data=VALID_DATA_JSON,
        )
        result = check()
        assert result == {
            "passed": True,
            "validation_errors": None,
        }

        # Test with invalid JSON data
        check = SchemaValidationCheck(
            schema=VALID_SCHEMA_JSON,
            data=INVALID_DATA_JSON,
        )
        result = check()
        assert result["passed"] is False
        assert result["validation_errors"] is not None

    def test_schema_check_pydantic_model_inputs(self) -> None:
        """Test schema validation with Pydantic model inputs."""
        check = SchemaValidationCheck(
            schema=VALID_SCHEMA_MODEL,  # Schema extracted from model
            data=VALID_DATA_MODEL,     # Data extracted from model
        )
        result = check()
        assert result == {
            "passed": True,
            "validation_errors": None,
        }

        # Test with invalid data model
        check = SchemaValidationCheck(
            schema=VALID_SCHEMA_MODEL,
            data=INVALID_DATA_MODEL,   # Age -1 violates schema constraints
        )
        result = check()
        assert result["passed"] is False
        assert result["validation_errors"] is not None

    @pytest.mark.parametrize(("schema", "data", "expected_passed"), [
        # Valid combinations - all schema types with valid data
        (VALID_SCHEMA_DICT, VALID_DATA_DICT, True),
        (VALID_SCHEMA_DICT, VALID_DATA_JSON, True),
        (VALID_SCHEMA_DICT, VALID_DATA_MODEL, True),
        (VALID_SCHEMA_JSON, VALID_DATA_DICT, True),
        (VALID_SCHEMA_JSON, VALID_DATA_JSON, True),
        (VALID_SCHEMA_JSON, VALID_DATA_MODEL, True),
        (VALID_SCHEMA_MODEL, VALID_DATA_DICT, True),
        (VALID_SCHEMA_MODEL, VALID_DATA_JSON, True),
        (VALID_SCHEMA_MODEL, VALID_DATA_MODEL, True),
        # Invalid combinations - all schema types with invalid data
        (VALID_SCHEMA_DICT, INVALID_DATA_DICT, False),
        (VALID_SCHEMA_DICT, INVALID_DATA_JSON, False),
        (VALID_SCHEMA_DICT, INVALID_DATA_MODEL, False),
        (VALID_SCHEMA_JSON, INVALID_DATA_DICT, False),
        (VALID_SCHEMA_JSON, INVALID_DATA_JSON, False),
        (VALID_SCHEMA_JSON, INVALID_DATA_MODEL, False),
        (VALID_SCHEMA_MODEL, INVALID_DATA_DICT, False),
        (VALID_SCHEMA_MODEL, INVALID_DATA_JSON, False),
        (VALID_SCHEMA_MODEL, INVALID_DATA_MODEL, False),
    ])
    def test_schema_check_all_input_combinations(
        self, schema: Any, data: Any, expected_passed: bool,
    ) -> None:
        """Test SchemaCheck with all combinations of input types."""
        # Execute check
        check = SchemaValidationCheck(schema=schema, data=data)
        result = check()

        # Assert results based on expected validity
        if expected_passed:
            assert result["passed"] is True
            assert result["validation_errors"] is None
        else:
            assert result["passed"] is False
            assert result["validation_errors"] is not None
            assert isinstance(result["validation_errors"], list)
            assert len(result["validation_errors"]) > 0


class TestSchemaCheckEngineIntegration:
    """Test SchemaCheck integration with the evaluation engine."""

    def test_schema_check_via_evaluate_dict_inputs(self) -> None:
        """Test SchemaCheck through engine evaluation with dict inputs."""
        test_cases = [
            TestCase(
                id="test_001",
                input="User data",
                expected={"schema": VALID_SCHEMA_DICT},
                checks=[
                    Check(
                        type=CheckType.SCHEMA_VALIDATION,
                        arguments={
                            "schema": "$.test_case.expected.schema",
                            "data": "$.output.value",
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value=VALID_DATA_DICT)]
        results = evaluate(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.summary.error_test_cases == 0
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results["passed"] is True
        assert results.results[0].check_results[0].results["validation_errors"] is None

    def test_schema_check_via_evaluate_invalid_data(self) -> None:
        """Test SchemaCheck through engine evaluation with invalid data."""
        test_cases = [
            TestCase(
                id="test_001",
                input="User data",
                expected={"schema": VALID_SCHEMA_DICT},
                checks=[
                    Check(
                        type=CheckType.SCHEMA_VALIDATION,
                        arguments={
                            "schema": "$.test_case.expected.schema",
                            "data": "$.output.value",
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value=INVALID_DATA_DICT)]
        results = evaluate(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results["passed"] is False
        assert results.results[0].check_results[0].results["validation_errors"] is not None

    def test_schema_check_instance_via_evaluate(self) -> None:
        """Test direct SchemaCheck instance usage in evaluate function."""
        test_cases = [
            TestCase(
                id="test_001",
                input="User data",
                expected={"schema": VALID_SCHEMA_DICT},
                checks=[
                    SchemaValidationCheck(
                        schema="$.test_case.expected.schema",
                        data="$.output.value",
                    ),
                ],
            ),
        ]

        outputs = [Output(value=VALID_DATA_DICT)]
        results = evaluate(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results["passed"] is True

    def test_schema_check_json_string_via_evaluate(self) -> None:
        """Test SchemaCheck with JSON string inputs through engine."""
        test_cases = [
            TestCase(
                id="test_001",
                input="User data",
                expected={
                    "schema": VALID_SCHEMA_JSON,
                    "data": VALID_DATA_JSON,
                },
                checks=[
                    Check(
                        type=CheckType.SCHEMA_VALIDATION,
                        arguments={
                            "schema": "$.test_case.expected.schema",
                            "data": "$.test_case.expected.data",
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="dummy")]  # Not used in this test
        results = evaluate(test_cases, outputs)

        assert results.results[0].check_results[0].results["passed"] is True

    def test_schema_check_complex_nested_schema(self) -> None:
        """Test SchemaCheck with complex nested schema structure."""
        nested_schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "profile": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "settings": {
                                    "type": "object",
                                    "properties": {
                                        "theme": {"type": "string", "enum": ["light", "dark"]},
                                        "notifications": {"type": "boolean"},
                                    },
                                    "required": ["theme", "notifications"],
                                },
                            },
                            "required": ["name", "settings"],
                        },
                    },
                    "required": ["profile"],
                },
            },
            "required": ["user"],
        }

        valid_nested_data = {
            "user": {
                "profile": {
                    "name": "Alice",
                    "settings": {
                        "theme": "dark",
                        "notifications": True,
                    },
                },
            },
        }

        test_cases = [
            TestCase(
                id="test_001",
                input="Complex nested data",
                expected={"schema": nested_schema},
                checks=[
                    Check(
                        type=CheckType.SCHEMA_VALIDATION,
                        arguments={
                            "schema": "$.test_case.expected.schema",
                            "data": "$.output.value",
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value=valid_nested_data)]
        results = evaluate(test_cases, outputs)

        assert results.results[0].check_results[0].results["passed"] is True


class TestSchemaCheckErrorHandling:
    """Test error handling and edge cases for SchemaCheck."""

    def test_schema_check_invalid_json_schema_string(self) -> None:
        """Test SchemaCheck with invalid JSON schema string."""
        check = SchemaValidationCheck(
            schema='{"type": "object", invalid json}',  # Invalid JSON
            data=VALID_DATA_DICT,
        )
        with pytest.raises(ValidationError, match="Schema validation failed"):
            check()

    def test_schema_check_invalid_json_data_string(self) -> None:
        """Test SchemaCheck with invalid JSON data string."""
        check = SchemaValidationCheck(
            schema=VALID_SCHEMA_DICT,
            data='{"name": "Alice", invalid json}',  # Invalid JSON
        )
        with pytest.raises(ValidationError, match="Schema validation failed"):
            check()

    def test_schema_check_non_object_schema(self) -> None:
        """Test SchemaCheck with non-object schema."""
        check = SchemaValidationCheck(
            schema='["not", "an", "object"]',  # Array instead of object
            data=VALID_DATA_DICT,
        )
        with pytest.raises(ValidationError, match="Schema validation failed"):
            check()

    def test_schema_check_unresolved_jsonpath_error(self) -> None:
        """Test RuntimeError when JSONPath fields are not resolved."""
        check = SchemaValidationCheck(
            schema="$.test_case.schema",  # Will be converted to JSONPath
            data=VALID_DATA_DICT,
        )

        # Call directly without resolution should raise RuntimeError
        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'json_schema' field"):
            check()

        check = SchemaValidationCheck(
            schema=VALID_SCHEMA_DICT,
            data="$.output.data",  # Will be converted to JSONPath
        )

        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'data' field"):
            check()

    def test_schema_check_invalid_jsonpath_in_engine(self) -> None:
        """Test that invalid JSONPath expressions are caught during evaluation."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.SCHEMA_VALIDATION,
                        arguments={
                            "schema": "$..[invalid",  # Invalid JSONPath syntax
                            "data": VALID_DATA_DICT,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="test")]

        # Should raise validation error for invalid JSONPath
        with pytest.raises(ValidationError, match="Invalid JSONPath expression"):
            evaluate(test_cases, outputs)

    def test_schema_check_default_results(self) -> None:
        """Test that default_results property returns correct structure."""
        check = SchemaValidationCheck(schema=VALID_SCHEMA_DICT, data=VALID_DATA_DICT)
        default = check.default_results
        assert default == {"passed": False, "validation_errors": None}

    def test_schema_check_empty_schema_and_data(self) -> None:
        """Test schema validation with empty schema and data."""
        empty_schema = {"type": "object"}
        empty_data = {}

        check = SchemaValidationCheck(schema=empty_schema, data=empty_data)
        result = check()
        assert result == {"passed": True, "validation_errors": None}


class TestSchemaCheckJSONPathIntegration:
    """Test SchemaCheck with various JSONPath expressions and data structures."""

    def test_schema_check_nested_jsonpath_extraction(self) -> None:
        """Test SchemaCheck with deeply nested JSONPath expressions."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected={
                    "validation": {
                        "schema": VALID_SCHEMA_DICT,
                        "test_data": VALID_DATA_DICT,
                    },
                },
                checks=[
                    Check(
                        type=CheckType.SCHEMA_VALIDATION,
                        arguments={
                            "schema": "$.test_case.expected.validation.schema",
                            "data": "$.test_case.expected.validation.test_data",
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="dummy")]  # Not used in this test
        results = evaluate(test_cases, outputs)

        assert results.results[0].check_results[0].results["passed"] is True

    def test_schema_check_array_access_jsonpath(self) -> None:
        """Test SchemaCheck with JSONPath array access."""
        schemas_array = [VALID_SCHEMA_DICT, {"type": "string"}]
        data_array = [VALID_DATA_DICT, {"extra": "data"}]

        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected={
                    "schemas": schemas_array,
                    "test_data": data_array,
                },
                checks=[
                    Check(
                        type=CheckType.SCHEMA_VALIDATION,
                        arguments={
                            "schema": "$.test_case.expected.schemas[0]",  # First schema
                            "data": "$.test_case.expected.test_data[0]",    # First data
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="dummy")]
        results = evaluate(test_cases, outputs)

        assert results.results[0].check_results[0].results["passed"] is True
