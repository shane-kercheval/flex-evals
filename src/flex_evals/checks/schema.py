"""Schema Check - validates data against a JSON schema."""

from typing import Any
from pydantic import Field, field_validator

from .base import BaseCheck, JSONPath, _convert_to_jsonpath
from ..registry import register
from ..exceptions import ValidationError
from ..constants import CheckType
from ..utils.schema_validation import validate_json_schema, JSONLike


@register(CheckType.SCHEMA, version='1.0.0')
class SchemaCheck(BaseCheck):
    """
    Validates data against a JSON schema.

    The schema can be provided as:
    - A Pydantic model instance (schema extracted from model)
    - A dictionary containing JSON schema
    - A JSON string containing schema

    The data to validate can be provided as:
    - A Pydantic model instance (data extracted from model)
    - A dictionary
    - A JSON string
    """

    reference_schema: JSONLike | JSONPath = Field(
        ...,
        description="JSON schema for validation (dict, JSON string, or Pydantic model instance)",
        alias="schema",  # Allow 'schema' in arguments but store as 'reference_schema'
    )
    data: JSONLike | JSONPath = Field(
        ...,
        description=(
            "Data to validate against the schema "
            "(dict, JSON string, or Pydantic model instance)"
        ),
    )

    @field_validator('reference_schema', 'data', mode='before')
    @classmethod
    def convert_jsonpath(cls, value: object) -> object | JSONPath:
        """Convert JSONPath-like strings to JSONPath objects."""
        return _convert_to_jsonpath(value)

    @property
    def default_results(self) -> dict[str, Any]:
        """Return default results structure for schema checks on error."""
        return {'passed': False, 'validation_errors': None}

    def __call__(self) -> dict[str, Any]:
        """
        Execute schema validation check using resolved Pydantic fields.

        All JSONPath objects should have been resolved by execute() before this is called.

        Returns:
            Dictionary with
                - 'passed' key indicating validation result
                - 'validation_errors' list if validation failed

        Raises:
            RuntimeError: If any field contains unresolved JSONPath objects
            ValidationError: If schema or data cannot be processed
        """
        # Validate that all fields are resolved (no JSONPath objects remain)
        if isinstance(self.reference_schema, JSONPath):
            raise RuntimeError(
                f"JSONPath not resolved for 'reference_schema' field: {self.reference_schema}",
            )
        if isinstance(self.data, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'data' field: {self.data}")

        try:
            # Use the utility function to validate data against schema
            is_valid, validation_errors = validate_json_schema(self.data, self.reference_schema)

            return {
                'passed': is_valid,
                'validation_errors': validation_errors,
            }

        except Exception as e:
            # Convert any exceptions to ValidationError for consistent error handling
            raise ValidationError(f"Schema validation failed: {e}") from e
