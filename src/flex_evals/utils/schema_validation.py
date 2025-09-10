"""Schema validation utilities for JSON schema validation."""

import json
from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from pydantic import BaseModel


JSONLike = str | dict[str, Any] | BaseModel
SchemaLike = JSONLike | type[BaseModel]


def _coerce(
        value: SchemaLike, *, as_schema: bool,
    ) -> dict[str, Any] | list[Any] | str | int | float | bool | None:
    """
    Coerce input to a Python object suitable for validation.

    - If BaseModel instance: dump to schema (for schema) or to data (for data).
    - If BaseModel class: extract JSON schema (only valid for schema).
    - If str: json.loads.
    - Otherwise: pass through.
    Additionally, if as_schema=True, ensure result is a mapping.

    Args:
        value: The input value to coerce
        as_schema: Whether to treat the value as a schema (True) or data (False)

    Returns:
        Coerced Python object suitable for validation

    Raises:
        TypeError: When schema is not a JSON object after coercion
        json.JSONDecodeError: When string cannot be parsed as JSON
    """
    if isinstance(value, type) and issubclass(value, BaseModel):
        # Pydantic model class - extract schema
        if not as_schema:
            raise TypeError("Pydantic model class can only be used as schema, not as data")
        value = value.model_json_schema()
    elif isinstance(value, BaseModel):
        # Pydantic model instance
        value = value.__class__.model_json_schema() if as_schema else value.model_dump()
    elif isinstance(value, str):
        value = json.loads(value)

    if as_schema and not isinstance(value, Mapping):
        raise TypeError("Schema must be a JSON object (mapping) after coercion.")
    return value


def validate_json_schema(data: JSONLike, schema: SchemaLike) -> tuple[bool, list[str] | None]:
    """
    Validate `data` against JSON `schema`.

    Args:
        data: The data to validate (BaseModel **instance**, dict, or JSON string)
        schema: The schema to validate against (BaseModel **class**, dict, or JSON string)

    Returns:
        Tuple of (is_valid, error_list). If invalid, error_list contains all validation issues.
        If valid, error_list is None.

    Raises:
        TypeError: When schema is not a JSON object after coercion
        json.JSONDecodeError: When JSON strings cannot be parsed

    Example:
        >>> schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        >>> data = {"name": "Alice"}
        >>> is_valid, errors = validate_json_schema(data, schema)
        >>> print(is_valid)
        True
    """
    schema_obj = _coerce(schema, as_schema=True)
    data_obj = _coerce(data, as_schema=False)

    # Draft202012Validator is the implementation of the JSON Schema 2020-12 draft, which is the
    # latest stable version of the standard.
    validator = Draft202012Validator(schema_obj, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(data_obj), key=lambda e: list(e.path))

    if not errors:
        return True, None

    validation_errors = []
    for err in errors:
        path_elems = [str(p) for p in err.path]
        path = "/" + "/".join(path_elems) if path_elems else "(root)"
        validation_errors.append(f"{path}: {err.message}")
    return False, validation_errors
