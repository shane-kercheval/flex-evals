"""
Generate JSON schemas for all registered check types and versions.

Provides dynamic schema generation by combining registry information (versions, async status)
with schema class introspection (field definitions, types, descriptions).
"""

from typing import Any, Union, get_args, get_origin
import types
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from .registry import list_registered_checks, get_check_info, get_latest_version
from .schemas.check import SchemaCheck, get_jsonpath_behavior


def _get_schema_class_for_check_type(check_type: str, version: str) -> type[SchemaCheck]:
    """
    Find the schema class for a given check type and version.

    This searches through all SchemaCheck subclasses to find one that matches
    the check_type and version exactly.

    Args:
        check_type: String identifier for the check type
        version: Version string to match against schema VERSION attribute

    Returns:
        Schema class that matches the check_type and version

    Raises:
        ValueError: If no schema class found for the given check_type and version
    """
    # Get all SchemaCheck subclasses
    def get_all_subclasses(cls) -> set[type]:  # noqa: ANN001
        """Recursively get all subclasses of a class."""
        subclasses = set(cls.__subclasses__())
        for subclass in list(subclasses):
            subclasses.update(get_all_subclasses(subclass))
        return subclasses

    schema_classes = get_all_subclasses(SchemaCheck)
    matching_classes = []

    # Find all schema classes matching this check_type
    for schema_class in schema_classes:
        try:
            if hasattr(schema_class, 'CHECK_TYPE') and str(schema_class.CHECK_TYPE) == check_type:
                matching_classes.append(schema_class)
        except Exception:
            # Skip classes that have issues with CHECK_TYPE
            continue

    if not matching_classes:
        raise ValueError(f"No schema class found for check type '{check_type}'")

    # Look for exact version match
    for schema_class in matching_classes:
        try:
            if hasattr(schema_class, 'VERSION') and str(schema_class.VERSION) == version:
                return schema_class
        except Exception:
            # Skip classes that have issues with VERSION
            continue

    # No exact version match found
    available_versions = []
    for schema_class in matching_classes:
        try:
            if hasattr(schema_class, 'VERSION'):
                available_versions.append(str(schema_class.VERSION))
        except Exception:
            pass

    raise ValueError(
        f"No schema class found for check type '{check_type}' version '{version}'. "
        f"Available versions: {available_versions}",
    )


def _is_nullable_type(annotation: Any) -> bool:  # noqa: ANN401
    """Check if a type annotation is nullable (Union[T, None] or T | None)."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Handle Python 3.10+ union syntax (str | None) - uses types.UnionType
    if origin is types.UnionType:
        return len(args) == 2 and type(None) in args

    # Handle typing.Union syntax (Union[str, None], Optional[str])
    if origin is Union:
        return len(args) == 2 and type(None) in args

    return False


def _extract_field_schema(
        field_name: str,
        field_info: FieldInfo,
        model_class: type,
    ) -> dict[str, Any]:
    """Extract schema information for a single field."""
    field_schema = {
        "type": _get_python_type_string(field_info.annotation),
        "nullable": _is_nullable_type(field_info.annotation),
        "description": field_info.description or "",
    }

    # Add default value if field is optional (has actual default)
    if field_info.default is not PydanticUndefined:
        # Handle special pydantic defaults
        if hasattr(field_info.default, '__name__'):
            # It's a callable default, describe it
            field_schema["default"] = f"<function: {field_info.default.__name__}>"
        else:
            field_schema["default"] = field_info.default

    # Add JSONPath behavior if present
    jsonpath_behavior = get_jsonpath_behavior(model_class, field_name)
    if jsonpath_behavior:
        field_schema["jsonpath"] = jsonpath_behavior.value

    # Add field constraints from FieldInfo
    if hasattr(field_info, 'constraints') and field_info.constraints:
        constraints = {}
        for constraint_name, constraint_value in field_info.constraints.items():
            if constraint_value is not None:
                constraints[constraint_name] = constraint_value
        if constraints:
            field_schema["constraints"] = constraints

    # Add validation from field annotations
    if hasattr(field_info, 'json_schema_extra') and field_info.json_schema_extra:
        extra = field_info.json_schema_extra
        if isinstance(extra, dict):
            # Merge relevant extra information
            for key, value in extra.items():
                if key not in field_schema and key != "jsonpath":  # jsonpath already handled
                    field_schema[key] = value

    return field_schema


def _get_python_type_string(annotation: Any) -> str:  # noqa: ANN401, PLR0911, PLR0912
    """Convert Python type annotation to a string representation."""
    if annotation is None:
        return "any"

    # Handle basic types
    if annotation is str:
        return "string"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    if annotation is dict:
        return "object"
    if annotation is list:
        return "array"

    # Handle generic types (List[str], Dict[str, Any], etc.)
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is list:
        if args:
            item_type = _get_python_type_string(args[0])
            return f"array<{item_type}>"
        return "array"
    if origin is dict:
        if len(args) >= 2:
            key_type = _get_python_type_string(args[0])
            value_type = _get_python_type_string(args[1])
            return f"object<{key_type},{value_type}>"
        return "object"
    if origin is tuple:
        if args:
            type_strings = [_get_python_type_string(arg) for arg in args]
            return f"tuple<{','.join(type_strings)}>"
        return "tuple"

    # Handle Python 3.10+ union syntax (str | None) - uses types.UnionType
    if origin is types.UnionType:
        types_list = [_get_python_type_string(arg) for arg in args if arg is not type(None)]
        if len(types_list) == 1:
            # This is a nullable type (T | None) - return just the base type
            return types_list[0]
        return f"union<{','.join(types_list)}>"

    # Handle typing.Union syntax (Union[str, None], Optional[str])
    if origin is Union:
        types_list = [_get_python_type_string(arg) for arg in args if arg is not type(None)]
        if len(types_list) == 1:
            # This is a nullable type (Optional[T]) - return just the base type
            return types_list[0]
        return f"union<{','.join(types_list)}>"

    # Handle Literal types
    if hasattr(annotation, '__origin__') and str(annotation.__origin__) == 'typing.Literal':
        if args:
            return f"literal<{','.join(str(arg) for arg in args)}>"
        return "literal"

    # Handle enum types
    if hasattr(annotation, '__bases__') and any(hasattr(base, '__members__') for base in annotation.__bases__):  # noqa: E501
        # This is likely an enum
        try:
            values = list(annotation.__members__.keys())
            return f"enum<{','.join(values)}>"
        except Exception:
            pass

    # Fallback to clean class name for custom types
    if hasattr(annotation, '__name__'):
        return annotation.__name__  # Keep original case for class names

    return str(annotation).lower()


def _get_version_schemas_for_check_type(check_type: str) -> dict[str, dict[str, Any]]:
    """
    Get schemas for all versions of a check type.

    For each registered version, attempts to find a matching schema class.
    If a version-specific schema class exists, it uses that; otherwise falls back
    to the default schema class for the check type.
    """
    version_schemas = {}

    # Get registry information for all versions
    registered_checks = list_registered_checks()
    if check_type not in registered_checks:
        return {}

    # For each version, try to find the appropriate schema class
    for version in registered_checks[check_type]:
        registry_info = get_check_info(check_type, version)

        # Try to get version-specific schema class
        try:
            schema_class = _get_schema_class_for_check_type(check_type, version)
        except ValueError:
            # If no schema class found, create minimal schema from registry info only
            version_schemas[version] = {
                "version": version,
                "is_async": registry_info["is_async"],
                "fields": {},
                "note": "No schema class found - registry-only information",
            }
            continue

        # Extract fields from schema class
        fields_schema = {}
        for field_name, field_info in schema_class.model_fields.items():
            # Skip metadata field as it's handled specially
            if field_name == "metadata":
                continue
            fields_schema[field_name] = _extract_field_schema(field_name, field_info, schema_class)

        version_schemas[version] = {
            "version": version,
            "is_async": registry_info["is_async"],
            "fields": fields_schema,
            "schema_class": schema_class.__name__,
        }

    return version_schemas


def generate_checks_schema(include_latest_only: bool = False) -> dict[str, Any]:
    """
    Generate complete schema for all registered check types and versions.

    Args:
        include_latest_only: If True, only include the latest version of each check type

    Returns:
        Dict with complete schema information for all checks:
        {
            "check_type": {
                "version": {
                    "version": "1.0.0",
                    "is_async": false,
                    "fields": {
                        "field_name": {
                            "type": "string",
                            "required": true,
                            "description": "...",
                            "jsonpath": "optional"
                        }
                    }
                }
            }
        }
    """
    all_schemas = {}

    # Get all registered check types
    registered_checks = list_registered_checks()

    for check_type in registered_checks:
        if include_latest_only:
            # Only include latest version
            try:
                latest_version = get_latest_version(check_type)
                version_schemas = _get_version_schemas_for_check_type(check_type)
                if latest_version in version_schemas:
                    all_schemas[check_type] = {latest_version: version_schemas[latest_version]}
            except Exception:
                # Skip if there's an issue getting latest version
                continue
        else:
            # Include all versions
            version_schemas = _get_version_schemas_for_check_type(check_type)
            if version_schemas:
                all_schemas[check_type] = version_schemas

    return all_schemas


def generate_check_schema(check_type: str, version: str | None = None) -> dict[str, Any] | None:
    """
    Generate schema for a specific check type and version.

    Args:
        check_type: The check type identifier
        version: Specific version, or None for latest

    Returns:
        Schema dict for the specific check/version, or None if not found
    """
    if version is None:
        try:
            version = get_latest_version(check_type)
        except ValueError:
            return None

    version_schemas = _get_version_schemas_for_check_type(check_type)
    return version_schemas.get(version)
