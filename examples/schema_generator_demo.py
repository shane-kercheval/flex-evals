#!/usr/bin/env python3
"""
Demonstration of dynamic schema generation for flex-evals checks.

This script showcases the new schema generation functionality that provides
complete schema information for all registered check types and versions.

Run via: `uv run examples/schema_generator_demo.py`
"""

import json
from flex_evals import generate_checks_schema, generate_check_schema


def demo_single_check_schema() -> None:
    """Demonstrate generating schema for a single check type."""
    print("=" * 60)
    print("SINGLE CHECK SCHEMA GENERATION")
    print("=" * 60)

    # Get schema for ContainsCheck
    print("📋 ContainsCheck Schema (v1.0.0):")
    contains_schema = generate_check_schema("contains", "1.0.0")
    print(json.dumps(contains_schema, indent=2))

    print("\n" + "-" * 40)

    # Get schema for latest version (same as above since only 1.0.0 exists)
    print("📋 ContainsCheck Schema (latest version):")
    contains_latest = generate_check_schema("contains", version=None)
    print(f"Latest version: {contains_latest['version']}")
    print(f"Is async: {contains_latest['is_async']}")
    print(f"Fields: {list(contains_latest['fields'].keys())}")

    print("\n" + "-" * 40)

    # Show detailed field information
    print("🔍 Detailed Field Information:")
    for field_name, field_info in contains_latest['fields'].items():
        print(f"  • {field_name}:")
        print(f"    - Type: {field_info['type']}")
        print(f"    - Nullable: {field_info['nullable']}")
        is_required = 'default' not in field_info
        print(f"    - Required: {is_required}")
        if 'default' in field_info:
            print(f"    - Default: {field_info['default']}")
        if 'jsonpath' in field_info:
            print(f"    - JSONPath: {field_info['jsonpath']}")
        print(f"    - Description: {field_info['description']}")


def demo_all_checks_schema() -> None:
    """Demonstrate generating schemas for all registered checks."""
    print("\n\n" + "=" * 60)
    print("ALL CHECKS SCHEMA GENERATION")
    print("=" * 60)

    # Get schemas for all check types (latest versions only)
    all_schemas = generate_checks_schema(include_latest_only=True)

    print(f"📊 Found {len(all_schemas)} registered check types:")

    for check_type, versions in sorted(all_schemas.items()):
        version = next(iter(versions.keys()))  # Only one version per check type
        schema = versions[version]

        print(f"\n🔧 {check_type.upper()} (v{version})")
        print(f"   └── Async: {'Yes' if schema['is_async'] else 'No'}")
        print(f"   └── Fields: {len(schema['fields'])}")

        # Show field summary
        required_fields = []
        optional_fields = []
        for field_name, field_info in schema['fields'].items():
            # Field is required if it has no default value
            if 'default' not in field_info:
                required_fields.append(field_name)
            else:
                optional_fields.append(field_name)

        if required_fields:
            print(f"   └── Required: {', '.join(required_fields)}")
        if optional_fields:
            print(f"   └── Optional: {', '.join(optional_fields)}")


def demo_schema_for_api() -> None:
    """Demonstrate how this would be used in an API context."""
    print("\n\n" + "=" * 60)
    print("API USAGE DEMONSTRATION")
    print("=" * 60)

    print("🌐 Example: API endpoint /api/v1/checks/schema")
    print("\n# Get all check schemas (what an API might return)")

    # This is what an API endpoint might return
    api_response = {
        "status": "success",
        "data": {
            "check_types": generate_checks_schema(include_latest_only=True),
        },
        "meta": {
            "total_types": len(generate_checks_schema(include_latest_only=True)),
            "generated_at": "2025-01-01T00:00:00Z",
        },
    }

    # Show a subset for brevity
    print("Response structure:")
    print(f"- status: {api_response['status']}")
    print(f"- total_types: {api_response['meta']['total_types']}")
    print(f"- check_types: {list(api_response['data']['check_types'].keys())}")

    print("\n🔍 Example: Get specific check schema")
    print("GET /api/v1/checks/contains/schema?version=1.0.0")

    specific_response = {
        "status": "success",
        "data": generate_check_schema("contains", "1.0.0"),
    }

    print("Response:")
    print(json.dumps(specific_response, indent=2)[:500] + "...")


def demo_yaml_generation() -> None:
    """Demonstrate how schemas could be used for YAML config generation."""
    print("\n\n" + "=" * 60)
    print("YAML CONFIGURATION GENERATION")
    print("=" * 60)

    print("📝 Using schema information to generate YAML templates:")

    # Get contains check schema
    schema = generate_check_schema("contains", "1.0.0")

    print(f"\n# Based on {schema['schema_class']} v{schema['version']}")
    print("checks:")
    print("  - type: contains")
    print(f"    version: \"{schema['version']}\"  # Optional, uses latest if omitted")
    print("    arguments:")

    # Generate YAML template based on field schema
    for field_name, field_info in schema['fields'].items():
        field_type = field_info['type']
        # Field is required if it has no default value
        required = 'default' not in field_info
        description = field_info['description']

        if required:
            if field_type == "string":
                print(f"      {field_name}: \"example_value\"  # {description}")
            elif field_type.startswith("array"):
                print(f"      {field_name}: [\"example\"]  # {description}")
            elif field_type == "boolean":
                print(f"      {field_name}: true  # {description}")
            elif field_type == "integer":
                print(f"      {field_name}: 42  # {description}")
            elif field_type == "number":
                print(f"      {field_name}: 3.14  # {description}")
        else:
            default_val = field_info.get('default', 'null')
            print(f"      # {field_name}: {default_val}  # Optional: {description}")


def demo_validation_helper() -> None:
    """Demonstrate how schemas could be used for validation."""
    print("\n\n" + "=" * 60)
    print("VALIDATION HELPER DEMONSTRATION")
    print("=" * 60)

    print("🔍 Using schema for runtime validation:")

    # Example check configuration that might come from user input
    user_check_config = {
        "type": "contains",
        "arguments": {
            "text": "$.output.value",
            "phrases": ["hello", "world"],
            "case_sensitive": False,
        },
    }

    print(f"\nUser-provided config: {json.dumps(user_check_config, indent=2)}")

    # Validate against schema
    check_type = user_check_config["type"]
    schema = generate_check_schema(check_type, None)  # Get latest version

    if not schema:
        print(f"❌ Unknown check type: {check_type}")
        return

    print(f"\n✅ Validating against {schema['schema_class']} v{schema['version']}:")

    # Check required fields
    provided_fields = set(user_check_config.get("arguments", {}).keys())
    # Fields are required if they have no default value
    required_fields = {name for name, info in schema['fields'].items() if 'default' not in info}

    missing_fields = required_fields - provided_fields
    if missing_fields:
        print(f"❌ Missing required fields: {missing_fields}")
    else:
        print("✅ All required fields present")

    # Check for unknown fields
    valid_fields = set(schema['fields'].keys())
    unknown_fields = provided_fields - valid_fields
    if unknown_fields:
        print(f"⚠️  Unknown fields (will be ignored): {unknown_fields}")
    else:
        print("✅ All fields are valid")

    # Check field types (simplified)
    print("\n🔍 Field type validation:")
    for field_name, field_value in user_check_config.get("arguments", {}).items():
        if field_name in schema['fields']:
            expected_type = schema['fields'][field_name]['type']
            actual_type = type(field_value).__name__

            # Simplified type checking
            type_map = {
                "string": "str",
                "boolean": "bool",
                "integer": "int",
                "number": ["int", "float"],
                "array<string>": "list",
            }

            expected_python_type = type_map.get(expected_type, expected_type)

            if isinstance(expected_python_type, list):
                type_match = actual_type in expected_python_type
            else:
                type_match = actual_type == expected_python_type

            if type_match:
                print(f"  ✅ {field_name}: {actual_type} matches {expected_type}")
            else:
                print(f"  ❌ {field_name}: {actual_type} doesn't match {expected_type}")


def main() -> None:
    """Run all demonstrations."""
    print("🚀 FLEX-EVALS DYNAMIC SCHEMA GENERATION DEMO")
    print("=" * 60)
    print("This demo shows how to dynamically generate complete schema")
    print("information for all registered check types and versions.")

    demo_single_check_schema()
    demo_all_checks_schema()
    demo_schema_for_api()
    demo_yaml_generation()
    demo_validation_helper()

    print("\n\n" + "=" * 60)
    print("✨ SUMMARY")
    print("=" * 60)
    print("The schema generation system provides:")
    print("• 📋 Complete field definitions with types and descriptions")
    print("• 🔄 Version-aware schema information")
    print("• ⚡ Async/sync detection from registry")
    print("• 🎯 JSONPath behavior specification")
    print("• 🌐 API-ready JSON serialization")
    print("• ✅ Runtime validation capabilities")
    print("\nUse cases:")
    print("• API documentation generation")
    print("• Interactive form builders")
    print("• Configuration file templates")
    print("• Runtime validation")
    print("• IDE autocompletion")


if __name__ == "__main__":
    main()
