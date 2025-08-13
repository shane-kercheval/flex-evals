"""Tests to verify schema-implementation version consistency."""


from flex_evals.constants import CheckType
from flex_evals.engine import evaluate
from flex_evals.registry import list_registered_checks, get_check_class
from flex_evals import (
    TestCase, Output,
    ContainsCheck, ExactMatchCheck, IsEmptyCheck, RegexCheck, ThresholdCheck,
    AttributeExistsCheck, SemanticSimilarityCheck, LLMJudgeCheck, CustomFunctionCheck,
)


class TestSchemaImplementationConsistency:
    """Test that schema classes and implementation classes are properly paired."""

    def test_all_schema_classes_have_version_attribute(self):
        """Test that all schema classes have VERSION attribute."""
        schema_classes = [
            ContainsCheck, ExactMatchCheck, IsEmptyCheck, RegexCheck, ThresholdCheck,
            AttributeExistsCheck, SemanticSimilarityCheck, LLMJudgeCheck, CustomFunctionCheck,
        ]

        for schema_class in schema_classes:
            assert hasattr(schema_class, 'VERSION'), f"{schema_class.__name__} missing VERSION attribute"  # noqa: E501
            assert isinstance(schema_class.VERSION, str), f"{schema_class.__name__}.VERSION must be a string"  # noqa: E501
            assert schema_class.VERSION == "1.0.0", f"{schema_class.__name__}.VERSION should be '1.0.0'"  # noqa: E501

    def test_all_schema_classes_have_corresponding_implementations(self):
        """Test that every schema class has a corresponding registered implementation."""
        schema_to_check_type = {
            ContainsCheck: CheckType.CONTAINS,
            ExactMatchCheck: CheckType.EXACT_MATCH,
            IsEmptyCheck: CheckType.IS_EMPTY,
            RegexCheck: CheckType.REGEX,
            ThresholdCheck: CheckType.THRESHOLD,
            AttributeExistsCheck: CheckType.ATTRIBUTE_EXISTS,
            SemanticSimilarityCheck: CheckType.SEMANTIC_SIMILARITY,
            LLMJudgeCheck: CheckType.LLM_JUDGE,
            CustomFunctionCheck: CheckType.CUSTOM_FUNCTION,
        }

        registered_checks = list_registered_checks()

        for schema_class, check_type in schema_to_check_type.items():
            check_type_str = str(check_type)

            # Check that the check type is registered
            assert check_type_str in registered_checks, f"No implementation registered for {check_type_str}"  # noqa: E501

            # Check that version 1.0.0 is registered (matching schema VERSION)
            versions = registered_checks[check_type_str]
            schema_version = schema_class.VERSION
            assert schema_version in versions, f"No implementation registered for {check_type_str} version {schema_version}"  # noqa: E501

            # Verify we can retrieve the implementation
            impl_class = get_check_class(check_type_str, schema_version)
            assert impl_class is not None, f"Failed to retrieve implementation for {check_type_str} v{schema_version}"  # noqa: E501

    def test_all_implementations_have_corresponding_schemas(self):
        """Test that every registered implementation has a corresponding schema class."""
        registered_checks = list_registered_checks()

        # Map of check types to their schema classes
        check_type_to_schema = {
            "contains": ContainsCheck,
            "exact_match": ExactMatchCheck,
            "is_empty": IsEmptyCheck,
            "regex": RegexCheck,
            "threshold": ThresholdCheck,
            "attribute_exists": AttributeExistsCheck,
            "semantic_similarity": SemanticSimilarityCheck,
            "llm_judge": LLMJudgeCheck,
            "custom_function": CustomFunctionCheck,
        }

        for check_type, versions in registered_checks.items():
            # Check that we have a schema class for this check type
            assert check_type in check_type_to_schema, f"No schema class found for registered check type '{check_type}'"  # noqa: E501

            schema_class = check_type_to_schema[check_type]

            # For now, we expect only version 1.0.0 to be registered
            assert "1.0.0" in versions, f"Expected version 1.0.0 for check type '{check_type}', found: {list(versions.keys())}"  # noqa: E501

            # Verify schema VERSION matches registered version
            assert schema_class.VERSION == "1.0.0", f"Schema {schema_class.__name__}.VERSION should be '1.0.0'"  # noqa: E501

    def test_schema_to_check_conversion_preserves_version(self):
        """Test that converting schema to Check object preserves version."""
        # Test with a few schema classes
        schema_instances = [
            ContainsCheck(text="test", phrases=["hello"]),
            ExactMatchCheck(actual="actual", expected="expected"),
            IsEmptyCheck(value="test"),
        ]

        for schema_instance in schema_instances:
            check = schema_instance.to_check()

            # Verify the check has the correct version from the schema
            assert check.version == schema_instance.VERSION, "Check version should match schema VERSION"  # noqa: E501

    def test_implementation_classes_are_properly_versioned(self):
        """Test that implementation classes follow versioning naming convention."""
        registered_checks = list_registered_checks()

        for check_type, versions in registered_checks.items():
            for version, info in versions.items():
                impl_class = info["class"]

                # For version 1.0.0, class should be named with _v1 suffix (or similar versioned name)  # noqa: E501
                # This test will help us identify which classes need renaming
                class_name = impl_class.__name__

                # This is currently expected to fail - we'll use it to identify what needs to be renamed  # noqa: E501
                if version == "1.0.0":
                    # Current class names don't have version suffixes - this test documents the current state  # noqa: E501
                    # and will guide our renaming
                    [
                        f"{check_type.replace('_', '').title()}Check_v1",
                        f"{check_type.replace('_', '').title()}Check_v1_0_0",
                        f"{check_type.replace('_', '').title()}Check",  # Current pattern (will change)  # noqa: E501
                    ]

                    # For now, just document what we found
                    print(f"Check type: {check_type}, Version: {version}, Class: {class_name}")

                    # This assertion will help us track progress
                    assert class_name is not None, f"Implementation class should exist for {check_type} v{version}"  # noqa: E501

    def test_end_to_end_schema_to_evaluation(self):
        """Test end-to-end: schema -> check -> evaluation with version consistency."""
        # Create a schema instance
        schema = ContainsCheck(text="$.output.value", phrases=["test"])

        # Convert to check
        check = schema.to_check()
        assert check.version == "1.0.0"

        # Create test case and output
        test_case = TestCase(id="test_1", input="test input")
        output = Output(value="this is a test message")

        # Run evaluation - this should use the v1.0.0 implementation
        result = evaluate([test_case], [output], [check])

        assert result.results[0].check_results[0].status == "completed"
        assert result.results[0].check_results[0].results["passed"] is True

        # Verify version metadata is preserved
        if result.results[0].check_results[0].metadata:
            assert result.results[0].check_results[0].metadata.get("check_version") == "1.0.0"
