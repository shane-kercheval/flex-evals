"""Tests for JSONPath validation functionality."""

import pytest
from pydantic import ValidationError

from flex_evals.schemas.check import (
    JSONPathBehavior,
    RequiredJSONPath,
    OptionalJSONPath,
    get_jsonpath_behavior,
    validate_jsonpath,
    is_jsonpath_expression,
    JSONPathValidatedModel,
)
from flex_evals.schemas.checks.attribute_exists import AttributeExistsCheck
from flex_evals.schemas.checks.contains import ContainsCheck
from flex_evals.schemas.checks.exact_match import ExactMatchCheck
from flex_evals.schemas.checks.regex import RegexCheck
from flex_evals.schemas.checks.threshold import ThresholdCheck
from flex_evals.schemas.checks.semantic_similarity import SemanticSimilarityCheck
from flex_evals.schemas.checks.is_empty import IsEmptyCheck


class TestJSONPathUtilityFunctions:
    """Test utility functions for JSONPath validation."""

    def test_validate_jsonpath_valid_expressions(self):
        """Test that valid JSONPath expressions pass validation."""
        valid_expressions = [
            "$.output.value",
            "$.test_case.input",
            "$.output.metadata.score",
            "$.test_case.expected",
            "$['field-with-dash']",
            "$.output.array[0]",
            "$.output.nested.type",
        ]

        for expr in valid_expressions:
            assert validate_jsonpath(expr), f"Should validate: {expr}"

    def test_validate_jsonpath_invalid_expressions(self):
        """Test that invalid JSONPath expressions fail validation."""
        invalid_expressions = [
            "$.output.",  # Trailing dot
            "$.output[",  # Unclosed bracket
            "",  # Empty string
            "$.output.value.result[?(@.type=unclosed_string",  # Unclosed filter
        ]

        for expr in invalid_expressions:
            assert not validate_jsonpath(expr), f"Should not validate: {expr}"

    def test_is_jsonpath_expression_detection(self):
        """Test JSONPath expression detection logic."""
        # Should be detected as JSONPath
        jsonpath_expressions = [
            "$.output.value",
            "$.test_case.input",
        ]

        for expr in jsonpath_expressions:
            assert is_jsonpath_expression(expr), f"Should detect as JSONPath: {expr}"

        # Should NOT be detected as JSONPath
        non_jsonpath_expressions = [
            "regular text",
            "\\$.escaped.literal",  # Escaped literal
            "text with $.embedded jsonpath",
            "$5.99",  # Dollar amount (doesn't start with $.)
            "",
            123,  # Non-string
            None,  # Non-string
        ]

        for expr in non_jsonpath_expressions:
            assert not is_jsonpath_expression(expr), f"Should NOT detect as JSONPath: {expr}"

    def test_get_jsonpath_behavior(self):
        """Test reading JSONPath behavior from field metadata."""
        # Create a test model with JSONPath fields
        class TestModel(JSONPathValidatedModel):
            required_field: str = RequiredJSONPath("Test required JSONPath")
            optional_field: str = OptionalJSONPath("Test optional JSONPath")
            regular_field: str = "regular field"

        # Test required field
        assert get_jsonpath_behavior(TestModel, 'required_field') == JSONPathBehavior.REQUIRED

        # Test optional field
        assert get_jsonpath_behavior(TestModel, 'optional_field') == JSONPathBehavior.OPTIONAL

        # Test regular field (no JSONPath behavior)
        assert get_jsonpath_behavior(TestModel, 'regular_field') is None

        # Test non-existent field
        assert get_jsonpath_behavior(TestModel, 'non_existent') is None


class TestJSONPathValidatedModel:
    """Test the base JSONPathValidatedModel class."""

    def test_required_jsonpath_valid(self):
        """Test that valid JSONPath expressions are accepted for required fields."""
        class TestModel(JSONPathValidatedModel):
            path: str = RequiredJSONPath("Test path")

        # Should work with valid JSONPath
        model = TestModel(path="$.output.value")
        assert model.path == "$.output.value"

    def test_required_jsonpath_invalid(self):
        """Test that invalid JSONPath expressions are rejected for required fields."""
        class TestModel(JSONPathValidatedModel):
            path: str = RequiredJSONPath("Test path")

        # Should fail with invalid JSONPath (doesn't start with $)
        with pytest.raises(ValidationError, match="requires valid JSONPath expression"):
            TestModel(path="invalid_jsonpath")

        # Should fail with invalid JSONPath (trailing dot)
        with pytest.raises(ValidationError, match="requires valid JSONPath expression"):
            TestModel(path="$.output.")  # Trailing dot

    def test_optional_jsonpath_valid(self):
        """Test that valid JSONPath expressions are accepted for optional fields."""
        class TestModel(JSONPathValidatedModel):
            text: str = OptionalJSONPath("Test text")

        # Should work with valid JSONPath
        model = TestModel(text="$.output.value")
        assert model.text == "$.output.value"

        # Should work with literal text
        model = TestModel(text="literal text")
        assert model.text == "literal text"

    def test_optional_jsonpath_invalid(self):
        """Test that invalid JSONPath expressions are rejected for optional fields."""
        class TestModel(JSONPathValidatedModel):
            text: str = OptionalJSONPath("Test text")

        # Should fail with invalid JSONPath that looks like JSONPath
        with pytest.raises(ValidationError, match="appears to be JSONPath but is invalid"):
            TestModel(text="$.invalid.")

        # Should work with literal text that doesn't look like JSONPath
        model = TestModel(text="This costs $5.99 today")
        assert model.text == "This costs $5.99 today"

    def test_escaped_literals(self):
        """Test that escaped literals are handled correctly."""
        class TestModel(JSONPathValidatedModel):
            text: str = OptionalJSONPath("Test text")

        # Escaped literal should be accepted (not validated as JSONPath)
        model = TestModel(text="\\$.literal.text")
        assert model.text == "\\$.literal.text"


class TestCheckClassValidation:
    """Test JSONPath validation in actual check classes."""

    def test_attribute_exists_check_valid(self):
        """Test AttributeExistsCheck with valid JSONPath."""
        check = AttributeExistsCheck(path="$.output.error")
        assert check.path == "$.output.error"

    def test_attribute_exists_check_invalid(self):
        """Test AttributeExistsCheck with invalid JSONPath."""
        with pytest.raises(ValidationError, match="requires valid JSONPath expression"):
            AttributeExistsCheck(path="invalid_path")  # Doesn't start with $

    def test_contains_check_valid_jsonpath(self):
        """Test ContainsCheck with valid JSONPath for text field."""
        check = ContainsCheck(text="$.output.value", phrases=["error"])
        assert check.text == "$.output.value"

    def test_contains_check_valid_literal(self):
        """Test ContainsCheck with literal text."""
        check = ContainsCheck(text="Some literal text", phrases=["literal"])
        assert check.text == "Some literal text"

    def test_contains_check_invalid_jsonpath(self):
        """Test ContainsCheck with invalid JSONPath."""
        with pytest.raises(ValidationError, match="appears to be JSONPath but is invalid"):
            ContainsCheck(text="$.invalid.", phrases=["test"])

    def test_exact_match_check_valid(self):
        """Test ExactMatchCheck with valid JSONPath expressions."""
        check = ExactMatchCheck(actual="$.output.value", expected="$.test_case.expected")
        assert check.actual == "$.output.value"
        assert check.expected == "$.test_case.expected"

    def test_exact_match_check_mixed(self):
        """Test ExactMatchCheck with mixed JSONPath and literal."""
        check = ExactMatchCheck(actual="$.output.value", expected="expected literal")
        assert check.actual == "$.output.value"
        assert check.expected == "expected literal"

    def test_exact_match_check_invalid(self):
        """Test ExactMatchCheck with invalid JSONPath."""
        with pytest.raises(ValidationError, match="appears to be JSONPath but is invalid"):
            ExactMatchCheck(actual="$.invalid.", expected="valid text")

    def test_regex_check_valid(self):
        """Test RegexCheck with valid JSONPath and literal."""
        check = RegexCheck(text="$.output.message", pattern=r"error.*")
        assert check.text == "$.output.message"

        check = RegexCheck(text="literal text", pattern=r"literal")
        assert check.text == "literal text"

    def test_threshold_check_valid(self):
        """Test ThresholdCheck with valid JSONPath."""
        check = ThresholdCheck(value="$.output.score", min_value=0.8)
        assert check.value == "$.output.score"

    def test_threshold_check_invalid(self):
        """Test ThresholdCheck with invalid JSONPath."""
        with pytest.raises(ValidationError, match="appears to be JSONPath but is invalid"):
            ThresholdCheck(value="$.invalid.", min_value=0.5)

    def test_semantic_similarity_check_valid(self):
        """Test SemanticSimilarityCheck with valid JSONPath expressions."""
        def dummy_embedding(text):  # noqa
            return [0.1, 0.2, 0.3]

        check = SemanticSimilarityCheck(
            text="$.output.response",
            reference="$.test_case.expected",
            embedding_function=dummy_embedding,
        )
        assert check.text == "$.output.response"
        assert check.reference == "$.test_case.expected"

    def test_is_empty_check_valid(self):
        """Test IsEmptyCheck with valid JSONPath."""
        check = IsEmptyCheck(value="$.output.error")
        assert check.value == "$.output.error"

        check = IsEmptyCheck(value="literal value")
        assert check.value == "literal value"


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_dollar_amounts_not_validated_as_jsonpath(self):
        """Test that dollar amounts are not incorrectly validated as JSONPath."""
        class TestModel(JSONPathValidatedModel):
            text: str = OptionalJSONPath("Test text")

        # Dollar amounts should be treated as literals
        model = TestModel(text="This item costs $5.99")
        assert model.text == "This item costs $5.99"

        model = TestModel(text="Price: $10.00")
        assert model.text == "Price: $10.00"

    def test_non_string_fields_ignored(self):
        """Test that non-string fields are ignored by validation."""
        class TestModel(JSONPathValidatedModel):
            number_field: int = RequiredJSONPath("This should be ignored")
            boolean_field: bool = OptionalJSONPath("This should be ignored")

        # Should not validate non-string fields
        model = TestModel(number_field=42, boolean_field=True)
        assert model.number_field == 42
        assert model.boolean_field is True

    def test_empty_string_handling(self):
        """Test handling of empty strings."""
        class TestModel(JSONPathValidatedModel):
            required_field: str = RequiredJSONPath("Test required")
            optional_field: str = OptionalJSONPath("Test optional", min_length=0)  # Allow empty

        # Empty string should fail for required JSONPath
        with pytest.raises(ValidationError, match="requires valid JSONPath expression"):
            TestModel(required_field="", optional_field="test")

        # Empty string should pass for optional (not detected as JSONPath) when min_length=0
        model = TestModel(required_field="$.valid.path", optional_field="")
        assert model.optional_field == ""

    def test_error_messages_contain_field_names(self):
        """Test that error messages include field names for debugging."""
        class TestModel(JSONPathValidatedModel):
            my_path: str = RequiredJSONPath("Test path")

        with pytest.raises(ValidationError) as exc_info:
            TestModel(my_path="invalid")  # Doesn't start with $

        error_str = str(exc_info.value)
        assert "my_path" in error_str
        assert "requires valid JSONPath expression" in error_str
        assert "invalid" in error_str
