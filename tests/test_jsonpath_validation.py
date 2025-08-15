"""Tests for JSONPath validation functionality."""

import pytest
from pydantic import ValidationError, Field, field_validator

from flex_evals.checks.base import (
    JSONPathBehavior,
    get_jsonpath_behavior,
    validate_jsonpath,
    is_jsonpath_expression,
    JSONPathValidatedModel,
    JSONPath,
)
from flex_evals.checks.attribute_exists import AttributeExistsCheck
from flex_evals.checks.contains import ContainsCheck
from flex_evals.checks.exact_match import ExactMatchCheck
from flex_evals.checks.regex import RegexCheck


class TestJSONPathValidation:
    """Test JSONPath expression validation."""

    def test_validate_jsonpath_valid_expressions(self):
        """Test that valid JSONPath expressions pass validation."""
        valid_expressions = [
            "$.output.value",
            "$.test_case.expected",
            "$.test_case.input.data[0]",
            "$.root..nested",
            "$[0].item",
        ]

        for expr in valid_expressions:
            assert validate_jsonpath(expr), f"Should be valid JSONPath: {expr}"

    def test_validate_jsonpath_invalid_expressions(self):
        """Test that invalid JSONPath expressions fail validation."""
        invalid_expressions = [
            "not_a_jsonpath",
            "$invalid[",
            "$.invalid]",
            "$.invalid...",
            "",
            "regular string",
            "output.value",  # Missing $ prefix
        ]

        for expr in invalid_expressions:
            assert not validate_jsonpath(expr), f"Should be invalid JSONPath: {expr}"

    def test_validate_jsonpath_robustness(self):
        """Test that validate_jsonpath handles edge cases robustly."""
        # Non-string inputs should return False, not crash
        assert not validate_jsonpath(None)
        assert not validate_jsonpath(123)
        assert not validate_jsonpath([])
        assert not validate_jsonpath({})
        assert not validate_jsonpath(True)
        
        # Empty string and non-$ prefixed should return False
        assert not validate_jsonpath("")
        assert not validate_jsonpath("root.value")
        
        # Basic valid JSONPath patterns should work
        assert validate_jsonpath("$")
        assert validate_jsonpath("$.value")
        assert validate_jsonpath("$[0]")
        assert validate_jsonpath("$.*")

    def test_is_jsonpath_expression_detection(self):
        """Test detection of JSONPath-like expressions."""
        jsonpath_expressions = [
            "$.output.value",
            "$.test_case.input",
            "$.complex[0].data",
        ]

        non_jsonpath_expressions = [
            "regular string",
            "not-a-jsonpath",
            "\\$.escaped.literal",  # Escaped literal
            "",
            123,  # Non-string
            None,  # Non-string
        ]

        for expr in jsonpath_expressions:
            assert is_jsonpath_expression(expr), f"Should detect as JSONPath: {expr}"

        for expr in non_jsonpath_expressions:
            assert not is_jsonpath_expression(expr), f"Should NOT detect as JSONPath: {expr}"

    def test_is_jsonpath_expression_escape_mechanism(self):
        """Test the escape mechanism for literal values that start with $."""
        # The key business value: users can escape literal $ values
        assert is_jsonpath_expression("$.value") is True      # JSONPath
        assert is_jsonpath_expression("\\$.literal") is False  # Escaped literal
        
        # Non-string inputs should be handled gracefully
        assert is_jsonpath_expression(None) is False
        assert is_jsonpath_expression(123) is False

    def test_get_jsonpath_behavior_with_type_annotations(self):
        """Test reading JSONPath behavior from type annotations."""
        # Test required JSONPath field (exactly JSONPath type)
        class TestModelRequired(JSONPathValidatedModel):
            path: JSONPath = Field(..., description="Required JSONPath field")

        assert get_jsonpath_behavior(TestModelRequired, 'path') == JSONPathBehavior.REQUIRED

        # Test optional JSONPath field (union with JSONPath)
        class TestModelOptional(JSONPathValidatedModel):
            text: str | JSONPath = Field(..., description="Optional JSONPath field")
            value: int | JSONPath = Field(42, description="Optional JSONPath with default")

        assert get_jsonpath_behavior(TestModelOptional, 'text') == JSONPathBehavior.OPTIONAL
        assert get_jsonpath_behavior(TestModelOptional, 'value') == JSONPathBehavior.OPTIONAL

        # Test regular field (no JSONPath)
        class TestModelRegular(JSONPathValidatedModel):
            regular_field: str = Field("default", description="Regular string field")

        assert get_jsonpath_behavior(TestModelRegular, 'regular_field') is None

        # Test non-existent field
        assert get_jsonpath_behavior(TestModelRequired, 'non_existent') is None

    def test_get_jsonpath_behavior_union_types(self):
        """Test the core business logic: detecting optional vs required JSONPath fields."""
        from typing import Union
        
        class TestModel(JSONPathValidatedModel):
            # The two patterns that matter for our architecture
            required_jsonpath: JSONPath = Field(..., description="Required JSONPath")
            optional_jsonpath: str | JSONPath = Field(..., description="Optional JSONPath")
            regular_field: str = Field("default", description="Regular field")
            
            # Test both union syntaxes work
            old_style_union: Union[str, JSONPath] = Field(..., description="Old Union syntax")
        
        # These are the behaviors that actually matter for check execution
        assert get_jsonpath_behavior(TestModel, 'required_jsonpath') == JSONPathBehavior.REQUIRED
        assert get_jsonpath_behavior(TestModel, 'optional_jsonpath') == JSONPathBehavior.OPTIONAL  
        assert get_jsonpath_behavior(TestModel, 'old_style_union') == JSONPathBehavior.OPTIONAL
        assert get_jsonpath_behavior(TestModel, 'regular_field') is None
        assert get_jsonpath_behavior(TestModel, 'nonexistent_field') is None


class TestJSONPathValidatedModel:
    """Test the base JSONPathValidatedModel class."""

    def test_required_jsonpath_valid(self):
        """Test that valid JSONPath expressions are accepted for required fields."""
        class TestModel(JSONPathValidatedModel):
            path: JSONPath = Field(..., description="Test path")

        # Should work with valid JSONPath
        model = TestModel(path=JSONPath(expression="$.output.value"))
        assert str(model.path) == "$.output.value"

    def test_required_jsonpath_invalid(self):
        """Test that invalid JSONPath expressions are rejected for required fields."""
        class TestModel(JSONPathValidatedModel):
            path: JSONPath = Field(..., description="Test path")

        # Should fail with invalid JSONPath
        with pytest.raises(ValidationError, match="Invalid JSONPath expression"):
            TestModel(path=JSONPath(expression="invalid_path"))

    def test_optional_jsonpath_valid(self):
        """Test that valid JSONPath expressions are accepted for optional fields."""
        class TestModel(JSONPathValidatedModel):
            text: str | JSONPath = Field("default", description="Test text")

            @field_validator('text', mode='before')
            @classmethod
            def convert_jsonpath(cls, v):
                if isinstance(v, str) and v.startswith('$.'):
                    return JSONPath(expression=v)
                return v

        # Should work with valid JSONPath
        model = TestModel(text="$.output.value")
        assert isinstance(model.text, JSONPath)
        assert str(model.text) == "$.output.value"

        # Should work with regular string
        model2 = TestModel(text="regular string")
        assert model2.text == "regular string"

    def test_optional_jsonpath_invalid(self):
        """Test that invalid JSONPath expressions are rejected for optional fields."""
        class TestModel(JSONPathValidatedModel):
            text: str | JSONPath = Field("default", description="Test text")

            @field_validator('text', mode='before')
            @classmethod
            def convert_jsonpath(cls, v):
                if isinstance(v, str) and v.startswith('$.'):
                    return JSONPath(expression=v)
                return v

        # Should fail with invalid JSONPath that looks like JSONPath
        with pytest.raises(ValidationError, match="Invalid JSONPath expression"):
            TestModel(text="$.invalid[")

    def test_optional_jsonpath_no_validation_for_literals(self):
        """Test that regular strings don't get JSONPath validation."""
        class TestModel(JSONPathValidatedModel):
            text: str | JSONPath = Field("default", description="Test text")

            @field_validator('text', mode='before')
            @classmethod
            def convert_jsonpath(cls, v):
                if isinstance(v, str) and v.startswith('$.'):
                    return JSONPath(expression=v)
                return v

        # Should work fine with regular string
        model = TestModel(text="just a regular string")
        assert model.text == "just a regular string"

    def test_field_without_description(self):
        """Test field without description still works."""
        class TestModel(JSONPathValidatedModel):
            text: str | JSONPath = Field(...)

            @field_validator('text', mode='before')
            @classmethod
            def convert_jsonpath(cls, v):
                if isinstance(v, str) and v.startswith('$.'):
                    return JSONPath(expression=v)
                return v

        model = TestModel(text="$.output.value")
        assert isinstance(model.text, JSONPath)


class TestJSONPathBehaviorInferenceWithRealChecks:
    """Test JSONPath behavior inference with real check classes."""

    def test_attribute_exists_check_required_behavior(self):
        """Test AttributeExistsCheck has required JSONPath behavior."""
        # AttributeExistsCheck has path: JSONPath (required)
        behavior = get_jsonpath_behavior(AttributeExistsCheck, 'path')
        assert behavior == JSONPathBehavior.REQUIRED

    def test_contains_check_optional_behavior(self):
        """Test ContainsCheck has optional JSONPath behavior."""
        # ContainsCheck has text: str | JSONPath (optional)
        behavior = get_jsonpath_behavior(ContainsCheck, 'text')
        assert behavior == JSONPathBehavior.OPTIONAL

        # ContainsCheck has phrases: str | list[str] | JSONPath (optional)
        behavior = get_jsonpath_behavior(ContainsCheck, 'phrases')
        assert behavior == JSONPathBehavior.OPTIONAL

        # ContainsCheck has case_sensitive: bool | JSONPath (optional)
        behavior = get_jsonpath_behavior(ContainsCheck, 'case_sensitive')
        assert behavior == JSONPathBehavior.OPTIONAL

    def test_exact_match_check_optional_behavior(self):
        """Test ExactMatchCheck has optional JSONPath behavior."""
        # ExactMatchCheck has actual: Any | JSONPath (optional)
        behavior = get_jsonpath_behavior(ExactMatchCheck, 'actual')
        assert behavior == JSONPathBehavior.OPTIONAL

        # ExactMatchCheck has expected: Any | JSONPath (optional)
        behavior = get_jsonpath_behavior(ExactMatchCheck, 'expected')
        assert behavior == JSONPathBehavior.OPTIONAL

    def test_regex_check_optional_behavior(self):
        """Test RegexCheck has optional JSONPath behavior."""
        # RegexCheck has text: str | JSONPath (optional)
        behavior = get_jsonpath_behavior(RegexCheck, 'text')
        assert behavior == JSONPathBehavior.OPTIONAL

        # RegexCheck has pattern: str | JSONPath (optional)
        behavior = get_jsonpath_behavior(RegexCheck, 'pattern')
        assert behavior == JSONPathBehavior.OPTIONAL


class TestJSONPathObjectBehavior:
    """Test JSONPath object creation and validation."""

    def test_jsonpath_object_creation_valid(self):
        """Test creating JSONPath objects with valid expressions."""
        path = JSONPath(expression="$.output.value")
        assert path.expression == "$.output.value"
        assert str(path) == "$.output.value"
        assert repr(path) == "JSONPath('$.output.value')"

    def test_jsonpath_object_creation_invalid(self):
        """Test creating JSONPath objects with invalid expressions fails."""
        with pytest.raises(ValueError, match="Invalid JSONPath expression"):
            JSONPath(expression="invalid_path")

        with pytest.raises(ValueError, match="Invalid JSONPath expression"):
            JSONPath(expression="$.invalid[")

    def test_jsonpath_object_str_repr(self):
        """Test string representation of JSONPath objects."""
        path = JSONPath(expression="$.test.path")
        assert str(path) == "$.test.path"
        assert repr(path) == "JSONPath('$.test.path')"


class TestIntegrationWithCheckClasses:
    """Test integration of JSONPath validation with actual check classes."""

    def test_attribute_exists_check_validation(self):
        """Test AttributeExistsCheck validates JSONPath correctly."""
        # Valid JSONPath should work
        check = AttributeExistsCheck(path="$.output.value")
        assert isinstance(check.path, JSONPath)
        assert str(check.path) == "$.output.value"

        # Invalid JSONPath should fail
        with pytest.raises(ValidationError):
            AttributeExistsCheck(path="invalid_path")

    def test_contains_check_validation(self):
        """Test ContainsCheck validates JSONPath correctly."""
        # Valid JSONPath should work
        check = ContainsCheck(
            text="$.output.value",
            phrases=["test"]
        )
        assert isinstance(check.text, JSONPath)
        assert str(check.text) == "$.output.value"

        # Regular string should work too
        check2 = ContainsCheck(
            text="regular text",
            phrases=["test"]
        )
        assert check2.text == "regular text"

        # Invalid JSONPath should fail
        with pytest.raises(ValidationError):
            ContainsCheck(
                text="$.invalid[",
                phrases=["test"]
            )

    def test_exact_match_check_validation(self):
        """Test ExactMatchCheck validates JSONPath correctly."""
        # Valid JSONPath should work
        check = ExactMatchCheck(
            actual="$.output.value",
            expected="$.test_case.expected"
        )
        assert isinstance(check.actual, JSONPath)
        assert isinstance(check.expected, JSONPath)

        # Mixed JSONPath and literal should work
        check2 = ExactMatchCheck(
            actual="$.output.value",
            expected="literal_value"
        )
        assert isinstance(check2.actual, JSONPath)
        assert check2.expected == "literal_value"

    def test_regex_check_validation(self):
        """Test RegexCheck validates JSONPath correctly."""
        # Valid JSONPath should work
        check = RegexCheck(
            text="$.output.value",
            pattern=r"\w+"
        )
        assert isinstance(check.text, JSONPath)
        assert check.pattern == r"\w+"

        # Regular string should work
        check2 = RegexCheck(
            text="regular text",
            pattern="$.test_case.pattern"
        )
        assert check2.text == "regular text"
        assert isinstance(check2.pattern, JSONPath)