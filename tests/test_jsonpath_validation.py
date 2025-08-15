"""Tests for JSONPath validation functionality."""

from typing import Any, Union
import pytest
from pydantic import BaseModel, ValidationError, Field, field_validator

from flex_evals import (
    JSONPath,
    AttributeExistsCheck,
    ContainsCheck,
    CustomFunctionCheck,
    EqualsCheck,
    ExactMatchCheck,
    IsEmptyCheck,
    LLMJudgeCheck,
    RegexCheck,
    SemanticSimilarityCheck,
    ThresholdCheck,
)
from flex_evals.checks.base import (
    JSONPathBehavior,
    get_jsonpath_behavior,
    validate_jsonpath,
    is_jsonpath_expression,
    _convert_to_jsonpath,
)


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
        class TestModelRequired(BaseModel):
            path: JSONPath = Field(..., description="Required JSONPath field")

        assert get_jsonpath_behavior(TestModelRequired, 'path') == JSONPathBehavior.REQUIRED

        # Test optional JSONPath field (union with JSONPath)
        class TestModelOptional(BaseModel):
            text: str | JSONPath = Field(..., description="Optional JSONPath field")
            value: int | JSONPath = Field(42, description="Optional JSONPath with default")

        assert get_jsonpath_behavior(TestModelOptional, 'text') == JSONPathBehavior.OPTIONAL
        assert get_jsonpath_behavior(TestModelOptional, 'value') == JSONPathBehavior.OPTIONAL

        # Test regular field (no JSONPath)
        class TestModelRegular(BaseModel):
            regular_field: str = Field("default", description="Regular string field")

        assert get_jsonpath_behavior(TestModelRegular, 'regular_field') is None

        # Test non-existent field
        assert get_jsonpath_behavior(TestModelRequired, 'non_existent') is None

    def test_get_jsonpath_behavior_union_types(self):
        """Test the core business logic: detecting optional vs required JSONPath fields."""
        class TestModel(BaseModel):
            # The two patterns that matter for our architecture
            required_jsonpath: JSONPath = Field(..., description="Required JSONPath")
            optional_jsonpath: str | JSONPath = Field(..., description="Optional JSONPath")
            regular_field: str = Field("default", description="Regular field")
            # Test both union syntaxes work
            old_style_union: Union[str, JSONPath] = Field(..., description="Old Union syntax")  # noqa: UP007

        # These are the behaviors that actually matter for check execution
        assert get_jsonpath_behavior(TestModel, 'required_jsonpath') == JSONPathBehavior.REQUIRED
        assert get_jsonpath_behavior(TestModel, 'optional_jsonpath') == JSONPathBehavior.OPTIONAL
        assert get_jsonpath_behavior(TestModel, 'old_style_union') == JSONPathBehavior.OPTIONAL
        assert get_jsonpath_behavior(TestModel, 'regular_field') is None
        assert get_jsonpath_behavior(TestModel, 'nonexistent_field') is None

    def test_get_jsonpath_behavior_complex_multi_type_unions(self):
        """Test complex multi-type unions as found in real check implementations."""
        class ComplexUnionModel(BaseModel):
            # From is_empty.py - very complex union
            multi_value: (
                str | list | dict | set | tuple | int | float | bool | None | JSONPath
            ) = Field(...)

            # From threshold.py - multi-primitive with None
            numeric_value: float | int | JSONPath | None = Field(None)

            # From contains.py - list union with JSONPath
            phrases: str | list[str] | JSONPath = Field(...)

            # Mixed Any patterns from exact_match.py
            any_value: Any | JSONPath = Field(...)

        # All of these should detect OPTIONAL because they contain JSONPath in union
        assert get_jsonpath_behavior(ComplexUnionModel, 'multi_value') == JSONPathBehavior.OPTIONAL
        assert get_jsonpath_behavior(ComplexUnionModel, 'numeric_value') == JSONPathBehavior.OPTIONAL  # noqa: E501
        assert get_jsonpath_behavior(ComplexUnionModel, 'phrases') == JSONPathBehavior.OPTIONAL
        assert get_jsonpath_behavior(ComplexUnionModel, 'any_value') == JSONPathBehavior.OPTIONAL

    def test_get_jsonpath_behavior_optional_and_none_patterns(self):
        """Test Optional and None union patterns with JSONPath."""
        class OptionalModel(BaseModel):
            # Optional with JSONPath
            optional_jsonpath: JSONPath | None = Field(None)

            # Explicit None union with JSONPath
            jsonpath_or_none: JSONPath | None = Field(None)

            # Optional with union containing JSONPath
            optional_union: str | JSONPath | None = Field(None)

            # Three-way union with None
            three_way_union: str | JSONPath | None = Field(None)

        # Optional[JSONPath] should be OPTIONAL (it's JSONPath | None)
        assert get_jsonpath_behavior(OptionalModel, 'optional_jsonpath') == JSONPathBehavior.OPTIONAL  # noqa: E501

        # JSONPath | None should be OPTIONAL
        assert get_jsonpath_behavior(OptionalModel, 'jsonpath_or_none') == JSONPathBehavior.OPTIONAL  # noqa: E501

        # Optional[str | JSONPath] = str | JSONPath | None should be OPTIONAL
        assert get_jsonpath_behavior(OptionalModel, 'optional_union') == JSONPathBehavior.OPTIONAL

        # Three-way union should be OPTIONAL
        assert get_jsonpath_behavior(OptionalModel, 'three_way_union') == JSONPathBehavior.OPTIONAL

    def test_get_jsonpath_behavior_generic_types_with_jsonpath(self):
        """Test generic types containing JSONPath."""
        class GenericModel(BaseModel):
            # List containing JSONPath - should return None (no direct JSONPath support)
            jsonpath_list: list[JSONPath] = Field(default_factory=list)

            # Dict with JSONPath values - should return None
            jsonpath_dict: dict[str, JSONPath] = Field(default_factory=dict)

            # Tuple with JSONPath - should return None
            jsonpath_tuple: tuple[JSONPath, str] = Field(...)

            # Complex nested generic - should return None
            nested_generic: list[dict[str, JSONPath]] = Field(default_factory=list)

        # These should all return None because JSONPath is not directly in the union
        assert get_jsonpath_behavior(GenericModel, 'jsonpath_list') is None
        assert get_jsonpath_behavior(GenericModel, 'jsonpath_dict') is None
        assert get_jsonpath_behavior(GenericModel, 'jsonpath_tuple') is None
        assert get_jsonpath_behavior(GenericModel, 'nested_generic') is None

    def test_get_jsonpath_behavior_forward_references(self):
        """Test forward references and string annotations."""
        class ForwardRefModel(BaseModel):
            # String annotation for JSONPath
            string_jsonpath: "JSONPath" = Field(...)

            # String annotation for union
            string_union: "str | JSONPath" = Field(...)

            # Regular field with string annotation
            string_field: "str" = Field("default")

        # These should still work with string annotations
        assert get_jsonpath_behavior(ForwardRefModel, 'string_jsonpath') == JSONPathBehavior.REQUIRED  # noqa: E501
        assert get_jsonpath_behavior(ForwardRefModel, 'string_union') == JSONPathBehavior.OPTIONAL
        assert get_jsonpath_behavior(ForwardRefModel, 'string_field') is None

    def test_get_jsonpath_behavior_edge_cases(self):
        """Test edge cases and potential failure scenarios."""
        class EdgeCaseModel(BaseModel):
            # Union with duplicate JSONPath (should still work)
            duplicate_jsonpath: JSONPath | str | JSONPath = Field(...)

            # Normal field should work
            normal_field: str = Field("test")

        # Test non-existent field (should handle gracefully)
        assert get_jsonpath_behavior(EdgeCaseModel, 'non_existent_field') is None

        # Should detect JSONPath in duplicate union
        assert get_jsonpath_behavior(EdgeCaseModel, 'duplicate_jsonpath') == JSONPathBehavior.OPTIONAL  # noqa: E501

        # Normal field should work
        assert get_jsonpath_behavior(EdgeCaseModel, 'normal_field') is None

        # Test with None as model_class (edge case - should not crash)
        assert get_jsonpath_behavior(None, 'any_field') is None


class TestConvertToJsonpath:
    """Test the _convert_to_jsonpath helper function."""

    def test_convert_jsonpath_string_valid(self):
        """Test converting valid JSONPath strings."""
        result = _convert_to_jsonpath("$.output.value")
        assert isinstance(result, JSONPath)
        assert result.expression == "$.output.value"

    def test_convert_jsonpath_string_invalid(self):
        """Test converting invalid JSONPath strings raises error."""
        with pytest.raises(ValidationError, match="Invalid JSONPath expression"):
            _convert_to_jsonpath("$.invalid[")

    def test_convert_regular_string_unchanged(self):
        """Test regular strings are returned unchanged."""
        result = _convert_to_jsonpath("regular string")
        assert result == "regular string"

    def test_convert_non_string_unchanged(self):
        """Test non-string values are returned unchanged."""
        values = [42, True, None, [], {}]
        for value in values:
            result = _convert_to_jsonpath(value)
            assert result == value

    def test_convert_existing_jsonpath_unchanged(self):
        """Test existing JSONPath objects are returned unchanged."""
        existing_jsonpath = JSONPath(expression="$.test.value")
        result = _convert_to_jsonpath(existing_jsonpath)
        assert result is existing_jsonpath


class TestFieldValidatorIntegration:
    """Test _convert_to_jsonpath integration with Pydantic field validators."""

    def test_field_validator_with_convert_function(self):
        """Test field validator using _convert_to_jsonpath function."""
        class TestModel(BaseModel):
            text: str | JSONPath = Field(..., description="Test text field")

            @field_validator('text', mode='before')
            @classmethod
            def convert_jsonpath(cls, value: object) -> object | JSONPath:
                return _convert_to_jsonpath(value)

        # Test valid JSONPath conversion
        model1 = TestModel(text="$.output.value")
        assert isinstance(model1.text, JSONPath)
        assert model1.text.expression == "$.output.value"

        # Test regular string unchanged
        model2 = TestModel(text="regular text")
        assert model2.text == "regular text"

        # Test invalid JSONPath raises error
        with pytest.raises(ValidationError, match="Invalid JSONPath expression"):
            TestModel(text="$.invalid[")


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
            phrases=["test"],
        )
        assert isinstance(check.text, JSONPath)
        assert str(check.text) == "$.output.value"

        # Regular string should work too
        check2 = ContainsCheck(
            text="regular text",
            phrases=["test"],
        )
        assert check2.text == "regular text"

        # Invalid JSONPath should fail
        with pytest.raises(ValidationError):
            ContainsCheck(
                text="$.invalid[",
                phrases=["test"],
            )

    def test_exact_match_check_validation(self):
        """Test ExactMatchCheck validates JSONPath correctly."""
        # Valid JSONPath should work
        check = ExactMatchCheck(
            actual="$.output.value",
            expected="$.test_case.expected",
        )
        assert isinstance(check.actual, JSONPath)
        assert isinstance(check.expected, JSONPath)

        # Mixed JSONPath and literal should work
        check2 = ExactMatchCheck(
            actual="$.output.value",
            expected="literal_value",
        )
        assert isinstance(check2.actual, JSONPath)
        assert check2.expected == "literal_value"

    def test_regex_check_validation(self):
        """Test RegexCheck validates JSONPath correctly."""
        # Valid JSONPath should work
        check = RegexCheck(
            text="$.output.value",
            pattern=r"\w+",
        )
        assert isinstance(check.text, JSONPath)
        assert check.pattern == r"\w+"

        # Regular string should work
        check2 = RegexCheck(
            text="regular text",
            pattern="$.test_case.pattern",
        )
        assert check2.text == "regular text"
        assert isinstance(check2.pattern, JSONPath)


class TestJSONPathBehaviorRealWorldValidation:
    """Comprehensive tests against actual check implementations in the codebase."""

    def test_is_empty_check_complex_union(self):
        """Test IsEmpty check's complex multi-type union with JSONPath."""
        # IsEmpty has: str | list | dict | set | tuple | int | float | bool | None | JSONPath
        behavior = get_jsonpath_behavior(IsEmptyCheck, 'value')
        assert behavior == JSONPathBehavior.OPTIONAL

    def test_threshold_check_numeric_unions(self):
        """Test Threshold check's numeric unions with JSONPath."""
        # min_value: float | int | JSONPath | None
        behavior_min = get_jsonpath_behavior(ThresholdCheck, 'min_value')
        assert behavior_min == JSONPathBehavior.OPTIONAL

        # max_value: float | int | JSONPath | None
        behavior_max = get_jsonpath_behavior(ThresholdCheck, 'max_value')
        assert behavior_max == JSONPathBehavior.OPTIONAL

    def test_contains_check_list_union(self):
        """Test Contains check's list union with JSONPath."""
        # phrases: str | list[str] | JSONPath
        behavior = get_jsonpath_behavior(ContainsCheck, 'phrases')
        assert behavior == JSONPathBehavior.OPTIONAL

        # case_sensitive: bool | JSONPath
        behavior_case = get_jsonpath_behavior(ContainsCheck, 'case_sensitive')
        assert behavior_case == JSONPathBehavior.OPTIONAL

    def test_equals_check_complex_types(self):
        """Test Equals check's complex type unions."""
        # actual: str | list | dict | set | tuple | int | float | bool | None | JSONPath
        behavior_actual = get_jsonpath_behavior(EqualsCheck, 'actual')
        assert behavior_actual == JSONPathBehavior.OPTIONAL

        # expected: str | list | dict | set | tuple | int | float | bool | None | JSONPath
        behavior_expected = get_jsonpath_behavior(EqualsCheck, 'expected')
        assert behavior_expected == JSONPathBehavior.OPTIONAL

    def test_semantic_similarity_threshold_config_union(self):
        """Test Semantic Similarity check's custom class union with JSONPath."""
        # threshold: ThresholdConfig | JSONPath | None
        behavior = get_jsonpath_behavior(SemanticSimilarityCheck, 'threshold')
        assert behavior == JSONPathBehavior.OPTIONAL

    def test_llm_judge_any_field(self):
        """Test LLM Judge check's Any field (not JSONPath union)."""
        # llm_function: Any (no JSONPath union)
        behavior = get_jsonpath_behavior(LLMJudgeCheck, 'llm_function')
        assert behavior is None

    def test_custom_function_check_dict_args(self):
        """Test Custom Function check's dict field (not JSONPath union)."""
        # function_args: dict[str, Any] (no JSONPath union)
        behavior = get_jsonpath_behavior(CustomFunctionCheck, 'function_args')
        assert behavior is None

    def test_all_basic_check_text_fields(self):
        """Test that all basic checks have correct JSONPath behavior for text fields."""
        # text: str | JSONPath (optional)
        assert get_jsonpath_behavior(ContainsCheck, 'text') == JSONPathBehavior.OPTIONAL
        assert get_jsonpath_behavior(RegexCheck, 'text') == JSONPathBehavior.OPTIONAL

    def test_all_basic_check_required_paths(self):
        """Test that checks requiring paths have REQUIRED behavior."""
        # path: JSONPath (required)
        assert get_jsonpath_behavior(AttributeExistsCheck, 'path') == JSONPathBehavior.REQUIRED

    def test_python_union_vs_typing_union_consistency(self):
        """Test that both Python 3.10+ and typing.Union syntaxes work identically."""
        class PythonUnionModel(BaseModel):
            field: str | JSONPath = Field(...)

        class TypingUnionModel(BaseModel):
            field: Union[str, JSONPath] = Field(...)  # noqa: UP007

        # Both should detect as OPTIONAL
        assert get_jsonpath_behavior(PythonUnionModel, 'field') == JSONPathBehavior.OPTIONAL
        assert get_jsonpath_behavior(TypingUnionModel, 'field') == JSONPathBehavior.OPTIONAL

    def test_inheritance_behavior(self):
        """Test JSONPath behavior detection works with class inheritance."""
        class BaseCheckModel(BaseModel):
            base_field: str | JSONPath = Field(...)

        class DerivedCheckModel(BaseCheckModel):
            derived_field: JSONPath = Field(...)

        # Both inherited and new fields should work
        assert get_jsonpath_behavior(DerivedCheckModel, 'base_field') == JSONPathBehavior.OPTIONAL
        assert (
            get_jsonpath_behavior(DerivedCheckModel, 'derived_field') == JSONPathBehavior.REQUIRED
        )
