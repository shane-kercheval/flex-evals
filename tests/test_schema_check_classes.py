"""Unit tests for individual SchemaCheck classes."""

import pytest
from pydantic import BaseModel, Field, ValidationError

from flex_evals import (
    ContainsCheck, ExactMatchCheck, RegexCheck, ThresholdCheck,
    SemanticSimilarityCheck, LLMJudgeCheck, CustomFunctionCheck,
    RegexFlags, ThresholdConfig, CheckType, SimilarityMetric,
)


class TestContainsCheck:
    """Test ContainsCheck schema class."""

    def test_contains_check_creation(self):
        """Test basic ContainsCheck creation."""
        check = ContainsCheck(
            text="$.output.value",
            phrases=["hello", "world"],
        )

        assert check.text == "$.output.value"
        assert check.phrases == ["hello", "world"]
        assert check.case_sensitive is True
        assert check.negate is False
        assert check.version is None

    def test_contains_check_with_options(self):
        """Test ContainsCheck with all options."""
        check = ContainsCheck(
            text="$.output.value",
            phrases=["hello"],
            case_sensitive=False,
            negate=True,
            version="1.0.0",
        )

        assert check.case_sensitive is False
        assert check.negate is True
        assert check.version == "1.0.0"

    def test_contains_check_validation_empty_text(self):
        """Test ContainsCheck validation for empty text."""
        with pytest.raises(ValidationError):
            ContainsCheck(text="", phrases=["hello"])

    def test_contains_check_validation_empty_phrases(self):
        """Test ContainsCheck validation for empty phrases."""
        with pytest.raises(ValidationError):
            ContainsCheck(text="$.value", phrases=[])

    def test_contains_check_validation_invalid_phrases(self):
        """Test ContainsCheck validation for invalid phrases."""
        with pytest.raises(ValidationError, match="all phrases must be non-empty strings"):
            ContainsCheck(text="$.value", phrases=["valid", ""])

    def test_contains_check_to_check_conversion(self):
        """Test ContainsCheck to_check() conversion."""
        schema_check = ContainsCheck(
            text="$.output.value",
            phrases=["hello", "world"],
            case_sensitive=False,
            negate=True,
            version="1.0.0",
        )

        check = schema_check.to_check()

        assert check.type == CheckType.CONTAINS
        assert check.arguments["text"] == "$.output.value"
        assert check.arguments["phrases"] == ["hello", "world"]
        assert check.arguments["case_sensitive"] is False
        assert check.arguments["negate"] is True
        assert check.version == "1.0.0"

    def test_contains_check_type_property(self):
        """Test ContainsCheck check_type property."""
        check = ContainsCheck(text="$.value", phrases=["test"])
        assert check.check_type == CheckType.CONTAINS


class TestExactMatchCheck:
    """Test ExactMatchCheck schema class."""

    def test_exact_match_check_creation(self):
        """Test basic ExactMatchCheck creation."""
        check = ExactMatchCheck(
            actual="$.output.value",
            expected="$.expected",
        )

        assert check.actual == "$.output.value"
        assert check.expected == "$.expected"
        assert check.case_sensitive is True
        assert check.negate is False

    def test_exact_match_check_with_options(self):
        """Test ExactMatchCheck with all options."""
        check = ExactMatchCheck(
            actual="$.output.value",
            expected="$.expected",
            case_sensitive=False,
            negate=True,
            version="2.0.0",
        )

        assert check.case_sensitive is False
        assert check.negate is True
        assert check.version == "2.0.0"

    def test_exact_match_check_validation_empty_actual(self):
        """Test ExactMatchCheck validation for empty actual."""
        with pytest.raises(ValidationError):
            ExactMatchCheck(actual="", expected="$.expected")

    def test_exact_match_check_validation_empty_expected(self):
        """Test ExactMatchCheck validation for empty expected."""
        with pytest.raises(ValidationError):
            ExactMatchCheck(actual="$.actual", expected="")

    def test_exact_match_check_to_check_conversion(self):
        """Test ExactMatchCheck to_check() conversion."""
        schema_check = ExactMatchCheck(
            actual="$.output.value",
            expected="$.expected",
            case_sensitive=False,
            negate=True,
            version="1.0.0",
        )

        check = schema_check.to_check()

        assert check.type == CheckType.EXACT_MATCH
        assert check.arguments["actual"] == "$.output.value"
        assert check.arguments["expected"] == "$.expected"
        assert check.arguments["case_sensitive"] is False
        assert check.arguments["negate"] is True
        assert check.version == "1.0.0"

    def test_exact_match_check_type_property(self):
        """Test ExactMatchCheck check_type property."""
        check = ExactMatchCheck(actual="$.actual", expected="$.expected")
        assert check.check_type == CheckType.EXACT_MATCH


class TestRegexCheck:
    """Test RegexCheck schema class."""

    def test_regex_check_creation(self):
        """Test basic RegexCheck creation."""
        check = RegexCheck(
            text="$.output.value",
            pattern=r"\d+",
        )

        assert check.text == "$.output.value"
        assert check.pattern == r"\d+"
        assert check.negate is False
        assert check.flags is None

    def test_regex_check_with_flags(self):
        """Test RegexCheck with flags."""
        flags = RegexFlags(case_insensitive=True, multiline=True)
        check = RegexCheck(
            text="$.output.value",
            pattern="hello",
            negate=True,
            flags=flags,
            version="1.0.0",
        )

        assert check.flags.case_insensitive is True
        assert check.flags.multiline is True
        assert check.flags.dot_all is False
        assert check.negate is True
        assert check.version == "1.0.0"

    def test_regex_check_validation_empty_text(self):
        """Test RegexCheck validation for empty text."""
        with pytest.raises(ValidationError):
            RegexCheck(text="", pattern="test")

    def test_regex_check_validation_empty_pattern(self):
        """Test RegexCheck validation for empty pattern."""
        with pytest.raises(ValidationError):
            RegexCheck(text="$.value", pattern="")

    def test_regex_check_to_check_conversion(self):
        """Test RegexCheck to_check() conversion."""
        flags = RegexFlags(case_insensitive=True, dot_all=True)
        schema_check = RegexCheck(
            text="$.output.value",
            pattern=r"test\d+",
            negate=True,
            flags=flags,
            version="1.0.0",
        )

        check = schema_check.to_check()

        assert check.type == CheckType.REGEX
        assert check.arguments["text"] == "$.output.value"
        assert check.arguments["pattern"] == r"test\d+"
        assert check.arguments["negate"] is True
        assert check.arguments["flags"]["case_insensitive"] is True
        assert check.arguments["flags"]["multiline"] is False
        assert check.arguments["flags"]["dot_all"] is True
        assert check.version == "1.0.0"

    def test_regex_check_to_check_conversion_no_flags(self):
        """Test RegexCheck to_check() conversion without flags."""
        schema_check = RegexCheck(
            text="$.output.value",
            pattern="test",
        )

        check = schema_check.to_check()

        assert "flags" not in check.arguments

    def test_regex_check_type_property(self):
        """Test RegexCheck check_type property."""
        check = RegexCheck(text="$.value", pattern="test")
        assert check.check_type == CheckType.REGEX


class TestThresholdCheck:
    """Test ThresholdCheck schema class."""

    def test_threshold_check_creation_min_only(self):
        """Test ThresholdCheck creation with min_value only."""
        check = ThresholdCheck(
            value="$.output.score",
            min_value=80.0,
        )

        assert check.value == "$.output.score"
        assert check.min_value == 80.0
        assert check.max_value is None
        assert check.min_inclusive is True
        assert check.max_inclusive is True
        assert check.negate is False

    def test_threshold_check_creation_max_only(self):
        """Test ThresholdCheck creation with max_value only."""
        check = ThresholdCheck(
            value="$.output.score",
            max_value=90.0,
        )

        assert check.max_value == 90.0
        assert check.min_value is None

    def test_threshold_check_creation_both_bounds(self):
        """Test ThresholdCheck creation with both bounds."""
        check = ThresholdCheck(
            value="$.output.score",
            min_value=80.0,
            max_value=90.0,
            min_inclusive=False,
            max_inclusive=False,
            negate=True,
            version="1.0.0",
        )

        assert check.min_value == 80.0
        assert check.max_value == 90.0
        assert check.min_inclusive is False
        assert check.max_inclusive is False
        assert check.negate is True
        assert check.version == "1.0.0"

    def test_threshold_check_validation_empty_value(self):
        """Test ThresholdCheck validation for empty value."""
        with pytest.raises(ValidationError):
            ThresholdCheck(value="", min_value=80.0)

    def test_threshold_check_validation_no_bounds(self):
        """Test ThresholdCheck validation for no bounds."""
        with pytest.raises(ValueError, match="At least one of 'min_value' or 'max_value' must be specified"):  # noqa: E501
            ThresholdCheck(value="$.value")

    def test_threshold_check_to_check_conversion_min_only(self):
        """Test ThresholdCheck to_check() conversion with min_value only."""
        schema_check = ThresholdCheck(
            value="$.output.score",
            min_value=80.0,
            min_inclusive=False,
            version="1.0.0",
        )

        check = schema_check.to_check()

        assert check.type == CheckType.THRESHOLD
        assert check.arguments["value"] == "$.output.score"
        assert check.arguments["min_value"] == 80.0
        assert "max_value" not in check.arguments
        assert check.arguments["min_inclusive"] is False
        assert check.arguments["max_inclusive"] is True
        assert check.arguments["negate"] is False
        assert check.version == "1.0.0"

    def test_threshold_check_to_check_conversion_both_bounds(self):
        """Test ThresholdCheck to_check() conversion with both bounds."""
        schema_check = ThresholdCheck(
            value="$.output.score",
            min_value=80.0,
            max_value=90.0,
            negate=True,
        )

        check = schema_check.to_check()

        assert check.arguments["min_value"] == 80.0
        assert check.arguments["max_value"] == 90.0
        assert check.arguments["negate"] is True

    def test_threshold_check_type_property(self):
        """Test ThresholdCheck check_type property."""
        check = ThresholdCheck(value="$.value", min_value=0.0)
        assert check.check_type == CheckType.THRESHOLD


class TestSemanticSimilarityCheck:
    """Test SemanticSimilarityCheck schema class."""

    def test_semantic_similarity_check_creation(self):
        """Test basic SemanticSimilarityCheck creation."""
        def mock_embedding_function(text: str):  # noqa: ANN202, ARG001
            return [0.1, 0.2, 0.3]

        check = SemanticSimilarityCheck(
            text="$.output.value",
            reference="$.expected",
            embedding_function=mock_embedding_function,
        )

        assert check.text == "$.output.value"
        assert check.reference == "$.expected"
        assert check.embedding_function == mock_embedding_function
        assert check.similarity_metric == SimilarityMetric.COSINE
        assert check.threshold is None

    def test_semantic_similarity_check_with_threshold(self):
        """Test SemanticSimilarityCheck with threshold."""
        def mock_embedding_function(text: str):  # noqa: ANN202, ARG001
            return [0.1, 0.2, 0.3]

        threshold = ThresholdConfig(min_value=0.8, negate=True)
        check = SemanticSimilarityCheck(
            text="$.output.value",
            reference="$.expected",
            embedding_function=mock_embedding_function,
            similarity_metric=SimilarityMetric.DOT,
            threshold=threshold,
            version="1.0.0",
        )

        assert check.similarity_metric == SimilarityMetric.DOT
        assert check.threshold.min_value == 0.8
        assert check.threshold.negate is True
        assert check.version == "1.0.0"

    def test_semantic_similarity_check_validation_empty_text(self):
        """Test SemanticSimilarityCheck validation for empty text."""
        def mock_embedding_function(text: str):  # noqa: ANN202, ARG001
            return [0.1, 0.2, 0.3]

        with pytest.raises(ValidationError):
            SemanticSimilarityCheck(
                text="",
                reference="$.expected",
                embedding_function=mock_embedding_function,
            )

    def test_semantic_similarity_check_to_check_conversion(self):
        """Test SemanticSimilarityCheck to_check() conversion."""
        def mock_embedding_function(text: str):  # noqa: ANN202, ARG001
            return [0.1, 0.2, 0.3]

        threshold = ThresholdConfig(min_value=0.8, max_value=1.0, negate=False)
        schema_check = SemanticSimilarityCheck(
            text="$.output.value",
            reference="$.expected",
            embedding_function=mock_embedding_function,
            similarity_metric=SimilarityMetric.EUCLIDEAN,
            threshold=threshold,
            version="1.0.0",
        )

        check = schema_check.to_check()

        assert check.type == CheckType.SEMANTIC_SIMILARITY
        assert check.arguments["text"] == "$.output.value"
        assert check.arguments["reference"] == "$.expected"
        assert check.arguments["embedding_function"] == mock_embedding_function
        assert check.arguments["similarity_metric"] == "euclidean"
        assert check.arguments["threshold"]["min_value"] == 0.8
        assert check.arguments["threshold"]["max_value"] == 1.0
        assert check.arguments["threshold"]["negate"] is False
        assert check.version == "1.0.0"

    def test_semantic_similarity_check_to_check_conversion_no_threshold(self):
        """Test SemanticSimilarityCheck to_check() conversion without threshold."""
        def mock_embedding_function(text: str):  # noqa: ANN202, ARG001
            return [0.1, 0.2, 0.3]

        schema_check = SemanticSimilarityCheck(
            text="$.output.value",
            reference="$.expected",
            embedding_function=mock_embedding_function,
        )

        check = schema_check.to_check()

        assert "threshold" not in check.arguments

    def test_semantic_similarity_check_type_property(self):
        """Test SemanticSimilarityCheck check_type property."""
        def mock_embedding_function(text: str):  # noqa: ANN202, ARG001
            return [0.1, 0.2, 0.3]

        check = SemanticSimilarityCheck(
            text="$.text",
            reference="$.ref",
            embedding_function=mock_embedding_function,
        )
        assert check.check_type == CheckType.SEMANTIC_SIMILARITY


class TestLLMJudgeCheck:
    """Test LLMJudgeCheck schema class."""

    def test_llm_judge_check_creation(self):
        """Test basic LLMJudgeCheck creation."""
        class TestResponse(BaseModel):
            score: int = Field(ge=1, le=10)

        def mock_llm_function(prompt: str, response_format: type) -> tuple[BaseModel, dict]:  # noqa: ARG001
            return TestResponse(score=8), {"cost": 0.01}

        check = LLMJudgeCheck(
            prompt="Rate this: {{$.output.value}}",
            response_format=TestResponse,
            llm_function=mock_llm_function,
        )

        assert check.prompt == "Rate this: {{$.output.value}}"
        assert check.response_format == TestResponse
        assert check.llm_function == mock_llm_function

    def test_llm_judge_check_validation_empty_prompt(self):
        """Test LLMJudgeCheck validation for empty prompt."""
        class TestResponse(BaseModel):
            score: int

        def mock_llm_function(prompt: str, response_format: type) -> tuple[BaseModel, dict]:  # noqa: ARG001
            return TestResponse(score=8), {}

        with pytest.raises(ValidationError):
            LLMJudgeCheck(
                prompt="",
                response_format=TestResponse,
                llm_function=mock_llm_function,
            )

    def test_llm_judge_check_to_check_conversion(self):
        """Test LLMJudgeCheck to_check() conversion."""
        class TestResponse(BaseModel):
            score: int = Field(ge=1, le=10)
            reasoning: str

        def mock_llm_function(prompt: str, response_format: type) -> tuple[BaseModel, dict]:  # noqa: ARG001
            return TestResponse(score=8, reasoning="Good"), {"cost": 0.01}

        schema_check = LLMJudgeCheck(
            prompt="Evaluate: {{$.output.value}}",
            response_format=TestResponse,
            llm_function=mock_llm_function,
            version="1.0.0",
        )

        check = schema_check.to_check()

        assert check.type == CheckType.LLM_JUDGE
        assert check.arguments["prompt"] == "Evaluate: {{$.output.value}}"
        assert check.arguments["response_format"] == TestResponse
        assert check.arguments["llm_function"] == mock_llm_function
        assert check.version == "1.0.0"

    def test_llm_judge_check_type_property(self):
        """Test LLMJudgeCheck check_type property."""
        class TestResponse(BaseModel):
            score: int

        def mock_llm_function(prompt: str, response_format: type) -> tuple[BaseModel, dict]:  # noqa: ARG001
            return TestResponse(score=8), {}

        check = LLMJudgeCheck(
            prompt="test",
            response_format=TestResponse,
            llm_function=mock_llm_function,
        )
        assert check.check_type == CheckType.LLM_JUDGE


class TestCustomFunctionCheck:
    """Test CustomFunctionCheck schema class."""

    def test_custom_function_check_creation(self):
        """Test basic CustomFunctionCheck creation."""
        def validation_function(text: str, min_length: int) -> dict:
            return {"passed": len(text) >= min_length}

        check = CustomFunctionCheck(
            validation_function=validation_function,
            function_args={"text": "$.output.value", "min_length": 5},
        )

        assert check.validation_function == validation_function
        assert check.function_args == {"text": "$.output.value", "min_length": 5}

    def test_custom_function_check_with_string_function(self):
        """Test CustomFunctionCheck with string function."""
        function_string = "lambda text, min_length: {'passed': len(text) >= min_length}"

        check = CustomFunctionCheck(
            validation_function=function_string,
            function_args={"text": "$.output.value", "min_length": 10},
            version="1.0.0",
        )

        assert check.validation_function == function_string
        assert check.function_args["min_length"] == 10
        assert check.version == "1.0.0"

    def test_custom_function_check_to_check_conversion(self):
        """Test CustomFunctionCheck to_check() conversion."""
        def validation_function(text: str) -> dict:
            return {"passed": bool(text)}

        schema_check = CustomFunctionCheck(
            validation_function=validation_function,
            function_args={"text": "$.output.value"},
            version="1.0.0",
        )

        check = schema_check.to_check()

        assert check.type == CheckType.CUSTOM_FUNCTION
        assert check.arguments["validation_function"] == validation_function
        assert check.arguments["function_args"] == {"text": "$.output.value"}
        assert check.version == "1.0.0"

    def test_custom_function_check_type_property(self):
        """Test CustomFunctionCheck check_type property."""
        def validation_function(x: int) -> dict:
            return {"passed": x > 0}

        check = CustomFunctionCheck(
            validation_function=validation_function,
            function_args={"x": "$.value"},
        )
        assert check.check_type == CheckType.CUSTOM_FUNCTION


class TestSchemaCheckVersionHandling:
    """Test version handling across all schema check types."""

    def test_default_version_none(self):
        """Test that default version is None for all check types."""
        checks = [
            ContainsCheck(text="$.value", phrases=["test"]),
            ExactMatchCheck(actual="$.actual", expected="$.expected"),
            RegexCheck(text="$.value", pattern="test"),
            ThresholdCheck(value="$.value", min_value=0.0),
        ]

        for check in checks:
            assert check.version is None
            converted = check.to_check()
            assert converted.version is None

    def test_version_preservation(self):
        """Test that version is preserved in conversion."""
        def mock_embedding_function(text: str):  # noqa: ANN202, ARG001
            return [0.1, 0.2, 0.3]

        def mock_validation_function(x: int) -> dict:
            return {"passed": x > 0}

        class TestResponse(BaseModel):
            score: int

        def mock_llm_function(prompt: str, response_format: type) -> tuple[BaseModel, dict]:  # noqa: ARG001
            return TestResponse(score=8), {}

        checks_with_versions = [
            (ContainsCheck(text="$.value", phrases=["test"], version="1.2.3"), "1.2.3"),
            (ExactMatchCheck(actual="$.actual", expected="$.expected", version="2.0.0"), "2.0.0"),
            (RegexCheck(text="$.value", pattern="test", version="1.0.0-beta"), "1.0.0-beta"),
            (ThresholdCheck(value="$.value", min_value=0.0, version="3.1.4"), "3.1.4"),
            (SemanticSimilarityCheck(
                text="$.text", reference="$.ref",
                embedding_function=mock_embedding_function, version="1.1.0"), "1.1.0"),
            (LLMJudgeCheck(
                prompt="test", response_format=TestResponse,
                llm_function=mock_llm_function, version="2.1.0"), "2.1.0"),
            (CustomFunctionCheck(
                validation_function=mock_validation_function,
                function_args={"x": "$.value"}, version="1.0.0"), "1.0.0"),
        ]

        for schema_check, expected_version in checks_with_versions:
            assert schema_check.version == expected_version
            converted = schema_check.to_check()
            assert converted.version == expected_version
