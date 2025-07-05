"""Integration tests for SchemaCheck classes with the evaluate() function."""

import pytest
from pydantic import BaseModel, Field, ValidationError

from flex_evals import (
    evaluate, ContainsCheck, ExactMatchCheck, RegexCheck, ThresholdCheck,
    CustomFunctionCheck,
    Check, TestCase, Output,
    RegexFlags,
)


class SimpleResponse(BaseModel):
    """Simple response format for LLMJudgeCheck tests."""

    score: int = Field(ge=1, le=10)
    reasoning: str


class TestSchemaCheckBasicUsage:
    """Test basic usage of schema checks with evaluate()."""

    def test_evaluate_with_contains_check(self):
        """Test evaluate() works with ContainsCheck."""
        test_case = TestCase(id="test1", input="user input", expected="expected output")
        output = Output(value="This contains the expected phrase")

        check = ContainsCheck(
            text="$.output.value",
            phrases=["expected phrase"],
        )

        result = evaluate([test_case], [output], [check])

        assert result.status == "completed"
        assert len(result.results) == 1
        assert len(result.results[0].check_results) == 1
        assert result.results[0].check_results[0].results["passed"] is True

    def test_evaluate_with_exact_match_check(self):
        """Test evaluate() works with ExactMatchCheck."""
        test_case = TestCase(id="test1", input="user input", expected="hello world")
        output = Output(value="hello world")

        check = ExactMatchCheck(
            actual="$.output.value",
            expected="$.test_case.expected",
        )

        result = evaluate([test_case], [output], [check])

        assert result.status == "completed"
        assert len(result.results) == 1
        assert result.results[0].check_results[0].results["passed"] is True

    def test_evaluate_with_regex_check(self):
        """Test evaluate() works with RegexCheck."""
        test_case = TestCase(id="test1", input="user input", expected="expected")
        output = Output(value="Hello World 123")

        check = RegexCheck(
            text="$.output.value",
            pattern=r"\d+",
        )

        result = evaluate([test_case], [output], [check])

        assert result.status == "completed"
        assert result.results[0].check_results[0].results["passed"] is True

    def test_evaluate_with_threshold_check(self):
        """Test evaluate() works with ThresholdCheck."""
        test_case = TestCase(id="test1", input="user input", expected="expected")
        output = Output(value={"score": 85})

        check = ThresholdCheck(
            value="$.output.value.score",
            min_value=80.0,
            max_value=100.0,
        )

        result = evaluate([test_case], [output], [check])

        assert result.status == "completed"
        assert result.results[0].check_results[0].results["passed"] is True


class TestSchemaCheckMixedUsage:
    """Test mixed usage of SchemaCheck and Check objects."""

    def test_evaluate_with_mixed_check_types(self):
        """Test evaluate() with both SchemaCheck and Check objects."""
        test_case = TestCase(id="test1", input="user input", expected="expected")
        output = Output(value="Hello expected world 85")

        # Mix schema check and traditional check
        schema_check = ContainsCheck(
            text="$.output.value",
            phrases=["expected"],
        )

        traditional_check = Check(
            type="regex",
            arguments={
                "text": "$.output.value",
                "pattern": r"\d+",
            },
        )

        result = evaluate([test_case], [output], [schema_check, traditional_check])

        assert result.status == "completed"
        assert len(result.results[0].check_results) == 2
        assert all(cr.results["passed"] for cr in result.results[0].check_results)

    def test_evaluate_with_multiple_schema_checks(self):
        """Test evaluate() with multiple different schema check types."""
        test_case = TestCase(id="test1", input="user input", expected="hello world")
        output = Output(value={"text": "hello world", "score": 95})

        checks = [
            ContainsCheck(text="$.output.value.text", phrases=["hello"]),
            ExactMatchCheck(actual="$.output.value.text", expected="$.test_case.expected"),
            ThresholdCheck(value="$.output.value.score", min_value=90.0),
            RegexCheck(text="$.output.value.text", pattern=r"world"),
        ]

        result = evaluate([test_case], [output], checks)

        assert result.status == "completed"
        assert len(result.results[0].check_results) == 4
        assert all(cr.results["passed"] for cr in result.results[0].check_results)


class TestSchemaCheckPatterns:
    """Test different check patterns with schema checks."""

    def test_shared_pattern_with_schema_checks(self):
        """Test shared schema checks applied to all test cases."""
        test_cases = [
            TestCase(id="test1", input="input1", expected="expected1"),
            TestCase(id="test2", input="input2", expected="expected2"),
        ]
        outputs = [
            Output(value="result with expected1"),
            Output(value="result with expected2"),
        ]

        # Shared schema check for all test cases
        check = ContainsCheck(
            text="$.output.value",
            phrases=["result"],
        )

        result = evaluate(test_cases, outputs, [check])

        assert result.status == "completed"
        assert len(result.results) == 2
        for test_result in result.results:
            assert len(test_result.check_results) == 1
            assert test_result.check_results[0].results["passed"] is True

    def test_per_test_case_pattern_with_schema_checks(self):
        """Test different schema checks for each test case."""
        test_cases = [
            TestCase(id="test1", input="input1", expected="hello"),
            TestCase(id="test2", input="input2", expected="world"),
        ]
        outputs = [
            Output(value="hello there"),
            Output(value="world here"),
        ]

        # Different schema checks for each test case
        checks = [
            [ContainsCheck(text="$.output.value", phrases=["hello"])],
            [ContainsCheck(text="$.output.value", phrases=["world"])],
        ]

        result = evaluate(test_cases, outputs, checks)

        assert result.status == "completed"
        assert len(result.results) == 2
        for test_result in result.results:
            assert len(test_result.check_results) == 1
            assert test_result.check_results[0].results["passed"] is True

    def test_convenience_pattern_with_schema_checks(self):
        """Test schema checks defined in TestCase.checks."""
        test_cases = [
            TestCase(
                id="test1",
                input="input1",
                expected="hello",
                checks=[ContainsCheck(text="$.output.value", phrases=["hello"])],
            ),
            TestCase(
                id="test2",
                input="input2",
                expected="world",
                checks=[ContainsCheck(text="$.output.value", phrases=["world"])],
            ),
        ]
        outputs = [
            Output(value="hello there"),
            Output(value="world here"),
        ]

        # No checks parameter - extract from TestCase.checks
        result = evaluate(test_cases, outputs, None)

        assert result.status == "completed"
        assert len(result.results) == 2
        for test_result in result.results:
            assert len(test_result.check_results) == 1
            assert test_result.check_results[0].results["passed"] is True


class TestSchemaCheckAdvancedFeatures:
    """Test advanced features of schema checks."""

    def test_regex_check_with_flags(self):
        """Test RegexCheck with flags."""
        test_case = TestCase(id="test1", input="input", expected="expected")
        output = Output(value="HELLO world")

        check = RegexCheck(
            text="$.output.value",
            pattern="hello",
            flags=RegexFlags(case_insensitive=True),
        )

        result = evaluate([test_case], [output], [check])

        assert result.status == "completed"
        assert result.results[0].check_results[0].results["passed"] is True

    def test_threshold_check_with_optional_bounds(self):
        """Test ThresholdCheck with only min_value."""
        test_case = TestCase(id="test1", input="input", expected="expected")
        output = Output(value={"score": 95})

        check = ThresholdCheck(
            value="$.output.value.score",
            min_value=80.0,
            # No max_value
        )

        result = evaluate([test_case], [output], [check])

        assert result.status == "completed"
        assert result.results[0].check_results[0].results["passed"] is True

    def test_contains_check_with_negation(self):
        """Test ContainsCheck with negation."""
        test_case = TestCase(id="test1", input="input", expected="expected")
        output = Output(value="This does not contain the unwanted text")

        check = ContainsCheck(
            text="$.output.value",
            phrases=["forbidden phrase"],
            negate=True,
        )

        result = evaluate([test_case], [output], [check])

        assert result.status == "completed"
        assert result.results[0].check_results[0].results["passed"] is True

    def test_custom_function_check_with_schema(self):
        """Test CustomFunctionCheck with schema check."""
        def validation_function(text: str, min_length: int) -> dict:
            return {"passed": len(text) >= min_length, "length": len(text)}

        test_case = TestCase(id="test1", input="input", expected="expected")
        output = Output(value="Hello world")

        check = CustomFunctionCheck(
            validation_function=validation_function,
            function_args={
                "text": "$.output.value",
                "min_length": 5,
            },
        )

        result = evaluate([test_case], [output], [check])

        assert result.status == "completed"
        assert result.results[0].check_results[0].results["passed"] is True
        assert result.results[0].check_results[0].results["length"] == 11


class TestSchemaCheckValidation:
    """Test validation of schema check fields."""

    def test_contains_check_validation_errors(self):
        """Test ContainsCheck validation errors."""
        with pytest.raises(ValidationError):
            ContainsCheck(text="$.output.value", phrases=[])

        with pytest.raises(ValueError, match="all phrases must be non-empty strings"):
            ContainsCheck(text="$.output.value", phrases=["valid", ""])

    def test_threshold_check_validation_errors(self):
        """Test ThresholdCheck validation errors."""
        with pytest.raises(ValueError, match="At least one of 'min_value' or 'max_value' must be specified"):  # noqa: E501
            ThresholdCheck(value="$.output.value")

    def test_exact_match_check_validation_errors(self):
        """Test ExactMatchCheck validation errors."""
        with pytest.raises(ValidationError):
            ExactMatchCheck(actual="", expected="$.expected")

        with pytest.raises(ValidationError):
            ExactMatchCheck(actual="$.actual", expected="")


class TestSchemaCheckEquivalence:
    """Test that schema checks produce identical results to equivalent Check objects."""

    def test_contains_check_equivalence(self):
        """Test ContainsCheck produces same results as equivalent Check."""
        test_case = TestCase(id="test1", input="input", expected="expected")
        output = Output(value="This contains expected text")

        # Schema check
        schema_check = ContainsCheck(
            text="$.output.value",
            phrases=["expected"],
            case_sensitive=False,
            negate=False,
        )

        # Equivalent traditional check
        traditional_check = Check(
            type="contains",
            arguments={
                "text": "$.output.value",
                "phrases": ["expected"],
                "case_sensitive": False,
                "negate": False,
            },
        )

        # Run both
        schema_result = evaluate([test_case], [output], [schema_check])
        traditional_result = evaluate([test_case], [output], [traditional_check])

        # Compare results
        assert schema_result.status == traditional_result.status
        assert len(schema_result.results) == len(traditional_result.results)

        schema_check_result = schema_result.results[0].check_results[0]
        traditional_check_result = traditional_result.results[0].check_results[0]

        assert schema_check_result.results == traditional_check_result.results
        assert schema_check_result.check_type == traditional_check_result.check_type

    def test_threshold_check_equivalence(self):
        """Test ThresholdCheck produces same results as equivalent Check."""
        test_case = TestCase(id="test1", input="input", expected="expected")
        output = Output(value={"score": 85})

        # Schema check
        schema_check = ThresholdCheck(
            value="$.output.value.score",
            min_value=80.0,
            max_value=90.0,
            min_inclusive=True,
            max_inclusive=True,
            negate=False,
        )

        # Equivalent traditional check
        traditional_check = Check(
            type="threshold",
            arguments={
                "value": "$.output.value.score",
                "min_value": 80.0,
                "max_value": 90.0,
                "min_inclusive": True,
                "max_inclusive": True,
                "negate": False,
            },
        )

        # Run both
        schema_result = evaluate([test_case], [output], [schema_check])
        traditional_result = evaluate([test_case], [output], [traditional_check])

        # Compare results
        schema_check_result = schema_result.results[0].check_results[0]
        traditional_check_result = traditional_result.results[0].check_results[0]

        assert schema_check_result.results == traditional_check_result.results
        assert schema_check_result.check_type == traditional_check_result.check_type


class TestSchemaCheckErrorHandling:
    """Test error handling with schema checks."""

    def test_schema_check_jsonpath_error(self):
        """Test schema check with invalid JSONPath."""
        test_case = TestCase(id="test1", input="input", expected="expected")
        output = Output(value="text")

        check = ContainsCheck(
            text="$.output.nonexistent.path",
            phrases=["text"],
        )

        result = evaluate([test_case], [output], [check])

        assert result.status == "error"
        assert result.results[0].check_results[0].status == "error"
        assert result.results[0].check_results[0].error is not None

    def test_schema_check_conversion_preservation(self):
        """Test that schema check conversion preserves all field values."""
        schema_check = RegexCheck(
            text="$.output.value",
            pattern="test",
            negate=True,
            flags=RegexFlags(case_insensitive=True, multiline=True),
            version="1.0.0",
        )

        converted_check = schema_check.to_check()

        assert converted_check.type == "regex"
        assert converted_check.arguments["text"] == "$.output.value"
        assert converted_check.arguments["pattern"] == "test"
        assert converted_check.arguments["negate"] is True
        assert converted_check.arguments["flags"]["case_insensitive"] is True
        assert converted_check.arguments["flags"]["multiline"] is True
        assert converted_check.arguments["flags"]["dot_all"] is False
        assert converted_check.version == "1.0.0"
