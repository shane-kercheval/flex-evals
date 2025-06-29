"""Tests for JSONPath resolver implementation."""

import pytest
from flex_evals.jsonpath_resolver import JSONPathResolver
from flex_evals.schemas import TestCase, Output
from flex_evals.exceptions import JSONPathError


class TestJSONPathResolver:
    """Test JSONPath resolver functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.resolver = JSONPathResolver()

        # Create test evaluation context
        self.test_case = TestCase(
            id="test_001",
            input={"question": "What is the capital of France?"},
            expected="Paris",
            metadata={"version": "1.0", "tags": ["geography"]},
        )

        self.output = Output(
            value={
                "answer": "Paris",
                "confidence": 0.95,
                "reasoning": "Based on geographical knowledge...",
                "details": {
                    "country": "France",
                    "continent": "Europe",
                },
            },
            metadata={"execution_time_ms": 245, "model_version": "gpt-4"},
        )

        self.context = self.resolver.create_evaluation_context(self.test_case, self.output)

    def test_jsonpath_detection_positive(self):
        """Test JSONPath expressions are detected correctly."""
        jsonpath_expressions = [
            "$.output.value",
            "$.test_case.expected",
            "$.output.value.confidence",
            "$.test_case.input.question",
            "$.output.metadata.execution_time_ms",
        ]

        for expr in jsonpath_expressions:
            assert self.resolver.is_jsonpath(expr), f"Should detect '{expr}' as JSONPath"

    def test_jsonpath_detection_negative(self):
        """Test non-JSONPath strings are not detected as JSONPath."""
        non_jsonpath_values = [
            "literal.string",
            "regular text",
            "Paris",
            "some.property",
            "json.key",
            "$not_a_path",
            # "$.but.escaped", # This IS a valid JSONPath, so removing from test
            "",
        ]

        for value in non_jsonpath_values:
            assert not self.resolver.is_jsonpath(value), f"Should not detect '{value}' as JSONPath"

    def test_jsonpath_escape_syntax(self):
        """Test escaped JSONPath strings are treated as literals."""
        escaped_values = [
            "\\$.literal",
            "\\$.this.is.literal",
            "\\$.output.value",
        ]

        for value in escaped_values:
            assert not self.resolver.is_jsonpath(value), f"Should not detect escaped '{value}' as JSONPath"  # noqa: E501

            # Test resolution returns literal value without escape
            result = self.resolver.resolve_argument(value, self.context)
            expected_literal = value[1:]  # Remove backslash
            assert result == {"value": expected_literal}

    def test_evaluation_context_structure(self):
        """Test evaluation context matches exact protocol schema."""
        context = self.resolver.create_evaluation_context(self.test_case, self.output)

        # Verify top-level structure
        assert set(context.keys()) == {"test_case", "output"}

        # Verify test_case structure
        test_case_data = context["test_case"]
        assert test_case_data["id"] == "test_001"
        assert test_case_data["input"] == {"question": "What is the capital of France?"}
        assert test_case_data["expected"] == "Paris"
        assert test_case_data["metadata"] == {"version": "1.0", "tags": ["geography"]}

        # Verify output structure
        output_data = context["output"]
        assert output_data["value"]["answer"] == "Paris"
        assert output_data["value"]["confidence"] == 0.95
        assert output_data["metadata"]["execution_time_ms"] == 245

    def test_jsonpath_test_case_access(self):
        """Test accessing test case properties via JSONPath."""
        test_cases = [
            ("$.test_case.id", "test_001"),
            ("$.test_case.input.question", "What is the capital of France?"),
            ("$.test_case.expected", "Paris"),
            ("$.test_case.metadata.version", "1.0"),
            ("$.test_case.metadata.tags[0]", "geography"),
        ]

        for jsonpath, expected_value in test_cases:
            result = self.resolver.resolve_argument(jsonpath, self.context)
            assert result["jsonpath"] == jsonpath
            assert result["value"] == expected_value

    def test_jsonpath_output_access(self):
        """Test accessing output properties via JSONPath."""
        test_cases = [
            ("$.output.value.answer", "Paris"),
            ("$.output.value.confidence", 0.95),
            ("$.output.value.reasoning", "Based on geographical knowledge..."),
            ("$.output.metadata.execution_time_ms", 245),
            ("$.output.metadata.model_version", "gpt-4"),
        ]

        for jsonpath, expected_value in test_cases:
            result = self.resolver.resolve_argument(jsonpath, self.context)
            assert result["jsonpath"] == jsonpath
            assert result["value"] == expected_value

    def test_jsonpath_nested_access(self):
        """Test accessing nested properties via JSONPath."""
        test_cases = [
            ("$.output.value.details.country", "France"),
            ("$.output.value.details.continent", "Europe"),
        ]

        for jsonpath, expected_value in test_cases:
            result = self.resolver.resolve_argument(jsonpath, self.context)
            assert result["jsonpath"] == jsonpath
            assert result["value"] == expected_value

    def test_jsonpath_resolution_success(self):
        """Test successful JSONPath resolution returns correct format."""
        result = self.resolver.resolve_argument("$.output.value.answer", self.context)

        # Verify format matches protocol specification
        assert isinstance(result, dict)
        assert set(result.keys()) == {"jsonpath", "value"}
        assert result["jsonpath"] == "$.output.value.answer"
        assert result["value"] == "Paris"

    def test_jsonpath_resolution_error(self):
        """Test invalid JSONPath expressions raise JSONPathError."""
        invalid_expressions = [
            "$.invalid[syntax",  # Invalid syntax - unclosed bracket
            "$.field.",          # Invalid syntax - trailing dot
        ]

        for expr in invalid_expressions:
            with pytest.raises(JSONPathError, match="Invalid JSONPath expression"):
                self.resolver.resolve_argument(expr, self.context)

        # Test expression that's not detected as JSONPath (literal value)
        result = self.resolver.resolve_argument("$invalid", self.context)
        assert result == {"value": "$invalid"}

    def test_jsonpath_nonexistent_path(self):
        """Test JSONPath to nonexistent data raises JSONPathError."""
        nonexistent_paths = [
            "$.output.value.nonexistent",
            "$.test_case.missing_field",
            "$.output.value.details.population",
            "$.test_case.metadata.tags[10]",
        ]

        for path in nonexistent_paths:
            with pytest.raises(JSONPathError, match="did not match any data"):
                self.resolver.resolve_argument(path, self.context)

    def test_resolved_arguments_format(self):
        """Test returned format matches protocol specification."""
        # Test JSONPath argument
        jsonpath_result = self.resolver.resolve_argument("$.output.value.answer", self.context)
        assert jsonpath_result == {
            "jsonpath": "$.output.value.answer",
            "value": "Paris",
        }

        # Test literal argument
        literal_result = self.resolver.resolve_argument("literal_value", self.context)
        assert literal_result == {"value": "literal_value"}

        # Test numeric literal
        numeric_result = self.resolver.resolve_argument(42, self.context)
        assert numeric_result == {"value": 42}

        # Test boolean literal
        bool_result = self.resolver.resolve_argument(True, self.context)
        assert bool_result == {"value": True}

    def test_mixed_literal_jsonpath_args(self):
        """Test resolving arguments with both literal and JSONPath values."""
        arguments = {
            "actual": "$.output.value.answer",
            "expected": "$.test_case.expected",
            "case_sensitive": True,
            "negate": False,
            "literal_string": "some text",
        }

        resolved = self.resolver.resolve_arguments(arguments, self.context)

        # Verify JSONPath arguments
        assert resolved["actual"]["jsonpath"] == "$.output.value.answer"
        assert resolved["actual"]["value"] == "Paris"

        assert resolved["expected"]["jsonpath"] == "$.test_case.expected"
        assert resolved["expected"]["value"] == "Paris"

        # Verify literal arguments
        assert resolved["case_sensitive"] == {"value": True}
        assert resolved["negate"] == {"value": False}
        assert resolved["literal_string"] == {"value": "some text"}

    def test_jsonpath_caching(self):
        """Test JSONPath compilation is cached for performance."""
        # First resolution should compile and cache
        result1 = self.resolver.resolve_argument("$.output.value.answer", self.context)
        assert result1["value"] == "Paris"

        # Verify expression is cached
        assert "$.output.value.answer" in self.resolver._cache

        # Second resolution should use cache
        result2 = self.resolver.resolve_argument("$.output.value.answer", self.context)
        assert result2["value"] == "Paris"

        # Results should be identical
        assert result1 == result2

    def test_non_string_values(self):
        """Test non-string values are not treated as JSONPath."""
        non_string_values = [
            123,
            45.67,
            True,
            False,
            None,
            {"key": "value"},
            ["list", "item"],
        ]

        for value in non_string_values:
            assert not self.resolver.is_jsonpath(value)
            result = self.resolver.resolve_argument(value, self.context)
            assert result == {"value": value}

    def test_complex_evaluation_context(self):
        """Test complex nested evaluation context structure."""
        # Create more complex test case and output
        complex_test_case = TestCase(
            id="complex_001",
            input={
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Tell me about Paris"},
                ],
                "config": {"temperature": 0.7, "max_tokens": 100},
            },
            expected={
                "response": "Paris is the capital of France",
                "metadata": {"confidence": 0.9},
            },
            metadata={
                "dataset": "geography",
                "difficulty": "easy",
                "tags": ["cities", "capitals"],
            },
        )

        complex_output = Output(
            value={
                "message": "Paris is the capital of France",
                "confidence": 0.95,
                "reasoning": "Well-known geographical fact",
                "sources": ["wikipedia", "britannica"],
                "metadata": {
                    "token_count": 45,
                    "finish_reason": "stop",
                },
            },
            metadata={
                "provider": "openai",
                "model": "gpt-4",
                "cost": 0.002,
                "latency_ms": 850,
            },
        )

        complex_context = self.resolver.create_evaluation_context(complex_test_case, complex_output)  # noqa: E501

        # Test complex nested access patterns
        test_cases = [
            ("$.test_case.input.messages[1].content", "Tell me about Paris"),
            ("$.test_case.input.config.temperature", 0.7),
            ("$.test_case.expected.metadata.confidence", 0.9),
            ("$.test_case.metadata.tags[1]", "capitals"),
            ("$.output.value.sources[0]", "wikipedia"),
            ("$.output.value.metadata.token_count", 45),
            ("$.output.metadata.provider", "openai"),
        ]

        for jsonpath, expected_value in test_cases:
            result = self.resolver.resolve_argument(jsonpath, complex_context)
            assert result["value"] == expected_value, f"Failed for {jsonpath}"
