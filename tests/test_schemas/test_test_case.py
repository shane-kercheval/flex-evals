"""Tests for TestCase schema implementation."""

import dataclasses
import pytest
from flex_evals import TestCase, Check


class TestTestCase:
    """Test TestCase schema implementation."""

    def test_test_case_required_fields(self):
        """Test TestCase with only required fields."""
        test_case = TestCase(id="test_001", input="What is the capital of France?")

        assert test_case.id == "test_001"
        assert test_case.input == "What is the capital of France?"
        assert test_case.expected is None
        assert test_case.metadata is None
        assert test_case.checks is None

    def test_test_case_string_input(self):
        """Test TestCase with simple string input."""
        test_case = TestCase(id="test_001", input="Simple string input")

        assert isinstance(test_case.input, str)
        assert test_case.input == "Simple string input"

    def test_test_case_object_input(self):
        """Test TestCase with Dict input containing nested data."""
        input_data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            "temperature": 0.7,
        }
        test_case = TestCase(id="test_001", input=input_data)

        assert isinstance(test_case.input, dict)
        assert test_case.input == input_data
        assert len(test_case.input["messages"]) == 2

    def test_test_case_expected_string(self):
        """Test TestCase with expected as string."""
        test_case = TestCase(
            id="test_001",
            input="What is the capital of France?",
            expected="Paris",
        )

        assert test_case.expected == "Paris"
        assert isinstance(test_case.expected, str)

    def test_test_case_expected_object(self):
        """Test TestCase with expected as Dict."""
        expected_data = {
            "answer": "Paris",
            "confidence": 0.95,
            "reasoning": "Paris is the capital and largest city of France.",
        }
        test_case = TestCase(
            id="test_001",
            input="What is the capital of France?",
            expected=expected_data,
        )

        assert test_case.expected == expected_data
        assert isinstance(test_case.expected, dict)

    def test_test_case_expected_null(self):
        """Test TestCase with expected as None/null."""
        test_case = TestCase(
            id="test_001",
            input="What is the capital of France?",
            expected=None,
        )

        assert test_case.expected is None

    def test_test_case_metadata_optional(self):
        """Test TestCase with and without metadata."""
        # Without metadata
        test_case1 = TestCase(id="test_001", input="test input")
        assert test_case1.metadata is None

        # With metadata
        metadata = {
            "version": "1.0.1",
            "tags": ["geography"],
            "created_at": "2025-06-25T10:00:00Z",
        }
        test_case2 = TestCase(id="test_002", input="test input", metadata=metadata)
        assert test_case2.metadata == metadata

    def test_test_case_checks_extension(self):
        """Test convenience checks field."""
        checks = [
            Check(type='exact_match', arguments={"actual": "$.output.value", "expected": "Paris"}),
            Check(type='contains', arguments={"text": "$.output.value", "phrases": ["France"]}),
        ]

        test_case = TestCase(
            id="test_001",
            input="What is the capital of France?",
            checks=checks,
        )

        assert test_case.checks == checks
        assert len(test_case.checks) == 2
        assert test_case.checks[0].type == 'exact_match'

    def test_test_case_validation_errors(self):
        """Test missing required fields raise ValidationError."""
        # Missing id
        with pytest.raises(ValueError, match="TestCase.id must be a non-empty string"):
            TestCase(id="", input="test input")

        with pytest.raises(ValueError, match="TestCase.id must be a non-empty string"):
            TestCase(id=None, input="test input")

        # Missing input
        with pytest.raises(ValueError, match="TestCase.input is required and cannot be None"):
            TestCase(id="test_001", input=None)

    def test_test_case_empty_id_error(self):
        """Test empty string id raises error."""
        with pytest.raises(ValueError, match="TestCase.id must be a non-empty string"):
            TestCase(id="", input="test input")

    def test_test_case_invalid_input_type(self):
        """Test invalid input types raise error."""
        with pytest.raises(ValueError, match="TestCase.input must be a string or dictionary"):
            TestCase(id="test_001", input=123)

        with pytest.raises(ValueError, match="TestCase.input must be a string or dictionary"):
            TestCase(id="test_001", input=["list", "not", "allowed"])

    def test_test_case_serialization(self):
        """Test TestCase can be converted to dict for JSON serialization."""
        test_case = TestCase(
            id="test_001",
            input={"question": "What is the capital of France?"},
            expected="Paris",
            metadata={"version": "1.0"},
        )

        # Convert to dict for serialization
        data = dataclasses.asdict(test_case)

        assert data["id"] == "test_001"
        assert data["input"]["question"] == "What is the capital of France?"
        assert data["expected"] == "Paris"
        assert data["metadata"]["version"] == "1.0"
        assert data["checks"] is None
