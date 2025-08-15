"""Comprehensive tests for AttributeExistsCheck implementation.

This module consolidates all tests for the AttributeExistsCheck including:
- Implementation execution tests (from test_standard_checks.py) 
- Engine integration tests
- Edge cases and error handling

Tests are organized by functionality rather than implementation details.
"""

import pytest
from typing import Any

from flex_evals.checks.attribute_exists import AttributeExistsCheck
from flex_evals.checks.base import JSONPath
from flex_evals import CheckType, Status, evaluate, Check, Output, TestCase
from flex_evals.exceptions import ValidationError
from pydantic import ValidationError as PydanticValidationError


class TestAttributeExistsValidation:
    """Test Pydantic validation and field handling for AttributeExistsCheck."""

    def test_attribute_exists_check_creation(self):
        """Test basic AttributeExistsCheck creation."""
        check = AttributeExistsCheck(
            path="$.output.value.error",
        )

        assert isinstance(check.path, JSONPath)


        assert check.path.expression == "$.output.value.error"
        assert check.negate is False

    def test_attribute_exists_check_with_negate(self):
        """Test AttributeExistsCheck with negate option."""
        check = AttributeExistsCheck(
            path="$.output.value.error",
            negate=True,
        )

        assert check.negate is True

    def test_attribute_exists_check_type_property(self):
        """Test AttributeExistsCheck check_type property returns correct type."""
        check = AttributeExistsCheck(path="$.test")
        assert check.check_type == CheckType.ATTRIBUTE_EXISTS


class TestAttributeExistsEngineIntegration:
    """Test AttributeExistsCheck integration with the evaluation engine."""

    def test_attribute_exists_via_evaluate_exists(self):
        """Test attribute exists check when attribute is present."""
        test_cases = [
            TestCase(
                id="test_001",
                input="Test with error",
                checks=[
                    AttributeExistsCheck(path="$.output.value.error"),
                ],
            ),
        ]

        # System output with error field
        outputs = [
            Output(value={"result": "failed", "error": "Something went wrong"}),
        ]

        # Run evaluation
        results = evaluate(test_cases, outputs)
        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_attribute_exists_via_evaluate_does_not_exist(self):
        """Test attribute exists check when attribute is not present."""
        test_cases = [
            TestCase(
                id="test_001",
                input="Test without error",
                checks=[
                    Check(
                        type=CheckType.ATTRIBUTE_EXISTS,
                        arguments={
                            "path": "$.output.value.error",
                        },
                    ),
                ],
            ),
        ]

        # System output without error field
        outputs = [
            Output(value={"result": "success", "message": "All good"}),
        ]

        # Run evaluation
        results = evaluate(test_cases, outputs)
        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": False}

    def test_attribute_exists_via_evaluate_negate_exists(self):
        """Test attribute exists check with negate when attribute is present."""
        test_cases = [
            TestCase(
                id="test_001",
                input="Test with error but negate",
                checks=[
                    Check(
                        type=CheckType.ATTRIBUTE_EXISTS,
                        arguments={
                            "path": "$.output.value.error",
                            "negate": True,
                        },
                    ),
                ],
            ),
        ]

        # System output with error field (should fail because negate=True)
        outputs = [
            Output(value={"result": "failed", "error": "Something went wrong"}),
        ]

        # Run evaluation
        results = evaluate(test_cases, outputs)
        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": False}

    def test_attribute_exists_via_evaluate_negate_does_not_exist(self):
        """Test attribute exists check with negate when attribute is not present."""
        test_cases = [
            TestCase(
                id="test_001",
                input="Test without error and negate",
                checks=[
                    Check(
                        type=CheckType.ATTRIBUTE_EXISTS,
                        arguments={
                            "path": "$.output.value.error",
                            "negate": True,
                        },
                    ),
                ],
            ),
        ]

        # System output without error field (should pass because negate=True)
        outputs = [
            Output(value={"result": "success", "message": "All good"}),
        ]

        # Run evaluation
        results = evaluate(test_cases, outputs)
        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_attribute_exists_nested_path_exists(self):
        """Test deeply nested path that exists."""
        test_cases = [
            TestCase(
                id="test_001",
                input="Test nested path",
                checks=[
                    Check(
                        type=CheckType.ATTRIBUTE_EXISTS,
                        arguments={
                            "path": "$.output.value.metadata.processing.duration",
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "result": "success",
                "metadata": {
                    "processing": {
                        "duration": 1.25,
                        "steps": ["parse", "validate", "execute"],
                    },
                    "version": "1.0.0",
                },
            }),
        ]

        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": True}

    def test_attribute_exists_nested_path_missing(self):
        """Test deeply nested path where intermediate path is missing."""
        test_cases = [
            TestCase(
                id="test_001",
                input="Test nested path missing",
                checks=[
                    Check(
                        type=CheckType.ATTRIBUTE_EXISTS,
                        arguments={
                            "path": "$.output.value.metadata.processing.duration",
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "result": "success",
                "metadata": {
                    # missing "processing" object
                    "version": "1.0.0",
                },
            }),
        ]

        results = evaluate(test_cases, outputs)
        assert results.results[0].check_results[0].results == {"passed": False}

    def test_attribute_exists_various_data_types(self):
        """Test attribute existence with various data types."""
        test_cases = [
            TestCase(
                id="test_001",
                input="Test various types",
                checks=[
                    Check(type=CheckType.ATTRIBUTE_EXISTS, arguments={"path": "$.output.value.number"}),
                    Check(type=CheckType.ATTRIBUTE_EXISTS, arguments={"path": "$.output.value.boolean"}),
                    Check(type=CheckType.ATTRIBUTE_EXISTS, arguments={"path": "$.output.value.null_value"}),
                    Check(type=CheckType.ATTRIBUTE_EXISTS, arguments={"path": "$.output.value.array"}),
                    Check(type=CheckType.ATTRIBUTE_EXISTS, arguments={"path": "$.output.value.missing", "negate": True}),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "number": 42,
                "boolean": False,
                "null_value": None,
                "array": [],
                # "missing" key intentionally absent
            }),
        ]

        results = evaluate(test_cases, outputs)

        # All checks should pass
        for i in range(5):
            assert results.results[0].check_results[i].status == Status.COMPLETED
            assert results.results[0].check_results[i].results == {"passed": True}

    @pytest.mark.parametrize(("output_value", "path", "negate", "expected_passed"), [
        ({"error": "failed"}, "$.output.value.error", False, True),  # Attribute exists, no negate
        ({"success": True}, "$.output.value.error", False, False),  # Attribute missing, no negate
        ({"error": "failed"}, "$.output.value.error", True, False),  # Attribute exists, with negate
        ({"success": True}, "$.output.value.error", True, True),  # Attribute missing, with negate
        ({"nested": {"field": "value"}}, "$.output.value.nested.field", False, True),  # Nested exists
        ({"nested": {}}, "$.output.value.nested.field", False, False),  # Nested missing
    ])
    def test_attribute_exists_parametrized(
        self, output_value: dict, path: str, negate: bool, expected_passed: bool,
    ):
        """Test attribute exists check with various combinations."""
        test_cases = [
            TestCase(
                id="test_001",
                input="Parametrized test",
                checks=[
                    Check(
                        type=CheckType.ATTRIBUTE_EXISTS,
                        arguments={
                            "path": path,
                            "negate": negate,
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value=output_value)]

        results = evaluate(test_cases, outputs)
        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results == {"passed": expected_passed}


class TestAttributeExistsErrorHandling:
    """Test error handling and edge cases for AttributeExistsCheck."""

    def test_attribute_exists_invalid_path_error(self):
        """Test with invalid JSONPath expression raises validation error."""
        test_cases = [
            TestCase(
                id="test_001",
                input="Test invalid path",
                checks=[
                    Check(
                        type=CheckType.ATTRIBUTE_EXISTS,
                        arguments={
                            "path": "not_a_jsonpath",  # Missing $.
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={"result": "success"}),
        ]

        # Should raise validation error for invalid JSONPath
        with pytest.raises(ValidationError, match=r"Check arguments validation failed"):
            evaluate(test_cases, outputs)

    def test_attribute_exists_missing_path_argument(self):
        """Test missing path argument raises validation error."""
        test_cases = [
            TestCase(
                id="test_001",
                input="Test missing path",
                checks=[
                    Check(
                        type=CheckType.ATTRIBUTE_EXISTS,
                        arguments={
                            # Missing "path" argument
                            "negate": False,
                        },
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={"result": "success"}),
        ]

        # Should raise validation error for missing required field
        with pytest.raises(ValidationError, match=r"Check arguments validation failed"):
            evaluate(test_cases, outputs)

    def test_attribute_exists_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(PydanticValidationError):
            AttributeExistsCheck()  # type: ignore