"""
Comprehensive tests for SupersetCheck implementation.

Tests are organized by functionality:
- Pydantic validation tests
- Execution logic tests
- Error handling tests
- Engine integration tests
"""

import pytest

from flex_evals import (
    SupersetCheck,
    JSONPath,
    EvaluationContext,
    CheckType,
    evaluate_sync,
    Check,
    ValidationError,
    TestCase,
    Output,
)
from pydantic import ValidationError as PydanticValidationError


class TestSupersetValidation:
    """Test Pydantic validation and field handling for SupersetCheck."""

    def test_creation_with_literal_lists(self) -> None:
        check = SupersetCheck(
            collection=["a", "b"],
            allowed=["a", "b", "c"],
        )
        assert check.collection == ["a", "b"]
        assert check.allowed == ["a", "b", "c"]
        assert check.case_sensitive is True
        assert check.negate is False

    def test_jsonpath_auto_conversion(self) -> None:
        check = SupersetCheck(
            collection="$.output.value.tags",
            allowed="$.test_case.expected.allowed",
            case_sensitive="$.test_case.expected.cs",
            negate="$.test_case.expected.neg",
        )
        assert isinstance(check.collection, JSONPath)
        assert check.collection.expression == "$.output.value.tags"
        assert isinstance(check.allowed, JSONPath)
        assert isinstance(check.case_sensitive, JSONPath)
        assert isinstance(check.negate, JSONPath)

    def test_invalid_jsonpath_raises(self) -> None:
        with pytest.raises(PydanticValidationError, match="Invalid JSONPath expression"):
            SupersetCheck(collection="$.invalid[", allowed=["a"])

    def test_check_type_property(self) -> None:
        check = SupersetCheck(collection=["a"], allowed=["a"])
        assert check.check_type == CheckType.SUPERSET

    def test_invalid_boolean_field_raises(self) -> None:
        with pytest.raises(PydanticValidationError):
            SupersetCheck(collection=["a"], allowed=["a"], case_sensitive="not_a_boolean")


class TestSupersetExecution:
    """Test SupersetCheck execution logic."""

    def test_all_collection_in_allowed(self) -> None:
        check = SupersetCheck(collection=["a", "b"], allowed=["a", "b", "c"])
        result = check()
        assert result["passed"] is True
        assert result["found"] == ["a", "b"]
        assert result["unexpected"] == []

    def test_some_collection_not_in_allowed(self) -> None:
        check = SupersetCheck(collection=["a", "b", "x"], allowed=["a", "b"])
        result = check()
        assert result["passed"] is False
        assert result["found"] == ["a", "b"]
        assert result["unexpected"] == ["x"]

    def test_all_collection_unexpected(self) -> None:
        check = SupersetCheck(collection=["x", "y"], allowed=["a", "b"])
        result = check()
        assert result["passed"] is False
        assert result["unexpected"] == ["x", "y"]
        assert result["found"] == []

    def test_empty_collection(self) -> None:
        """Empty collection has nothing unexpected — passes."""
        check = SupersetCheck(collection=[], allowed=["a", "b"])
        result = check()
        assert result["passed"] is True
        assert result["unexpected"] == []
        assert result["found"] == []

    def test_empty_allowed_nonempty_collection(self) -> None:
        """Non-empty collection with empty allowed — everything is unexpected."""
        check = SupersetCheck(collection=["a"], allowed=[])
        result = check()
        assert result["passed"] is False
        assert result["unexpected"] == ["a"]

    def test_both_empty(self) -> None:
        check = SupersetCheck(collection=[], allowed=[])
        result = check()
        assert result["passed"] is True

    def test_negate_inverts_pass(self) -> None:
        check = SupersetCheck(collection=["a"], allowed=["a", "b"], negate=True)
        result = check()
        assert result["passed"] is False

    def test_negate_inverts_fail(self) -> None:
        check = SupersetCheck(collection=["a", "x"], allowed=["a"], negate=True)
        result = check()
        assert result["passed"] is True

    def test_case_insensitive_strings(self) -> None:
        check = SupersetCheck(
            collection=["Python", "Flask"],
            allowed=["python", "flask", "api"],
            case_sensitive=False,
        )
        result = check()
        assert result["passed"] is True
        assert result["unexpected"] == []

    def test_case_sensitive_strings(self) -> None:
        check = SupersetCheck(
            collection=["Python"],
            allowed=["python"],
            case_sensitive=True,
        )
        result = check()
        assert result["passed"] is False
        assert result["unexpected"] == ["Python"]

    def test_case_insensitive_does_not_affect_non_strings(self) -> None:
        check = SupersetCheck(collection=[1], allowed=["1"], case_sensitive=False)
        result = check()
        assert result["passed"] is False
        assert result["unexpected"] == [1]

    def test_duplicates_in_collection(self) -> None:
        """Duplicates in collection are deduplicated (set semantics)."""
        check = SupersetCheck(collection=["a", "a", "b"], allowed=["a", "b"])
        result = check()
        assert result["passed"] is True
        assert result["found"] == ["a", "b"]

    def test_non_string_elements(self) -> None:
        check = SupersetCheck(collection=[1, 2], allowed=[1, 2, 3])
        result = check()
        assert result["passed"] is True

    def test_dict_elements(self) -> None:
        check = SupersetCheck(
            collection=[{"key": "val"}],
            allowed=[{"key": "val"}, {"other": "data"}],
        )
        result = check()
        assert result["passed"] is True

    def test_dict_elements_unexpected(self) -> None:
        check = SupersetCheck(
            collection=[{"key": "val"}, {"unknown": True}],
            allowed=[{"key": "val"}],
        )
        result = check()
        assert result["passed"] is False
        assert result["unexpected"] == [{"unknown": True}]

    def test_set_inputs(self) -> None:
        check = SupersetCheck(collection={"a", "b"}, allowed={"a", "b", "c"})
        result = check()
        assert result["passed"] is True

    def test_none_elements(self) -> None:
        check = SupersetCheck(collection=[None, "a"], allowed=[None, "a", "b"])
        result = check()
        assert result["passed"] is True

    def test_negate_with_empty_collection(self) -> None:
        """Empty collection passes; negate inverts to False."""
        check = SupersetCheck(collection=[], allowed=["a"], negate=True)
        result = check()
        assert result["passed"] is False

    def test_case_insensitive_with_negate(self) -> None:
        check = SupersetCheck(
            collection=["Python"],
            allowed=["python", "flask"],
            case_sensitive=False,
            negate=True,
        )
        result = check()
        # All collection items found (case-insensitive), passed=True, negate → False
        assert result["passed"] is False


class TestSupersetErrorHandling:
    """Test error conditions for SupersetCheck."""

    def test_collection_not_list_or_set(self) -> None:
        check = SupersetCheck(collection=["placeholder"], allowed=["a"])
        check.collection = "not a list"  # type: ignore[assignment]
        with pytest.raises(ValidationError, match="must be a list or set"):
            check()

    def test_allowed_not_list_or_set(self) -> None:
        check = SupersetCheck(collection=["a"], allowed=["placeholder"])
        check.allowed = "not a list"  # type: ignore[assignment]
        with pytest.raises(ValidationError, match="must be a list or set"):
            check()

    def test_collection_is_int(self) -> None:
        check = SupersetCheck(collection=["placeholder"], allowed=["a"])
        check.collection = 42  # type: ignore[assignment]
        with pytest.raises(ValidationError, match="must be a list or set"):
            check()

    def test_unresolved_jsonpath_collection(self) -> None:
        check = SupersetCheck(collection="$.output.value", allowed=["a"])
        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'collection'"):
            check()

    def test_unresolved_jsonpath_allowed(self) -> None:
        check = SupersetCheck(collection=["a"], allowed="$.test_case.expected")
        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'allowed'"):
            check()

    def test_unresolved_jsonpath_case_sensitive(self) -> None:
        check = SupersetCheck(
            collection=["a"], allowed=["a"], case_sensitive="$.test_case.expected.cs",
        )
        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'case_sensitive'"):
            check()

    def test_unresolved_jsonpath_negate(self) -> None:
        check = SupersetCheck(
            collection=["a"], allowed=["a"], negate="$.test_case.expected.neg",
        )
        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'negate'"):
            check()


class TestSupersetEngineIntegration:
    """Test SupersetCheck integration with the evaluation engine."""

    def test_execute_with_evaluation_context(self) -> None:
        check = SupersetCheck(
            collection="$.output.value.tags",
            allowed="$.test_case.expected.allowed",
        )
        test_case = TestCase(
            id="test_001",
            input="test",
            expected={"allowed": ["python", "flask", "api", "web"]},
        )
        output = Output(value={"tags": ["python", "flask"]})
        context = EvaluationContext(test_case, output)

        result = check.execute(context)
        assert result.status == "completed"
        assert result.results["passed"] is True
        assert result.results["unexpected"] == []
        assert result.resolved_arguments["collection"]["value"] == ["python", "flask"]

    def test_execute_with_unexpected_items(self) -> None:
        check = SupersetCheck(
            collection="$.output.value.tags",
            allowed="$.test_case.expected.allowed",
        )
        test_case = TestCase(
            id="test_001",
            input="test",
            expected={"allowed": ["python", "flask"]},
        )
        output = Output(value={"tags": ["python", "flask", "hallucinated"]})
        context = EvaluationContext(test_case, output)

        result = check.execute(context)
        assert result.status == "completed"
        assert result.results["passed"] is False
        assert result.results["unexpected"] == ["hallucinated"]

    def test_wrong_runtime_type_via_execute(self) -> None:
        check = SupersetCheck(collection="$.output.value", allowed=["a"])
        test_case = TestCase(id="test_001", input="test")
        output = Output(value="not a list")
        context = EvaluationContext(test_case, output)

        result = check.execute(context)
        assert result.status == "error"
        assert result.results == {"passed": False, "unexpected": [], "found": []}
        assert result.error is not None
        assert "must be a list or set" in result.error.message

    @pytest.mark.parametrize(("output_tags", "allowed_tags", "expected_passed"), [
        (["a", "b"], ["a", "b", "c"], True),
        (["a", "b", "x"], ["a", "b"], False),
        ([], ["a", "b"], True),
        (["a"], [], False),
    ])
    def test_via_evaluate_sync(
        self,
        output_tags: list[str],
        allowed_tags: list[str],
        expected_passed: bool,
    ) -> None:
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected=allowed_tags,
                checks=[
                    Check(
                        type=CheckType.SUPERSET,
                        arguments={
                            "collection": "$.output.value",
                            "allowed": "$.test_case.expected",
                        },
                    ),
                ],
            ),
        ]
        outputs = [Output(value=output_tags)]
        results = evaluate_sync(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].check_results[0].results["passed"] == expected_passed

    def test_check_instance_via_evaluate_sync(self) -> None:
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected=["python", "flask", "api"],
                checks=[
                    SupersetCheck(
                        collection="$.output.value",
                        allowed="$.test_case.expected",
                    ),
                ],
            ),
        ]
        outputs = [Output(value=["python", "flask"])]
        results = evaluate_sync(test_cases, outputs)

        assert results.summary.completed_test_cases == 1
        assert results.results[0].check_results[0].results["passed"] is True
