"""
Comprehensive tests for DisjointCheck implementation.

Tests are organized by functionality:
- Pydantic validation tests
- Execution logic tests
- Error handling tests
- Engine integration tests
"""

import pytest

from flex_evals import (
    DisjointCheck,
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


class TestDisjointValidation:
    """Test Pydantic validation and field handling for DisjointCheck."""

    def test_creation_with_literal_lists(self) -> None:
        check = DisjointCheck(
            collection=["a", "b"],
            forbidden=["x", "y"],
        )
        assert check.collection == ["a", "b"]
        assert check.forbidden == ["x", "y"]
        assert check.case_sensitive is True
        assert check.negate is False

    def test_jsonpath_auto_conversion(self) -> None:
        check = DisjointCheck(
            collection="$.output.value.tags",
            forbidden="$.test_case.expected.excluded",
            case_sensitive="$.test_case.expected.cs",
            negate="$.test_case.expected.neg",
        )
        assert isinstance(check.collection, JSONPath)
        assert check.collection.expression == "$.output.value.tags"
        assert isinstance(check.forbidden, JSONPath)
        assert isinstance(check.case_sensitive, JSONPath)
        assert isinstance(check.negate, JSONPath)

    def test_invalid_jsonpath_raises(self) -> None:
        with pytest.raises(PydanticValidationError, match="Invalid JSONPath expression"):
            DisjointCheck(collection="$.invalid[", forbidden=["a"])

    def test_check_type_property(self) -> None:
        check = DisjointCheck(collection=["a"], forbidden=["x"])
        assert check.check_type == CheckType.DISJOINT

    def test_invalid_boolean_field_raises(self) -> None:
        with pytest.raises(PydanticValidationError):
            DisjointCheck(collection=["a"], forbidden=["x"], case_sensitive="not_a_boolean")


class TestDisjointExecution:
    """Test DisjointCheck execution logic."""

    def test_no_overlap(self) -> None:
        check = DisjointCheck(collection=["a", "b"], forbidden=["x", "y"])
        result = check()
        assert result["passed"] is True
        assert result["overlapping"] == []

    def test_some_overlap(self) -> None:
        check = DisjointCheck(collection=["a", "b", "c"], forbidden=["b", "x"])
        result = check()
        assert result["passed"] is False
        assert result["overlapping"] == ["b"]

    def test_complete_overlap(self) -> None:
        check = DisjointCheck(collection=["a", "b"], forbidden=["a", "b"])
        result = check()
        assert result["passed"] is False
        assert result["overlapping"] == ["a", "b"]

    def test_empty_collection(self) -> None:
        """Nothing can overlap with empty collection."""
        check = DisjointCheck(collection=[], forbidden=["a", "b"])
        result = check()
        assert result["passed"] is True
        assert result["overlapping"] == []

    def test_empty_forbidden(self) -> None:
        """Nothing is forbidden."""
        check = DisjointCheck(collection=["a", "b"], forbidden=[])
        result = check()
        assert result["passed"] is True
        assert result["overlapping"] == []

    def test_both_empty(self) -> None:
        check = DisjointCheck(collection=[], forbidden=[])
        result = check()
        assert result["passed"] is True

    def test_negate_inverts_disjoint(self) -> None:
        """No overlap → pass, negate → fail."""
        check = DisjointCheck(collection=["a"], forbidden=["x"], negate=True)
        result = check()
        assert result["passed"] is False

    def test_negate_inverts_overlap(self) -> None:
        """Overlap → fail, negate → pass."""
        check = DisjointCheck(collection=["a", "b"], forbidden=["a"], negate=True)
        result = check()
        assert result["passed"] is True

    def test_case_insensitive_finds_overlap(self) -> None:
        check = DisjointCheck(
            collection=["Python", "Flask"],
            forbidden=["python"],
            case_sensitive=False,
        )
        result = check()
        assert result["passed"] is False
        assert result["overlapping"] == ["python"]

    def test_case_sensitive_no_overlap(self) -> None:
        check = DisjointCheck(
            collection=["Python"],
            forbidden=["python"],
            case_sensitive=True,
        )
        result = check()
        assert result["passed"] is True

    def test_case_insensitive_does_not_affect_non_strings(self) -> None:
        check = DisjointCheck(collection=[1], forbidden=["1"], case_sensitive=False)
        result = check()
        assert result["passed"] is True

    def test_duplicates_in_forbidden(self) -> None:
        """Duplicates in forbidden are deduplicated."""
        check = DisjointCheck(collection=["a", "b"], forbidden=["a", "a"])
        result = check()
        assert result["passed"] is False
        assert result["overlapping"] == ["a"]

    def test_duplicates_in_collection(self) -> None:
        check = DisjointCheck(collection=["a", "a", "b"], forbidden=["a"])
        result = check()
        assert result["passed"] is False
        assert result["overlapping"] == ["a"]

    def test_non_string_elements(self) -> None:
        check = DisjointCheck(collection=[1, 2, 3], forbidden=[4, 5])
        result = check()
        assert result["passed"] is True

    def test_non_string_elements_overlap(self) -> None:
        check = DisjointCheck(collection=[1, 2, 3], forbidden=[2, 4])
        result = check()
        assert result["passed"] is False
        assert result["overlapping"] == [2]

    def test_dict_elements(self) -> None:
        check = DisjointCheck(
            collection=[{"key": "val"}],
            forbidden=[{"other": "data"}],
        )
        result = check()
        assert result["passed"] is True

    def test_dict_elements_overlap(self) -> None:
        check = DisjointCheck(
            collection=[{"key": "val"}, {"other": "data"}],
            forbidden=[{"key": "val"}],
        )
        result = check()
        assert result["passed"] is False
        assert result["overlapping"] == [{"key": "val"}]

    def test_set_inputs(self) -> None:
        check = DisjointCheck(collection={"a", "b"}, forbidden={"x", "y"})
        result = check()
        assert result["passed"] is True

    def test_none_elements(self) -> None:
        check = DisjointCheck(collection=[None, "a"], forbidden=[None])
        result = check()
        assert result["passed"] is False
        assert result["overlapping"] == [None]

    def test_negate_with_empty_forbidden(self) -> None:
        """Empty forbidden → disjoint (pass), negate → fail."""
        check = DisjointCheck(collection=["a"], forbidden=[], negate=True)
        result = check()
        assert result["passed"] is False

    def test_case_insensitive_with_negate(self) -> None:
        check = DisjointCheck(
            collection=["Python"],
            forbidden=["python"],
            case_sensitive=False,
            negate=True,
        )
        result = check()
        # Overlap found (case-insensitive), passed=False, negate → True
        assert result["passed"] is True


class TestDisjointErrorHandling:
    """Test error conditions for DisjointCheck."""

    def test_collection_not_list_or_set(self) -> None:
        check = DisjointCheck(collection=["placeholder"], forbidden=["a"])
        check.collection = "not a list"  # type: ignore[assignment]
        with pytest.raises(ValidationError, match="must be a list or set"):
            check()

    def test_forbidden_not_list_or_set(self) -> None:
        check = DisjointCheck(collection=["a"], forbidden=["placeholder"])
        check.forbidden = "not a list"  # type: ignore[assignment]
        with pytest.raises(ValidationError, match="must be a list or set"):
            check()

    def test_collection_is_int(self) -> None:
        check = DisjointCheck(collection=["placeholder"], forbidden=["a"])
        check.collection = 42  # type: ignore[assignment]
        with pytest.raises(ValidationError, match="must be a list or set"):
            check()

    def test_unresolved_jsonpath_collection(self) -> None:
        check = DisjointCheck(collection="$.output.value", forbidden=["a"])
        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'collection'"):
            check()

    def test_unresolved_jsonpath_forbidden(self) -> None:
        check = DisjointCheck(collection=["a"], forbidden="$.test_case.expected")
        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'forbidden'"):
            check()

    def test_unresolved_jsonpath_case_sensitive(self) -> None:
        check = DisjointCheck(
            collection=["a"], forbidden=["x"], case_sensitive="$.test_case.expected.cs",
        )
        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'case_sensitive'"):
            check()

    def test_unresolved_jsonpath_negate(self) -> None:
        check = DisjointCheck(
            collection=["a"], forbidden=["x"], negate="$.test_case.expected.neg",
        )
        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'negate'"):
            check()


class TestDisjointEngineIntegration:
    """Test DisjointCheck integration with the evaluation engine."""

    def test_execute_with_evaluation_context(self) -> None:
        check = DisjointCheck(
            collection="$.output.value.tags",
            forbidden="$.test_case.expected.excluded",
        )
        test_case = TestCase(
            id="test_001",
            input="test",
            expected={"excluded": ["spam", "junk"]},
        )
        output = Output(value={"tags": ["python", "flask"]})
        context = EvaluationContext(test_case, output)

        result = check.execute(context)
        assert result.status == "completed"
        assert result.results["passed"] is True
        assert result.results["overlapping"] == []
        assert result.resolved_arguments["collection"]["value"] == ["python", "flask"]
        assert result.resolved_arguments["forbidden"]["value"] == ["spam", "junk"]

    def test_execute_with_overlap(self) -> None:
        check = DisjointCheck(
            collection="$.output.value.tags",
            forbidden="$.test_case.expected.excluded",
        )
        test_case = TestCase(
            id="test_001",
            input="test",
            expected={"excluded": ["flask", "spam"]},
        )
        output = Output(value={"tags": ["python", "flask"]})
        context = EvaluationContext(test_case, output)

        result = check.execute(context)
        assert result.status == "completed"
        assert result.results["passed"] is False
        assert result.results["overlapping"] == ["flask"]

    def test_wrong_runtime_type_via_execute(self) -> None:
        check = DisjointCheck(collection="$.output.value", forbidden=["a"])
        test_case = TestCase(id="test_001", input="test")
        output = Output(value="not a list")
        context = EvaluationContext(test_case, output)

        result = check.execute(context)
        assert result.status == "error"
        assert result.results == {"passed": False, "overlapping": []}
        assert result.error is not None
        assert "must be a list or set" in result.error.message

    @pytest.mark.parametrize(("output_tags", "forbidden_tags", "expected_passed"), [
        (["a", "b"], ["x", "y"], True),
        (["a", "b"], ["a", "x"], False),
        (["a", "b"], ["a", "b"], False),
        ([], ["a"], True),
        (["a"], [], True),
        ([], [], True),
    ])
    def test_via_evaluate_sync(
        self,
        output_tags: list[str],
        forbidden_tags: list[str],
        expected_passed: bool,
    ) -> None:
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected=forbidden_tags,
                checks=[
                    Check(
                        type=CheckType.DISJOINT,
                        arguments={
                            "collection": "$.output.value",
                            "forbidden": "$.test_case.expected",
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
                expected=["spam", "junk"],
                checks=[
                    DisjointCheck(
                        collection="$.output.value",
                        forbidden="$.test_case.expected",
                    ),
                ],
            ),
        ]
        outputs = [Output(value=["python", "flask"])]
        results = evaluate_sync(test_cases, outputs)

        assert results.summary.completed_test_cases == 1
        assert results.results[0].check_results[0].results["passed"] is True
