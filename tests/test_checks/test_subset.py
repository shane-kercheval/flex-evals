"""
Comprehensive tests for SubsetCheck implementation.

Tests are organized by functionality:
- Pydantic validation tests
- Execution logic tests
- Error handling tests
- Engine integration tests
"""

import pytest

from flex_evals import (
    SubsetCheck,
    JSONPath,
    EvaluationContext,
    CheckType,
    Status,
    evaluate_sync,
    Check,
    ValidationError,
    TestCase,
    Output,
)
from pydantic import ValidationError as PydanticValidationError


class TestSubsetValidation:
    """Test Pydantic validation and field handling for SubsetCheck."""

    def test_creation_with_literal_lists(self) -> None:
        check = SubsetCheck(
            collection=["a", "b", "c"],
            required=["a", "b"],
        )
        assert check.collection == ["a", "b", "c"]
        assert check.required == ["a", "b"]
        assert check.case_sensitive is True
        assert check.negate is False

    def test_jsonpath_auto_conversion(self) -> None:
        check = SubsetCheck(
            collection="$.output.value.tags",
            required="$.test_case.expected.must_contain",
            case_sensitive="$.test_case.expected.case_sensitive",
            negate="$.test_case.expected.should_negate",
        )
        assert isinstance(check.collection, JSONPath)
        assert check.collection.expression == "$.output.value.tags"
        assert isinstance(check.required, JSONPath)
        assert check.required.expression == "$.test_case.expected.must_contain"
        assert isinstance(check.case_sensitive, JSONPath)
        assert isinstance(check.negate, JSONPath)

    def test_invalid_jsonpath_raises(self) -> None:
        with pytest.raises(PydanticValidationError, match="Invalid JSONPath expression"):
            SubsetCheck(collection="$.invalid[", required=["a"])

    def test_check_type_property(self) -> None:
        check = SubsetCheck(collection=["a"], required=["a"])
        assert check.check_type == CheckType.SUBSET

    def test_invalid_boolean_field_raises(self) -> None:
        with pytest.raises(PydanticValidationError):
            SubsetCheck(collection=["a"], required=["a"], case_sensitive="not_a_boolean")


class TestSubsetExecution:
    """Test SubsetCheck execution logic."""

    def test_all_required_present(self) -> None:
        check = SubsetCheck(collection=["a", "b", "c"], required=["a", "b"])
        result = check()
        assert result["passed"] is True
        assert result["found"] == ["a", "b"]
        assert result["missing"] == []

    def test_some_required_missing(self) -> None:
        check = SubsetCheck(collection=["a", "c"], required=["a", "b"])
        result = check()
        assert result["passed"] is False
        assert result["found"] == ["a"]
        assert result["missing"] == ["b"]

    def test_all_required_missing(self) -> None:
        check = SubsetCheck(collection=["x", "y"], required=["a", "b"])
        result = check()
        assert result["passed"] is False
        assert result["missing"] == ["a", "b"]
        assert result["found"] == []

    def test_empty_collection_nonempty_required(self) -> None:
        check = SubsetCheck(collection=[], required=["a"])
        result = check()
        assert result["passed"] is False
        assert result["missing"] == ["a"]

    def test_empty_required(self) -> None:
        """Empty required is vacuously true — empty set is subset of everything."""
        check = SubsetCheck(collection=["a", "b"], required=[])
        result = check()
        assert result["passed"] is True
        assert result["found"] == []
        assert result["missing"] == []

    def test_both_empty(self) -> None:
        check = SubsetCheck(collection=[], required=[])
        result = check()
        assert result["passed"] is True

    def test_negate_inverts_pass(self) -> None:
        check = SubsetCheck(collection=["a", "b"], required=["a", "b"], negate=True)
        result = check()
        assert result["passed"] is False

    def test_negate_inverts_fail(self) -> None:
        check = SubsetCheck(collection=["a"], required=["a", "b"], negate=True)
        result = check()
        assert result["passed"] is True

    def test_case_insensitive_strings(self) -> None:
        check = SubsetCheck(
            collection=["Python", "Flask"],
            required=["python", "flask"],
            case_sensitive=False,
        )
        result = check()
        assert result["passed"] is True
        assert result["found"] == ["python", "flask"]
        assert result["missing"] == []

    def test_case_sensitive_strings(self) -> None:
        check = SubsetCheck(
            collection=["Python", "Flask"],
            required=["python", "flask"],
            case_sensitive=True,
        )
        result = check()
        assert result["passed"] is False

    def test_case_insensitive_does_not_affect_non_strings(self) -> None:
        """case_sensitive=False only affects str elements; int 1 != str '1'."""
        check = SubsetCheck(
            collection=[1, 2, 3],
            required=["1"],
            case_sensitive=False,
        )
        result = check()
        assert result["passed"] is False
        assert result["missing"] == ["1"]

    def test_duplicate_elements_in_required(self) -> None:
        """Duplicates in required are deduplicated (set semantics)."""
        check = SubsetCheck(collection=["a", "b"], required=["a", "a", "b", "b"])
        result = check()
        assert result["passed"] is True
        assert result["found"] == ["a", "b"]

    def test_duplicate_elements_in_collection(self) -> None:
        check = SubsetCheck(collection=["a", "a", "b"], required=["a", "b"])
        result = check()
        assert result["passed"] is True

    def test_non_string_elements(self) -> None:
        check = SubsetCheck(collection=[1, 2, 3], required=[1, 2])
        result = check()
        assert result["passed"] is True
        assert result["found"] == [1, 2]

    def test_mixed_type_elements(self) -> None:
        check = SubsetCheck(collection=[1, "a", True], required=[1, "a"])
        result = check()
        assert result["passed"] is True

    def test_dict_elements(self) -> None:
        """Unhashable elements (dicts) are supported via iteration-based comparison."""
        check = SubsetCheck(
            collection=[{"key": "val"}, {"other": "data"}],
            required=[{"key": "val"}],
        )
        result = check()
        assert result["passed"] is True
        assert result["found"] == [{"key": "val"}]

    def test_dict_elements_missing(self) -> None:
        check = SubsetCheck(
            collection=[{"other": "data"}],
            required=[{"key": "val"}],
        )
        result = check()
        assert result["passed"] is False
        assert result["missing"] == [{"key": "val"}]

    def test_set_inputs(self) -> None:
        check = SubsetCheck(collection={"a", "b", "c"}, required={"a", "b"})
        result = check()
        assert result["passed"] is True

    def test_single_item_required(self) -> None:
        check = SubsetCheck(collection=["a", "b"], required=["a"])
        result = check()
        assert result["passed"] is True
        assert result["found"] == ["a"]

    def test_case_insensitive_deduplication(self) -> None:
        """Deduplication respects case_sensitive setting."""
        check = SubsetCheck(
            collection=["A", "b"],
            required=["a", "A"],
            case_sensitive=False,
        )
        result = check()
        assert result["passed"] is True
        # "a" and "A" are duplicates when case-insensitive; first occurrence kept
        assert len(result["found"]) == 1

    def test_none_elements(self) -> None:
        """None is a valid element, compared with ==."""
        check = SubsetCheck(collection=["a", None, "b"], required=[None])
        result = check()
        assert result["passed"] is True
        assert result["found"] == [None]

    def test_negate_with_empty_required(self) -> None:
        """Empty required is vacuously true; negate inverts to False."""
        check = SubsetCheck(collection=["a"], required=[], negate=True)
        result = check()
        assert result["passed"] is False

    def test_case_insensitive_with_negate(self) -> None:
        """case_sensitive and negate compose without interference."""
        check = SubsetCheck(
            collection=["Python", "Flask"],
            required=["python"],
            case_sensitive=False,
            negate=True,
        )
        result = check()
        # Found (case-insensitive), so passed=True, then negate → False
        assert result["passed"] is False


class TestSubsetErrorHandling:
    """Test error conditions for SubsetCheck."""

    def test_collection_not_list_or_set(self) -> None:
        check = SubsetCheck(collection=["placeholder"], required=["a"])
        check.collection = "not a list"  # type: ignore[assignment]
        with pytest.raises(ValidationError, match="must be a list or set"):
            check()

    def test_required_not_list_or_set(self) -> None:
        check = SubsetCheck(collection=["a"], required=["placeholder"])
        check.required = "not a list"  # type: ignore[assignment]
        with pytest.raises(ValidationError, match="must be a list or set"):
            check()

    def test_collection_is_int(self) -> None:
        check = SubsetCheck(collection=["placeholder"], required=["a"])
        check.collection = 42  # type: ignore[assignment]
        with pytest.raises(ValidationError, match="must be a list or set"):
            check()

    def test_unresolved_jsonpath_collection(self) -> None:
        check = SubsetCheck(collection="$.output.value", required=["a"])
        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'collection'"):
            check()

    def test_unresolved_jsonpath_required(self) -> None:
        check = SubsetCheck(collection=["a"], required="$.test_case.expected")
        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'required'"):
            check()

    def test_unresolved_jsonpath_case_sensitive(self) -> None:
        check = SubsetCheck(
            collection=["a"], required=["a"], case_sensitive="$.test_case.expected.cs",
        )
        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'case_sensitive'"):
            check()

    def test_unresolved_jsonpath_negate(self) -> None:
        check = SubsetCheck(
            collection=["a"], required=["a"], negate="$.test_case.expected.neg",
        )
        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'negate'"):
            check()


class TestSubsetEngineIntegration:
    """Test SubsetCheck integration with the evaluation engine."""

    def test_execute_with_evaluation_context(self) -> None:
        check = SubsetCheck(
            collection="$.output.value.tags",
            required="$.test_case.expected.must_contain",
        )
        test_case = TestCase(
            id="test_001",
            input="test",
            expected={"must_contain": ["python", "flask"]},
        )
        output = Output(value={"tags": ["python", "flask", "api"]})
        context = EvaluationContext(test_case, output)

        result = check.execute(context)
        assert result.status == "completed"
        assert result.results["passed"] is True
        assert result.results["found"] == ["python", "flask"]
        assert result.results["missing"] == []
        assert result.resolved_arguments["collection"]["value"] == ["python", "flask", "api"]
        assert result.resolved_arguments["required"]["value"] == ["python", "flask"]

    def test_execute_with_missing_items(self) -> None:
        check = SubsetCheck(
            collection="$.output.value.tags",
            required="$.test_case.expected.must_contain",
        )
        test_case = TestCase(
            id="test_001",
            input="test",
            expected={"must_contain": ["python", "django"]},
        )
        output = Output(value={"tags": ["python", "flask"]})
        context = EvaluationContext(test_case, output)

        result = check.execute(context)
        assert result.status == "completed"
        assert result.results["passed"] is False
        assert result.results["missing"] == ["django"]

    @pytest.mark.parametrize(("output_tags", "required_tags", "expected_passed"), [
        (["a", "b", "c"], ["a", "b"], True),
        (["a", "b"], ["a", "b", "c"], False),
        (["a"], [], True),
        ([], ["a"], False),
    ])
    def test_via_evaluate_sync(
        self,
        output_tags: list[str],
        required_tags: list[str],
        expected_passed: bool,
    ) -> None:
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected=required_tags,
                checks=[
                    Check(
                        type=CheckType.SUBSET,
                        arguments={
                            "collection": "$.output.value",
                            "required": "$.test_case.expected",
                        },
                    ),
                ],
            ),
        ]
        outputs = [Output(value=output_tags)]
        results = evaluate_sync(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.summary.error_test_cases == 0
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results["passed"] == expected_passed

    def test_check_instance_via_evaluate_sync(self) -> None:
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected=["python", "flask"],
                checks=[
                    SubsetCheck(
                        collection="$.output.value",
                        required="$.test_case.expected",
                    ),
                ],
            ),
        ]
        outputs = [Output(value=["python", "flask", "api"])]
        results = evaluate_sync(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].check_results[0].results["passed"] is True
        assert results.results[0].check_results[0].results["found"] == ["python", "flask"]

    def test_wrong_runtime_type_via_execute(self) -> None:
        """When JSONPath resolves collection to a non-list, execute() returns error CheckResult."""
        check = SubsetCheck(
            collection="$.output.value",
            required=["a"],
        )
        test_case = TestCase(id="test_001", input="test")
        output = Output(value="not a list")
        context = EvaluationContext(test_case, output)

        result = check.execute(context)
        assert result.status == "error"
        assert result.results == {"passed": False, "missing": [], "found": []}
        assert result.error is not None
        assert "must be a list or set" in result.error.message

    def test_case_insensitive_via_evaluate_sync(self) -> None:
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected={"must_contain": ["PYTHON"], "cs": False},
                checks=[
                    Check(
                        type=CheckType.SUBSET,
                        arguments={
                            "collection": "$.output.value",
                            "required": "$.test_case.expected.must_contain",
                            "case_sensitive": "$.test_case.expected.cs",
                        },
                    ),
                ],
            ),
        ]
        outputs = [Output(value=["python", "flask"])]
        results = evaluate_sync(test_cases, outputs)

        assert results.results[0].check_results[0].results["passed"] is True
