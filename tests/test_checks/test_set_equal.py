"""
Comprehensive tests for SetEqualCheck implementation.

Tests are organized by functionality:
- Pydantic validation tests
- Execution logic tests
- Error handling tests
- Engine integration tests
"""

import pytest

from flex_evals import (
    SetEqualCheck,
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


class TestSetEqualValidation:
    """Test Pydantic validation and field handling for SetEqualCheck."""

    def test_creation_with_literal_lists(self) -> None:
        check = SetEqualCheck(
            actual=["a", "b"],
            expected=["a", "b"],
        )
        assert check.actual == ["a", "b"]
        assert check.expected == ["a", "b"]
        assert check.case_sensitive is True
        assert check.negate is False

    def test_jsonpath_auto_conversion(self) -> None:
        check = SetEqualCheck(
            actual="$.output.value.tags",
            expected="$.test_case.expected.tags",
            case_sensitive="$.test_case.expected.cs",
            negate="$.test_case.expected.neg",
        )
        assert isinstance(check.actual, JSONPath)
        assert check.actual.expression == "$.output.value.tags"
        assert isinstance(check.expected, JSONPath)
        assert isinstance(check.case_sensitive, JSONPath)
        assert isinstance(check.negate, JSONPath)

    def test_invalid_jsonpath_raises(self) -> None:
        with pytest.raises(PydanticValidationError, match="Invalid JSONPath expression"):
            SetEqualCheck(actual="$.invalid[", expected=["a"])

    def test_check_type_property(self) -> None:
        check = SetEqualCheck(actual=["a"], expected=["a"])
        assert check.check_type == CheckType.SET_EQUAL

    def test_invalid_boolean_field_raises(self) -> None:
        with pytest.raises(PydanticValidationError):
            SetEqualCheck(actual=["a"], expected=["a"], case_sensitive="not_a_boolean")


class TestSetEqualExecution:
    """Test SetEqualCheck execution logic."""

    def test_same_elements_same_order(self) -> None:
        check = SetEqualCheck(actual=["a", "b", "c"], expected=["a", "b", "c"])
        result = check()
        assert result["passed"] is True
        assert result["missing"] == []
        assert result["unexpected"] == []

    def test_same_elements_different_order(self) -> None:
        check = SetEqualCheck(actual=["c", "a", "b"], expected=["a", "b", "c"])
        result = check()
        assert result["passed"] is True

    def test_duplicates_in_one_side(self) -> None:
        """Set semantics — duplicates are ignored."""
        check = SetEqualCheck(actual=["a", "a", "b"], expected=["b", "a"])
        result = check()
        assert result["passed"] is True

    def test_duplicates_in_both_sides(self) -> None:
        check = SetEqualCheck(actual=["a", "a"], expected=["a", "a", "a"])
        result = check()
        assert result["passed"] is True

    def test_missing_items(self) -> None:
        check = SetEqualCheck(actual=["a"], expected=["a", "b"])
        result = check()
        assert result["passed"] is False
        assert result["missing"] == ["b"]
        assert result["unexpected"] == []

    def test_unexpected_items(self) -> None:
        check = SetEqualCheck(actual=["a", "b", "c"], expected=["a"])
        result = check()
        assert result["passed"] is False
        assert result["missing"] == []
        assert result["unexpected"] == ["b", "c"]

    def test_both_missing_and_unexpected(self) -> None:
        check = SetEqualCheck(actual=["a", "x"], expected=["a", "b"])
        result = check()
        assert result["passed"] is False
        assert result["missing"] == ["b"]
        assert result["unexpected"] == ["x"]

    def test_completely_different(self) -> None:
        check = SetEqualCheck(actual=["x", "y"], expected=["a", "b"])
        result = check()
        assert result["passed"] is False
        assert result["missing"] == ["a", "b"]
        assert result["unexpected"] == ["x", "y"]

    def test_both_empty(self) -> None:
        check = SetEqualCheck(actual=[], expected=[])
        result = check()
        assert result["passed"] is True

    def test_empty_actual_nonempty_expected(self) -> None:
        check = SetEqualCheck(actual=[], expected=["a"])
        result = check()
        assert result["passed"] is False
        assert result["missing"] == ["a"]

    def test_nonempty_actual_empty_expected(self) -> None:
        check = SetEqualCheck(actual=["a"], expected=[])
        result = check()
        assert result["passed"] is False
        assert result["unexpected"] == ["a"]

    def test_negate_inverts_equal(self) -> None:
        check = SetEqualCheck(actual=["a", "b"], expected=["a", "b"], negate=True)
        result = check()
        assert result["passed"] is False

    def test_negate_inverts_not_equal(self) -> None:
        check = SetEqualCheck(actual=["a"], expected=["a", "b"], negate=True)
        result = check()
        assert result["passed"] is True

    def test_case_insensitive_strings(self) -> None:
        check = SetEqualCheck(
            actual=["Python", "Flask"],
            expected=["python", "flask"],
            case_sensitive=False,
        )
        result = check()
        assert result["passed"] is True

    def test_case_sensitive_strings(self) -> None:
        check = SetEqualCheck(
            actual=["Python"],
            expected=["python"],
            case_sensitive=True,
        )
        result = check()
        assert result["passed"] is False

    def test_case_insensitive_does_not_affect_non_strings(self) -> None:
        check = SetEqualCheck(actual=[1], expected=["1"], case_sensitive=False)
        result = check()
        assert result["passed"] is False

    def test_non_string_elements(self) -> None:
        check = SetEqualCheck(actual=[1, 2, 3], expected=[3, 2, 1])
        result = check()
        assert result["passed"] is True

    def test_dict_elements(self) -> None:
        check = SetEqualCheck(
            actual=[{"key": "val"}, {"other": "data"}],
            expected=[{"other": "data"}, {"key": "val"}],
        )
        result = check()
        assert result["passed"] is True

    def test_dict_elements_not_equal(self) -> None:
        check = SetEqualCheck(
            actual=[{"key": "val"}],
            expected=[{"key": "other"}],
        )
        result = check()
        assert result["passed"] is False

    def test_set_inputs(self) -> None:
        check = SetEqualCheck(actual={"a", "b"}, expected={"b", "a"})
        result = check()
        assert result["passed"] is True

    def test_none_elements(self) -> None:
        check = SetEqualCheck(actual=[None, "a"], expected=["a", None])
        result = check()
        assert result["passed"] is True

    def test_negate_with_both_empty(self) -> None:
        """Both empty = equal; negate inverts to False."""
        check = SetEqualCheck(actual=[], expected=[], negate=True)
        result = check()
        assert result["passed"] is False

    def test_case_insensitive_with_negate(self) -> None:
        check = SetEqualCheck(
            actual=["Python"],
            expected=["python"],
            case_sensitive=False,
            negate=True,
        )
        result = check()
        # Equal (case-insensitive), passed=True, negate → False
        assert result["passed"] is False


class TestSetEqualErrorHandling:
    """Test error conditions for SetEqualCheck."""

    def test_actual_not_list_or_set(self) -> None:
        check = SetEqualCheck(actual=["placeholder"], expected=["a"])
        check.actual = "not a list"  # type: ignore[assignment]
        with pytest.raises(ValidationError, match="must be a list or set"):
            check()

    def test_expected_not_list_or_set(self) -> None:
        check = SetEqualCheck(actual=["a"], expected=["placeholder"])
        check.expected = "not a list"  # type: ignore[assignment]
        with pytest.raises(ValidationError, match="must be a list or set"):
            check()

    def test_actual_is_int(self) -> None:
        check = SetEqualCheck(actual=["placeholder"], expected=["a"])
        check.actual = 42  # type: ignore[assignment]
        with pytest.raises(ValidationError, match="must be a list or set"):
            check()

    def test_unresolved_jsonpath_actual(self) -> None:
        check = SetEqualCheck(actual="$.output.value", expected=["a"])
        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'actual'"):
            check()

    def test_unresolved_jsonpath_expected(self) -> None:
        check = SetEqualCheck(actual=["a"], expected="$.test_case.expected")
        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'expected'"):
            check()

    def test_unresolved_jsonpath_case_sensitive(self) -> None:
        check = SetEqualCheck(
            actual=["a"], expected=["a"], case_sensitive="$.test_case.expected.cs",
        )
        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'case_sensitive'"):
            check()

    def test_unresolved_jsonpath_negate(self) -> None:
        check = SetEqualCheck(
            actual=["a"], expected=["a"], negate="$.test_case.expected.neg",
        )
        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'negate'"):
            check()


class TestSetEqualEngineIntegration:
    """Test SetEqualCheck integration with the evaluation engine."""

    def test_execute_with_evaluation_context(self) -> None:
        check = SetEqualCheck(
            actual="$.output.value.tags",
            expected="$.test_case.expected.tags",
        )
        test_case = TestCase(
            id="test_001",
            input="test",
            expected={"tags": ["flask", "python"]},
        )
        output = Output(value={"tags": ["python", "flask"]})
        context = EvaluationContext(test_case, output)

        result = check.execute(context)
        assert result.status == "completed"
        assert result.results["passed"] is True
        assert result.results["missing"] == []
        assert result.results["unexpected"] == []
        assert result.resolved_arguments["actual"]["value"] == ["python", "flask"]
        assert result.resolved_arguments["expected"]["value"] == ["flask", "python"]

    def test_execute_with_differences(self) -> None:
        check = SetEqualCheck(
            actual="$.output.value.tags",
            expected="$.test_case.expected.tags",
        )
        test_case = TestCase(
            id="test_001",
            input="test",
            expected={"tags": ["python", "django"]},
        )
        output = Output(value={"tags": ["python", "flask"]})
        context = EvaluationContext(test_case, output)

        result = check.execute(context)
        assert result.status == "completed"
        assert result.results["passed"] is False
        assert result.results["missing"] == ["django"]
        assert result.results["unexpected"] == ["flask"]

    def test_wrong_runtime_type_via_execute(self) -> None:
        check = SetEqualCheck(actual="$.output.value", expected=["a"])
        test_case = TestCase(id="test_001", input="test")
        output = Output(value="not a list")
        context = EvaluationContext(test_case, output)

        result = check.execute(context)
        assert result.status == "error"
        assert result.results == {"passed": False, "missing": [], "unexpected": []}
        assert result.error is not None
        assert "must be a list or set" in result.error.message

    @pytest.mark.parametrize(("output_tags", "expected_tags", "expected_passed"), [
        (["a", "b"], ["b", "a"], True),
        (["a", "b"], ["a", "b", "c"], False),
        (["a", "b", "c"], ["a", "b"], False),
        ([], [], True),
        (["a"], [], False),
        ([], ["a"], False),
    ])
    def test_via_evaluate_sync(
        self,
        output_tags: list[str],
        expected_tags: list[str],
        expected_passed: bool,
    ) -> None:
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected=expected_tags,
                checks=[
                    Check(
                        type=CheckType.SET_EQUAL,
                        arguments={
                            "actual": "$.output.value",
                            "expected": "$.test_case.expected",
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
                expected=["python", "flask"],
                checks=[
                    SetEqualCheck(
                        actual="$.output.value",
                        expected="$.test_case.expected",
                    ),
                ],
            ),
        ]
        outputs = [Output(value=["flask", "python"])]
        results = evaluate_sync(test_cases, outputs)

        assert results.summary.completed_test_cases == 1
        assert results.results[0].check_results[0].results["passed"] is True
