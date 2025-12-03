"""
Comprehensive tests for CustomFunctionCheck implementation.

This module consolidates all tests for the CustomFunctionCheck including:
- Pydantic validation tests
- Implementation execution tests (sync and async functions)
- Engine integration tests
- Edge cases and error handling
- String function definitions (lambda and named functions)

Tests are organized by functionality rather than implementation details.
"""

import asyncio
import pytest

from flex_evals import (
    CustomFunctionCheck,
    JSONPath,
    EvaluationContext,
    CheckType,
    Status,
    evaluate_sync,
    Check,
    ValidationError,
    CheckExecutionError,
    TestCase,
    Output,
)
from pydantic import ValidationError as PydanticValidationError


# Test helper functions
def simple_validation(text: str, expected: str) -> dict[str, bool]:
    """Simple sync validation function for testing."""
    return {'passed': text == expected}


def validation_with_multiple_args(value: int, threshold: int, operator: str) -> dict[str, bool]:
    """Validation function with multiple arguments."""
    if operator == 'gt':
        passed = value > threshold
    elif operator == 'lt':
        passed = value < threshold
    elif operator == 'eq':
        passed = value == threshold
    else:
        passed = False
    return {'passed': passed, 'value': value, 'threshold': threshold}


async def async_validation(text: str, expected: str) -> dict[str, bool]:
    """Async validation function for testing."""
    await asyncio.sleep(0.001)  # Simulate async work
    return {'passed': text == expected, 'async': True}


def complex_validation(data: dict[str, int]) -> dict[str, int | bool]:
    """Validation function that processes complex data."""
    total = sum(data.values())
    return {
        'passed': total > 0,
        'total': total,
        'count': len(data),
    }


def validation_returns_non_dict(value: str) -> str:  # noqa: ARG001
    """Validation function that returns non-dict (should fail)."""
    return "not_a_dict"


class TestCustomFunctionValidation:
    """Test Pydantic validation and field handling for CustomFunctionCheck."""

    def test_custom_function_check_with_jsonpath(self) -> None:
        """Test CustomFunctionCheck creation with JSONPath expressions."""
        check = CustomFunctionCheck(
            validation_function="$.test_case.expected.validator",
            function_args="$.test_case.expected.args",
        )

        assert isinstance(check.validation_function, JSONPath)
        assert check.validation_function.expression == "$.test_case.expected.validator"
        assert isinstance(check.function_args, JSONPath)
        assert check.function_args.expression == "$.test_case.expected.args"

    def test_custom_function_check_default_function_args(self) -> None:
        """Test CustomFunctionCheck with default empty function_args."""
        check = CustomFunctionCheck(
            validation_function=lambda: {'passed': True},
        )

        assert check.function_args == {}

    @pytest.mark.asyncio
    async def test_custom_function_jsonpath_comprehensive(self) -> None:
        """Comprehensive JSONPath string conversion and execution test."""
        # Create check with JSONPath fields as strings
        check = CustomFunctionCheck(
            validation_function="$.test_case.expected.function",
            function_args="$.test_case.expected.arguments",
        )

        # Verify conversion happened
        assert isinstance(check.validation_function, JSONPath)
        assert check.validation_function.expression == "$.test_case.expected.function"
        assert isinstance(check.function_args, JSONPath)
        assert check.function_args.expression == "$.test_case.expected.arguments"

        # Test execution with EvaluationContext
        test_case = TestCase(
            id="test_001",
            input="test",
            expected={
                "function": simple_validation,
                "arguments": {"text": "hello", "expected": "hello"},
            },
        )
        output = Output(value="dummy")
        context = EvaluationContext(test_case, output)

        result = await check.execute(context)
        assert result.status == Status.COMPLETED
        assert result.results['passed'] is True

        # Verify resolved_arguments
        assert result.resolved_arguments["validation_function"]["value"] == simple_validation
        assert result.resolved_arguments["function_args"]["value"] == {
            "text": "hello",
            "expected": "hello",
        }

    def test_custom_function_check_invalid_jsonpath(self) -> None:
        """Test that invalid JSONPath expressions are caught during validation."""
        with pytest.raises(PydanticValidationError, match="Invalid JSONPath expression"):
            CustomFunctionCheck(
                validation_function="$.invalid[",
                function_args={},
            )

        with pytest.raises(PydanticValidationError, match="Invalid JSONPath expression"):
            CustomFunctionCheck(
                validation_function=simple_validation,
                function_args="$.invalid[",
            )

    def test_custom_function_check_type_property(self) -> None:
        """Test CustomFunctionCheck check_type property returns correct type."""
        check = CustomFunctionCheck(
            validation_function=simple_validation,
            function_args={},
        )
        assert check.check_type == CheckType.CUSTOM_FUNCTION

    def test_custom_function_check_required_fields(self) -> None:
        """Test that required fields are enforced."""
        with pytest.raises(PydanticValidationError):
            CustomFunctionCheck()  # type: ignore


class TestCustomFunctionExecution:
    """Test CustomFunctionCheck execution logic and __call__ method."""

    @pytest.mark.asyncio
    async def test_custom_function_sync_execution(self) -> None:
        """Test execution with synchronous validation function."""
        check = CustomFunctionCheck(
            validation_function=simple_validation,
            function_args={'text': 'hello', 'expected': 'hello'},
        )
        result = await check()
        assert result == {'passed': True}

    @pytest.mark.asyncio
    async def test_custom_function_async_execution(self) -> None:
        """Test execution with asynchronous validation function."""
        check = CustomFunctionCheck(
            validation_function=async_validation,
            function_args={'text': 'world', 'expected': 'world'},
        )
        result = await check()
        assert result == {'passed': True, 'async': True}

    @pytest.mark.asyncio
    async def test_custom_function_multiple_args(self) -> None:
        """Test execution with multiple function arguments."""
        check = CustomFunctionCheck(
            validation_function=validation_with_multiple_args,
            function_args={'value': 10, 'threshold': 5, 'operator': 'gt'},
        )
        result = await check()
        assert result['passed'] is True
        assert result['value'] == 10
        assert result['threshold'] == 5

    @pytest.mark.asyncio
    async def test_custom_function_complex_data(self) -> None:
        """Test execution with complex data structures."""
        check = CustomFunctionCheck(
            validation_function=complex_validation,
            function_args={'data': {'a': 1, 'b': 2, 'c': 3}},
        )
        result = await check()
        assert result['passed'] is True
        assert result['total'] == 6
        assert result['count'] == 3

    @pytest.mark.asyncio
    async def test_custom_function_lambda_string(self) -> None:
        """Test execution with lambda function as string."""
        check = CustomFunctionCheck(
            validation_function='lambda x, y: {"passed": x == y, "result": "equal" if x == y else "not equal"}',  # noqa: E501
            function_args={'x': 42, 'y': 42},
        )
        result = await check()
        assert result['passed'] is True
        assert result['result'] == 'equal'

    @pytest.mark.asyncio
    async def test_custom_function_named_function_string(self) -> None:
        """Test execution with named function as string."""
        func_string = """
def validate_range(value, min_val, max_val):
    in_range = min_val <= value <= max_val
    return {
        'passed': in_range,
        'value': value,
        'in_range': in_range
    }
"""
        check = CustomFunctionCheck(
            validation_function=func_string,
            function_args={'value': 50, 'min_val': 0, 'max_val': 100},
        )
        result = await check()
        assert result['passed'] is True
        assert result['value'] == 50
        assert result['in_range'] is True

    @pytest.mark.asyncio
    async def test_custom_function_with_imports_in_string(self) -> None:
        """Test execution with string function that uses imported modules."""
        func_string = """
def validate_pattern(text, pattern):
    import re
    match = re.search(pattern, text)
    return {
        'passed': match is not None,
        'matched': match.group(0) if match else None
    }
"""
        check = CustomFunctionCheck(
            validation_function=func_string,
            function_args={'text': 'hello world 123', 'pattern': r'\d+'},
        )
        result = await check()
        assert result['passed'] is True
        assert result['matched'] == '123'

    @pytest.mark.asyncio
    async def test_custom_function_no_args(self) -> None:
        """Test execution with function that takes no arguments."""
        check = CustomFunctionCheck(
            validation_function=lambda: {'passed': True, 'constant': 'value'},
        )
        result = await check()
        assert result == {'passed': True, 'constant': 'value'}


class TestCustomFunctionEngineIntegration:
    """Test CustomFunctionCheck integration with the evaluation engine."""

    def test_custom_function_via_evaluate_sync(self) -> None:
        """Test CustomFunctionCheck through engine evaluation."""
        test_cases = [
            TestCase(
                id="test_001",
                input="What is the capital of France?",
                expected="Paris",
                checks=[
                    Check(
                        type=CheckType.CUSTOM_FUNCTION,
                        arguments={
                            "validation_function": simple_validation,
                            "function_args": {
                                "text": "$.output.value",
                                "expected": "$.test_case.expected",
                            },
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="Paris")]
        results = evaluate_sync(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.summary.error_test_cases == 0
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results['passed'] is True

        # Verify resolved_arguments for JSONPath strings in function_args dict
        check_result = results.results[0].check_results[0]
        assert check_result.resolved_arguments["validation_function"]["value"] == simple_validation
        # function_args should have resolved the JSONPath strings
        assert check_result.resolved_arguments["function_args"]["resolved_from"] == "literal"
        # The dict values themselves were resolved by our custom resolve_fields

    def test_custom_function_check_instance_via_evaluate_sync(self) -> None:
        """Test direct check instance usage in evaluate function."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected={"text": "hello", "match": "hello"},
                checks=[
                    CustomFunctionCheck(
                        validation_function=simple_validation,
                        function_args={
                            "text": "$.test_case.expected.text",
                            "expected": "$.test_case.expected.match",
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="dummy")]
        results = evaluate_sync(test_cases, outputs)

        assert results.summary.total_test_cases == 1
        assert results.summary.completed_test_cases == 1
        assert results.results[0].status == Status.COMPLETED
        assert results.results[0].check_results[0].results['passed'] is True

        # Verify resolved_arguments - dict values should have been resolved from JSONPath
        check_result = results.results[0].check_results[0]
        assert check_result.resolved_arguments["validation_function"]["value"] == simple_validation
        # function_args is a literal dict, but its values were JSONPath strings that got resolved
        assert check_result.resolved_arguments["function_args"]["resolved_from"] == "literal"

    def test_custom_function_with_lambda_string_via_evaluate_sync(self) -> None:
        """Test CustomFunctionCheck with lambda string through engine."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.CUSTOM_FUNCTION,
                        arguments={
                            "validation_function": 'lambda value: {"passed": value > 10, "value": value}',  # noqa: E501
                            "function_args": {"value": "$.output.value"},
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value=15)]
        results = evaluate_sync(test_cases, outputs)

        assert results.results[0].check_results[0].results['passed'] is True
        assert results.results[0].check_results[0].results['value'] == 15

    def test_custom_function_async_via_evaluate_sync(self) -> None:
        """Test async validation function through engine."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected="match",
                checks=[
                    CustomFunctionCheck(
                        validation_function=async_validation,
                        function_args={
                            "text": "$.output.value",
                            "expected": "$.test_case.expected",
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="match")]
        results = evaluate_sync(test_cases, outputs)

        assert results.results[0].check_results[0].results['passed'] is True
        assert results.results[0].check_results[0].results['async'] is True

    def test_custom_function_complex_validation_via_evaluate_sync(self) -> None:
        """Test complex validation function with nested data structures."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    CustomFunctionCheck(
                        validation_function=complex_validation,
                        function_args={"data": "$.output.value.scores"},
                    ),
                ],
            ),
        ]

        outputs = [Output(value={"scores": {"math": 85, "science": 92, "english": 88}})]
        results = evaluate_sync(test_cases, outputs)

        assert results.results[0].check_results[0].results['passed'] is True
        assert results.results[0].check_results[0].results['total'] == 265
        assert results.results[0].check_results[0].results['count'] == 3


class TestCustomFunctionErrorHandling:
    """Test error handling and edge cases for CustomFunctionCheck."""

    @pytest.mark.asyncio
    async def test_custom_function_unresolved_jsonpath_validation_function(self) -> None:
        """Test RuntimeError when validation_function JSONPath is not resolved."""
        check = CustomFunctionCheck(
            validation_function="$.test_case.function",
            function_args={},
        )

        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'validation_function'"):
            await check()

    @pytest.mark.asyncio
    async def test_custom_function_unresolved_jsonpath_function_args(self) -> None:
        """Test RuntimeError when function_args JSONPath is not resolved."""
        check = CustomFunctionCheck(
            validation_function=simple_validation,
            function_args="$.test_case.args",
        )

        with pytest.raises(RuntimeError, match="JSONPath not resolved for 'function_args'"):
            await check()

    @pytest.mark.asyncio
    async def test_custom_function_invalid_string_function(self) -> None:
        """Test ValidationError when string function is invalid."""
        check = CustomFunctionCheck(
            validation_function='this is not valid python',
            function_args={},
        )

        with pytest.raises(ValidationError, match="Failed to create function from string"):
            await check()

    @pytest.mark.asyncio
    async def test_custom_function_string_with_no_function_defined(self) -> None:
        """Test ValidationError when string has no function definition."""
        check = CustomFunctionCheck(
            validation_function='x = 42',  # No function defined
            function_args={},
        )

        with pytest.raises(ValidationError, match="No function found in definition"):
            await check()

    @pytest.mark.asyncio
    async def test_custom_function_not_callable(self) -> None:
        """Test ValidationError when validation_function is not callable and not a valid string."""
        check = CustomFunctionCheck(
            validation_function="not a function",  # String that can't be converted to function
            function_args={},
        )

        # This will fail when trying to convert the string to a function
        with pytest.raises(ValidationError, match="Failed to create function from string"):
            await check()

    @pytest.mark.asyncio
    async def test_custom_function_returns_non_dict(self) -> None:
        """Test ValidationError when function returns non-dict."""
        check = CustomFunctionCheck(
            validation_function=validation_returns_non_dict,
            function_args={'value': 'test'},
        )

        with pytest.raises(ValidationError, match="must return dict"):
            await check()

    @pytest.mark.asyncio
    async def test_custom_function_execution_error(self) -> None:
        """Test CheckExecutionError when function raises exception."""
        def failing_function(value: int) -> dict[str, bool]:  # noqa: ARG001
            raise ValueError("Intentional error")

        check = CustomFunctionCheck(
            validation_function=failing_function,
            function_args={'value': 42},
        )

        with pytest.raises(CheckExecutionError, match="Error executing validation function"):
            await check()

    @pytest.mark.asyncio
    async def test_custom_function_missing_required_argument(self) -> None:
        """Test CheckExecutionError when required function argument is missing."""
        check = CustomFunctionCheck(
            validation_function=simple_validation,
            function_args={'text': 'hello'},  # Missing 'expected' argument
        )

        with pytest.raises(CheckExecutionError, match="Error executing validation function"):
            await check()

    def test_custom_function_invalid_jsonpath_in_engine(self) -> None:
        """Test that invalid JSONPath expressions are caught during evaluation."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    Check(
                        type=CheckType.CUSTOM_FUNCTION,
                        arguments={
                            "validation_function": "$..[invalid",
                            "function_args": {},
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="test")]

        with pytest.raises(ValidationError, match="Invalid JSONPath expression"):
            evaluate_sync(test_cases, outputs)

    def test_custom_function_default_results(self) -> None:
        """Test that default_results property returns correct structure."""
        check = CustomFunctionCheck(
            validation_function=simple_validation,
            function_args={},
        )
        default = check.default_results
        assert default == {}


class TestCustomFunctionJSONPathIntegration:
    """Test CustomFunctionCheck with various JSONPath expressions and data structures."""

    def test_custom_function_nested_jsonpath(self) -> None:
        """Test custom function with deeply nested JSONPath expressions."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected={
                    "validation": {
                        "function": simple_validation,
                        "params": {"text": "success", "expected": "success"},
                    },
                },
                checks=[
                    Check(
                        type=CheckType.CUSTOM_FUNCTION,
                        arguments={
                            "validation_function": "$.test_case.expected.validation.function",
                            "function_args": "$.test_case.expected.validation.params",
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="dummy")]
        results = evaluate_sync(test_cases, outputs)

        assert results.results[0].check_results[0].results['passed'] is True

    def test_custom_function_array_access_jsonpath(self) -> None:
        """Test custom function with JSONPath array access."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected={
                    "functions": [simple_validation, complex_validation],
                    "args_list": [
                        {"text": "hello", "expected": "hello"},
                        {"data": {"a": 1, "b": 2}},
                    ],
                },
                checks=[
                    Check(
                        type=CheckType.CUSTOM_FUNCTION,
                        arguments={
                            "validation_function": "$.test_case.expected.functions[0]",
                            "function_args": "$.test_case.expected.args_list[0]",
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="dummy")]
        results = evaluate_sync(test_cases, outputs)

        assert results.results[0].check_results[0].results['passed'] is True

    def test_custom_function_mixed_jsonpath_and_literals(self) -> None:
        """Test custom function with mixed JSONPath and literal arguments."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                expected="Paris",
                checks=[
                    Check(
                        type=CheckType.CUSTOM_FUNCTION,
                        arguments={
                            "validation_function": simple_validation,
                            "function_args": {
                                "text": "$.output.value",
                                "expected": "$.test_case.expected",
                            },
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value="Paris")]
        results = evaluate_sync(test_cases, outputs)

        assert results.results[0].check_results[0].results['passed'] is True

        # Verify that JSONPath strings in dict were resolved
        check_result = results.results[0].check_results[0]
        assert check_result.resolved_arguments["validation_function"]["value"] == simple_validation
        # The function_args dict itself is a literal, but contained JSONPath strings as values
        assert check_result.resolved_arguments["function_args"]["resolved_from"] == "literal"

    def test_custom_function_jsonpath_to_complex_structure(self) -> None:
        """Test custom function with JSONPath resolving to complex data."""
        test_cases = [
            TestCase(
                id="test_001",
                input="test",
                checks=[
                    CustomFunctionCheck(
                        validation_function=complex_validation,
                        function_args={"data": "$.output.value.metrics"},
                    ),
                ],
            ),
        ]

        outputs = [
            Output(value={
                "metrics": {
                    "response_time": 100,
                    "throughput": 50,
                    "accuracy": 95,
                },
            }),
        ]

        results = evaluate_sync(test_cases, outputs)

        assert results.results[0].check_results[0].results['passed'] is True
        assert results.results[0].check_results[0].results['total'] == 245

    def test_custom_function_jsonpath_string_function_comprehensive(self) -> None:
        """
        Comprehensive test with JSONPath to string function AND JSONPath args.

        This tests the most complex scenario:
        - validation_function is a JSONPath that resolves to a string function definition
        - function_args contains JSONPath strings that need resolution
        - Full execution through the engine
        """
        # Define a string function that will be stored in test_case.expected
        func_string = """
def validate_score(actual_score, expected_score, tolerance):
    difference = abs(actual_score - expected_score)
    passed = difference <= tolerance
    return {
        'passed': passed,
        'actual': actual_score,
        'expected': expected_score,
        'difference': difference,
        'within_tolerance': passed
    }
"""

        test_cases = [
            TestCase(
                id="test_001",
                input="Calculate score",
                expected={
                    "validator": func_string,  # String function definition
                    "expected_value": 85,
                    "tolerance_level": 5,
                },
                checks=[
                    Check(
                        type=CheckType.CUSTOM_FUNCTION,
                        arguments={
                            # JSONPath to string function definition
                            "validation_function": "$.test_case.expected.validator",
                            # JSONPath strings in function_args dict
                            "function_args": {
                                "actual_score": "$.output.value.score",
                                "expected_score": "$.test_case.expected.expected_value",
                                "tolerance": "$.test_case.expected.tolerance_level",
                            },
                        },
                    ),
                ],
            ),
        ]

        outputs = [Output(value={"score": 88})]  # Score is 88, expected 85, tolerance 5
        results = evaluate_sync(test_cases, outputs)

        # Should pass: |88 - 85| = 3, which is <= 5
        assert results.results[0].status == Status.COMPLETED
        check_result = results.results[0].check_results[0]
        assert check_result.status == Status.COMPLETED
        assert check_result.results['passed'] is True
        assert check_result.results['actual'] == 88
        assert check_result.results['expected'] == 85
        assert check_result.results['difference'] == 3
        assert check_result.results['within_tolerance'] is True

        # Verify resolved_arguments captured everything correctly
        # validation_function should be resolved from JSONPath to the string function definition
        assert "jsonpath" in check_result.resolved_arguments["validation_function"]
        assert (
            check_result.resolved_arguments["validation_function"]["jsonpath"]
            == "$.test_case.expected.validator"
        )
        assert func_string in check_result.resolved_arguments["validation_function"]["value"]

        # function_args is a literal dict, but its values were JSONPath strings
        # NOTE: resolved_arguments shows the original dict with JSONPath strings,
        # not the final resolved values. The actual resolution happens in
        # CustomFunctionCheck.resolve_fields() and the resolved values are used
        # for execution, but they're not captured in resolved_arguments.
        # This is a known limitation of the current implementation.
        assert check_result.resolved_arguments["function_args"]["resolved_from"] == "literal"
        # The dict still contains JSONPath strings (not yet resolved in resolved_arguments)
        function_args_value = str(check_result.resolved_arguments["function_args"]["value"])
        assert "$.output.value.score" in function_args_value


class TestCustomFunctionStringFunctions:
    """Test CustomFunctionCheck with various string function definitions."""

    @pytest.mark.asyncio
    async def test_lambda_with_math_operations(self) -> None:
        """Test lambda function with mathematical operations."""
        check = CustomFunctionCheck(
            validation_function='lambda x, y: {"passed": x + y > 100, "sum": x + y}',
            function_args={'x': 60, 'y': 50},
        )
        result = await check()
        assert result['passed'] is True
        assert result['sum'] == 110

    @pytest.mark.asyncio
    async def test_lambda_with_string_operations(self) -> None:
        """Test lambda function with string operations."""
        check = CustomFunctionCheck(
            validation_function=(
                'lambda text: {"passed": text.upper() == "HELLO", "upper": text.upper()}'
            ),
            function_args={'text': 'hello'},
        )
        result = await check()
        assert result['passed'] is True
        assert result['upper'] == 'HELLO'

    @pytest.mark.asyncio
    async def test_named_function_with_conditionals(self) -> None:
        """Test named function with conditional logic."""
        func_string = """
def grade_score(score):
    if score >= 90:
        grade = 'A'
        passed = True
    elif score >= 80:
        grade = 'B'
        passed = True
    elif score >= 70:
        grade = 'C'
        passed = True
    else:
        grade = 'F'
        passed = False
    return {'passed': passed, 'grade': grade, 'score': score}
"""
        check = CustomFunctionCheck(
            validation_function=func_string,
            function_args={'score': 85},
        )
        result = await check()
        assert result['passed'] is True
        assert result['grade'] == 'B'
        assert result['score'] == 85

    @pytest.mark.asyncio
    async def test_named_function_with_list_operations(self) -> None:
        """Test named function with list operations."""
        func_string = """
def analyze_list(values):
    return {
        'passed': len(values) > 0,
        'length': len(values),
        'sum': sum(values),
        'average': sum(values) / len(values) if values else 0
    }
"""
        check = CustomFunctionCheck(
            validation_function=func_string,
            function_args={'values': [10, 20, 30, 40]},
        )
        result = await check()
        assert result['passed'] is True
        assert result['length'] == 4
        assert result['sum'] == 100
        assert result['average'] == 25.0

    @pytest.mark.asyncio
    async def test_function_using_json_module(self) -> None:
        """Test string function that uses json module."""
        func_string = """
def validate_json(json_string):
    import json
    try:
        data = json.loads(json_string)
        return {'passed': True, 'data': data}
    except:
        return {'passed': False, 'data': None}
"""
        check = CustomFunctionCheck(
            validation_function=func_string,
            function_args={'json_string': '{"key": "value"}'},
        )
        result = await check()
        assert result['passed'] is True
        assert result['data'] == {'key': 'value'}

    @pytest.mark.asyncio
    async def test_function_using_datetime_module(self) -> None:
        """Test string function that uses datetime module."""
        func_string = """
def check_date_format(date_string):
    import datetime
    try:
        datetime.datetime.strptime(date_string, '%Y-%m-%d')
        return {'passed': True, 'valid_format': True}
    except ValueError:
        return {'passed': False, 'valid_format': False}
"""
        check = CustomFunctionCheck(
            validation_function=func_string,
            function_args={'date_string': '2024-01-15'},
        )
        result = await check()
        assert result['passed'] is True
        assert result['valid_format'] is True

    @pytest.mark.asyncio
    async def test_async_string_function(self) -> None:
        """Test async function defined as string."""
        func_string = """
async def async_validator(text, expected):
    import asyncio
    await asyncio.sleep(0.001)  # Simulate async work
    match = text == expected
    return {
        'passed': match,
        'text': text,
        'expected': expected,
        'async': True
    }
"""
        check = CustomFunctionCheck(
            validation_function=func_string,
            function_args={'text': 'hello', 'expected': 'hello'},
        )
        result = await check()
        assert result['passed'] is True
        assert result['text'] == 'hello'
        assert result['expected'] == 'hello'
        assert result['async'] is True

    @pytest.mark.asyncio
    async def test_validation_function_returns_failed(self) -> None:
        """Test validation function that returns passed=False."""
        func_string = """
def validate_threshold(value, min_threshold, max_threshold):
    within_range = min_threshold <= value <= max_threshold
    return {
        'passed': within_range,
        'value': value,
        'min_threshold': min_threshold,
        'max_threshold': max_threshold,
        'within_range': within_range
    }
"""
        # Test case where validation fails
        check = CustomFunctionCheck(
            validation_function=func_string,
            function_args={'value': 150, 'min_threshold': 0, 'max_threshold': 100},
        )
        result = await check()
        assert result['passed'] is False
        assert result['value'] == 150
        assert result['min_threshold'] == 0
        assert result['max_threshold'] == 100
        assert result['within_range'] is False
