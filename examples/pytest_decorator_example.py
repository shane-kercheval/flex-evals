"""
Simple, deterministic examples of the @evaluate pytest decorator.

This script demonstrates the core @evaluate decorator functionality with predictable,
deterministic examples that will always pass when run as tests.

To run: pytest examples/pytest_decorator_example.py -v
"""

import pytest
from pydantic import BaseModel, Field

from flex_evals import TestCase, ContainsCheck, ExactMatchCheck, LLMJudgeCheck
from flex_evals.pytest_decorator import evaluate


# Simple response format for LLM judge demo
class QualityResult(BaseModel):
    """Simple quality assessment result."""

    passed: bool = Field(description="Whether the response passed quality check")
    score: int = Field(ge=1, le=5, description="Quality score")


def simple_quality_judge(prompt: str, response_format: type) -> tuple:
    """Deterministic mock LLM that evaluates based on content length and keywords."""
    # Simple rules: good responses are longer and contain "Python"
    good_response = len(prompt) > 50 and "Python" in prompt

    if good_response:
        result = response_format(passed=True, score=4)
    else:
        result = response_format(passed=False, score=2)

    return result, {"cost": 0.01, "model": "simple-judge"}


# Example 1: Basic deterministic success
@evaluate(
    test_cases=[TestCase(id="basic", input="What is Python?")],
    checks=[ContainsCheck(
        text="$.output.value",
        phrases=["Python", "programming"],
    )],
    samples=3,
    success_threshold=1.0,  # Expect 100% success
)
def test_python_explanation(test_case: TestCase) -> str:
    """Deterministic function that always produces good Python explanations."""
    # Use test_case.input to generate more contextual response
    if "Python" in test_case.input:
        return "Python is a popular programming language known for simplicity."
    return "Python is a versatile programming language."


# Example 2: Multiple check types
@evaluate(
    test_cases=[TestCase(
        id="multi_check",
        input="code example",
        expected="example",
    )],
    checks=[
        ContainsCheck(
            text="$.output.value.code",
            phrases=["print"],
        ),
        ExactMatchCheck(
            expected="$.test_case.expected",
            actual="$.output.value.type",
        ),
    ],
    samples=2,
    success_threshold=1.0,
)
def test_code_generation(test_case: TestCase) -> dict:
    """Generate code examples with multiple validation checks."""
    # Use test_case.input to generate contextual code
    if "example" in test_case.input:
        return {
            "code": "print('Hello, World!')",
            "type": "example",
            "language": "python",
        }
    return {
        "code": "print('Generated code')",
        "type": "example",
        "language": "python",
    }


# Example 3: TestCase-defined checks
test_case_with_checks = TestCase(
    id="self_contained",
    input="demo",
    checks=[ContainsCheck(
        text="$.output.value",
        phrases=["success"],
    )],
)

@evaluate(
    test_cases=[test_case_with_checks],
    checks=None,  # Use TestCase's own checks
    samples=2,
    success_threshold=1.0,
)
def test_with_testcase_checks(test_case: TestCase) -> str:
    """Demonstrate TestCase-defined checks pattern."""
    # Use test_case.input for more dynamic response
    return f"Operation for '{test_case.input}' completed successfully"


# Example 4: LLM Judge integration
@evaluate(
    test_cases=[TestCase(id="quality", input="AI explanation")],
    checks=[LLMJudgeCheck(
        prompt="Evaluate this explanation: {{$.output.value}}",
        response_format=QualityResult,
        llm_function=simple_quality_judge,
    )],
    samples=2,
    success_threshold=1.0,
)
def test_llm_quality_assessment(test_case: TestCase) -> str:
    """Demonstrate LLM judge evaluation with deterministic mock."""
    # Use test_case.input to generate contextual response
    if "AI" in test_case.input:
        return (
            "Python is an excellent programming language for AI development, "
            "offering clear syntax and powerful libraries like TensorFlow and PyTorch."
        )
    return (
        "Python is an excellent programming language for beginners and experts alike, "
        "offering clear syntax and powerful libraries."
    )


# Example 5: Statistical threshold demonstration
@evaluate(
    test_cases=[TestCase(id="variable", input="test")],
    checks=[ExactMatchCheck(
        expected="pass",
        actual="$.output.value",
    )],
    samples=4,
    success_threshold=0.75,  # 75% threshold (3 out of 4 must pass)
)
def test_statistical_threshold(test_case: TestCase) -> str:  # noqa: ARG001
    """Demonstrate statistical threshold with predictable variance."""
    # Simple counter-based approach for deterministic behavior
    if hasattr(test_statistical_threshold, '_counter'):
        test_statistical_threshold._counter += 1
    else:
        test_statistical_threshold._counter = 1

    # Return "pass" for first 3 calls, "fail" for 4th call (75% success)
    return "pass" if test_statistical_threshold._counter <= 3 else "fail"


# Simple fixtures for demonstration
@pytest.fixture
def simple_fixture() -> str:
    """Simple string fixture for basic testing."""
    return "fixture_value"


@pytest.fixture
def user_data_fixture() -> str:
    """Another simple string fixture."""
    return "user_data_123"


# Example 7: Pytest fixture integration
@evaluate(
    test_cases=[TestCase(id="fixture_test", input="user_data")],
    checks=[ExactMatchCheck(
        expected="fixture_value:user_data",
        actual="$.output.value",
    )],
    samples=2,
    success_threshold=1.0,
)
def test_with_simple_fixture(test_case: TestCase, simple_fixture) -> str:  # noqa: ANN001
    """Demonstrate basic fixture integration with simple string values."""
    return f"{simple_fixture}:{test_case.input}"


# Example 8: Multiple fixtures
@evaluate(
    test_cases=[TestCase(id="multi_fixture", input="combined")],
    checks=[ExactMatchCheck(
        expected="fixture_value+user_data_123+combined",
        actual="$.output.value",
    )],
    samples=2,
    success_threshold=1.0,
)
def test_with_multiple_fixtures(test_case: TestCase, simple_fixture, user_data_fixture) -> str:  # noqa: ANN001
    """Demonstrate multiple fixture integration with simple string concatenation."""
    return f"{simple_fixture}+{user_data_fixture}+{test_case.input}"


if __name__ == "__main__":
    print("Simple @evaluate decorator examples")
    print("Run with: pytest examples/pytest_decorator_example.py -v")
    print()
    print("Examples included:")
    print("  test_python_explanation: Basic contains check")
    print("  test_code_generation: Multiple check types")
    print("  test_with_testcase_checks: TestCase-defined checks")
    print("  test_llm_quality_assessment: LLM judge integration")
    print("  test_statistical_threshold: Statistical evaluation")
    print("  test_cycling_test_cases: Multiple test cases")
    print("  test_with_simple_fixture: Pytest fixture integration")
    print("  test_with_multiple_fixtures: Multiple fixtures integration")
