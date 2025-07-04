"""
Simple, deterministic examples of the @evaluate pytest decorator.

This script demonstrates the core @evaluate decorator functionality with predictable,
deterministic examples that will always pass when run as tests.

To run: pytest examples/pytest_decorator_example.py -v
"""

from pydantic import BaseModel, Field

from flex_evals import TestCase, Check, CheckType
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
    checks=[Check(
        type=CheckType.CONTAINS,
        arguments={"text": "$.output.value", "phrases": ["Python", "programming"]},
    )],
    samples=3,
    success_threshold=1.0,  # Expect 100% success
)
def test_python_explanation() -> str:
    """Deterministic function that always produces good Python explanations."""
    return "Python is a popular programming language known for simplicity."


# Example 2: Multiple check types
@evaluate(
    test_cases=[TestCase(
        id="multi_check",
        input="code example",
        expected="example",
    )],
    checks=[
        Check(
            type=CheckType.CONTAINS,
            arguments={"text": "$.output.value.code", "phrases": ["print"]},
        ),
        Check(
            type=CheckType.EXACT_MATCH,
            arguments={"expected": "$.test_case.expected", "actual": "$.output.value.type"},
        ),
    ],
    samples=2,
    success_threshold=1.0,
)
def test_code_generation() -> dict:
    """Generate code examples with multiple validation checks."""
    return {
        "code": "print('Hello, World!')",
        "type": "example",
        "language": "python",
    }


# Example 3: TestCase-defined checks
test_case_with_checks = TestCase(
    id="self_contained",
    input="demo",
    checks=[Check(
        type=CheckType.CONTAINS,
        arguments={"text": "$.output.value", "phrases": ["success"]},
    )],
)

@evaluate(
    test_cases=[test_case_with_checks],
    checks=None,  # Use TestCase's own checks
    samples=2,
    success_threshold=1.0,
)
def test_with_testcase_checks() -> str:
    """Demonstrate TestCase-defined checks pattern."""
    return "Operation completed successfully"


# Example 4: LLM Judge integration
@evaluate(
    test_cases=[TestCase(id="quality", input="AI explanation")],
    checks=[Check(
        type=CheckType.LLM_JUDGE,
        arguments={
            "prompt": "Evaluate this explanation: {{$.output.value}}",
            "response_format": QualityResult,
            "llm_function": simple_quality_judge,
        },
    )],
    samples=2,
    success_threshold=1.0,
)
def test_llm_quality_assessment() -> str:
    """Demonstrate LLM judge evaluation with deterministic mock."""
    # Long response with "Python" keyword will pass the mock judge
    return "Python is an excellent programming language for beginners and experts alike, offering clear syntax and powerful libraries."  # noqa: E501


# Example 5: Statistical threshold demonstration
@evaluate(
    test_cases=[TestCase(id="variable", input="test")],
    checks=[Check(
        type=CheckType.EXACT_MATCH,
        arguments={"expected": "pass", "actual": "$.output.value"},
    )],
    samples=4,
    success_threshold=0.75,  # 75% threshold (3 out of 4 must pass)
)
def test_statistical_threshold() -> str:
    """Demonstrate statistical threshold with predictable variance."""
    # Simple counter-based approach for deterministic behavior
    if hasattr(test_statistical_threshold, '_counter'):
        test_statistical_threshold._counter += 1
    else:
        test_statistical_threshold._counter = 1

    # Return "pass" for first 3 calls, "fail" for 4th call (75% success)
    return "pass" if test_statistical_threshold._counter <= 3 else "fail"


# Example 6: Multiple test cases cycling
@evaluate(
    test_cases=[
        TestCase(id="math", input="2+2", expected="4"),
        TestCase(id="string", input="hello", expected="HELLO"),
    ],
    checks=[Check(
        type=CheckType.EXACT_MATCH,
        arguments={"expected": "$.test_case.expected", "actual": "$.output.value"},
    )],
    samples=4,  # Will cycle: math, string, math, string
    success_threshold=1.0,
)
def test_cycling_test_cases() -> str:
    """Demonstrate cycling through multiple test cases."""
    # Counter for deterministic cycling behavior
    if hasattr(test_cycling_test_cases, '_counter'):
        test_cycling_test_cases._counter += 1
    else:
        test_cycling_test_cases._counter = 1

    # Return appropriate response based on test case cycling
    if test_cycling_test_cases._counter % 2 == 1:
        return "4"  # For math test case (2+2 = 4)
    return "HELLO"  # For string test case (hello -> HELLO)


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
