# flex-evals Playbook

This playbook provides comprehensive guidance for AI coding agents and human developers working with the `flex-evals` codebase - a Python implementation of the Flexible Evaluation Protocol (FEP).

## Project Overview

**flex-evals** is a Python implementation of the Flexible Evaluation Protocol (FEP) - a vendor-neutral, schema-driven standard for evaluating any system that produces complex or variable outputs, from deterministic APIs to non-deterministic LLMs and agentic workflows.

**Key Technologies:**
- **Python 3.11+** - Core language with modern type hints and async support
- **Pydantic** - Data validation and schema definition
- **pytest + pytest-asyncio** - Testing framework with async support
- **ruff** - Fast Python linter and formatter
- **uv** - Modern Python package manager
- **GitHub Actions** - CI/CD pipeline

**Architecture Patterns:**
- **Registry Pattern** - Pluggable check implementations with decorator-based registration
- **Abstract Base Classes** - `BaseCheck` and `BaseAsyncCheck` for extensible evaluation checks
- **Async/Sync Auto-Detection** - Engine automatically optimizes execution based on check types
- **JSONPath Integration** - Dynamic data extraction from test cases and outputs
- **Dataclass + Pydantic Schemas** - Type-safe data models following FEP specification

## Project Structure

```
flex-evals/
├── .github/workflows/       # GitHub Actions CI/CD
│   └── tests.yaml          # Test workflow (Python 3.11, 3.12, 3.13)
├── examples/               # Jupyter notebooks and sample data
│   ├── quickstart.ipynb    # Introduction to FEP concepts
│   ├── llm-as-a-judge.ipynb
│   ├── example_yaml_test_cases.ipynb
│   ├── test_cases.yaml     # Sample YAML test cases
│   └── *.json, *.csv       # Sample evaluation data
├── src/flex_evals/         # Main package source
│   ├── __init__.py         # Public API exports
│   ├── engine.py           # Core evaluate() function and orchestration
│   ├── constants.py        # Enums (CheckType, Status, ErrorType, etc.)
│   ├── exceptions.py       # Custom exception hierarchy
│   ├── registry.py         # Check registration and discovery system
│   ├── jsonpath_resolver.py # JSONPath expression handling
│   ├── checks/             # Check implementations
│   │   ├── base.py         # BaseCheck and BaseAsyncCheck abstract classes
│   │   ├── standard/       # Built-in synchronous checks
│   │   │   ├── exact_match.py  # String equality comparison
│   │   │   ├── contains.py     # Substring/phrase detection
│   │   │   ├── regex.py        # Pattern matching
│   │   │   └── threshold.py    # Numeric bounds checking
│   │   └── extended/       # Async checks (LLM, API calls)
│   │       ├── llm_judge.py    # LLM-based qualitative evaluation
│   │       ├── semantic_similarity.py
│   │       └── custom_function.py
│   └── schemas/            # Pydantic data models (FEP protocol)
│       ├── check.py        # Check, CheckResult, CheckError
│       ├── test_case.py    # TestCase schema
│       ├── output.py       # Output schema
│       ├── results.py      # EvaluationRunResult, TestCaseResult
│       └── experiments.py  # ExperimentMetadata
├── tests/                  # Comprehensive test suite
│   ├── conftest.py         # Test configuration and fixtures
│   ├── test_evaluation_engine.py  # Core engine tests
│   ├── test_*_checks.py    # Individual check implementations
│   └── test_schemas/       # Schema validation tests
├── pyproject.toml          # Project metadata and dependencies (uv/hatch)
├── .ruff.toml             # Linting and formatting configuration
├── Makefile               # Development commands
├── README.md              # Comprehensive project documentation
└── uv.lock                # Locked dependency versions
```

**File Organization Patterns:**
- **Schemas in `schemas/`** - All Pydantic models follow FEP specification
- **Checks in `checks/standard/` and `checks/extended/`** - Sync vs async separation
- **Tests mirror source structure** - `test_X.py` for each `X.py` source file
- **Examples as Jupyter notebooks** - Interactive documentation and tutorials

**Configuration Files:**
- **`pyproject.toml`** - Project metadata, dependencies, build system (hatchling), pytest config
- **`.ruff.toml`** - Comprehensive linting rules with project-specific ignores
- **`Makefile`** - Development workflow commands
- **`uv.lock`** - Deterministic dependency resolution

## Getting Started

**Prerequisites:**
- Python 3.11 or higher
- `uv` package manager (recommended) or `pip`

**Installation & Setup:**
```bash
# Clone and navigate to project
cd ~/repos/flex-evals

# Install dependencies (creates .venv automatically)
uv install

# Install with development dependencies
uv install --dev

# Verify installation
uv run python -c "from flex_evals import evaluate; print('✓ Installation successful')"
```

**Quick Verification:**
```bash
# Run all quality checks
make tests

# Or run components individually
make linting       # ruff check and format
make unittests     # pytest with coverage
```

## Development Workflow

**Starting Development:**
```bash
# Activate environment (if not using `uv run`)
source .venv/bin/activate

# Run project in development mode
uv run python -m pytest tests/test_evaluation_engine.py -v

# Example usage
uv run python examples/quickstart.py  # If script version exists
```

**Common Development Commands:**
```bash
# Package management
uv add <package>           # Add runtime dependency
uv add --dev <package>     # Add development dependency
uv remove <package>        # Remove dependency
uv sync                    # Sync environment with lock file

# Code quality
uv run ruff check src/     # Check linting (auto-fix enabled)
uv run ruff check tests/   # Lint tests
uv run ruff format src/    # Format code

# Testing
uv run pytest                    # Run all tests
uv run pytest tests/test_engine.py  # Run specific test file
uv run pytest -k "test_async"   # Run tests matching pattern
uv run coverage run -m pytest   # Run with coverage
uv run coverage html            # Generate HTML coverage report
```

**Environment Variables:**
- Set in `.env` file (gitignored) for local development
- No required environment variables for basic functionality
- API keys needed only for extended checks (LLM judge, semantic similarity)

## Code Standards & Guidelines

**Follow Existing Patterns:** The codebase has established conventions that should be maintained rather than introducing new styles.

### Python Code Style

**Type Hints (Required):**
```python
# All functions must have comprehensive type hints
def evaluate(
    test_cases: list[TestCase],
    outputs: list[Output], 
    checks: list[Check] | list[list[Check]] | None = None,
    experiment_metadata: ExperimentMetadata | None = None,
) -> EvaluationRunResult:
```

**Docstring Style (Google/NumPy hybrid):**
```python
def execute_check(self, check: Check) -> CheckResult:
    """
    Execute a single check and return the result.
    
    This method handles argument resolution, error handling, and result formatting
    according to the FEP protocol.
    
    Args:
        check: The check to execute with type and arguments
        
    Returns:
        Complete CheckResult with status, results, and metadata
        
    Raises:
        ValidationError: If check arguments are invalid
        CheckExecutionError: If check execution fails
    """
```

**Import Organization:**
```python
# 1. Standard library imports
import asyncio
import uuid
from datetime import datetime, UTC
from typing import Any

# 2. Third-party imports  
from pydantic import BaseModel, Field

# 3. Local imports (relative)
from ..schemas import TestCase, Output, Check
from ..exceptions import ValidationError
from .base import BaseCheck
```

**Error Handling Pattern:**
```python
# Use custom exceptions with descriptive messages
try:
    result = some_operation()
except SomeSpecificError as e:
    raise ValidationError(f"Specific context: {e}") from e
except Exception as e:
    raise CheckExecutionError(f"Unexpected error: {e}") from e
```

### File and Class Naming

**File Names:** Snake_case matching module functionality
- `evaluation_engine.py` not `evaluationEngine.py`
- `test_evaluation_engine.py` for corresponding tests

**Class Names:** PascalCase with descriptive suffixes
- `BaseCheck`, `ExactMatchCheck`, `LlmJudgeCheck`
- `TestCase`, `CheckResult`, `EvaluationRunResult`

**Function Names:** Snake_case with verb-noun pattern
- `evaluate()`, `execute_check()`, `resolve_arguments()`
- Private methods: `_validate_inputs()`, `_compute_summary()`

**Variable Names:** Snake_case, descriptive
- `test_cases`, `check_results`, `resolved_arguments`
- `max_async_concurrent`, `evaluation_id`

### Registry Pattern Usage

**Registering New Checks:**
```python
from ..registry import register
from ..constants import CheckType
from .base import BaseCheck

@register(CheckType.MY_CHECK, version="1.0.0")
class MyCheck(BaseCheck):
    def __call__(self, text: str, pattern: str) -> dict[str, Any]:
        # Implementation
        return {"passed": True, "details": "..."}
```

**Constants Definition:**
```python
# Add to constants.py for new check types
class CheckType(str, Enum):
    EXACT_MATCH = "exact_match"
    MY_NEW_CHECK = "my_new_check"
```

## Testing

### Test Organization

**Test File Structure:**
```python
# tests/test_my_module.py
"""Tests for my_module functionality."""

import pytest
from flex_evals.my_module import MyClass
from flex_evals.exceptions import ValidationError

class TestMyClass:
    """Test MyClass functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.instance = MyClass()
    
    def test_normal_case(self):
        """Test normal operation."""
        result = self.instance.do_something("input")
        assert result == "expected"
    
    def test_error_case(self):
        """Test error handling."""
        with pytest.raises(ValidationError, match="specific error"):
            self.instance.do_something("invalid")
```

**Async Test Pattern:**
```python
@pytest.mark.asyncio
async def test_async_functionality():
    """Test async operations."""
    check = LlmJudgeCheck()
    result = await check(prompt="test", llm_function=mock_llm)
    assert result["passed"] is True
```

### Running Tests

**Local Testing:**
```bash
# Full test suite
make unittests

# Specific patterns
uv run pytest tests/test_engine.py::TestEvaluationEngine::test_async_detection -v
uv run pytest -k "async" --tb=short
uv run pytest tests/ --durations=10  # Show slowest tests

# Coverage analysis
uv run coverage run -m pytest
uv run coverage report                # Terminal report
uv run coverage html                 # Generate htmlcov/ directory
open htmlcov/index.html             # View coverage in browser
```

**Test Categories:**
- **Unit tests** - Individual function/class testing
- **Integration tests** - Component interaction testing  
- **Async tests** - Concurrency and performance testing
- **Error handling tests** - Exception scenarios

**Test Fixtures:**
```python
# Common fixtures in conftest.py
@pytest.fixture
def sample_test_cases():
    return [
        TestCase(id="test_1", input="input_1", expected="output_1"),
        TestCase(id="test_2", input="input_2", expected="output_2"),
    ]

@pytest.fixture
def mock_llm_function():
    async def mock_llm(prompt: str, response_format: type) -> tuple[Any, dict]:
        return response_format(score=5), {"cost_usd": 0.01}
    return mock_llm
```

### Writing New Tests

**For New Checks:**
```python
def test_new_check_success():
    """Test successful check execution."""
    check = NewCheck()
    result = check(arg1="value1", arg2="value2")
    
    assert "passed" in result
    assert result["passed"] is True
    assert "additional_info" in result

def test_new_check_failure():
    """Test check failure conditions."""
    check = NewCheck()
    
    with pytest.raises(ValidationError):
        check(invalid_arg="bad_value")
```

**For Schema Changes:**
```python
def test_schema_validation():
    """Test Pydantic schema validation."""
    # Valid case
    valid_data = {"id": "test", "input": "test input"}
    test_case = TestCase(**valid_data)
    assert test_case.id == "test"
    
    # Invalid case
    with pytest.raises(ValidationError):
        TestCase(id="", input=None)  # Invalid values
```

## Code Quality & Linting

### Ruff Configuration

**Enabled Rules:** (see `.ruff.toml`)
- **E, W** - pycodestyle errors and warnings
- **F** - Pyflakes  
- **N** - pep8-naming
- **D** - pydocstyle (docstring conventions)
- **ANN** - flake8-annotations (type hints)
- **UP** - pyupgrade (modern Python syntax)
- **PT** - flake8-pytest-style
- **RET, SIM** - Code simplification rules
- **PL** - Pylint rules

**Project-Specific Ignores:**
- `D203, D212` - Conflicting docstring rules  
- `PLR0913` - Too many arguments (common in evaluation functions)
- `ANN204, ANN206` - Return type annotations for special methods

**Test-Specific Relaxed Rules:**
- No return type annotations required in tests
- Relaxed naming conventions for test methods
- Allow longer functions in tests

### Code Quality Commands

```bash
# Automatic fixing (safe transformations)
make linting

# Manual review required
uv run ruff check src/ --no-fix        # Show issues without fixing
uv run ruff check src/ --show-fixes    # Preview what would be fixed

# Format code
uv run ruff format src/ tests/ examples/
```

**Pre-commit Workflow:**
```bash
# Before committing
make tests           # Run all quality checks
git add .
git commit -m "feat: add new functionality"
```

### Quality Guidelines

**Code Complexity:**
- Keep functions under 20 lines when possible
- Extract complex logic into private methods
- Use descriptive variable names over comments

**Documentation:**
- Every public function needs a docstring
- Include examples in docstrings for complex APIs
- Update README when adding new features

**Error Messages:**
- Include context in error messages
- Use specific exception types
- Preserve original exceptions with `from e`

## Common Tasks & Commands

### Adding New Check Types

**1. Define the Check Class:**
```python
# src/flex_evals/checks/standard/my_check.py
from typing import Any
from ..base import BaseCheck
from ...registry import register
from ...constants import CheckType

@register(CheckType.MY_CHECK, version="1.0.0")
class MyCheck(BaseCheck):
    """
    Check description and purpose.
    
    Arguments Schema:
    - input_text: string | JSONPath - Text to analyze
    - threshold: float (default: 0.5) - Minimum score threshold
    
    Results Schema:
    - passed: boolean - Whether check passed
    - score: float - Computed score
    """
    
    def __call__(self, input_text: str, threshold: float = 0.5) -> dict[str, Any]:
        # Implementation
        score = compute_score(input_text)
        return {
            "passed": score >= threshold,
            "score": score,
        }
```

**2. Add Constant:**
```python
# src/flex_evals/constants.py
class CheckType(str, Enum):
    # ... existing types
    MY_CHECK = "my_check"
```

**3. Write Tests:**
```python
# tests/test_my_check.py
from flex_evals.checks.standard.my_check import MyCheck

def test_my_check():
    check = MyCheck()
    result = check(input_text="test", threshold=0.3)
    assert "passed" in result
    assert "score" in result
```

**4. Update Exports:**
```python
# src/flex_evals/__init__.py  
from .constants import CheckType  # Includes new type
```

### Adding Async Checks

**For LLM or API-based checks:**
```python
from ..base import BaseAsyncCheck

@register(CheckType.ASYNC_CHECK, version="1.0.0")
class AsyncCheck(BaseAsyncCheck):
    async def __call__(self, text: str, api_endpoint: str) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_endpoint, json={"text": text})
            return {"passed": response.json()["score"] > 0.5}
```

### Working with Schemas

**Extending Test Cases:**
```python
# Add new optional field
@dataclass
class TestCase:
    id: str
    input: str | dict[str, Any]
    expected: str | dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    checks: list[Check] | None = None
    # New field:
    tags: list[str] | None = None  # Add with default None for backwards compatibility
```

**Creating Custom Output Types:**
```python
# For structured outputs
structured_output = Output(
    value={
        "response": "Generated text",
        "confidence": 0.95,
        "metadata": {"tokens": 150}
    },
    metadata={"model": "gpt-4", "timestamp": "2023-..."}
)
```

### Package Management

**Adding Dependencies:**
```bash
# Runtime dependency
uv add requests>=2.32.0

# Development dependency  
uv add --dev pytest-mock>=3.14.0

# Optional dependency group
uv add --optional=llm openai>=1.0.0
```

**Managing Versions:**
```bash
# Update all dependencies
uv lock --upgrade

# Update specific package
uv lock --upgrade-package requests

# Show outdated packages
uv tree --show-outdated
```

### Building and Publishing

**Local Development Build:**
```bash
# Build package
make package-build

# Test built package
uv run --isolated --from dist/flex_evals-*.whl python -c "from flex_evals import evaluate"
```

**Publishing (maintainers only):**
```bash
# Set token in environment
export UV_PUBLISH_TOKEN="your-pypi-token"

# Build and publish
make package
```

## Project-Specific Guidelines

### FEP Protocol Compliance

**Schema Adherence:**
- All schemas must match FEP specification exactly
- Use Pydantic for validation and serialization
- Maintain backwards compatibility in schema changes

**Evaluation Context Structure:**
```python
# Standard context available to all checks via JSONPath
{
    'test_case': {
        'id': 'test_001',
        'input': '...',
        'expected': '...',
        'metadata': {...}
    },
    'output': {
        'value': '...',  # The actual system output
        'metadata': {...}  # Optional output metadata
    }
}
```

**JSONPath Usage Patterns:**
```python
# Common JSONPath expressions used in checks
"$.test_case.input"           # Test input
"$.test_case.expected"        # Expected output  
"$.output.value"              # Actual output
"$.output.metadata.confidence" # Nested output metadata
"$.test_case.metadata.category" # Test case metadata
```

### Async/Sync Check Guidelines

**Sync Checks (BaseCheck):**
- Use for fast, local operations
- String comparisons, regex, simple calculations
- No I/O operations or network calls

**Async Checks (BaseAsyncCheck):**
- Use for LLM API calls, external services
- Network requests, file I/O operations
- Anything that might block or take significant time

**Performance Considerations:**
- Engine automatically detects and optimizes execution
- Sync checks run directly, async checks run concurrently
- Use `max_async_concurrent` parameter to limit API rate limits

### Error Handling Philosophy

**Fail Fast:**
- Validate inputs early in functions
- Use specific exception types for different error conditions

**Graceful Degradation:**
- Individual check failures don't stop evaluation
- Comprehensive error reporting in results
- Recoverable vs non-recoverable error classification

**Error Context:**
- Include JSONPath expressions in error messages
- Preserve original error information
- Provide actionable error messages

### Documentation Standards

**API Documentation:**
- Comprehensive docstrings for all public functions
- Include examples in docstrings
- Document JSONPath patterns and expected schemas

**Example-Driven:**
- Jupyter notebooks for complex workflows
- README examples for common use cases
- Test cases serve as additional documentation

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Module not found
uv sync                    # Sync environment
source .venv/bin/activate  # Activate environment

# Import issues in tests
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Add src to path
```

**Test Failures:**
```bash
# Async test failures
uv run pytest --asyncio-mode=auto  # Ensure asyncio mode

# Coverage issues  
uv run coverage erase     # Clear previous coverage
uv run coverage run -m pytest
```

**Registry Issues:**
```bash
# Checks not found
# Ensure checks are imported to trigger registration
from flex_evals import checks  # Imports trigger @register decorators
```

**JSONPath Resolution Errors:**
```python
# Debug JSONPath expressions
from flex_evals.jsonpath_resolver import JSONPathResolver
resolver = JSONPathResolver()
context = resolver.create_evaluation_context(test_case, output)
print(json.dumps(context, indent=2))  # Inspect available paths
```

### Debug Techniques

**Evaluation Debug:**
```python
# Enable verbose error information
result = evaluate(test_cases, outputs, checks)
for test_result in result.results:
    for check_result in test_result.check_results:
        if check_result.error:
            print(f"Error in {check_result.check_type}: {check_result.error.message}")
```

**Check Development Debug:**
```python
# Test individual checks
check = MyCheck()
try:
    result = check(arg1="value1")
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

**Performance Debug:**
```python
import time

start_time = time.time()
result = evaluate(test_cases, outputs, checks, max_async_concurrent=5)
duration = time.time() - start_time
print(f"Evaluation took {duration:.3f} seconds")
```

### Log Analysis

**Test Output:**
```bash
# Verbose test output
uv run pytest -v -s tests/

# Show captured stdout  
uv run pytest --capture=no tests/

# Specific test debugging
uv run pytest tests/test_engine.py::test_specific_case -v -s
```

**Coverage Analysis:**
```bash
# Missing coverage report
uv run coverage report --show-missing

# Focus on specific modules
uv run coverage report --include="src/flex_evals/engine.py"
```

## Contributing Guidelines

### Code Review Expectations

**Before Submitting PR:**
- Run `make tests` and ensure all pass
- Add tests for new functionality
- Update documentation if needed
- Follow existing code patterns

**PR Description Should Include:**
- Clear description of changes
- Rationale for design decisions
- Test coverage information
- Breaking changes (if any)

**Review Criteria:**
- Code follows established patterns
- Comprehensive test coverage
- Clear, descriptive naming
- Proper error handling
- Documentation updates

### Documentation Requirements

**New Features:**
- Update README with usage examples
- Add or update Jupyter notebook examples
- Include docstrings with examples

**Schema Changes:**
- Document impact on FEP protocol compliance
- Update example files
- Add migration notes if needed

### Release Process

**Version Bumping:**
- Follow semantic versioning (semver)
- Update version in `pyproject.toml`
- Tag releases with `v1.2.3` format

**Testing Before Release:**
- Full test suite across Python versions (3.11, 3.12, 3.13)
- Integration testing with example notebooks
- Performance regression testing

---

## Quick Reference

**Essential Commands:**
```bash
make tests              # Run all quality checks
make linting           # Lint and format code  
make unittests         # Run pytest with coverage
uv add <package>       # Add dependency
uv run pytest tests/  # Run tests
```

**Key Files for AI Agents:**
- `src/flex_evals/engine.py` - Core evaluation logic
- `src/flex_evals/checks/base.py` - Check implementation patterns
- `src/flex_evals/schemas/` - Data models and validation
- `tests/test_evaluation_engine.py` - Example usage patterns
- `examples/quickstart.ipynb` - Interactive tutorial

**Architecture Entry Points:**
- `evaluate()` function in `engine.py` - Main API
- `BaseCheck`/`BaseAsyncCheck` - Extend for new checks
- `@register()` decorator - Add checks to registry
- JSONPath resolver - Data extraction patterns

This playbook reflects the actual codebase patterns and should be updated as the project evolves.
