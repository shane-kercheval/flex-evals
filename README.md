# flex-evals

A Python implementation of the [**Flexible Evaluation Protocol (FEP)**](https://docs.google.com/document/d/1-vRugkj7E50v1wd1ftZxqS-wJtRfIFwwF2din7e0Iuw/edit?tab=t.0) - a vendor-neutral, schema-driven standard for evaluating any system that produces complex or variable outputs, from deterministic APIs to non-deterministic LLMs and agentic workflows.

[![Tests](https://github.com/shane-kercheval/flex-evals/actions/workflows/tests.yaml/badge.svg)](https://github.com/shane-kercheval/flex-evals/actions/workflows/tests.yaml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Quick Start

```python
from flex_evals import evaluate, TestCase, Output, ContainsCheck

test_cases = [
    TestCase(
        id='test_001',
        input="What is the capital of France?",
        checks=[
            ContainsCheck(
                text='$.output.value',    # JSONPath - extract from system output
                phrases=['Paris', 'France'], # Literal - exact phrases to find
            ),
        ],
    ),
]

# System outputs to evaluate
outputs = [
    Output(value="The capital of France is Paris."),
]

# Run evaluation
results = evaluate(test_cases, outputs)
print(f"Evaluation completed: {results.status}")
print(f"Passed: {results.results[0].check_results[0].results['passed']}")
```

## Pytest Integration

Use the `@evaluate` decorator to test functions with automatic evaluation:

```python
from flex_evals import TestCase, ContainsCheck
from flex_evals.pytest_decorator import evaluate

@evaluate(
    test_cases=[TestCase(input="What is Python?")],
    checks=[
        ContainsCheck(
            text="$.output.value",  # JSONPath expression
            phrases=["Python", "programming"],
        ),
    ],
    samples=10,
    success_threshold=0.8,  # Expect 80% success
)
async def test_python_explanation(test_case: TestCase) -> str:
    # This function will be called `samples * len(test_cases)` times.
    # Each test case will be evaluated against this function's output.
    # The value returned by this function will be populated into the `Output` dataclass and
    # can be referenced by the Check via JSONPath (e.g. `text="$.output.value"`)
    return my_llm(test_case.input)
```

### Fixture Limitations

When using pytest fixtures with the `@evaluate` decorator, be aware that **fixture instances are reused across all test executions** within a single decorated function run. This means:

- If your pytest fixture maintains state (counters, lists, etc.), that state will accumulate across multiple executions
- Each execution does NOT get a fresh fixture instance - the same fixture is passed to all executions
- This is standard pytest behavior when fixtures are resolved to values before being passed to the decorator

**Example of problematic fixture:**
```python
@pytest.fixture
async def stateful_fixture():
    class Counter:
        def __init__(self):
            self.count = 0
        def increment(self):
            self.count += 1
            return self.count
    return Counter()

@evaluate(test_cases=[...], samples=5)  
async def test_func(test_case, stateful_fixture):
    # This will return 1, 2, 3, 4, 5 across executions
    # NOT 1, 1, 1, 1, 1 as might be expected
    return stateful_fixture.increment()
```

For stateless fixtures or fixtures that should maintain state across executions, this behavior is expected and correct.

## Examples

**See `examples` directory for more detailed usage examples**:
- **[Pytest Decorator Examples](examples/pytest_decorator_example.py)** - Complete pytest integration examples
- **[Quickstart Notebook](examples/quickstart.ipynb)** - Introduction to FEP concepts
- **[Advanced Examples](examples/example_yaml_test_cases.ipynb)** - Using YAML for defining test cases
- **[LLM-as-a-Judge](examples/llm-as-a-judge.ipynb)** - Using YAML for defining test cases

## Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Core Concepts](#-core-concepts)
- [Usage Examples](#-usage-examples)
- [Available Checks](#-available-checks)
- [JSONPath Support](#-jsonpath-support)
- [Async Evaluation](#-async-evaluation)
- [Architecture](#-architecture)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)

## Features

### **Protocol Compliance**
- **Full FEP Implementation** - Complete implementation of the Flexible Evaluation Protocol specification
- **Structured Results** - Comprehensive result format with metadata, timestamps, and error details
- **Reproducible Evaluations** - Consistent, auditable evaluation runs

### **Flexible Data Access**
- **JSONPath Expressions** - Dynamic data extraction with `$.test_case.input`, `$.output.value`, etc.
- **Multiple Input Types** - Support for strings, objects, and complex nested data structures
- **Custom Metadata** - Attach arbitrary metadata to test cases, outputs, and evaluations

### **Built-in Checks**
- **Standard Checks** - `exact_match`, `contains`, `regex`, `threshold`
- **LLM Checks** - `semantic_similarity`, `llm_judge` (with user-provided async functions)
- **Extensible** - Easy to add custom check implementations

### **Performance & Scalability**
- **Async Support** - Automatic detection and optimal execution of sync/async checks
- **Parallel Execution** - Batch processing for large evaluation runs
- **Memory Efficient** - Streaming support for large datasets

### **Developer Experience**
- **Pythonic API** - Clean, type-safe interfaces with excellent IDE support
- **Test-Friendly** - Easy unit testing of individual checks
- **Comprehensive Documentation** - Detailed examples and API reference

## Installation

```bash
uv add flex-evals
pip install flex-evals
```

### Requirements
- Python 3.11+
- Dependencies: `jsonpath-ng`, `pydantic`, `pyyaml`, `requests`

## Core Concepts

### **Test Cases**
Define the inputs and expected outputs for evaluation:

```python
test_case = TestCase(
    id='unique_identifier',
    input="System input data",
    expected="Expected output",  # Optional
    metadata={'category': 'reasoning'}  # Optional
)
```

### **Outputs**
Represent the actual system responses being evaluated:

```python
output = Output(
    value="System generated response",
    metadata={'model': 'gpt-4', 'tokens': 150}  # Optional
)
```

### **Checks**
Define evaluation criteria with type-safe, validated classes:

```python
from flex_evals import ExactMatchCheck

check = ExactMatchCheck(
    actual='$.output.value',  # JSONPath to extract data
    expected='Paris',         # Literal value
    case_sensitive=False
)
```

## Usage Examples

### **Simple Text Comparison**

```python
from flex_evals import evaluate, TestCase, Output, ExactMatchCheck

# Geography quiz evaluation
test_cases = [TestCase(id='q1', input="Capital of France?", expected='Paris')]
outputs = [Output(value='Paris')]
checks = [
    ExactMatchCheck(
        actual='$.output.value',
        expected='$.test_case.expected',
    ),
]

results = evaluate(test_cases, outputs, checks)
```

### **Pattern 1: Shared Checks (1-to-Many)**

```python
from flex_evals import ContainsCheck, RegexCheck
import re

# Each test case shares the same checks
checks = [
    # Check if answer is correct
    ContainsCheck(
        text='$.output.value',
        phrases=['Paris'],
        case_sensitive=False
    ),
    # Check if response is properly formatted
    RegexCheck(
        text='$.output.value',
        pattern=r'The capital of .+ is .+\.',
        flags=re.IGNORECASE
    )
]

results = evaluate(test_cases, outputs, checks)
```

### **Pattern 2: Per-Test-Case Checks (1-to-1)**

```python
from flex_evals import ExactMatchCheck, RegexCheck

# Each test case has it's own checks
test_cases = [
    TestCase(
        id='math_problem',
        input="What is 2+2?",
        checks=[
            ExactMatchCheck(
                actual='$.output.value', 
                expected='4'
            )
        ]
    ),
    TestCase(
        id='creative_writing',
        input="Write a haiku about code",
        checks=[
            RegexCheck(
                text='$.output.value', 
                pattern=r'(.+\n){2}.+'
            )
        ]
    )
]

outputs = [
    Output(value="4"),
    Output(value="Code flows like stream\nBugs dance in morning sunlight\nCommit, push, deploy"),
]

# No global checks needed - using per-test-case checks
results = evaluate(test_cases, outputs, checks=None)
```

## JSONPath Support

Access data anywhere in the "evaluation context" (i.e. test case definition and output) using [JSONPath](https://www.rfc-editor.org/rfc/rfc9535.html) expressions:

```python
# Evaluation context structure:
{
    'test_case': {
        'id': 'test_001',
        'input': "What is the capital of France?",
        'expected': 'Paris',
        'metadata': {'category': 'geography'}
    },
    'output': {
        'value': "The capital of France is Paris",
        'metadata': {'model': 'gpt-4', 'tokens': 25}
    }
}

# JSONPath examples:
'$.test_case.input'              # "What is the capital of France?"
'$.test_case.expected'           # "Paris"
'$.output.value'                 # "The capital of France is Paris"
'$.output.metadata.model'        # "gpt-4"
'$.test_case.metadata.category'  # "geography"
```

### **Literal vs JSONPath**
- Strings starting with `$.` are JSONPath expressions
- Use `\\$.` to escape literal strings that start with `$.`
- All other values are treated as literals


### **JSONPath Example**

```python
# Evaluate structured outputs
test_case = TestCase(
    id='api_test',
    input={'endpoint': '/users', 'method': 'GET'},
    expected={'status': 200, 'count': 5}
)

output = Output(
    value={'status': 200, 'data': {'users': [...]}, 'count': 5},
    metadata={'response_time': 245}
)

checks = [
    ExactMatchCheck(
        # use JSONPath to access nested output value
        actual='$.output.value.status',
        # use JSONPath to access expected value
        expected='$.test_case.expected.status'
    ),
    ThresholdCheck(
        # use JSONPath to access nested metadata
        value='$.output.metadata.response_time',
        max_value=500
    )
]
```

## Available Checks

flex-evals provides type-safe check classes with IDE support, validation, and clear APIs:

### **YAML Configuration**

Checks can also be defined in YAML format for configuration-driven evaluations:

```yaml
# test_cases.yaml
checks:
  - type: exact_match
    arguments:
      actual: "$.output.value"
      expected: "Paris"
      case_sensitive: false
    version: "1.0.0"  # Optional: specify version
  - type: contains
    arguments:
      text: "$.output.value"
      phrases: ["France"]
    # version omitted - uses latest version
```

Load and use YAML-defined checks:

```python
import yaml
from flex_evals import Check

# Load checks from YAML
with open('test_cases.yaml', 'r') as f:
    config = yaml.safe_load(f)

checks = [Check(**check_config) for check_config in config['checks']]

# Use in evaluation
results = evaluate(test_cases, outputs, checks)
```

See [example_yaml_test_cases.ipynb](examples/example_yaml_test_cases.ipynb) for comprehensive YAML configuration examples.

### **Standard Checks**

#### **`ExactMatchCheck`**

Compare two values for exact equality:

```python
from flex_evals import ExactMatchCheck

ExactMatchCheck(
    actual='$.output.value',
    expected='Paris',
    case_sensitive=True,  # Default
    negate=False,         # Default
)
```

#### **`ContainsCheck`**  

Check if text contains all specified phrases:

```python
from flex_evals import ContainsCheck

ContainsCheck(
    text='$.output.value',
    phrases=['Paris', 'France'],
    case_sensitive=True,  # Default
    negate=False,         # Pass if ALL phrases found
)
```

#### **`RegexCheck`**

Test text against regular expression patterns:

```python
import re
from flex_evals import RegexCheck

RegexCheck(
    text='$.output.value',
    pattern=r'^[A-Z][a-z]+$',
    flags=re.IGNORECASE,  # Use standard re flags
    negate=False,
)
```

#### **`ThresholdCheck`**

Validate numeric values against bounds:

```python
from flex_evals import ThresholdCheck

ThresholdCheck(
    value='$.output.confidence',
    min_value=0.8,
    max_value=1.0,
    min_inclusive=True,   # Default
    max_inclusive=True,   # Default
    negate=False,
)
```

### **Extended Checks (Async)**

#### **`semantic_similarity`**

Measure semantic similarity using embeddings:

```python
TBD
```

#### **`LLMJudgeCheck`**

Use an LLM for qualitative evaluation:

```python
from flex_evals import LLMJudgeCheck
from pydantic import BaseModel, Field

class HelpfulnessScore(BaseModel):  # Pydantic model defining Judge format.
    score: int = Field(description="Rate the response on a scale of 1-5h.")
    reasoning: str = Field(description="Brief explanation of the score.")

async def llm_judge(prompt: str, response_format: type[BaseModel]):
    response = ...
    metadata = {
        'cost_usd': ...,
        'response_time_ms': ...,
        'model_name': ...,
        'model_version': ...,
    }
    return response, metadata

LLMJudgeCheck(
    prompt="Rate this response for helpfulness: {{$.output.value.response}}",
    response_format=HelpfulnessScore,
    llm_function=llm_judge,
)
```

## Async Evaluation

flex-evals automatically detects and optimizes async checks:

```python
# Mix of sync and async checks
checks = [
    # Sync __call__
    ExactMatchCheck(
        actual='$.output.value',
        expected='Paris',
    ),
    # Async __call__
    LLMJudgeCheck(
        prompt="{{$.output.value}}",
        response_format=MyFormat,
        llm_function=judge_func,
    )
]

# Engine automatically:
# 1. Detects async checks  
# 2. Runs all async checks in event loop
# 3. Maintains proper result ordering after execution
results = evaluate(test_cases, outputs, checks)
```

### **Custom Async Checks**

```python
from flex_evals.checks.base import BaseAsyncCheck
from flex_evals.registry import register

@register('custom_async_check', version='1.0.0')
class CustomAsyncCheck(BaseAsyncCheck):
    async def __call__(self, text: str, api_endpoint: str) -> dict:
        # Your async implementation
        async with httpx.AsyncClient() as client:
            response = await client.post(api_endpoint, json={'text': text})
            return {'score': response.json()['score']}
```

## Check Versioning

flex-evals supports versioned checks to maintain backward compatibility while allowing evolution of check implementations and field definitions.

### **Check Architecture**

Each check type is implemented as a single class that combines validation and execution:
- **Check Class**: e.g., `ContainsCheck`, `ExactMatchCheck`
  - Inherits from `BaseCheck` or `BaseAsyncCheck`
  - Defines Pydantic fields with JSONPath support (`str | JSONPath`)
  - Contains both validation logic and execution logic in one class
  - Registered with the system using `@register(check_type, version="1.0.0")`

### **Using Versioned Checks**

```python
# Option 1: Use check class directly (recommended)
from flex_evals import ContainsCheck
check = ContainsCheck(
    text="$.output.value",     # JSONPath expression
    phrases=["hello", "world"] # Literal value
)

# Option 2: Use Check dataclass with version (YAML-compatible)
from flex_evals import Check
check = Check(
    type="contains",
    arguments={
        "text": "$.output.value", 
        "phrases": ["hello", "world"]
    },
    version="1.0.0"  # Use specific version
)

# Option 3: Use Check dataclass without version (uses latest)
check = Check(
    type="contains", 
    arguments={
        "text": "$.output.value", 
        "phrases": ["hello", "world"]
    }
    # version defaults to latest
)
```

### **JSONPath Integration**

All check fields support both literal values and JSONPath expressions:

```python
# Mixed literal and JSONPath values
check = ExactMatchCheck(
    actual="$.output.value",           # JSONPath - extract from output
    expected="$.test_case.expected",   # JSONPath - extract from test case
    case_sensitive=False               # Literal - use directly
)

# All literal values
check = ExactMatchCheck(
    actual="Paris",      # Literal comparison
    expected="Paris",    # Literal expected
    case_sensitive=True  # Literal boolean
)

# All JSONPath expressions
check = ExactMatchCheck(
    actual="$.output.value",
    expected="$.test_case.expected", 
    case_sensitive="$.test_case.metadata.case_sensitive"
)
```

### **Creating New Check Versions**

To create a new version of an existing check:

**1. Register Updated Version**

```python
# src/flex_evals/checks/contains.py

@register(CheckType.CONTAINS, version="2.0.0")  # Updated version
class ContainsCheck(BaseCheck):
    """ContainsCheck with enhanced word boundary support."""
    
    # Existing fields
    text: str | JSONPath = Field(..., description='Text to search')
    phrases: str | list[str] | JSONPath = Field(..., description='Phrases to find')
    case_sensitive: bool | JSONPath = Field(True, description='Case sensitive search')
    # Additional functionality
    word_boundaries: bool | JSONPath = Field(False, description='Match whole words only')
    
    @field_validator('text', 'phrases', 'case_sensitive', 'word_boundaries', mode='before')
    @classmethod
    def convert_jsonpath(cls, v):
        if isinstance(v, str) and v.startswith('$.'):
            return JSONPath(expression=v)
        return v

    def __call__(self) -> dict[str, Any]:
        # Implementation with enhanced logic
        pass
```

**2. Version Accessibility**

The registry system maintains version history automatically. All versions remain accessible:

```python
# Specific version request
check_v1 = Check(
    type="contains",
    arguments={"text": "$.output", "phrases": ["hello"]},
    version="1.0.0"  # Explicitly request version 1.0.0
)

# Latest version (default behavior)
check_latest = Check(
    type="contains", 
    arguments={
        "text": "$.output", 
        "phrases": ["hello"],
        "word_boundaries": True  # Features available in latest version
    }
    # Automatically uses latest version
)
```

### **Semantic Versioning Rules**

- **Major** (1.0.0 → 2.0.0): Breaking changes (remove fields, change behavior)
- **Minor** (1.0.0 → 1.1.0): Add optional fields, new functionality  
- **Patch** (1.0.0 → 1.0.1): Bug fixes, implementation improvements

### **Version Management**

```python
from flex_evals.registry import get_latest_version, list_versions

# Get latest version of a check type
latest = get_latest_version("contains")  # "2.0.0"

# List all available versions
versions = list_versions("contains")  # ["1.0.0", "2.0.0"]
```

### **Field Validation and JSONPath**

The check architecture provides automatic JSONPath conversion:

```python
# When you pass a string starting with '$.' it becomes a JSONPath
check = ContainsCheck(
    text="$.output.value",  # Automatically converted to JSONPath object
    phrases=["hello"]       # Remains as literal list
)

# Escape literal values that start with '$.' using '\\'
check = ContainsCheck(
    text="\\$.99 price",  # Becomes literal "$.99 price"
    phrases=["dollar"]
)
```

## Architecture

### **Core Components**

```
src/flex_evals/
├── schemas/          # Pydantic models for FEP protocol  
├── engine.py         # Main evaluate() function
├── checks/
│   ├── base.py       # BaseCheck, BaseAsyncCheck, JSONPath classes
│   ├── *.py          # Check implementations (sync/async)
│   └── __init__.py   # Check exports
├── jsonpath_resolver.py  # JSONPath expression handling
├── registry.py       # Check registration and discovery
└── exceptions.py     # Custom exception hierarchy
```

### **Evaluation Flow**

1. **Validation** - Ensure inputs meet protocol requirements
2. **Check Resolution** - Map check types to implementations  
3. **Async Detection** - Determine execution strategy
4. **Execution** - Run checks with proper error handling
5. **Aggregation** - Collect results and compute summaries

### **Result Format**

```python
EvaluationRunResult(
    evaluation_id='uuid',
    started_at='2025-01-01T00:00:00Z',
    completed_at='2025-01-01T00:00:05Z', 
    status='completed',  # completed | error | skip
    summary=EvaluationSummary(
        total_test_cases=100,
        completed_test_cases=95,
        error_test_cases=3,
        skipped_test_cases=2
    ),
    results=[TestCaseResult(...), ...],
    experiment=ExperimentMetadata(...)
)
```

## Development

### **Setup**

```bash
# Clone repository
git clone https://github.com/your-org/flex-evals.git
cd flex-evals

# Install with development dependencies
uv install --dev

# Run tests
make unittests

# Run linting
make linting

# Run all quality checks
make tests
```

### **Project Commands**

```bash
# Development workflow
make linting          # Run ruff linting
make unittests        # Run pytest with coverage  
make tests            # Run all quality checks

# Package management
uv add <package>      # Add dependency
uv add --dev <tool>   # Add development dependency
uv run <command>      # Run command in environment
```

### **Adding Custom Checks**

1. **Create check implementation:**

```python
from flex_evals.checks.base import BaseCheck, JSONPath
from flex_evals.registry import register
from pydantic import Field, field_validator
from typing import Any

@register('my_check', version='1.0.0')
class MyCheck(BaseCheck):
    """Custom check with JSONPath support."""
    
    # Define fields with JSONPath support
    text: str | JSONPath = Field(..., description='Text to analyze')
    pattern: str | JSONPath = Field(..., description='Pattern to match')
    threshold: float | JSONPath = Field(0.5, description='Minimum score threshold')
    
    @field_validator('text', 'pattern', 'threshold', mode='before')
    @classmethod
    def convert_jsonpath(cls, v):
        """Convert JSONPath-like strings to JSONPath objects."""
        if isinstance(v, str) and v.startswith('$.'):
            return JSONPath(expression=v)
        return v
    
    def __call__(self) -> dict[str, Any]:
        """Execute the check using resolved field values."""
        # All JSONPath expressions are resolved before this is called
        score = your_analysis(self.text, self.pattern)
        return {
            'score': score, 
            'passed': score >= self.threshold,
            'threshold_used': self.threshold
        }
```

2. **Write tests:**

```python
import pytest
from flex_evals import evaluate, TestCase, Output

def test_my_check_direct():
    """Test check directly with literal values."""
    check = MyCheck(text='test input', pattern='test', threshold=0.7)
    result = check()
    assert 'score' in result
    assert 'passed' in result

def test_my_check_with_jsonpath():
    """Test check with JSONPath expressions via engine."""
    test_cases = [TestCase(id='test1', input='test input')]
    outputs = [Output(value='test response')]
    checks = [
        MyCheck(
            text='$.output.value',      # JSONPath expression
            pattern='$.test_case.input', # JSONPath expression  
            threshold=0.8               # Literal value
        )
    ]
    
    results = evaluate(test_cases, outputs, checks)
    assert results.results[0].check_results[0].status == 'completed'
```

3. **Register and use:**

```python
# Import registers the check automatically
from my_package.my_check import MyCheck

# Option 1: Use check class directly
check = MyCheck(
    text='$.output.value',
    pattern='success',
    threshold=0.8
)

# Option 2: Use via Check dataclass (YAML-compatible)
from flex_evals import Check
check = Check(
    type='my_check', 
    arguments={
        'text': '$.output.value',
        'pattern': 'success', 
        'threshold': 0.8
    }
)
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Quick Contribution Steps**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run linting and unit tests (`make tests`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### **Development Principles**

- **Comphensive Unit Tests** - Ensure all new features have tests
- **Consistent Style** - Follow PEP 8 and use `ruff` for linting
- **Documentation** - Clear examples and comprehensive docs

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Related Links

- [Flexible Evaluation Protocol Specification](TODO)
- [Examples Repository](examples/)

## Support

- **Issues**: [GitHub Issues](https://github.com/your-org/flex-evals/issues)
