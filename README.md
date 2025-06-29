# flex-evals

A Python implementation of the [**Flexible Evaluation Protocol (FEP)**](https://docs.google.com/document/d/1-vRugkj7E50v1wd1ftZxqS-wJtRfIFwwF2din7e0Iuw/edit?tab=t.0) - a vendor-neutral, schema-driven standard for evaluating any system that produces complex or variable outputs, from deterministic APIs to non-deterministic LLMs and agentic workflows.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Quick Start

```python
from flex_evals import evaluate, TestCase, Output, Check

# Define your test cases
test_cases = [
    TestCase(
        id="test_001",
        input="What is the capital of France?",
        expected="Paris"
    )
]

# System outputs to evaluate
outputs = [
    Output(value="The capital of France is Paris.")
]

# Define evaluation criteria
checks = [
    Check(
        type="exact_match",
        arguments={
            "actual": "$.output.value",  # JSONPath expression
            "expected": "$.test_case.expected",
            "case_sensitive": False
        }
    )
]

# Run evaluation
results = evaluate(test_cases, outputs, checks)
print(f"Evaluation completed: {results.status}")
print(f"Passed: {results.results[0].check_results[0].results['passed']}")
```

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
    id="unique_identifier",
    input="System input data",
    expected="Expected output",  # Optional
    metadata={"category": "reasoning"}  # Optional
)
```

### **Outputs**
Represent the actual system responses being evaluated:

```python
output = Output(
    value="System generated response",
    metadata={"model": "gpt-4", "tokens": 150}  # Optional
)
```

### **Checks**
Define evaluation criteria with flexible argument resolution:

```python
check = Check(
    type="exact_match",
    arguments={
        "actual": "$.output.value",  # JSONPath to extract data
        "expected": "Paris",         # Literal value
        "case_sensitive": False
    },
    weight=1.0  # Optional weighting
)
```

## Usage Examples

### **Simple Text Comparison**

```python
from flex_evals import evaluate, TestCase, Output, Check

# Geography quiz evaluation
test_cases = [TestCase(id="q1", input="Capital of France?", expected="Paris")]
outputs = [Output(value="Paris")]
checks = [Check(type="exact_match", arguments={"actual": "$.output.value", "expected": "$.test_case.expected"})]

results = evaluate(test_cases, outputs, checks)
```

### **Multi-Criteria Evaluation**

```python
# Evaluate both correctness and format
checks = [
    # Check if answer is correct
    Check(
        type="contains",
        arguments={
            "text": "$.output.value",
            "phrases": ["Paris"],
            "case_sensitive": False
        }
    ),
    # Check if response is properly formatted
    Check(
        type="regex",
        arguments={
            "text": "$.output.value",
            "pattern": r"The capital of .+ is .+\.",
            "flags": {"case_insensitive": True}
        }
    )
]

results = evaluate(test_cases, outputs, checks)
```

### **Per-Test-Case Checks**

```python
# Different evaluation criteria for each test case
test_cases = [
    TestCase(
        id="math_problem",
        input="What is 2+2?",
        checks=[
            Check(type="exact_match", arguments={"actual": "$.output.value", "expected": "4"})
        ]
    ),
    TestCase(
        id="creative_writing",
        input="Write a haiku about code",
        checks=[
            Check(type="regex", arguments={"text": "$.output.value", "pattern": r"(.+\n){2}.+"})
        ]
    )
]

outputs = [Output(value="4"), Output(value="Code flows like stream\nBugs dance in morning sunlight\nCommit, push, deploy")]

# No global checks needed - using per-test-case checks
results = evaluate(test_cases, outputs, checks=None)
```

### **Complex Data Structures**

```python
# Evaluate structured outputs
test_case = TestCase(
    id="api_test",
    input={"endpoint": "/users", "method": "GET"},
    expected={"status": 200, "count": 5}
)

output = Output(
    value={"status": 200, "data": {"users": [...]}, "count": 5},
    metadata={"response_time": 245}
)

checks = [
    Check(
        type="exact_match",
        arguments={
            "actual": "$.output.value.status",
            "expected": "$.test_case.expected.status"
        }
    ),
    Check(
        type="threshold",
        arguments={
            "value": "$.output.metadata.response_time",
            "max_value": 500
        }
    )
]
```

### **LLM Evaluation with Semantic Similarity**

```python
import openai

async def get_embedding(text):
    """User-provided embedding function"""
    response = await openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Semantic similarity check
checks = [
    Check(
        type="semantic_similarity",
        arguments={
            "text": "$.output.value",
            "reference": "$.test_case.expected",
            "embedding_function": get_embedding,
            "threshold": {"min_value": 0.8}
        }
    )
]

results = evaluate(test_cases, outputs, checks)  # Automatically runs async
```

## Available Checks

### **Standard Checks**

#### **`exact_match`**
Compare two values for exact equality:
```python
Check(type="exact_match", arguments={
    "actual": "$.output.value",
    "expected": "Paris",
    "case_sensitive": True,  # Default
    "negate": False          # Default
})
```

#### **`contains`**  
Check if text contains all specified phrases:
```python
Check(type="contains", arguments={
    "text": "$.output.value",
    "phrases": ["Paris", "France"],
    "case_sensitive": True,  # Default
    "negate": False          # Pass if ALL phrases found
})
```

#### **`regex`**
Test text against regular expression patterns:
```python
Check(type="regex", arguments={
    "text": "$.output.value",
    "pattern": r"^[A-Z][a-z]+$",
    "flags": {
        "case_insensitive": False,
        "multiline": False,
        "dot_all": False
    },
    "negate": False
})
```

#### **`threshold`**
Validate numeric values against bounds:
```python
Check(type="threshold", arguments={
    "value": "$.output.confidence",
    "min_value": 0.8,
    "max_value": 1.0,
    "min_inclusive": True,   # Default
    "max_inclusive": True,   # Default
    "negate": False
})
```

### **Extended Checks (Async)**

#### **`semantic_similarity`**
Measure semantic similarity using embeddings:
```python
Check(type="semantic_similarity", arguments={
    "text": "$.output.value",
    "reference": "$.test_case.expected",
    "embedding_function": your_async_embedding_function,
    "threshold": {"min_value": 0.8},
    "similarity_metric": "cosine"  # Default
})
```

#### **`llm_judge`**
Use LLM for qualitative evaluation:
```python
Check(type="llm_judge", arguments={
    "prompt": "Rate this response for helpfulness: {{$.output.value}}",
    "response_format": {
        "type": "object",
        "properties": {
            "score": {"type": "number"},
            "reasoning": {"type": "string"}
        }
    },
    "llm_function": your_async_llm_function
})
```

## 🔍 JSONPath Support

Access data anywhere in the evaluation context using JSONPath expressions:

```python
# Evaluation context structure:
{
    "test_case": {
        "id": "test_001",
        "input": "What is the capital of France?",
        "expected": "Paris",
        "metadata": {"category": "geography"}
    },
    "output": {
        "value": "The capital of France is Paris",
        "metadata": {"model": "gpt-4", "tokens": 25}
    }
}

# JSONPath examples:
"$.test_case.input"              # "What is the capital of France?"
"$.test_case.expected"           # "Paris"
"$.output.value"                 # "The capital of France is Paris"
"$.output.metadata.model"        # "gpt-4"
"$.test_case.metadata.category"  # "geography"
```

### **Literal vs JSONPath**
- Strings starting with `$.` are JSONPath expressions
- Use `\\$.` to escape literal strings that start with `$.`
- All other values are treated as literals

## ⚡ Async Evaluation

flex-evals automatically detects and optimizes async checks:

```python
# Mix of sync and async checks
checks = [
    Check(type="exact_match", arguments={"actual": "$.output.value", "expected": "Paris"}),  # Sync
    Check(type="semantic_similarity", arguments={...})  # Async
]

# Engine automatically:
# 1. Detects async checks  
# 2. Runs everything in async context
# 3. Maintains proper result ordering
results = evaluate(test_cases, outputs, checks)
```

### **Custom Async Checks**

```python
from flex_evals.checks.base import BaseAsyncCheck
from flex_evals.registry import register

@register("custom_async_check", version="1.0.0")
class CustomAsyncCheck(BaseAsyncCheck):
    async def __call__(self, text: str, api_endpoint: str) -> dict:
        # Your async implementation
        async with httpx.AsyncClient() as client:
            response = await client.post(api_endpoint, json={"text": text})
            return {"score": response.json()["score"]}
```

## Architecture

### **Core Components**

```
src/flex_evals/
├── schemas/          # Pydantic models for FEP protocol
├── engine.py         # Main evaluate() function
├── checks/
│   ├── base.py       # BaseCheck and BaseAsyncCheck
│   ├── standard/     # Built-in synchronous checks
│   └── extended/     # Async checks (LLM, API calls)
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
    evaluation_id="uuid",
    started_at="2025-01-01T00:00:00Z",
    completed_at="2025-01-01T00:00:05Z", 
    status="completed",  # completed | error | skip
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
from flex_evals.checks.base import BaseCheck
from flex_evals.registry import register

@register("my_check", version="1.0.0")
class MyCheck(BaseCheck):
    def __call__(self, text: str, pattern: str, threshold: float = 0.5) -> dict:
        # Your check logic here
        score = your_analysis(text, pattern)
        return {"score": score, "passed": score >= threshold}
```

2. **Write tests:**

```python
def test_my_check():
    check = MyCheck()
    result = check(text="test input", pattern="test")
    assert "score" in result
    assert "passed" in result
```

3. **Register and use:**

```python
# Import registers the check automatically
from my_package.my_check import MyCheck

check = Check(type="my_check", arguments={
    "text": "$.output.value",
    "pattern": "success",
    "threshold": 0.8
})
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Quick Contribution Steps**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run quality checks (`make tests`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### **Development Principles**

- **Test-driven development** - Write tests first, run frequently
- **Protocol compliance** - Maintain full FEP specification adherence
- **Clean interfaces** - Pythonic, type-safe APIs
- **Performance** - Optimize for large-scale evaluations
- **Documentation** - Clear examples and comprehensive docs

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Related Links

- [Flexible Evaluation Protocol Specification](TODO)
- [Examples Repository](examples/)

## Support

- **Issues**: [GitHub Issues](https://github.com/your-org/flex-evals/issues)
