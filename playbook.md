# flex-evals Project Playbook

**⚡ AI Coding Agent Guide ⚡**

This playbook serves as the primary guide for AI coding agents and human developers working with the flex-evals codebase. It documents existing patterns, conventions, and workflows to ensure consistent development.

---

## 1. Project Overview

**flex-evals** is a Python implementation of the **Flexible Evaluation Protocol (FEP)** - a vendor-neutral, schema-driven standard for evaluating any system that produces complex or variable outputs, from deterministic APIs to non-deterministic LLMs and agentic workflows.

### Key Technologies
- **Python 3.11+** with modern type hints and async/await patterns
- **Pydantic** for schema validation and data modeling
- **JSONPath** for dynamic data extraction from evaluation contexts
- **uv** for fast dependency management and build processes
- **Pytest** for comprehensive testing with async support
- **Ruff** for linting and code formatting

### Architecture Philosophy
- **Registry Pattern**: Pluggable check system using decorator-based registration
- **Async-First**: Automatic detection and optimal execution of sync/async checks
- **Schema-Driven**: Pydantic models enforce FEP protocol compliance
- **JSONPath Integration**: Dynamic argument resolution for flexible data access
- **Error Context**: Rich exception hierarchy with detailed debugging information

---

## 2. Project Structure

```
~/repos/flex-evals/
├── src/flex_evals/           # Main package source
│   ├── __init__.py          # Public API exports
│   ├── engine.py            # Core evaluate() function
│   ├── registry.py          # Check registration system
│   ├── constants.py         # String enums (CheckType, Status, etc.)
│   ├── exceptions.py        # Custom exception hierarchy
│   ├── jsonpath_resolver.py # JSONPath expression handling
│   ├── checks/              # Check implementations
│   │   ├── base.py         # BaseCheck and BaseAsyncCheck abstractions
│   │   ├── standard/       # Built-in synchronous checks
│   │   │   ├── exact_match.py
│   │   │   ├── contains.py
│   │   │   ├── regex.py
│   │   │   └── threshold.py
│   │   └── extended/       # Async checks (LLM, API calls)
│   │       ├── llm_judge.py
│   │       ├── semantic_similarity.py
│   │       └── custom_function.py
│   └── schemas/            # Pydantic models for FEP protocol
│       ├── test_case.py    # TestCase data model
│       ├── output.py       # Output data model
│       ├── check.py        # Check and CheckResult models
│       ├── results.py      # Evaluation result models
│       └── experiments.py  # Experiment metadata
├── tests/                  # Comprehensive test suite
│   ├── conftest.py        # Test configuration and fixtures
│   ├── test_*.py          # Unit tests organized by module
│   └── test_schemas/      # Schema validation tests
├── examples/              # Usage examples and demos
│   ├── quickstart.ipynb   # Getting started guide
│   ├── llm-as-a-judge.ipynb
│   ├── test_cases.yaml    # YAML configuration examples
│   └── *.json            # Sample evaluation data
├── .github/workflows/     # CI/CD configuration
├── pyproject.toml        # Project metadata and dependencies
├── Makefile             # Development commands
├── .ruff.toml          # Linting configuration
└── uv.lock            # Dependency lock file
```

### Key Files and Their Roles

**Core Entry Points:**
- `src/flex_evals/__init__.py` - Public API exports
- `src/flex_evals/engine.py` - Main `evaluate()` function
- `src/flex_evals/registry.py` - Check registration system

**Configuration Files:**
- `pyproject.toml` - Project metadata, dependencies, pytest config
- `.ruff.toml` - Linting rules and code style configuration
- `Makefile` - Development workflow commands
- `uv.lock` - Exact dependency versions for reproducible builds

**File Naming Conventions:**
- **Schema files**: `{entity}.py` (e.g., `test_case.py`, `check.py`)
- **Check implementations**: `{check_name}.py` (e.g., `exact_match.py`)
- **Test files**: `test_{module_name}.py` (e.g., `test_engine.py`)
- **Example files**: Descriptive names with extensions (`.ipynb`, `.yaml`, `.json`)

### Directory Organization Patterns

**Where code belongs:**
- **Core logic**: `src/flex_evals/` (engine, registry, utilities)
- **Data models**: `src/flex_evals/schemas/` (Pydantic models only)
- **Check implementations**: `src/flex_evals/checks/standard/` or `extended/`
- **Tests**: `tests/` with parallel structure to `src/`
- **Examples**: `examples/` for demos and tutorials
- **Documentation**: Root level files (`README.md`, this playbook)

---

## 3. Getting Started

### Prerequisites
- **Python 3.11 or higher**
- **uv** package manager (recommended) or pip

### Environment Configuration

The project uses `uv` for dependency management, which automatically handles virtual environments. No additional environment configuration is required for development.

**Environment Variables:**
- No required environment variables for core functionality
- Individual checks may require API keys (e.g., for LLM judge checks)
- Test execution uses pytest configuration from `pyproject.toml`

---

## 4. Development Workflow

### Running the Project Locally

**Core evaluation function:**
```python
from flex_evals import evaluate, TestCase, Output, Check, CheckType

# Define test cases
test_cases = [TestCase(id='test_001', input="What is 2+2?", expected="4")]

# System outputs to evaluate
outputs = [Output(value="The answer is 4")]

# Define checks
checks = [Check(type=CheckType.CONTAINS, arguments={'text': '$.output.value', 'phrases': ['4']})]

# Run evaluation
results = evaluate(test_cases, outputs, checks)
print(f"Status: {results.status}")
```

### Development Server/Environment Commands

**Main development commands:**
```bash
# Install/update dependencies
uv add <package>              # Add new dependency
uv add --dev <package>        # Add development dependency
uv sync                       # Sync dependencies with lock file

# Code quality
make linting                  # Run ruff linting with auto-fix
make unittests               # Run pytest with coverage
make tests                   # Run all quality checks

# Package management
make package-build           # Build distribution packages
make package                 # Build and publish (requires UV_PUBLISH_TOKEN)
```

### Environment Variables and Configuration

**Development configuration:**
- **Pytest**: Configured in `pyproject.toml` with async support, timeouts, and path setup
- **Ruff**: Configured in `.ruff.toml` with comprehensive rule set
- **Coverage**: HTML reports generated in `htmlcov/` directory

**No persistent configuration files** - uv manages environment automatically.

### Code Reload/Hot-reload Capabilities

**For development:**
- Use `uv run` to execute scripts with latest code changes
- Jupyter notebooks automatically reload when using `%load_ext autoreload`
- Tests run with live code - no compilation step required

---

## 5. Code Standards & Guidelines

### Language-Specific Conventions

**Type Hints (Follow existing patterns):**
```python
# ✅ Modern Python style (project standard)
def process_items(items: list[str]) -> dict[str, Any]:
    results: dict[str, int] = {}
    return results

# ❌ Avoid legacy typing imports
from typing import List, Dict  # Don't use these
```

### Naming Conventions

**Follow existing patterns throughout the codebase:**

- **Files**: `snake_case.py` (e.g., `exact_match.py`, `test_case.py`)
- **Classes**: `PascalCase` (e.g., `ExactMatchCheck`, `TestCase`)
- **Functions/Methods**: `snake_case` (e.g., `evaluate`, `resolve_arguments`)
- **Variables**: `snake_case` (e.g., `test_cases`, `check_results`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `DEFAULT_TIMEOUT`)
- **Enums**: `PascalCase` class, `SCREAMING_SNAKE_CASE` values (e.g., `CheckType.EXACT_MATCH`)


---

## 6. Testing

### Testing Framework and Configuration

**Framework:** pytest with async support, coverage reporting, and timeout handling

**Configuration location:** `pyproject.toml`


### Test File Organization and Naming

**Follow existing test organization patterns:**
```
tests/
├── conftest.py                    # Shared fixtures and test configuration
├── test_evaluation_engine.py     # Tests for engine.py
├── test_standard_checks.py       # Tests for standard checks
├── test_extended_checks.py       # Tests for async checks
├── test_registry.py              # Tests for registration system
└── test_schemas/                 # Schema validation tests
    ├── test_check.py
    ├── test_output.py
    └── test_results.py
```

**Naming conventions:**
- Test files: `test_{module_name}.py`
- Test classes: `Test{ClassName}`
- Test methods: `test_{specific_behavior}`

### How to Run Different Types of Tests

```bash
# Run all tests with coverage
make unittests

# Run specific test file
uv run pytest tests/test_evaluation_engine.py

# Run specific test method
uv run pytest tests/test_evaluation_engine.py::TestEvaluationEngine::test_evaluate_function_signature

# Run tests with output
uv run pytest -v tests/

# Run tests in parallel (if installed)
uv run pytest -n auto tests/
```

### Testing Patterns and Conventions

**Follow existing pytest test patterns:**


### Coverage Tools and Expectations

**Coverage configuration:** Uses `coverage` with HTML reporting
- **Location:** Coverage reports generated in `htmlcov/`
- **Command:** `uv run coverage html` (included in `make unittests`)
- **Target:** Maintain high coverage on new code, especially core logic

---

## 7. Code Quality & Maintenance

### Linting Tools and Configuration

**Primary tool:** Ruff (replaces flake8, black, isort, etc.)
- **Configuration:** `.ruff.toml`
- **Command:** `make linting` (includes auto-fix)

### Code Formatting Standards

**Follow existing formatting patterns:**
- **Line length:** 99 characters (configured in ruff)
- **Indentation:** 4 spaces (Python standard)
- **String quotes:** Double quotes for strings, single quotes for short literals
- **Import sorting:** Automatic via ruff (standard library, third-party, local)

### Pre-commit Hooks and CI Quality Gates

**CI Configuration:** `.github/workflows/tests.yaml`
- **Triggers:** Push to main, PRs to main (ignoring README.md changes)
- **Python versions:** 3.11, 3.12, 3.13
- **Steps:** Install dependencies, run linting, run unit tests

**Quality gates:**
1. **Linting must pass:** `make linting`
2. **Unit tests must pass:** `make unittests`
3. **Coverage reporting:** Generated automatically

### How to Run Quality Checks Locally

```bash
# Run all quality checks (linting + tests)
make tests

# Run only linting with auto-fix
make linting

# Run only unit tests with coverage
make unittests

# Manual ruff commands
uv run ruff check src/flex_evals/ --fix
uv run ruff check tests/ --fix
uv run ruff check examples/ --fix
```

### Build System and Dependency Management

**Build system:** Hatchling (configured in `pyproject.toml`)
```toml
[build-system]
requires = ["hatchling>=1.17.1"]
build-backend = "hatchling.build"
```

**Dependency management:**
```bash
# Add new dependencies
uv add jsonpath-ng>=1.6.0           # Runtime dependency
uv add --dev pytest>=8.4.0          # Development dependency

# Sync with lock file
uv sync

# Build package
make package-build                   # Creates dist/ directory
```

---

## 8. Project-Specific Guidelines

### Important Architectural Decisions

**1. Registry Pattern for Checks**
- All checks self-register using the `@register` decorator
- Enables dynamic check discovery and pluggable architecture
- Supports versioning and conflict detection

**2. Async/Sync Auto-Detection**
- Engine automatically detects async checks by introspecting the `__call__` method
- Optimizes execution by running sync checks directly, async checks concurrently
- Maintains result ordering regardless of execution pattern

**3. JSONPath Argument Resolution**
- Arguments starting with `$.` are treated as JSONPath expressions
- Provides access to full evaluation context: `$.test_case.*`, `$.output.*`
- Use `\\$.` to escape literal strings starting with `$.`

**4. Schema-Driven Development**
- All data structures are Pydantic models ensuring type safety
- Validation happens at model creation time
- Provides clear error messages for invalid data

### Performance Considerations and Optimization

**Async Concurrency Control:**
```python
# Use max_async_concurrent to prevent overwhelming external APIs
result = evaluate(
    test_cases, outputs, checks,
    max_async_concurrent=10  # Limit concurrent async operations
)
```

**Parallel Processing:**
```python
# Use max_parallel_workers for CPU-intensive workloads
result = evaluate(
    test_cases, outputs, checks,
    max_parallel_workers=4  # Process test cases in parallel
)
```

**Memory Efficiency:**
- Use generators for large datasets when possible
- Process results incrementally rather than storing all in memory
- Registry state is serialized for parallel workers to maintain check availability

### Integration Patterns Between Components

**Check Registration Pattern:**
```python
# Import automatically registers checks
import flex_evals.checks.standard  # Registers all standard checks
import flex_evals.checks.extended  # Registers all extended checks

# Custom registration
from flex_evals.registry import register
@register("custom_check", version="1.0.0")
class CustomCheck(BaseCheck): ...
```

**Error Handling Integration:**
```python
# Follow existing error propagation patterns
try:
    result = evaluate(test_cases, outputs, checks)
except ValidationError as e:
    # Handle validation errors
    logger.error(f"Validation failed: {e}")
except Exception as e:
    # Handle unexpected errors
    logger.error(f"Evaluation failed: {e}")
```

---

## 9. Command Reference

### Essential Commands

**Setup and Installation:**
```bash
uv install --dev              # Install all dependencies including dev tools
uv sync                       # Sync dependencies with lock file
uv add <package>              # Add new runtime dependency
uv add --dev <package>        # Add new development dependency
```

**Development Workflow:**
```bash
make tests                    # Run all quality checks (linting + tests)
make linting                  # Run ruff linting with auto-fix
make unittests               # Run pytest with coverage report
```

**Running Code:**
```bash
uv run python <script.py>    # Run Python script in project environment
uv run pytest <test_file>    # Run specific tests
```

### Package Management

**Adding Dependencies:**
```bash
uv add "requests>=2.32.4"               # Add runtime dependency with version constraint
uv add --dev "pytest>=8.4.0"            # Add development dependency
uv add --optional ml "scikit-learn"     # Add to optional dependency group
```

### Testing Commands

**Basic Testing:**
```bash
uv run pytest                           # Run all tests
uv run pytest tests/test_engine.py      # Run specific test file
uv run pytest -k "test_exact_match"     # Run tests matching pattern
uv run pytest -v                        # Verbose output
uv run pytest --tb=short               # Short traceback format
```

**Coverage and Reporting:**
```bash
uv run coverage run -m pytest          # Run tests with coverage (included in make unittests)
uv run coverage report                 # Show coverage report in terminal
uv run coverage html                   # Generate HTML coverage report
open htmlcov/index.html                # View coverage report (macOS)
```

**Performance Testing:**
```bash
uv run pytest --durations=10           # Show 10 slowest tests
uv run pytest --timeout=30             # Set timeout for tests
```

### Build & Quality

**Code Quality:**
```bash
uv run ruff check src/                 # Check for linting issues
uv run ruff check --fix src/           # Auto-fix linting issues
uv run ruff format src/                # Format code (included in check --fix)
```

---

## 10. Safety Guidelines

### ⚠️ NEVER DO

**Version Control Operations:**
```bash
# ❌ NEVER perform these operations
git commit -m "message"       # Don't commit changes
git push origin main          # Don't push to remote
git branch new-feature        # Don't create branches
git merge feature-branch      # Don't merge branches
git tag v1.0.0               # Don't create tags
git reset --hard HEAD~1      # Don't reset commits
```

**Destructive File Operations:**
```bash
# ❌ NEVER perform these operations
rm -rf src/                   # Don't delete source code
rm -rf .git/                  # Don't delete git history
rm -rf tests/                 # Don't delete tests
mv src/ old_src/             # Don't move critical directories
```

(Removing individual files is generally safe, but avoid removing critical directories or files.)

**System-Level Changes:**
```bash
# ❌ NEVER perform these operations
sudo pip install <package>   # Don't install globally
pip install --system <pkg>   # Don't modify system Python
chmod -R 777 .               # Don't change permissions broadly
chown -R root:root .         # Don't change ownership
```

**External Service Calls:**
```bash
# ❌ NEVER perform these operations
curl -X POST https://api.production.com/deploy    # Don't call production APIs
make package && uv publish                        # Don't publish packages (costs money)
docker push production/image                       # Don't push to production registries
```

**Data Safety:**
```bash
# ❌ NEVER perform these operations
rm uv.lock                   # Don't delete lock files
rm pyproject.toml            # Don't delete project configuration
mv .env.example .env && git add .env  # Don't commit secrets
```

### ✅ SAFE OPERATIONS

**Recommended development activities:**
- Reading and analyzing code
- Running tests (`make tests`, `uv run pytest`)
- Running linting (`make linting`)
- Creating new files in appropriate directories
- Modifying existing code following established patterns
- Adding dependencies via `uv add` (non-destructive)
- Building packages locally (`make package-build`)
- Running examples and notebooks
- Debugging and troubleshooting

**Safe file operations:**
- Creating new test files in `tests/`
- Creating new check implementations in `checks/`
- Modifying existing files to fix bugs or add features
- Creating example files in `examples/`
- Adding documentation

---

## 11. Playbook Maintenance

**📋 Important Note for AI Coding Agents:**

This playbook should be **kept up-to-date** as the project evolves. When making significant changes to the codebase:

1. **Update relevant sections** of this playbook to reflect new patterns
2. **Add new commands** to the Command Reference section
3. **Update troubleshooting** with new common issues discovered
4. **Modify code examples** to match current implementations
5. **Add new safety guidelines** if new operations become possible

---
