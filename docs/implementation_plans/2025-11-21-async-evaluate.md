# Implementation Plan: Async-First Evaluate API

## Overview

**Goal**: Refactor the `evaluate()` function to be async-first, solving event loop conflicts in Jupyter notebooks and async contexts, while providing a sync convenience wrapper for simple scripts.

**Key Decision**: Make `async def evaluate()` the primary API with `def evaluate_sync()` as a convenience wrapper. This reflects the true async nature of the operations and solves recurring event loop issues.

**Why This Change**:
- Current sync API creates new event loops with `asyncio.run()`, causing conflicts in Jupyter notebooks and async contexts
- The function performs inherently async operations (LLM calls, API requests), so it should be async
- Enables composition with other async operations via `asyncio.gather()`
- Eliminates complex threading workarounds in pytest decorator
- Future-proofs the API as Python moves toward async-first patterns

**Breaking Changes**: Yes, but backwards compatibility is NOT required. Prioritize clean design.

---

## Milestone 1: Core Engine Async Refactor

### Goal
Convert the core evaluation engine to async-first, making `evaluate()` async and creating `evaluate_sync()` wrapper.

### Success Criteria
- [ ] `evaluate()` is async and uses existing event loop (no `asyncio.run()` internally)
- [ ] `evaluate_sync()` wrapper works for synchronous contexts
- [ ] All existing tests pass with updated async calls
- [ ] Both sync and async checks execute correctly
- [ ] Parallel processing still works

### Key Changes

**1. Refactor `_evaluate()` to be properly async**

Current code ([engine.py:332-366](src/flex_evals/engine.py#L332-L366)) uses `asyncio.run()` internally:

```python
# Current (problematic)
def _evaluate(work_items, max_async_concurrent=None):
    # ... sync checks ...
    if flattened_async_checks:
        async_results = asyncio.run(  # L Creates new event loop
            _execute_all_async_checks(flattened_async_checks, max_async_concurrent)
        )
```

Change to:

```python
# New (async-first)
async def _evaluate_async(work_items, max_async_concurrent=None):
    """Execute evaluation using existing event loop."""
    # ... sync checks ...
    if flattened_async_checks:
        async_results = await _execute_all_async_checks(  #  Uses existing loop
            flattened_async_checks, max_async_concurrent
        )
    # ... rest of logic
```

**2. Update main `evaluate()` function**

```python
async def evaluate(
        test_cases: list[TestCase],
        outputs: list[Output],
        checks: list[CheckTypes] | list[list[CheckTypes]] | None = None,
        metadata: dict[str, Any] | None = None,
        max_async_concurrent: int | None = None,
        max_parallel_workers: int = 1,
    ) -> EvaluationRunResult:
    """
    Execute checks against test cases and their corresponding outputs (async).

    This is the primary evaluation function. Use this in:
    - Async contexts and async functions
    - Jupyter notebooks (supports top-level await)
    - When composing with other async operations

    For synchronous/blocking contexts (simple scripts), use evaluate_sync() instead.

    Args:
        test_cases: List of test cases to evaluate
        outputs: List of system outputs corresponding to test cases
        checks: Either:
            - List[CheckTypes]: Same checks applied to all test cases (shared pattern)
            - List[List[CheckTypes]]: checks[i] applies to test_cases[i] (per-test-case pattern)
            - None: Extract checks from TestCase.checks field (convenience pattern)
        metadata: Optional metadata for the evaluation run
        max_async_concurrent: Maximum number of concurrent async check executions (default: no limit)
        max_parallel_workers: Number of parallel worker processes (default: 1, no parallelization)

    Returns:
        Complete evaluation results with all test case results and summary statistics

    Raises:
        ValidationError: If inputs don't meet FEP protocol requirements

    Example:
        >>> # In async function or Jupyter notebook
        >>> results = await evaluate(test_cases, outputs, checks)
        >>>
        >>> # Compose with other async operations
        >>> results, data = await asyncio.gather(
        ...     evaluate(test_cases, outputs, checks),
        ...     fetch_additional_data()
        ... )
    """
    started_at = datetime.now(UTC)
    evaluation_id = str(uuid.uuid4())

    # Validate inputs
    _validate_inputs(test_cases, outputs, checks)
    resolved_checks = _resolve_checks(test_cases, checks)

    # Execute evaluation
    if max_parallel_workers > 1:
        # Note: Parallel processing still uses ProcessPoolExecutor,
        # which handles async internally via _evaluate_with_registry
        test_case_results = _evaluate_parallel(
            test_cases, outputs, resolved_checks, max_async_concurrent, max_parallel_workers,
        )
    else:
        work_items = list(zip(test_cases, outputs, resolved_checks))
        test_case_results = await _evaluate_async(work_items, max_async_concurrent)

    completed_at = datetime.now(UTC)
    summary = _compute_evaluation_summary(test_case_results)
    status = _compute_evaluation_status(test_case_results)

    return EvaluationRunResult(
        evaluation_id=evaluation_id,
        started_at=started_at,
        completed_at=completed_at,
        status=status,
        summary=summary,
        results=test_case_results,
        metadata=metadata,
    )


def evaluate_sync(
        test_cases: list[TestCase],
        outputs: list[Output],
        checks: list[CheckTypes] | list[list[CheckTypes]] | None = None,
        metadata: dict[str, Any] | None = None,
        max_async_concurrent: int | None = None,
        max_parallel_workers: int = 1,
    ) -> EvaluationRunResult:
    """
    Execute checks against test cases and their corresponding outputs (synchronous).

    Convenience wrapper for synchronous contexts. Creates a new event loop internally
    using asyncio.run().

    Use this ONLY in synchronous scripts where you cannot use async/await.
    For async contexts or Jupyter notebooks, use evaluate() instead.

    Args:
        Same as evaluate()

    Returns:
        Same as evaluate()

    Example:
        >>> # In synchronous script
        >>> results = evaluate_sync(test_cases, outputs, checks)

    Note:
        This function will fail if called from within an existing event loop.
        In async contexts, use the async evaluate() function instead.
    """
    return asyncio.run(
        evaluate(
            test_cases=test_cases,
            outputs=outputs,
            checks=checks,
            metadata=metadata,
            max_async_concurrent=max_async_concurrent,
            max_parallel_workers=max_parallel_workers,
        )
    )
```

**3. Update `_evaluate_with_registry()` for parallel processing**

This function runs in worker processes, so it needs to handle async:

```python
def _evaluate_with_registry(
        work_items: list[tuple[TestCase, Output, list[BaseCheck | BaseAsyncCheck]]],
        max_async_concurrent: int | None = None,
        registry_state: dict | None = None,
    ) -> list[TestCaseResult]:
    """Evaluate work items in a separate process with registry restoration."""
    # Restore registry state in the worker process
    if registry_state:
        restore_registry_state(registry_state)

    # Worker processes need to create their own event loop
    return asyncio.run(_evaluate_async(work_items, max_async_concurrent))
```

**4. Update exports in `__init__.py`**

Add both `evaluate` and `evaluate_sync` to exports:

```python
from .engine import evaluate, evaluate_sync

__all__ = [
    # ... existing exports ...
    "evaluate",
    "evaluate_sync",
]
```

### Testing Strategy

**Test Cases to Update/Add**:

1. **Basic async evaluation** (`test_evaluation_engine.py`):
   - Update all existing `evaluate()` calls to `await evaluate()`
   - Add test fixtures with `@pytest.mark.asyncio` where needed
   - Test that async evaluate works with existing event loop

2. **Sync wrapper tests**:
   - Test `evaluate_sync()` works in non-async context
   - Verify it produces same results as async version
   - Test that it fails gracefully if called from async context (documents limitation)

3. **Mixed sync/async checks**:
   - Verify both sync and async checks execute correctly
   - Test that async checks run concurrently
   - Test `max_async_concurrent` limiting still works

4. **Parallel processing**:
   - Verify `max_parallel_workers > 1` still works
   - Test that worker processes handle async correctly
   - Ensure registry state restoration works

5. **Jupyter notebook simulation**:
   - Test with existing event loop (simulates notebook environment)
   - Verify no `RuntimeError: asyncio.run() cannot be called from a running event loop`

6. **Composition tests**:
   - Test `asyncio.gather(evaluate(...), other_async())` works
   - Verify evaluate integrates cleanly with other async operations

**Testing Implementation Notes**:
- Use `pytest-asyncio` for async test support (already in dependencies)
- Mark async tests with `@pytest.mark.asyncio`
- Test both code paths: single-threaded (`max_parallel_workers=1`) and parallel
- Focus on behavior verification, not implementation details
- Test realistic scenarios: LLM judge checks, semantic similarity (actual async I/O)

### Dependencies
None - this is the foundation milestone.

### Risk Factors

1. **Parallel processing complexity**: Worker processes need to create their own event loops. Solution: Use `asyncio.run()` in `_evaluate_with_registry()`.

2. **Test fixture updates**: Many tests will need async conversion. Mitigation: Start with core tests, then expand.

3. **Event loop conflicts in tests**: pytest-asyncio may have quirks. Mitigation: Use `@pytest.mark.asyncio` consistently.

4. **Performance regression**: Ensure async overhead doesn't slow down sync-only workloads. Mitigation: Benchmark before/after.

---

## Milestone 2: Update pytest Decorator

### Goal
Update the `@evaluate` pytest decorator to use the new async API, eliminating the complex threading workarounds currently needed for event loop conflicts.

### Success Criteria
- [ ] pytest decorator uses `await evaluate()` instead of sync version
- [ ] Threading workarounds removed (lines ~330-340 in pytest_decorator.py)
- [ ] All pytest decorator tests pass
- [ ] Decorator works with both sync and async test fixtures
- [ ] IDE integration still works

### Key Changes

**1. Simplify the async evaluation call**

Current code has complex threading to avoid event loop conflicts ([pytest_decorator.py:330-340](src/flex_evals/pytest_decorator.py#L330-L340)):

```python
# Current (complex workaround)
try:
    return asyncio.run(resolve_and_run())
except RuntimeError:
    # Workaround for event loop conflicts
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, resolve_and_run())
        return future.result(timeout=300)
```

Change to:

```python
# New (simple, clean)
async def _run_async_evaluation(...):
    """Run evaluation using existing event loop."""
    evaluation_result = await evaluate(  #  Just await it
        test_cases=expanded_test_cases,
        outputs=outputs,
        checks=checks,
    )
    # ... rest of logic
```

**2. Update wrapper function**

The decorator's wrapper should handle async properly:

```python
def decorator(func):
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        """Async wrapper for evaluation."""
        # ... setup code ...

        # Call evaluate directly (no threading needed)
        evaluation_result = await evaluate(
            test_cases=expanded_test_cases,
            outputs=outputs,
            checks=checks,
        )

        # ... result processing ...

    # Return async wrapper for pytest-asyncio to handle
    return async_wrapper
```

**3. Remove threading imports and workarounds**

Delete:
- `import concurrent.futures`
- Threading executor logic
- Timeout handling (let pytest handle timeouts)

### Testing Strategy

**Test Cases**:

1. **Basic decorator functionality**:
   - Test decorator with sync test function
   - Test decorator with async test function
   - Verify evaluation runs correctly in both cases

2. **Async fixture integration**:
   - Test with async fixtures (this was the pain point)
   - Verify no event loop conflicts
   - Test fixture resolution works correctly

3. **Statistical sampling**:
   - Test `samples > 1` parameter
   - Verify success threshold calculation
   - Test failure cases

4. **Error handling**:
   - Test function exceptions are caught
   - Test check execution errors
   - Verify error reporting is clear

5. **IDE integration**:
   - Manually verify tests run in IDE (if applicable)
   - Test pytest discovery works
   - Verify test names are correct

### Dependencies
Must complete Milestone 1 first (async evaluate API).

### Risk Factors

1. **pytest-asyncio compatibility**: Different versions may behave differently. Mitigation: Document required version.

2. **Fixture resolution**: Async fixture handling may be tricky. Mitigation: Test thoroughly with various fixture patterns.

3. **IDE integration**: Some IDEs may not handle async tests well. Mitigation: Document known issues.

---

## Milestone 3: Update Documentation and Examples

### Goal
Update all documentation, examples, and docstrings to reflect the async-first API and guide users on when to use each function.

### Success Criteria
- [ ] README.md updated with async examples as primary
- [ ] Docstrings updated in all relevant files
- [ ] Usage examples show both `evaluate()` and `evaluate_sync()`
- [ ] Clear guidance on when to use each function
- [ ] Jupyter notebook examples updated
- [ ] Migration guide provided (if applicable)

### Key Changes

**1. Update README.md**

Primary example should use async:

```python
# Quick Start (Async - Recommended)
import asyncio
from flex_evals import evaluate, TestCase, Output, ContainsCheck

async def main():
    test_cases = [TestCase(input="What is the capital of France?")]
    outputs = [Output(value="Paris is the capital of France")]
    checks = [ContainsCheck(text="$.output.value", phrases="Paris")]

    results = await evaluate(test_cases, outputs, checks)
    print(f"Passed: {results.summary.completed_test_cases}/{results.summary.total_test_cases}")

# Run it
asyncio.run(main())

# Quick Start (Sync - Simple Scripts)
from flex_evals import evaluate_sync, TestCase, Output, ContainsCheck

test_cases = [TestCase(input="What is the capital of France?")]
outputs = [Output(value="Paris is the capital of France")]
checks = [ContainsCheck(text="$.output.value", phrases="Paris")]

results = evaluate_sync(test_cases, outputs, checks)  # No await needed
print(f"Passed: {results.summary.completed_test_cases}/{results.summary.total_test_cases}")
```

**2. Add "When to Use Which API" section**

```markdown
## Choosing Between evaluate() and evaluate_sync()

### Use `evaluate()` (async) when:
-  Working in Jupyter notebooks (supports top-level `await`)
-  Building async applications or services
-  Integrating with other async operations (databases, APIs, etc.)
-  You want to compose multiple evaluations concurrently

### Use `evaluate_sync()` when:
-  Writing simple synchronous scripts
-  You cannot use async/await in your context
-   Note: Will fail if called from within an existing event loop

### Examples

**Jupyter Notebook (Recommended)**
```python
# Just await it directly - Jupyter handles the event loop
results = await evaluate(test_cases, outputs, checks)
```

**Async Application**
```python
async def evaluate_llm_outputs():
    # Compose with other async operations
    results, metrics = await asyncio.gather(
        evaluate(test_cases, outputs, checks),
        fetch_performance_metrics()
    )
    return results, metrics
```

**Simple Script**
```python
def main():
    results = evaluate_sync(test_cases, outputs, checks)
    print(results.summary)

if __name__ == "__main__":
    main()
```
```

**3. Update all code examples in README**

Search for all `evaluate(` calls and update to either:
- `await evaluate(` (for async contexts)
- `evaluate_sync(` (for sync examples)

Add proper imports and async context where needed.

**4. Update CLAUDE.md project instructions**

Update the project overview to mention:
- Primary API is async
- Sync wrapper available for convenience
- When to use each

**5. Update inline docstrings**

Update docstrings in:
- `engine.py` - both `evaluate()` and `evaluate_sync()`
- `pytest_decorator.py` - mention async behavior
- Any other files that reference the evaluate function

**6. Create migration note (optional)**

Since backwards compatibility isn't required, consider a short migration note in the commit message or changelog:

```markdown
## Breaking Change: evaluate() is now async

### What Changed
- `evaluate()` is now `async def evaluate()` - must be awaited
- New `evaluate_sync()` function for synchronous contexts

### Migration
**Before:**
```python
results = evaluate(test_cases, outputs, checks)
```

**After (Async - Recommended):**
```python
results = await evaluate(test_cases, outputs, checks)
```

**After (Sync - Simple Scripts):**
```python
results = evaluate_sync(test_cases, outputs, checks)
```

### Why This Change
Solves event loop conflicts in Jupyter notebooks and async contexts,
enables composition with other async operations, and reflects the
true async nature of evaluation operations.
```

### Testing Strategy

**Documentation Testing**:

1. **README examples**: Manually verify all code examples work
2. **Docstring examples**: Run doctest if applicable
3. **Example scripts**: Create/update example scripts that demonstrate both APIs
4. **Jupyter notebook**: Create example notebook showing async usage

**Review Checklist**:
- [ ] All code examples are syntactically correct
- [ ] Imports are complete and correct
- [ ] Async examples use proper async/await syntax
- [ ] Guidance is clear and actionable
- [ ] Links between related sections work

### Dependencies
Should be done after Milestones 1 and 2 are complete and tested.

### Risk Factors

1. **Documentation drift**: Examples may become outdated. Mitigation: Add tests for critical examples.

2. **User confusion**: Two functions may confuse users. Mitigation: Clear, prominent guidance on which to use.

---

## Milestone 4: Performance Validation and Optimization

### Goal
Ensure the async refactor doesn't introduce performance regressions, and optimize if needed.

### Success Criteria
- [ ] Benchmark comparison shows no significant regression
- [ ] Async overhead is minimal for sync-only checks
- [ ] Parallel processing performance maintained or improved
- [ ] Memory usage is reasonable
- [ ] Performance documentation updated

### Key Changes

**1. Create benchmark suite**

```python
# tests/test_performance.py
import pytest
import time
import asyncio
from flex_evals import evaluate, evaluate_sync, TestCase, Output, ExactMatchCheck

@pytest.mark.benchmark
def test_sync_only_checks_performance():
    """Benchmark evaluation with only sync checks."""
    test_cases = [TestCase(input=f"test_{i}") for i in range(100)]
    outputs = [Output(value=f"test_{i}") for i in range(100)]
    checks = [ExactMatchCheck(actual="$.output.value", expected="$.test_case.input")]

    start = time.perf_counter()
    results = evaluate_sync(test_cases, outputs, checks)
    duration = time.perf_counter() - start

    assert results.summary.total_test_cases == 100
    print(f"Sync-only benchmark: {duration:.3f}s ({duration/100*1000:.2f}ms per test case)")

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_async_checks_performance():
    """Benchmark evaluation with async checks."""
    # Test with actual async checks (semantic similarity, LLM judge)
    # Measure concurrent execution efficiency
    pass

@pytest.mark.benchmark
def test_parallel_processing_performance():
    """Benchmark parallel processing performance."""
    # Test with max_parallel_workers > 1
    # Compare against serial execution
    pass
```

**2. Profile critical paths**

- Profile `_evaluate_async()` with cProfile
- Identify any new bottlenecks introduced by async conversion
- Optimize if needed

**3. Memory profiling**

- Ensure event loop overhead is reasonable
- Check for memory leaks in async execution
- Profile parallel processing memory usage

### Testing Strategy

**Benchmarks to Run**:

1. **Sync-only workload**: 100 test cases, only sync checks
2. **Async-only workload**: 100 test cases, only async checks
3. **Mixed workload**: 100 test cases, mixed sync/async checks
4. **Parallel processing**: 1000 test cases, various worker counts
5. **Memory usage**: Monitor memory over extended runs

**Comparison Metrics**:
- Total execution time
- Time per test case
- Async check concurrency (should be concurrent)
- Memory usage (MB)
- CPU utilization

**Acceptance Criteria**:
- No more than 5% regression in sync-only scenarios
- Async checks show clear concurrency (N checks in ~1x time, not Nx time)
- Parallel processing maintains or improves performance
- Memory usage scales linearly with test case count

### Testing Strategy

Run benchmarks:
```bash
# Run benchmark tests
uv run pytest tests/test_performance.py -m benchmark -v

# Profile specific functions
uv run python -m cProfile -s cumulative tests/profile_evaluate.py
```

Document results in benchmark report.

### Dependencies
Must complete Milestones 1-3 first.

### Risk Factors

1. **Event loop overhead**: Async has inherent overhead. Mitigation: Only async code should await; sync code should execute directly.

2. **Benchmarking noise**: Performance tests can be noisy. Mitigation: Run multiple iterations, use median values.

3. **Platform differences**: Performance may vary by OS/Python version. Mitigation: Document test environment.

---

## Implementation Order Summary

1. **Milestone 1**: Core engine async refactor (highest priority, unblocks everything)
2. **Milestone 2**: Update pytest decorator (depends on M1, high value)
3. **Milestone 3**: Documentation updates (depends on M1-2, critical for users)
4. **Milestone 4**: Performance validation (depends on M1-3, ensures quality)

## Key Implementation Principles

1. **Async-first design**: The primary API should be async; sync is a convenience wrapper
2. **Clean separation**: `_evaluate_async()` uses existing loop; `evaluate_sync()` creates new loop
3. **No threading workarounds**: Let async/await handle concurrency naturally
4. **Clear documentation**: Users need guidance on when to use each API
5. **Comprehensive testing**: Test both async and sync paths, edge cases, and real-world scenarios
6. **Performance awareness**: Ensure async overhead doesn't hurt sync-only workloads

## Questions to Resolve Before Implementation

1. Should we keep both `evaluate()` and `evaluate_sync()` or provide just one?
   - **Decision**: Provide both; async as primary, sync as convenience

2. How should parallel processing work with async?
   - **Decision**: Worker processes create their own event loops via `asyncio.run()`

3. What should happen if `evaluate_sync()` is called from async context?
   - **Decision**: Let it fail with clear error message; document limitation

4. Should we deprecate any existing APIs?
   - **Decision**: No deprecation needed; clean break is acceptable

## Success Indicators

-  No `RuntimeError: asyncio.run() cannot be called from a running event loop` in notebooks
-  pytest decorator no longer needs threading workarounds
-  Can compose `evaluate()` with other async operations via `asyncio.gather()`
-  Clear, simple examples in documentation
-  All tests pass
-  No significant performance regression
