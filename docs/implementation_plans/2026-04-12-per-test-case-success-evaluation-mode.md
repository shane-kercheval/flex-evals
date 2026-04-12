# Implementation Plan: Per-Test-Case Success Evaluation Mode

## Overview

**Goal**: Add a `pass_mode` parameter to the `@evaluate` decorator that controls how pass/fail is aggregated, and rename `success_threshold` to `pass_threshold` for clarity.

**Why**: The current sample-level aggregation creates an exponential penalty for adding test cases. With N independent test cases each passing at rate P, the probability of a clean sample is P^N. This means adding more test coverage lowers the overall pass rate, even when individual test cases are performing well. Per-test-case mode evaluates each test case independently across samples, so a flaky test case is identified specifically rather than dragging down unrelated test cases.

**Modes**:
- `PassMode.PER_TEST_CASE` (new default) — each test case is evaluated independently across samples; overall eval passes if ALL test cases meet the threshold individually
- `PassMode.SAMPLE` (current behavior) — a sample passes if ALL test cases pass; overall eval passes if enough samples meet the threshold

**Renames**:
- `success_threshold` → `pass_threshold`

**Per-test-case breakdown**: Always included in `_test_results` metadata regardless of mode, providing diagnostic visibility into which test cases are weak.

**Files Modified**:
- `src/flex_evals/constants.py` — Add `PassMode` enum
- `src/flex_evals/pytest_decorator.py` — Add `pass_mode` parameter, rename `success_threshold`, implement per-test-case aggregation, new failure report, updated metadata
- `src/flex_evals/__init__.py` — Export `PassMode`
- `tests/test_pytest_decorator.py` — Update existing tests for rename, add per-test-case mode tests
- `README.md` — Rewrite evaluation section documenting both modes
- `examples/pytest_decorator_example.py` — Update examples, add per-test-case mode example

**Breaking Changes**: Yes.
- `success_threshold` parameter renamed to `pass_threshold`
- Default behavior changes from sample-level to per-test-case evaluation
- `_test_results` metadata structure changes (new fields, renamed fields)

---

## Milestone 1: `PassMode` Enum, Parameter Rename, Per-Test-Case Aggregation

### Goal & Outcome

Implement the core behavior change. After this milestone:
- `PassMode` enum exists with `SAMPLE` and `PER_TEST_CASE` values
- `@evaluate` accepts `pass_mode` (default: `PassMode.PER_TEST_CASE`) and `pass_threshold` (replaces `success_threshold`)
- Per-test-case mode: each test case evaluated independently across samples, overall passes if all test cases individually meet threshold
- Sample mode: existing behavior preserved under `PassMode.SAMPLE`
- `_test_results` metadata always includes per-test-case breakdown regardless of mode
- `_test_config` metadata includes `pass_mode`
- Failure report shows per-test-case results when in per-test-case mode
- All existing tests updated and passing

### Implementation Outline

**1. Add `PassMode` enum to `constants.py`**

```python
class PassMode(str, Enum):
    """Evaluation pass/fail aggregation mode for the @evaluate decorator."""
    SAMPLE = 'sample'
    PER_TEST_CASE = 'per_test_case'
```

Export from `src/flex_evals/__init__.py`.

**2. Update `@evaluate` decorator signature in `pytest_decorator.py`**

```python
def evaluate(
    test_cases: list[TestCase],
    checks: list[Check] | None = None,
    samples: int = 1,
    pass_threshold: float = 1.0,
    pass_mode: PassMode = PassMode.PER_TEST_CASE,
    max_concurrency: int | None = None,
    output_dir: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[[F], F]:
```

- Add `pass_mode` parameter. Accept `PassMode | str` and normalize via `pass_mode = PassMode(pass_mode)` with a try/except that gives a clear error for invalid values. This matches the project convention where all `str, Enum` types are used interchangeably with plain strings.
- Rename `success_threshold` → `pass_threshold` throughout the file
- Update the decorator docstring to document both modes clearly
- Update all internal references from `success_threshold` to `pass_threshold`

**3. Add `_compute_per_test_case_results()` helper**

A new helper function that computes per-test-case pass rates across samples. This is used by both modes (for metadata) and by per-test-case mode (for the pass/fail decision).

```python
def _compute_per_test_case_results(
    evaluation_results: list[TestCaseResult],
    num_test_cases: int,
    samples: int,
) -> list[dict[str, Any]]:
    """
    Compute per-test-case pass rates across samples.

    Returns list preserving original test case order, each entry:
        {"index": int, "id": str | None, "passed": int, "failed": int, "total": int, "rate": float}
    """
```

The logic:
- For each test case index `tc_idx` in `range(num_test_cases)`:
  - For each sample index `s_idx` in `range(samples)`:
    - Get the result at `evaluation_results[s_idx * num_test_cases + tc_idx]`
    - Check if all checks passed via `_check_sample_passed(tcr.check_results)`
  - Record index, id, pass count, fail count, total, and rate
  - Use the test case ID from the result's `execution_context.test_case.id` (may be `None`)
- **Grouping is by positional index, not by ID.** This avoids issues with duplicate or `None` IDs. The ID is included for display purposes only.
- **Important ordering invariant:** The results list is assumed to be in `test_cases * samples` order (all test cases for sample 0, then sample 1, etc.), matching the expansion in the decorator. Add a comment documenting this at the aggregation site.

**4. Update `_process_evaluation_result()`**

Add a branch based on `pass_mode`:

- `PassMode.SAMPLE`: existing logic (iterate samples, check if all test cases pass per sample)
- `PassMode.PER_TEST_CASE`: use `_compute_per_test_case_results()`, check if ALL test cases individually meet `pass_threshold`, call a new failure report function if any fail

For per-test-case mode:
```python
per_tc = _compute_per_test_case_results(evaluation_result.results, num_tc, samples)
failed_entries = [entry for entry in per_tc if entry["rate"] < pass_threshold]
if failed_entries:
    _generate_per_test_case_failure_report(...)
```

**5. Update `_save_result_if_configured()`**

The `_test_results` metadata should include per-test-case breakdown always.

**Single aggregation path:** Both `_save_result_if_configured()` and `_process_evaluation_result()` should consume the same helper functions for computing sample-level and per-test-case stats. Extract a `_compute_sample_results()` helper (analogous to `_compute_per_test_case_results()`) that returns `(passed_samples, failed_samples_count, sample_pass_rate)`. This avoids the current pattern of duplicating sample-level counting logic between persistence and assertion.

```python
per_tc = _compute_per_test_case_results(evaluation_result.results, num_tc, samples)
sample_passed, sample_failed, sample_pass_rate = _compute_sample_results(
    evaluation_result.results, num_tc, samples,
)

'_test_config': {
    'test_function': func.__name__,
    'test_module': func.__module__,
    'samples': samples,
    'pass_threshold': pass_threshold,
    'pass_mode': str(pass_mode),
    'max_concurrency': max_concurrency,
    'num_test_cases': num_tc,
},
'_test_results': {
    'pass_mode': str(pass_mode),
    'pass_threshold': pass_threshold,
    'passed': <bool>,  # depends on mode
    # Sample-level stats (always computed)
    'passed_samples': sample_passed,
    'failed_samples': sample_failed,
    'total_samples': samples,
    'sample_pass_rate': sample_pass_rate,
    # Per-test-case stats (always computed)
    'per_test_case': per_tc,  # list of {"index", "id", "passed", "failed", "total", "rate"}
    'failed_test_cases': [
        {"index": e["index"], "id": e["id"]}
        for e in per_tc if e["rate"] < pass_threshold
    ],
},
```

The `passed` field reflects the active mode's decision.

**6. Add `_generate_per_test_case_failure_report()`**

New function for per-test-case mode failure reporting:

```
Statistical evaluation failed for test_suggest_tags:
  Pass mode: per_test_case
  Pass threshold: 80.00%

  Per-test-case results:
    ✅ python-flask-tutorial: 10/10 (100.00%)
    ✅ javascript-react-guide: 10/10 (100.00%)
    ❌ postgres-index-note: 7/10 (70.00%)  ← below threshold
    ✅ machine-learning-paper: 10/10 (100.00%)

  Failed test cases: 1 of 4
```

The report should show ALL test cases (not just failures) so the user sees the full picture. Failed test cases should be clearly marked. Limit to a reasonable number if there are many test cases (e.g., show first 20, summarize rest).

**7. Update existing `_generate_failure_report()` for sample mode**

Update parameter names (`success_threshold` → `pass_threshold`, `success_rate` → `sample_pass_rate` in the report text). Add `pass_mode` to the report header for clarity.

### Testing Strategy

**Update all existing tests**: Every test that references `success_threshold` needs updating to `pass_threshold`. Every test that relies on sample-level pass/fail behavior should explicitly pass `pass_mode=PassMode.SAMPLE` since the default is changing.

**New tests for `PassMode.PER_TEST_CASE`**:

- **Basic per-test-case pass**: All test cases pass threshold → overall passes
- **Basic per-test-case fail**: One test case below threshold → overall fails, report identifies the failing test case
- **Mixed per-test-case results**: Some test cases at 100%, some at threshold, some below → correctly identifies failures
- **Single flaky test case**: 10 test cases where 9 pass at 100% and 1 passes at 70% with threshold 80% → fails, only the flaky test case is identified
- **Threshold boundary**: Test case at exactly `pass_threshold` → passes (>=)
- **Per-test-case with samples=1**: Each test case has 0% or 100% rate — should work correctly
- **Per-test-case with single test case**: Degenerates to simple pass/fail
- **Per-test-case failure report format**: Verify the report includes all test cases with pass rates and marks failures
- **Per-test-case where the exponential penalty problem is demonstrated**: Multiple test cases each passing at high individual rates but failing under sample mode, passing under per-test-case mode — validates the core motivation

**New tests for metadata**:

- **Per-test-case `_test_results` structure**: Verify `per_test_case` dict, `failed_test_cases` list, `pass_mode`, `pass_threshold` fields
- **Sample mode `_test_results` includes per-test-case breakdown**: Verify the per-test-case stats are present even in sample mode
- **`_test_config` includes pass_mode**: Verify the config metadata

**Tests for duplicate/None test case IDs**:

- **Duplicate IDs**: Two test cases with the same ID are tracked separately by positional index, both appear in per_test_case results with correct individual stats
- **None IDs**: Test cases with `id=None` are tracked by positional index, appear in per_test_case results and failure reports with index as identifier

**Validation tests**:

- **Invalid pass_mode value**: Invalid string raises `ValueError`
- **PassMode enum values work**: Both `PassMode.SAMPLE` and `PassMode.PER_TEST_CASE` accepted
- **String pass_mode accepted**: `pass_mode="sample"` and `pass_mode="per_test_case"` are accepted and normalized to `PassMode` enum values (matches project convention for `str, Enum` types)

**Sample mode preservation tests**:

- Verify existing sample-mode behavior is preserved when `pass_mode=PassMode.SAMPLE` is explicit
- The existing tests (after updating parameter names and adding explicit `pass_mode=PassMode.SAMPLE`) should cover this

---

## Milestone 2: Documentation & Examples

### Goal & Outcome

Update all documentation and examples to reflect the new API. After this milestone:
- README documents both modes with clear examples
- Decorator docstring is comprehensive
- Example file demonstrates both modes
- Users can understand when to use each mode

### Implementation Outline

**1. Update `README.md`**

Rewrite the "Pytest Integration" section (lines ~37-78):
- Update the introductory example to use `pass_threshold` and `pass_mode`
- Replace the "Understanding `success_threshold`" section with a new section explaining both modes
- Show a clear example of each mode with a table like the one in the feature request
- Explain when to use each mode (independent test cases → per_test_case, sequential/interdependent → sample)
- Update the initial code example

**2. Update `examples/pytest_decorator_example.py`**

- Rename all `success_threshold` → `pass_threshold` in existing examples
- Add a new example demonstrating `pass_mode=PassMode.PER_TEST_CASE` with multiple test cases and samples
- Add a new example demonstrating `pass_mode=PassMode.SAMPLE` with a comment explaining when this mode is appropriate
- Import `PassMode` in the examples

**3. Update decorator docstring**

The docstring in `evaluate()` should clearly document:
- `pass_threshold` parameter with explanation for both modes
- `pass_mode` parameter with description of each mode and when to use each:
  - `PER_TEST_CASE`: use when test cases are independent scenarios (e.g., different inputs, tag suggestions, classification evals). Each test case is evaluated on its own merit. This is the default and the right choice for most LLM evaluations.
  - `SAMPLE`: use when test cases are interdependent or sequential (e.g., end-to-end pipeline tests where "the whole pipeline works" is the meaningful question). A single failure in any test case means the entire sample failed.
- Updated example showing both modes

### Testing Strategy

No new code tests needed — this is documentation only. Verify:
- All examples in README are syntactically valid
- `examples/pytest_decorator_example.py` runs successfully: `uv run pytest examples/pytest_decorator_example.py`
