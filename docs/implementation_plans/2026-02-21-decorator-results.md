# Implementation Plan: Result Persistence for `@evaluate` Decorator

## Overview

**Goal**: Add `output_dir` and `metadata` parameters to the `@evaluate` decorator so that `EvaluationRunResult` is automatically saved to JSON after each evaluation run.

**Why**: Results saved to disk enable cross-run comparison, visualization, and analysis without modifying test code. Saving happens _before_ `_process_evaluation_result()` calls `pytest.fail()`, so results are always captured — even on threshold failures.

**Files Modified**:
- `src/flex_evals/schemas/results.py` — Extract `FlexEvalsJSONEncoder` to module level
- `src/flex_evals/schemas/__init__.py` — Export `FlexEvalsJSONEncoder`
- `src/flex_evals/pytest_decorator.py` — New parameters, save logic
- `tests/test_pytest_decorator.py` — New tests for all persistence scenarios
- `tests/conftest.py` — Remove `pytest_plugins` (moved to test file)

**Breaking Changes**: None. Both new parameters default to `None`, preserving existing behavior. `FlexEvalsJSONEncoder` is a new public export.

---

## Milestone 1: Add `output_dir` and `metadata` Parameters with Save Logic

### Goal & Outcome

Add the `output_dir` and `metadata` parameters to the `@evaluate` decorator along with a save function that runs _before_ `_process_evaluation_result`. Do NOT refactor `_process_evaluation_result` — leave it untouched and compute sample counts independently in the save path.

After this milestone:
- `evaluate()` accepts `output_dir: str | Path | None` and `metadata: dict[str, Any] | None`
- `FLEX_EVALS_OUTPUT_DIR` env var is supported as fallback for `output_dir`
- When `output_dir` is set (via param or env var), a JSON file is written to disk after evaluation completes but _before_ threshold checking
- The JSON file is a clean serialized `EvaluationRunResult` with merged metadata:
  - User-provided `metadata` at top level
  - Auto-captured `_test_config` (test function name, module, samples, threshold, num test cases)
  - Computed `_test_results` (passed/failed/total samples, success rate, threshold, passed bool)
- Keys prefixed with `_test_` are reserved for auto-captured data and will overwrite user-provided keys with the same names (documented in the `metadata` parameter docstring)
- Save failures emit a `warnings.warn(..., RuntimeWarning)` but never alter test outcomes
- `_process_evaluation_result` is NOT modified — the save path computes its own counts inline
- All existing tests continue to pass unchanged

### Implementation Outline

**1. Extract `FlexEvalsJSONEncoder` to module level in `results.py`**

Move the `CustomEncoder` class from inside `serialize()` to a module-level `FlexEvalsJSONEncoder` class. This allows the decorator to reuse it when serializing user-provided metadata (which may contain datetime, Path, Pydantic models, etc.). Export it from `schemas/__init__.py`.

**2. Add parameters to `evaluate()`**

Add `output_dir: str | Path | None = None` and `metadata: dict[str, Any] | None = None` to the `evaluate()` function signature.

Add imports at the top of the file:
- `import os`, `uuid`, `json`, `warnings`
- `from datetime import datetime, UTC`
- `from pathlib import Path`
- `FlexEvalsJSONEncoder` from `.schemas`

**3. Implement `_save_result_if_configured` as a closure inside `decorator()`**

This function needs access to `output_dir`, `metadata`, `func`, `test_cases`, `samples`, and `success_threshold` from the enclosing scope. This follows the existing pattern — the decorator already has ~10 nested closures.

It should:

1. Resolve the effective output dir: explicit param > `FLEX_EVALS_OUTPUT_DIR` env var > `None`
2. If `None`, return early (no-op)
3. Compute sample pass/fail counts inline (~5 lines — do NOT extract a shared helper):
   ```python
   num_tc = len(test_cases)
   passed = sum(
       1 for i in range(samples)
       if all(
           _check_sample_passed(tcr.check_results)
           for tcr in evaluation_result.results[i * num_tc:(i + 1) * num_tc]
       )
   )
   failed = samples - passed
   success_rate = passed / samples
   ```
   This duplicates the counting pattern from `_process_evaluation_result`, but the two paths have genuinely different needs: the save path needs four numbers, while `_process_evaluation_result` builds detailed failure info with exceptions per sample for the failure report. Premature unification would add complexity for little gain.
4. Set `evaluation_result.metadata` directly with the merged metadata dict, then call `serialize()`. This produces a clean `EvaluationRunResult` serialization. The base layer is any existing engine-level metadata, then user-provided metadata, then auto-captured `_test_config` and `_test_results` (which take precedence). `_process_evaluation_result` does not read `evaluation_result.metadata`, so this mutation is safe:
   ```python
   evaluation_result.metadata = {
       **(evaluation_result.metadata or {}),
       **(metadata or {}),
       '_test_config': {
           'test_function': func.__name__,
           'test_module': func.__module__,
           'samples': samples,
           'success_threshold': success_threshold,
           'num_test_cases': num_tc,
       },
       '_test_results': {
           'passed_samples': passed,
           'failed_samples': failed,
           'total_samples': samples,
           'success_rate': success_rate,
           'success_threshold': success_threshold,
           'passed': success_rate >= success_threshold,
       },
   }
   serialized = evaluation_result.serialize()
   ```
5. Generate filename: `{ISO_timestamp}_{short_uuid}.json` (colons replaced with hyphens for filesystem safety)
6. Create output dir (including parents) via `Path.mkdir(parents=True, exist_ok=True)`
7. Write `json.dumps(serialized, indent=2, cls=FlexEvalsJSONEncoder)` to file. Uses `indent=2` for human-readable output (these files are intended for analysis and inspection). Uses `FlexEvalsJSONEncoder` so user-provided metadata with non-JSON-safe types (datetime, Path, etc.) is handled gracefully instead of losing the entire result file.
8. Wrap everything in `try/except` that calls `warnings.warn("Failed to save evaluation result: {error}", RuntimeWarning, stacklevel=2)` on any failure. Using `warnings.warn` instead of `print(file=sys.stderr)` because pytest captures stderr by default, while warnings appear in pytest's warnings summary section — more visible and idiomatic.

**4. Call save before process in both sync and async paths**

In `_evaluate_results_sync` and `_evaluate_results_async`, call `_save_result_if_configured(evaluation_result)` between the `evaluate_sync()`/`evaluate_async()` call and `_process_evaluation_result()`. Do NOT modify `_process_evaluation_result` at all.

### Testing Strategy

All tests use `tmp_path` for temp directories and simple deterministic checks to stay fast. Follow the existing class+method naming convention in the test file.

1. **`test__output_dir_saves_json_file`** — Decorate a simple passing sync function with `output_dir=tmp_path`. Verify exactly one `.json` file is created, it's valid JSON, and contains expected top-level keys (`evaluation_id`, `results`, `metadata`, etc.).

2. **`test__output_dir_none_does_not_save`** — Decorate with `output_dir=None` and no env var set. Verify no files are created in any temp directory. Use `monkeypatch.delenv('FLEX_EVALS_OUTPUT_DIR', raising=False)` to ensure env var is not set.

3. **`test__output_dir_creates_directory`** — Set `output_dir` to `tmp_path / "nested" / "deep" / "dir"` (non-existent). Verify the directories are created and a JSON file is written.

4. **`test__metadata_included_in_saved_file`** — Pass `metadata={"model": "test-model", "provider": "test"}`. Load the saved JSON and verify these keys are present in `metadata` at the top level.

5. **`test__auto_metadata_included`** — Save a result and verify `metadata._test_config` contains `test_function`, `test_module`, `samples`, `success_threshold`, and `num_test_cases` with correct values.

6. **`test__saved_file_contains_test_results`** — Use a function that passes some samples and fails others (deterministic via `call_count`). Verify `metadata._test_results` has correct `passed_samples`, `failed_samples`, `total_samples`, `success_rate`, `success_threshold`, and `passed` boolean.

7. **`test__save_failure_does_not_mask_test_result`** — Set `output_dir` to an unwritable path (e.g., `/dev/null/impossible`). Verify a passing test still passes normally. Use `pytest.warns(RuntimeWarning)` to verify a warning was emitted.

8. **`test__save_failure_does_not_mask_failing_test`** — Same unwritable path, but with a function that fails checks. Verify the test still fails via `pytest.raises(Failed)` and a `RuntimeWarning` is emitted.

9. **`test__results_saved_on_threshold_failure`** — Create a function that fails the success threshold. Verify that the JSON file is still written to disk despite `pytest.fail()` being called. Use `pytest.raises(_pytest.outcomes.Failed)` to catch the failure, then check that the file exists and contains valid data.

10. **`test__env_var_output_dir`** — Use `monkeypatch.setenv('FLEX_EVALS_OUTPUT_DIR', str(tmp_path))` without setting the `output_dir` parameter. Verify results are saved to the env var path.

11. **`test__explicit_output_dir_overrides_env_var`** — Set both `output_dir=tmp_path / "explicit"` and `monkeypatch.setenv('FLEX_EVALS_OUTPUT_DIR', str(tmp_path / "env"))`. Verify file is written to the explicit path and not the env var path.

12. **`test__async_output_dir_saves_json_file`** — Same as test #1 but with an `async def` decorated function. Verifies the save logic works correctly through the async execution path (which involves threading and event loop management).
