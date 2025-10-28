# Analysis: Remove ExperimentMetadata Redundancy

**Date:** 2025-10-28
**Status:** Proposed
**Decision:** Remove `ExperimentMetadata` class and `experiment` field from `EvaluationRunResult`

## Problem Statement

The current design has two separate metadata storage locations in `EvaluationRunResult`:

1. `experiment: ExperimentMetadata | None` - Contains nested `name` and `metadata` fields
2. `metadata: dict[str, Any] | None` - For "implementation-specific" data

This creates unnecessary complexity and confusion about where to store evaluation context.

## Current Usage Analysis

### Schema Structure

**File:** `src/flex_evals/schemas/experiments.py`
```python
@dataclass
class ExperimentMetadata:
    name: str | None = None
    metadata: dict[str, Any] | None = None
```

**File:** `src/flex_evals/schemas/results.py`
```python
@dataclass
class EvaluationRunResult:
    # ... other fields ...
    experiment: ExperimentMetadata | None = None
    metadata: dict[str, Any] | None = None
```

### Public API

**Function:** `evaluate()` in `src/flex_evals/engine.py`
```python
def evaluate(
    test_cases: list[TestCase],
    outputs: list[Output],
    checks: list[CheckTypes] | list[list[CheckTypes]] | None = None,
    experiment_metadata: ExperimentMetadata | None = None,
    # ...
) -> EvaluationRunResult:
```

The `experiment_metadata` parameter is directly assigned to `EvaluationRunResult.experiment`.

### Data Flattening

In `to_dict_list()` method (lines 238-247 of `results.py`):
```python
# Add experiment metadata if present at evaluation level
if self.experiment:
    if self.experiment.name:
        test_case_data['experiment_name'] = self.experiment.name
    if self.experiment.metadata:
        test_case_data['experiment_metadata'] = self.experiment.metadata

# Add evaluation metadata if present
if self.metadata:
    test_case_data['evaluation_metadata'] = self.metadata
```

This creates three separate columns in flattened output:
- `experiment_name`
- `experiment_metadata`
- `evaluation_metadata`

### Test Coverage

**Tests Found:**
- `test_evaluate_experiment_metadata()` - Verifies experiment metadata passes through
- `test_evaluate_with_none_ids_experiment_metadata()` - Tests with None IDs
- `test_with_experiment_metadata()` - Schema validation
- `test_to_dict_list_single_test_case_single_check()` - Flattening with experiment data

**Example Usage:**
`examples/example_yaml_test_cases.ipynb` shows real-world usage:
```python
evaluation_results = evaluate(
    test_cases=test_cases,
    outputs=outputs,
    checks=global_checks,
    experiment_metadata=ExperimentMetadata(
        name=config['experiment']['name'],
        metadata=config['experiment']['metadata'],
    ),
)
```

## Issues with Current Design

### 1. Confusing Separation
- What's the semantic difference between "experiment metadata" and "evaluation metadata"?
- Both describe context for the same evaluation run
- Users must decide which to use, creating cognitive overhead

### 2. Nested Metadata Structure
- `EvaluationRunResult.experiment.metadata` creates unnecessary nesting
- Just to store key-value pairs that could go in top-level `metadata`

### 3. Unnecessary Boilerplate
- Users must import `ExperimentMetadata`
- Must instantiate a separate object just to add a name or version
- More verbose than needed

### 4. Rigid Structure
- Forces a specific structure (name + metadata dict)
- Users lose flexibility in organizing their context data

## Proposed Solution

**Remove `ExperimentMetadata` entirely.** Use only the existing `metadata` field.

### Benefits

1. **Simpler API**: One place for all evaluation context
2. **More Flexible**: Users organize metadata however they want
3. **Less Code**: Remove entire `experiments.py` module
4. **Clearer Intent**: No confusion about which metadata field to use
5. **Same Functionality**: Everything possible with `ExperimentMetadata` can be done with plain dict

### Migration Pattern

**Before:**
```python
from flex_evals import evaluate, ExperimentMetadata

result = evaluate(
    test_cases=cases,
    outputs=outputs,
    checks=checks,
    experiment_metadata=ExperimentMetadata(
        name="my_experiment",
        metadata={"version": "1.0", "purpose": "testing"}
    )
)

# Access
result.experiment.name
result.experiment.metadata["version"]
```

**After:**
```python
from flex_evals import evaluate

result = evaluate(
    test_cases=cases,
    outputs=outputs,
    checks=checks,
    metadata={
        "experiment_name": "my_experiment",
        "version": "1.0",
        "purpose": "testing"
    }
)

# Access
result.metadata["experiment_name"]
result.metadata["version"]
```

Or users can nest if they prefer:
```python
metadata={
    "experiment": {
        "name": "my_experiment",
        "version": "1.0"
    }
}
```

## Implementation Changes Required

### 1. Delete Files
- `src/flex_evals/schemas/experiments.py`

### 2. Update `src/flex_evals/schemas/results.py`
- Remove import: `from flex_evals.schemas.experiments import ExperimentMetadata`
- Remove field: `experiment: ExperimentMetadata | None = None` from `EvaluationRunResult`
- Update docstring to remove experiment field reference
- Simplify `to_dict_list()` method:
  - Remove experiment name/metadata extraction (lines 238-243)
  - Just propagate top-level `metadata` as `evaluation_metadata`

### 3. Update `src/flex_evals/engine.py`
- Remove import: `ExperimentMetadata` from imports
- Remove parameter: `experiment_metadata: ExperimentMetadata | None = None` from `evaluate()`
- Add parameter: `metadata: dict[str, Any] | None = None` to `evaluate()`
- Update docstring to document new `metadata` parameter
- Update result creation:
  ```python
  return EvaluationRunResult(
      # ... other fields ...
      metadata=metadata,
  )
  ```

### 4. Update Public API Exports
- `src/flex_evals/schemas/__init__.py` - Remove `ExperimentMetadata` export
- `src/flex_evals/__init__.py` - Remove from `__all__` list

### 5. Update Tests
- `tests/test_evaluation_engine.py`:
  - Remove/update `test_evaluate_experiment_metadata()`
  - Remove/update `test_evaluate_with_none_ids_experiment_metadata()`
  - Add tests for new `metadata` parameter

- `tests/test_schemas/test_results.py`:
  - Remove/update `test_with_experiment_metadata()`
  - Update `test_to_dict_list_*()` tests to use `metadata` instead
  - Verify flattening behavior with new structure

### 6. Update Examples
- `examples/example_yaml_test_cases.ipynb`:
  - Update to use `metadata` parameter instead of `experiment_metadata`
  - Show how to organize experiment context within metadata dict

### 7. Update Documentation
- `README.md` - Remove `ExperimentMetadata` references, document `metadata` usage
- Any other docs mentioning experiment metadata

## Validation Checklist

- [ ] All tests pass
- [ ] Linting passes
- [ ] Examples run successfully
- [ ] Documentation updated
- [ ] No references to `ExperimentMetadata` remain
- [ ] `to_dict_list()` works correctly with new structure
- [ ] Migration path documented (if needed)

## Notes

- No backwards compatibility required (per project constraints)
- This is a breaking API change - requires version bump
- Consider adding migration guide if there are external users
- The flattened output format will change - document this clearly

## Decision Rationale

The principle of simplicity and avoiding premature structure applies here. `ExperimentMetadata` adds a layer of structure that provides no real benefit over a plain dictionary. Users who need structure can create it themselves within the `metadata` field. The framework shouldn't impose rigid organization on what is essentially free-form context data.
