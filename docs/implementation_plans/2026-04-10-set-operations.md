# Implementation Plan: Set Operation Checks

## Overview

**Goal**: Add four new check types — `SubsetCheck`, `SupersetCheck`, `SetEqualCheck`, and `DisjointCheck` — for evaluating list/set membership in model outputs.

**Why**: The existing `ContainsCheck` does string substring matching (`phrase in text_str`), which is fundamentally different from set membership. Stringifying a list and doing substring matching produces false positives (e.g., tag `"on"` matches inside `"python"`). These checks operate on collections as first-class data structures.

**Checks**:
- `SubsetCheck` (`subset`) — "Are all expected items present in the actual output?"
- `SupersetCheck` (`superset`) — "Does the actual output contain only items from the allowed set?"
- `SetEqualCheck` (`set_equal`) — "Are actual and expected the same set, regardless of order/duplicates?"
- `DisjointCheck` (`disjoint`) — "Do actual and expected share no elements?"

**Set semantics**: These are mathematical set operations. Duplicates are ignored — `["a", "a", "b"]` is treated as `{"a", "b"}`. Multiset/bag semantics (where duplicate counts matter) are out of scope.

**Field naming**: Each check uses field names that are self-documenting for its specific semantics:
- `SubsetCheck`: `collection`/`required` — "collection must contain all required items"
- `SupersetCheck`: `collection`/`allowed` — "collection must contain only allowed items"
- `SetEqualCheck`: `actual`/`expected` — "actual must equal expected" (symmetric, matches `EqualsCheck`)
- `DisjointCheck`: `collection`/`forbidden` — "collection must not contain any forbidden items"

**Element comparison**: Elements are compared using Python `==` by default. When `case_sensitive=False`, only `str` elements are lowercased for comparison; non-string elements are compared with `==` regardless. There is no implicit `str()` coercion — `1 != "1"`. To support unhashable elements (dicts, nested lists), comparison uses list iteration rather than Python `set()` construction.

**Empty collection handling**: Empty collections are valid inputs for all checks. The mathematical result applies: empty set is a subset of everything, empty sets are equal, etc.

**Files Modified**:
- `src/flex_evals/constants.py` — Add `SUBSET`, `SUPERSET`, `SET_EQUAL`, `DISJOINT` to `CheckType`
- `src/flex_evals/checks/_set_utils.py` — Private shared helper for element comparison/partitioning
- `src/flex_evals/checks/subset.py` — New `SubsetCheck`
- `src/flex_evals/checks/superset.py` — New `SupersetCheck`
- `src/flex_evals/checks/set_equal.py` — New `SetEqualCheck`
- `src/flex_evals/checks/disjoint.py` — New `DisjointCheck`
- `src/flex_evals/checks/__init__.py` — Export new checks
- `src/flex_evals/__init__.py` — Add new checks to top-level exports and `__all__`
- `tests/conftest.py` — Add new checks to `restore_standard_checks()`
- `tests/test_checks/test_subset.py` — Tests
- `tests/test_checks/test_superset.py` — Tests
- `tests/test_checks/test_set_equal.py` — Tests
- `tests/test_checks/test_disjoint.py` — Tests

**Breaking Changes**: None. Additive only.

---

## Milestone 1: Shared Helper + `SubsetCheck`

### Goal & Outcome

Implement the shared comparison helper and `SubsetCheck` — the most fundamental set operation. After this milestone:
- `_partition_elements()` helper exists for all set checks to use
- `SubsetCheck` is registered as `subset` and available for use in YAML configs and code
- Checks whether all items in `expected` exist in `actual` (i.e., `expected ⊆ actual`)
- Supports `case_sensitive` for string element comparison
- Supports `negate` which inverts the boolean result
- Returns `{'passed': bool, 'missing': list, 'found': list}` for diagnostics
- All fields support JSONPath resolution
- Exported from top-level `flex_evals` package

### Implementation Outline

**1. Add `SUBSET` to `CheckType` in `constants.py`**

```python
SUBSET = 'subset'
```

**2. Create `src/flex_evals/checks/_set_utils.py`**

A private module with a shared helper function used by all four set checks. This avoids duplicating element comparison and case normalization logic across four files.

```python
def _partition_elements(
    source: list | set,
    target: list | set,
    case_sensitive: bool,
) -> tuple[list, list]:
    """
    Partition `source` elements into those found and not found in `target`.

    Returns (found, not_found) using original (pre-normalized) values from source.
    Comparison uses Python == for non-string elements. For string elements,
    applies case-insensitive comparison when case_sensitive=False.
    Uses list iteration (not Python set()) to support unhashable elements.
    Duplicates in source are ignored — each unique element is checked once.
    """
```

The function should:
- Deduplicate `source` (preserving first occurrence of each unique element, respecting case sensitivity for strings)
- For each unique element in `source`, check if a matching element exists in `target`
- String matching: compare with `.lower()` when `case_sensitive=False`
- Non-string matching: compare with `==` regardless of `case_sensitive`
- Return `(found, not_found)` tuples of original (pre-normalized) source elements

**3. Create `src/flex_evals/checks/subset.py`**

Follow the `ContainsCheck` pattern. Key structural elements:
- `@register(CheckType.SUBSET, version='1.0.0')`
- Extends `BaseCheck`
- Fields: `actual`, `expected`, `case_sensitive` (default `True`), `negate` (default `False`)
- `@field_validator` with `_convert_to_jsonpath` on all fields
- `default_results` returns `{'passed': False, 'missing': [], 'found': []}`
- `__call__()` validates no unresolved JSONPath objects remain, then performs the check

Field type annotations:

```python
actual: list | set | JSONPath = Field(
    ...,
    description="The collection to check against (e.g., model output tags)",
)
expected: list | set | JSONPath = Field(
    ...,
    description="Items that must be present in actual",
)
case_sensitive: bool | JSONPath = Field(True, ...)
negate: bool | JSONPath = Field(False, ...)
```

Core logic in `__call__()`:

- Validate `actual` is a list or set; raise `ValidationError` if not
- Validate `expected` is a list or set; raise `ValidationError` if not (but empty is allowed)
- Call `_partition_elements(source=expected, target=actual, case_sensitive=self.case_sensitive)` to get `(found, missing)`
- `passed = len(missing) == 0` (all expected items found), then apply `negate`
- Return `{'passed': passed, 'missing': missing, 'found': found}`

**4. Export from `checks/__init__.py`, `src/flex_evals/__init__.py`, and `tests/conftest.py`**

- Add import and `__all__` entry in `checks/__init__.py`
- Add import and `__all__` entry in `src/flex_evals/__init__.py`
- Add `SubsetCheck` re-registration to `restore_standard_checks()` in `tests/conftest.py`

### Testing Strategy

Create `tests/test_checks/test_subset.py`. Follow the organizational pattern from `test_contains.py` with these test classes:

**`TestSubsetValidation`**:
- Basic creation with literal lists
- JSONPath string auto-conversion on all fields
- Default values (`case_sensitive=True`, `negate=False`)

**`TestSubsetExecution`**:
- All expected items present → passes
- Some expected items missing → fails, `missing` populated correctly
- Empty actual list, non-empty expected → fails
- Empty expected list → passes (vacuously true: empty set is subset of everything)
- Both empty → passes
- `negate=True` inverts result (present → fail, missing → pass)
- `case_sensitive=False` with mixed-case string elements
- `case_sensitive=False` does not affect non-string elements (int `1` != str `"1"`)
- Duplicate elements in actual or expected (set semantics — duplicates don't matter)
- Non-string elements (ints, mixed types) work correctly
- Dict elements in both actual and expected (unhashable element support)
- Expected is a `set` instead of a `list`
- Actual is a `set` instead of a `list`
- Single-item expected list

**`TestSubsetErrorHandling`**:
- `actual` is not a list/set (e.g., a string or int) → `ValidationError`
- `expected` is not a list/set → `ValidationError`
- Unresolved JSONPath field → `RuntimeError`

**`TestSubsetEngineIntegration`**:
- End-to-end with `EvaluationContext` and JSONPath resolution
- Verify `resolved_arguments` structure
- End-to-end via `evaluate_sync()`

---

## Milestone 2: `SupersetCheck`

### Goal & Outcome

Implement `SupersetCheck`. After this milestone:
- `SupersetCheck` is registered as `superset` and available for use
- Checks whether all items in `actual` exist in `expected` (i.e., `actual ⊆ expected`), meaning the output contains nothing unexpected
- Supports `case_sensitive` and `negate`
- Returns `{'passed': bool, 'unexpected': list, 'found': list}` for diagnostics
- All fields support JSONPath resolution
- Exported from top-level `flex_evals` package

### Implementation Outline

**1. Add `SUPERSET` to `CheckType` in `constants.py`**

```python
SUPERSET = 'superset'
```

**2. Create `src/flex_evals/checks/superset.py`**

Same structure as `SubsetCheck`. The difference is the direction of the membership check — we partition `actual` against `expected` (instead of `expected` against `actual`):

Field type annotations — identical to SubsetCheck:

```python
actual: list | set | JSONPath = Field(
    ...,
    description="The collection to check (e.g., model output tags)",
)
expected: list | set | JSONPath = Field(
    ...,
    description="The allowed set of items — actual must contain only items from this set",
)
case_sensitive: bool | JSONPath = Field(True, ...)
negate: bool | JSONPath = Field(False, ...)
```

`default_results` returns `{'passed': False, 'unexpected': [], 'found': []}`.

Core logic in `__call__()`:

- Same type validation as SubsetCheck (empty collections allowed)
- Call `_partition_elements(source=actual, target=expected, case_sensitive=self.case_sensitive)` to get `(found, unexpected)`
- `passed = len(unexpected) == 0` (no unexpected items), then apply `negate`
- Return `{'passed': passed, 'unexpected': unexpected, 'found': found}`

**3. Export from `checks/__init__.py`, `src/flex_evals/__init__.py`, and `tests/conftest.py`**

### Testing Strategy

Create `tests/test_checks/test_superset.py`. Same structure as SubsetCheck tests with adjusted semantics:

**`TestSupersetValidation`**:
- Same as SubsetCheck validation tests

**`TestSupersetExecution`**:
- All actual items in expected → passes
- Some actual items not in expected → fails, `unexpected` populated
- Empty actual list → passes (empty set is subset of anything)
- Empty expected, non-empty actual → fails (actual has unexpected items)
- Both empty → passes
- `negate=True` inverts result
- `case_sensitive=False` with mixed-case elements
- `case_sensitive=False` does not affect non-string elements
- Duplicates handled correctly
- Non-string elements
- Dict elements (unhashable)

**`TestSupersetErrorHandling`**:
- Same error conditions as SubsetCheck (invalid types, unresolved JSONPath)

**`TestSupersetEngineIntegration`**:
- End-to-end with JSONPath resolution and `evaluate_sync()`

---

## Milestone 3: `SetEqualCheck`

### Goal & Outcome

Implement `SetEqualCheck`. After this milestone:
- `SetEqualCheck` is registered as `set_equal` and available for use
- Checks whether `actual` and `expected` contain the same elements regardless of order and duplicates
- Supports `case_sensitive` and `negate`
- Returns `{'passed': bool, 'missing': list, 'unexpected': list}` for diagnostics
- All fields support JSONPath resolution
- Exported from top-level `flex_evals` package

### Implementation Outline

**1. Add `SET_EQUAL` to `CheckType` in `constants.py`**

```python
SET_EQUAL = 'set_equal'
```

**2. Create `src/flex_evals/checks/set_equal.py`**

Same structural pattern. This check runs the partition in both directions — items missing from actual AND unexpected items in actual.

Field type annotations — identical to SubsetCheck/SupersetCheck.

`default_results` returns `{'passed': False, 'missing': [], 'unexpected': []}`.

Core logic in `__call__()`:

- Same type validation (empty collections allowed)
- Call `_partition_elements(source=expected, target=actual, ...)` to get `(_, missing)`
- Call `_partition_elements(source=actual, target=expected, ...)` to get `(_, unexpected)`
- `passed = len(missing) == 0 and len(unexpected) == 0`, then apply `negate`
- Return `{'passed': passed, 'missing': missing, 'unexpected': unexpected}`

**Docstring**: Note the distinction from `EqualsCheck` — `SetEqualCheck` ignores order and duplicates (`["a", "b", "b"]` equals `["b", "a"]`), while `EqualsCheck` uses Python `==` which is order-sensitive for lists.

**3. Export from `checks/__init__.py`, `src/flex_evals/__init__.py`, and `tests/conftest.py`**

### Testing Strategy

Create `tests/test_checks/test_set_equal.py`:

**`TestSetEqualValidation`**:
- Same as previous checks

**`TestSetEqualExecution`**:
- Same elements, same order → passes
- Same elements, different order → passes
- Same elements with duplicates in one side → passes (set semantics)
- Missing items → fails, `missing` populated
- Unexpected items → fails, `unexpected` populated
- Both missing and unexpected → fails, both populated
- Both empty → passes
- `negate=True` inverts result
- `case_sensitive=False` with mixed-case elements
- `case_sensitive=False` does not affect non-string elements
- Non-string elements
- Dict elements (unhashable)

**`TestSetEqualErrorHandling`**:
- Same error conditions as previous checks

**`TestSetEqualEngineIntegration`**:
- End-to-end with JSONPath resolution and `evaluate_sync()`

---

## Milestone 4: `DisjointCheck`

### Goal & Outcome

Implement `DisjointCheck`. After this milestone:
- `DisjointCheck` is registered as `disjoint` and available for use
- Checks whether `actual` and `expected` share no elements (i.e., their intersection is empty)
- Common use case: "the output must NOT contain any of these excluded items"
- Supports `case_sensitive` and `negate`
- Returns `{'passed': bool, 'overlapping': list}` for diagnostics
- All fields support JSONPath resolution
- Exported from top-level `flex_evals` package

### Implementation Outline

**1. Add `DISJOINT` to `CheckType` in `constants.py`**

```python
DISJOINT = 'disjoint'
```

**2. Create `src/flex_evals/checks/disjoint.py`**

Same structural pattern. This check finds the intersection — any element present in both collections means the check fails.

Field type annotations:

```python
actual: list | set | JSONPath = Field(
    ...,
    description="The collection to check (e.g., model output tags)",
)
expected: list | set | JSONPath = Field(
    ...,
    description="Items that must NOT be present in actual (the excluded/forbidden set)",
)
case_sensitive: bool | JSONPath = Field(True, ...)
negate: bool | JSONPath = Field(False, ...)
```

`default_results` returns `{'passed': False, 'overlapping': []}`.

Core logic in `__call__()`:

- Same type validation (empty collections allowed)
- Call `_partition_elements(source=expected, target=actual, ...)` to get `(overlapping, _)`
- `passed = len(overlapping) == 0` (no shared elements), then apply `negate`
- Return `{'passed': passed, 'overlapping': overlapping}`

Note: `_partition_elements(source=expected, target=actual)` returns `(found, not_found)`. The `found` portion is exactly the overlap/intersection — elements from `expected` that exist in `actual`. We only need the `found` list here.

**3. Export from `checks/__init__.py`, `src/flex_evals/__init__.py`, and `tests/conftest.py`**

### Testing Strategy

Create `tests/test_checks/test_disjoint.py`:

**`TestDisjointValidation`**:
- Same as previous checks

**`TestDisjointExecution`**:
- No overlap → passes, `overlapping` is empty
- Some overlap → fails, `overlapping` populated with shared elements
- Complete overlap (same elements) → fails
- Empty actual → passes (nothing can overlap)
- Empty expected → passes (nothing forbidden)
- Both empty → passes
- `negate=True` inverts result (overlap → pass, no overlap → fail)
- `case_sensitive=False` with mixed-case elements (e.g., `actual=["Python"]`, `expected=["python"]` → overlap found)
- `case_sensitive=False` does not affect non-string elements
- Duplicates handled correctly
- Non-string elements
- Dict elements (unhashable)

**`TestDisjointErrorHandling`**:
- Same error conditions as previous checks

**`TestDisjointEngineIntegration`**:
- End-to-end with JSONPath resolution and `evaluate_sync()`
