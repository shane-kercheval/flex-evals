# JSONPath Performance Refactor

**Date:** 2025-10-25
**Impact:** 109x performance improvement for sync check evaluations
**Status:** Complete

## Problem Identified

The `test_sync_checks_have_no_async_overhead` performance test was taking **4.7 seconds** for 200 test cases, far exceeding the expected threshold. This was entirely due to redundant JSONPath parsing.

### Root Cause Analysis

Using `cProfile`, we discovered that 99.3% of execution time (4.69s out of 4.72s) was spent parsing JSONPath expressions. The system was parsing the same 2 unique JSONPath patterns **802 times** instead of once each:

- `"$.output.value.result"` - parsed 401 times
- `"$.test_case.input.value"` - parsed 401 times

**Why 802 parses for 2 unique patterns?**

JSONPath parsing occurred in **two separate code paths** that didn't share results:

1. **Validation Path** (during Pydantic field creation):
   - `validate_jsonpath()` in `base.py` called `jsonpath_ng.parse()` directly
   - No caching mechanism
   - Called 402 times (2 patterns ◊ 200 test cases + 2 initial validations)

2. **Resolution Path** (during check execution):
   - `JSONPathResolver._cache` had instance-based caching
   - But validation had already parsed the same expressions and thrown away the results
   - Called 400 times (2 patterns ◊ 200 test cases)

The validation and resolution paths never shared their parsed expressions, resulting in double parsing.

## Solution: Consolidated Module-Level Caching

### Architecture Changes

**Removed:**
- `JSONPathResolver` class and instance-based caching
- `get_shared_resolver()` singleton pattern
- Duplicate `validate_jsonpath()` function in `base.py`
- Unused `is_jsonpath_expression()` function

**Created in `jsonpath_resolver.py`:**
```python
@lru_cache(maxsize=128)
def parse_jsonpath_cached(expression: str) -> Any:
    """Single source of truth for JSONPath parsing with @lru_cache."""
    return jsonpath_parse(expression)

def validate_jsonpath(expression: str) -> bool:
    """Validates using parse_jsonpath_cached() - shares cache with resolution."""
    if not isinstance(expression, str) or not expression.startswith('$'):
        return False
    try:
        parse_jsonpath_cached(expression)  # ê Cached!
        return True
    except (JSONPathNGError, Exception):
        return False

def resolve_argument(value: object, context: dict[str, Any]) -> dict[str, Any]:
    """Resolves JSONPath or literal using parse_jsonpath_cached() - shares cache."""
    if isinstance(value, str) and is_jsonpath(value):
        jsonpath_expr = parse_jsonpath_cached(value)  # ê Same cache!
        # ... apply to context and return
```

**Key Design Principles:**
1. **Single cached parser** - All code paths use `parse_jsonpath_cached()`
2. **Module-level functions** - No unnecessary class abstractions
3. **Standard library solution** - Uses `@lru_cache` instead of custom caching
4. **Shared cache** - Validation and resolution use the same cache

### Code Changes

**Files Modified:**
1. `src/flex_evals/utils/jsonpath_resolver.py` - Converted class to module-level functions
2. `src/flex_evals/checks/base.py` - Updated to use module-level functions directly
3. `tests/test_jsonpath_resolver.py` - Updated test to use module-level functions
4. `tests/test_jsonpath_validation.py` - Updated imports
5. `tests/test_evaluation_engine_optimization.py` - Updated performance threshold to <1s

**Migration Pattern:**
```python
# Before (instance-based, no sharing)
self._resolver = get_shared_resolver()
self._resolver.resolve_argument(value, context)

# After (module-level, automatic sharing)
resolve_argument(value, context)  # Uses @lru_cache automatically
```

## Performance Results

### Profiling Data

**Before optimization:**
```
Total Time: 4.72 seconds
JSONPath Parse Calls: 802
  - parse() time: 4.69s (99.3% of total)
  - other operations: 0.03s (0.7%)
Per-test-case time: 23.6ms
```

**After optimization:**
```
Total Time: 0.043 seconds
JSONPath Parse Calls: 2 (one per unique pattern!)
  - parse() time: 0.017s (39.5% of total)
  - other operations: 0.026s (60.5%)
Per-test-case time: 0.2ms
```

**Improvements:**
- **109x faster** overall (4.72s í 0.043s)
- **401x fewer** parse calls (802 í 2)
- **276x faster** parsing time (4.69s í 0.017s)

### Test Results

- `test_sync_checks_have_no_async_overhead`: **Passes in 70ms** (was 4700ms)
- All 40 JSONPath validation tests: **Pass**
- All 15 optimization tests: **Pass**

## Why This Design Is Better

### Before: Complex Class-Based Pattern
```
base.py:
  - validate_jsonpath() í parses directly, no cache

jsonpath_resolver.py:
  - JSONPathResolver class with instance cache
  - get_shared_resolver() singleton
  - resolve_argument() uses instance cache

Result: Two separate code paths, two separate parse operations
```

### After: Simple Module-Level Functions
```
jsonpath_resolver.py:
  - parse_jsonpath_cached() with @lru_cache
  - validate_jsonpath() uses parse_jsonpath_cached()
  - resolve_argument() uses parse_jsonpath_cached()

Result: Single code path, single parse operation
```

### Benefits
1. **No class overhead** - Functions are simpler than classes for stateless operations
2. **Automatic cache sharing** - `@lru_cache` is module-level, shared everywhere
3. **Standard library** - Uses Python's built-in `@lru_cache` instead of custom caching
4. **Thread-safe** - `@lru_cache` is thread-safe by default
5. **Memory-bounded** - `maxsize=128` prevents unbounded growth
6. **Less code** - Removed ~200 lines of class boilerplate

## Technical Details

### Why Keep `is_jsonpath()` Heuristic?

Even with caching, the cheap prefix check (`value.startswith("$.")`) is valuable:

**For validation:**
- Quickly rejects obvious non-JSONPath values without parse attempts
- Example: `validate_jsonpath("hello")` í returns False immediately

**For resolution:**
- Prevents parse attempts on literal values
- Critical because `@lru_cache` doesn't cache exceptions
- Without it: literal values like `expected="hello"` would try to parse 200 times and fail 200 times

### Cache Characteristics

- **What's cached:** Parsed JSONPath expression objects (NOT resolved values)
- **Cache key:** JSONPath string (e.g., `"$.output.value.result"`)
- **Cache size:** 128 unique expressions (typical usage: 2-10)
- **Cache scope:** Module-level (shared across all code)
- **Thread safety:** Yes (expressions are stateless, `@lru_cache` is thread-safe)

### Real-World Impact

For a typical evaluation with:
- 1000 test cases
- 2 JSONPath patterns in checks

**Before:** 2000+ parses ◊ 2.2ms = ~4.4 seconds
**After:** 2 parses ◊ 2.2ms = ~4.4ms

**1000x improvement** in parsing overhead.

## Lessons Learned

1. **Profile before optimizing** - The issue wasn't obvious until profiling revealed 99.3% time in parsing
2. **Avoid premature abstraction** - The `JSONPathResolver` class added complexity without benefit
3. **Share caches across code paths** - Validation and resolution should use the same parsed expressions
4. **Use stdlib when possible** - `@lru_cache` is simpler and better than custom caching
5. **Design for simplicity** - Module-level functions are cleaner than singletons for stateless operations

## Related Files

- Profiling script: `profile_sync_checks.py` (can be deleted)
- Performance test: `tests/test_evaluation_engine_optimization.py::test_sync_checks_have_no_async_overhead`
