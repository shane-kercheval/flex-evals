# Combined Check Architecture Implementation Plan

## Overview
Replace the dual schema/implementation class architecture with a single combined approach where check classes inherit from both validation and execution base classes. This eliminates code duplication while providing type safety and better developer experience.

## Design Principles
- **No backwards compatibility concerns** - implement best practices from scratch
- **Single source of truth** - one class per check type
- **Type safety first** - Pydantic validation at construction time
- **Clean API** - intuitive for both YAML configs and direct instantiation
- **Consistent structure** - all checks follow same patterns

## Phase 1: Base Architecture Setup

### 1.1 Reorganize Base Classes (`src/flex_evals/checks/base.py`)
- **Move** `JSONPathValidatedModel` from `schemas/check.py` to `checks/base.py`
- **Enhance** existing `BaseCheck` class to inherit from `JSONPathValidatedModel`
- **Enhance** existing `BaseAsyncCheck` class to inherit from `JSONPathValidatedModel`
- **Add** required class attributes (`VERSION`, `CHECK_TYPE`)
- **Add** `to_arguments()` method for engine integration
- **Keep** `EvaluationContext` in same file

**Result**: `checks/base.py` becomes single source for all check base functionality

### 1.2 Update Engine Integration (`src/flex_evals/engine.py`)
- **Modify** `_convert_check_input()` to handle both `Check` dataclass and `BaseCheck`/`BaseAsyncCheck` instances
- **Update** execution functions to work with combined check instances
- **Simplify** type signatures: `list[Check | BaseCheck | BaseAsyncCheck]`
- **Remove** dependency on separate schema classes

### 1.3 Update Imports and Exports
- **Update** `checks/__init__.py` to export base classes
- **Clean up** circular imports between schemas and checks
- **Standardize** import paths for consistency

## Phase 2: Convert Standard Checks

### 2.1 Reference Implementation (ExactMatch)
Convert `exact_match` as the canonical example:
- **Create** `checks/exact_match.py` with combined `ExactMatchCheck(BaseCheck)`
- **Include** Pydantic field definitions with validation
- **Include** execution logic in `__call__` method
- **Test** end-to-end functionality

### 2.2 Convert Remaining Sync Checks
Transform each standard sync check:
- `checks/contains.py` → `ContainsCheck(BaseCheck)`
- `checks/equals.py` → `EqualsCheck(BaseCheck)`
- `checks/regex.py` → `RegexCheck(BaseCheck)`
- `checks/threshold.py` → `ThresholdCheck(BaseCheck)`
- `checks/is_empty.py` → `IsEmptyCheck(BaseCheck)`
- `checks/attribute_exists.py` → `AttributeExistsCheck(BaseCheck)`

### 2.3 Convert Async Checks
Transform async checks to use `BaseAsyncCheck`:
- `checks/semantic_similarity.py` → `SemanticSimilarityCheck(BaseAsyncCheck)`
- `checks/llm_judge.py` → `LLMJudgeCheck(BaseAsyncCheck)`

### 2.4 Consistent Field Patterns
Ensure all checks follow standard field organization:
```python
class CheckName(BaseCheck):
    VERSION = "1.0.0"
    CHECK_TYPE = CheckType.CHECK_NAME
    
    # Input fields (what to check)
    input_field: str = OptionalJSONPath("Description")
    
    # Configuration fields (how to check)
    config_field: bool = Field(True, description="Config description")
    negate: bool = Field(False, description="Invert result")
```

## Phase 3: Test Consolidation

### 3.1 Merge Schema and Implementation Tests
- **Combine** `tests/test_schemas_checks.py` and `tests/test_standard_checks.py`
- **Organize** by check type, not by schema vs implementation
- **Structure**: `tests/test_checks/test_exact_match.py` etc.

### 3.2 Standardize Test Patterns
Each check test should cover:
- **Validation**: Field validation at construction time
- **Execution**: Direct `__call__` method testing
- **Integration**: Engine execution testing
- **Edge cases**: JSONPath resolution, error handling

### 3.3 Engine Integration Tests
- **Update** `tests/test_evaluation_engine.py` to use combined checks
- **Remove** tests specific to old dual-class approach
- **Add** tests for mixed `Check` dataclass and `BaseCheck` instance usage

## Phase 4: Registry and Utility Updates

### 4.1 Simplify Registry
- **Update** registry to work with combined check classes
- **Ensure** version tracking works with class attributes
- **Consider** auto-registration decorator improvements

### 4.2 Schema Generation
- **Update** `schema_generator.py` to work with combined classes
- **Generate** schemas from Pydantic fields instead of separate schema classes
- **Maintain** JSON schema output compatibility

### 4.3 JSONPath Utilities
- **Keep** `OptionalJSONPath` and `RequiredJSONPath` helpers
- **Ensure** validation logic works with combined approach
- **Move** utilities to appropriate location in `checks/base.py`

## Phase 5: Cleanup and Documentation

### 5.1 Remove Legacy Code
- **Delete** all separate schema classes in `schemas/checks/`
- **Delete** old implementation classes in `checks/standard/`
- **Clean up** unused imports and utilities
- **Remove** `to_check()` conversion methods

### 5.2 Update Documentation
- **Revise** all examples to show combined approach
- **Update** README with new instantiation patterns
- **Remove** references to "schema vs implementation classes"
- **Add** migration examples for custom check authors

### 5.3 File Structure Cleanup
Final structure:
```
src/flex_evals/
├── checks/
│   ├── __init__.py              # Export all check classes
│   ├── base.py                  # BaseCheck, BaseAsyncCheck, utilities
│   ├── exact_match.py           # ExactMatchCheck(BaseCheck)
│   ├── contains.py              # ContainsCheck(BaseCheck)
│   ├── semantic_similarity.py   # SemanticSimilarityCheck(BaseAsyncCheck)
│   └── ...
├── schemas/
│   ├── check.py                 # Keep Check dataclass only
│   └── ...
└── tests/
    ├── test_checks/
    │   ├── test_exact_match.py
    │   └── ...
    └── test_evaluation_engine.py
```

## Phase 6: Validation and Polish

### 6.1 Comprehensive Testing
- **Run** full test suite to ensure no regressions
- **Test** both YAML-loaded checks and direct instantiation
- **Validate** async/sync separation still works correctly
- **Check** performance characteristics

### 6.2 Integration Validation
- **Test** with real evaluation scenarios
- **Validate** JSONPath resolution works correctly
- **Ensure** error messages are clear and helpful
- **Check** memory usage and performance

### 6.3 API Consistency Review
- **Ensure** all checks follow consistent patterns
- **Validate** type hints are accurate and helpful
- **Check** IDE experience (autocomplete, type checking)
- **Review** import patterns for simplicity

## Success Criteria

1. **Single class per check** - no duplicate schema/implementation files
2. **Type safety** - full Pydantic validation at construction
3. **Clean API** - intuitive instantiation and usage
4. **Zero regression** - all existing functionality preserved
5. **Better DX** - improved developer experience over previous approach
6. **Performance** - no significant performance degradation
7. **Maintainability** - reduced code complexity and file count

## Risk Mitigation

- **Incremental conversion** - convert checks one at a time
- **Comprehensive testing** - validate each step
- **Keep reference implementation** - use ExactMatch as canonical example
- **Document patterns** - establish clear conventions early

This plan transforms the architecture while maintaining all functionality and significantly improving the developer experience.