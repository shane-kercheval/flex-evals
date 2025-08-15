"""
Flexible Evaluation Protocol (FEP) - Python Implementation.

A vendor-neutral, schema-driven standard for measuring the quality of any system
that produces complex or variable outputs.
"""

from .engine import evaluate
from .constants import CheckType, Status, ErrorType, SimilarityMetric
# Import checks to trigger registration of standard checks
from . import checks  # noqa: F401
from .schemas.check import Check, CheckError, CheckResult
from .schemas.test_case import TestCase
from .schemas.output import Output
from .schemas.results import (
    TestCaseResult,
    EvaluationRunResult,
    TestCaseSummary,
    EvaluationSummary,
)
from .schemas.experiments import ExperimentMetadata
from .schema_generator import generate_checks_schema, generate_check_schema

# Import individual check classes for direct end-user access
from .checks import (
    AttributeExistsCheck,
    BaseAsyncCheck,
    BaseCheck,
    ContainsCheck,
    CustomFunctionCheck,
    EqualsCheck,
    EvaluationContext,
    ExactMatchCheck,
    IsEmptyCheck,
    LLMJudgeCheck,
    RegexCheck,
    RegexFlags,
    SemanticSimilarityCheck,
    ThresholdCheck,
    get_check_class,
    list_registered_checks,
    register,
)

# Import useful classes for advanced users
from .checks.base import JSONPath
from .exceptions import CheckExecutionError, FlexEvalsError, JSONPathError, ValidationError

__version__ = "0.1.0"
__all__ = [
    # Check classes
    "AttributeExistsCheck",
    "BaseAsyncCheck",
    "BaseCheck",
    # Schema classes
    "Check",
    "CheckError",
    "CheckExecutionError",
    "CheckResult",
    # Constants and enums
    "CheckType",
    "ContainsCheck",
    "CustomFunctionCheck",
    "EqualsCheck",
    "ErrorType",
    # Check system utilities
    "EvaluationContext",
    "EvaluationRunResult",
    "EvaluationSummary",
    "ExactMatchCheck",
    "ExperimentMetadata",
    "FlexEvalsError",
    "IsEmptyCheck",
    "JSONPath",
    "JSONPathError",
    "LLMJudgeCheck",
    "Output",
    "RegexCheck",
    "RegexFlags",
    "SemanticSimilarityCheck",
    "SimilarityMetric",
    "Status",
    "TestCase",
    "TestCaseResult",
    "TestCaseSummary",
    "ThresholdCheck",
    "ValidationError",
    # Core evaluation functionality
    "evaluate",
    # Schema generators
    "generate_check_schema",
    "generate_checks_schema",
    # Registry functions
    "get_check_class",
    "list_registered_checks",
    "register",
]
