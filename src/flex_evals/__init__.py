"""
Flexible Evaluation Protocol (FEP) - Python Implementation.

A vendor-neutral, schema-driven standard for measuring the quality of any system
that produces complex or variable outputs.
"""

from .engine import evaluate
from . import schemas
from .constants import CheckType, Status, ErrorType, SimilarityMetric
# Import checks to trigger registration of standard checks
from . import checks  # noqa: F401
from .schemas.check import (
    Check, CheckError, CheckResult, SchemaCheck,
    JSONPathBehavior, RequiredJSONPath, OptionalJSONPath, JSONPathValidatedModel,
    get_jsonpath_behavior, validate_jsonpath, is_jsonpath_expression,
)
from .schemas.checks import (
    AttributeExistsCheck, ContainsCheck, ExactMatchCheck, IsEmptyCheck, RegexCheck, RegexFlags,
    ThresholdCheck, SemanticSimilarityCheck, LLMJudgeCheck, CustomFunctionCheck, ThresholdConfig,
)
from .schemas.test_case import TestCase
from .schemas.output import Output
from .schemas.results import TestCaseResult, EvaluationRunResult
from .schema_generator import generate_checks_schema, generate_check_schema

__version__ = "0.1.0"
__all__ = [
    "AttributeExistsCheck",
    "Check",
    "CheckError",
    "CheckResult",
    "CheckType",
    "ContainsCheck",
    "CustomFunctionCheck",
    "ErrorType",
    "EvaluationRunResult",
    "ExactMatchCheck",
    "IsEmptyCheck",
    "JSONPathBehavior",
    "JSONPathValidatedModel",
    "LLMJudgeCheck",
    "OptionalJSONPath",
    "Output",
    "RegexCheck",
    "RegexFlags",
    "RequiredJSONPath",
    "SchemaCheck",
    "SemanticSimilarityCheck",
    "SimilarityMetric",
    "Status",
    "TestCase",
    "TestCaseResult",
    "ThresholdCheck",
    "ThresholdConfig",
    "evaluate",
    "generate_check_schema",
    "generate_checks_schema",
    "get_jsonpath_behavior",
    "is_jsonpath_expression",
    "schemas",
    "validate_jsonpath",
]
