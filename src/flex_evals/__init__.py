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
)
from .schemas.checks import (
    ContainsCheck, ExactMatchCheck, IsEmptyCheck, RegexCheck, RegexFlags, ThresholdCheck,
    SemanticSimilarityCheck, LLMJudgeCheck, CustomFunctionCheck, ThresholdConfig,
)
from .schemas.test_case import TestCase
from .schemas.output import Output
from .schemas.results import TestCaseResult, EvaluationRunResult

__version__ = "0.1.0"
__all__ = [
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
    "LLMJudgeCheck",
    "Output",
    "RegexCheck",
    "RegexFlags",
    "SchemaCheck",
    "SemanticSimilarityCheck",
    "SimilarityMetric",
    "Status",
    "TestCase",
    "TestCaseResult",
    "ThresholdCheck",
    "ThresholdConfig",
    "evaluate",
    "schemas",
]
