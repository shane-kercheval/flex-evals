"""
Flexible Evaluation Protocol (FEP) - Python Implementation.

A vendor-neutral, schema-driven standard for measuring the quality of any system
that produces complex or variable outputs.
"""

from .engine import evaluate
from . import schemas
from .constants import CheckType, Status, ErrorType, SimilarityMetric
from .schemas.check import Check, CheckError, CheckResult, CheckResultMetadata
from .schemas.test_case import TestCase
from .schemas.output import Output
from .schemas.results import TestCaseResult, EvaluationRunResult

__version__ = "0.1.0"
__all__ = [
    "Check",
    "CheckError",
    "CheckResult",
    "CheckResultMetadata",
    "CheckType",
    "ErrorType",
    "EvaluationRunResult",
    "Output",
    "SimilarityMetric",
    "Status",
    "TestCase",
    "TestCaseResult",
    "evaluate",
    "schemas",
]
