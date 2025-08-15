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
from .schemas.check import Check, CheckError, CheckResult
from .schemas.test_case import TestCase
from .schemas.output import Output
from .schemas.results import TestCaseResult, EvaluationRunResult
from .schemas.experiments import ExperimentMetadata
from .schema_generator import generate_checks_schema, generate_check_schema

__version__ = "0.1.0"
__all__ = [
    "Check",
    "CheckError",
    "CheckResult",
    "CheckType",
    "ErrorType",
    "EvaluationRunResult",
    "ExperimentMetadata",
    "Output",
    "SimilarityMetric",
    "Status",
    "TestCase",
    "TestCaseResult",
    "evaluate",
    "generate_check_schema",
    "generate_checks_schema",
]
