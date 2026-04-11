"""
Check implementations for FEP.

This module provides both built-in standard checks and the infrastructure
for registering custom checks.
"""

from ..registry import (
    register,
    get_check_class,
    list_registered_checks,
)
from .base import BaseCheck, BaseAsyncCheck, EvaluationContext

# Import all combined checks to trigger registration
from .attribute_exists import AttributeExistsCheck
from .contains import ContainsCheck
from .custom_function import CustomFunctionCheck
from .equals import EqualsCheck
from .exact_match import ExactMatchCheck
from .is_empty import IsEmptyCheck
from .llm_judge import LLMJudgeCheck
from .regex import RegexCheck, RegexFlags
from .schema import SchemaValidationCheck
from .semantic_similarity import SemanticSimilarityCheck
from .set_equal import SetEqualCheck
from .subset import SubsetCheck
from .superset import SupersetCheck
from .threshold import ThresholdCheck

__all__ = [
    # Combined check classes
    "AttributeExistsCheck",
    "BaseAsyncCheck",
    "BaseCheck",
    "ContainsCheck",
    "CustomFunctionCheck",
    "EqualsCheck",
    "EvaluationContext",
    "ExactMatchCheck",
    "IsEmptyCheck",
    "LLMJudgeCheck",
    "RegexCheck",
    "RegexFlags",
    "SchemaValidationCheck",
    "SemanticSimilarityCheck",
    "SetEqualCheck",
    "SubsetCheck",
    "SupersetCheck",
    "ThresholdCheck",
    # Registry functions
    "get_check_class",
    "list_registered_checks",
    "register",
]
