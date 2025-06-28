"""
Extended check implementations for FEP.

Imports all extended checks to trigger their registration.
"""

from .semantic_similarity import SemanticSimilarityCheck
from .llm_judge import LlmJudgeCheck

__all__ = [
    "LlmJudgeCheck",
    "SemanticSimilarityCheck",
]
