"""Test configuration for package."""
from flex_evals.checks.extended.custom_function import CustomFunctionCheck
from flex_evals.checks.extended.llm_judge import LlmJudgeCheck
from flex_evals.checks.extended.semantic_similarity import SemanticSimilarityCheck
from flex_evals.checks.standard.contains import ContainsCheck
from flex_evals.checks.standard.exact_match import ExactMatchCheck
from flex_evals.checks.standard.is_empty import IsEmptyCheck
from flex_evals.checks.standard.regex import RegexCheck
from flex_evals.checks.standard.threshold import ThresholdCheck
from flex_evals import CheckType
from flex_evals.registry import register


def restore_standard_checks():
    """Restore standard checks to registry after clearing."""
    # Re-register them (they have @register decorators but need to be called again)
    register(CheckType.EXACT_MATCH, version="1.0.0")(ExactMatchCheck)
    register(CheckType.CONTAINS, version="1.0.0")(ContainsCheck)
    register(CheckType.REGEX, version="1.0.0")(RegexCheck)
    register(CheckType.IS_EMPTY, version="1.0.0")(IsEmptyCheck)
    register(CheckType.THRESHOLD, version="1.0.0")(ThresholdCheck)
    register(CheckType.CUSTOM_FUNCTION, version="1.0.0")(CustomFunctionCheck)
    register(CheckType.SEMANTIC_SIMILARITY, version="1.0.0")(SemanticSimilarityCheck)
    register(CheckType.LLM_JUDGE, version="1.0.0")(LlmJudgeCheck)
