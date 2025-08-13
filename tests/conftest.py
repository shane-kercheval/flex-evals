"""Test configuration for package."""
from flex_evals.checks.extended.custom_function import CustomFunctionCheck_v1_0_0
from flex_evals.checks.extended.llm_judge import LlmJudgeCheck_v1_0_0
from flex_evals.checks.extended.semantic_similarity import SemanticSimilarityCheck_v1_0_0
from flex_evals.checks.standard.attribute_exists import AttributeExistsCheck_v1_0_0
from flex_evals.checks.standard.contains import ContainsCheck_v1_0_0
from flex_evals.checks.standard.exact_match import ExactMatchCheck_v1_0_0
from flex_evals.checks.standard.is_empty import IsEmptyCheck_v1_0_0
from flex_evals.checks.standard.regex import RegexCheck_v1_0_0
from flex_evals.checks.standard.threshold import ThresholdCheck_v1_0_0
from flex_evals import CheckType
from flex_evals.registry import register


def restore_standard_checks():
    """Restore standard checks to registry after clearing."""
    # Re-register them (they have @register decorators but need to be called again)
    register(CheckType.ATTRIBUTE_EXISTS, version="1.0.0")(AttributeExistsCheck_v1_0_0)
    register(CheckType.EXACT_MATCH, version="1.0.0")(ExactMatchCheck_v1_0_0)
    register(CheckType.CONTAINS, version="1.0.0")(ContainsCheck_v1_0_0)
    register(CheckType.REGEX, version="1.0.0")(RegexCheck_v1_0_0)
    register(CheckType.IS_EMPTY, version="1.0.0")(IsEmptyCheck_v1_0_0)
    register(CheckType.THRESHOLD, version="1.0.0")(ThresholdCheck_v1_0_0)
    register(CheckType.CUSTOM_FUNCTION, version="1.0.0")(CustomFunctionCheck_v1_0_0)
    register(CheckType.SEMANTIC_SIMILARITY, version="1.0.0")(SemanticSimilarityCheck_v1_0_0)
    register(CheckType.LLM_JUDGE, version="1.0.0")(LlmJudgeCheck_v1_0_0)
