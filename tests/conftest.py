"""Test configuration for package."""
from flex_evals import (
    AttributeExistsCheck,
    ContainsCheck,
    CustomFunctionCheck,
    EqualsCheck,
    ExactMatchCheck,
    IsEmptyCheck,
    LLMJudgeCheck,
    RegexCheck,
    SemanticSimilarityCheck,
    ThresholdCheck,
    CheckType,
    register,
)

# Import essential test utility combined check classes
from tests.schemas_for_test_checks import (  # noqa: F401
    TestCheck, TestAsyncCheck, TestFailingCheck,
    # Additional test utilities will be imported as needed
)


def restore_standard_checks():
    """Restore standard checks to registry after clearing."""
    # Re-register them (they have @register decorators but need to be called again)
    register(CheckType.ATTRIBUTE_EXISTS, version="1.0.0")(AttributeExistsCheck)
    register(CheckType.CONTAINS, version="1.0.0")(ContainsCheck)
    register(CheckType.CUSTOM_FUNCTION, version="1.0.0")(CustomFunctionCheck)
    register(CheckType.EQUALS, version="1.0.0")(EqualsCheck)
    register(CheckType.EXACT_MATCH, version="1.0.0")(ExactMatchCheck)
    register(CheckType.IS_EMPTY, version="1.0.0")(IsEmptyCheck)
    register(CheckType.LLM_JUDGE, version="1.0.0")(LLMJudgeCheck)
    register(CheckType.REGEX, version="1.0.0")(RegexCheck)
    register(CheckType.SEMANTIC_SIMILARITY, version="1.0.0")(SemanticSimilarityCheck)
    register(CheckType.THRESHOLD, version="1.0.0")(ThresholdCheck)
