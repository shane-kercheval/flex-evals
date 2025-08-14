"""LLMJudgeCheck schema class for type-safe LLM judge check definitions."""

from typing import Any, ClassVar, TypeVar
from collections.abc import Callable

from pydantic import BaseModel, Field

from ...constants import CheckType
from ..check import SchemaCheck

# Type variable for the response format model
T = TypeVar('T', bound=BaseModel)


class LLMJudgeCheck(SchemaCheck):
    """Uses an LLM to evaluate outputs against complex, nuanced criteria."""

    VERSION: ClassVar[str] = "1.0.0"
    CHECK_TYPE: ClassVar[CheckType] = CheckType.LLM_JUDGE

    prompt: str = Field(..., min_length=1, description="Prompt template with optional {{$.jsonpath}} placeholders")  # noqa: E501
    response_format: type[BaseModel] = Field(..., description="Pydantic model class defining expected LLM response structure")  # noqa: E501
    llm_function: Callable[[str, type[BaseModel]], tuple[BaseModel, dict[str, Any]]] = Field(..., description="Function to call LLM")  # noqa: E501

    model_config = {"arbitrary_types_allowed": True}  # noqa: RUF012

