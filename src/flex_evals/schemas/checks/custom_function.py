"""CustomFunctionCheck schema class for type-safe custom function check definitions."""

from typing import Any, ClassVar
from collections.abc import Callable

from pydantic import Field

from ...constants import CheckType
from ..check import SchemaCheck


class CustomFunctionCheck(SchemaCheck):
    """Executes user-provided python validation functions."""

    VERSION: ClassVar[str] = "1.0.0"
    CHECK_TYPE: ClassVar[CheckType] = CheckType.CUSTOM_FUNCTION

    validation_function: Callable | str = Field(..., description="User-provided function or string function definition")  # noqa: E501
    function_args: dict[str, Any] = Field(..., description="Arguments to pass to validation_function (JSONPath expressions allowed)")  # noqa: E501

    model_config = {"arbitrary_types_allowed": True}  # noqa: RUF012

