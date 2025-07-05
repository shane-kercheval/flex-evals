"""CustomFunctionCheck schema class for type-safe custom function check definitions."""

from typing import Any
from collections.abc import Callable

from pydantic import Field

from ...constants import CheckType
from ..check import SchemaCheck


class CustomFunctionCheck(SchemaCheck):
    """
    Type-safe schema for custom function check.

    Executes user-provided validation functions for flexible argument checking.

    Fields:
    - validation_function: User-provided function or string function definition
    - function_args: Arguments to pass to validation_function (JSONPath expressions allowed)
    - version: Optional version string for the check
    """

    validation_function: Callable | str = Field(..., description="User-provided function or string function definition")  # noqa: E501
    function_args: dict[str, Any] = Field(..., description="Arguments to pass to validation_function (JSONPath expressions allowed)")  # noqa: E501

    model_config = {"arbitrary_types_allowed": True}  # noqa: RUF012

    @property
    def check_type(self) -> CheckType:
        """Return the CheckType for this check."""
        return CheckType.CUSTOM_FUNCTION

