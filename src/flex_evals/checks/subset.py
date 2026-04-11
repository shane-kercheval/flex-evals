"""Subset Check - checks if all expected items are present in a collection."""

from typing import Any
from pydantic import field_validator, Field

from .base import BaseCheck, JSONPath, _convert_to_jsonpath
from ..registry import register
from ..exceptions import ValidationError
from ..constants import CheckType
from ._set_utils import _partition_elements


@register(CheckType.SUBSET, version='1.0.0')
class SubsetCheck(BaseCheck):
    """
    Checks whether all expected items are present in the actual collection.

    Performs a set membership test: expected ⊆ actual. Duplicates are ignored
    (set semantics). Elements are compared using Python == by default; when
    case_sensitive=False, only string elements are lowercased for comparison.
    """

    actual: list | set | JSONPath = Field(
        ...,
        description="The collection to check against (e.g., model output tags)",
    )
    expected: list | set | JSONPath = Field(
        ...,
        description="Items that must be present in actual",
    )
    case_sensitive: bool | JSONPath = Field(
        True,
        description="Whether string element matching is case-sensitive",
    )
    negate: bool | JSONPath = Field(
        False,
        description="If true, inverts the result (passes when not all expected items are found)",
    )

    @field_validator('actual', 'expected', 'case_sensitive', 'negate', mode='before')
    @classmethod
    def convert_jsonpath(cls, value: object) -> object | JSONPath:
        """Convert JSONPath-like strings to JSONPath objects."""
        return _convert_to_jsonpath(value)

    @property
    def default_results(self) -> dict[str, Any]:
        """Return default results structure for subset checks on error."""
        return {'passed': False, 'missing': [], 'found': []}

    def __call__(self) -> dict[str, Any]:
        """
        Execute subset check using resolved Pydantic fields.

        Returns:
            Dictionary with 'passed' (bool), 'missing' (list), and 'found' (list) keys.

        Raises:
            RuntimeError: If any field contains unresolved JSONPath objects
        """
        if isinstance(self.actual, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'actual' field: {self.actual}")
        if isinstance(self.expected, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'expected' field: {self.expected}")
        if isinstance(self.case_sensitive, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'case_sensitive' field: {self.case_sensitive}")  # noqa: E501
        if isinstance(self.negate, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'negate' field: {self.negate}")

        if not isinstance(self.actual, list | set):
            raise ValidationError(
                f"Subset check 'actual' must be a list or set, got {type(self.actual).__name__}",
            )
        if not isinstance(self.expected, list | set):
            raise ValidationError(
                "Subset check 'expected' must be a list or set, "
                f"got {type(self.expected).__name__}",
            )

        found, missing = _partition_elements(
            source=self.expected,
            target=self.actual,
            case_sensitive=self.case_sensitive,
        )

        passed = len(missing) == 0
        if self.negate:
            passed = not passed

        return {'passed': passed, 'missing': missing, 'found': found}
