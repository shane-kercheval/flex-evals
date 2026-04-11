"""Disjoint Check - checks if a collection and a forbidden set share no elements."""

from typing import Any
from pydantic import field_validator, Field

from .base import BaseCheck, JSONPath, _convert_to_jsonpath
from ..registry import register
from ..exceptions import ValidationError
from ..constants import CheckType
from ._set_utils import _partition_elements


@register(CheckType.DISJOINT, version='1.0.0')
class DisjointCheck(BaseCheck):
    """
    Checks whether the collection and the forbidden set share no elements.

    Passes when the intersection is empty (collection ∩ forbidden = ∅).
    Common use case: "the output must NOT contain any of these excluded items."
    Duplicates are ignored (set semantics). Elements are compared using Python ==
    by default; when case_sensitive=False, only string elements are lowercased
    for comparison.
    """

    collection: list | set | JSONPath = Field(
        ...,
        description="The collection to check (e.g., model output tags)",
    )
    forbidden: list | set | JSONPath = Field(
        ...,
        description="Items that must NOT be present in collection",
    )
    case_sensitive: bool | JSONPath = Field(
        True,
        description="Whether string element matching is case-sensitive",
    )
    negate: bool | JSONPath = Field(
        False,
        description="If true, inverts the result (passes when overlap exists)",
    )

    @field_validator('collection', 'forbidden', 'case_sensitive', 'negate', mode='before')
    @classmethod
    def convert_jsonpath(cls, value: object) -> object | JSONPath:
        """Convert JSONPath-like strings to JSONPath objects."""
        return _convert_to_jsonpath(value)

    @property
    def default_results(self) -> dict[str, Any]:
        """Return default results structure for disjoint checks on error."""
        return {'passed': False, 'overlapping': []}

    def __call__(self) -> dict[str, Any]:
        """
        Execute disjoint check using resolved Pydantic fields.

        Returns:
            Dictionary with 'passed' (bool) and 'overlapping' (list) keys.

        Raises:
            RuntimeError: If any field contains unresolved JSONPath objects
        """
        if isinstance(self.collection, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'collection' field: {self.collection}")
        if isinstance(self.forbidden, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'forbidden' field: {self.forbidden}")
        if isinstance(self.case_sensitive, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'case_sensitive' field: {self.case_sensitive}")  # noqa: E501
        if isinstance(self.negate, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'negate' field: {self.negate}")

        if not isinstance(self.collection, list | set):
            raise ValidationError(
                "Disjoint check 'collection' must be a list or set, "
                f"got {type(self.collection).__name__}",
            )
        if not isinstance(self.forbidden, list | set):
            raise ValidationError(
                "Disjoint check 'forbidden' must be a list or set, "
                f"got {type(self.forbidden).__name__}",
            )

        overlapping, _ = _partition_elements(
            source=self.forbidden,
            target=self.collection,
            case_sensitive=self.case_sensitive,
        )

        passed = len(overlapping) == 0
        if self.negate:
            passed = not passed

        return {'passed': passed, 'overlapping': overlapping}
