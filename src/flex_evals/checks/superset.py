"""Superset Check - checks if the collection contains only allowed items."""

from typing import Any
from pydantic import field_validator, Field

from .base import BaseCheck, JSONPath, _convert_to_jsonpath
from ..registry import register
from ..exceptions import ValidationError
from ..constants import CheckType
from ._set_utils import _partition_elements


@register(CheckType.SUPERSET, version='1.0.0')
class SupersetCheck(BaseCheck):
    """
    Checks whether the collection contains only items from the allowed set.

    Performs a set membership test: collection ⊆ allowed. Passes when there are
    no unexpected items in collection. Duplicates are ignored (set semantics).
    Elements are compared using Python == by default; when case_sensitive=False,
    only string elements are lowercased for comparison.
    """

    collection: list | set | JSONPath = Field(
        ...,
        description="The collection to check (e.g., model output tags)",
    )
    allowed: list | set | JSONPath = Field(
        ...,
        description="The allowed set of items — collection must contain only items from this set",
    )
    case_sensitive: bool | JSONPath = Field(
        True,
        description="Whether string element matching is case-sensitive",
    )
    negate: bool | JSONPath = Field(
        False,
        description="If true, inverts the result",
    )

    @field_validator('collection', 'allowed', 'case_sensitive', 'negate', mode='before')
    @classmethod
    def convert_jsonpath(cls, value: object) -> object | JSONPath:
        """Convert JSONPath-like strings to JSONPath objects."""
        return _convert_to_jsonpath(value)

    @property
    def default_results(self) -> dict[str, Any]:
        """Return default results structure for superset checks on error."""
        return {'passed': False, 'unexpected': [], 'found': []}

    def __call__(self) -> dict[str, Any]:
        """
        Execute superset check using resolved Pydantic fields.

        Returns:
            Dictionary with 'passed' (bool), 'unexpected' (list), and 'found' (list) keys.

        Raises:
            RuntimeError: If any field contains unresolved JSONPath objects
        """
        if isinstance(self.collection, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'collection' field: {self.collection}")
        if isinstance(self.allowed, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'allowed' field: {self.allowed}")
        if isinstance(self.case_sensitive, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'case_sensitive' field: {self.case_sensitive}")  # noqa: E501
        if isinstance(self.negate, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'negate' field: {self.negate}")

        if not isinstance(self.collection, list | set):
            raise ValidationError(
                "Superset check 'collection' must be a list or set, "
                f"got {type(self.collection).__name__}",
            )
        if not isinstance(self.allowed, list | set):
            raise ValidationError(
                "Superset check 'allowed' must be a list or set, "
                f"got {type(self.allowed).__name__}",
            )

        found, unexpected = _partition_elements(
            source=self.collection,
            target=self.allowed,
            case_sensitive=self.case_sensitive,
        )

        passed = len(unexpected) == 0
        if self.negate:
            passed = not passed

        return {'passed': passed, 'unexpected': unexpected, 'found': found}
