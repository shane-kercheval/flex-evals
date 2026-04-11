"""Subset Check - checks if all required items are present in a collection."""

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
    Checks whether all required items are present in the collection.

    Performs a set membership test: required ⊆ collection. Duplicates are
    ignored (set semantics). Elements are compared using Python == by default;
    when case_sensitive=False, only string elements are lowercased for comparison.
    """

    collection: list | set | JSONPath = Field(
        ...,
        description="The collection to check against (e.g., model output tags)",
    )
    required: list | set | JSONPath = Field(
        ...,
        description="Items that must be present in collection",
    )
    case_sensitive: bool | JSONPath = Field(
        True,
        description="Whether string element matching is case-sensitive",
    )
    negate: bool | JSONPath = Field(
        False,
        description="If true, inverts the result",
    )

    @field_validator('collection', 'required', 'case_sensitive', 'negate', mode='before')
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
        if isinstance(self.collection, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'collection' field: {self.collection}")
        if isinstance(self.required, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'required' field: {self.required}")
        if isinstance(self.case_sensitive, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'case_sensitive' field: {self.case_sensitive}")  # noqa: E501
        if isinstance(self.negate, JSONPath):
            raise RuntimeError(f"JSONPath not resolved for 'negate' field: {self.negate}")

        if not isinstance(self.collection, list | set):
            raise ValidationError(
                "Subset check 'collection' must be a list or set, "
                f"got {type(self.collection).__name__}",
            )
        if not isinstance(self.required, list | set):
            raise ValidationError(
                "Subset check 'required' must be a list or set, "
                f"got {type(self.required).__name__}",
            )

        found, missing = _partition_elements(
            source=self.required,
            target=self.collection,
            case_sensitive=self.case_sensitive,
        )

        passed = len(missing) == 0
        if self.negate:
            passed = not passed

        return {'passed': passed, 'missing': missing, 'found': found}
