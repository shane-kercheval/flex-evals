"""TestCase schema implementation for FEP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    # Import check classes for type hints only
    from ..checks.base import CheckTypes



@dataclass
class TestCase:
    """
    A test case provides the input and optional expected output for evaluation.

    Required Fields:
    - id: Unique identifier for the test case
    - input: The input provided to the system being evaluated

    Optional Fields:
    - expected: Reference output for comparison or validation
    - metadata: Descriptive information about the test case
    - checks: Convenience extension of protocol for per-test-case pattern of checks
    """

    input: str | dict[str, Any] | list[Any] | Any
    id: str | None = None
    expected: str | dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    checks: list[CheckTypes] | None = None  # Per-test-case checks (convenience pattern)

    def __post_init__(self):
        """Validate required fields and constraints."""
        if self.id is not None:
            if not isinstance(self.id, str):
                raise ValueError("TestCase.id must be a string or None")
            if not self.id:
                raise ValueError("TestCase.id must be a non-empty string when provided")

        if self.input is None:
            raise ValueError("TestCase.input is required and cannot be None")
