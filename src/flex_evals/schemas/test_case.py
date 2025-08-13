"""TestCase schema implementation for FEP."""

from dataclasses import dataclass
from typing import Any
from ..schemas.check import Check, SchemaCheck


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

    id: str
    input: str | dict[str, Any]
    expected: str | dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    checks: list[Check | SchemaCheck] | None = None  # Forward reference, convenience extension

    def __post_init__(self):
        """Validate required fields and constraints."""
        if not self.id or not isinstance(self.id, str):
            raise ValueError("TestCase.id must be a non-empty string")

        if self.input is None:
            raise ValueError("TestCase.input is required and cannot be None")

        if not isinstance(self.input, str | dict):
            raise ValueError("TestCase.input must be a string or dictionary")
